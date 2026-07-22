#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cache_dryrun: 只读、不跑模型、不写入的「判重+过期判断」预演脚本，用于验证
hdfs_utils 读到的 baseline 缓存 + 按用户最新交互时间判过期的分类逻辑是否符合
预期——完全不触碰 generate.py 现有跑法（不改它、不导入它的推理路径），跑这个
脚本不会影响任何正在用的产出。

分类规则（cache_key = uid_<geohash>_<cache_version_tag>_<period>）：
  - baseline 里没有这个 cache_key            -> 新增（cache miss）
  - baseline 有，但用户最新交互时间比 baseline 记录的新（或 baseline 缺失
    该字段，视为无法判断）                    -> 过期（需要刷新）
  - baseline 有，且最新交互时间一致            -> 命中（跳过）
「新增」+「过期」累计达到 infer_max_num（或窗口用户耗尽）即停止，与 generate.py
未来要接入的实际推理停止条件保持一致，这样这里数出来的数字就是以后真跑模型时
会跑多少条的准确预估。

验证通过、确认分类逻辑无误后，再把这段逻辑正式合入 generate.py（替换掉现在
「全量重跑、不判重」的旧逻辑），并接上真正的写回。

用法：
    python inference/cache_dryrun.py [common.conf]
"""

import os
import sys
import configparser
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("train", os.path.join("train", "model"), "data", "inference"):
    sys.path.insert(0, os.path.join(ROOT, p))

from tokenizer_sid import SIDTokenizer                     # noqa: E402
from constrained_decode import build_sid_trie              # noqa: E402
import geo_utils as geo                                    # noqa: E402
import hdfs_utils as hu                                     # noqa: E402


def load_config(conf_path: str) -> dict:
    cp = configparser.ConfigParser()
    if not cp.read(conf_path, encoding="utf-8"):
        raise FileNotFoundError(f"找不到配置文件: {conf_path}")
    root = os.path.dirname(os.path.abspath(conf_path))

    def path(key, default):
        v = cp.get("inference", key, fallback=default)
        return v if os.path.isabs(v) else os.path.join(root, v)

    return {
        "infer_start": cp.get("inference", "infer_start"),
        "infer_end": cp.get("inference", "infer_end"),
        "infer_max_num": cp.getint("inference", "infer_max_num", fallback=-1),
        "behaviors": [b.strip() for b in
                      cp.get("inference", "behaviors", fallback="pay").split(",")
                      if b.strip()],
        "periods": [p.strip() for p in
                    cp.get("inference", "periods", fallback="").split(",")
                    if p.strip()],
        "vocab_path": (lambda v: v if os.path.isabs(v) else os.path.join(root, v))(
            cp.get("train", "vocab_path", fallback="outputs/vocab.json")),
        "favor_coord_field": cp.get("inference", "favor_coord_field",
                                    fallback="user_favor_coor_top3"),
        "favor_coord_topk": cp.getint("inference", "favor_coord_topk", fallback=3),
        "favor_coord_radius_km": cp.getfloat("inference", "favor_coord_radius_km",
                                             fallback=4.0),
        "cache_version_tag": cp.get("inference", "cache_version_tag",
                                    fallback="Version3"),
        "hdfs_output_root": cp.get("inference", "hdfs_output_root").rstrip("/"),
    }


def iter_users_with_last_ts(conf_path: str, ic: dict, id2sid: dict):
    """跟 generate.py 的 iter_infer_users_with_coords 同源同逻辑，唯一区别：
       额外产出 last_interact_ts（该用户全部历史里最新一条交互的 ts，
       timeline 按 ts 升序，取最后一条即可）。判重逻辑验证通过后，
       这段改动会正式合入 generate.py 本体，替换掉它现在的同名函数。"""
    import step3_build_samples as step3
    from step1_get_user_action import load_config as load_data_config, stream_rows
    from step2_inject_sid import map_sample

    cfg = load_data_config(conf_path)
    cfg["train_start"], cfg["train_end"] = ic["infer_start"], ic["infer_end"]
    cfg["max_num"] = -1
    seq_fields = cfg["seq_fields"]
    tz_offset = cfg["tz_offset_hours"]
    field = ic["favor_coord_field"]

    for _dt, _hdfs, _schema, row in stream_rows(cfg, verbose=True):
        mapped = map_sample(row, id2sid, seq_fields)
        timeline = step3.build_timeline(mapped, seq_fields, tz_offset)
        if not timeline:
            continue
        last_ts = timeline[-1]["ts"]
        sessions = step3.sessionize_by_day(timeline)
        items = [(t["action"], t["geo_sid"], t["meal_period"])
                 for _, toks in sessions for t in toks]
        yield row.get("uid"), items, row.get(field), last_ts


def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "common.conf")
    ic = load_config(conf_path)

    tok = SIDTokenizer.load(ic["vocab_path"])
    tok_periods = list(getattr(tok, "periods", []))
    periods = ic["periods"] or tok_periods
    print(f"[INFO] infer_start={ic['infer_start']}  infer_end={ic['infer_end']}  "
          f"infer_max_num={ic['infer_max_num']}  periods={periods}")

    import step3_build_samples as step3
    from step2_inject_sid import load_item_sid_shop_map
    id2sid, _id2shop = load_item_sid_shop_map(step3.get_item_map_path(conf_path))

    trie = build_sid_trie(tok, id2sid.values())
    precision = geo.geohash_precision_from_vocab(tok)
    real_geohashes = {tok.id2token[tid][1:-1] for tid in trie.keys()}
    print(f"[INFO] trie 覆盖 {len(set(id2sid.values()))} 个 SID，真实 geohash 集合大小={len(real_geohashes)}")

    fs = hu.get_fs()
    src_dt = hu.find_source_dt(fs, ic["hdfs_output_root"], ic["infer_end"])
    print(f"[INFO] baseline dt={src_dt}")

    for behavior in ic["behaviors"]:
        baseline = hu.read_dt_cache(fs, ic["hdfs_output_root"], src_dt, behavior) if src_dt else {}
        base_uids = {hu.parse_uid(ck) for ck in baseline}
        print(f"\n[{behavior}] baseline: cache_key数={len(baseline)}  uid数={len(base_uids)}")

        new_keys, stale_keys, hit_keys = set(), set(), set()
        need_infer_cap = ic["infer_max_num"]
        n_users_seen = 0

        for uid, items, raw_coord, last_ts in iter_users_with_last_ts(conf_path, ic, id2sid):
            n_users_seen += 1
            pairs = geo.parse_favor_coords(raw_coord, ic["favor_coord_topk"])
            seen_gh = set()
            for lng, lat in pairs:
                gh = geo.encode(lat, lng, precision)
                if gh in seen_gh:
                    continue
                seen_gh.add(gh)
                nbr = geo.neighbor_geohash_set(lat, lng, ic["favor_coord_radius_km"], precision)
                if not (nbr & real_geohashes):
                    continue
                for period in periods:
                    ck = f"{uid}_{gh}_{ic['cache_version_tag']}_{period}"
                    if ck in baseline:
                        base_ts = baseline[ck]["last_interact_ts"]
                        if base_ts is None or base_ts != last_ts:
                            stale_keys.add(ck)
                        else:
                            hit_keys.add(ck)
                    else:
                        new_keys.add(ck)
            if 0 <= need_infer_cap <= len(new_keys) + len(stale_keys):
                break

        need_uids = {hu.parse_uid(ck) for ck in new_keys | stale_keys}
        hit_uids = {hu.parse_uid(ck) for ck in hit_keys}
        print(f"  扫过 {n_users_seen} 个用户后停止（infer_max_num={need_infer_cap}）")
        print(f"  新增: cache_key={len(new_keys)}  uid={len({hu.parse_uid(ck) for ck in new_keys})}")
        print(f"  过期: cache_key={len(stale_keys)}  uid={len({hu.parse_uid(ck) for ck in stale_keys})}")
        print(f"  命中跳过: cache_key={len(hit_keys)}  uid={len(hit_uids)}")
        print(f"  本次需要推理合计: cache_key={len(new_keys)+len(stale_keys)}  uid={len(need_uids)}")
        # 过期条目是替换（数量不变），只有新增条目会让总数增加
        merged_after = len(baseline) + len(new_keys)
        print(f"  预计推理后累计: cache_key={merged_after}  "
              f"uid={len(base_uids | need_uids)}")


if __name__ == "__main__":
    main()
