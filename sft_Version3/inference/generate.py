#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate: 批量推理（纯产出推荐结果，无 label、不算指标）——收藏坐标邻域约束缓存，
按 dt=YYYYMMDD 分区累积在 HDFS 上，cache_key 粒度判重 + 按用户最新交互时间判过期。

与 eval/run_eval 的区别：
  - 输入用【全部历史 S1..Sm】作前缀（真实推理场景，预测未来；eval 为留 label
    只用到 S(m-1)），有 >=1 个交互的用户即可推；
  - 输出带 geo_sid -> item_id 反查（一对多：多个 item 可能共享同一 SID）。

生成逻辑（线上缓存预热用）：不改前缀（仍只强制接 <时段><行为>），而是把 beam
search 第 0 层（geohash）的候选根限定为"该收藏坐标 radius_km 邻域内、词表中
真实存在的 geohash 集合"——用户具体会落在哪个 geohash 仍由模型在这个子集里自己
选，只是圈定了地理范围；每用户按 user_favor_coor_top3 字段解析出的收藏坐标
（geohash 去重）与 periods 组合，各出一份缓存。
每条推理的粒度 = uid x 收藏坐标圆心 geohash x period（缓存键 cache_key =
uid_<圆心geohash>_<cache_version_tag>_<period>）。

HDFS 分区缓存（hdfs_utils.py）：
  - rec_{行为}.parquet 落 [inference] hdfs_output_root 下 dt=infer_end/ 分区，
    每个 dt 每个行为一个文件，是该 dt 的完整缓存快照（cache_key x rank，
    含 cache_key/rank/score/sid/item_ids/shop_id/last_interact_ts 七列）；
  - 每次推理先找 <= infer_end 的最近已有 dt 分区读入当 baseline；对每个
    (uid, geohash, period) 算出 cache_key：baseline 没有 -> 新增；baseline
    有但该用户全部历史最新交互时间变了（或 baseline 缺这个字段，判不出来）
    -> 过期，都需要重新推理；否则 -> 命中，直接沿用 baseline 里的旧结果；
  - 新增+过期两类累计数量达到 infer_max_num（或窗口用户耗尽）即停止推理；
  - baseline（命中的 + 本次没扫描到的）+ 本次新推理结果，合并后整体重写到
    dt=infer_end 分区（不是增量追加，是每次全量重写这一个分区的完整快照）。
  - debug_{行为}.jsonl / _meta.json 仍落本地 output_dir（不上 HDFS）；
    debug 只记录本次真正推理（新增/过期）过的 cache_key，命中跳过的没有
    新的模型输入可看，故不写。

数据并行（[inference] infer_num>1，由 submit_infer.py 编排提交，不是本文件
自己触发）：待推理用户按 crc32(uid) % infer_num 分片，各 shard 独立进程/
独立 GPU 只处理自己那一份，infer_max_num 是每个 shard 各自的上限（不是全局
总量）；每个 shard 写 rec_{行为}_part{shard_index:03d}.parquet（不再是单一
的 rec_{行为}.parquet），baseline 由编排方提前一次性合并成 _baseline 快照，
本文件的 shard worker 只读这份快照，不自己找历史 dt/glob 旧文件；
debug_{行为}.jsonl/_meta.json 文件名也带 _part{shard_index} 后缀，避免多个
shard 并发写同一个 NFS 路径互相覆盖。

用法:
    python generate.py [common.conf]                  # 本地/单卡，不分片
    python generate.py [common.conf] <shard_index>     # 数据并行 worker，
                                                        # 由 submit_infer.py 调用
"""

import os
import sys
import json
import time
import configparser
import contextlib
from collections import defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("train", os.path.join("train", "model"), "data", "inference"):
    sys.path.insert(0, os.path.join(ROOT, p))

from tokenizer_sid import SIDTokenizer                     # noqa: E402
from train_sft import load_checkpoint                      # noqa: E402
from constrained_decode import (build_sid_trie, constrained_beam_search,  # noqa: E402
                                make_prefix)
import geo_utils as geo                                    # noqa: E402
import hdfs_utils as hu                                     # noqa: E402


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------
def load_infer_config(conf_path: str) -> dict:
    cp = configparser.ConfigParser()
    if not cp.read(conf_path, encoding="utf-8"):
        raise FileNotFoundError(f"找不到配置文件: {conf_path}")
    root = os.path.dirname(os.path.abspath(conf_path))

    def path(key, default):
        v = cp.get("inference", key, fallback=default)
        return v if os.path.isabs(v) else os.path.join(root, v)

    topk = cp.getint("inference", "topk", fallback=20)
    return {
        "infer_start": cp.get("inference", "infer_start"),
        "infer_end": cp.get("inference", "infer_end"),
        # infer_num>1（数据并行）时这是每个分片独立的上限，不是全局总量
        "infer_max_num": cp.getint("inference", "infer_max_num", fallback=-1),
        # 数据并行分片数：>1 时按 uid 哈希拆成 infer_num 份，由
        # submit_infer.py 提交 infer_num 个独立单 GPU 任务；=1 = 不分片
        "infer_num": cp.getint("inference", "infer_num", fallback=1),
        "behaviors": [b.strip() for b in
                      cp.get("inference", "behaviors", fallback="clk,pay").split(",")
                      if b.strip()],
        "periods": [p.strip() for p in
                    cp.get("inference", "periods", fallback="").split(",")
                    if p.strip()],
        "topk": topk,
        "beam_size": max(cp.getint("inference", "beam_size", fallback=topk), topk),
        "batch_size": cp.getint("inference", "batch_size", fallback=32),
        "ckpt_path": path("ckpt_path", "outputs/ckpt/latest/best.pt"),
        "output_dir": path("output_dir", "outputs/inference"),
        "vocab_path": (lambda v: v if os.path.isabs(v) else os.path.join(root, v))(
            cp.get("train", "vocab_path", fallback="outputs/vocab.json")),
        "max_len": cp.getint("train", "max_len", fallback=512),
        # qwen3 checkpoint 没存 hf_config 时的 fallback，同 eval/run_eval.py：
        # 用当前这次运行自己的 common.conf 解析出的路径覆盖 checkpoint 里存的那份
        "qwen3_path": (lambda v: "" if not v else
                       (v if os.path.isabs(v) else os.path.join(root, v)))(
            cp.get("train", "qwen3_path", fallback="").strip()),
        "favor_coord_field": cp.get("inference", "favor_coord_field",
                                    fallback="user_favor_coor_top3"),
        "favor_coord_topk": cp.getint("inference", "favor_coord_topk", fallback=3),
        "favor_coord_radius_km": cp.getfloat("inference", "favor_coord_radius_km",
                                             fallback=4.0),
        "cache_version_tag": cp.get("inference", "cache_version_tag",
                                    fallback="Version3"),
        "write_debug_input": cp.getboolean("inference", "write_debug_input",
                                          fallback=True),
        "max_output_caches": cp.getint("inference", "max_output_caches",
                                       fallback=-1),
        # rec_*.parquet 落 HDFS 的根目录，其下按 dt=YYYYMMDD 建分区（见 hdfs_utils.py）
        "hdfs_output_root": cp.get("inference", "hdfs_output_root").rstrip("/"),
    }


# ------------------------------------------------------------------
# 数据：全历史用户流 + 收藏坐标原始字段 + 最新交互时间
# ------------------------------------------------------------------
def iter_infer_users_with_coords(conf_path: str, ic: dict, id2sid: dict):
    """流式产出 (uid, items, favor_coord_raw, last_interact_ts)：items = 全部
       历史交互 [(action, geo_sid, meal_period)...] 按时间原序；favor_coord_raw
       = 原始行的收藏坐标字段（lng@lat^lng@lat^... 格式）；last_interact_ts =
       该用户全部历史里最新一条交互的 ts（timeline 已按 ts 升序，取最后一条），
       用于跟 HDFS baseline 缓存里记录的时间比对、判断是否过期。复刻
       step3.iter_user_samples 内部的取数链路（而非改造其共享签名），避免把
       本功能的耦合带进 train/eval 也在用的公共函数。这里本身不做
       infer_max_num 早停——那个上限现在限制的是"新增+过期"的 cache_key 数
       （命中缓存的用户不消耗名额），必须扫过更多用户才能凑够，早停条件由
       调用方 generate_for_users 按分类结果动态判断，不能按原始用户数停。"""
    import step3_build_samples as step3
    from step1_get_user_action import load_config, stream_rows
    from step2_inject_sid import map_sample

    cfg = load_config(conf_path)
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
        last_interact_ts = timeline[-1]["ts"]
        sessions = step3.sessionize_by_day(timeline)
        items = [(t["action"], t["geo_sid"], t["meal_period"])
                 for _, toks in sessions for t in toks]
        yield row.get("uid"), items, row.get(field), last_interact_ts


# ------------------------------------------------------------------
# 输出：本地调试 jsonl
# ------------------------------------------------------------------
class DebugJsonlWriter:
    """调试用：把喂给模型的输入解码成人可读形式逐行写 jsonl。只记录本次真正
       推理过的 cache_key（新增/过期两类）——命中跳过的没有新的模型输入可看，
       不写。解码自实际编码后的 prefix（tok.decode），而非增广前的原始
       items——这样才能反映 make_prefix 截断后模型真正看到的历史（尾部悬空的
       <时段><行为> 强制条件 token 因不完整会被 decode 自动跳过，另作字段列出）。"""

    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, row: dict):
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self):
        self.f.close()


def _decode_history(tok, prefix: dict) -> str:
    """prefix['input_ids'] -> 逐 token 转回字符串直接拼接（不加分隔符），
       贴合模型真实看到的输入：交互之间本就没有分隔 token（encode_items 里
       交互紧密相接，靠固定 stride 定界，不靠分隔符），此处如实还原，不经过
       tok.decode()——decode() 是按"完整交互"解析的，会把结尾悬空、没有跟
       SID 的强制生成条件 <period><behavior> 当残缺片段丢弃，而那正是本次
       真正要预测的目标条件，必须保留。"""
    return "".join(tok.id2token[t] for t in prefix["input_ids"])


def _build_item_result(beams, topk: int, sid2items: dict, id2shop: dict) -> list:
    """该 cache_key 下 top-K 涉及的全部真实 item 清单（跨所有 rank 展平，
       与落盘的 rank 范围一致）：[{item_id, shop_id, geo_sid}, ...]。
       geo_sid = 该 item 所属 rank 的 sid（一个 sid 下的所有 item 共享同一 geo_sid）。"""
    out = []
    for sid, _score in beams[:topk]:
        for item_id in sid2items.get(sid, []):
            out.append({"item_id": item_id, "shop_id": id2shop.get(item_id),
                       "geo_sid": sid})
    return out


# ------------------------------------------------------------------
# 推理主体（与数据源解耦，便于单测）
# ------------------------------------------------------------------
def generate_for_users(model, tok, trie, user_iter, ic: dict,
                       device, autocast_ctx, baselines: dict,
                       sid2items: dict, id2shop: dict,
                       debug_writers: dict = None) -> tuple:
    """收藏坐标邻域约束缓存 + baseline 判重/过期判断：每用户按 (behavior x
       period x 收藏坐标去重后的 geohash) 算出 cache_key，与 baselines[behavior]
       比对——不存在则"新增"，存在但 last_interact_ts 对不上（或 baseline 缺
       这个字段）则"过期"，二者都要重新跑 beam search；存在且时间一致则
       "命中"，直接沿用 baseline 里的旧结果，不重新推理。新增+过期累计达到
       ic["infer_max_num"] 后该 behavior 停止再推理新用户（其它 behavior 未达
       上限的继续，直到全部达到上限或用户流耗尽）。

       返回 (merged, stats)：
         merged[behavior] = {cache_key: {"rows": [...], "last_interact_ts": ts}}
           起点是 baselines[behavior] 的浅拷贝，本次新增/过期的 cache_key 会
           被覆盖/新增进去，其余（命中的 + 本次没扫描到的）原样保留——即为
           本次要整体重写回 HDFS 该 dt 分区的完整快照。
         stats[behavior] = {"n": 实际推理次数, "infer_s": 纯推理秒,
           "skipped": 空邻域跳过数, "new_keys"/"stale_keys"/"hit_keys": 三个
           cache_key 集合（用于分类计数 + 反解 uid 数）}。

       debug_writers（可选，以 behavior 为键）：只对本次真正推理（新增/过期）
       的 cache_key 写调试记录；ic["max_output_caches"]>0 时限制条数。"""
    precision = geo.geohash_precision_from_vocab(tok)
    real_geohashes = {tok.id2token[tid][1:-1] for tid in trie.keys()}
    periods = ic["periods_resolved"]
    max_caches = ic.get("max_output_caches", -1)
    cap = ic["infer_max_num"]

    merged = {b: dict(baselines.get(b, {})) for b in ic["behaviors"]}
    stats = {b: {"n": 0, "infer_s": 0.0, "skipped": 0,
                 "new_keys": set(), "stale_keys": set(), "hit_keys": set()}
             for b in ic["behaviors"]}
    written = {b: 0 for b in ic["behaviors"]}
    done_behaviors = set()

    def resolve_coords(raw):
        """收藏坐标原始字段 -> 去重后的 (rank, lng, lat, geohash, legal_list, root)
           列表；root = 该坐标邻域内真实 geo token 限定的 trie 视图，legal_list =
           排序后的合法 geohash 白名单（供落盘）；邻域内无真实 item 时
           root=None（调用方计入 skipped 并跳过）。"""
        pairs = geo.parse_favor_coords(raw, ic["favor_coord_topk"])
        seen, out = set(), []
        for rank, (lng, lat) in enumerate(pairs):
            gh = geo.encode(lat, lng, precision)
            if gh in seen:                    # top3 坐标落到同一网格：去重只算一次
                continue
            seen.add(gh)
            nbr = geo.neighbor_geohash_set(lat, lng, ic["favor_coord_radius_km"], precision)
            valid = nbr & real_geohashes
            if not valid:
                out.append((rank, lng, lat, gh, [], None))
                continue
            legal_list = sorted(valid)
            root = {tok.token2id[f"<{g}>"]: trie[tok.token2id[f"<{g}>"]] for g in valid}
            out.append((rank, lng, lat, gh, legal_list, root))
        return out

    def flush(batch):
        for behavior in ic["behaviors"]:
            if behavior in done_behaviors:
                continue
            base = baselines.get(behavior, {})
            s = stats[behavior]
            for period in periods:
                rows = []                     # (uid, prefix, root, last_ts, extra_cols)
                for uid, items, coords, last_ts in batch:
                    to_infer = []
                    for rank, lng, lat, gh, legal_list, root in coords:
                        if root is None:
                            s["skipped"] += 1
                            continue
                        cache_key = f"{uid}_{gh}_{ic['cache_version_tag']}_{period}"
                        base_entry = base.get(cache_key)
                        if base_entry is None:
                            s["new_keys"].add(cache_key)
                        elif base_entry["last_interact_ts"] != last_ts:  # None 也算不等
                            s["stale_keys"].add(cache_key)
                        else:
                            s["hit_keys"].add(cache_key)
                            continue          # 命中：沿用 baseline，不进入本次推理
                        to_infer.append((rank, lng, lat, gh, legal_list, root, cache_key))
                    if not to_infer:
                        continue
                    prefix = make_prefix(tok, items, behavior, ic["max_len"],
                                         period=period)
                    for rank, lng, lat, gh, legal_list, root, cache_key in to_infer:
                        rows.append((uid, prefix, root, last_ts, {
                            "period": period, "coord_rank": rank, "geohash": gh,
                            "lng": lng, "lat": lat, "legal_geohashes": legal_list,
                            "cache_key": cache_key}))
                if not rows:
                    continue
                prefixes = [r[1] for r in rows]
                roots = [r[2] for r in rows]
                t0 = time.time()
                results = constrained_beam_search(model, tok, trie, prefixes,
                                                  ic["beam_size"], device, autocast_ctx,
                                                  roots=roots)
                s["infer_s"] += time.time() - t0
                s["n"] += len(rows)
                for (uid, prefix, root, last_ts, extra), beams in zip(rows, results):
                    cache_key = extra["cache_key"]
                    out_rows = []
                    for rk, (sid, score) in enumerate(beams[:ic["topk"]], start=1):
                        item_ids = [str(x) for x in sid2items.get(sid, [])]
                        out_rows.append({
                            "rank": rk, "score": float(score), "sid": sid,
                            "item_ids": item_ids,
                            "shop_id": [id2shop.get(x) for x in item_ids],
                        })
                    merged[behavior][cache_key] = {"rows": out_rows,
                                                   "last_interact_ts": last_ts}
                    if debug_writers and behavior in debug_writers and \
                       (max_caches < 0 or written[behavior] < max_caches):
                        written[behavior] += 1
                        debug_writers[behavior].write({
                            "uid": uid, "target_behavior": behavior, **extra,
                            "item_result": _build_item_result(
                                beams, ic["topk"], sid2items, id2shop),
                            "history": _decode_history(tok, prefix),
                            "prompt_tokens": len(prefix["input_ids"]),
                        })
            if 0 <= cap <= len(s["new_keys"]) + len(s["stale_keys"]):
                done_behaviors.add(behavior)

    batch, done = [], 0
    for uid, items, raw_coord, last_ts in user_iter:
        if len(done_behaviors) == len(ic["behaviors"]):
            break
        coords = resolve_coords(raw_coord)
        if not coords:                        # 没有任何可用收藏坐标，跳过该用户
            continue
        batch.append((uid, items, coords, last_ts))
        if len(batch) == ic["batch_size"]:
            flush(batch)
            done += len(batch)
            batch = []
            if done % (ic["batch_size"] * 20) < ic["batch_size"]:
                print(f"  已扫描 {done} 用户")
    if batch:
        flush(batch)
    return merged, stats


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "common.conf")
    # 数据并行分片编号：submit_infer.py 提交每个分片任务时以第二个位置参数传入
    # （0-based，< infer_num）；不传（本地单卡直接跑）则视为不分片，忽略 infer_num
    shard_index = int(sys.argv[2]) if len(sys.argv) > 2 else None
    ic = load_infer_config(conf_path)
    infer_num = ic["infer_num"] if shard_index is not None else 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    else:
        autocast_ctx = contextlib.nullcontext
    max_desc = "全量窗口" if ic["infer_max_num"] == -1 else f"最多 {ic['infer_max_num']} 条新增/过期缓存"
    print(f"[INFO] device={device}")
    if shard_index is not None:
        print(f"[INFO] 数据并行分片: shard_index={shard_index}/{infer_num}")
    print(f"[INFO] 推理窗口: {ic['infer_start']} ~ {ic['infer_end']}  {max_desc}")

    tok = SIDTokenizer.load(ic["vocab_path"])
    for b in ic["behaviors"]:
        if b not in tok.behaviors:
            raise ValueError(f"未知行为 {b!r}（词表行为: {tok.behaviors}）")
    tok_periods = list(getattr(tok, "periods", []))
    if not tok_periods:
        raise ValueError("收藏坐标邻域缓存需要词表含时段位（先用带时段的词表/"
                         "checkpoint，如 ntp_w_period 分支训出的模型）")
    periods = ic["periods"] or tok_periods
    for p in periods:
        if p not in tok_periods:
            raise ValueError(f"未知时段 {p!r}（词表时段: {tok_periods}）")
    ic["periods_resolved"] = periods
    print(f"[INFO] behaviors={ic['behaviors']}  periods={periods}  "
          f"radius={ic['favor_coord_radius_km']}km  topk坐标={ic['favor_coord_topk']}  "
          f"topk={ic['topk']}  beam={ic['beam_size']}  batch={ic['batch_size']}")

    model, _cfg, meta = load_checkpoint(ic["ckpt_path"], map_location=device,
                                        qwen3_path=ic["qwen3_path"] or None)
    model.to(device).eval()
    print(f"[INFO] ckpt={ic['ckpt_path']}  (epoch={meta['epoch']} "
          f"val_loss={meta['val_loss']:.4f})")

    import step3_build_samples as step3
    from step2_inject_sid import load_item_sid_shop_map
    id2sid, id2shop = load_item_sid_shop_map(step3.get_item_map_path(conf_path))
    sid2items = defaultdict(list)             # 反查：一个 SID 可对应多个 item
    for item_id, sid in id2sid.items():
        sid2items[sid].append(item_id)
    trie = build_sid_trie(tok, id2sid.values())
    print(f"[INFO] trie 覆盖 {len(sid2items)} 个 SID（{len(id2sid)} 个 item）")

    fs = hu.get_fs()
    print(f"[INFO] hdfs_output_root={ic['hdfs_output_root']}")
    baselines = {}
    src_dt = None
    if shard_index is not None:
        # 数据并行 worker：只读编排方（submit_infer.py）已经准备好的那一份
        # _baseline 快照（历史 dt/历史 part 文件已经在编排阶段合并好了，这里
        # 不用也不该自己再去 find_source_dt/glob 旧文件），并按本 shard 负责
        # 的 uid 子集筛一遍——这样后面 generate_for_users 里 merged[behavior]
        # 从一开始就只包含本 shard 的数据，写回时不用再额外过滤
        src_dt = ic["infer_end"]  # 快照落在 target dt 下，编排阶段已按来源合并好
        for b in ic["behaviors"]:
            full = hu.read_baseline_snapshot(fs, ic["hdfs_output_root"], ic["infer_end"], b)
            baselines[b] = {ck: v for ck, v in full.items()
                            if hu.shard_of_uid(hu.parse_uid(ck), infer_num) == shard_index}
            base_uids = {hu.parse_uid(ck) for ck in baselines[b]}
            print(f"  [{b}] 本 shard baseline: cache_key={len(baselines[b])}  "
                  f"uid={len(base_uids)}（快照总量 cache_key={len(full)}）")
    else:
        src_dt = hu.find_source_dt(fs, ic["hdfs_output_root"], ic["infer_end"])
        print(f"[INFO] baseline dt={src_dt}")
        for b in ic["behaviors"]:
            baselines[b] = hu.read_dt_cache(fs, ic["hdfs_output_root"], src_dt, b) if src_dt else {}
            base_uids = {hu.parse_uid(ck) for ck in baselines[b]}
            print(f"  [{b}] baseline: cache_key={len(baselines[b])}  uid={len(base_uids)}")

    run_dir = os.path.join(ic["output_dir"], f"{ic['infer_start']}_{ic['infer_end']}")
    os.makedirs(run_dir, exist_ok=True)
    t_start = time.time()

    # 数据并行时每个 shard 是独立进程/独立机器，但本地 output_dir 是共享 NFS
    # 路径——debug jsonl/_meta.json 文件名必须带上 shard 后缀，否则多个 shard
    # 并发写同一个文件会互相覆盖
    suffix = f"_part{shard_index:03d}" if shard_index is not None else ""
    debug_paths = {b: os.path.join(run_dir, f"debug_{b}{suffix}.jsonl")
                  for b in ic["behaviors"]} if ic["write_debug_input"] else {}
    debug_writers = {b: DebugJsonlWriter(p) for b, p in debug_paths.items()}
    users = iter_infer_users_with_coords(conf_path, ic, id2sid)
    if shard_index is not None:
        users = (rec for rec in users
                if hu.shard_of_uid(rec[0], infer_num) == shard_index)
    try:
        merged, stats = generate_for_users(
            model, tok, trie, users,
            ic, device, autocast_ctx, baselines, sid2items, id2shop,
            debug_writers=debug_writers or None)
    finally:
        for w in debug_writers.values():
            w.close()
    wall_s = time.time() - t_start

    print("\n================ 推理完成，判重明细 ================")
    total_n = total_inf = total_skip = 0
    for b in ic["behaviors"]:
        s = stats[b]
        new_uids = {hu.parse_uid(ck) for ck in s["new_keys"]}
        stale_uids = {hu.parse_uid(ck) for ck in s["stale_keys"]}
        hit_uids = {hu.parse_uid(ck) for ck in s["hit_keys"]}
        merged_uids = {hu.parse_uid(ck) for ck in merged[b]}
        base_uids = {hu.parse_uid(ck) for ck in baselines[b]}
        print(f"  [{b}]")
        print(f"    推理前 baseline: cache_key={len(baselines[b])}  uid={len(base_uids)}")
        print(f"    新增: cache_key={len(s['new_keys'])}  uid={len(new_uids)}  |  "
              f"过期: cache_key={len(s['stale_keys'])}  uid={len(stale_uids)}  |  "
              f"命中跳过: cache_key={len(s['hit_keys'])}  uid={len(hit_uids)}")
        if s["n"]:
            print(f"    实际推理 {s['n']} 次  纯推理 {s['infer_s']:.1f}s  "
                  f"QPS {s['n'] / max(s['infer_s'], 1e-9):.1f}  跳过(空邻域) {s['skipped']}")
        print(f"    推理后累计: cache_key={len(merged[b])}  uid={len(merged_uids)}")
        total_n += s["n"]; total_inf += s["infer_s"]; total_skip += s["skipped"]
    print(f"  合计: {total_n} 次生成  跳过 {total_skip}  纯推理 QPS "
          f"{total_n / max(total_inf, 1e-9):.1f}  |  全流程 {wall_s:.1f}s"
          f"（含取数/编码/写盘）端到端 QPS {total_n / max(wall_s, 1e-9):.1f}")
    print("=====================================================")

    print("[INFO] 写回 HDFS ...")
    for b in ic["behaviors"]:
        hu.write_dt_cache(fs, ic["hdfs_output_root"], ic["infer_end"], b, merged[b],
                          shard_index=shard_index, infer_num=infer_num)
        out_name = (f"rec_{b}_part{shard_index:03d}.parquet" if shard_index is not None
                   else f"rec_{b}.parquet")
        print(f"  [{b}] -> {ic['hdfs_output_root']}/dt={ic['infer_end']}/{out_name}"
              f"（{len(merged[b])} 个 cache_key）")

    meta = {
        "infer_start": ic["infer_start"], "infer_end": ic["infer_end"],
        "infer_max_num": ic["infer_max_num"], "behaviors": ic["behaviors"],
        "periods": ic["periods_resolved"],
        "favor_coord_field": ic["favor_coord_field"],
        "favor_coord_topk": ic["favor_coord_topk"],
        "favor_coord_radius_km": ic["favor_coord_radius_km"],
        "cache_version_tag": ic["cache_version_tag"],
        "topk": ic["topk"], "beam_size": ic["beam_size"],
        "batch_size": ic["batch_size"], "ckpt_path": ic["ckpt_path"],
        "ckpt_epoch": meta["epoch"], "ckpt_val_loss": round(meta["val_loss"], 4),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(wall_s, 1),
        "hdfs_output_root": ic["hdfs_output_root"],
        "baseline_dt": src_dt, "target_dt": ic["infer_end"],
        "infer_num": infer_num, "shard_index": shard_index,
        "stats": {b: {
            "baseline_cache_key": len(baselines[b]),
            "baseline_uid": len({hu.parse_uid(ck) for ck in baselines[b]}),
            "new_cache_key": len(stats[b]["new_keys"]),
            "stale_cache_key": len(stats[b]["stale_keys"]),
            "hit_cache_key": len(stats[b]["hit_keys"]),
            "merged_cache_key": len(merged[b]),
            "merged_uid": len({hu.parse_uid(ck) for ck in merged[b]}),
            "n_infer": stats[b]["n"], "infer_s": round(stats[b]["infer_s"], 1),
            "skipped": stats[b]["skipped"],
            "qps": round(stats[b]["n"] / max(stats[b]["infer_s"], 1e-9), 1),
        } for b in ic["behaviors"]},
    }
    meta_name = f"_meta{suffix}.json" if shard_index is not None else "_meta.json"
    with open(os.path.join(run_dir, meta_name), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 本地调试/统计: {run_dir}/{meta_name}")


if __name__ == "__main__":
    main()
