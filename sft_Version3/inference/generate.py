#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate: 批量推理（纯产出推荐结果，无 label、不算指标）——收藏坐标邻域约束缓存。

与 eval/run_eval 的区别：
  - 输入用【全部历史 S1..Sm】作前缀（真实推理场景，预测未来；eval 为留 label
    只用到 S(m-1)），有 >=1 个交互的用户即可推；
  - 用户圈定：顺序取窗口内前 infer_max_num 个可推用户（产出结果不是评估，
    不需要无偏抽样）；-1 = 全量；
  - 输出带 geo_sid -> item_id 反查（一对多：多个 item 可能共享同一 SID）。

生成逻辑（线上缓存预热用）：不改前缀（仍只强制接 <时段><行为>），而是把 beam
search 第 0 层（geohash）的候选根限定为"该收藏坐标 radius_km 邻域内、词表中
真实存在的 geohash 集合"——用户具体会落在哪个 geohash 仍由模型在这个子集里自己
选，只是圈定了地理范围；每用户按 user_favor_coor_top3 字段解析出的收藏坐标
（geohash 去重）与 periods 组合，各出一份缓存。
每条推理的粒度 = uid x 收藏坐标圆心 geohash x period（缓存键 cache_key =
uid_<圆心geohash>_<cache_version_tag>_<period>）。两个输出文件按行为分文件，
以 cache_key 一一对应（jsonl 是 cache_key 粒度，parquet 是 cache_key x rank
粒度）：
  - rec_{行为}.parquet：仅 cache_key/rank/score/sid/item_ids/shop_id 六列
    （精简为下游直接消费的推荐负载，period/坐标等上下文字段不落 parquet）；
  - debug_{行为}.jsonl：uid/target_behavior/period/coord_rank/圆心geohash/
    lng/lat/legal_geohashes(合法白名单)/cache_key + item_result（该 cache_key
    下 top-K 涉及的全部真实 item：[{item_id, shop_id, geo_sid}, ...]）+
    history(模型真实输入逐 token 拼接成的字符串，无分隔符，含开头 <bos> 与
    结尾本次强制生成条件 <period><behavior>)/prompt_tokens。
[inference] max_output_caches 只限制 debug_{行为}.jsonl 落盘前 N 个 cache_key
（供人工抽样核对，不影响 beam search 计算/排序）；rec_{行为}.parquet 永远是
本次推理的全部结果，不受这个开关影响。

流程：
  1. 读 common.conf [inference]（窗口/上限/行为/topk/beam/ckpt/输出目录）
     与 [train]（vocab_path/max_len）；
  2. 加载 ckpt + item map 建 SID 前缀树与 sid->item_ids 反查表；
  3. 流式拉窗口数据（复用 step1->2 + step3 的时间线/session 工具），逐用户拼
     全历史前缀，按收藏坐标邻域限定 geohash 候选根后生成；
  4. 边收边推边写（不整窗攒内存）：结果存 parquet（比 jsonl 省一个量级存储），
     目录按推理窗口命名：{output_dir}/{infer_start}_{infer_end}/rec_{行为}.parquet，
     行 = 一条推荐；同目录附 _meta.json（本次配置 + 统计），同窗口重跑会覆盖；
  5. 结束打印纯推理 QPS、端到端 QPS（含取数/写盘）。

用法:
    python generate.py [common.conf]
"""

import os
import sys
import json
import time
import configparser
import contextlib
from collections import defaultdict

import torch
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("train", os.path.join("train", "model"), "data", "inference"):
    sys.path.insert(0, os.path.join(ROOT, p))

from tokenizer_sid import SIDTokenizer                     # noqa: E402
from train_sft import load_checkpoint                      # noqa: E402
from constrained_decode import (build_sid_trie, constrained_beam_search,  # noqa: E402
                                make_prefix)
import geo_utils as geo                                    # noqa: E402


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
        "infer_max_num": cp.getint("inference", "infer_max_num", fallback=-1),
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
    }


# ------------------------------------------------------------------
# 数据：全历史用户流 + 收藏坐标原始字段
# ------------------------------------------------------------------
def iter_infer_users_with_coords(conf_path: str, ic: dict, id2sid: dict):
    """流式产出 (uid, items, favor_coord_raw)：items = 全部历史交互
       [(action, geo_sid, meal_period)...] 按时间原序；favor_coord_raw = 原始行的
       收藏坐标字段（lng@lat^lng@lat^... 格式）。复刻 step3.iter_user_samples
       内部的取数链路（而非改造其共享签名），避免把本功能的耦合带进 train/eval
       也在用的公共函数。顺序取前 infer_max_num 个可推用户（>=1 个交互）即停。"""
    import step3_build_samples as step3
    from step1_get_user_action import load_config, stream_rows
    from step2_inject_sid import map_sample

    cfg = load_config(conf_path)
    cfg["train_start"], cfg["train_end"] = ic["infer_start"], ic["infer_end"]
    cfg["max_num"] = -1
    seq_fields = cfg["seq_fields"]
    tz_offset = cfg["tz_offset_hours"]
    field = ic["favor_coord_field"]

    n = 0
    for _dt, _hdfs, _schema, row in stream_rows(cfg, verbose=True):
        mapped = map_sample(row, id2sid, seq_fields)
        timeline = step3.build_timeline(mapped, seq_fields, tz_offset)
        sessions = step3.sessionize_by_day(timeline)
        items = [(t["action"], t["geo_sid"], t["meal_period"])
                 for _, toks in sessions for t in toks]
        if not items:
            continue
        yield row.get("uid"), items, row.get(field)
        n += 1
        if 0 < ic["infer_max_num"] <= n:
            break


# ------------------------------------------------------------------
# 输出：流式 parquet
# ------------------------------------------------------------------
class ParquetRecWriter:
    """流式 parquet 写入：每条推荐一行，攒满 buffer_rows 写一个 row group，
       不整窗攒内存。用完必须 close()。一个文件混装多个 cache_key（uid x 圆心
       geohash x period）的推荐，只靠 cache_key 区分——period/coord_rank/圆心
       geohash/白名单等上下文字段不落 parquet（要看这些去 debug_{行为}.jsonl，
       同一 cache_key 在两个文件里一一对应）。"""
    SCHEMA = pa.schema([
        ("cache_key", pa.string()),           # uid_<圆心geohash>_<version>_<period>
        ("rank", pa.int32()),                 # 1 = 置信最高
        ("score", pa.float32()),              # 各 SID token 的累计 logprob
        ("sid", pa.string()),                 # 模型实际生成的 SID（其 geohash 未必等于圆心）
        ("item_ids", pa.list_(pa.string())),  # sid 反查（一对多）
        ("shop_id", pa.list_(pa.string())),   # 与 item_ids 逐一对应的店铺 id
    ])

    def __init__(self, path: str, sid2items: dict, id2shop: dict, topk: int,
                 buffer_rows: int = 50000):
        self.sid2items = sid2items
        self.id2shop = id2shop
        self.topk = topk
        self.buffer_rows = buffer_rows
        self.writer = pq.ParquetWriter(path, self.SCHEMA)
        self.buf = []

    def write_user(self, cache_key: str, beams):
        for rank, (sid, score) in enumerate(beams[:self.topk], start=1):
            item_ids = [str(x) for x in self.sid2items.get(sid, [])]
            self.buf.append({
                "cache_key": cache_key, "rank": rank, "score": float(score),
                "sid": sid, "item_ids": item_ids,
                "shop_id": [self.id2shop.get(x) for x in item_ids],
            })
        if len(self.buf) >= self.buffer_rows:
            self._flush()

    def _flush(self):
        if self.buf:
            self.writer.write_table(pa.Table.from_pylist(self.buf, schema=self.SCHEMA))
            self.buf = []

    def close(self):
        self._flush()
        self.writer.close()


class DebugJsonlWriter:
    """调试用：把喂给模型的输入解码成人可读形式逐行写 jsonl，与 parquet 并存。
       解码自实际编码后的 prefix（tok.decode），而非增广前的原始 items——
       这样才能反映 make_prefix 截断后模型真正看到的历史（尾部悬空的
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
       与 parquet 落盘的 rank 范围一致）：[{item_id, shop_id, geo_sid}, ...]。
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
                       device, autocast_ctx, writers: dict,
                       sid2items: dict, id2shop: dict,
                       debug_writers: dict = None) -> dict:
    """收藏坐标邻域约束缓存：每用户按 (behavior x period x 收藏坐标去重后的
       geohash) 生成一份 top-K，第一层 geohash 候选被限定在该坐标 radius_km 内、
       且真实存在 item 的 geohash 集合中（不改前缀，只改 beam search 的候选根，
       geohash 仍由模型选）。writers 以 behavior 为键（一行为一文件）；返回同
       结构统计 {behavior: {"n": 覆盖次数, "infer_s": 纯推理秒, "skipped": 空邻域跳过数}}。
       debug_writers（可选，同以 behavior 为键）：写入模型真实输入的 token 字符串 + 合法
       geohash 白名单 + item_result（该 cache_key 下全部真实 item 明细），
       供人工核对；ic["max_output_caches"]>0 时只对 debug_writers 落盘前 N 个
       cache_key（writers/parquet 不受影响，永远是全部结果；beam search 本身
       的计算/排序也不受影响，只是少写一份调试记录）。"""
    precision = geo.geohash_precision_from_vocab(tok)
    real_geohashes = {tok.id2token[tid][1:-1] for tid in trie.keys()}
    periods = ic["periods_resolved"]
    max_caches = ic.get("max_output_caches", -1)
    stats = {b: {"n": 0, "infer_s": 0.0, "skipped": 0} for b in ic["behaviors"]}
    written = {b: 0 for b in ic["behaviors"]}

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
            for period in periods:
                rows = []                     # (uid, prefix, root, extra_cols)
                for uid, items, coords in batch:
                    prefix = make_prefix(tok, items, behavior, ic["max_len"],
                                         period=period)
                    for rank, lng, lat, gh, legal_list, root in coords:
                        if root is None:
                            stats[behavior]["skipped"] += 1
                            continue
                        cache_key = f"{uid}_{gh}_{ic['cache_version_tag']}_{period}"
                        rows.append((uid, prefix, root, {
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
                stats[behavior]["infer_s"] += time.time() - t0
                stats[behavior]["n"] += len(rows)
                for (uid, prefix, root, extra), beams in zip(rows, results):
                    writers[behavior].write_user(extra["cache_key"], beams)
                    if debug_writers and (max_caches < 0 or written[behavior] < max_caches):
                        written[behavior] += 1
                        debug_writers[behavior].write({
                            "uid": uid, "target_behavior": behavior, **extra,
                            "item_result": _build_item_result(
                                beams, ic["topk"], sid2items, id2shop),
                            "history": _decode_history(tok, prefix),
                            "prompt_tokens": len(prefix["input_ids"]),
                        })

    batch, done = [], 0
    for uid, items, raw_coord in user_iter:
        coords = resolve_coords(raw_coord)
        if not coords:                        # 没有任何可用收藏坐标，跳过该用户
            continue
        batch.append((uid, items, coords))
        if len(batch) == ic["batch_size"]:
            flush(batch)
            done += len(batch)
            batch = []
            if done % (ic["batch_size"] * 20) < ic["batch_size"]:
                print(f"  已推理 {done} 用户")
    if batch:
        flush(batch)
    return stats


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "common.conf")
    ic = load_infer_config(conf_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    else:
        autocast_ctx = contextlib.nullcontext
    max_desc = "全量窗口" if ic["infer_max_num"] == -1 else f"前 {ic['infer_max_num']} 个可推用户"
    print(f"[INFO] device={device}")
    print(f"[INFO] 推理窗口: {ic['infer_start']} ~ {ic['infer_end']}  取 {max_desc}")

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

    run_dir = os.path.join(ic["output_dir"], f"{ic['infer_start']}_{ic['infer_end']}")
    os.makedirs(run_dir, exist_ok=True)
    t_start = time.time()

    paths = {b: os.path.join(run_dir, f"rec_{b}.parquet") for b in ic["behaviors"]}
    writers = {b: ParquetRecWriter(p, sid2items, id2shop, ic["topk"])
              for b, p in paths.items()}
    debug_paths = {b: os.path.join(run_dir, f"debug_{b}.jsonl")
                  for b in ic["behaviors"]} if ic["write_debug_input"] else {}
    debug_writers = {b: DebugJsonlWriter(p) for b, p in debug_paths.items()}
    try:
        stats = generate_for_users(
            model, tok, trie, iter_infer_users_with_coords(conf_path, ic, id2sid),
            ic, device, autocast_ctx, writers, sid2items, id2shop,
            debug_writers=debug_writers or None)
    finally:
        for w in writers.values():
            w.close()
        for w in debug_writers.values():
            w.close()
    wall_s = time.time() - t_start

    total_req = sum(s["n"] for s in stats.values())
    total_inf = sum(s["infer_s"] for s in stats.values())
    total_skip = sum(s["skipped"] for s in stats.values())
    print("\n================ 推理完成 ================")
    for b in ic["behaviors"]:
        s = stats[b]
        if s["n"]:
            print(f"  {b}: {s['n']} 次生成  纯推理 {s['infer_s']:.1f}s  "
                  f"QPS {s['n'] / max(s['infer_s'], 1e-9):.1f}  "
                  f"跳过(空邻域) {s['skipped']}")
        print(f"  结果: {paths[b]}" +
              (f"  调试输入: {debug_paths[b]}" if b in debug_paths else ""))
    print(f"  合计: {total_req} 次生成  跳过 {total_skip}  纯推理 QPS "
          f"{total_req / max(total_inf, 1e-9):.1f}  |  全流程 {wall_s:.1f}s"
          f"（含取数/编码/写盘）端到端 QPS {total_req / max(wall_s, 1e-9):.1f}")
    print("==========================================")

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
        "stats": {b: {"n": s["n"], "infer_s": round(s["infer_s"], 1),
                     "skipped": s["skipped"],
                     "qps": round(s["n"] / max(s["infer_s"], 1e-9), 1)}
                  for b, s in stats.items()},
    }
    with open(os.path.join(run_dir, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 配置与统计: {run_dir}/_meta.json")


if __name__ == "__main__":
    main()
