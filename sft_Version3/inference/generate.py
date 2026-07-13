#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate: 批量推理（纯产出推荐结果，无 label、不算指标）。

与 eval/run_eval 的区别：
  - 输入用【全部历史 S1..Sm】作前缀（真实推理场景，预测未来；eval 为留 label
    只用到 S(m-1)），有 >=1 个交互的用户即可推；
  - 用户圈定：顺序取窗口内前 infer_max_num 个可推用户（产出结果不是评估，
    不需要无偏抽样）；-1 = 全量；
  - 输出带 geo_sid -> item_id 反查（一对多：多个 item 可能共享同一 SID）。

两种模式（[inference] use_favor_coord 切换，互斥）：
  A. 常规模式（默认）：按 (行为 x 时段) 组合各出一份，前缀强制接 <时段><行为>，
     geohash 及其后各层由模型自由生成（trie 全量候选）；
     输出 rec_{行为}_{时段}.parquet。
  B. 收藏坐标模式（use_favor_coord=true，线上缓存预热用）：不改前缀（仍只接
     <时段><行为>），而是把 beam search 第 0 层（geohash）的候选根限定为"该收藏
     坐标 radius_km 邻域内、词表中真实存在的 geohash 集合"——用户具体会落在
     哪个 geohash 仍由模型在这个子集里自己选，只是圈定了地理范围；每用户按
     user_favor_coor_top3 字段解析出的收藏坐标（geohash 去重）与 periods 组合，
     各出一份缓存，键名 uid_<geohash>_<cache_version_tag>_<period>；
     输出按行为分文件 rec_{行为}.parquet，period/coord_rank/geohash 等作为列区分
     （不再按 period/坐标拆文件）。

流程：
  1. 读 common.conf [inference]（窗口/上限/行为/topk/beam/ckpt/输出目录）
     与 [train]（vocab_path/max_len）；
  2. 加载 ckpt + item map 建 SID 前缀树与 sid->item_ids 反查表；
  3. 流式拉窗口数据（复用 step1->2 + step3 的时间线/session 工具），逐用户拼
     全历史前缀；模式 A 按 (行为 x 时段) 生成，模式 B 额外按收藏坐标邻域限定
     geohash 候选根后生成；
  4. 边收边推边写（不整窗攒内存）：结果存 parquet（比 jsonl 省一个量级存储），
     目录按推理窗口命名：{output_dir}/{infer_start}_{infer_end}/rec_*.parquet，
     行 = 一条推荐，每组每用户 topk 行；同目录附 _meta.json（本次配置 + 统计），
     同窗口重跑会覆盖；
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
        "use_favor_coord": cp.getboolean("inference", "use_favor_coord", fallback=False),
        "favor_coord_field": cp.get("inference", "favor_coord_field",
                                    fallback="user_favor_coor_top3"),
        "favor_coord_topk": cp.getint("inference", "favor_coord_topk", fallback=3),
        "favor_coord_radius_km": cp.getfloat("inference", "favor_coord_radius_km",
                                             fallback=4.0),
        "cache_version_tag": cp.get("inference", "cache_version_tag",
                                    fallback="Version3"),
    }


# ------------------------------------------------------------------
# 数据：全历史用户流
# ------------------------------------------------------------------
def iter_infer_users(conf_path: str, ic: dict, id2sid: dict):
    """流式产出 (uid, items)：items = 全部历史交互 [(action, geo_sid)...] 按时间原序。
       顺序取前 infer_max_num 个可推用户（>=1 个交互）即停。"""
    import step3_build_samples as step3
    from step1_get_user_action import load_config

    cfg = load_config(conf_path)
    cfg["train_start"], cfg["train_end"] = ic["infer_start"], ic["infer_end"]
    cfg["max_num"] = -1                       # 上限按「可推用户数」算，行数不设限
    cfg["behavior_drop_x"] = 0                # 推理不需要 train/val/test 样本
    cfg["min_train_seq_len"] = 10 ** 9

    n = 0
    for uid, sessions, _samples in step3.iter_user_samples(
            cfg, conf_path, id2sid=id2sid, verbose=True):
        items = [(t["action"], t["geo_sid"], t["meal_period"])
                 for _, toks in sessions for t in toks]
        if not items:
            continue                          # 清洗后没有任何可用交互
        yield uid, items
        n += 1
        if 0 < ic["infer_max_num"] <= n:
            break


def iter_infer_users_with_coords(conf_path: str, ic: dict, id2sid: dict):
    """流式产出 (uid, items, favor_coord_raw)：与 iter_infer_users 同一数据源，
       额外透出原始行的收藏坐标字段（lng@lat^lng@lat^... 格式，供收藏坐标邻域
       缓存用）。复刻 step3.iter_user_samples 内部的取数链路（而非改造其共享
       签名），避免把本功能的耦合带进 train/eval 也在用的公共函数。"""
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
    """流式 parquet 写入：每条推荐一行（uid/rank/sid/score/item_ids [+ extra 列]），
       攒满 buffer_rows 写一个 row group，不整窗攒内存。用完必须 close()。"""
    SCHEMA = pa.schema([
        ("uid", pa.string()),
        ("rank", pa.int32()),                 # 1 = 置信最高
        ("sid", pa.string()),
        ("score", pa.float32()),              # 各 SID token 的累计 logprob
        ("item_ids", pa.list_(pa.string())),  # sid 反查（一对多）
    ])
    # 收藏坐标模式：一个文件混装多个 (period, coord) 组合，靠这些列区分
    FAVOR_COORD_SCHEMA = pa.schema([
        ("uid", pa.string()),
        ("period", pa.string()),
        ("coord_rank", pa.int32()),           # 该用户收藏坐标去重后的序号（0 起）
        ("geohash", pa.string()),             # 收藏坐标中心点的 geohash（缓存键用）
        ("lng", pa.float64()),
        ("lat", pa.float64()),
        ("cache_key", pa.string()),           # uid_<geohash>_<version>_<period>
        ("rank", pa.int32()),
        ("sid", pa.string()),
        ("score", pa.float32()),
        ("item_ids", pa.list_(pa.string())),
    ])

    def __init__(self, path: str, sid2items: dict, topk: int,
                 schema: "pa.Schema" = None, buffer_rows: int = 50000):
        self.sid2items = sid2items
        self.topk = topk
        self.buffer_rows = buffer_rows
        self.schema = schema or self.SCHEMA
        self.writer = pq.ParquetWriter(path, self.schema)
        self.buf = []

    def write_user(self, uid, beams, extra: dict = None):
        for rank, (sid, score) in enumerate(beams[:self.topk], start=1):
            row = {"uid": str(uid), "rank": rank, "sid": sid, "score": float(score),
                  "item_ids": [str(x) for x in self.sid2items.get(sid, [])]}
            if extra:
                row.update(extra)
            self.buf.append(row)
        if len(self.buf) >= self.buffer_rows:
            self._flush()

    def _flush(self):
        if self.buf:
            self.writer.write_table(pa.Table.from_pylist(self.buf, schema=self.schema))
            self.buf = []

    def close(self):
        self._flush()
        self.writer.close()


# ------------------------------------------------------------------
# 推理主体（与数据源/输出格式解耦，便于单测）
# ------------------------------------------------------------------
def generate_for_users(model, tok, trie, user_iter, ic: dict,
                       device, autocast_ctx, writers: dict) -> dict:
    """按 batch 收用户 -> 每个 (行为[, 时段]) 组各跑一次约束 beam search ->
       逐用户写对应 writer。writers/返回值均以组名（如 'pay_bf'）为键：
       {组名: {"n": 用户数, "infer_s": 纯推理秒}}。"""
    groups = ic["groups"]                 # [(组名, behavior, period)]
    stats = {g: {"n": 0, "infer_s": 0.0} for g, _, _ in groups}

    def flush(batch):
        for gname, behavior, period in groups:
            prefixes = [make_prefix(tok, items, behavior, ic["max_len"],
                                    period=period)
                        for _, items in batch]
            t0 = time.time()
            results = constrained_beam_search(model, tok, trie, prefixes,
                                              ic["beam_size"], device, autocast_ctx)
            stats[gname]["infer_s"] += time.time() - t0
            stats[gname]["n"] += len(batch)
            for (uid, _), beams in zip(batch, results):
                writers[gname].write_user(uid, beams)

    batch, done = [], 0
    for uid, items in user_iter:
        batch.append((uid, items))
        if len(batch) == ic["batch_size"]:
            flush(batch)
            done += len(batch)
            batch = []
            if done % (ic["batch_size"] * 20) < ic["batch_size"]:
                print(f"  已推理 {done} 用户")
    if batch:
        flush(batch)
    return stats


def generate_for_users_favor_coord(model, tok, trie, user_iter, ic: dict,
                                   device, autocast_ctx, writers: dict) -> dict:
    """收藏坐标邻域约束缓存（详见模块 docstring 模式 B）。
       writers 以 behavior 为键（一行为一文件）；返回同结构统计
       {behavior: {"n": 覆盖次数, "infer_s": 纯推理秒, "skipped": 空邻域跳过数}}。"""
    precision = geo.geohash_precision_from_vocab(tok)
    real_geohashes = {tok.id2token[tid][1:-1] for tid in trie.keys()}
    periods = ic["periods_resolved"]
    stats = {b: {"n": 0, "infer_s": 0.0, "skipped": 0} for b in ic["behaviors"]}

    def resolve_coords(raw):
        """收藏坐标原始字段 -> 去重后的 (rank, lng, lat, geohash, root) 列表；
           root = 该坐标邻域内真实 geo token 限定的 trie 视图，邻域内无真实
           item 时 root=None（调用方计入 skipped 并跳过）。"""
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
                out.append((rank, lng, lat, gh, None))
                continue
            root = {tok.token2id[f"<{g}>"]: trie[tok.token2id[f"<{g}>"]] for g in valid}
            out.append((rank, lng, lat, gh, root))
        return out

    def flush(batch):
        for behavior in ic["behaviors"]:
            for period in periods:
                rows = []                     # (uid, prefix, root, extra_cols)
                for uid, items, coords in batch:
                    prefix = make_prefix(tok, items, behavior, ic["max_len"],
                                         period=period)
                    for rank, lng, lat, gh, root in coords:
                        if root is None:
                            stats[behavior]["skipped"] += 1
                            continue
                        cache_key = f"{uid}_{gh}_{ic['cache_version_tag']}_{period}"
                        rows.append((uid, prefix, root, {
                            "period": period, "coord_rank": rank, "geohash": gh,
                            "lng": lng, "lat": lat, "cache_key": cache_key}))
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
                for (uid, _, _, extra), beams in zip(rows, results):
                    writers[behavior].write_user(uid, beams, extra=extra)

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
                print(f"  已推理 {done} 用户（收藏坐标模式）")
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

    if ic["use_favor_coord"]:
        if not tok_periods:
            raise ValueError("use_favor_coord=True 需要词表含时段位（先用带时段的"
                             "词表/checkpoint，如 ntp_w_period 分支训出的模型）")
        periods = ic["periods"] or tok_periods
        for p in periods:
            if p not in tok_periods:
                raise ValueError(f"未知时段 {p!r}（词表时段: {tok_periods}）")
        ic["periods_resolved"] = periods
        print(f"[INFO] 收藏坐标模式: behaviors={ic['behaviors']}  periods={periods}  "
              f"radius={ic['favor_coord_radius_km']}km  topk坐标={ic['favor_coord_topk']}  "
              f"topk={ic['topk']}  beam={ic['beam_size']}  batch={ic['batch_size']}")
    else:
        if tok_periods:                   # 词表含时段位：条件必须给全
            periods = ic["periods"] or tok_periods
            for p in periods:
                if p not in tok_periods:
                    raise ValueError(f"未知时段 {p!r}（词表时段: {tok_periods}）")
        else:
            if ic["periods"]:
                raise ValueError("词表不含时段位，[inference] periods 应留空")
            periods = [None]
        # (组名, behavior, period)；无时段位时组名 = 行为名
        ic["groups"] = [(b if p is None else f"{b}_{p}", b, p)
                        for b in ic["behaviors"] for p in periods]
        print(f"[INFO] 生成组合={[g for g, _, _ in ic['groups']]}  topk={ic['topk']}  "
              f"beam={ic['beam_size']}  batch={ic['batch_size']}")

    model, _cfg, meta = load_checkpoint(ic["ckpt_path"], map_location=device)
    model.to(device).eval()
    print(f"[INFO] ckpt={ic['ckpt_path']}  (epoch={meta['epoch']} "
          f"val_loss={meta['val_loss']:.4f})")

    import step3_build_samples as step3
    id2sid = step3.load_item_sid_map(step3.get_item_map_path(conf_path))
    sid2items = defaultdict(list)             # 反查：一个 SID 可对应多个 item
    for item_id, sid in id2sid.items():
        sid2items[sid].append(item_id)
    trie = build_sid_trie(tok, id2sid.values())
    print(f"[INFO] trie 覆盖 {len(sid2items)} 个 SID（{len(id2sid)} 个 item）")

    run_dir = os.path.join(ic["output_dir"], f"{ic['infer_start']}_{ic['infer_end']}")
    os.makedirs(run_dir, exist_ok=True)
    t_start = time.time()

    if ic["use_favor_coord"]:
        paths = {b: os.path.join(run_dir, f"rec_{b}.parquet") for b in ic["behaviors"]}
        writers = {b: ParquetRecWriter(p, sid2items, ic["topk"],
                                       schema=ParquetRecWriter.FAVOR_COORD_SCHEMA)
                  for b, p in paths.items()}
        try:
            stats = generate_for_users_favor_coord(
                model, tok, trie, iter_infer_users_with_coords(conf_path, ic, id2sid),
                ic, device, autocast_ctx, writers)
        finally:
            for w in writers.values():
                w.close()
        wall_s = time.time() - t_start

        total_req = sum(s["n"] for s in stats.values())
        total_inf = sum(s["infer_s"] for s in stats.values())
        total_skip = sum(s["skipped"] for s in stats.values())
        print("\n================ 推理完成（收藏坐标模式） ================")
        for b in ic["behaviors"]:
            s = stats[b]
            if s["n"]:
                print(f"  {b}: {s['n']} 次生成  纯推理 {s['infer_s']:.1f}s  "
                      f"QPS {s['n'] / max(s['infer_s'], 1e-9):.1f}  "
                      f"跳过(空邻域) {s['skipped']}")
            print(f"  结果: {paths[b]}")
        print(f"  合计: {total_req} 次生成  跳过 {total_skip}  纯推理 QPS "
              f"{total_req / max(total_inf, 1e-9):.1f}  |  全流程 {wall_s:.1f}s"
              f"（含取数/编码/写盘）端到端 QPS {total_req / max(wall_s, 1e-9):.1f}")
        print("==========================================================")

        meta = {
            "mode": "favor_coord",
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
    else:
        paths = {g: os.path.join(run_dir, f"rec_{g}.parquet")
                for g, _, _ in ic["groups"]}
        writers = {g: ParquetRecWriter(p, sid2items, ic["topk"])
                  for g, p in paths.items()}
        try:
            stats = generate_for_users(model, tok, trie,
                                       iter_infer_users(conf_path, ic, id2sid),
                                       ic, device, autocast_ctx, writers)
        finally:
            for w in writers.values():
                w.close()
        wall_s = time.time() - t_start

        total_req = sum(s["n"] for s in stats.values())  # 1 用户 x 1 组合 = 1 次生成
        total_inf = sum(s["infer_s"] for s in stats.values())
        print("\n================ 推理完成 ================")
        for g, _, _ in ic["groups"]:
            s = stats[g]
            if s["n"]:
                print(f"  {g}: {s['n']} 用户  纯推理 {s['infer_s']:.1f}s  "
                      f"QPS {s['n'] / max(s['infer_s'], 1e-9):.1f}  "
                      f"({s['infer_s'] / s['n'] * 1000:.1f}ms/用户)")
            print(f"  结果: {paths[g]}")
        print(f"  合计: {total_req} 次生成  纯推理 QPS "
              f"{total_req / max(total_inf, 1e-9):.1f}  |  全流程 {wall_s:.1f}s"
              f"（含取数/编码/写盘）端到端 QPS {total_req / max(wall_s, 1e-9):.1f}")
        print("==========================================")

        meta = {
            "mode": "period",
            "infer_start": ic["infer_start"], "infer_end": ic["infer_end"],
            "infer_max_num": ic["infer_max_num"], "behaviors": ic["behaviors"],
            "groups": [g for g, _, _ in ic["groups"]],
            "topk": ic["topk"], "beam_size": ic["beam_size"],
            "batch_size": ic["batch_size"], "ckpt_path": ic["ckpt_path"],
            "ckpt_epoch": meta["epoch"], "ckpt_val_loss": round(meta["val_loss"], 4),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_s": round(wall_s, 1),
            "stats": {b: {"n": s["n"], "infer_s": round(s["infer_s"], 1),
                         "qps": round(s["n"] / max(s["infer_s"], 1e-9), 1)}
                      for b, s in stats.items()},
        }

    with open(os.path.join(run_dir, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 配置与统计: {run_dir}/_meta.json")


if __name__ == "__main__":
    main()
