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

数据并行（[inference] gpu_num>1）：判重分类不再由本文件做，而是 submit_infer.py
提交任何 GPU 任务之前先调用 plan_infer.py 跑一次性的预处理——流式扫全部用户、
按 baseline 分类，命中的直接写 rec_{行为}_part_cold.parquet，新增/过期的按
"当前任务数最少"分配到 gpu_num 个分片（各自独立的 infer_max_num 上限），写成
本地/NFS 的任务清单（uid+历史交互+已解析好的 geohash/period/legal_geohashes
等）。本文件在 shard_index 不为空时只读自己那份任务清单直接跑 beam search，
不做判重、也不用连 HDFS 读 baseline——结果写 rec_{行为}_part{shard_index}.parquet
（只含本分片新推理出的条目，不混 baseline）。

用法:
    python generate.py [common.conf]                  # 本地/单卡，不分片，
                                                       # 自己做判重+推理+写回
    python generate.py [common.conf] <shard_index>     # 数据并行 GPU worker，
                                                        # 由 submit_infer.py 调用，
                                                        # 只读 plan_infer.py 准备好的
                                                        # 任务清单，纯推理+写自己的 part
"""

import os
import sys
import json
import time
import contextlib

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("train", os.path.join("train", "model"), "data", "inference"):
    sys.path.insert(0, os.path.join(ROOT, p))

from train_sft import load_checkpoint                      # noqa: E402
from constrained_decode import constrained_beam_search, make_prefix  # noqa: E402
import hdfs_utils as hu                                     # noqa: E402
import infer_common as ifc                                  # noqa: E402

load_infer_config = ifc.load_infer_config


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


def _rows_from_beams(beams, topk: int, sid2items: dict, id2shop: dict) -> list:
    """beam search 结果 -> 落盘用的 rank/score/sid/item_ids/shop_id 行列表。"""
    out = []
    for rk, (sid, score) in enumerate(beams[:topk], start=1):
        item_ids = [str(x) for x in sid2items.get(sid, [])]
        out.append({"rank": rk, "score": float(score), "sid": sid,
                   "item_ids": item_ids, "shop_id": [id2shop.get(x) for x in item_ids]})
    return out


# ------------------------------------------------------------------
# 单卡（不分片）推理主体：判重 + 推理一体，与数据源解耦，便于单测
# ------------------------------------------------------------------
def generate_for_users(model, tok, trie, user_iter, ic: dict,
                       device, autocast_ctx, baselines: dict,
                       sid2items: dict, id2shop: dict,
                       real_geohashes: set, precision: int,
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
    periods = ic["periods_resolved"]
    max_caches = ic.get("max_output_caches", -1)
    cap = ic["infer_max_num"]

    merged = {b: dict(baselines.get(b, {})) for b in ic["behaviors"]}
    stats = {b: {"n": 0, "infer_s": 0.0, "skipped": 0,
                 "new_keys": set(), "stale_keys": set(), "hit_keys": set()}
             for b in ic["behaviors"]}
    written = {b: 0 for b in ic["behaviors"]}
    done_behaviors = set()

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
                    out_rows = _rows_from_beams(beams, ic["topk"], sid2items, id2shop)
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
        coords = ifc.resolve_coords(tok, trie, real_geohashes, precision, raw_coord,
                                    ic["favor_coord_topk"], ic["favor_coord_radius_km"])
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
# 数据并行 GPU worker：只读任务清单，纯推理，不判重
# ------------------------------------------------------------------
def run_shard_worker(ic: dict, shard_index: int, model, tok, trie,
                     device, autocast_ctx, sid2items: dict, id2shop: dict,
                     debug_writers: dict = None) -> tuple:
    """读 plan_infer.py 为本分片准备好的任务清单（每个 behavior 一个 JSONL
       文件），批量跑 beam search，返回 (rows_by_behavior, stats)：
         rows_by_behavior[behavior] = {cache_key: {"rows":[...], "last_interact_ts":ts}}
           只含本分片本次新推理出的条目（不含任何 baseline/命中数据——那些
           已经由 plan_infer.py 直接写进 rec_{behavior}_part_cold.parquet 了）；
         stats[behavior] = {"n": 推理次数, "infer_s": 纯推理秒}。"""
    rdir = ifc.run_dir(ic)
    max_caches = ic.get("max_output_caches", -1)
    rows_by_behavior = {}
    stats = {}
    t_worker_start = time.time()

    for behavior in ic["behaviors"]:
        wl_path = ifc.work_list_path(rdir, behavior, shard_index)
        items = []
        if os.path.exists(wl_path):
            with open(wl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        print(f"  [{behavior}] 本分片待推理 {len(items)} 条")

        out_rows = {}
        s = {"n": 0, "infer_s": 0.0}
        written = 0
        bs = ic["batch_size"]
        for i in range(0, len(items), bs):
            batch = items[i:i + bs]
            prefixes, roots = [], []
            for it in batch:
                prefix = make_prefix(tok, it["items"], it["behavior"], ic["max_len"],
                                     period=it["period"])
                root = ifc.root_from_legal_geohashes(tok, trie, it["legal_geohashes"])
                prefixes.append(prefix)
                roots.append(root)
            t0 = time.time()
            results = constrained_beam_search(model, tok, trie, prefixes,
                                              ic["beam_size"], device, autocast_ctx,
                                              roots=roots)
            s["infer_s"] += time.time() - t0
            s["n"] += len(batch)
            if s["n"] % (bs * 20) < bs or i + bs >= len(items):
                elapsed = time.time() - t_worker_start
                qps = s["n"] / max(s["infer_s"], 1e-9)
                print(f"  [{behavior}] 已推理 {s['n']}/{len(items)} 条  "
                      f"QPS {qps:.2f}  已耗时 {elapsed:.0f}s")
            for it, prefix, beams in zip(batch, prefixes, results):
                out_rows[it["cache_key"]] = {
                    "rows": _rows_from_beams(beams, ic["topk"], sid2items, id2shop),
                    "last_interact_ts": it["last_interact_ts"]}
                if debug_writers and behavior in debug_writers and \
                   (max_caches < 0 or written < max_caches):
                    written += 1
                    debug_writers[behavior].write({
                        "uid": it["uid"], "target_behavior": behavior,
                        "period": it["period"], "coord_rank": it["coord_rank"],
                        "geohash": it["geohash"], "lng": it["lng"], "lat": it["lat"],
                        "legal_geohashes": it["legal_geohashes"],
                        "cache_key": it["cache_key"],
                        "item_result": _build_item_result(
                            beams, ic["topk"], sid2items, id2shop),
                        "history": _decode_history(tok, prefix),
                        "prompt_tokens": len(prefix["input_ids"]),
                    })
        rows_by_behavior[behavior] = out_rows
        stats[behavior] = s
    return rows_by_behavior, stats


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "common.conf")
    # 数据并行分片编号：submit_infer.py 提交每个分片任务时以第二个位置参数传入
    # （0-based，< gpu_num）；不传（本地单卡直接跑）则视为不分片，忽略 gpu_num
    shard_index = int(sys.argv[2]) if len(sys.argv) > 2 else None
    ic = load_infer_config(conf_path)
    gpu_num = ic["gpu_num"] if shard_index is not None else 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    else:
        autocast_ctx = contextlib.nullcontext
    max_desc = "全量窗口" if ic["infer_max_num"] == -1 else f"最多 {ic['infer_max_num']} 条新增/过期缓存"
    print(f"[INFO] device={device}")
    if shard_index is not None:
        print(f"[INFO] 数据并行 GPU worker: shard_index={shard_index}/{gpu_num}"
              f"（判重已由 plan_infer.py 提前做完，本进程只读任务清单纯推理）")
    print(f"[INFO] 推理窗口: {ic['infer_start']} ~ {ic['infer_end']}  {max_desc}")

    model, _cfg, meta = load_checkpoint(ic["ckpt_path"], map_location=device,
                                        qwen3_path=ic["qwen3_path"] or None)
    model.to(device).eval()
    print(f"[INFO] ckpt={ic['ckpt_path']}  (epoch={meta['epoch']} "
          f"val_loss={meta['val_loss']:.4f})")

    ctx = ifc.load_pipeline_context(ic, conf_path)
    tok, trie = ctx["tok"], ctx["trie"]
    id2sid, id2shop, sid2items = ctx["id2sid"], ctx["id2shop"], ctx["sid2items"]
    ic["periods_resolved"] = ctx["periods_resolved"]
    print(f"[INFO] behaviors={ic['behaviors']}  periods={ic['periods_resolved']}  "
          f"radius={ic['favor_coord_radius_km']}km  topk坐标={ic['favor_coord_topk']}  "
          f"topk={ic['topk']}  beam={ic['beam_size']}  batch={ic['batch_size']}")
    print(f"[INFO] trie 覆盖 {len(sid2items)} 个 SID（{len(id2sid)} 个 item）")

    run_dir = ifc.run_dir(ic)
    os.makedirs(run_dir, exist_ok=True)
    t_start = time.time()

    # 数据并行时每个 shard 是独立进程/独立机器，但本地 output_dir 是共享 NFS
    # 路径——debug jsonl/_meta.json 文件名必须带上 shard 后缀，否则多个 shard
    # 并发写同一个文件会互相覆盖
    suffix = f"_part{shard_index:03d}" if shard_index is not None else ""
    debug_paths = {b: os.path.join(run_dir, f"debug_{b}{suffix}.jsonl")
                  for b in ic["behaviors"]} if ic["write_debug_input"] else {}
    debug_writers = {b: DebugJsonlWriter(p) for b, p in debug_paths.items()}

    fs = hu.get_fs()
    print(f"[INFO] hdfs_output_root={ic['hdfs_output_root']}")

    if shard_index is not None:
        # 数据并行 GPU worker：不判重、不连 baseline，只读 plan_infer.py 准备
        # 好的任务清单纯推理
        try:
            rows_by_behavior, stats = run_shard_worker(
                ic, shard_index, model, tok, trie, device, autocast_ctx,
                sid2items, id2shop, debug_writers=debug_writers or None)
        finally:
            for w in debug_writers.values():
                w.close()
        wall_s = time.time() - t_start

        print("\n================ 分片推理完成 ================")
        total_n = total_inf = 0
        for b in ic["behaviors"]:
            s = stats[b]
            if s["n"]:
                print(f"  [{b}] 推理 {s['n']} 次  纯推理 {s['infer_s']:.1f}s  "
                      f"QPS {s['n'] / max(s['infer_s'], 1e-9):.1f}")
            total_n += s["n"]; total_inf += s["infer_s"]
        print(f"  合计: {total_n} 次生成  纯推理 QPS {total_n / max(total_inf, 1e-9):.1f}  "
              f"|  全流程 {wall_s:.1f}s")
        print("================================================")

        print("[INFO] 写回 HDFS ...")
        for b in ic["behaviors"]:
            hu.write_dt_cache(fs, ic["hdfs_output_root"], ic["infer_end"], b,
                              rows_by_behavior[b], shard_index=shard_index, gpu_num=gpu_num)
            print(f"  [{b}] -> {ic['hdfs_output_root']}/dt={ic['infer_end']}/"
                  f"rec_{b}_part{shard_index:03d}.parquet（{len(rows_by_behavior[b])} 个 cache_key）")

        meta = {
            "infer_start": ic["infer_start"], "infer_end": ic["infer_end"],
            "gpu_num": gpu_num, "shard_index": shard_index,
            "ckpt_path": ic["ckpt_path"], "ckpt_epoch": meta["epoch"],
            "ckpt_val_loss": round(meta["val_loss"], 4),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_s": round(wall_s, 1),
            "stats": {b: {"n_infer": stats[b]["n"], "infer_s": round(stats[b]["infer_s"], 1),
                         "qps": round(stats[b]["n"] / max(stats[b]["infer_s"], 1e-9), 1),
                         "output_cache_key": len(rows_by_behavior[b])}
                      for b in ic["behaviors"]},
        }
        meta_name = f"_meta{suffix}.json"
        with open(os.path.join(run_dir, meta_name), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 本地调试/统计: {run_dir}/{meta_name}")
        return

    # 单卡（不分片）：判重 + 推理一体，逻辑不变
    src_dt = hu.find_source_dt(fs, ic["hdfs_output_root"], ic["infer_end"])
    print(f"[INFO] baseline dt={src_dt}")
    baselines = {}
    for b in ic["behaviors"]:
        baselines[b] = hu.read_dt_cache(fs, ic["hdfs_output_root"], src_dt, b) if src_dt else {}
        base_uids = {hu.parse_uid(ck) for ck in baselines[b]}
        print(f"  [{b}] baseline: cache_key={len(baselines[b])}  uid={len(base_uids)}")

    users = ifc.iter_infer_users_with_coords(conf_path, ic, id2sid)
    try:
        merged, stats = generate_for_users(
            model, tok, trie, users, ic, device, autocast_ctx, baselines,
            sid2items, id2shop, ctx["real_geohashes"], ctx["precision"],
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
        hu.write_dt_cache(fs, ic["hdfs_output_root"], ic["infer_end"], b, merged[b])
        print(f"  [{b}] -> {ic['hdfs_output_root']}/dt={ic['infer_end']}/rec_{b}.parquet"
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
        "gpu_num": gpu_num, "shard_index": shard_index,
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
    with open(os.path.join(run_dir, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 本地调试/统计: {run_dir}/_meta.json")


if __name__ == "__main__":
    main()
