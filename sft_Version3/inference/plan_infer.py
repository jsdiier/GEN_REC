#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plan_infer: 数据并行（[inference] gpu_num>1）的判重预处理，跑在提交任何 GPU
任务之前，只跑一次（不随 gpu_num 放大），不需要 GPU。

职责：
  1. 读历史缓存（hdfs_output_root 下 <= infer_end 的最近一个 dt 分区，不管
     是老的单文件还是之前不同 gpu_num 留下的 part 文件，glob 全部合并）；
  2. 流式扫本次窗口全部用户，对每个 (uid, geohash, period) 算出 cache_key，
     跟历史缓存比对：
       - 命中（存在且 last_interact_ts 没变）-> 留在 cold 里原样不动；
       - 新增/过期（不存在，或 last_interact_ts 变了/历史缺这个字段判不出）
         -> 需要重新推理，按"当前任务数最少"分配到某个分片（简单负载均衡），
            直到该分片凑够 infer_max_num 条为止；全部分片都凑满后，新出现
            的新增/过期条目本轮不处理——新增的本来就不在 cold 里，直接跳过；
            过期的旧数据留在 cold 里不摘除，相当于"这轮先沿用旧结果，下次
            再判一次"，不会丢数据；
  3. 落盘：cold 写成 rec_{behavior}_part_cold.parquet（HDFS，dt=infer_end 下，
     跟其它 part 文件同规格，是正式结果的一部分，不是临时文件）；每个分片的
     待推理清单写成本地/NFS 的 JSONL（uid+历史交互+已解析好的 period/geohash/
     legal_geohashes 等，GPU worker 直接读来拼 prefix 跑 beam search，不用
     重新流式扫全量用户表）；
  4. 全部写完确认无误后，删掉本轮读到的旧历史文件（cold_path 本身如果跟某个
     旧文件路径重合——同一天重跑时会这样——要跳过，不能删自己刚写的东西）。

用法（供 submit_infer.py 内联调用，也可独立跑核对）：
    python plan_infer.py [common.conf]
"""

import os
import sys
import json
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("train", os.path.join("train", "model"), "data", "inference"):
    sys.path.insert(0, os.path.join(ROOT, p))

import infer_common as ifc                                  # noqa: E402
import hdfs_utils as hu                                      # noqa: E402


def run_planning(conf_path: str, ic: dict) -> dict:
    ctx = ifc.load_pipeline_context(ic, conf_path)
    tok, trie = ctx["tok"], ctx["trie"]
    precision, real_geohashes = ctx["precision"], ctx["real_geohashes"]
    id2sid = ctx["id2sid"]
    periods = ctx["periods_resolved"]
    behaviors = ic["behaviors"]
    gpu_num = ic["gpu_num"]
    cap = ic["infer_max_num"]

    fs = hu.get_fs()
    src_dt = hu.find_source_dt(fs, ic["hdfs_output_root"], ic["infer_end"])
    print(f"[PLAN] baseline dt={src_dt}")

    baselines, old_paths = {}, {}
    for b in behaviors:
        old_paths[b] = hu.glob_rec_files(fs, ic["hdfs_output_root"], src_dt, b) if src_dt else []
        baselines[b] = hu.read_cache_files(fs, old_paths[b])
        print(f"  [{b}] 历史文件 {len(old_paths[b])} 个，cache_key={len(baselines[b])}  "
              f"uid={len({hu.parse_uid(ck) for ck in baselines[b]})}")

    cold = {b: dict(baselines[b]) for b in behaviors}
    shard_counts = {b: [0] * gpu_num for b in behaviors}
    work_lists = {b: [[] for _ in range(gpu_num)] for b in behaviors}
    stats = {b: {"new_keys": set(), "stale_keys": set(), "hit_keys": set(), "skipped": 0}
            for b in behaviors}
    done = {b: False for b in behaviors}

    t0 = time.time()
    n_scanned = 0
    for uid, items, raw_coord, last_ts in ifc.iter_infer_users_with_coords(conf_path, ic, id2sid):
        if all(done.values()):
            break
        coords = ifc.resolve_coords(tok, trie, real_geohashes, precision, raw_coord,
                                    ic["favor_coord_topk"], ic["favor_coord_radius_km"])
        if not coords:
            continue
        n_scanned += 1
        for b in behaviors:
            if done[b]:
                continue
            base = baselines[b]
            s = stats[b]
            counts = shard_counts[b]
            for period in periods:
                for rank, lng, lat, gh, legal_list, root in coords:
                    if root is None:
                        s["skipped"] += 1
                        continue
                    cache_key = f"{uid}_{gh}_{ic['cache_version_tag']}_{period}"
                    base_entry = base.get(cache_key)
                    is_hit = base_entry is not None and base_entry["last_interact_ts"] == last_ts
                    if is_hit:
                        s["hit_keys"].add(cache_key)     # 已经在 cold 里，不用动
                        continue
                    if 0 <= cap and min(counts) >= cap:
                        continue                          # 全部分片已满，这条本轮先不处理
                    if base_entry is None:
                        s["new_keys"].add(cache_key)
                    else:
                        s["stale_keys"].add(cache_key)
                        cold[b].pop(cache_key, None)      # 过期：从 cold 摘掉，等下面重新推理写回
                    idx = min(range(gpu_num), key=lambda i: counts[i])
                    counts[idx] += 1
                    work_lists[b][idx].append({
                        "uid": uid, "items": items, "behavior": b, "period": period,
                        "geohash": gh, "legal_geohashes": legal_list,
                        "lng": lng, "lat": lat, "coord_rank": rank,
                        "cache_key": cache_key, "last_interact_ts": last_ts,
                    })
            if 0 <= cap and min(shard_counts[b]) >= cap:
                done[b] = True
    scan_s = time.time() - t0
    print(f"[PLAN] 扫描 {n_scanned} 个可推用户，耗时 {scan_s:.1f}s")

    dt_dir = f"{ic['hdfs_output_root']}/dt={ic['infer_end']}"
    fs.create_dir(dt_dir, recursive=True)
    rdir = ifc.run_dir(ic)
    os.makedirs(rdir, exist_ok=True)

    cold_paths = {}
    for b in behaviors:
        cold_paths[b] = f"{dt_dir}/rec_{b}_part_cold.parquet"
        hu.write_cache_file(fs, cold_paths[b], cold[b])
        for i in range(gpu_num):
            wl_path = ifc.work_list_path(rdir, b, i)
            with open(wl_path, "w", encoding="utf-8") as f:
                for item in work_lists[b][i]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        new_uids = {hu.parse_uid(ck) for ck in stats[b]["new_keys"]}
        stale_uids = {hu.parse_uid(ck) for ck in stats[b]["stale_keys"]}
        hit_uids = {hu.parse_uid(ck) for ck in stats[b]["hit_keys"]}
        cold_uids = {hu.parse_uid(ck) for ck in cold[b]}
        print(f"  [{b}] 新增: cache_key={len(stats[b]['new_keys'])} uid={len(new_uids)}  |  "
              f"过期: cache_key={len(stats[b]['stale_keys'])} uid={len(stale_uids)}  |  "
              f"命中: cache_key={len(stats[b]['hit_keys'])} uid={len(hit_uids)}")
        print(f"  [{b}] cold(命中+待定的过期) 写入 {cold_paths[b]}  "
              f"cache_key={len(cold[b])} uid={len(cold_uids)}")
        print(f"  [{b}] 各分片待推理条数: {[len(w) for w in work_lists[b]]}")

    # 全部落盘成功后再删旧文件；cold_path 跟某个旧文件路径重合的情况
    # （同一天重跑）要跳过——那是刚写完的新文件，不是待清理的旧文件
    for b in behaviors:
        for p in old_paths[b]:
            if p != cold_paths[b]:
                fs.delete_file(p)
    print("[PLAN] 旧历史文件已清理")

    return {
        "src_dt": src_dt, "scan_s": scan_s, "n_scanned": n_scanned,
        "stats": stats, "cold_counts": {b: len(cold[b]) for b in behaviors},
        "work_counts": {b: [len(w) for w in work_lists[b]] for b in behaviors},
        "baseline_cache_key": {b: len(baselines[b]) for b in behaviors},
        "baseline_uid": {b: len({hu.parse_uid(ck) for ck in baselines[b]}) for b in behaviors},
    }


def summarize_final(ic: dict, plan_result: dict) -> None:
    """全部分片成功后调用：读 dt=infer_end 下这一轮真正落地的最终状态
       （part_cold + 各分片各自写的 part{i}，glob 全部合并），打印"总的"
       （跨全部分片，不是某一个分片自己的推理统计）推理前后 cache_key/uid
       对比——这是 submit_infer.py 自己进程的输出，出现在你提交时那次
       nohup/终端日志里，不在任何一个 infer_platform_*_part{i}.log 里
       （那些只是各 GPU 自己的推理统计，看不到全局判重/合并结果）。"""
    fs = hu.get_fs()
    print("\n================ 数据并行本轮汇总 ================")
    for b in ic["behaviors"]:
        final = hu.read_dt_cache(fs, ic["hdfs_output_root"], ic["infer_end"], b)
        final_uids = {hu.parse_uid(ck) for ck in final}
        s = plan_result["stats"][b]
        new_uids = {hu.parse_uid(ck) for ck in s["new_keys"]}
        stale_uids = {hu.parse_uid(ck) for ck in s["stale_keys"]}
        hit_uids = {hu.parse_uid(ck) for ck in s["hit_keys"]}
        print(f"  [{b}]")
        print(f"    推理前 baseline: cache_key={plan_result['baseline_cache_key'][b]}  "
              f"uid={plan_result['baseline_uid'][b]}")
        print(f"    新增: cache_key={len(s['new_keys'])}  uid={len(new_uids)}  |  "
              f"过期: cache_key={len(s['stale_keys'])}  uid={len(stale_uids)}  |  "
              f"命中: cache_key={len(s['hit_keys'])}  uid={len(hit_uids)}")
        print(f"    推理后最终累计: cache_key={len(final)}  uid={len(final_uids)}")
    print("====================================================")


def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "common.conf")
    ic = ifc.load_infer_config(conf_path)
    if ic["gpu_num"] <= 1:
        raise ValueError("plan_infer.py 只在 [inference] gpu_num>1（数据并行）时有意义")
    run_planning(conf_path, ic)


if __name__ == "__main__":
    main()
