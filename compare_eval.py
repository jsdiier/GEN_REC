#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_eval.py
读各实验 outputs/predictions/pred_{clk,pay}.jsonl，重算指标并对比展示
（渲染风格对齐 tf_rank/ana_result.py：fancy_grid + baseline/new + 百分位 delta）

用法:
    python3 compare_eval.py <exp>                      # 单实验：展示其 clk/pay 指标
    python3 compare_eval.py <baseline_exp> <new_exp>   # 双实验：并列对比 + delta
示例:
    python3 compare_eval.py sft_V3_tiger
    python3 compare_eval.py ntp_w_period_geo qwen_w_period_geo
    python3 compare_eval.py sft_Version3 sft_V3_tiger --topk 5,10,20

说明:
  - 实验名可为文件夹名（相对本脚本所在目录）或绝对路径；
  - 指标从预测明细现算（HR/Recall/NDCG@K + 分层前缀命中率），与 run_eval 报表同口径；
  - 分层前缀命中: depth j 命中 = top-K 内存在与任一 label 前 j 层 SID 一致的预测，
    层数按各实验数据自动识别（基线 4 层 / tiger 5 层），缺的层显示 N/A；
  - delta = (new - baseline) * 100，百分位绝对提升（百分点）；
  - 口径注意: 两实验「精确命中」粒度可能不同（4层=共享SID，5层=精确item），
    同粒度对比看两边都有的前缀层（如 L4）；热门基线不在明细里，见各自 eval 日志。
"""

import os
import json
import math
import glob
import argparse

from tabulate import tabulate

BEHAVIORS = ("clk", "pay")
METRIC_KEYS = (("hr", "HR"), ("recall", "Recall"), ("ndcg", "NDCG"))


def sid_parts(sid: str) -> list:
    return sid[1:-1].split("><")


def user_metrics(ranked: list, labels: list, ks: list) -> dict:
    label_set = set(labels)
    max_k = max(ks)
    hit_pos = [p for p, sid in enumerate(ranked[:max_k], 1) if sid in label_set]
    out = {}
    for k in ks:
        hits = [p for p in hit_pos if p <= k]
        dcg = sum(1.0 / math.log2(p + 1) for p in hits)
        idcg = sum(1.0 / math.log2(p + 1)
                   for p in range(1, min(len(label_set), k) + 1))
        out[k] = {"hr": 1.0 if hits else 0.0,
                  "recall": len(hits) / len(label_set),
                  "ndcg": dcg / idcg if idcg > 0 else 0.0}
    return out


def user_prefix(ranked: list, labels: list, ks: list, L: int) -> dict:
    max_k = max(ks)
    label_pfx = [set() for _ in range(L)]
    for sid in labels:
        parts = sid_parts(sid)
        for j in range(L):
            label_pfx[j].add(tuple(parts[:j + 1]))
    hit_at = [None] * L
    for pos, sid in enumerate(ranked[:max_k], 1):
        parts = sid_parts(sid)
        for j in range(L):
            if hit_at[j] is None and tuple(parts[:j + 1]) in label_pfx[j]:
                hit_at[j] = pos
    return {j + 1: {k: (1.0 if hit_at[j] and hit_at[j] <= k else 0.0)
                    for k in ks} for j in range(L)}


def load_behavior(exp_dir: str, pred_dir: str, behavior: str, ks: list) -> dict:
    """聚合一个行为下【全部分组】的预测明细 -> 微平均指标（run_eval.py 现在按
       (行为,时段) 分组落盘，文件名是 pred_{behavior}.jsonl（无时段词表退化态）
       或 pred_{behavior}_{period}.jsonl（如 pred_pay_bf.jsonl），这里 glob 全部
       匹配上的文件，所有组的评测实例直接池到一起算——跟 run_eval.py 报表里
       "跨时段组微平均" 同一个口径，不是先各组求平均再平均。
       返回 None 表示一个文件都没找到/全为空。"""
    base = os.path.join(exp_dir, pred_dir)
    paths = sorted(set(glob.glob(os.path.join(base, f"pred_{behavior}.jsonl")) +
                       glob.glob(os.path.join(base, f"pred_{behavior}_*.jsonl"))))
    if not paths:
        return None
    n, n_labels, L, level_tags = 0, 0, None, None
    sums = {k: {"hr": 0.0, "recall": 0.0, "ndcg": 0.0} for k in ks}
    pfx_sums, max_stored = None, None
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                ranked = [sid for sid, _ in r["topk"]]
                labels = r["labels"]
                if not labels:
                    continue
                if L is None:
                    parts = sid_parts(labels[0])
                    L = len(parts)
                    level_tags = ["geo"] + [p.split("_")[0] for p in parts[1:]]
                    pfx_sums = {j: {k: 0.0 for k in ks} for j in range(1, L + 1)}
                    max_stored = len(ranked)
                m = user_metrics(ranked, labels, ks)
                p = user_prefix(ranked, labels, ks, L)
                for k in ks:
                    for key, _ in METRIC_KEYS:
                        sums[k][key] += m[k][key]
                    for j in range(1, L + 1):
                        pfx_sums[j][k] += p[j][k]
                n += 1
                n_labels += len(labels)
    if n == 0:
        return None
    return {
        "n": n, "avg_labels": n_labels / n,
        "num_levels": L, "level_tags": level_tags, "max_stored": max_stored,
        "metrics": {k: {key: v / n for key, v in d.items()}
                    for k, d in sums.items()},
        "prefix": {j: {k: v / n for k, v in d.items()}
                   for j, d in pfx_sums.items()},
    }


def load_meta(exp_dir: str, pred_dir: str) -> dict:
    """读 run_eval.py 落的 _meta.json（各组耗时/吞吐，pred_*.jsonl 里没有这个）。
       返回 None 表示没有这个文件（旧版 run_eval.py 跑的实验，没落过这份）。"""
    path = os.path.join(exp_dir, pred_dir, "_meta.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def behavior_qps(meta: dict, behavior: str) -> dict:
    """把一个行为下全部时段组的耗时统计池到一起算总吞吐——跟精度指标的
       "跨时段组微平均" 同一个口径：总用户数 / 总推理耗时，不是先各组算吞吐
       再对组数取平均（组大小不一样，直接平均会失真）。"""
    groups = [g for key, g in meta.get("groups", {}).items()
             if key == behavior or key.startswith(f"{behavior}_")]
    n = sum(g["n"] for g in groups)
    infer_s = sum(g["infer_s"] for g in groups)
    wall_s = sum(g["wall_s"] for g in groups)
    if n == 0 or infer_s <= 0:
        return None
    return {"n": n, "infer_s": infer_s, "wall_s": wall_s,
           "qps": n / infer_s, "ms_per_user": infer_s / n * 1000}


def format_delta(delta: float) -> str:
    """百分位绝对提升（百分点），如 +2.020% / -0.150%"""
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}%"


def depth_name(tags: list, j: int) -> str:
    name = f"L{j}(" + "+".join(tags[:j]) + ")"
    return name if len(name) <= 16 else f"L{j}(..+{tags[j - 1]})"


def print_behavior(behavior: str, ks: list, results: dict):
    """results: {exp_name: 聚合指标}，1 个 = 展示，2 个 = baseline/new 对比。"""
    exps = list(results)
    pair = len(exps) == 2

    print("\n" + "=" * 60)
    print(f"  {behavior.upper()}  " + "   ".join(
        f"{e}: {r['n']}用户/均{r['avg_labels']:.2f}正样本/{r['num_levels']}层SID"
        for e, r in results.items()))
    print("=" * 60)

    if pair:
        cols = [f"baseline\n{exps[0]}", f"new\n{exps[1]}", "absolutely\ndelta"]
    else:
        cols = [exps[0]]

    # ── 主指标表：行 = 指标@K ──
    rows = []
    for k in ks:
        for key, label in METRIC_KEYS:
            vals = [results[e]["metrics"][k][key] for e in exps]
            row = [f"{label}@{k}"] + [f"{v:.4f}" for v in vals]
            if pair:
                row.append(format_delta((vals[1] - vals[0]) * 100))
            rows.append(row)
    print(tabulate(rows, headers=[behavior.upper()] + cols,
                   tablefmt="fancy_grid", stralign="center",
                   disable_numparse=True))

    # ── 分层前缀命中率：行 = 深度 x K ──
    max_L = max(r["num_levels"] for r in results.values())
    tags = max(results.values(), key=lambda r: r["num_levels"])["level_tags"]
    rows2 = []
    for j in range(1, max_L + 1):
        for k in ks:
            vals = [r["prefix"].get(j, {}).get(k) for r in results.values()]
            row = [f"{depth_name(tags, j)} @{k}"] + \
                  [f"{v:.4f}" if v is not None else "N/A" for v in vals]
            if pair:
                row.append(format_delta((vals[1] - vals[0]) * 100)
                           if None not in vals else "—")
            rows2.append(row)
    print(tabulate(rows2, headers=["前缀命中(方向诊断)"] + cols,
                   tablefmt="fancy_grid", stralign="center",
                   disable_numparse=True))


def print_qps(behavior: str, results: dict):
    """results: {exp_name: behavior_qps() 返回值}，1 个 = 展示，2 个 = 对比。
       缺 _meta.json 的实验（旧版 run_eval.py 跑的）直接跳过，不参与这张表。"""
    exps = [e for e in results if results[e] is not None]
    if not exps:
        return
    pair = len(exps) == 2
    cols = ([f"baseline\n{exps[0]}", f"new\n{exps[1]}", "倍数\nnew/baseline"]
           if pair else [exps[0]])
    rows = [
        ["吞吐(用户/s)"] + [f"{results[e]['qps']:.2f}" for e in exps],
        ["单用户耗时(ms)"] + [f"{results[e]['ms_per_user']:.1f}" for e in exps],
        ["评测用户数"] + [str(results[e]["n"]) for e in exps],
    ]
    if pair:
        base_qps, new_qps = results[exps[0]]["qps"], results[exps[1]]["qps"]
        rows[0].append(f"{new_qps / base_qps:.2f}x")
        base_ms, new_ms = results[exps[0]]["ms_per_user"], results[exps[1]]["ms_per_user"]
        rows[1].append(f"{new_ms / base_ms:.2f}x")
        rows[2].append("—")
    print(tabulate(rows, headers=[f"{behavior.upper()} 推理速度"] + cols,
                   tablefmt="fancy_grid", stralign="center",
                   disable_numparse=True))


def main():
    ap = argparse.ArgumentParser(description="对比各实验的 eval 预测明细指标")
    ap.add_argument("exps", nargs="+",
                    help="1~2 个实验（文件夹名或绝对路径；两个时为 baseline new）")
    ap.add_argument("--topk", default="5,10,20", help="指标 K 列表，逗号分隔")
    ap.add_argument("--pred-dir", default="outputs/predictions",
                    help="实验目录下预测明细的相对路径")
    args = ap.parse_args()
    if len(args.exps) > 2:
        ap.error("最多两个实验（baseline new）")
    ks = sorted(int(k) for k in args.topk.split(","))
    root = os.path.dirname(os.path.abspath(__file__))

    results = {b: {} for b in BEHAVIORS}
    qps_results = {b: {} for b in BEHAVIORS}
    for exp in args.exps:
        exp_dir = exp if os.path.isabs(exp) else os.path.join(root, exp)
        name = os.path.basename(exp_dir.rstrip("/"))
        print(f"[INFO] 实验 {name}: {exp_dir}")
        for b in BEHAVIORS:
            r = load_behavior(exp_dir, args.pred_dir, b, ks)
            if r is None:
                print(f"  ⚠  {b}: 找不到或为空: "
                      f"{os.path.join(exp_dir, args.pred_dir, f'pred_{b}*.jsonl')}")
                continue
            if max(ks) > r["max_stored"]:
                print(f"  ⚠  {b}: 明细只存了 top{r['max_stored']}，"
                      f"K={max(ks)} 按截断计算")
            results[b][name] = r

        meta = load_meta(exp_dir, args.pred_dir)
        if meta is None:
            print(f"  ⚠  没有 _meta.json（旧版 run_eval.py 跑的实验没落过这份），"
                  f"跳过 QPS 对比")
        else:
            for b in BEHAVIORS:
                qps_results[b][name] = behavior_qps(meta, b)

    for b in BEHAVIORS:
        if results[b]:
            print_behavior(b, ks, results[b])

    if len(args.exps) == 2:
        print("\n[口径提醒] 两实验「精确命中」粒度不同时（4层=共享SID，5层=精确item），"
              "delta 只在同粒度层（如 L4）上公平；热门基线见各自 eval 日志。")

    print("\n" + "=" * 60)
    print("  推理速度对比（跨时段组池到一起算总吞吐，同精度指标口径）")
    print("=" * 60)
    for b in BEHAVIORS:
        print_qps(b, qps_results[b])


if __name__ == "__main__":
    main()
