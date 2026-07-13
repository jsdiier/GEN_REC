#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_eval: 读各实验 outputs/predictions/pred_{clk,pay}.jsonl，重算指标并列对比。

用法（在 GEN_REC 目录下）:
    python3 compare_eval.py sft_V3_tiger                # 单实验：展示其 clk/pay 指标
    python3 compare_eval.py sft_V3_tiger sft_Version3   # 多实验：并列对比
    python3 compare_eval.py A B --topk 5,10,20          # 自定义 K

说明:
  - 指标从预测明细现算（HR/Recall/NDCG@K + 分层前缀命中率），与 run_eval 报表同口径；
  - 分层前缀命中: depth j 命中 = top-K 内存在与任一 label 前 j 层 SID 一致的预测，
    层数按各实验数据自动识别（基线 4 层 / tiger 5 层）；
  - 跨实验对比注意口径: 精确命中的粒度不同（基线=共享 SID，tiger=精确 item），
    同粒度对比请看两边都有的前缀层（如 L4）；热门基线不在明细里，见各自 eval 日志。
"""

import os
import json
import math
import argparse

from tabulate import tabulate

BEHAVIORS = ("clk", "pay")


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


def load_behavior(path: str, ks: list) -> dict:
    """聚合一个 pred_{behavior}.jsonl -> 平均指标。返回 None 表示文件缺失。"""
    if not os.path.isfile(path):
        return None
    n, n_labels, L, level_tags = 0, 0, None, None
    sums = {k: {"hr": 0.0, "recall": 0.0, "ndcg": 0.0} for k in ks}
    pfx_sums = None
    max_stored = None
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
                for key in ("hr", "recall", "ndcg"):
                    sums[k][key] += m[k][key]
                for j in range(1, L + 1):
                    pfx_sums[j][k] += p[j][k]
            n += 1
            n_labels += len(labels)
    if n == 0:
        return None
    return {
        "n": n, "avg_labels": n_labels / n,
        "num_levels": L, "level_tags": level_tags,
        "max_stored": max_stored,
        "metrics": {k: {key: v / n for key, v in d.items()}
                    for k, d in sums.items()},
        "prefix": {j: {k: v / n for k, v in d.items()}
                   for j, d in pfx_sums.items()},
    }


def main():
    ap = argparse.ArgumentParser(description="对比各实验的 eval 预测明细指标")
    ap.add_argument("exps", nargs="+", help="实验文件夹名（如 sft_V3_tiger sft_Version3）")
    ap.add_argument("--topk", default="5,10,20", help="指标 K 列表，逗号分隔")
    ap.add_argument("--pred-dir", default="outputs/predictions",
                    help="实验目录下预测明细的相对路径")
    args = ap.parse_args()
    ks = sorted(int(k) for k in args.topk.split(","))
    root = os.path.dirname(os.path.abspath(__file__))

    # 加载: results[behavior][exp] = 聚合指标
    results = {b: {} for b in BEHAVIORS}
    for exp in args.exps:
        for b in BEHAVIORS:
            path = os.path.join(root, exp, args.pred_dir, f"pred_{b}.jsonl")
            r = load_behavior(path, ks)
            if r is None:
                print(f"[WARN] {exp}/{b}: 找不到或为空: {path}")
                continue
            if max(ks) > r["max_stored"]:
                print(f"[WARN] {exp}/{b}: 明细只存了 top{r['max_stored']}，"
                      f"K={max(ks)} 的指标按截断计算")
            results[b][exp] = r

    for b in BEHAVIORS:
        if not results[b]:
            continue
        print(f"\n================ {b} ================")
        info = "  ".join(f"{exp}: {r['n']}用户/均{r['avg_labels']:.2f}正样本"
                         f"/{r['num_levels']}层SID"
                         for exp, r in results[b].items())
        print(info)

        headers = ["实验"] + [f"{m}@{k}" for k in ks
                              for m in ("HR", "Recall", "NDCG")]
        rows = [[exp] + [f"{r['metrics'][k][key]:.4f}"
                         for k in ks for key in ("hr", "recall", "ndcg")]
                for exp, r in results[b].items()]
        print(tabulate(rows, headers=headers, tablefmt="github",
                       stralign="right"))

        print(f"\n分层前缀命中率 HR@K（L1 对=地方对，逐层收窄；最深层=精确命中）:")
        headers2 = ["实验", "前缀深度"] + [f"@{k}" for k in ks]
        rows2 = []
        for exp, r in results[b].items():
            for j in range(1, r["num_levels"] + 1):
                name = f"L{j}(" + "+".join(r["level_tags"][:j]) + ")"
                if len(name) > 16:
                    name = f"L{j}(..+{r['level_tags'][j - 1]})"
                rows2.append([exp, name] +
                             [f"{r['prefix'][j][k]:.4f}" for k in ks])
        print(tabulate(rows2, headers=headers2, tablefmt="github",
                       stralign="right"))

    if len(args.exps) > 1:
        print("\n[口径提醒] 各实验「精确命中」粒度不同（4层=共享SID，5层=精确item），"
              "跨实验同粒度对比请看两边都有的前缀层（如 L4）。")


if __name__ == "__main__":
    main()
