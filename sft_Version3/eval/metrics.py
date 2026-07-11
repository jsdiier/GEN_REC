#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics: top-K 推荐指标（单用户粒度算，run_eval 汇总平均）。

约定：ranked 为模型生成的 item(geo_sid) 降序列表；labels 为该用户 label session
内该行为的正样本集合（去重）。多正样本的定义：
  HR@K     = top-K 内命中任意一个正样本则 1 否则 0（命中率）
  Recall@K = |top-K ∩ labels| / |labels|
  NDCG@K   = DCG/IDCG，DCG = Σ_命中位置p 1/log2(p+1)（p 从 1 数），
             IDCG = 前 min(|labels|, K) 个位置全命中的 DCG（理想排列）
"""

import math


def rank_metrics(ranked: list, labels, ks: list) -> dict:
    """返回 {K: {"hr": 0/1, "recall": float, "ndcg": float}}。labels 需非空。"""
    label_set = set(labels)
    assert label_set, "labels 不能为空（无正样本的用户应在上游跳过）"
    max_k = max(ks)
    hit_pos = [p for p, sid in enumerate(ranked[:max_k], start=1) if sid in label_set]

    out = {}
    for k in ks:
        hits_k = [p for p in hit_pos if p <= k]
        dcg = sum(1.0 / math.log2(p + 1) for p in hits_k)
        idcg = sum(1.0 / math.log2(p + 1)
                   for p in range(1, min(len(label_set), k) + 1))
        out[k] = {
            "hr": 1.0 if hits_k else 0.0,
            "recall": len(hits_k) / len(label_set),
            "ndcg": dcg / idcg if idcg > 0 else 0.0,
        }
    return out


class MetricAccumulator:
    """逐用户累加，report() 给平均。"""

    def __init__(self, ks: list):
        self.ks = ks
        self.n = 0
        self.sums = {k: {"hr": 0.0, "recall": 0.0, "ndcg": 0.0} for k in ks}

    def add(self, per_k: dict):
        self.n += 1
        for k in self.ks:
            for name in ("hr", "recall", "ndcg"):
                self.sums[k][name] += per_k[k][name]

    def report(self) -> dict:
        """{K: {"hr":均值, "recall":均值, "ndcg":均值}}；n=0 时返回空。"""
        if self.n == 0:
            return {}
        return {k: {name: v / self.n for name, v in d.items()}
                for k, d in self.sums.items()}
