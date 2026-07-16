#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval: 独立 test 窗口的端到端评测（GAMER 推理 -> HR/Recall/NDCG 报表）。

流程：
  1. 读 common.conf：[data] test_start/test_end/test_sample_rate（test 集圈定），
     [eval] ckpt/topk/beam/batch（评测行为），[train] vocab_path/max_len；
  2. 取数复用 step1->2->3 内存链（把窗口换成 test 窗口）：直接取 step3 三分法
     产出的 test 样本（input=S1..S(m-1), label=Sm；m>=3 的用户才有）。
     三分法下最后一个 session 天然是 train/val 都没见过的，无需日期过滤，
     test 窗口可以与训练窗口同天。用户抽样：
       - crc32(uid) % 10000 < test_sample_rate*10000（确定性抽样；test_max_num>0
         时攒够即早停供小规模快测，-1 扫完整窗保证严格可复现）；
  3. 加载 ckpt（默认 best.pt）+ 用 item map 全量真实 item 建 SID 前缀树；
  4. 条件评测：按 (行为 x 时段) 组合各跑一遍（词表无时段位时退化为按行为）——
     input 末尾强制接 <时段><行为> token，trie 约束 beam search 生成 top-K item，
     与 label session 中该组的正样本集合比对；label 里没有该组的用户跳过；
  5. 报表：分行为 x 分 K 的 HR/Recall/NDCG + 热门 item 基线对照 +
     clk/pay 测试数据占比分布；预测明细落盘 outputs/predictions/ 供 case 分析。

用法:
    python run_eval.py [common.conf]
"""

import os
import sys
import json
import time
import zlib
import configparser
import contextlib
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("train", os.path.join("train", "model"), "data", "inference", "eval"):
    sys.path.insert(0, os.path.join(ROOT, p))

from tokenizer_sid import SIDTokenizer                     # noqa: E402
from train_sft import load_checkpoint                      # noqa: E402
from constrained_decode import (build_sid_trie, constrained_beam_search,  # noqa: E402
                                make_prefix)
from metrics import (rank_metrics, MetricAccumulator,      # noqa: E402
                     prefix_hr, PrefixHRAccumulator, _sid_parts)

try:
    from tabulate import tabulate
except ImportError:                                        # 未安装时降级为简易对齐
    tabulate = None


def _render_table(rows: list, headers: list) -> str:
    if tabulate:
        return tabulate(rows, headers=headers, tablefmt="github",
                        stralign="right", numalign="right")
    widths = [max(len(str(x)) for x in [h] + [r[i] for r in rows])
              for i, h in enumerate(headers)]
    def line(cells):
        return "  ".join(str(c).rjust(w) for c, w in zip(cells, widths))
    return "\n".join([line(headers), line(["-" * w for w in widths])] +
                     [line(r) for r in rows])


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------
def load_eval_config(conf_path: str) -> dict:
    cp = configparser.ConfigParser()
    if not cp.read(conf_path, encoding="utf-8"):
        raise FileNotFoundError(f"找不到配置文件: {conf_path}")
    root = os.path.dirname(os.path.abspath(conf_path))

    def path(section, key, default):
        v = cp.get(section, key, fallback=default)
        return v if os.path.isabs(v) else os.path.join(root, v)

    topk = [int(k) for k in cp.get("eval", "topk", fallback="5,10,20").split(",")]
    return {
        "test_start": cp.get("data", "test_start"),
        "test_end": cp.get("data", "test_end"),
        "test_sample_rate": cp.getfloat("data", "test_sample_rate"),
        "test_max_num": cp.getint("data", "test_max_num", fallback=-1),
        "vocab_path": path("train", "vocab_path", "outputs/vocab.json"),
        "max_len": cp.getint("train", "max_len", fallback=512),
        "ckpt_path": path("eval", "ckpt_path", "outputs/ckpt/best.pt"),
        "topk": sorted(topk),
        "beam_size": cp.getint("eval", "beam_size", fallback=max(topk)),
        "batch_size": cp.getint("eval", "batch_size", fallback=32),
        "predictions_out_dir": path("eval", "predictions_out_dir", "outputs/predictions"),
    }


# ------------------------------------------------------------------
# test 样本收集（复用 step3 内存链，窗口换成 test 窗口）
# ------------------------------------------------------------------
def collect_test_samples(conf_path: str, ec: dict) -> tuple:
    """扫完整个 test 窗口，返回 (samples, stats, id2sid)，id2sid 复用给 trie。
       sample: {uid, input: [(action, geo_sid)...], positives_grouped,
                favor_coord_raw, label_date}。"""
    import step3_build_samples as step3
    from step1_get_user_action import load_config

    cfg = load_config(conf_path)
    cfg["train_start"], cfg["train_end"] = ec["test_start"], ec["test_end"]
    cfg["max_num"] = -1                       # 上限由 test_max_num 控制
    cfg["behavior_drop_x"] = 0                # test 不需要 train 样本，关掉增强
    cfg["min_train_seq_len"] = 10 ** 9        # 让 build_train_sequences 直接返回空

    thresh = max(int(ec["test_sample_rate"] * 10000), 1)

    id2sid = step3.load_item_sid_map(step3.get_item_map_path(conf_path))
    samples = []
    stats = Counter()
    for uid, _sessions, user_samples in step3.iter_user_samples(
            cfg, conf_path, id2sid=id2sid, verbose=True):
        stats["rows_scanned"] += 1
        if zlib.crc32(str(uid).encode("utf-8")) % 10000 >= thresh:
            continue
        stats["uid_sampled"] += 1
        test = next((s for s in user_samples if s["split"] == "test"), None)
        if test is None:
            stats["skip_lt3_sessions"] += 1   # m<3 出不了 test（m=2 归 val）
            continue
        samples.append({
            "uid": test["uid"],
            "input": step3._slim(test["input"]),
            "positives_grouped": test["positives_grouped"],
            "favor_coord_raw": test["favor_coord_raw"],
            "label_date": test["label_date"],
        })
        if 0 < ec["test_max_num"] <= len(samples):
            stats["early_stopped"] = 1        # 小测早停：集合依赖扫描前缀
            break
    stats["test_users"] = len(samples)
    return samples, stats, id2sid


# ------------------------------------------------------------------
# 行为条件评测
# ------------------------------------------------------------------
def group_labels(s: dict, behavior: str, period: str = None):
    """取一个 test 样本在 (时段, 行为) 组的正样本列表；无则返回 None。
       positives_grouped 的 key 固定是 "{meal_period}_{action}"（meal_period 在
       step3 无条件算出，跟词表有没有时段位无关）。词表没有时段位时
       （period=None，没法在生成条件里指定时段）退化成把该行为下全部时段的
       正样本合并，等价于旧版 positives_by_action。"""
    if period is not None:
        return s["positives_grouped"].get(f"{period}_{behavior}")
    merged = []
    for key, geo_sids in s["positives_grouped"].items():
        if key.endswith(f"_{behavior}"):
            merged.extend(g for g in geo_sids if g not in merged)
    return merged or None


def eval_behavior(model, tok, trie, samples: list, behavior: str, ec: dict,
                  device, autocast_ctx, pop_ranked: list, pred_path: str,
                  period: str = None) -> dict:
    """对 label 里含该 (行为[, 时段]) 组的用户跑约束 beam search 并累计指标。
       词表含时段位时按 <时段><行为> 条件生成，与 label 中该组的正样本比对。
       返回 {"model": report, "pop": report, "n": 用户数, "avg_labels": 平均正样本数}。"""
    eligible = [s for s in samples if group_labels(s, behavior, period)]
    acc, acc_pop = MetricAccumulator(ec["topk"]), MetricAccumulator(ec["topk"])
    L = tok.num_levels
    pfx, pfx_pop = PrefixHRAccumulator(ec["topk"], L), PrefixHRAccumulator(ec["topk"], L)
    n_labels = 0
    infer_s = 0.0                       # 纯推理耗时（beam search 部分，含 GPU 同步）
    n_batches = 0
    t0 = time.time()
    max_k = max(ec["topk"])

    with open(pred_path, "w", encoding="utf-8") as fout:
        for lo in range(0, len(eligible), ec["batch_size"]):
            batch = eligible[lo: lo + ec["batch_size"]]
            prefixes = [make_prefix(tok, s["input"], behavior, ec["max_len"],
                                    period=period)
                        for s in batch]
            t_inf = time.time()
            results = constrained_beam_search(model, tok, trie, prefixes,
                                              ec["beam_size"], device, autocast_ctx)
            infer_s += time.time() - t_inf
            n_batches += 1
            for s, beams in zip(batch, results):
                labels = group_labels(s, behavior, period)
                ranked = [sid for sid, _ in beams]
                acc.add(rank_metrics(ranked, labels, ec["topk"]))
                acc_pop.add(rank_metrics(pop_ranked, labels, ec["topk"]))
                pfx.add(prefix_hr(ranked, labels, ec["topk"], L))
                pfx_pop.add(prefix_hr(pop_ranked, labels, ec["topk"], L))
                n_labels += len(labels)
                fout.write(json.dumps({
                    "uid": s["uid"], "behavior": behavior, "period": period,
                    "label_date": s["label_date"], "labels": labels,
                    "topk": [(sid, round(sc, 4)) for sid, sc in beams[:max_k]],
                }, ensure_ascii=False) + "\n")
            done = lo + len(batch)
            if done % (ec["batch_size"] * 20) < ec["batch_size"]:
                tag = behavior if period is None else f"{behavior}x{period}"
                print(f"  [{tag}] {done}/{len(eligible)} 用户 "
                      f"({time.time() - t0:.0f}s)")
    return {"model": acc.report(), "pop": acc_pop.report(),
            "prefix_model": pfx.report(), "prefix_pop": pfx_pop.report(),
            "n": len(eligible), "avg_labels": n_labels / max(len(eligible), 1),
            "infer_s": infer_s, "n_batches": n_batches,
            "wall_s": time.time() - t0}


def popularity_ranked(samples: list, behavior: str, k: int,
                      period: str = None) -> list:
    """基线：test 用户 input 区中该 (行为[, 时段]) 组交互的 item 频次 top-k。
       input 项为 (action, geo_sid[, period]) 元组。"""
    cnt = Counter(it[1] for s in samples for it in s["input"]
                  if it[0] == behavior and
                  (period is None or (len(it) > 2 and it[2] == period)))
    return [sid for sid, _ in cnt.most_common(k)]


def print_report(behavior: str, r: dict, ks: list, level_tags: list):
    print(f"\n---- {behavior} （{r['n']} 用户，平均正样本 {r['avg_labels']:.2f}）----")
    headers = ["模型"] + [f"{m}@{k}" for k in ks for m in ("HR", "Recall", "NDCG")]
    rows = []
    for name, rep in (("GAMER", r["model"]), ("热门基线", r["pop"])):
        rows.append([name] + [f"{rep.get(k, {}).get(key, 0.0):.4f}"
                              for k in ks for key in ("hr", "recall", "ndcg")])
    print(_render_table(rows, headers))

    # 分层前缀命中率：depth j = 预测的前 j 层 SID 与任一 label 前 j 层一致
    # （L1 对 = 地方对了，逐层收窄；最深层 = 精确命中，应等于上表 HR）
    print(f"\n  分层前缀命中率 HR@K（GAMER / 热门基线）:")
    headers2 = ["前缀深度"] + [f"@{k}" for k in ks]
    rows2 = []
    for j in range(1, len(level_tags) + 1):
        name = f"L{j}(" + "+".join(level_tags[:j]) + ")"
        if len(name) > 14:
            name = f"L{j}(..+{level_tags[j - 1]})"
        rows2.append([name] + [
            f"{r['prefix_model'].get(j, {}).get(k, 0.0):.4f} / "
            f"{r['prefix_pop'].get(j, {}).get(k, 0.0):.4f}" for k in ks])
    print(_render_table(rows2, headers2))
    if r["n"]:
        avg_user_ms = r["infer_s"] / r["n"] * 1000
        avg_batch_ms = r["infer_s"] / max(r["n_batches"], 1) * 1000
        print(f"  推理耗时: 总计 {r['infer_s']:.1f}s（整段流程 {r['wall_s']:.1f}s）  "
              f"平均 {avg_batch_ms:.0f}ms/batch  {avg_user_ms:.1f}ms/用户  "
              f"吞吐 {r['n'] / max(r['infer_s'], 1e-9):.1f} 用户/s")


def composite_report(reports: dict, behavior: str, periods: list) -> dict:
    """同一 behavior 下所有 (period) 组直接按样本量加权平均（微平均，不是先各组
       求平均再对组数取平均）：等价于把该 behavior 下全部组的评测实例池到一起
       统一算，样本量大的组自然权重更大。pay/clk 分开算，不跨行为聚合。
       以后加 geohash_rank 维度，这里的 groups 换成 (period, geohash_rank) 的
       笛卡尔积即可，公式不用变。"""
    groups = [reports[(behavior, p)] for p in periods if reports[(behavior, p)]["n"] > 0]
    total_n = sum(g["n"] for g in groups)
    if total_n == 0:
        return {}
    out = {"n": total_n}
    for name in ("model", "pop"):
        ks = list(groups[0][name].keys())
        out[name] = {k: {m: sum(g[name][k][m] * g["n"] for g in groups) / total_n
                         for m in ("hr", "recall", "ndcg")}
                     for k in ks}
    return out


def print_composite(behavior: str, comp: dict, ks: list):
    if not comp:
        print(f"\n---- {behavior}（跨时段组微平均）：无有效分组，跳过 ----")
        return
    print(f"\n---- {behavior}（跨全部时段组微平均，共 {comp['n']} 条评测实例）----")
    headers = ["模型"] + [f"{m}@{k}" for k in ks for m in ("HR", "Recall", "NDCG")]
    rows = []
    for name, tag in (("model", "GAMER"), ("pop", "热门基线")):
        rows.append([tag] + [f"{comp[name][k][m]:.4f}"
                             for k in ks for m in ("hr", "recall", "ndcg")])
    print(_render_table(rows, headers))


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "common.conf")
    ec = load_eval_config(conf_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    else:
        autocast_ctx = contextlib.nullcontext
    print(f"[INFO] device={device}")
    cap = (f"max_num={ec['test_max_num']}（攒够早停，小测模式）"
           if ec["test_max_num"] > 0 else "扫完整窗（正式评测）")
    print(f"[INFO] test 窗口: {ec['test_start']} ~ {ec['test_end']}  "
          f"sample_rate={ec['test_sample_rate']}（crc32(uid)）  {cap}")
    print(f"[INFO] ckpt={ec['ckpt_path']}  topk={ec['topk']}  "
          f"beam={ec['beam_size']}  batch={ec['batch_size']}")

    tok = SIDTokenizer.load(ec["vocab_path"])
    model, cfg, meta = load_checkpoint(ec["ckpt_path"], map_location=device)
    model.to(device).eval()
    print(f"[INFO] 模型加载完成: epoch={meta['epoch']}  val_loss={meta['val_loss']:.4f}  "
          f"参数量={model.num_parameters() / 1e6:.2f}M")

    samples, stats, id2sid = collect_test_samples(conf_path, ec)
    print(f"\n[INFO] test 集收集完成: 扫描 {stats['rows_scanned']} 行, "
          f"crc32 抽中 {stats['uid_sampled']}, "
          f"剔除 <3 session {stats['skip_lt3_sessions']}, "
          f"最终 {stats['test_users']} 用户")

    # (行为[, 时段]) 组的用户覆盖分布；词表无时段位时退化为单纯行为组
    periods = list(getattr(tok, "periods", [])) or [None]
    groups = [(b, p) for b in tok.behaviors for p in periods]
    n = max(len(samples), 1)
    dist = "  ".join(
        f"{b if p is None else f'{b}x{p}'} "
        f"{sum(1 for s in samples if group_labels(s, b, p))} "
        f"({sum(1 for s in samples if group_labels(s, b, p)) / n:.1%})"
        for b, p in groups)
    print(f"[INFO] label 组覆盖: {dist}")

    trie = build_sid_trie(tok, id2sid.values())
    print(f"[INFO] SID trie 构建完成（覆盖 {len(set(id2sid.values()))} 个真实 item）")
    sample_parts = _sid_parts(next(iter(id2sid.values())))
    level_tags = ["geo"] + [p.split("_")[0] for p in sample_parts[1:]]

    os.makedirs(ec["predictions_out_dir"], exist_ok=True)
    reports = {}
    for b, p in groups:
        tag = b if p is None else f"{b}_{p}"
        pop = popularity_ranked(samples, b, max(ec["topk"]), period=p)
        pred_path = os.path.join(ec["predictions_out_dir"], f"pred_{tag}.jsonl")
        reports[(b, p)] = eval_behavior(model, tok, trie, samples, b, ec,
                                        device, autocast_ctx, pop, pred_path,
                                        period=p)
        print(f"[INFO] {tag} 预测明细已落盘: {pred_path}")

    print("\n================ 评测报表 ================")
    for b, p in groups:
        print_report(b if p is None else f"{b} x {p}",
                     reports[(b, p)], ec["topk"], level_tags)
    print("\n（HR=top-K 命中任一正样本；Recall=命中数/正样本数；NDCG 按命中位置加权。"
          "热门基线 = test 用户 input 区该行为 item 频次 top-K，模型显著高于它才说明学到了个性化）")
    print("==========================================")

    print("\n============ 综合指标（分行为，跨时段组微平均） ============")
    for b in tok.behaviors:
        print_composite(b, composite_report(reports, b, periods), ec["topk"])
    print("\n（同一行为下所有时段组的评测实例直接池到一起算，不是先各组求平均再对"
          "组数取平均——样本量大的组权重自然更大。pay/clk 分开看，不跨行为聚合。）")
    print("==========================================")


if __name__ == "__main__":
    main()
