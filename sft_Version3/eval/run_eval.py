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
  4. 行为条件评测：对 clk / pay 各跑一遍——input 末尾强制接该行为 token，
     trie 约束 beam search 生成 top-K item，与 label session 中该行为的正样本
     集合比对；label 里没有该行为的用户跳过（不稀释指标）；
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
from constrained_decode import build_sid_trie, constrained_beam_search  # noqa: E402
from metrics import rank_metrics, MetricAccumulator        # noqa: E402


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
       sample: {uid, input: [(action, geo_sid)...], positives_by_action, label_date}。"""
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
            "positives_by_action": test["positives_by_action"],
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
def make_prefix(tok, items: list, behavior: str, max_len: int) -> dict:
    """[BOS] + 历史交互（按交互粒度左截断）+ 强制行为 token。
       预留 num_levels 个生成位，保证总长 <= max_len。"""
    stride = 1 + tok.num_levels
    keep = max((max_len - 2 - tok.num_levels) // stride, 1)
    ids, beh, types = tok.encode_items(items[-keep:])
    b = tok.behavior2id[behavior]
    return {
        "input_ids": [tok.bos_id] + ids + [tok.token2id[f"<{behavior}>"]],
        "behavior_ids": [-1] + beh + [b],
        "token_types": [-1] + types + [0],
    }


def eval_behavior(model, tok, trie, samples: list, behavior: str, ec: dict,
                  device, autocast_ctx, pop_ranked: list, pred_path: str) -> dict:
    """对 label 里含该行为的用户跑约束 beam search 并累计指标。
       返回 {"model": report, "pop": report, "n": 用户数, "avg_labels": 平均正样本数}。"""
    eligible = [s for s in samples
                if s["positives_by_action"].get(behavior)]
    acc, acc_pop = MetricAccumulator(ec["topk"]), MetricAccumulator(ec["topk"])
    n_labels = 0
    infer_s = 0.0                       # 纯推理耗时（beam search 部分，含 GPU 同步）
    n_batches = 0
    t0 = time.time()
    max_k = max(ec["topk"])

    with open(pred_path, "w", encoding="utf-8") as fout:
        for lo in range(0, len(eligible), ec["batch_size"]):
            batch = eligible[lo: lo + ec["batch_size"]]
            prefixes = [make_prefix(tok, s["input"], behavior, ec["max_len"])
                        for s in batch]
            t_inf = time.time()
            results = constrained_beam_search(model, tok, trie, prefixes,
                                              ec["beam_size"], device, autocast_ctx)
            infer_s += time.time() - t_inf
            n_batches += 1
            for s, beams in zip(batch, results):
                labels = s["positives_by_action"][behavior]
                ranked = [sid for sid, _ in beams]
                acc.add(rank_metrics(ranked, labels, ec["topk"]))
                acc_pop.add(rank_metrics(pop_ranked, labels, ec["topk"]))
                n_labels += len(labels)
                fout.write(json.dumps({
                    "uid": s["uid"], "behavior": behavior,
                    "label_date": s["label_date"], "labels": labels,
                    "topk": [(sid, round(sc, 4)) for sid, sc in beams[:max_k]],
                }, ensure_ascii=False) + "\n")
            done = lo + len(batch)
            if done % (ec["batch_size"] * 20) < ec["batch_size"]:
                print(f"  [{behavior}] {done}/{len(eligible)} 用户 "
                      f"({time.time() - t0:.0f}s)")
    return {"model": acc.report(), "pop": acc_pop.report(),
            "n": len(eligible), "avg_labels": n_labels / max(len(eligible), 1),
            "infer_s": infer_s, "n_batches": n_batches,
            "wall_s": time.time() - t0}


def popularity_ranked(samples: list, behavior: str, k: int) -> list:
    """基线：test 用户 input 区中该行为交互的 item 频次 top-k。"""
    cnt = Counter(sid for s in samples for a, sid in s["input"] if a == behavior)
    return [sid for sid, _ in cnt.most_common(k)]


def print_report(behavior: str, r: dict, ks: list):
    print(f"\n---- {behavior} （{r['n']} 用户，平均正样本 {r['avg_labels']:.2f}）----")
    head = f"  {'':>10}" + "".join(f"{'HR@' + str(k):>9}{'Recall@' + str(k):>10}"
                                   f"{'NDCG@' + str(k):>9}" for k in ks)
    print(head)
    for name, rep in (("GAMER", r["model"]), ("热门基线", r["pop"])):
        row = f"  {name:>8}"
        for k in ks:
            m = rep.get(k, {"hr": 0, "recall": 0, "ndcg": 0})
            row += f"{m['hr']:>9.4f}{m['recall']:>10.4f}{m['ndcg']:>9.4f}"
        print(row)
    if r["n"]:
        avg_user_ms = r["infer_s"] / r["n"] * 1000
        avg_batch_ms = r["infer_s"] / max(r["n_batches"], 1) * 1000
        print(f"  推理耗时: 总计 {r['infer_s']:.1f}s（整段流程 {r['wall_s']:.1f}s）  "
              f"平均 {avg_batch_ms:.0f}ms/batch  {avg_user_ms:.1f}ms/用户  "
              f"吞吐 {r['n'] / max(r['infer_s'], 1e-9):.1f} 用户/s")


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

    # clk/pay 占比分布
    n_clk = sum(1 for s in samples if s["positives_by_action"].get("clk"))
    n_pay = sum(1 for s in samples if s["positives_by_action"].get("pay"))
    n_both = sum(1 for s in samples if s["positives_by_action"].get("clk")
                 and s["positives_by_action"].get("pay"))
    n = max(len(samples), 1)
    print(f"[INFO] label 行为分布: 含clk {n_clk} ({n_clk / n:.1%})  "
          f"含pay {n_pay} ({n_pay / n:.1%})  两者都有 {n_both} ({n_both / n:.1%})")

    trie = build_sid_trie(tok, id2sid.values())
    print(f"[INFO] SID trie 构建完成（覆盖 {len(set(id2sid.values()))} 个真实 item）")

    os.makedirs(ec["predictions_out_dir"], exist_ok=True)
    reports = {}
    for behavior in tok.behaviors:
        pop = popularity_ranked(samples, behavior, max(ec["topk"]))
        pred_path = os.path.join(ec["predictions_out_dir"], f"pred_{behavior}.jsonl")
        reports[behavior] = eval_behavior(model, tok, trie, samples, behavior, ec,
                                          device, autocast_ctx, pop, pred_path)
        print(f"[INFO] {behavior} 预测明细已落盘: {pred_path}")

    print("\n================ 评测报表 ================")
    for behavior in tok.behaviors:
        print_report(behavior, reports[behavior], ec["topk"])
    print("\n（HR=top-K 命中任一正样本；Recall=命中数/正样本数；NDCG 按命中位置加权。"
          "热门基线 = test 用户 input 区该行为 item 频次 top-K，模型显著高于它才说明学到了个性化）")
    print("==========================================")


if __name__ == "__main__":
    main()
