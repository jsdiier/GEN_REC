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

流程：
  1. 读 common.conf [inference]（窗口/上限/行为/topk/beam/ckpt/输出目录）
     与 [train]（vocab_path/max_len）；
  2. 加载 ckpt + item map 建 SID 前缀树与 sid->item_ids 反查表；
  3. 流式拉窗口数据（复用 step1->2 + step3 的时间线/session 工具），逐用户拼
     全历史前缀，按 conf 配置的每个行为强制接行为 token，trie 约束 beam search
     生成 top-K item；
  4. 边收边推边写（不整窗攒内存）：结果存 parquet（比 jsonl 省一个量级存储），
     目录按推理窗口命名：{output_dir}/{infer_start}_{infer_end}/rec_{behavior}.parquet，
     行 = 一条推荐（uid, rank, sid, score, item_ids），每用户每行为 topk 行；
     同目录附 _meta.json（本次配置 + 统计），同窗口重跑会覆盖；
  5. 结束打印每行为与合计的纯推理 QPS、端到端 QPS（含取数/写盘）。

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
        "topk": topk,
        "beam_size": max(cp.getint("inference", "beam_size", fallback=topk), topk),
        "batch_size": cp.getint("inference", "batch_size", fallback=32),
        "ckpt_path": path("ckpt_path", "outputs/ckpt/latest/best.pt"),
        "output_dir": path("output_dir", "outputs/inference"),
        "vocab_path": (lambda v: v if os.path.isabs(v) else os.path.join(root, v))(
            cp.get("train", "vocab_path", fallback="outputs/vocab.json")),
        "max_len": cp.getint("train", "max_len", fallback=512),
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
        items = [(t["action"], t["geo_sid"]) for _, toks in sessions for t in toks]
        if not items:
            continue                          # 清洗后没有任何可用交互
        yield uid, items
        n += 1
        if 0 < ic["infer_max_num"] <= n:
            break


# ------------------------------------------------------------------
# 输出：流式 parquet
# ------------------------------------------------------------------
class ParquetRecWriter:
    """流式 parquet 写入：每条推荐一行（uid/rank/sid/score/item_ids），
       攒满 buffer_rows 写一个 row group，不整窗攒内存。用完必须 close()。"""
    SCHEMA = pa.schema([
        ("uid", pa.string()),
        ("rank", pa.int32()),                 # 1 = 置信最高
        ("sid", pa.string()),
        ("score", pa.float32()),              # 4 个 SID token 的累计 logprob
        ("item_ids", pa.list_(pa.string())),  # sid 反查（一对多）
    ])

    def __init__(self, path: str, sid2items: dict, topk: int,
                 buffer_rows: int = 50000):
        self.sid2items = sid2items
        self.topk = topk
        self.buffer_rows = buffer_rows
        self.writer = pq.ParquetWriter(path, self.SCHEMA)
        self.buf = []

    def write_user(self, uid, beams):
        for rank, (sid, score) in enumerate(beams[:self.topk], start=1):
            self.buf.append({
                "uid": str(uid), "rank": rank, "sid": sid, "score": float(score),
                "item_ids": [str(x) for x in self.sid2items.get(sid, [])],
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


# ------------------------------------------------------------------
# 推理主体（与数据源/输出格式解耦，便于单测）
# ------------------------------------------------------------------
def generate_for_users(model, tok, trie, user_iter, ic: dict,
                       device, autocast_ctx, writers: dict) -> dict:
    """按 batch 收用户 -> 每个行为各跑一次约束 beam search -> 逐用户写 writer。
       返回 {behavior: {"n": 用户数, "infer_s": 纯推理秒}}。"""
    stats = {b: {"n": 0, "infer_s": 0.0} for b in ic["behaviors"]}

    def flush(batch):
        for behavior in ic["behaviors"]:
            prefixes = [make_prefix(tok, items, behavior, ic["max_len"])
                        for _, items in batch]
            t0 = time.time()
            results = constrained_beam_search(model, tok, trie, prefixes,
                                              ic["beam_size"], device, autocast_ctx)
            stats[behavior]["infer_s"] += time.time() - t0
            stats[behavior]["n"] += len(batch)
            for (uid, _), beams in zip(batch, results):
                writers[behavior].write_user(uid, beams)

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
    print(f"[INFO] behaviors={ic['behaviors']}  topk={ic['topk']}  "
          f"beam={ic['beam_size']}  batch={ic['batch_size']}")

    tok = SIDTokenizer.load(ic["vocab_path"])
    for b in ic["behaviors"]:
        if b not in tok.behaviors:
            raise ValueError(f"未知行为 {b!r}（词表行为: {tok.behaviors}）")
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
    paths = {b: os.path.join(run_dir, f"rec_{b}.parquet") for b in ic["behaviors"]}
    writers = {b: ParquetRecWriter(p, sid2items, ic["topk"])
               for b, p in paths.items()}
    t_start = time.time()
    try:
        stats = generate_for_users(model, tok, trie,
                                   iter_infer_users(conf_path, ic, id2sid),
                                   ic, device, autocast_ctx, writers)
    finally:
        for w in writers.values():
            w.close()
    wall_s = time.time() - t_start

    total_req = sum(s["n"] for s in stats.values())      # 1 用户 x 1 行为 = 1 次生成
    total_inf = sum(s["infer_s"] for s in stats.values())
    print("\n================ 推理完成 ================")
    for b in ic["behaviors"]:
        s = stats[b]
        if s["n"]:
            print(f"  {b}: {s['n']} 用户  纯推理 {s['infer_s']:.1f}s  "
                  f"QPS {s['n'] / max(s['infer_s'], 1e-9):.1f}  "
                  f"({s['infer_s'] / s['n'] * 1000:.1f}ms/用户)")
        print(f"  结果: {paths[b]}")
    print(f"  合计: {total_req} 次生成  纯推理 QPS "
          f"{total_req / max(total_inf, 1e-9):.1f}  |  全流程 {wall_s:.1f}s"
          f"（含取数/编码/写盘）端到端 QPS {total_req / max(wall_s, 1e-9):.1f}")
    print("==========================================")

    meta = {
        "infer_start": ic["infer_start"], "infer_end": ic["infer_end"],
        "infer_max_num": ic["infer_max_num"], "behaviors": ic["behaviors"],
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
