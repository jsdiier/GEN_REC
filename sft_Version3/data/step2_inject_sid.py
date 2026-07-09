#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2: 把用户行为序列（seq_fields，如 u_pay_item_seq_100 / u_clk_item_seq_100）
里每个 item_id 映射成 geo_sid。多字段「各自独立」处理。

  - 复用 step1 的流式取数与序列解析；
  - 加载本地 item_id -> geo_sid 映射表（common.conf [item_map] item_map_path）；
  - 每个选中序列字段各自映射、各自出一份缺失率分布；
  - keep_only_full_match 打开时按「非空字段零缺失即可」保留样本：
      至少一个选中字段非空，且所有非空字段零缺失 → 保留；
      全空 或 任一非空字段有缺失 → 丢弃。
  - 本步只做映射 + 打印 + 统计，不落盘。

不在映射表中、或映射到空 geo_sid 的 item，都记为缺失。

用法:
    python3 step2_inject_sid.py [common.conf]
"""

import os
import sys
import json
import math
import configparser
from collections import Counter

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] 需要 pyarrow，请先 pip install pyarrow", file=sys.stderr)
    sys.exit(1)

from step1_get_user_action import load_config, stream_rows, parse_item_seq


# ------------------------------------------------------------------
# 映射表
# ------------------------------------------------------------------
def get_item_map_path(conf_path: str) -> str:
    cp = configparser.ConfigParser()
    if not cp.read(conf_path, encoding="utf-8"):
        raise FileNotFoundError(f"找不到配置文件: {conf_path}")
    return cp.get("item_map", "item_map_path")


def load_item_sid_map(path: str) -> dict:
    """加载 item_id -> geo_sid 映射（只读两列，item_id 统一转成 str 作 key）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"映射表不存在: {path}")
    table = pq.read_table(path, columns=["item_id", "geo_sid"])
    ids = table.column("item_id").to_pylist()
    sids = table.column("geo_sid").to_pylist()
    id2sid = {}
    for i, s in zip(ids, sids):
        if i is None:
            continue
        id2sid[str(i)] = s
    return id2sid


# ------------------------------------------------------------------
# 映射单条样本（多字段，各自独立）
# ------------------------------------------------------------------
def map_field(seq_str: str, id2sid: dict) -> dict:
    """映射单个序列字段，返回该字段的 geo_sid 序列与缺失统计。"""
    items = parse_item_seq(seq_str)
    geo_seq = []
    mapped_items = []
    missing = 0
    for it in items:
        iid = it.get("item_id")
        sid = id2sid.get(iid)
        if not sid:  # 不在表中 或 空 geo_sid，都算缺失
            missing += 1
            sid = None
        geo_seq.append(sid)
        mapped_items.append({"item_id": iid, "geo_sid": sid, "title": it.get("title")})
    n = len(items)
    return {
        "n_items": n,
        "n_missing": missing,
        "miss_rate": (missing / n) if n else None,
        "geo_sid_seq": geo_seq,
        "items": mapped_items,
    }


def map_sample(row: dict, id2sid: dict, seq_fields: list) -> dict:
    """对每个选中序列字段各自映射，结果挂在 fields[field] 下。"""
    return {
        "uid": row.get("uid"),
        "fields": {f: map_field(row.get(f, ""), id2sid) for f in seq_fields},
    }


def is_full_match(mapped: dict, seq_fields: list) -> bool:
    """非空字段零缺失即可：至少一个字段非空，且所有非空字段零缺失。"""
    fields_with_items = [f for f in seq_fields if mapped["fields"][f]["n_items"] > 0]
    if not fields_with_items:
        return False  # 全空
    return all(mapped["fields"][f]["n_missing"] == 0 for f in fields_with_items)


# ------------------------------------------------------------------
# 缺失率分布（每个字段各一份）
# ------------------------------------------------------------------
BUCKET_LABELS = ["=0% (全命中)"] + [f"({(b-1)*10}%,{b*10}%]" for b in range(1, 11)]


def bucket_of(rate: float) -> str:
    if rate <= 0:
        return "=0% (全命中)"
    b = min(math.ceil(rate * 10), 10)  # 1..10
    return f"({(b-1)*10}%,{b*10}%]"


def new_field_stat() -> dict:
    return {"bucket": Counter(), "n_nonempty": 0, "n_empty": 0,
            "total_items": 0, "total_missing": 0}


def update_field_stat(stat: dict, field_res: dict):
    if field_res["n_items"] == 0:
        stat["n_empty"] += 1
    else:
        stat["n_nonempty"] += 1
        stat["total_items"] += field_res["n_items"]
        stat["total_missing"] += field_res["n_missing"]
        stat["bucket"][bucket_of(field_res["miss_rate"])] += 1


def print_field_distribution(field: str, stat: dict):
    n_nonempty = stat["n_nonempty"]
    print(f"\n---- 字段 {field} ----")
    print(f"序列数: {n_nonempty + stat['n_empty']}  (非空 {n_nonempty}, 空 {stat['n_empty']})")
    if stat["total_items"]:
        print(f"item 级总缺失率: {stat['total_missing']}/{stat['total_items']} = "
              f"{stat['total_missing'] / stat['total_items'] * 100:.2f}%")
    print("每条序列缺失率直方图（分母=非空序列数）:")
    print(f"  {'缺失率区间':<16} {'序列数':>10} {'占比':>10}")
    for label in BUCKET_LABELS:
        cnt = stat["bucket"].get(label, 0)
        ratio = (cnt / n_nonempty * 100) if n_nonempty else 0.0
        print(f"  {label:<16} {cnt:>10} {ratio:>9.2f}%")


def print_filter_summary(keep_switch: bool, n_total: int, n_kept: int,
                         n_drop_empty: int, n_drop_missing: int):
    print("\n================ 过滤汇总（keep_only_full_match={}）================"
          .format("on" if keep_switch else "off"))

    def pct(x):
        return (x / n_total * 100) if n_total else 0.0

    print(f"处理样本总数: {n_total}")
    print(f"  保留(非空字段零缺失): {n_kept}  ({pct(n_kept):.2f}%)")
    print(f"  丢弃合计:            {n_total - n_kept}  ({pct(n_total - n_kept):.2f}%)")
    print(f"    - 全空(所有字段皆空):  {n_drop_empty}  ({pct(n_drop_empty):.2f}%)")
    print(f"    - 有缺失(某非空字段缺): {n_drop_missing}  ({pct(n_drop_missing):.2f}%)")
    if not keep_switch:
        print("  （开关关闭：以上仅为统计，未实际过滤）")
    print("====================================================================\n")


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "common.conf"
    )
    cfg = load_config(conf_path)
    seq_fields = cfg["seq_fields"]
    item_map_path = get_item_map_path(conf_path)

    print(f"[INFO] 加载映射表: {item_map_path}")
    id2sid = load_item_sid_map(item_map_path)
    print(f"[INFO] 映射表条目数: {len(id2sid)}")

    keep_switch = cfg["keep_only_full_match"]
    unlimited = cfg["max_num"] == -1
    max_desc = "全量窗口数据" if unlimited else str(cfg["max_num"])
    print(f"[INFO] 序列字段: {seq_fields}")
    print(f"[INFO] 窗口: {cfg['train_start']} ~ {cfg['train_end']}，期望处理 {max_desc} 行")
    print(f"[INFO] keep_only_full_match = {'on' if keep_switch else 'off'}\n")

    field_stats = {f: new_field_stat() for f in seq_fields}
    n = 0
    n_kept = 0
    n_drop_empty = 0
    n_drop_missing = 0
    printed = 0

    for dt, hdfs_path, _schema, row in stream_rows(cfg):
        mapped = map_sample(row, id2sid, seq_fields)
        keep = is_full_match(mapped, seq_fields)

        # 诊断统计：每个字段各自累计（基于全部处理样本）
        for f in seq_fields:
            update_field_stat(field_stats[f], mapped["fields"][f])

        # 过滤计数（样本级）
        if keep:
            n_kept += 1
        elif all(mapped["fields"][f]["n_items"] == 0 for f in seq_fields):
            n_drop_empty += 1
        else:
            n_drop_missing += 1

        # 打印：开关打开时只打印「保留」的样本，关闭时打印全部
        if printed < cfg["log_sample_count"] and (keep or not keep_switch):
            tag = "保留样本" if keep_switch else "映射样本"
            print(f"---------- {tag} #{printed + 1}  (dt={dt}, part={os.path.basename(hdfs_path)}) ----------")
            print(json.dumps(mapped, ensure_ascii=False, indent=2, default=str))
            print()
            printed += 1

        n += 1
        if not unlimited and n >= cfg["max_num"]:
            break
        if unlimited and n % 100000 == 0:
            print(f"[INFO] 已处理 {n} 行")

    print(f"[INFO] 实际处理 {n} 行")
    print("\n================ 缺失率分布（诊断，按字段独立统计）================")
    for f in seq_fields:
        print_field_distribution(f, field_stats[f])
    print("\n====================================================================")
    print_filter_summary(keep_switch, n, n_kept, n_drop_empty, n_drop_missing)


if __name__ == "__main__":
    main()
