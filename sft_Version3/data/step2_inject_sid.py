#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2: 把用户行为序列（seq_fields，如 u_pay_item_seq_100 / u_clk_item_seq_100）
里每个 item_id 映射成 geo_sid。多字段「各自独立」处理。

清洗模式 clean_mode（common.conf [data]）：
  - drop_missing（默认）: 逐 item 剔除找不到 geo_sid 的交互，保留序列其余部分；
  - full_match          : 整条丢弃，非空字段零缺失才保留整条样本。

诊断输出（两种模式都打印）：
  - 每个字段的缺失率分布；
  - 每个字段「剔除缺失 item 前 / 后」的序列长度分布。

不在映射表中、或映射到空 geo_sid 的 item，都记为缺失。本步只做映射+统计，不落盘。

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


def clean_mapped(mapped: dict, seq_fields: list) -> dict:
    """drop_missing 清洗视图：每个字段剔掉缺失 item，保留其余 geo_sid。"""
    out = {"uid": mapped["uid"], "fields": {}}
    for f in seq_fields:
        fr = mapped["fields"][f]
        clean = [s for s in fr["geo_sid_seq"] if s]
        out["fields"][f] = {
            "len_before": fr["n_items"],
            "len_after": len(clean),
            "removed": fr["n_missing"],
            "geo_sid_seq": clean,
        }
    return out


def is_full_match(mapped: dict, seq_fields: list) -> bool:
    """full_match 判据：至少一个字段非空，且所有非空字段零缺失。"""
    fields_with_items = [f for f in seq_fields if mapped["fields"][f]["n_items"] > 0]
    if not fields_with_items:
        return False  # 全空
    return all(mapped["fields"][f]["n_missing"] == 0 for f in fields_with_items)


# ------------------------------------------------------------------
# 分桶
# ------------------------------------------------------------------
MISS_BUCKET_LABELS = ["=0% (全命中)"] + [f"({(b-1)*10}%,{b*10}%]" for b in range(1, 11)]
LEN_BUCKET_LABELS = ["0"] + [f"[{(b-1)*10+1},{b*10}]" for b in range(1, 11)] + [">100"]


def miss_bucket(rate: float) -> str:
    if rate <= 0:
        return "=0% (全命中)"
    b = min(math.ceil(rate * 10), 10)  # 1..10
    return f"({(b-1)*10}%,{b*10}%]"


def len_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n > 100:
        return ">100"
    b = math.ceil(n / 10)  # 1..10
    return f"[{(b-1)*10+1},{b*10}]"


def new_field_stat() -> dict:
    return {
        "miss_bucket": Counter(),      # 缺失率分布（分母=非空序列）
        "n_nonempty": 0, "n_empty": 0,
        "total_items": 0, "total_missing": 0,
        "len_before": Counter(),       # 剔除前长度分布（含空序列）
        "len_after": Counter(),        # 剔除后长度分布
        "became_empty": 0,             # 原本非空、剔除后变空的序列数
    }


def update_field_stat(stat: dict, field_res: dict):
    n = field_res["n_items"]
    after = n - field_res["n_missing"]
    stat["len_before"][len_bucket(n)] += 1
    stat["len_after"][len_bucket(after)] += 1
    if n == 0:
        stat["n_empty"] += 1
    else:
        stat["n_nonempty"] += 1
        stat["total_items"] += n
        stat["total_missing"] += field_res["n_missing"]
        stat["miss_bucket"][miss_bucket(field_res["miss_rate"])] += 1
        if after == 0:
            stat["became_empty"] += 1


# ------------------------------------------------------------------
# 打印分布
# ------------------------------------------------------------------
def print_miss_distribution(field: str, stat: dict):
    n_nonempty = stat["n_nonempty"]
    print(f"\n---- 字段 {field} 缺失率分布 ----")
    print(f"序列数: {n_nonempty + stat['n_empty']}  (非空 {n_nonempty}, 空 {stat['n_empty']})")
    if stat["total_items"]:
        print(f"item 级总缺失率: {stat['total_missing']}/{stat['total_items']} = "
              f"{stat['total_missing'] / stat['total_items'] * 100:.2f}%")
    print("每条序列缺失率直方图（分母=非空序列数）:")
    print(f"  {'缺失率区间':<16} {'序列数':>10} {'占比':>10}")
    for label in MISS_BUCKET_LABELS:
        cnt = stat["miss_bucket"].get(label, 0)
        ratio = (cnt / n_nonempty * 100) if n_nonempty else 0.0
        print(f"  {label:<16} {cnt:>10} {ratio:>9.2f}%")


def print_length_distribution(field: str, stat: dict, n_total: int):
    print(f"\n---- 字段 {field} 序列长度分布（剔除缺失 item 前 / 后）----")
    print(f"  {'长度区间':<12} {'剔前序列数':>10} {'剔前占比':>9} {'剔后序列数':>10} {'剔后占比':>9}")
    for label in LEN_BUCKET_LABELS:
        b = stat["len_before"].get(label, 0)
        a = stat["len_after"].get(label, 0)
        bp = (b / n_total * 100) if n_total else 0.0
        ap = (a / n_total * 100) if n_total else 0.0
        print(f"  {label:<12} {b:>10} {bp:>8.2f}% {a:>10} {ap:>8.2f}%")
    tb = stat["total_items"]
    ta = tb - stat["total_missing"]
    if tb:
        print(f"  总交互数: 剔前 {tb} -> 剔后 {ta}  "
              f"(剔除 {stat['total_missing']}, {stat['total_missing'] / tb * 100:.2f}%)")
    print(f"  剔除后变空的非空序列: {stat['became_empty']}")


def print_drop_missing_summary(n_total: int, n_all_empty_after: int):
    print("\n================ 清洗汇总（clean_mode=drop_missing）================")
    print(f"处理样本总数: {n_total}")
    pct = (n_all_empty_after / n_total * 100) if n_total else 0.0
    print(f"剔除后【所有字段都空】的样本: {n_all_empty_after}  ({pct:.2f}%)")
    print("（step2 仅诊断，未落盘；剔除后全空的样本通常在 step3 丢弃）")
    print("====================================================================\n")


def print_full_match_summary(n_total: int, n_kept: int,
                             n_drop_empty: int, n_drop_missing: int):
    print("\n================ 过滤汇总（clean_mode=full_match）================")

    def pct(x):
        return (x / n_total * 100) if n_total else 0.0

    print(f"处理样本总数: {n_total}")
    print(f"  保留(非空字段零缺失): {n_kept}  ({pct(n_kept):.2f}%)")
    print(f"  丢弃合计:            {n_total - n_kept}  ({pct(n_total - n_kept):.2f}%)")
    print(f"    - 全空(所有字段皆空):  {n_drop_empty}  ({pct(n_drop_empty):.2f}%)")
    print(f"    - 有缺失(某非空字段缺): {n_drop_missing}  ({pct(n_drop_missing):.2f}%)")
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
    clean_mode = cfg["clean_mode"]
    item_map_path = get_item_map_path(conf_path)

    print(f"[INFO] 加载映射表: {item_map_path}")
    id2sid = load_item_sid_map(item_map_path)
    print(f"[INFO] 映射表条目数: {len(id2sid)}")

    unlimited = cfg["max_num"] == -1
    max_desc = "全量窗口数据" if unlimited else str(cfg["max_num"])
    print(f"[INFO] 序列字段: {seq_fields}")
    print(f"[INFO] 窗口: {cfg['train_start']} ~ {cfg['train_end']}，期望处理 {max_desc} 行")
    print(f"[INFO] clean_mode = {clean_mode}\n")

    field_stats = {f: new_field_stat() for f in seq_fields}
    n = 0
    printed = 0
    # drop_missing 计数
    n_all_empty_after = 0
    # full_match 计数
    n_kept = 0
    n_drop_empty = 0
    n_drop_missing = 0

    for dt, hdfs_path, _schema, row in stream_rows(cfg):
        mapped = map_sample(row, id2sid, seq_fields)

        # 诊断统计（两种模式都累计）：缺失率 + 长度前后
        for f in seq_fields:
            update_field_stat(field_stats[f], mapped["fields"][f])

        if clean_mode == "full_match":
            keep = is_full_match(mapped, seq_fields)
            if keep:
                n_kept += 1
            elif all(mapped["fields"][f]["n_items"] == 0 for f in seq_fields):
                n_drop_empty += 1
            else:
                n_drop_missing += 1
            if printed < cfg["log_sample_count"] and keep:
                print(f"---------- 保留样本 #{printed + 1}  (dt={dt}, part={os.path.basename(hdfs_path)}) ----------")
                print(json.dumps(mapped, ensure_ascii=False, default=str))
                print()
                printed += 1
        else:  # drop_missing
            if all((mapped["fields"][f]["n_items"] - mapped["fields"][f]["n_missing"]) == 0
                   for f in seq_fields):
                n_all_empty_after += 1
            if printed < cfg["log_sample_count"]:
                print(f"---------- 清洗样本 #{printed + 1}  (dt={dt}, part={os.path.basename(hdfs_path)}) ----------")
                print(json.dumps(clean_mapped(mapped, seq_fields), ensure_ascii=False, default=str))
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
        print_miss_distribution(f, field_stats[f])
    print("\n====================================================================")

    print("\n================ 序列长度分布（剔除缺失 item 前 / 后）================")
    for f in seq_fields:
        print_length_distribution(f, field_stats[f], n)
    print("\n====================================================================")

    if clean_mode == "full_match":
        print_full_match_summary(n, n_kept, n_drop_empty, n_drop_missing)
    else:
        print_drop_missing_summary(n, n_all_empty_after)


if __name__ == "__main__":
    main()
