#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Hive 表 user_action_sample_d_whole_v2 对应的 HDFS parquet 路径，
在 [train_start, train_end] 窗口内按 dt 逐天流式读取用户行为样本。

设计要点：
  - 不落盘：用 `hadoop fs -cat <part>` 把单个 part 文件的字节流读进内存 BytesIO，
            再交给 pyarrow 解析（parquet footer 在文件尾，无法逐行网络流式，
            这是「内存里流式出行 + 按 part 早停」的最佳折中）。
  - 早停：  max_num > 0 时凑够 max_num 行立即停止，后续 part / dt 分区不再触碰。
  - 本步只做「取数 + 打印原始数据」，不写任何输出文件；schema 与前几行打出来后，
    再决定下游样本怎么拼（step3）。

用法:
    python3 step1_get_user_action.py [common.conf]
"""

import os
import sys
import json
import subprocess
import configparser
from datetime import datetime, timedelta

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] 需要 pyarrow，请先 pip install pyarrow", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------------
# 行为序列解析规范
#   序列字段（如 u_pay_item_seq_100 / u_clk_item_seq_100）共用同一套规范：
#   item 间用 ||| 分隔，item 内字段用 @_@ 分隔，共 10 个字段。
#   具体使用哪些序列字段由 common.conf [data] seq_fields 配置。
# ------------------------------------------------------------------
ITEM_SEP = "|||"
FIELD_SEP = "@_@"
ITEM_FIELDS = ["item_id", "phone_time", "local_hour", "brand_name",
               "s_main_category_id", "module", "phone_time_local",
               "title", "description", "brand"]


def parse_item_seq(seq_str):
    """把序列字符串解析成结构化 item 列表。字段数不符时标记 _parts_len。"""
    if not seq_str:
        return []
    items = []
    for chunk in str(seq_str).split(ITEM_SEP):
        if not chunk:
            continue
        parts = chunk.split(FIELD_SEP)
        item = {f: (parts[i] if i < len(parts) else None)
                for i, f in enumerate(ITEM_FIELDS)}
        if len(parts) != len(ITEM_FIELDS):
            item["_parts_len"] = len(parts)  # 异常行标记，便于排查
        items.append(item)
    return items


def parse_sample(row: dict, seq_fields: list) -> dict:
    """保留用户画像字段原样，把每个选中的序列字段替换为 {n_items, items} 结构。"""
    parsed = {k: v for k, v in row.items() if k not in seq_fields}
    for f in seq_fields:
        items = parse_item_seq(row.get(f, ""))
        parsed[f] = {"n_items": len(items), "items": items}
    return parsed


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------
def load_config(conf_path: str) -> dict:
    cp = configparser.ConfigParser()
    if not cp.read(conf_path, encoding="utf-8"):
        raise FileNotFoundError(f"找不到配置文件: {conf_path}")
    return {
        "hadoop_bin": cp.get("hadoop", "hadoop_bin"),
        "hdfs_root": cp.get("hive", "hdfs_root"),
        "table": cp.get("hive", "table"),
        "country_code": cp.get("hive", "country_code"),
        "train_start": cp.get("data", "train_start"),
        "train_end": cp.get("data", "train_end"),
        "max_num": cp.getint("data", "max_num"),
        "log_sample_count": cp.getint("data", "log_sample_count", fallback=3),
        "clean_mode": cp.get("data", "clean_mode", fallback="drop_missing"),
        "tz_offset_hours": cp.getint("data", "tz_offset_hours", fallback=-6),
        "session_granularity": cp.get("data", "session_granularity", fallback="day"),
        "train_label_scope": cp.get("data", "train_label_scope", fallback="slide"),
        "behavior_drop_x": cp.getint("data", "behavior_drop_x", fallback=3),
        "min_train_seq_len": cp.getint("data", "min_train_seq_len", fallback=2),
        # 0（默认）= 现状：留一法三分 train/val/test（一次性全量训练）；
        # 1 = 增量模式：只保留【最后一次交互时间落在 train_start~train_end】的用户，
        #     用其全部历史 S1..Sm（不留一法切分）整条构造 train 序列，不出 val/test
        "is_auto": cp.getint("data", "is_auto", fallback=0),
        "samples_out_dir": cp.get("data", "samples_out_dir", fallback=""),
        "seq_fields": [s.strip() for s in
                       cp.get("data", "seq_fields",
                              fallback="u_pay_item_seq_100").split(",")
                       if s.strip()],
        # 用户收藏坐标原始字段（test 样本 favor_coord_raw 用）：[data] 没配就复用
        # [inference] 已有的同名配置，两处语义一致，不强制重复填一遍
        "favor_coord_field": cp.get(
            "data", "favor_coord_field",
            fallback=cp.get("inference", "favor_coord_field",
                            fallback="user_favor_coor_top3")),
    }


# ------------------------------------------------------------------
# HDFS 访问
# ------------------------------------------------------------------
def daterange(start: str, end: str):
    """按天遍历 [start, end]（含端点），dt 格式 YYYYMMDD。"""
    d0 = datetime.strptime(start, "%Y%m%d")
    d1 = datetime.strptime(end, "%Y%m%d")
    if d1 < d0:
        raise ValueError(f"train_end({end}) 早于 train_start({start})")
    cur = d0
    while cur <= d1:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def build_partition_dir(cfg: dict, dt: str) -> str:
    return "/".join([
        cfg["hdfs_root"].rstrip("/"),
        cfg["table"],
        f"dt={dt}",
        f"country_code={cfg['country_code']}",
    ])


def fmt_size(n_bytes: int) -> str:
    """字节数 -> 人类可读（B/KiB/MiB/GiB/TiB）。"""
    size = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def list_part_files(hadoop_bin: str, hdfs_dir: str) -> tuple:
    """列出分区下所有 part 文件，按文件名排序保证可复现；分区不存在时返回空。
       返回 (part 文件路径列表, 分区总字节数)，大小直接取自 `fs -ls` 输出第 5 列。"""
    ret = subprocess.run([hadoop_bin, "fs", "-ls", hdfs_dir],
                         capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"[WARN] 列目录失败（可能该 dt 分区不存在），跳过: {hdfs_dir}\n"
              f"       stderr: {ret.stderr.strip()}")
        return [], 0
    files, total_bytes = [], 0
    for line in ret.stdout.splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        path = parts[-1]
        if os.path.basename(path).startswith("part-"):
            files.append(path)
            try:
                total_bytes += int(parts[4])
            except ValueError:
                pass
    files.sort()
    return files, total_bytes


def cat_parquet_to_reader(hadoop_bin: str, hdfs_path: str) -> "pa.BufferReader":
    """把 HDFS 上的 parquet part 文件 cat 进内存（不落盘），返回可 seek 的 BufferReader。"""
    ret = subprocess.run([hadoop_bin, "fs", "-cat", hdfs_path],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode != 0:
        raise RuntimeError(f"hadoop fs -cat 失败 {hdfs_path}: "
                           f"{ret.stderr.decode('utf-8', 'ignore')}")
    return pa.BufferReader(ret.stdout)


def stream_rows(cfg: dict, batch_size: int = 1024, part_filter=None, verbose: bool = True):
    """
    生成器：在窗口内逐 dt、逐 part 流式产出 (dt, hdfs_path, schema, row)。
    max_num 早停由调用方负责计数。
    part_filter(hdfs_path)->bool 可选：多 DataLoader worker 按 part 文件分片，
    每个 worker 只 cat 命中的分片，避免重复读 HDFS；verbose=False 时静默（worker 内用）。
    """
    for dt in daterange(cfg["train_start"], cfg["train_end"]):
        hdfs_dir = build_partition_dir(cfg, dt)
        part_files, total_bytes = list_part_files(cfg["hadoop_bin"], hdfs_dir)
        n_all = len(part_files)
        if part_filter:
            part_files = [p for p in part_files if part_filter(p)]
        if not part_files:
            continue
        if verbose:
            shard = (f"，本进程分到 {len(part_files)} 个" if part_filter else "")
            print(f"[INFO] dt={dt}: {n_all} 个 part 文件，"
                  f"HDFS 占用 {fmt_size(total_bytes)}{shard}")
        for hdfs_path in part_files:
            reader = cat_parquet_to_reader(cfg["hadoop_bin"], hdfs_path)
            pf = pq.ParquetFile(reader)
            for batch in pf.iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    yield dt, hdfs_path, pf.schema_arrow, row


# ------------------------------------------------------------------
# 打印
# ------------------------------------------------------------------
def print_schema(schema):
    print("\n================ 原始数据 SCHEMA ================")
    for field in schema:
        print(f"  {field.name}: {field.type}")
    print(f"  （共 {len(schema.names)} 列）")
    print("================================================\n")


def print_row(idx: int, dt: str, hdfs_path: str, row: dict, seq_fields: list):
    """打印原始行（紧凑 JSON）+ 解析后的结构（带缩进 JSON，便于看清 item 序列）。"""
    print(f"---------- 样本 #{idx}  (dt={dt}, part={os.path.basename(hdfs_path)}) ----------")
    print("[原始]")
    print(json.dumps(row, ensure_ascii=False, default=str))
    print("[解析]")
    print(json.dumps(parse_sample(row, seq_fields), ensure_ascii=False, indent=2, default=str))
    print()


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "common.conf"
    )
    cfg = load_config(conf_path)

    unlimited = cfg["max_num"] == -1
    max_desc = "全量窗口数据" if unlimited else str(cfg["max_num"])
    print(f"[INFO] 表: {cfg['table']}  country_code={cfg['country_code']}")
    print(f"[INFO] 窗口: {cfg['train_start']} ~ {cfg['train_end']}（按 dt 逐天）")
    print(f"[INFO] 序列字段: {cfg['seq_fields']}")
    print(f"[INFO] 期望读取: {max_desc} 行")

    schema_printed = False
    n = 0
    for dt, hdfs_path, schema, row in stream_rows(cfg):
        if not schema_printed:
            print_schema(schema)
            schema_printed = True
        if n < cfg["log_sample_count"]:
            print_row(n + 1, dt, hdfs_path, row, cfg["seq_fields"])

        n += 1
        if not unlimited and n >= cfg["max_num"]:
            break
        if unlimited and n % 100000 == 0:
            print(f"[INFO] 已累计读取 {n} 行")

    print(f"[INFO] 实际读取 {n} 行（本步仅取数+打印，未落盘）")


if __name__ == "__main__":
    main()
