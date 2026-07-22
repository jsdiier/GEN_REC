#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdfs_utils: 推理结果 HDFS 分区缓存的读取/合并工具，配合 generate.py 的
「按 dt=YYYYMMDD 分区累积缓存，cache_key 粒度判重+按最新交互时间判过期」逻辑。

设计（渐进式开发，本文件先只做「读」，「写」「判重」「与 generate.py 集成」
留到后续步骤）：
  - 每个 dt 分区下每个 behavior 只有一个文件 rec_{behavior}.parquet，
    永远是该 dt 的完整快照（每次推理整分区全量重写，不搞多文件累加）；
  - 找 <= 本次 infer_end 的最近一个已有 dt 分区，直接读它的内容做 baseline
    （不需要真的 `hadoop fs -cp` 复制文件，反正是全量重写，读进内存、
    写到新 dt 路径即可）；
  - baseline 按 cache_key 分组：一个 cache_key 对应 topk 行（rank 1..K），
    外加 last_interact_ts（该 cache_key 对应用户，生成时的全部历史最新
    交互时间，用于下次判断是否过期）；
  - 旧文件（本设计落地前手工产出的）没有 last_interact_ts 列，读取时要
    优雅兼容：该列缺失时每条记录的 last_interact_ts 记为 None，
    调用方应将 None 视为「无法判断是否过期」强制归入需要刷新。

用法（本步骤，独立验证用）：
    python hdfs_utils.py [common.conf]
读 [inference] hdfs_output_root/infer_end/behaviors 与 [hadoop] hadoop_bin，
找到 <= infer_end 的最近 dt 分区，打印每个 behavior 的 cache_key 数、
uid 数、是否含 last_interact_ts 列，供人工核对。
"""

import os
import re
import sys
import configparser
from collections import defaultdict

import pyarrow.parquet as pq
import pyarrow.fs as pf

_FS = None
_DT_RE = re.compile(r"^dt=(\d{8})$")


def get_fs() -> "pf.HadoopFileSystem":
    """进程内单例，避免重复建连接。"""
    global _FS
    if _FS is None:
        _FS = pf.HadoopFileSystem("default")
    return _FS


def list_dt_partitions(fs, hdfs_output_root: str) -> list:
    """列出 hdfs_output_root 下已存在的 dt=YYYYMMDD 子目录（升序）；
       根目录本身不存在或下面没有任何 dt 分区（首次推理）则返回 []。"""
    infos = fs.get_file_info(pf.FileSelector(hdfs_output_root, allow_not_found=True))
    dts = []
    for info in infos:
        if info.type != pf.FileType.Directory:
            continue
        m = _DT_RE.match(os.path.basename(info.path.rstrip("/")))
        if m:
            dts.append(m.group(1))
    return sorted(dts)


def find_source_dt(fs, hdfs_output_root: str, infer_end: str):
    """在已有分区里找 <= infer_end 的最大 dt：
         - 等于 infer_end：今天已经跑过（同天重跑），直接读它当 baseline；
         - 小于 infer_end：断档也没关系，从最近一个完整历史分区起步；
         - 找不到任何分区：首次推理，返回 None。"""
    dts = [d for d in list_dt_partitions(fs, hdfs_output_root) if d <= infer_end]
    return dts[-1] if dts else None


def parse_uid(cache_key: str) -> str:
    """cache_key = uid_<geohash>_<cache_version_tag>_<period>，uid 本身不含
       下划线，从右往左切三刀最稳妥（不受 uid 具体内容影响）。"""
    return cache_key.rsplit("_", 3)[0]


def read_dt_cache(fs, hdfs_output_root: str, dt: str, behavior: str) -> dict:
    """读 dt 分区下该 behavior 的 rec_{behavior}.parquet，按 cache_key 分组，
       返回 {cache_key: {"rows": [row_dict, ...], "last_interact_ts": ts_or_None}}。
       文件不存在（该 dt 分区没跑过这个 behavior）返回空字典；
       文件里没有 last_interact_ts 列（旧版本产出）则每条 last_interact_ts 记 None。"""
    path = f"{hdfs_output_root}/dt={dt}/rec_{behavior}.parquet"
    info = fs.get_file_info(path)
    if info.type == pf.FileType.NotFound:
        return {}
    with fs.open_input_file(path) as f:
        table = pq.read_table(f)
    has_ts = "last_interact_ts" in table.column_names
    cols = table.column_names
    grouped = defaultdict(lambda: {"rows": [], "last_interact_ts": None})
    for row in table.to_pylist():
        ck = row["cache_key"]
        grouped[ck]["rows"].append({k: row[k] for k in cols if k != "cache_key"})
        if has_ts:
            # 同一 cache_key 各行的 last_interact_ts 理应一致，取任意一行即可
            grouped[ck]["last_interact_ts"] = row.get("last_interact_ts")
    return dict(grouped)


# ------------------------------------------------------------------
# 独立验证入口：只读、不写
# ------------------------------------------------------------------
def _load_minimal_config(conf_path: str) -> dict:
    cp = configparser.ConfigParser()
    if not cp.read(conf_path, encoding="utf-8"):
        raise FileNotFoundError(f"找不到配置文件: {conf_path}")
    return {
        "hdfs_output_root": cp.get("inference", "hdfs_output_root").rstrip("/"),
        "infer_end": cp.get("inference", "infer_end"),
        "behaviors": [b.strip() for b in
                      cp.get("inference", "behaviors", fallback="pay").split(",")
                      if b.strip()],
    }


def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common.conf")
    ic = _load_minimal_config(conf_path)
    fs = get_fs()

    all_dts = list_dt_partitions(fs, ic["hdfs_output_root"])
    print(f"[INFO] hdfs_output_root={ic['hdfs_output_root']}")
    print(f"[INFO] 已有 dt 分区: {all_dts}")

    src_dt = find_source_dt(fs, ic["hdfs_output_root"], ic["infer_end"])
    if src_dt is None:
        print(f"[INFO] 没有 <= {ic['infer_end']} 的历史分区，视为首次推理（baseline 为空）")
        return
    print(f"[INFO] infer_end={ic['infer_end']}  取 baseline dt={src_dt}")

    for behavior in ic["behaviors"]:
        cache = read_dt_cache(fs, ic["hdfs_output_root"], src_dt, behavior)
        n_keys = len(cache)
        uids = {parse_uid(ck) for ck in cache}
        n_with_ts = sum(1 for v in cache.values() if v["last_interact_ts"] is not None)
        print(f"  [{behavior}] cache_key 数={n_keys}  uid 数={len(uids)}  "
              f"含 last_interact_ts 的条目数={n_with_ts}/{n_keys}")
        if cache:
            sample_key = next(iter(cache))
            print(f"    样例 cache_key={sample_key!r}  "
                  f"rows={cache[sample_key]['rows'][:1]}  "
                  f"last_interact_ts={cache[sample_key]['last_interact_ts']!r}")


if __name__ == "__main__":
    main()
