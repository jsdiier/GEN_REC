#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdfs_utils: 推理结果 HDFS 分区缓存的读取/合并工具，配合 generate.py 的
「按 dt=YYYYMMDD 分区累积缓存，cache_key 粒度判重+按最新交互时间判过期」逻辑。

设计：
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
import subprocess
import configparser
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pf

_FS = None
_DT_RE = re.compile(r"^dt=(\d{8})$")

# cache_key/last_interact_ts 是整条缓存条目（一个 uid x geohash x period）
# 共有的，其余五列是 topk 展开后逐 rank 的推荐负载
SCHEMA = pa.schema([
    ("cache_key", pa.string()),           # uid_<圆心geohash>_<version>_<period>
    ("rank", pa.int32()),                 # 1 = 置信最高
    ("score", pa.float32()),              # 各 SID token 的累计 logprob
    ("sid", pa.string()),                 # 模型实际生成的 SID（其 geohash 未必等于圆心）
    ("item_ids", pa.list_(pa.string())),  # sid 反查（一对多）
    ("shop_id", pa.list_(pa.string())),   # 与 item_ids 逐一对应的店铺 id
    ("last_interact_ts", pa.int64()),     # 生成时该用户全部历史最新一条交互的 ts
])


def _ensure_classpath() -> None:
    """pyarrow 的 HadoopFileSystem 底层走 libhdfs，需要 CLASSPATH 里包含全部
       hadoop jar 包才能建连接（否则报 getJNIEnv/HDFS connection failed）。
       通过交互式 shell 手动 export 不可靠——本函数走 Luban 平台提交时容器环境
       不会带着手动设的变量，所以改成代码里自动兜底：已设置则跳过（幂等），
       没设置就跑一次 `hadoop classpath --glob` 补上。"""
    if os.environ.get("CLASSPATH"):
        return
    hadoop_bin = os.environ.get("HADOOP_BIN")
    if not hadoop_bin:
        hadoop_home = os.environ.get("HADOOP_HOME", "/usr/local/hadoop-current")
        hadoop_bin = os.path.join(hadoop_home, "bin", "hadoop")
    ret = subprocess.run([hadoop_bin, "classpath", "--glob"],
                        capture_output=True, text=True)
    if ret.returncode != 0 or not ret.stdout.strip():
        raise RuntimeError(f"自动设置 CLASSPATH 失败（hadoop_bin={hadoop_bin}）: "
                          f"{ret.stderr.strip()}")
    os.environ["CLASSPATH"] = ret.stdout.strip()


def get_fs() -> "pf.HadoopFileSystem":
    """进程内单例，避免重复建连接。"""
    global _FS
    if _FS is None:
        _ensure_classpath()
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
    row_cols = [c for c in cols if c not in ("cache_key", "last_interact_ts")]
    for row in table.to_pylist():
        ck = row["cache_key"]
        grouped[ck]["rows"].append({k: row[k] for k in row_cols})
        if has_ts:
            # 同一 cache_key 各行的 last_interact_ts 理应一致，取任意一行即可
            grouped[ck]["last_interact_ts"] = row.get("last_interact_ts")
    return dict(grouped)


def write_dt_cache(fs, hdfs_output_root: str, dt: str, behavior: str, merged: dict,
                   buffer_rows: int = 50000) -> None:
    """把合并后的完整缓存（baseline 中未变的条目 + 本次新增/刷新的条目）整体
       重写到 dt 分区（该 behavior 的旧文件如果存在，直接覆盖）。merged 结构
       同 read_dt_cache 的返回值：{cache_key: {"rows": [...], "last_interact_ts": ts}}。
       流式攒 row group 写，不一次性把整表转成一个大 pa.Table（merged 本身已经
       整个在内存里，这里只是控制单次 write_table 的行数，避免瞬时峰值内存翻倍）。"""
    dt_dir = f"{hdfs_output_root}/dt={dt}"
    path = f"{dt_dir}/rec_{behavior}.parquet"
    fs.create_dir(dt_dir, recursive=True)
    with fs.open_output_stream(path) as sink:
        writer = pq.ParquetWriter(sink, SCHEMA)
        buf = []
        for cache_key, entry in merged.items():
            ts = entry["last_interact_ts"]
            for row in entry["rows"]:
                buf.append({"cache_key": cache_key, "last_interact_ts": ts, **row})
                if len(buf) >= buffer_rows:
                    writer.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
                    buf = []
        if buf:
            writer.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
        writer.close()


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
