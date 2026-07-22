#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdfs_utils: 推理结果 HDFS 分区缓存的读取/合并工具，配合 generate.py 的
「按 dt=YYYYMMDD 分区累积缓存，cache_key 粒度判重+按最新交互时间判过期」逻辑。

设计：
  - 不分片（单卡，gpu_num=1）：每个 dt 分区下每个 behavior 一个文件
    rec_{behavior}.parquet，是该 dt 的完整快照（每次推理整分区全量重写）；
    找 <= 本次 infer_end 的最近一个已有 dt 分区，直接读它的内容做 baseline
    （不需要真的 `hadoop fs -cp` 复制文件，反正是全量重写，读进内存、
    写到新 dt 路径即可）；
  - 数据并行（gpu_num>1）：待推理用户按 crc32(uid) % gpu_num 分片，
    提交 gpu_num 个独立单 GPU 任务并行跑，每个分片各写自己的
    rec_{behavior}_part{分片号}.parquet。提交前由编排方（submit_infer.py）
    调 consolidate_baseline 一次性把历史结果（不管是老的单文件还是之前
    某次不同 gpu_num 留下的 part 文件）合并成一份 rec_{behavior}_baseline
    快照，各分片 worker 只读这一份快照的【自己那部分】（不自己 glob 历史
    文件，也不会把全量快照整个物化成 Python 对象——见 read_baseline_snapshot
    的说明），避免 gpu_num 前后两次运行不一致时旧 part 文件残留造成同一
    cache_key 重复出现在多个文件里；全部分片成功后编排方删掉快照文件；
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
import zlib
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


def shard_of_uid(uid, gpu_num: int) -> int:
    """uid 的数据并行分片归属：跟本仓库已有的 test_sample_rate 一样用
       crc32(uid) 确定性哈希——同一个 uid 只要 gpu_num 不变，永远分到
       同一个 shard；gpu_num 变了，分配会整体重新洗牌（见
       consolidate_baseline 对这种情况的处理）。"""
    return zlib.crc32(str(uid).encode("utf-8")) % gpu_num


def glob_rec_files(fs, hdfs_output_root: str, dt: str, behavior: str) -> list:
    """dt 分区下该 behavior 现存的全部结果文件路径：覆盖老的单文件命名
       rec_{behavior}.parquet 和分片命名 rec_{behavior}_part{i}.parquet
       （两种可能在迁移过渡期同时存在）；不含 _baseline 快照文件（那是
       consolidate_baseline 内部使用的临时文件，不算"结果"）。"""
    dt_dir = f"{hdfs_output_root}/dt={dt}"
    infos = fs.get_file_info(pf.FileSelector(dt_dir, allow_not_found=True))
    prefix = f"rec_{behavior}"
    return sorted(
        info.path for info in infos
        if info.type == pf.FileType.File
        and os.path.basename(info.path).startswith(prefix)
        and info.path.endswith(".parquet")
        and not os.path.basename(info.path).endswith("_baseline.parquet")
    )


def _table_to_grouped(table, grouped=None) -> dict:
    """单个 pyarrow Table -> 按 cache_key 合并的 baseline 字典（可传入已有
       grouped 累加，供多文件合并复用）。真正把行物化成 Python dict 的地方
       只有这一处——调用方如果只需要一部分行，应该先在 pyarrow 层面
       table.filter() 筛完再传进来，不要传整张表再指望这里帮你扔掉多余的。"""
    if grouped is None:
        grouped = defaultdict(lambda: {"rows": [], "last_interact_ts": None})
    has_ts = "last_interact_ts" in table.column_names
    row_cols = [c for c in table.column_names
               if c not in ("cache_key", "last_interact_ts")]
    for row in table.to_pylist():
        ck = row["cache_key"]
        grouped[ck]["rows"].append({k: row[k] for k in row_cols})
        if has_ts:
            grouped[ck]["last_interact_ts"] = row.get("last_interact_ts")
    return grouped


def read_cache_files(fs, paths: list) -> dict:
    """读若干个 parquet 文件，按 cache_key 合并成一个 baseline 字典
       {cache_key: {"rows": [...], "last_interact_ts": ts_or_None}}。
       文件里没有 last_interact_ts 列（旧版本产出）则该文件贡献的条目
       last_interact_ts 记 None（调用方应视为"无法判断是否过期"强制刷新）。
       会把全部行整个物化成 Python 对象——用于编排阶段一次性合并全量历史
       （只发生一次，不随分片数放大）；分片 worker 请用 read_baseline_snapshot
       （带按分片过滤，不会整份物化）。"""
    grouped = defaultdict(lambda: {"rows": [], "last_interact_ts": None})
    for path in paths:
        with fs.open_input_file(path) as f:
            table = pq.read_table(f)
        _table_to_grouped(table, grouped)
    return dict(grouped)


def write_cache_file(fs, path: str, merged: dict, buffer_rows: int = 50000) -> None:
    """把 merged（结构同 read_cache_files 返回值）整体写到单个 parquet 文件
       路径（流式攒 row group 写；merged 本身已经整个在内存里，这里只是
       控制单次 write_table 的行数，避免瞬时峰值内存翻倍）。调用方自己保证
       父目录已存在。"""
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


def read_dt_cache(fs, hdfs_output_root: str, dt: str, behavior: str) -> dict:
    """单卡（不分片）场景用：读 dt 分区下该 behavior 现存的全部结果文件
       （不区分老命名/分片命名，全部 glob 合并），返回 baseline 字典。"""
    return read_cache_files(fs, glob_rec_files(fs, hdfs_output_root, dt, behavior))


def write_dt_cache(fs, hdfs_output_root: str, dt: str, behavior: str, merged: dict,
                   shard_index: int = None, gpu_num: int = 1,
                   buffer_rows: int = 50000) -> None:
    """写本次的完整结果。gpu_num<=1（不分片，单卡）时写 rec_{behavior}.parquet
       （老命名，向后兼容）；gpu_num>1 时写 rec_{behavior}_part{shard_index}.parquet
       （数据并行分片命名，此时 shard_index 必填，merged 应该已经是只含本
       shard 负责的 uid 那部分）。"""
    dt_dir = f"{hdfs_output_root}/dt={dt}"
    fs.create_dir(dt_dir, recursive=True)
    if gpu_num <= 1:
        path = f"{dt_dir}/rec_{behavior}.parquet"
    else:
        if shard_index is None:
            raise ValueError("gpu_num>1 时必须提供 shard_index")
        path = f"{dt_dir}/rec_{behavior}_part{shard_index:03d}.parquet"
    write_cache_file(fs, path, merged, buffer_rows=buffer_rows)


def _baseline_snapshot_path(hdfs_output_root: str, dt: str, behavior: str) -> str:
    return f"{hdfs_output_root}/dt={dt}/rec_{behavior}_baseline.parquet"


def consolidate_baseline(fs, hdfs_output_root: str, source_dt, target_dt: str,
                         behavior: str):
    """数据并行编排（submit_infer.py）专用，在提交任何分片任务之前调用一次：
       把 source_dt 分区下该 behavior 现存的全部结果文件（不管是老的单文件
       还是之前某次不同 gpu_num 分片留下的 part 文件）合并读出，写成
       target_dt 分区下统一的一份 _baseline 快照文件，供本次全部分片 worker
       读取——避免两次运行 gpu_num 不一致时，旧 part 文件残留导致同一个
       cache_key 重复出现在多个文件里。

       source_dt == target_dt（同一天重跑）时，被合并进快照的旧文件在快照
       写成功后会被删除（本来就要被这次全量重写替换掉，删掉安全）；
       source_dt < target_dt（推进到新的一天）时，老分区的文件原样保留，
       不做任何删除（那是历史分区，不该被这次运行动它）。

       source_dt 为 None（完全没有任何历史，首次推理）时什么也不做，返回
       None，分片 worker 读 baseline 快照会读到"文件不存在"从而视为空。"""
    if source_dt is None:
        return None
    old_paths = glob_rec_files(fs, hdfs_output_root, source_dt, behavior)
    if not old_paths:
        return None
    baseline = read_cache_files(fs, old_paths)
    target_dir = f"{hdfs_output_root}/dt={target_dt}"
    fs.create_dir(target_dir, recursive=True)
    snapshot_path = _baseline_snapshot_path(hdfs_output_root, target_dt, behavior)
    write_cache_file(fs, snapshot_path, baseline)
    if source_dt == target_dt:
        for p in old_paths:
            if p != snapshot_path:
                fs.delete_file(p)
    return snapshot_path


def read_baseline_snapshot(fs, hdfs_output_root: str, dt: str, behavior: str,
                           shard_index: int = None, gpu_num: int = 1) -> dict:
    """分片 worker 专用：只读 consolidate_baseline 已经准备好的那一份快照
       文件（不再自己 glob 旧结果文件——那些在编排阶段已经被合并/清理过了）。

       传入 shard_index/gpu_num（gpu_num>1）时，会先在 pyarrow 层面按
       crc32(uid)%gpu_num 筛出属于本分片的那些行（table.filter，纯列式
       操作，不产生 Python 对象），再把这一小部分转成 Python dict——不会像
       "整份读成 Python dict 再筛选"那样让每个分片都临时扛着全量数据的内存，
       全量历史很大时（比如百万用户级别）这个区别很关键。

       快照不存在（没有任何历史，首次推理）则返回空字典。"""
    path = _baseline_snapshot_path(hdfs_output_root, dt, behavior)
    info = fs.get_file_info(path)
    if info.type == pf.FileType.NotFound:
        return {}
    with fs.open_input_file(path) as f:
        table = pq.read_table(f)
    if shard_index is not None and gpu_num > 1:
        uids = (parse_uid(ck) for ck in table.column("cache_key").to_pylist())
        mask = pa.array(shard_of_uid(u, gpu_num) == shard_index for u in uids)
        table = table.filter(mask)
    return dict(_table_to_grouped(table))


def delete_baseline_snapshot(fs, hdfs_output_root: str, dt: str, behavior: str) -> None:
    """全部分片都成功后调用：清理掉 _baseline 快照文件（内容已经被完整
       分发进各分片各自的 part 文件里，快照本身不再需要）。"""
    path = _baseline_snapshot_path(hdfs_output_root, dt, behavior)
    info = fs.get_file_info(path)
    if info.type != pf.FileType.NotFound:
        fs.delete_file(path)


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
