#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_common: generate.py（GPU 推理 worker）和 plan_infer.py（CPU 判重预处理）
共用的配置解析 / 数据流 / 收藏坐标解析逻辑，避免两边各写一份、逻辑漂移。
"""

import os
import configparser

import geo_utils as geo


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
        # gpu_num>1（数据并行）时这是判重预处理阶段【每个分片独立的】上限，
        # 不是全局总量——预处理会给 gpu_num 个分片各自记数，哪个分片满了就
        # 不再往哪个分片塞新任务，全部分片满了就提前结束扫描
        "infer_max_num": cp.getint("inference", "infer_max_num", fallback=-1),
        # 数据并行 GPU 数：>1 时判重预处理（plan_infer.py）把待推理样本按分片
        # 分发，提交 gpu_num 个独立单 GPU 任务并行跑；=1 = 不分片，单卡跑
        "gpu_num": cp.getint("inference", "gpu_num", fallback=1),
        "behaviors": [b.strip() for b in
                      cp.get("inference", "behaviors", fallback="clk,pay").split(",")
                      if b.strip()],
        "periods": [p.strip() for p in
                    cp.get("inference", "periods", fallback="").split(",")
                    if p.strip()],
        "topk": topk,
        "beam_size": max(cp.getint("inference", "beam_size", fallback=topk), topk),
        "batch_size": cp.getint("inference", "batch_size", fallback=32),
        "ckpt_path": path("ckpt_path", "outputs/ckpt/latest/best.pt"),
        "output_dir": path("output_dir", "outputs/inference"),
        "vocab_path": (lambda v: v if os.path.isabs(v) else os.path.join(root, v))(
            cp.get("train", "vocab_path", fallback="outputs/vocab.json")),
        "max_len": cp.getint("train", "max_len", fallback=512),
        # qwen3 checkpoint 没存 hf_config 时的 fallback，同 eval/run_eval.py：
        # 用当前这次运行自己的 common.conf 解析出的路径覆盖 checkpoint 里存的那份
        "qwen3_path": (lambda v: "" if not v else
                       (v if os.path.isabs(v) else os.path.join(root, v)))(
            cp.get("train", "qwen3_path", fallback="").strip()),
        "favor_coord_field": cp.get("inference", "favor_coord_field",
                                    fallback="user_favor_coor_top3"),
        "favor_coord_topk": cp.getint("inference", "favor_coord_topk", fallback=3),
        "favor_coord_radius_km": cp.getfloat("inference", "favor_coord_radius_km",
                                             fallback=4.0),
        "cache_version_tag": cp.get("inference", "cache_version_tag",
                                    fallback="Version3"),
        "write_debug_input": cp.getboolean("inference", "write_debug_input",
                                          fallback=True),
        "max_output_caches": cp.getint("inference", "max_output_caches",
                                       fallback=-1),
        # rec_*.parquet 落 HDFS 的根目录，其下按 dt=YYYYMMDD 建分区（见 hdfs_utils.py）
        "hdfs_output_root": cp.get("inference", "hdfs_output_root").rstrip("/"),
    }


def run_dir(ic: dict) -> str:
    """本地/NFS 落盘根目录：debug jsonl、_meta.json、数据并行的分片任务清单
       都在这下面（同一个窗口共用一个目录）。"""
    return os.path.join(ic["output_dir"], f"{ic['infer_start']}_{ic['infer_end']}")


def work_list_path(rdir: str, behavior: str, shard_index: int) -> str:
    """判重预处理（plan_infer.py）给每个分片写的待推理任务清单路径（本地/NFS，
       JSONL，一行一条待推理条目）；数据并行 GPU worker（generate.py）按
       shard_index 读自己这一份，不用重新流式扫描/判重。"""
    return os.path.join(rdir, f"_work_{behavior}_part{shard_index:03d}.jsonl")


# ------------------------------------------------------------------
# 数据：全历史用户流 + 收藏坐标原始字段 + 最新交互时间
# ------------------------------------------------------------------
def iter_infer_users_with_coords(conf_path: str, ic: dict, id2sid: dict):
    """流式产出 (uid, items, favor_coord_raw, last_interact_ts)：items = 全部
       历史交互 [(action, geo_sid, meal_period)...] 按时间原序；favor_coord_raw
       = 原始行的收藏坐标字段（lng@lat^lng@lat^... 格式）；last_interact_ts =
       该用户全部历史里最新一条交互的 ts（timeline 已按 ts 升序，取最后一条），
       用于跟 HDFS baseline 缓存里记录的时间比对、判断是否过期。复刻
       step3.iter_user_samples 内部的取数链路（而非改造其共享签名），避免把
       本功能的耦合带进 train/eval 也在用的公共函数。不做任何提前停止——
       扫描范围/停止条件由调用方（plan_infer.py 的判重逻辑、或单卡模式的
       generate.py）自己控制。"""
    import step3_build_samples as step3
    from step1_get_user_action import load_config, stream_rows
    from step2_inject_sid import map_sample

    cfg = load_config(conf_path)
    cfg["train_start"], cfg["train_end"] = ic["infer_start"], ic["infer_end"]
    cfg["max_num"] = -1
    seq_fields = cfg["seq_fields"]
    tz_offset = cfg["tz_offset_hours"]
    field = ic["favor_coord_field"]

    for _dt, _hdfs, _schema, row in stream_rows(cfg, verbose=True):
        mapped = map_sample(row, id2sid, seq_fields)
        timeline = step3.build_timeline(mapped, seq_fields, tz_offset)
        if not timeline:
            continue
        last_interact_ts = timeline[-1]["ts"]
        sessions = step3.sessionize_by_day(timeline)
        items = [(t["action"], t["geo_sid"], t["meal_period"])
                 for _, toks in sessions for t in toks]
        yield row.get("uid"), items, row.get(field), last_interact_ts


# ------------------------------------------------------------------
# 收藏坐标 -> 邻域候选根
# ------------------------------------------------------------------
def resolve_coords(tok, trie, real_geohashes: set, precision: int, raw,
                   favor_coord_topk: int, favor_coord_radius_km: float) -> list:
    """收藏坐标原始字段 -> 去重后的 (rank, lng, lat, geohash, legal_list, root)
       列表；root = 该坐标邻域内真实 geo token 限定的 trie 视图，legal_list =
       排序后的合法 geohash 白名单（供落盘/序列化进任务清单）；邻域内无真实
       item 时 root=None（调用方计入 skipped 并跳过）。"""
    pairs = geo.parse_favor_coords(raw, favor_coord_topk)
    seen, out = set(), []
    for rank, (lng, lat) in enumerate(pairs):
        gh = geo.encode(lat, lng, precision)
        if gh in seen:                    # top3 坐标落到同一网格：去重只算一次
            continue
        seen.add(gh)
        nbr = geo.neighbor_geohash_set(lat, lng, favor_coord_radius_km, precision)
        valid = nbr & real_geohashes
        if not valid:
            out.append((rank, lng, lat, gh, [], None))
            continue
        legal_list = sorted(valid)
        root = {tok.token2id[f"<{g}>"]: trie[tok.token2id[f"<{g}>"]] for g in valid}
        out.append((rank, lng, lat, gh, legal_list, root))
    return out


def root_from_legal_geohashes(tok, trie, legal_geohashes: list) -> dict:
    """数据并行 GPU worker 专用：任务清单里只序列化了 legal_geohashes（纯字符
       串列表，可落盘），worker 自己重建 root（拿自己进程内已经建好的 trie
       做字典查找，很便宜，不需要重新算邻域/geohash）。"""
    return {tok.token2id[f"<{g}>"]: trie[tok.token2id[f"<{g}>"]] for g in legal_geohashes}


# ------------------------------------------------------------------
# 共享的模型无关流水线上下文（tokenizer/item map/trie/geo 索引）
# ------------------------------------------------------------------
def load_pipeline_context(ic: dict, conf_path: str) -> dict:
    """加载判重预处理（plan_infer.py）和单卡推理（generate.py 不分片路径）
       都要用到的、跟具体 GPU/模型权重无关的上下文：tokenizer、item map、
       SID 前缀树、geohash 精度与全量真实 geohash 集合。数据并行 GPU worker
       （generate.py 分片路径）也需要 tok/trie 来跑 beam search，但不需要
       real_geohashes/precision（那是判重预处理用来解析收藏坐标的，worker
       直接从任务清单里读现成的 legal_geohashes），按需取用即可。"""
    from tokenizer_sid import SIDTokenizer
    from constrained_decode import build_sid_trie
    import step3_build_samples as step3
    from step2_inject_sid import load_item_sid_shop_map
    from collections import defaultdict

    tok = SIDTokenizer.load(ic["vocab_path"])
    for b in ic["behaviors"]:
        if b not in tok.behaviors:
            raise ValueError(f"未知行为 {b!r}（词表行为: {tok.behaviors}）")
    tok_periods = list(getattr(tok, "periods", []))
    if not tok_periods:
        raise ValueError("收藏坐标邻域缓存需要词表含时段位（先用带时段的词表/"
                         "checkpoint，如 ntp_w_period 分支训出的模型）")
    periods = ic["periods"] or tok_periods
    for p in periods:
        if p not in tok_periods:
            raise ValueError(f"未知时段 {p!r}（词表时段: {tok_periods}）")

    id2sid, id2shop = load_item_sid_shop_map(step3.get_item_map_path(conf_path))
    sid2items = defaultdict(list)             # 反查：一个 SID 可对应多个 item
    for item_id, sid in id2sid.items():
        sid2items[sid].append(item_id)
    trie = build_sid_trie(tok, id2sid.values())
    precision = geo.geohash_precision_from_vocab(tok)
    real_geohashes = {tok.id2token[tid][1:-1] for tid in trie.keys()}

    return {
        "tok": tok, "periods_resolved": periods,
        "id2sid": id2sid, "id2shop": id2shop, "sid2items": sid2items,
        "trie": trie, "precision": precision, "real_geohashes": real_geohashes,
    }
