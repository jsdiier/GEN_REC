#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式训练数据接入（不落盘）：DataLoader worker 内直跑 step1->2->3 的内存链
（data/step3_build_samples.iter_user_samples），适配 TB 级训练数据。

GAMERStreamingTrainDataset（IterableDataset，只产 train 样本）：
  - worker 分片：对 part 文件路径 crc32 取模，每个 worker 只 `hadoop fs -cat`
    自己命中的分片，互不重复读；
  - 洗牌：流式数据无法全局 shuffle，用 shuffle_buffer 蓄水池式局部打乱
    （buffer 满后每来一条随机顶出一条）；
  - epoch：DataLoader 每个 epoch 重建迭代器 = 重新流一遍 HDFS 窗口数据；
  - item map（id2sid）在每个 worker 进程内懒加载一份。

collect_val_samples（启动时调用一次，返回内存 list）：
  - val 不适合流式（每个 epoch 都要在同一批样本上评估），启动时流一遍窗口数据，
    按 uid 的 crc32 抽样（sample_rate）收集 val 样本，凑满 max_users 提前停；
  - 返回的样本喂给 GAMERJsonlDataset(samples=...) 常驻内存。
    抽样率越低、上限越大，扫描的数据前缀越长——两个参数共同控制启动成本。
"""

import os
import sys
import random
import zlib

from torch.utils.data import IterableDataset, get_worker_info

from .sft_dataset import encode_train_record

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)


def _load_pipeline():
    """延迟 import data/ 目录的流水线模块（需要 pyarrow / hadoop 环境）。"""
    import step3_build_samples as step3
    from step1_get_user_action import load_config
    return step3, load_config


class GAMERStreamingTrainDataset(IterableDataset):
    def __init__(self, conf_path: str, tokenizer, max_len: int = 512,
                 shuffle_buffer: int = 10000, seed: int = 42):
        self.step3, load_config = _load_pipeline()
        self.conf_path = os.path.abspath(conf_path)
        self.cfg = load_config(self.conf_path)
        self.tok = tokenizer
        self.max_len = max_len
        self.shuffle_buffer = max(shuffle_buffer, 1)
        self.seed = seed
        self._id2sid = None            # worker 进程内懒加载

    def _train_records(self, part_filter, verbose):
        if self._id2sid is None:
            self._id2sid = self.step3.load_item_sid_map(
                self.step3.get_item_map_path(self.conf_path))
        for _uid, _sessions, samples in self.step3.iter_user_samples(
                self.cfg, self.conf_path, part_filter=part_filter,
                id2sid=self._id2sid, verbose=verbose):
            for s in samples:
                if s["split"] == "train":
                    yield encode_train_record(self.tok, s["token_seq"], self.max_len)

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        part_filter = None
        if nw > 1:
            part_filter = (lambda p, _w=wid, _n=nw:
                           zlib.crc32(p.encode("utf-8")) % _n == _w)
        rng = random.Random(self.seed * 1000 + wid)
        buf = []
        for rec in self._train_records(part_filter, verbose=(wid == 0)):
            if len(buf) < self.shuffle_buffer:
                buf.append(rec)
                continue
            i = rng.randrange(self.shuffle_buffer)
            buf[i], rec = rec, buf[i]
            yield rec
        rng.shuffle(buf)
        yield from buf


def collect_val_samples(conf_path: str, sample_rate: float = 0.05,
                        max_users: int = 20000, verbose: bool = True) -> list:
    """流一遍窗口数据收集 val 样本（uid crc32 抽样 + 上限早停），返回 step3 结构的
       样本 list，喂 GAMERJsonlDataset(samples=...)。"""
    step3, load_config = _load_pipeline()
    conf_path = os.path.abspath(conf_path)
    cfg = load_config(conf_path)
    id2sid = step3.load_item_sid_map(step3.get_item_map_path(conf_path))
    thresh = max(int(sample_rate * 10000), 1)
    out = []
    for uid, _sessions, samples in step3.iter_user_samples(
            cfg, conf_path, id2sid=id2sid, verbose=verbose):
        if zlib.crc32(str(uid).encode("utf-8")) % 10000 >= thresh:
            continue
        for s in samples:
            if s["split"] == "val":
                out.append(s)
        if len(out) >= max_users:
            break
    if verbose:
        print(f"[INFO] 收集 val 样本 {len(out)} 条 "
              f"(sample_rate={sample_rate}, max_users={max_users})")
    return out
