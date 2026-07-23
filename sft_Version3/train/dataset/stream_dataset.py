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

增量模式（[data] is_auto=1，只出 train 样本，没有 val split）：
  collect_incremental_monitor_samples 用同一套 crc32(uid) 抽样，从"本轮该训练
  的用户"里划一小撮出来只当监控集用（不参与训练，train="train" 形状的样本，
  复用 GAMERJsonlDataset(split="train") 的编码逻辑当 val_dl 跑——全 token 监督
  的 NTP 序列 loss，语义上就是"模型在没训练过的用户序列上的困惑度"，可以当
  val loss 一样比大小选 best.pt）；GAMERStreamingTrainDataset 传入同样的
  monitor_sample_rate 后，会在流式产出 train 样本时跳过这批被抽去做监控的
  uid，保证训练集和监控集不重叠。
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
                 shuffle_buffer: int = 10000, seed: int = 42,
                 monitor_sample_rate: float = 0.0):
        """monitor_sample_rate>0（仅增量模式 is_auto=1 用得上）：跳过被
           collect_incremental_monitor_samples 用同一套 crc32(uid) 抽样规则
           划去做监控集的那批 uid，保证训练集和监控集不重叠。"""
        self.step3, load_config = _load_pipeline()
        self.conf_path = os.path.abspath(conf_path)
        self.cfg = load_config(self.conf_path)
        self.tok = tokenizer
        self.max_len = max_len
        self.shuffle_buffer = max(shuffle_buffer, 1)
        self.seed = seed
        self._epoch = 0
        self._id2sid = None            # worker 进程内懒加载
        self._monitor_thresh = max(int(monitor_sample_rate * 10000), 1) \
            if monitor_sample_rate > 0 else -1

    def set_epoch(self, epoch: int):
        """训练循环每个 epoch 前调用，让洗牌顺序随 epoch 变化。
           （DataLoader 每个 epoch 重新把 dataset pickle 进 worker，属性随之带入；
             不调用则每个 epoch 的样本顺序完全相同。）"""
        self._epoch = epoch

    def _train_records(self, part_filter, verbose):
        if self._id2sid is None:
            self._id2sid = self.step3.load_item_sid_map(
                self.step3.get_item_map_path(self.conf_path))
        for uid, _sessions, samples in self.step3.iter_user_samples(
                self.cfg, self.conf_path, part_filter=part_filter,
                id2sid=self._id2sid, verbose=verbose):
            if self._monitor_thresh >= 0 and \
               zlib.crc32(str(uid).encode("utf-8")) % 10000 < self._monitor_thresh:
                continue                # 划给监控集了，训练时跳过
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
        rng = random.Random(self.seed * 1000 + wid + self._epoch * 7919)
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


def collect_incremental_monitor_samples(conf_path: str, sample_rate: float = 0.05,
                                        max_users: int = 20000,
                                        verbose: bool = True) -> list:
    """增量模式（[data] is_auto=1）专用：跟 collect_val_samples 同一套
       crc32(uid) 抽样机制，但源头换成 iter_user_samples 在 is_auto=1 下产出的
       "train" 形状样本（增量窗口内的用户全序列），抽出来的这批只当监控集用，
       不参与训练——GAMERStreamingTrainDataset 要传同样的 sample_rate 作
       monitor_sample_rate，训练时才会跳过这批被抽中的 uid，避免训练/监控
       集重叠。返回的样本结构跟 train 样本一致（{"token_seq": [...]}），可以
       直接喂 GAMERJsonlDataset(split="train", samples=...) 当 val_dl 用。"""
    step3, load_config = _load_pipeline()
    conf_path = os.path.abspath(conf_path)
    cfg = load_config(conf_path)
    if not cfg.get("is_auto"):
        raise ValueError("collect_incremental_monitor_samples 只在 is_auto=1 时有意义")
    id2sid = step3.load_item_sid_map(step3.get_item_map_path(conf_path))
    thresh = max(int(sample_rate * 10000), 1)
    out = []
    for uid, _sessions, samples in step3.iter_user_samples(
            cfg, conf_path, id2sid=id2sid, verbose=verbose):
        if zlib.crc32(str(uid).encode("utf-8")) % 10000 >= thresh:
            continue
        for s in samples:
            if s["split"] == "train":
                out.append(s)
        if len(out) >= max_users:
            break
    if verbose:
        print(f"[INFO] 收集增量监控样本 {len(out)} 条 "
              f"(sample_rate={sample_rate}, max_users={max_users})")
    return out


def count_incremental_samples(conf_path: str, verbose: bool = True) -> dict:
    """[data] is_auto=1 专用：训练开始前先完整流一遍窗口数据，数出这一轮
       增量训练实际会覆盖多少用户、多少条 train 序列（含 behavior-drop 增强
       变体，即 GAMERStreamingTrainDataset 实际会产出的条数，监控集也包含在
       内——没有再按 monitor_sample_rate 刨掉，因为那批也是"这一轮命中的
       样本"，只是不参与梯度更新）。

       代价说明：is_auto 的筛选条件（最后一次交互是否落在窗口内）依赖每个
       用户的完整历史，没有现成的统计信息可以估算，只能整个流一遍数据源
       跟正式训练一样的量——这一步会让启动多花一次完整扫描的时间（量级
       跟训练一个 epoch 差不多），是"训练前就拿到准确样本数"必然的代价。"""
    step3, load_config = _load_pipeline()
    conf_path = os.path.abspath(conf_path)
    cfg = load_config(conf_path)
    if not cfg.get("is_auto"):
        raise ValueError("count_incremental_samples 只在 is_auto=1 时有意义")
    id2sid = step3.load_item_sid_map(step3.get_item_map_path(conf_path))
    n_users, n_seqs = 0, 0
    for _uid, _sessions, samples in step3.iter_user_samples(
            cfg, conf_path, id2sid=id2sid, verbose=verbose):
        if samples:
            n_users += 1
            n_seqs += len(samples)
    if verbose:
        print(f"[INFO] 本轮增量训练样本统计：命中窗口 [{cfg['train_start']}, "
              f"{cfg['train_end']}] 的用户 {n_users} 个，"
              f"共 {n_seqs} 条 train 序列（含 behavior-drop 增强变体）")
    return {"n_users": n_users, "n_train_sequences": n_seqs}
