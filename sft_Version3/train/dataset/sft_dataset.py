#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAMERJsonlDataset: 读 step3 落盘的 train.jsonl / val.jsonl（或直接传入内存样本列表），
用 SIDTokenizer 编码成 modeling.GAMERModel 的输入契约
（input_ids / behavior_ids / token_types / labels）。

- train 行: {uid, aug_r, token_seq: [(action, geo_sid), ...]}
  -> encode_train_record：[BOS]+全序列，labels=input_ids（全 token 监督）
- val 行  : {uid, input: [...], label_tokens: [...], positives_grouped, favor_coord_raw, label_date}
  -> encode_val_record：labels 仅 label 区有效（teacher-forcing 算 val loss）

截断策略（对齐论文 user-level 最大长度约束，如 100 个交互 ≈ 1+100*5=501 token）：
  按【交互】粒度从左截断（保留最近的），绝不从 item 中间切断 token；
  val 先保 label 区（过长时截尾，极少发生），剩余预算再从左截 input 区。

encode_train_record / encode_val_record 是独立函数，供本类与
stream_dataset.GAMERStreamingTrainDataset（流式模式）共用同一套截断/编码逻辑。
不依赖 tokenizer 模块本身（传入实例，鸭子类型），方便单测与替换。
"""

import json

from torch.utils.data import Dataset


def _stride(tokenizer) -> int:
    """每交互 token 数：优先用 tokens_per_interaction（含时段位），旧词表回退。"""
    return getattr(tokenizer, "tokens_per_interaction", 1 + tokenizer.num_levels)


def encode_train_record(tokenizer, items: list, max_len: int) -> dict:
    """train 序列 [(action, geo_sid[, period])... 或 token dict...] -> 全 token 监督输入。"""
    keep = max((max_len - 1) // _stride(tokenizer), 1)     # 至少留 1 个交互
    return tokenizer.encode_train_sample({"token_seq": items[-keep:]})


def encode_val_record(tokenizer, inp: list, label: list, max_len: int) -> dict:
    """val 的 input 区 + label 区 -> teacher-forcing 样本（labels 仅 label 区有效）。"""
    stride = _stride(tokenizer)
    keep_lb = max((max_len - 1) // stride, 1)
    label = label[:keep_lb]                                # label 区过长截尾（极少发生）
    keep_in = (max_len - 1 - len(label) * stride) // stride
    inp = inp[-keep_in:] if keep_in > 0 else []
    return tokenizer.encode_val_sample({"input": inp, "label_tokens": label})


class GAMERJsonlDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, split: str, max_len: int = 512,
                 samples: list = None):
        """split: 'train' | 'val'；max_len: token 数上限（含 BOS）。
           jsonl_path 与 samples 二选一：samples 直接传 step3 结构的样本列表
           （stream 模式收集的 val 就走这里）。"""
        assert split in ("train", "val"), split
        self.tok = tokenizer
        self.split = split
        self.max_len = max_len
        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        if self.split == "train":
            return encode_train_record(self.tok, s["token_seq"], self.max_len)
        return encode_val_record(self.tok, s["input"], s["label_tokens"], self.max_len)
