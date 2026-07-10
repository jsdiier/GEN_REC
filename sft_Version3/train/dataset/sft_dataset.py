#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAMERJsonlDataset: 读 step3 落盘的 train.jsonl / val.jsonl，用 SIDTokenizer 编码成
modeling.GAMERModel 的输入契约（input_ids / behavior_ids / token_types / labels）。

- train 行: {uid, aug_r, token_seq: [(action, geo_sid), ...]}
  -> encode_train_sample：[BOS]+全序列，labels=input_ids（全 token 监督）
- val 行  : {uid, input: [...], label_tokens: [...], positives_by_action, label_date}
  -> encode_val_sample：labels 仅 label 区有效（teacher-forcing 算 val loss）

截断策略（对齐论文 user-level 最大长度约束，如 100 个交互 ≈ 1+100*5=501 token）：
  按【交互】粒度从左截断（保留最近的），绝不从 item 中间切断 token；
  val 先保 label 区（过长时截尾，极少发生），剩余预算再从左截 input 区。

不依赖 tokenizer 模块本身（构造时传入实例，鸭子类型），方便单测与替换。
"""

import json

from torch.utils.data import Dataset


class GAMERJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, split: str, max_len: int = 512):
        """split: 'train' | 'val'；max_len: token 数上限（含 BOS）。"""
        assert split in ("train", "val"), split
        self.tok = tokenizer
        self.split = split
        self.max_len = max_len
        self.stride = 1 + tokenizer.num_levels        # 每交互 token 数（1 行为 + l 层 SID）
        self.samples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def _fit_items(self, budget_tokens: int) -> int:
        """budget_tokens 预算内最多放多少个交互。"""
        return max(budget_tokens // self.stride, 0)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        if self.split == "train":
            items = [tuple(x) for x in s["token_seq"]]
            keep = max(self._fit_items(self.max_len - 1), 1)   # 至少留 1 个交互
            return self.tok.encode_train_sample({"token_seq": items[-keep:]})

        label = [tuple(x) for x in s["label_tokens"]]
        inp = [tuple(x) for x in s["input"]]
        keep_lb = max(self._fit_items(self.max_len - 1), 1)
        label = label[:keep_lb]                                # label 区过长截尾（极少发生）
        keep_in = self._fit_items(self.max_len - 1 - len(label) * self.stride)
        inp = inp[-keep_in:] if keep_in > 0 else []
        return self.tok.encode_val_sample({"input": inp, "label_tokens": label})
