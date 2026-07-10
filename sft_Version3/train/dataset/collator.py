#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAMERCollator: 把 GAMERJsonlDataset 的变长样本 pad 成等长 batch 张量。

padding 约定（与 modeling.GAMERModel 的输入契约一致）：
  input_ids    -> pad_id（词表 <pad>=0）
  behavior_ids -> -1（非 item token）
  token_types  -> -1（非 item token，MoE 路由到 expert_0 且不参与跨层注意力）
  labels       -> -100（不算 loss）
  attention_mask: 有效位 1 / padding 0
返回的 dict 可直接 **batch 解包进 model.forward。
"""

import torch


class GAMERCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)

        def pad(key: str, fill: int):
            return torch.tensor(
                [b[key] + [fill] * (max_len - len(b[key])) for b in batch],
                dtype=torch.long)

        attention_mask = torch.tensor(
            [[1] * len(b["input_ids"]) + [0] * (max_len - len(b["input_ids"]))
             for b in batch], dtype=torch.long)
        return {
            "input_ids": pad("input_ids", self.pad_id),
            "behavior_ids": pad("behavior_ids", -1),
            "token_types": pad("token_types", -1),
            "labels": pad("labels", -100),
            "attention_mask": attention_mask,
        }
