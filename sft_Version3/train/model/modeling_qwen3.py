#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3NTPModel: 用原生 Qwen3（transformers 库）做 NTP backbone，替代 modeling.py 的
手写 GAMERModel（CrossLevelBehaviorInteraction / PositionBehaviorMoE 全部去掉，
period/behavior 只是普通词表 token，靠标准 attention 隐式学）。

对外契约与 GAMERModel 完全一致（train_sft.py / generate.py / constrained_decode.py
/ eval/run_eval.py 均通过这个接口调用，无需改动）：
    forward(input_ids, behavior_ids, token_types, attention_mask=None,
            labels=None, last_positions=None) -> (loss, logits)
  - behavior_ids / token_types：接收但不使用（仅 GAMERModel 的 MoE/跨层注意力需要，
    这里保留参数位是为了让 dataset/collator/调用方一行都不用改）；
  - last_positions：与 GAMERModel 相同的优化——只在 lm_head 投影【之前】按位置 gather
    隐状态，避免整条序列过 (B,L,vocab) 的 lm_head（约束解码逐步生成的关键省显存点）。

预训练权重来源：common.conf [train] qwen3_path，本地目录（HF 格式 config.json +
safetensors），如 /home/luban/rank-ssl/chenpinyuan/MODEL/Qwen3-0.6B。
词表被替换成 SIDTokenizer 的小词表（resize_token_embeddings），所以 Qwen3 预训练收益
主要来自 transformer 主干（attention+FFN）权重初始化，而不是原生 152k 词表的语言知识。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config as HFQwen3Config, Qwen3ForCausalLM


@dataclass
class Qwen3NTPConfig:
    vocab_size: int                        # 由 SIDTokenizer 决定，替换 Qwen3 原生词表
    qwen3_path: str                        # 本地预训练权重目录（HF 格式）
    pad_token_id: int = 0
    max_position_embeddings: int = 1024    # 仅需 <= Qwen3 原生 rope 支持长度（32768）
    # Qwen3-0.6B 官方 config 是 28 层、hidden=1024，比 GAMER 的 8 层大很多，不开
    # 梯度检查点 batch_size 稍大一点就 OOM（forward 激活值是主要显存开销，不是
    # attention 矩阵本身——sdpa 已经不物化 (B,H,L,L)）；反正只是拿 activation
    # 换算力，重算一遍前向的开销远小于省下的显存，没有理由关。
    gradient_checkpointing: bool = True


class Qwen3NTPModel(nn.Module):
    def __init__(self, cfg: Qwen3NTPConfig, load_pretrained: bool = True):
        """load_pretrained=False：只按 qwen3_path 的 config.json 建架构、随机初始化，
           跳过真实权重加载——load_checkpoint() 里马上会用 state_dict 整体覆盖，
           真权重加载纯属浪费 I/O，仅 train_sft.py 主流程需要 True（真正吃预训练初始化）。"""
        super().__init__()
        self.cfg = cfg
        if load_pretrained:
            full = Qwen3ForCausalLM.from_pretrained(cfg.qwen3_path, attn_implementation="sdpa")
        else:
            hf_cfg = HFQwen3Config.from_pretrained(cfg.qwen3_path)
            full = Qwen3ForCausalLM(hf_cfg, attn_implementation="sdpa")
        full.resize_token_embeddings(cfg.vocab_size)   # 换成 SID 小词表，embedding+lm_head 同步 resize
        full.config.pad_token_id = cfg.pad_token_id
        full.config.use_cache = False
        self.backbone = full.model      # Qwen3Model：embedding + N 层 + 末尾 RMSNorm（无 lm_head）
        self.lm_head = full.lm_head
        self.hf_config = full.config
        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def forward(self, input_ids, behavior_ids=None, token_types=None,
                attention_mask=None, labels=None, last_positions=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False).last_hidden_state

        if last_positions is not None:
            idx = last_positions.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
            hidden = hidden.gather(1, idx).squeeze(1)      # (B, D)：只留需要的位置
            return None, self.lm_head(hidden)               # (B, vocab)

        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.reshape(-1, self.cfg.vocab_size),
                                   shift_labels.reshape(-1), ignore_index=-100)
        return loss, logits

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        seen, total = set(), 0
        for p in params:                      # 去重（tie_word_embeddings 共享权重）
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total


if __name__ == "__main__":
    # 冒烟测试：随机数据前向 + 反向，打印参数量（需要 qwen3_path 指向本地权重目录）
    import sys
    import torch

    torch.manual_seed(0)
    qwen3_path = sys.argv[1] if len(sys.argv) > 1 else "/home/luban/rank-ssl/chenpinyuan/MODEL/Qwen3-0.6B"
    cfg = Qwen3NTPConfig(vocab_size=34000, qwen3_path=qwen3_path)
    model = Qwen3NTPModel(cfg)
    print(f"参数量: {model.num_parameters() / 1e6:.2f}M")

    B, L = 2, 37
    input_ids = torch.randint(10, cfg.vocab_size, (B, L))
    loss, logits = model(input_ids, behavior_ids=None, token_types=None, labels=input_ids)
    loss.backward()
    print(f"loss: {loss.item():.4f}  logits: {tuple(logits.shape)}")
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    # last_positions 路径（约束解码用）
    lengths = torch.tensor([L, L - 3])
    _, last_logits = model(input_ids, behavior_ids=None, token_types=None,
                           last_positions=lengths - 1)
    assert tuple(last_logits.shape) == (B, cfg.vocab_size)
    print("smoke test passed")
