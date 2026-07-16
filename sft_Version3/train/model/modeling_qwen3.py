#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3NTPModel: 用原生 Qwen3（transformers 库）做 NTP backbone，最后
cfg.num_customized_layers 层额外并联挂上 GAMER 论文的两个定制模块
（CrossLevelBehaviorInteraction / PositionBehaviorMoE，复用 modeling.py
的实现，见 _customize_layer 的说明）；其余层是纯原生 Qwen3 block。

对外契约与 GAMERModel 基本一致（train_sft.py / generate.py /
constrained_decode.py / eval/run_eval.py 均通过这个接口调用，无需改动）：
    forward(input_ids, behavior_ids, token_types, attention_mask=None,
            labels=None, last_positions=None) -> (loss, logits)
  - behavior_ids / token_types：这个分支会真正用到（喂给定制层的两个模块），
    跟纯原生 Qwen3 backbone 版本（qwen_w_period_geo 分支）不同——那边这两个
    参数只接收不使用；
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

from modeling import GAMERConfig, CrossLevelBehaviorInteraction, PositionBehaviorMoE


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
    # ---- GAMER 定制模块（本分支新增，qwen_w_period_geo 主分支没有这几项）----
    # 由 tokenizer 推导（跟 GAMERConfig 同名字段同语义），PositionBehaviorMoE /
    # CrossLevelBehaviorInteraction 建表要用：
    tokens_per_item: int = 5               # num_levels，MoE 按 SID 层数建专家
    num_behaviors: int = 2
    num_periods: int = 0
    behavior_levels: tuple = (1, 2)
    # 只有最后这么多层加定制模块（离输出近，更容易学到行为/时段结构），
    # 其余层是纯原生 Qwen3 block；这两个数是跟用户对齐过的固定值，这个分支
    # 没有开关（qwen_w_period_geo 主分支才是"纯原生 Qwen3"的那个版本）：
    num_customized_layers: int = 8
    moe_intermediate_size: int = 3072      # 跟 Qwen3 原生 FFN 一样大，不做瓶颈


class _GamerLayerAddon(nn.Module):
    """挂在 Qwen3DecoderLayer.mlp 位置：Qwen3 原生 FFN 保留（预训练权重不丢），
       并联加 CrossLevelBehaviorInteraction + PositionBehaviorMoE，三路输出
       加和后交给 Qwen3DecoderLayer 自己原有的残差相加逻辑。

       为什么是"并联加和"而不是论文里 attn->cross-level->FFN->MoE 的严格串联：
       Qwen3DecoderLayer.forward() 的具体参数签名（position_embeddings /
       past_key_values 之类）在不同 transformers 版本之间变过（我们已经被
       attn_implementation 这个构造参数的版本差异坑过一次），如果重新实现一遍
       forward 去把 cross-level 插在 self_attn 和 mlp 之间，等于把这个版本
       敏感的调用链路又抄一遍，换个 transformers 版本大概率又要修一次。
       只包一层 self.mlp（forward(x)->tensor 这个接口极其稳定，是每个
       pre-norm transformer FFN 子层的最基本假设，基本不随版本变化），风险
       小得多。三路都吃同一份 post-attention-norm 的隐状态作为输入，语义上
       仍然是"在这一层里，行为层级交互 + 位置感知专家 都参与了这次更新"，
       跟论文严格顺序的差别只是"并联 vs 串联"，不是"有没有"。

       behavior_ids/token_types/pad_mask 不通过 Qwen3 内部 forward 的参数链路
       传（HF 每层调用不保证会把自定义 kwargs 透传到 self.mlp 这一层），而是
       从 ctx（外层 Qwen3NTPModel 每次 forward 前写入的共享 dict）读，这几层
       共享同一个 ctx 对象引用。"""

    def __init__(self, original_mlp: nn.Module, gcfg: GAMERConfig, ctx: dict):
        super().__init__()
        self.original_mlp = original_mlp
        self.cross_level = CrossLevelBehaviorInteraction(gcfg)
        self.position_moe = PositionBehaviorMoE(gcfg)
        self.ctx = ctx

    def forward(self, x):
        ffn_out = self.original_mlp(x)
        cross_out = self.cross_level(x, self.ctx["behavior_ids"],
                                     self.ctx["levels"], self.ctx["pad_mask"])
        moe_out = self.position_moe(x, self.ctx["behavior_ids"], self.ctx["token_types"])
        return ffn_out + cross_out + moe_out


class Qwen3NTPModel(nn.Module):
    def __init__(self, cfg: Qwen3NTPConfig, load_pretrained: bool = True,
                hf_config_dict: dict = None):
        """load_pretrained=False：只建架构、随机初始化，跳过真实权重加载——
           load_checkpoint() 里马上会用 state_dict 整体覆盖，加载真实预训练权重
           纯属浪费 I/O，仅 train_sft.py 主流程需要 True（真正吃预训练初始化）。
           hf_config_dict：架构 config 直接以 dict 传入（load_checkpoint 存在
           checkpoint 里的那份），不用再读 qwen3_path 的 config.json——训练用的
           qwen3_path 是相对路径，训练（platform，NFS 挂载）和 eval/inference
           （本地 home 挂载）解析出来的绝对路径不是同一个物理位置，checkpoint
           一旦跨语境加载就会读到不存在的路径；架构其实只是几个数字，直接存进
           checkpoint 就能让 eval/inference 完全不依赖 qwen3_path 是否可达。
           留空时退回旧路径（读 qwen3_path 的 config.json），只为兼容这个字段
           上线前存的旧 checkpoint。
           注意：_GamerLayerAddon 里新增的参数（CrossLevelBehaviorInteraction /
           PositionBehaviorMoE）不在 hf_config_dict / Qwen3 原生权重覆盖范围内，
           它们的初始化值始终是随机初始化，跟 load_pretrained 无关；训练完的
           checkpoint 会把这部分权重存进 state_dict，eval/inference 走
           load_state_dict 正常还原，不受影响。"""
        super().__init__()
        self.cfg = cfg
        if load_pretrained:
            full = Qwen3ForCausalLM.from_pretrained(cfg.qwen3_path, attn_implementation="sdpa")
        else:
            hf_cfg = (HFQwen3Config(**hf_config_dict) if hf_config_dict is not None
                      else HFQwen3Config.from_pretrained(cfg.qwen3_path))
            # 直接构造（非 from_pretrained）时，attn_implementation 不是所有
            # transformers 版本都支持当构造函数关键字传（远端环境版本较老，
            # 报 TypeError: unexpected keyword argument 'attn_implementation'）；
            # 写到 config 上是更早就支持、更兼容的方式，PreTrainedModel.__init__
            # 会读 config._attn_implementation 决定用哪个注意力实现。
            hf_cfg._attn_implementation = "sdpa"
            full = Qwen3ForCausalLM(hf_cfg)
        full.resize_token_embeddings(cfg.vocab_size)   # 换成 SID 小词表，embedding+lm_head 同步 resize
        full.config.pad_token_id = cfg.pad_token_id
        full.config.use_cache = False
        self.backbone = full.model      # Qwen3Model：embedding + N 层 + 末尾 RMSNorm（无 lm_head）
        self.lm_head = full.lm_head
        self.hf_config = full.config
        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # ---- 最后 num_customized_layers 层挂 GAMER 定制模块 ----
        self._ctx = {}   # forward() 每次调用前填，_GamerLayerAddon 共享读取这同一个 dict
        n = cfg.num_customized_layers
        if n > 0:
            gcfg = GAMERConfig(
                vocab_size=cfg.vocab_size,        # 这两个定制模块用不到，占位
                hidden_size=self.hf_config.hidden_size,
                intermediate_size=cfg.moe_intermediate_size,
                num_layers=n,                     # 占位，不影响这两个模块本身
                num_heads=self.hf_config.num_attention_heads,
                head_dim=getattr(self.hf_config, "head_dim",
                                 self.hf_config.hidden_size // self.hf_config.num_attention_heads),
                tokens_per_item=cfg.tokens_per_item,
                num_behaviors=cfg.num_behaviors,
                num_periods=cfg.num_periods,
                behavior_levels=cfg.behavior_levels,
            )
            # 行为 id -> 层级 查表（下标 0 留给非 item token：level 0），跟
            # modeling.GAMERModel 里同一张表同一个用法
            levels = torch.tensor([0] + list(cfg.behavior_levels), dtype=torch.long)
            self.register_buffer("level_table", levels, persistent=False)
            for layer in self.backbone.layers[-n:]:      # n > 层数时 Python 切片自动只取全部
                layer.mlp = _GamerLayerAddon(layer.mlp, gcfg, self._ctx)

    def forward(self, input_ids, behavior_ids=None, token_types=None,
                attention_mask=None, labels=None, last_positions=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.cfg.num_customized_layers > 0:
            beh = behavior_ids if behavior_ids is not None else torch.full_like(input_ids, -1)
            tt = token_types if token_types is not None else torch.full_like(input_ids, -1)
            self._ctx["behavior_ids"] = beh
            self._ctx["token_types"] = tt
            self._ctx["pad_mask"] = attention_mask.bool()
            self._ctx["levels"] = self.level_table[(beh + 1).clamp(min=0)]
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
    cfg = Qwen3NTPConfig(vocab_size=34000, qwen3_path=qwen3_path,
                         tokens_per_item=4, num_behaviors=2, num_periods=3,
                         behavior_levels=(1, 2), num_customized_layers=2)
    model = Qwen3NTPModel(cfg)
    print(f"参数量: {model.num_parameters() / 1e6:.2f}M")

    B, n_items = 2, 6
    stride = 1 + cfg.tokens_per_item + 1          # period + behavior + SID levels
    L = 1 + n_items * stride
    input_ids = torch.randint(10, cfg.vocab_size, (B, L))
    behavior_ids = torch.full((B, L), -1, dtype=torch.long)
    token_types = torch.full((B, L), -1, dtype=torch.long)
    for i in range(n_items):
        s = 1 + i * stride
        b = torch.randint(0, cfg.num_behaviors, (B,))
        behavior_ids[:, s:s + stride] = b[:, None]
        token_types[:, s] = cfg.tokens_per_item + 1     # period 位
        for j in range(1, stride):
            token_types[:, s + j] = j
    loss, logits = model(input_ids, behavior_ids=behavior_ids, token_types=token_types,
                         labels=input_ids)
    loss.backward()
    print(f"loss: {loss.item():.4f}  logits: {tuple(logits.shape)}")
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), \
        "有参数没收到梯度（定制模块没接上反向图？）"

    # last_positions 路径（约束解码用）
    lengths = torch.tensor([L, L - 3])
    _, last_logits = model(input_ids, behavior_ids=behavior_ids, token_types=token_types,
                           last_positions=lengths - 1)
    assert tuple(last_logits.shape) == (B, cfg.vocab_size)
    print("smoke test passed")
