#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAMER 模型实现（对齐论文 arXiv:2511.03155 3.4 节结构 + 4.1.4 节尺寸）。

结构：decoder-only，num_layers 个 block，每个 block 三个模块串联（均 pre-norm 残差）：
  1. Causal Self-Attention      —— 标准因果注意力 + RoPE（Qwen3 式，RMSNorm/SwiGLU）；
  2. Cross-level Behavior Interaction —— 独立的一套注意力：Q/K/V 各加一张行为嵌入表
     E_B,Q/K/V（论文 Eq.2），behavior-wise mask（论文 Eq.3：i<j 且 level(b_i)<level(b_j)
     才可见），输出乘 SiLU 门控 G=SiLU(H·W_G)（论文 Eq.4）；
  3. Position-and-Behavior-Aware MoE —— 按 token 在 item 内的位置路由固定专家（论文 Eq.5）：
     行为 token 走 expert_0，第 j 层 SID token 走 expert_j（进 FFN 前 concat 行为嵌入）。

论文尺寸（默认值）：hidden=256, inner=512(SiLU), 6 heads x head_dim 64, 8 层，
词表 = SID 码本 + 行为 token + special token，总参数量 ~0.03B。

输入约定（与 step3/step4 的序列构造对齐，每个交互 = 1 行为 token + tokens_per_item 个 SID token）：
  input_ids    : (B, L) 统一词表 id
  behavior_ids : (B, L) 每个 token 所属 item 的行为 id（0=clk, 1=pay, ...）；
                 非 item token（BOS/PAD 等 special）= -1
  token_types  : (B, L) token 在 item 内的位置：0=行为 token, 1..l=第 j 层 SID token；
                 非 item token = -1（MoE 路由到 expert_0，跨层注意力中不可见也不可看）
  attention_mask: (B, L) 1=有效 0=padding（可省，默认全 1）
  labels       : (B, L) NTP 目标，-100 处不算 loss（内部做 shift，直接传 labels=input_ids
                 即为全 token 监督；val 的 teacher-forcing 把 input 区置 -100 即可）

论文未明说的实现选择（都在注释里标注了）：
  - 残差/归一化：论文说“基于 Qwen3 架构”，故用 pre-RMSNorm 残差；
  - FFN：Qwen3 的 SwiGLU（gate/up/down），inner=512；
  - 跨层注意力不加 RoPE（论文 Eq.3 无位置项，位置信息由模块 1 负责）；
  - behavior mask 下整行不可见的 query（如序列开头的低层行为）输出置 0，由门控自然关闭。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------
@dataclass
class GAMERConfig:
    vocab_size: int = 34000            # 统一词表大小，由 tokenizer 决定（special+行为+geo+a/b/c）
    hidden_size: int = 256             # 论文: model dimension 256
    intermediate_size: int = 512       # 论文: inner dimension 512
    num_layers: int = 8                # 论文: 8 decoder layers
    num_heads: int = 6                 # 论文: 6 heads
    head_dim: int = 64                 # 论文: dimension 64（6*64=384，注意力内部维度）
    tokens_per_item: int = 4           # 每个 item 的 SID token 数 l（geo + a + b + c）
    num_behaviors: int = 2             # 行为种类 |B|（clk / pay）
    behavior_levels: tuple = (1, 2)    # 各行为 id 的层级（下标=行为 id）：clk=1 < pay=2
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    tie_word_embeddings: bool = True   # lm_head 与词嵌入共享权重（小词表下省参数）

    @property
    def attn_inner(self) -> int:
        return self.num_heads * self.head_dim


# ------------------------------------------------------------------
# 基础组件（Qwen3 式）
# ------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(var + self.eps)
        return (self.weight * x).type_as(self.weight)


def build_rope_cache(head_dim: int, max_len: int, theta: float):
    """预计算 RoPE 的 cos/sin 表：(max_len, head_dim)。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)                  # (max_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)           # (max_len, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """q/k: (B, H, L, hd)；cos/sin: (L, hd)。"""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class SwiGLU(nn.Module):
    """Qwen3 式 FFN：down(SiLU(gate(x)) * up(x))。in_dim 可与 hidden 不同（MoE SID 专家是 2D 输入）。"""

    def __init__(self, in_dim: int, inner_dim: int, out_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(in_dim, inner_dim, bias=False)
        self.up_proj = nn.Linear(in_dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, out_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ------------------------------------------------------------------
# 模块 1：Causal Self-Attention（论文 Eq.1）
# ------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GAMERConfig):
        super().__init__()
        self.cfg = cfg
        d, inner = cfg.hidden_size, cfg.attn_inner
        self.q_proj = nn.Linear(d, inner, bias=False)
        self.k_proj = nn.Linear(d, inner, bias=False)
        self.v_proj = nn.Linear(d, inner, bias=False)
        self.o_proj = nn.Linear(inner, d, bias=False)

    def forward(self, x, cos, sin, pad_mask):
        """x: (B,L,D); pad_mask: (B,L) bool，True=有效。"""
        B, L, _ = x.shape
        H, hd = self.cfg.num_heads, self.cfg.head_dim
        q = self.q_proj(x).view(B, L, H, hd).transpose(1, 2)   # (B,H,L,hd)
        k = self.k_proj(x).view(B, L, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:L], sin[:L])
        # 因果 mask + padding mask，交给 F.sdpa（bool mask: True=可见）
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        mask = causal[None, None] & pad_mask[:, None, None, :]  # (B,1,L,L)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, L, H * hd)
        return self.o_proj(out)


# ------------------------------------------------------------------
# 模块 2：Cross-level Behavior Interaction（论文 Eq.2-4）
# ------------------------------------------------------------------
class CrossLevelBehaviorInteraction(nn.Module):
    def __init__(self, cfg: GAMERConfig):
        super().__init__()
        self.cfg = cfg
        d, inner = cfg.hidden_size, cfg.attn_inner
        self.q_proj = nn.Linear(d, inner, bias=False)
        self.k_proj = nn.Linear(d, inner, bias=False)
        self.v_proj = nn.Linear(d, inner, bias=False)
        # 行为嵌入表 E_B,Q / E_B,K / E_B,V（论文 Eq.2），加在投影后的 Q/K/V 上
        self.eb_q = nn.Embedding(cfg.num_behaviors, inner)
        self.eb_k = nn.Embedding(cfg.num_behaviors, inner)
        self.eb_v = nn.Embedding(cfg.num_behaviors, inner)
        self.o_proj = nn.Linear(inner, d, bias=False)
        self.w_gate = nn.Linear(d, d, bias=False)   # G = SiLU(H·W_G)（论文 Eq.4）

    def forward(self, x, behavior_ids, levels, pad_mask):
        """x: (B,L,D); behavior_ids: (B,L) -1=非item; levels: (B,L) 行为层级(非item=0)。"""
        B, L, _ = x.shape
        H, hd = self.cfg.num_heads, self.cfg.head_dim
        beh = behavior_ids.clamp(min=0)             # -1 -> 0 只为安全查表，下面会被 mask 掉
        q = self.q_proj(x) + self.eb_q(beh)
        k = self.k_proj(x) + self.eb_k(beh)
        v = self.v_proj(x) + self.eb_v(beh)
        q = q.view(B, L, H, hd).transpose(1, 2)
        k = k.view(B, L, H, hd).transpose(1, 2)
        v = v.view(B, L, H, hd).transpose(1, 2)
        # behavior-wise mask（论文 Eq.3）：i<j 且 level_i < level_j 且双方均为有效 item token
        causal_strict = torch.ones(L, L, dtype=torch.bool, device=x.device).tril(-1)  # i<j
        level_ok = levels[:, None, :] < levels[:, :, None]        # (B, Lq, Lk): level_k < level_q
        item_ok = (levels > 0)                                    # 非 item token 不参与
        mask = (causal_strict[None] & level_ok
                & item_ok[:, None, :] & item_ok[:, :, None]
                & pad_mask[:, None, :])                           # (B,L,L)
        scores = q @ k.transpose(-1, -2) / math.sqrt(hd)          # (B,H,L,L)
        scores = scores.masked_fill(~mask[:, None], torch.finfo(scores.dtype).min)
        attn = scores.softmax(dim=-1)
        # 整行不可见的 query（如序列开头的低层行为）softmax 退化为均匀分布，置 0
        no_key = ~mask.any(dim=-1)                                # (B,L)
        attn = attn.masked_fill(no_key[:, None, :, None], 0.0)
        out = (attn @ v).transpose(1, 2).reshape(B, L, H * hd)
        gate = F.silu(self.w_gate(x))                             # (B,L,D)
        return self.o_proj(out) * gate


# ------------------------------------------------------------------
# 模块 3：Position-and-Behavior-Aware MoE（论文 Eq.5）
# ------------------------------------------------------------------
class PositionBehaviorMoE(nn.Module):
    """按 token 在 item 内的位置路由固定专家：
       expert_0: 行为 token（及 BOS/PAD 等非 item token），输入 D；
       expert_j (j=1..l): 第 j 层 SID token，输入 concat(hidden, E_B(行为)) = 2D。"""

    def __init__(self, cfg: GAMERConfig):
        super().__init__()
        self.cfg = cfg
        d, inner, l = cfg.hidden_size, cfg.intermediate_size, cfg.tokens_per_item
        self.behavior_emb = nn.Embedding(cfg.num_behaviors, d)
        self.expert_beh = SwiGLU(d, inner, d)
        self.experts_sid = nn.ModuleList([SwiGLU(2 * d, inner, d) for _ in range(l)])

    def forward(self, x, behavior_ids, token_types):
        """x: (B,L,D); token_types: (B,L) 0=行为token, 1..l=SID第j层, -1=非item。"""
        out = torch.zeros_like(x)
        # 行为 token 与非 item token 走 expert_0
        m0 = token_types <= 0
        if m0.any():
            out[m0] = self.expert_beh(x[m0])
        beh = behavior_ids.clamp(min=0)
        beh_e = self.behavior_emb(beh)                            # (B,L,D)
        for j, expert in enumerate(self.experts_sid, start=1):
            mj = token_types == j
            if mj.any():
                out[mj] = expert(torch.cat([x[mj], beh_e[mj]], dim=-1))
        return out


# ------------------------------------------------------------------
# Block 与整体模型
# ------------------------------------------------------------------
class GAMERBlock(nn.Module):
    def __init__(self, cfg: GAMERConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.cross = CrossLevelBehaviorInteraction(cfg)
        self.ln3 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.moe = PositionBehaviorMoE(cfg)

    def forward(self, x, cos, sin, behavior_ids, levels, token_types, pad_mask):
        x = x + self.attn(self.ln1(x), cos, sin, pad_mask)
        x = x + self.cross(self.ln2(x), behavior_ids, levels, pad_mask)
        x = x + self.moe(self.ln3(x), behavior_ids, token_types)
        return x


class GAMERModel(nn.Module):
    def __init__(self, cfg: GAMERConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size,
                                         padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList([GAMERBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # 行为 id -> 层级 的查表（下标 0 留给非 item token：level 0）
        levels = torch.tensor([0] + list(cfg.behavior_levels), dtype=torch.long)
        self.register_buffer("level_table", levels, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def forward(self, input_ids, behavior_ids, token_types,
                attention_mask=None, labels=None):
        """返回 (loss, logits)；labels 为 None 时 loss 为 None。"""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pad_mask = attention_mask.bool()
        # levels: behavior_ids(-1..num_behaviors-1) 平移 1 后查表 -> 非 item=0, clk=1, pay=2
        levels = self.level_table[(behavior_ids + 1).clamp(min=0)]
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin,
                      behavior_ids, levels, token_types, pad_mask)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # 内部 shift：预测下一个 token；labels 直接传 input_ids 即全 token 监督
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.cfg.vocab_size),
                                   shift_labels.view(-1), ignore_index=-100)
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
    # 冒烟测试：随机数据前向 + 反向，打印参数量
    torch.manual_seed(0)
    cfg = GAMERConfig(vocab_size=34000)
    model = GAMERModel(cfg)
    print(f"参数量: {model.num_parameters() / 1e6:.2f}M")

    B, n_items = 2, 6
    stride = 1 + cfg.tokens_per_item                      # 每交互 5 个 token
    L = 1 + n_items * stride                              # BOS + 6 个交互
    input_ids = torch.randint(10, cfg.vocab_size, (B, L))
    behavior_ids = torch.full((B, L), -1, dtype=torch.long)
    token_types = torch.full((B, L), -1, dtype=torch.long)
    for i in range(n_items):                              # 填 item 区的元信息
        s = 1 + i * stride
        b = torch.randint(0, cfg.num_behaviors, (B,))
        behavior_ids[:, s:s + stride] = b[:, None]
        token_types[:, s] = 0
        for j in range(1, stride):
            token_types[:, s + j] = j
    loss, logits = model(input_ids, behavior_ids, token_types, labels=input_ids)
    loss.backward()
    print(f"loss: {loss.item():.4f}  logits: {tuple(logits.shape)}")
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print("smoke test passed")
