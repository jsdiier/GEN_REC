#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
constrained_decode: SID 前缀树 + 行为条件约束 beam search。

生成一个 item = 依次生成 num_levels 个 SID token（geo -> a -> b -> c）。
自由解码会拼出不存在的 SID 组合，所以用 item map 里全部真实 item 的 SID
建一棵前缀树（trie），每一步只允许沿树上存在的边走——生成结果必然是真实 item。

用法（eval/run_eval.py 调用）：
    trie = build_sid_trie(tokenizer, geo_sids)
    results = constrained_beam_search(model, tokenizer, trie, prefixes,
                                      beam_size, device, autocast_ctx)
    # results[i] = [(geo_sid, 累计logprob), ...] 按分数降序，长度 beam_size

prefix 契约：每条 prefix 是 dict {input_ids, behavior_ids, token_types}，
  形如 [BOS] + 历史交互 + 强制行为 token（<clk> 或 <pay>），行为条件由调用方拼好。
  beam search 只负责在其后生成 num_levels 个 SID token。

实现要点：
  - 无 KV cache（模型未实现），每步全量 forward；只解 num_levels(=4) 步，可接受；
  - 跨样本 batch：右 padding + 因果注意力保证 pad 不影响前面位置，
    按各自真实长度取末位 logits；
  - 第 0 步所有 beam 相同（同一 prefix），只算 1 条避免重复；
  - 分数 = token logprob 累加；各 beam 生成长度相同，无需长度归一化。
"""

import torch
import torch.nn.functional as F


def build_sid_trie(tokenizer, geo_sids) -> dict:
    """全部真实 item 的 geo_sid 字符串 -> token id 前缀树（嵌套 dict，叶子为空 dict）。
       编码失败的 sid（层数异常/词表外 token）跳过并计数告警。"""
    from tokenizer_sid import split_sid
    root, n_bad = {}, 0
    for sid in set(geo_sids):
        try:
            parts = split_sid(sid)
            if len(parts) != tokenizer.num_levels:
                raise ValueError("层数不符")
            ids = [tokenizer.token2id[p] for p in parts]
        except (ValueError, KeyError):
            n_bad += 1
            continue
        node = root
        for tid in ids:
            node = node.setdefault(tid, {})
    if n_bad:
        print(f"[WARN] {n_bad} 个 geo_sid 无法进 trie（层数异常或词表外），已跳过")
    return root


def _pad_batch(seqs: list, pad_id: int, device):
    """变长序列列表 -> 右 padding 的 (input_ids, behavior_ids, token_types,
       attention_mask, lengths) 张量。"""
    max_len = max(len(s["input_ids"]) for s in seqs)
    n = len(seqs)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    behavior_ids = torch.full((n, max_len), -1, dtype=torch.long)
    token_types = torch.full((n, max_len), -1, dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long)
    lengths = torch.tensor([len(s["input_ids"]) for s in seqs], dtype=torch.long)
    for i, s in enumerate(seqs):
        L = len(s["input_ids"])
        input_ids[i, :L] = torch.tensor(s["input_ids"], dtype=torch.long)
        behavior_ids[i, :L] = torch.tensor(s["behavior_ids"], dtype=torch.long)
        token_types[i, :L] = torch.tensor(s["token_types"], dtype=torch.long)
        attention_mask[i, :L] = 1
    return (input_ids.to(device), behavior_ids.to(device), token_types.to(device),
            attention_mask.to(device), lengths.to(device))


@torch.no_grad()
def _last_logprobs(model, seqs, pad_id, device, autocast_ctx):
    """batch forward，取各序列末位 token 的 log_softmax(logits)。(n, V) float32。"""
    input_ids, behavior_ids, token_types, attn, lengths = _pad_batch(seqs, pad_id, device)
    with autocast_ctx():
        _, logits = model(input_ids=input_ids, behavior_ids=behavior_ids,
                          token_types=token_types, attention_mask=attn)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1))
    last = logits.gather(1, idx).squeeze(1).float()
    return F.log_softmax(last, dim=-1)


def _extend(prefix: dict, tid: int, behavior_id: int, token_type: int) -> dict:
    return {
        "input_ids": prefix["input_ids"] + [tid],
        "behavior_ids": prefix["behavior_ids"] + [behavior_id],
        "token_types": prefix["token_types"] + [token_type],
    }


@torch.no_grad()
def constrained_beam_search(model, tokenizer, trie: dict, prefixes: list,
                            beam_size: int, device, autocast_ctx) -> list:
    """对 batch 内每条 prefix 做 trie 约束 beam search，生成 num_levels 个 SID token。
       prefix 末 token 必须是行为 token（行为条件已由调用方拼好）。
       返回 results[i] = [(geo_sid, score), ...] 分数降序，最多 beam_size 条。"""
    model.eval()
    n = len(prefixes)
    beh_of = [p["behavior_ids"][-1] for p in prefixes]      # 生成 token 沿用行为 id

    # beams[i] = [(seq_dict, score, trie_node), ...]；第 0 步只有 prefix 一条
    beams = [[(p, 0.0, trie)] for p in prefixes]

    for lv in range(tokenizer.num_levels):
        flat, owner = [], []                                # 展平所有活跃 beam 做一次 forward
        for i in range(n):
            for b in beams[i]:
                flat.append(b[0])
                owner.append(i)
        logprobs = _last_logprobs(model, flat, tokenizer.pad_id, device, autocast_ctx)

        new_beams = [[] for _ in range(n)]
        pos = 0
        for i in range(n):
            cand = []                                       # (score, tid, beam_idx)
            for j, (seq, score, node) in enumerate(beams[i]):
                lp = logprobs[pos + j]
                allowed = list(node.keys())
                sub = lp[torch.tensor(allowed, device=lp.device)]
                k = min(beam_size, len(allowed))
                top = torch.topk(sub, k)
                for v, a in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((score + v, allowed[a], j))
            cand.sort(key=lambda x: -x[0])
            for score, tid, j in cand[:beam_size]:
                seq, _, node = beams[i][j]
                new_beams[i].append((_extend(seq, tid, beh_of[i], lv + 1),
                                     score, node[tid]))
            pos += len(beams[i])
        beams = new_beams

    results = []
    for i in range(n):
        out = []
        for seq, score, _ in beams[i]:
            sid_ids = seq["input_ids"][-tokenizer.num_levels:]
            out.append(("".join(tokenizer.id2token[t] for t in sid_ids), score))
        results.append(out)                                 # cand 已排序，beam 即降序
    return results
