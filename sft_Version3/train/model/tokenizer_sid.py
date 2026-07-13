#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tokenizer_sid: 统一词表（special + 行为 + 各层 SID token）与编解码。

职责（step3 字符串形态 <-> modeling.py 的三张量契约之间的桥）：
  1. 建词表：扫 item map parquet 的 geo_sid 列，按 '><' 切分成若干层
     （如 <9g3tck><a_20><b_187><c_151> = 4 层：geo/a/b/c），每层各自收集 token。
     层数由数据自动发现，不写死——以后引入去歧位 <d_*>（5 层）时重建词表即可，
     模型侧只需把 GAMERConfig.tokens_per_item 改成对应层数。
  2. encode：[(action, geo_sid), ...] -> input_ids / behavior_ids / token_types，
     每个交互 = 1 行为 token + num_levels 个 SID token：
       - encode_train_sample: [BOS] + 全部交互，labels = input_ids（全 token 监督）
       - encode_val_sample  : [BOS] + input区 + label区，labels 仅 label 区有效
         （input 区与 BOS 置 -100，teacher-forcing 算 val loss）
  3. decode：id 序列 -> [(action, geo_sid), ...]，供推理/评测还原比对。
  4. save/load vocab.json：训练/推理/评测共享同一份映射；按层存 token id 列表
     （level_token_ids），留给 constrained decoding 建合法路径 trie 用。

id 布局：<pad>=0, <bos>=1, <eos>=2, 行为 token（按层级升序，如 <clk> <pay>），
         然后逐层 SID token（层内按字符串排序，保证重建可复现）。

用法：
  构建词表:  python tokenizer_sid.py [common.conf] [输出vocab.json路径]
             （默认读项目根 common.conf 的 [item_map] item_map_path，
               输出到项目根 outputs/vocab.json）
  自测:      python tokenizer_sid.py selftest
"""

import json
import os
import sys
import configparser
from collections import Counter

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
# 行为 -> 层级（与 data/step3 的 BEHAVIOR_LEVELS、GAMERConfig.behavior_levels 保持一致；
# 行为 id 按层级升序编号：clk=0, pay=1）
BEHAVIOR_LEVELS = {"clk": 1, "pay": 2}
# 时段分桶（与 data/step2 的 get_meal_period 保持一致；顺序 = 时段 id）
PERIODS = ["bf", "lc", "dn"]


def split_sid(geo_sid: str) -> list:
    """'<9g3tck><a_20><b_187><c_151>' -> ['<9g3tck>', '<a_20>', '<b_187>', '<c_151>']。"""
    s = geo_sid.strip()
    if not (s.startswith("<") and s.endswith(">")):
        raise ValueError(f"非法 geo_sid（应为 <..><..> 形式）: {geo_sid!r}")
    return [f"<{p}>" for p in s[1:-1].split("><")]


class SIDTokenizer:
    def __init__(self, token2id: dict, behaviors: list, num_levels: int,
                 level_token_ids: list, periods: list = None):
        self.token2id = token2id
        self.id2token = {v: k for k, v in token2id.items()}
        self.behaviors = behaviors                      # 行为名列表，下标 = 行为 id
        self.behavior2id = {b: i for i, b in enumerate(behaviors)}
        self.num_levels = num_levels                    # 每个 item 的 SID 层数 l
        self.level_token_ids = level_token_ids          # [ [该层全部 token id], ... ]
        self.periods = periods or []                    # 时段名列表（空 = 无时段位）
        self.period_type = num_levels + 1               # 时段 token 的 token_types 值
        self.pad_id = token2id[PAD]
        self.bos_id = token2id[BOS]
        self.eos_id = token2id[EOS]

    @property
    def tokens_per_interaction(self) -> int:
        """每个交互的 token 数：[时段] + 行为 + l 层 SID。"""
        return (2 if self.periods else 1) + self.num_levels

    # ---------------- 构建 / 存取 ----------------
    @classmethod
    def from_item_map(cls, parquet_path: str, behaviors=None, periods=None):
        """扫 item map 的 geo_sid 列建词表。层数取多数派，层数异常的 sid 丢弃并告警。
           periods 非空时词表含时段 token（<bf> 等），序列形态变为
           时段+行为+SID；传 [] 可显式构建无时段词表。"""
        import pyarrow.parquet as pq
        behaviors = behaviors or sorted(BEHAVIOR_LEVELS, key=BEHAVIOR_LEVELS.get)
        periods = PERIODS if periods is None else periods
        col = pq.read_table(parquet_path, columns=["geo_sid"]).column("geo_sid").to_pylist()

        level_cnt = Counter()
        parsed = []
        for s in col:
            if not s:
                continue
            try:
                parts = split_sid(s)
            except ValueError:
                continue
            parsed.append(parts)
            level_cnt[len(parts)] += 1
        if not parsed:
            raise ValueError(f"item map 中没有可解析的 geo_sid: {parquet_path}")
        num_levels = level_cnt.most_common(1)[0][0]
        n_bad = sum(c for l, c in level_cnt.items() if l != num_levels)
        if n_bad:
            print(f"[WARN] {n_bad} 条 geo_sid 层数 != {num_levels}，已丢弃（分布: {dict(level_cnt)}）")

        level_tokens = [set() for _ in range(num_levels)]
        for parts in parsed:
            if len(parts) != num_levels:
                continue
            for lv, tok in enumerate(parts):
                level_tokens[lv].add(tok)

        token2id = {PAD: 0, BOS: 1, EOS: 2}
        for b in behaviors:
            token2id[f"<{b}>"] = len(token2id)
        for p in periods:
            token2id[f"<{p}>"] = len(token2id)
        level_token_ids = []
        for lv in range(num_levels):
            ids = []
            for tok in sorted(level_tokens[lv]):        # 排序保证重建可复现
                token2id[tok] = len(token2id)
                ids.append(token2id[tok])
            level_token_ids.append(ids)
        return cls(token2id, behaviors, num_levels, level_token_ids, periods)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "token2id": self.token2id,
                "behaviors": self.behaviors,
                "num_levels": self.num_levels,
                "level_token_ids": self.level_token_ids,
                "periods": self.periods,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["token2id"], d["behaviors"], d["num_levels"],
                   d["level_token_ids"], d.get("periods", []))

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def behavior_levels(self) -> tuple:
        """给 GAMERConfig.behavior_levels 用：下标 = 行为 id。"""
        return tuple(BEHAVIOR_LEVELS[b] for b in self.behaviors)

    # ---------------- 编码 ----------------
    def _norm_items(self, items) -> list:
        """兼容 step3 的 token dict（取 action/geo_sid/meal_period）与
           (action, geo_sid[, period]) 元组，统一为三元组（无时段词表时 period=None）。"""
        out = []
        for it in items:
            if isinstance(it, dict):
                out.append((it["action"], it["geo_sid"], it.get("meal_period")))
            else:
                out.append((it[0], it[1], it[2] if len(it) > 2 else None))
        return out

    def encode_items(self, items):
        """交互列表 -> (input_ids, behavior_ids, token_types)，不含 BOS。
           每个交互 = [时段 token +] 行为 token + num_levels 个 SID token；
           时段 token 的 token_types = num_levels+1，behavior_ids 沿用所属交互行为。"""
        ids, beh, types = [], [], []
        for action, geo_sid, period in self._norm_items(items):
            b = self.behavior2id[action]                # 未知行为直接 KeyError 暴露问题
            parts = split_sid(geo_sid)
            if len(parts) != self.num_levels:
                raise ValueError(f"geo_sid 层数 {len(parts)} != 词表层数 {self.num_levels}: {geo_sid!r}")
            if self.periods:
                if period is None:
                    raise ValueError(f"词表含时段位但交互缺 meal_period: {(action, geo_sid)!r}")
                ids.append(self.token2id[f"<{period}>"])  # 未知时段直接 KeyError
                beh.append(b)
                types.append(self.period_type)
            ids.append(self.token2id[f"<{action}>"])
            beh.append(b)
            types.append(0)                             # 行为 token
            for lv, tok in enumerate(parts):
                ids.append(self.token2id[tok])          # 未知 SID 直接 KeyError 暴露问题
                beh.append(b)
                types.append(lv + 1)                    # SID 第 lv+1 层
        return ids, beh, types

    def encode_train_sample(self, sample: dict) -> dict:
        """step3 train 样本 {token_seq: [...]} -> 全 token 监督的 NTP 样本。"""
        ids, beh, types = self.encode_items(sample["token_seq"])
        input_ids = [self.bos_id] + ids
        return {
            "input_ids": input_ids,
            "behavior_ids": [-1] + beh,
            "token_types": [-1] + types,
            "labels": list(input_ids),                  # 内部 shift，BOS 位置天然不当 target
        }

    def encode_val_sample(self, sample: dict) -> dict:
        """step3 val 样本 {input: [...], label_tokens: [...]} -> teacher-forcing 样本：
           labels 仅 label 区有效，[BOS]+input 区置 -100。"""
        in_ids, in_beh, in_types = self.encode_items(sample["input"])
        lb_ids, lb_beh, lb_types = self.encode_items(sample["label_tokens"])
        input_ids = [self.bos_id] + in_ids + lb_ids
        return {
            "input_ids": input_ids,
            "behavior_ids": [-1] + in_beh + lb_beh,
            "token_types": [-1] + in_types + lb_types,
            "labels": [-100] * (1 + len(in_ids)) + lb_ids,
        }

    # ---------------- 解码 ----------------
    def decode(self, ids) -> list:
        """id 序列 -> [(action, geo_sid, period), ...]（无时段词表时 period=None）。
           跳过 special；遇到 [时段+]行为 token 开始收一个 item 的 num_levels 个
           SID token，不完整的尾部 item 丢弃。"""
        beh_tokens = {self.token2id[f"<{b}>"]: b for b in self.behaviors}
        per_tokens = {self.token2id[f"<{p}>"]: p for p in self.periods}
        level_sets = [set(ids_) for ids_ in self.level_token_ids]
        out, i, n = [], 0, len(ids)
        while i < n:
            period = None
            j = i
            if self.periods and ids[j] in per_tokens and j + 1 < n:
                period = per_tokens[ids[j]]
                j += 1
            tid = ids[j]
            span = ids[j + 1: j + 1 + self.num_levels]
            if tid in beh_tokens and len(span) == self.num_levels and \
                    all(t in level_sets[lv] for lv, t in enumerate(span)):
                out.append((beh_tokens[tid],
                            "".join(self.id2token[t] for t in span), period))
                i = j + 1 + self.num_levels
            else:
                i += 1                                  # special / 残缺片段，跳过
        return out


# ------------------------------------------------------------------
# 入口：建词表 / 自测
# ------------------------------------------------------------------
def _build_from_conf(conf_path: str, out_path: str):
    cp = configparser.ConfigParser()
    if not cp.read(conf_path, encoding="utf-8"):
        raise FileNotFoundError(f"找不到配置文件: {conf_path}")
    parquet_path = cp.get("item_map", "item_map_path")
    print(f"[INFO] item map: {parquet_path}")
    tok = SIDTokenizer.from_item_map(parquet_path)
    tok.save(out_path)
    print(f"[INFO] vocab 已保存: {out_path}")
    print(f"[INFO] vocab_size={tok.vocab_size}  num_levels={tok.num_levels}  "
          f"behaviors={tok.behaviors}  periods={tok.periods}  "
          f"每交互 token 数={tok.tokens_per_interaction}")
    for lv, ids in enumerate(tok.level_token_ids):
        print(f"  SID 第{lv + 1}层 token 数: {len(ids)}")


def _selftest():
    """无 parquet 依赖的自测：手工建小词表，验证 encode/decode 往返与 val mask
       （无时段 / 有时段两种形态都测）。"""
    behaviors = ["clk", "pay"]

    def build(periods):
        token2id = {PAD: 0, BOS: 1, EOS: 2, "<clk>": 3, "<pay>": 4}
        for p in periods:
            token2id[f"<{p}>"] = len(token2id)
        level_tokens = [["<g1>", "<g2>"], ["<a_0>", "<a_1>"],
                        ["<b_0>", "<b_1>"], ["<c_0>", "<c_1>"]]
        level_token_ids = []
        for toks in level_tokens:
            ids = []
            for t in toks:
                token2id[t] = len(token2id)
                ids.append(token2id[t])
            level_token_ids.append(ids)
        return SIDTokenizer(token2id, behaviors, 4, level_token_ids, periods)

    # ---- 无时段（旧形态，每交互 5 token）----
    tok = build([])
    items = [("clk", "<g1><a_0><b_1><c_0>"), ("pay", "<g2><a_1><b_0><c_1>")]
    train = tok.encode_train_sample({"token_seq": items})
    assert tok.tokens_per_interaction == 5
    assert len(train["input_ids"]) == 1 + 2 * 5
    assert train["behavior_ids"] == [-1] + [0] * 5 + [1] * 5
    assert train["token_types"] == [-1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert train["labels"] == train["input_ids"]
    assert tok.decode(train["input_ids"]) == [i + (None,) for i in items]

    val = tok.encode_val_sample({"input": items[:1], "label_tokens": items[1:]})
    assert val["labels"][:6] == [-100] * 6              # BOS + input 区 5 token
    assert val["labels"][6:] == val["input_ids"][6:]    # label 区原样

    # ---- 有时段（每交互 6 token：时段+行为+4 SID）----
    tkp = build(["bf", "lc", "dn"])
    items_p = [("clk", "<g1><a_0><b_1><c_0>", "bf"),
               ("pay", "<g2><a_1><b_0><c_1>", "dn")]
    tr = tkp.encode_train_sample({"token_seq": items_p})
    assert tkp.tokens_per_interaction == 6
    assert len(tr["input_ids"]) == 1 + 2 * 6
    assert tr["input_ids"][1] == tkp.token2id["<bf>"]   # 时段在行为前
    assert tr["input_ids"][2] == tkp.token2id["<clk>"]
    assert tr["behavior_ids"] == [-1] + [0] * 6 + [1] * 6
    assert tr["token_types"] == [-1, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
    assert tkp.period_type == 5
    assert tkp.decode(tr["input_ids"]) == items_p       # 往返含时段
    # 缺时段报错
    try:
        tkp.encode_items([("clk", "<g1><a_0><b_1><c_0>")])
        raise AssertionError("缺 meal_period 应报错")
    except ValueError:
        pass

    # save/load 往返（含 periods）
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "vocab.json")
        tkp.save(p)
        tok2 = SIDTokenizer.load(p)
        assert tok2.periods == ["bf", "lc", "dn"] and tok2.num_levels == 4
        assert tok2.encode_train_sample({"token_seq": items_p}) == tr
    assert tok.behavior_levels == (1, 2)
    print("selftest passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", ".."))
        conf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "common.conf")
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "outputs", "vocab.json")
        _build_from_conf(conf, out)
