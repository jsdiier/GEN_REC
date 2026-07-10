#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_build_samples（分割 + label 分割）: session 粒度留一法，构造 train/val/test 样本。

流程（不落盘、全程流式）：
  1. 内存链式复用 step2 的 map_sample 拿到每个用户的 pay/clk 映射；
  2. 合并 pay+clk 成一条按时间排序的时间线，每 token 带 action（clk/pay），
     丢弃找不到 geo_sid 的交互；
  3. 按【本地自然日】切 session（session_granularity=day）：Su=[S1,...,Sm]；
  4. 留一法三分（每个用户独立）：
       test :  input = S1..S(m-1)（含 val 的 session），label 区 = Sm
       val  :  input = S1..S(m-2),                    label 区 = S(m-1)
       train:  label session 在区域 S1..S(m-2) 内取：
                 slide(默认) -> S2..S(m-2) 每个 session 轮流当 label，input=它之前的
                 last        -> 只用 S(m-2) 当 label
     边界：m<2 无样本；m==2 仅 test；m==3 test+val；m>=4 才有 train。
  5. label 分割：label 区按「目标行为 b」取正样本集合 𝒯u = 区内以 b 交互的所有 item（去重）。
     每条样本 = (input, target_action, label_geo_sids 集合)；train 阶段再展开/采样。
  6. 打印前若干用户的样本拆解 + 各 split 样本量汇总供核对。

（train 的 behavior-drop 增强下一步做，且只作用于 train、val/test 不变。）

用法:
    python3 step3_build_samples.py [common.conf]
"""

import os
import sys
from collections import Counter

from step1_get_user_action import load_config, stream_rows
from step2_inject_sid import (map_sample, load_item_sid_map,
                              get_item_map_path, local_time_and_period)


def field_to_action(field: str) -> str:
    """由序列字段名推断行为类型。"""
    if "pay" in field:
        return "pay"
    if "clk" in field:
        return "clk"
    return field


def build_timeline(mapped: dict, seq_fields: list, tz_offset_hours: int) -> list:
    """合并所有字段的 item 成一条按时间排序的时间线，丢弃缺失 geo_sid / 无时间的 item。
    每个 token: {action, item_id, geo_sid, ts, date, local_hour, meal_period, ts_str}。"""
    timeline = []
    for f in seq_fields:
        action = field_to_action(f)
        for it in mapped["fields"][f]["items"]:
            if not it["geo_sid"]:
                continue  # 丢弃找不到 geo_sid 的交互
            try:
                ts = int(it["phone_time_local"])
            except (TypeError, ValueError):
                continue  # 无法定位时间的丢弃（无法 sessionize）
            ts_str, meal_period = local_time_and_period(it["phone_time_local"], tz_offset_hours)
            timeline.append({
                "action": action,
                "item_id": it["item_id"],
                "geo_sid": it["geo_sid"],
                "ts": ts,
                "date": ts_str[:10] if ts_str else None,  # 本地自然日
                "local_hour": it["local_hour"],
                "meal_period": meal_period,
                "ts_str": ts_str,
            })
    timeline.sort(key=lambda x: x["ts"])
    return timeline


def sessionize_by_day(timeline: list) -> list:
    """时间线（已按时间排序）按本地自然日切 session：返回 [(date, [token...]), ...]。"""
    sessions = []
    for tok in timeline:
        if sessions and sessions[-1][0] == tok["date"]:
            sessions[-1][1].append(tok)
        else:
            sessions.append((tok["date"], [tok]))
    return sessions


def _fmt_tok(t: dict) -> str:
    return f"<{t['action']}>{t['geo_sid']}"


def positives_by_action(label_tokens: list) -> dict:
    """label 区按目标行为 b 分组成正样本 geo_sid 集合（去重，保序）。
       {action: [geo_sid, ...]}。"""
    g = {}
    for t in label_tokens:
        lst = g.setdefault(t["action"], [])
        if t["geo_sid"] not in lst:
            lst.append(t["geo_sid"])
    return g


def _make_samples(uid, split, input_tokens, label_date, label_tokens):
    """label 区内每个目标行为 b 出一条样本：label = 该 b 的正样本集合（下游 train 再展开/采样）。"""
    samples = []
    for action, geo_sids in positives_by_action(label_tokens).items():
        samples.append({
            "uid": uid,
            "split": split,
            "target_action": action,
            "input": input_tokens,
            "label_geo_sids": geo_sids,
            "label_date": label_date,
        })
    return samples


def build_all_samples(uid: str, sessions: list, train_label_scope: str) -> list:
    """session 粒度留一法，构造 train/val/test 样本（label 为正样本集合）。
       test  : input=S1..S(m-1), label=Sm
       val   : input=S1..S(m-2), label=S(m-1)
       train : slide -> S2..S(m-2) 每个 session 当 label；last -> 仅 S(m-2)
       （sessions 0-indexed: S1=sessions[0] ... Sm=sessions[m-1]）"""
    m = len(sessions)
    out = []
    if m < 2:
        return out

    # test
    test_input = [t for _, toks in sessions[:m - 1] for t in toks]
    out += _make_samples(uid, "test", test_input, sessions[m - 1][0], sessions[m - 1][1])

    # val (需 m>=3)
    if m >= 3:
        val_input = [t for _, toks in sessions[:m - 2] for t in toks]
        out += _make_samples(uid, "val", val_input, sessions[m - 2][0], sessions[m - 2][1])

    # train：label session 落在 train 区域 S1..S(m-2) 内，且需有更早 session 作 input
    #   可当 label 的下标 s ∈ [1, m-3]（0-indexed），对应 S2..S(m-2)
    if m >= 4:
        if train_label_scope == "last":
            train_idxs = [m - 3]
        else:  # slide
            train_idxs = list(range(1, m - 2))
        for s in train_idxs:
            tr_input = [t for _, toks in sessions[:s] for t in toks]
            out += _make_samples(uid, "train", tr_input, sessions[s][0], sessions[s][1])

    return out


def print_user_samples(uid: str, sessions: list, samples: list):
    m = len(sessions)
    print(f"\n========== 用户 uid={uid}  共 {m} 个 session ==========")
    for i, (date, toks) in enumerate(sessions):
        if i == m - 1:
            role = "[test-label]"
        elif i == m - 2:
            role = "[val-label]"
        else:
            role = "[train区域]"
        print(f"  session#{i} {role} date={date}  {len(toks)} 交互: {[_fmt_tok(t) for t in toks]}")

    print(f"\n  生成 {len(samples)} 条样本（input 只展示尾部 3 个交互 + 长度）:")
    for s in samples:
        inp = s["input"]
        tail = [_fmt_tok(t) for t in inp[-3:]]
        print(f"    [{s['split']:<5}/{s['target_action']:<3}] "
              f"input_len={len(inp)} tail={tail} "
              f"-> label({s['target_action']})@{s['label_date']}: {s['label_geo_sids']}")
    print("=" * 64)


def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "common.conf"
    )
    cfg = load_config(conf_path)
    seq_fields = cfg["seq_fields"]
    tz_offset = cfg["tz_offset_hours"]

    if cfg["session_granularity"] != "day":
        print(f"[WARN] 目前仅支持 session_granularity=day，收到 "
              f"{cfg['session_granularity']}，按 day 处理")

    item_map_path = get_item_map_path(conf_path)
    print(f"[INFO] 加载映射表: {item_map_path}")
    id2sid = load_item_sid_map(item_map_path)
    print(f"[INFO] 映射表条目数: {len(id2sid)}")

    train_label_scope = cfg["train_label_scope"]
    unlimited = cfg["max_num"] == -1
    max_desc = "全量窗口数据" if unlimited else str(cfg["max_num"])
    print(f"[INFO] 序列字段: {seq_fields}  session=按天  留一法三分 train/val/test")
    print(f"[INFO] train_label_scope={train_label_scope}（label 为正样本集合，train 阶段再展开/采样）")
    print(f"[INFO] 窗口: {cfg['train_start']} ~ {cfg['train_end']}，期望处理 {max_desc} 行\n")

    n = 0
    n_users = 0
    session_hist = Counter()          # 每用户 session 数分布
    split_samples = Counter()         # 各 split 样本数
    total_pos = Counter()             # 各 split 正样本(去重后)累计，估平均集合大小
    printed = 0

    for _dt, _hdfs, _schema, row in stream_rows(cfg):
        mapped = map_sample(row, id2sid, seq_fields)
        timeline = build_timeline(mapped, seq_fields, tz_offset)
        sessions = sessionize_by_day(timeline)
        n_users += 1
        session_hist[min(len(sessions), 5)] += 1

        samples = build_all_samples(row.get("uid"), sessions, train_label_scope)
        for s in samples:
            split_samples[s["split"]] += 1
            total_pos[s["split"]] += len(s["label_geo_sids"])
        if samples and printed < cfg["log_sample_count"]:
            print_user_samples(row.get("uid"), sessions, samples)
            printed += 1

        n += 1
        if not unlimited and n >= cfg["max_num"]:
            break
        if unlimited and n % 100000 == 0:
            print(f"[INFO] 已处理 {n} 行")

    print("\n================ 汇总 ================")
    print(f"处理用户数: {n_users}")
    print("每用户 session 数分布:")
    for k in sorted(session_hist):
        label = "5+" if k == 5 else str(k)
        print(f"  {label} 个 session: {session_hist[k]}")
    print("各 split 样本数（每条样本 = 一个 (input, target_action, 正样本集合)）:")
    for sp in ("train", "val", "test"):
        cnt = split_samples.get(sp, 0)
        avg = (total_pos.get(sp, 0) / cnt) if cnt else 0.0
        print(f"  {sp:<5}: {cnt} 条  (平均正样本集合大小 {avg:.2f})")
    print("=====================================")


if __name__ == "__main__":
    main()
