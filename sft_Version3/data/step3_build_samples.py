#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_build_samples（分割阶段）: session 粒度留一法，切分 train / val / test。

本轮只做「分割」，供人工核对（不做增广、不落盘、全程流式）：
  1. 内存链式复用 step2 的 map_sample 拿到每个用户的 pay/clk 映射；
  2. 合并 pay+clk 成一条按时间排序的时间线，每 token 带 action（clk/pay），
     丢弃找不到 geo_sid 的交互；
  3. 按【本地自然日】切 session（session_granularity=day）：Su=[S1,...,Sm]；
  4. 留一法三分（每个用户独立）：
       test :  input = S1..S(m-1)（含 val 的 session），label 区 = Sm
       val  :  input = S1..S(m-2),                    label 区 = S(m-1)
       train:  剩余交互 S1..S(m-2)（本轮只圈出区域，label 分割/增广下一步做）
     边界：m<2 无历史跳过；m==2 只有 test；m>=3 才有 val 和 train。
  5. label 区按「目标行为 b」取正样本集合 𝒯u = 区内以 b 交互的所有 item
     （这里把 label session 按 action 分组打印，pay/clk 两个集合都展示）。
  6. 打印前若干用户的三分拆解供核对。

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


def split_user_sessions(sessions: list):
    """session 粒度留一法三分。返回 dict 或 None（m<2）。
       test.input = S1..S(m-1)；val.input = S1..S(m-2)；train_region = S1..S(m-2)。"""
    m = len(sessions)
    if m < 2:
        return None
    test = {
        "input": [t for _, toks in sessions[:m - 1] for t in toks],
        "label_date": sessions[m - 1][0],
        "label_session": sessions[m - 1][1],
    }
    val = None
    if m >= 3:
        val = {
            "input": [t for _, toks in sessions[:m - 2] for t in toks],
            "label_date": sessions[m - 2][0],
            "label_session": sessions[m - 2][1],
        }
    train_region = sessions[:m - 2]  # (date, toks) 列表，m==2 时为空
    return {"test": test, "val": val, "train_region": train_region}


def group_by_action(tokens: list) -> dict:
    """label 区按 action 分组 -> {action: [token...]}，即各目标行为 b 的正样本集合。"""
    g = {}
    for t in tokens:
        g.setdefault(t["action"], []).append(t)
    return g


def _fmt_tok(t: dict) -> str:
    return f"<{t['action']}>{t['geo_sid']}"


def print_user_split(uid: str, sessions: list, split: dict):
    m = len(sessions)
    print(f"\n========== 用户 uid={uid}  共 {m} 个 session ==========")
    for i, (date, toks) in enumerate(sessions):
        if i == m - 1:
            role = "[test-label]"
        elif i == m - 2:
            role = "[val-label]"
        else:
            role = "[train]"
        print(f"  session#{i} {role} date={date}  {len(toks)} 交互: {[_fmt_tok(t) for t in toks]}")

    def show_split(name, part):
        if part is None:
            print(f"\n  -- {name}: 无（session 数不足）")
            return
        print(f"\n  -- {name} --")
        print(f"     input（{len(part['input'])} 交互）: {[_fmt_tok(t) for t in part['input']]}")
        groups = group_by_action(part["label_session"])
        print(f"     label 区 date={part['label_date']}，各目标行为的正样本集合 𝒯u:")
        for act, toks in groups.items():
            print(f"        b={act}: {[t['geo_sid'] for t in toks]}")

    show_split("TEST", split["test"])
    show_split("VAL", split["val"])
    tr = split["train_region"]
    tr_tokens = sum(len(toks) for _, toks in tr)
    print(f"\n  -- TRAIN 区域（S1..S(m-2)，{len(tr)} 个 session / {tr_tokens} 交互，"
          f"label 分割+增广下一步做）--")
    print(f"     {[(d, [_fmt_tok(t) for t in toks]) for d, toks in tr]}")
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

    unlimited = cfg["max_num"] == -1
    max_desc = "全量窗口数据" if unlimited else str(cfg["max_num"])
    print(f"[INFO] 序列字段: {seq_fields}  session=按天  留一法三分 train/val/test")
    print(f"[INFO] 窗口: {cfg['train_start']} ~ {cfg['train_end']}，期望处理 {max_desc} 行\n")

    n = 0
    n_users = 0
    session_hist = Counter()      # 每用户 session 数分布
    n_test = 0                    # 有 test（m>=2）
    n_val = 0                     # 有 val（m>=3）
    n_train = 0                   # train 区域非空（m>=3）
    printed = 0

    for _dt, _hdfs, _schema, row in stream_rows(cfg):
        mapped = map_sample(row, id2sid, seq_fields)
        timeline = build_timeline(mapped, seq_fields, tz_offset)
        sessions = sessionize_by_day(timeline)
        n_users += 1
        m = len(sessions)
        session_hist[min(m, 5)] += 1

        split = split_user_sessions(sessions)
        if split is not None:
            n_test += 1
            if split["val"] is not None:
                n_val += 1
            if split["train_region"]:
                n_train += 1
            if printed < cfg["log_sample_count"]:
                print_user_split(row.get("uid"), sessions, split)
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
    print(f"有 test 的用户(m>=2): {n_test}")
    print(f"有 val  的用户(m>=3): {n_val}")
    print(f"有 train区域 的用户(m>=3): {n_train}")
    print("=====================================")


if __name__ == "__main__":
    main()
