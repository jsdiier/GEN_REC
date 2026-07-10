#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_build_samples（GAMER 对齐版）: session 粒度留一法 + user-level 全序列 train + behavior-drop 增强。

流程（全程流式；[data] samples_out_dir 配置后把样本写成 train.jsonl / val.jsonl
供 train/dataset 读取，落盘只留 (action, geo_sid)，留空则仅打印诊断）：
  1. 内存链式复用 step2 的 map_sample 拿到每个用户的 pay/clk 映射；
  2. 合并 pay+clk 成一条按时间排序的时间线，每 token 带 action（clk/pay），
     丢弃找不到 geo_sid 的交互；
  3. 按【本地自然日】切 session（session_granularity=day）：Su=[S1,...,Sm]；
  4. 用户过滤：session 数 m < 2 的用户整体剔除（留不出 val session）；
  5. 留一法两分（每个用户独立，m>=2）。test 不在本步产出——后续基于
     common.conf [data] test_start/test_end 的独立时间窗口另建测试流程：
       val  :  input = S1..S(m-1), label 区 = Sm（train 窗口内每用户最后一个 session）
       train:  user-level 全序列 = S1..S(m-1) 按时间原序拼接（保留重复交互），
               不区分 target_action、不设 label 区——下游做 next-token prediction，
               labels = input_ids 整体左移，全 token 监督（对齐论文 Eq.6）。
  6. train 的 behavior-drop 增强（对齐论文 3.3）：x = behavior_drop_x，
       对整条 train 序列生成 x 个变体，丢弃比例 ri = i/(x+1) (i=1..x)；
       行为层级 clk=1 < pay=2，最高层 pay 一条不丢，其余行为 b 按比例 ri/L_b 随机丢弃（保序）。
       每用户最多输出 x+1 条 train 序列（原始 + x 个变体）；val 不增强。
       两个退化保护：
         - train 区域交互数 < min_train_seq_len（conf，默认2）的用户不出 train 序列
           （长度 1 学不到转移模式），val 照常保留；
         - 变体若未实际丢弃任何交互（与原始序列相同，如全是 pay），跳过不输出，避免重复。
  7. val 的形态（teacher-forcing 算 val loss 用）：每用户一条样本，
     label_tokens = val session 的原始有序交互（含重复、含行为），下游拼接
     input+label_tokens 做 NTP，loss 只算 label 区 token（input 区置 -100）；
     附带 positives_by_action（按行为去重的正样本集合），留给以后算 HR/NDCG。
  8. 打印前若干用户的样本拆解 + 各 split 样本量汇总供核对。

与旧版（SFT 式）的差异：
  - train 不再是 slide/last 的 (input, label集合) 对，train_label_scope 配置已废弃不读；
  - train 序列保留重复交互（论文 footnote 2），不去重、不采样；
  - test 不由本步产出（改为 test_start/test_end 独立窗口的测试流程）。

用法:
    python3 step3_build_samples.py [common.conf]
"""

import os
import sys
import json
import random
from collections import Counter

from step1_get_user_action import load_config, stream_rows
from step2_inject_sid import (map_sample, load_item_sid_map,
                              get_item_map_path, local_time_and_period)


# 行为层级（对齐论文 3.3：最低层为 1，随行为深度递增；最高层行为不参与 drop）
BEHAVIOR_LEVELS = {"clk": 1, "pay": 2}
HIGHEST_BEHAVIOR = max(BEHAVIOR_LEVELS, key=BEHAVIOR_LEVELS.get)


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


def _slim(tokens: list) -> list:
    """token dict 列表 -> [(action, geo_sid), ...]，落盘只留模型需要的两个字段。"""
    return [(t["action"], t["geo_sid"]) for t in tokens]


def positives_by_action(label_tokens: list) -> dict:
    """label 区按目标行为 b 分组成正样本 geo_sid 集合（去重，保序）。
       {action: [geo_sid, ...]}。仅用于 val/test 评测样本。"""
    g = {}
    for t in label_tokens:
        lst = g.setdefault(t["action"], [])
        if t["geo_sid"] not in lst:
            lst.append(t["geo_sid"])
    return g


def _make_val_sample(uid, input_tokens, label_date, label_tokens) -> dict:
    """val：每用户一条样本。label_tokens 为 val session 原始有序交互
       （teacher-forcing 算 val loss）；positives_by_action 为按行为去重的
       正样本集合（留给以后算 HR/Recall/NDCG）。"""
    return {
        "uid": uid,
        "split": "val",
        "input": input_tokens,
        "label_tokens": label_tokens,
        "positives_by_action": positives_by_action(label_tokens),
        "label_date": label_date,
    }


def drop_augment(tokens: list, r: float, rng: random.Random) -> list:
    """对整条 train 序列做 behavior-drop（论文 3.3）：
       除最高层行为外，每种行为 b 随机丢弃 round(n_b * r / L_b) 条交互，保序。"""
    idx_by_action = {}
    for i, t in enumerate(tokens):
        if t["action"] != HIGHEST_BEHAVIOR:
            idx_by_action.setdefault(t["action"], []).append(i)
    drop_idx = set()
    for action, idxs in idx_by_action.items():
        level = BEHAVIOR_LEVELS.get(action, 1)
        k = round(len(idxs) * r / level)
        if k > 0:
            drop_idx.update(rng.sample(idxs, k))
    return [t for i, t in enumerate(tokens) if i not in drop_idx]


def build_train_sequences(uid, sessions: list, x: int, min_seq_len: int) -> list:
    """train：S1..S(m-1) 原序拼接成一条 user-level 全序列（保留重复交互），
       外加 x 个 behavior-drop 变体（ri = i/(x+1)）。每用户最多输出 x+1 条：
       - 区域交互数 < min_seq_len 时不出任何 train 序列（val 不受影响）；
       - 变体与原始相同（什么都没丢到）时跳过。
       rng 以 (uid, i) 做种子，保证可复现。"""
    train_tokens = [t for _, toks in sessions[:-1] for t in toks]
    if len(train_tokens) < min_seq_len:
        return []
    out = [{"uid": uid, "split": "train", "aug_r": 0.0, "token_seq": train_tokens}]
    for i in range(1, x + 1):
        r = i / (x + 1)
        rng = random.Random(f"{uid}|aug{i}")
        aug = drop_augment(train_tokens, r, rng)
        if len(aug) == len(train_tokens):
            continue  # drop_augment 只删不改，长度不变即与原始完全相同
        out.append({"uid": uid, "split": "train", "aug_r": round(r, 4),
                    "token_seq": aug})
    return out


def build_all_samples(uid: str, sessions: list, behavior_drop_x: int,
                      min_train_seq_len: int) -> list:
    """session 粒度留一法两分（m>=2 的用户才保留；test 由独立窗口流程另建）：
       val   : input=S1..S(m-1), label=Sm（每用户一条，label_tokens 有序）
       train : S1..S(m-1) 全序列 + 去重后的 drop 变体（NTP 序列，无 label 区）
       （sessions 0-indexed: S1=sessions[0] ... Sm=sessions[m-1]）"""
    m = len(sessions)
    if m < 2:
        return []  # 留不出 val session 的用户整体剔除

    out = []
    # val：train 窗口内每用户最后一个 session 当 label 区（每用户一条）
    val_input = [t for _, toks in sessions[:m - 1] for t in toks]
    out.append(_make_val_sample(uid, val_input, sessions[m - 1][0], sessions[m - 1][1]))
    # train（user-level 全序列 + 增强）
    out += build_train_sequences(uid, sessions, behavior_drop_x, min_train_seq_len)
    return out


def print_user_samples(uid: str, sessions: list, samples: list):
    m = len(sessions)
    print(f"\n========== 用户 uid={uid}  共 {m} 个 session ==========")
    for i, (date, toks) in enumerate(sessions):
        role = "[val-label]" if i == m - 1 else "[train区域]"
        print(f"  session#{i} {role} date={date}  {len(toks)} 交互: {[_fmt_tok(t) for t in toks]}")

    print(f"\n  生成 {len(samples)} 条样本:")
    for s in samples:
        if s["split"] == "train":
            seq = s["token_seq"]
            n_by_action = Counter(t["action"] for t in seq)
            print(f"    [train/r={s['aug_r']:<6}] seq_len={len(seq)} "
                  f"({dict(n_by_action)}) seq={[_fmt_tok(t) for t in seq]}")
        else:
            inp = s["input"]
            tail = [_fmt_tok(t) for t in inp[-3:]]
            print(f"    [val] input_len={len(inp)} tail={tail} "
                  f"-> label区@{s['label_date']} {len(s['label_tokens'])} 交互(有序): "
                  f"{[_fmt_tok(t) for t in s['label_tokens']]}")
            print(f"          positives_by_action: "
                  f"{ {a: len(v) for a, v in s['positives_by_action'].items()} }")
    print("=" * 64)


def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "common.conf"
    )
    cfg = load_config(conf_path)
    seq_fields = cfg["seq_fields"]
    tz_offset = cfg["tz_offset_hours"]
    behavior_drop_x = cfg["behavior_drop_x"]
    min_train_seq_len = cfg["min_train_seq_len"]

    if cfg["session_granularity"] != "day":
        print(f"[WARN] 目前仅支持 session_granularity=day，收到 "
              f"{cfg['session_granularity']}，按 day 处理")

    item_map_path = get_item_map_path(conf_path)
    print(f"[INFO] 加载映射表: {item_map_path}")
    id2sid = load_item_sid_map(item_map_path)
    print(f"[INFO] 映射表条目数: {len(id2sid)}")

    unlimited = cfg["max_num"] == -1
    max_desc = "全量窗口数据" if unlimited else str(cfg["max_num"])
    print(f"[INFO] 序列字段: {seq_fields}  session=按天  "
          f"训练/验证两分（剔除 <2 session 用户；test 由 test_start/test_end 独立流程另建）")
    print(f"[INFO] train = user-level 全序列 NTP（GAMER 式，无 prompt/label 之分）")
    print(f"[INFO] behavior-drop 增强 x={behavior_drop_x}"
          f"（ri=i/(x+1)，每用户最多 {behavior_drop_x + 1} 条 train 序列；"
          f"最高层行为 {HIGHEST_BEHAVIOR} 不丢；与原始相同的变体跳过）")
    print(f"[INFO] train 区域最少交互数 min_train_seq_len={min_train_seq_len}"
          f"（不足的用户只出 val 不出 train）")
    out_dir = cfg["samples_out_dir"]
    writers = None
    if out_dir:
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(os.path.dirname(os.path.abspath(conf_path)), out_dir)
        os.makedirs(out_dir, exist_ok=True)
        writers = {sp: open(os.path.join(out_dir, f"{sp}.jsonl"), "w", encoding="utf-8")
                   for sp in ("train", "val")}
        print(f"[INFO] 样本落盘: {out_dir}/{{train,val}}.jsonl")
    else:
        print("[INFO] samples_out_dir 未配置，仅打印诊断不落盘")
    print(f"[INFO] 窗口: {cfg['train_start']} ~ {cfg['train_end']}，期望处理 {max_desc} 行\n")

    n = 0
    n_users = 0
    n_users_kept = 0                  # m>=2 被保留的用户
    session_hist = Counter()          # 每用户 session 数分布
    split_samples = Counter()         # 各 split 样本数
    total_pos = Counter()             # val label 区交互数累计，估平均 label 区长度
    train_seq_len = Counter()         # {aug_r: 累计 token 数}，估平均序列长度
    train_seq_cnt = Counter()         # {aug_r: 序列条数}
    printed = 0

    for _dt, _hdfs, _schema, row in stream_rows(cfg):
        mapped = map_sample(row, id2sid, seq_fields)
        timeline = build_timeline(mapped, seq_fields, tz_offset)
        sessions = sessionize_by_day(timeline)
        n_users += 1
        session_hist[min(len(sessions), 5)] += 1

        samples = build_all_samples(row.get("uid"), sessions,
                                    behavior_drop_x, min_train_seq_len)
        if samples:
            n_users_kept += 1
        for s in samples:
            split_samples[s["split"]] += 1
            if s["split"] == "train":
                train_seq_len[s["aug_r"]] += len(s["token_seq"])
                train_seq_cnt[s["aug_r"]] += 1
                rec = {"uid": s["uid"], "aug_r": s["aug_r"],
                       "token_seq": _slim(s["token_seq"])}
            else:
                total_pos[s["split"]] += len(s["label_tokens"])
                rec = {"uid": s["uid"], "input": _slim(s["input"]),
                       "label_tokens": _slim(s["label_tokens"]),
                       "positives_by_action": s["positives_by_action"],
                       "label_date": s["label_date"]}
            if writers:
                writers[s["split"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
        if samples and printed < cfg["log_sample_count"]:
            print_user_samples(row.get("uid"), sessions, samples)
            printed += 1

        n += 1
        if not unlimited and n >= cfg["max_num"]:
            break
        if unlimited and n % 100000 == 0:
            print(f"[INFO] 已处理 {n} 行")

    print("\n================ 汇总 ================")
    print(f"处理用户数: {n_users}  (保留 m>=2: {n_users_kept}, "
          f"剔除 <2 session: {n_users - n_users_kept})")
    print("每用户 session 数分布:")
    for k in sorted(session_hist):
        label = "5+" if k == 5 else str(k)
        print(f"  {label} 个 session: {session_hist[k]}")
    print("train（user-level NTP 序列，每用户 x+1 条）:")
    for r in sorted(train_seq_cnt):
        cnt = train_seq_cnt[r]
        avg = train_seq_len[r] / cnt if cnt else 0.0
        tag = "原始" if r == 0.0 else f"r={r}"
        print(f"  {tag:<8}: {cnt} 条  (平均 seq_len {avg:.2f})")
    print(f"  train 合计: {split_samples.get('train', 0)} 条")
    print("val（每用户一条，label_tokens 为原始有序交互，teacher-forcing 算 val loss）:")
    cnt = split_samples.get("val", 0)
    avg = (total_pos.get("val", 0) / cnt) if cnt else 0.0
    print(f"  val  : {cnt} 条  (平均 label 区交互数 {avg:.2f})")
    if writers:
        for sp, w in writers.items():
            w.close()
        print(f"已落盘: {out_dir}/train.jsonl, {out_dir}/val.jsonl")
    print("=====================================")


if __name__ == "__main__":
    main()
