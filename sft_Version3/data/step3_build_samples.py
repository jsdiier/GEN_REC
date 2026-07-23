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
  4. 用户过滤：session 数 m < 2 的用户整体剔除（留不出任何 label session）；
  5. 留一法三分（对齐论文 session-wise leave-one-out，每个用户独立）：
       m>=3: train:  S1..S(m-2) 按时间原序拼接（保留重复交互）成 user-level
                     全序列，不区分 target_action、不设 label 区——下游做
                     next-token prediction，labels 整体左移全 token 监督（Eq.6）
             val  :  input = S1..S(m-2), label 区 = S(m-1)
             test :  input = S1..S(m-1), label 区 = Sm
       m==2: 只出 val（input=S1, label=S2），不出 train/test——S2 给了 val
             （best.pt 按 val loss 挑选）就不能再当 test label，避免选择性泄漏。
     最后一个 session 天然是 train/val 都没见过的，test 不依赖日历日期；
     eval/run_eval.py 用 [data] test_start/test_end 的快照取 test 样本。
  6. train 的 behavior-drop 增强（对齐论文 3.3）：x = behavior_drop_x，
       对整条 train 序列生成 x 个变体，丢弃比例 ri = i/(x+1) (i=1..x)；
       行为层级 clk=1 < pay=2，最高层 pay 一条不丢，其余行为 b 按比例 ri/L_b 随机丢弃（保序）。
       每用户最多输出 x+1 条 train 序列（原始 + x 个变体）；val 不增强。
       两个退化保护：
         - train 区域交互数 < min_train_seq_len（conf，默认2）的用户不出 train 序列
           （长度 1 学不到转移模式），val 照常保留；
         - 变体若未实际丢弃任何交互（与原始序列相同，如全是 pay），跳过不输出，避免重复。
  7. val/test 的形态（同构）：每用户各一条样本，label_tokens = label session 的
     原始有序交互（含重复、含行为）——val 下游拼 input+label 做 teacher-forcing
     只算 label 区 loss；test 下游用 input 做前缀推理，与 positives_grouped
     （按 (时段,行为) 去重的正样本集合，配合 favor_coord_raw）比对算 HR/Recall/NDCG。
  8. 打印前若干用户的样本拆解 + 各 split 样本量汇总供核对。

与旧版（SFT 式）的差异：
  - train 不再是 slide/last 的 (input, label集合) 对，train_label_scope 配置已废弃不读；
  - train 序列保留重复交互（论文 footnote 2），不去重、不采样。

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
    """token dict 列表 -> [(action, geo_sid, meal_period), ...]，落盘只留模型需要的字段。
       用于 input / train 的 token_seq——这两处不参与评测分组，不需要占位字段。"""
    return [(t["action"], t["geo_sid"], t["meal_period"]) for t in tokens]


def _slim_label_tokens(tokens: list) -> list:
    """label_tokens 专用：在 _slim 的 3 元组后面多留 4 个占位字段
       (req_lng, req_lat, req_geohash, group_suffix)，对应每次交互的请求经纬度、
       换算出的 geohash、以及它落在用户 geohash_1/2/3 里的哪个（都不落=None）。
       上游这个per-交互请求经纬度字段还没 ready，先占位 None；等数据到位只用改
       这里怎么填值，不用再改一次 label_tokens 的形状——tokenizer 的 _norm_items
       只读前 3 个元素，多出来的占位字段不影响训练侧的 encode_val_sample。"""
    return [(t["action"], t["geo_sid"], t["meal_period"], None, None, None, None)
            for t in tokens]


def positives_grouped(label_tokens: list) -> dict:
    """label 区按 (时段, 行为) 分组成正样本 geo_sid 集合（去重，保序）：
       {"{meal_period}_{action}": [geo_sid, ...]}。替代旧版 positives_by_action /
       positives_by_action_period 两个字段。以后请求经纬度字段到位、加上
       geohash_rank 维度时，组名延伸成 "{period}_{action}_{geohash_rank}"，
       是纯新增不是破坏性改动。"""
    g = {}
    for t in label_tokens:
        key = f"{t['meal_period']}_{t['action']}"
        lst = g.setdefault(key, [])
        if t["geo_sid"] not in lst:
            lst.append(t["geo_sid"])
    return g


def _make_eval_sample(uid, split, input_tokens, label_date, label_tokens,
                      favor_coord_raw) -> dict:
    """val/test 同构样本（每用户各一条）。label_tokens 为 label session 原始有序
       交互（val 走 teacher-forcing 算 loss）；positives_grouped 为按 (时段,行为)
       去重的正样本集合（test 算 HR/Recall/NDCG）；favor_coord_raw 为用户收藏坐标
       原始字符串（geohash_1/2/3 及各自经纬度由下游按需解析，样本这一层只搬运）。"""
    return {
        "uid": uid,
        "split": split,
        "input": input_tokens,
        "label_tokens": label_tokens,
        "positives_grouped": positives_grouped(label_tokens),
        "favor_coord_raw": favor_coord_raw,
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


def build_train_sequences(uid, train_sessions: list, x: int, min_seq_len: int) -> list:
    """train：传入的 train 区域 sessions（三分法下 = S1..S(m-2)）原序拼接成一条
       user-level 全序列（保留重复交互），外加 x 个 behavior-drop 变体
       （ri = i/(x+1)）。每用户最多输出 x+1 条：
       - 区域交互数 < min_seq_len 时不出任何 train 序列（val/test 不受影响）；
       - 变体与原始相同（什么都没丢到）时跳过。
       rng 以 (uid, i) 做种子，保证可复现。"""
    train_tokens = [t for _, toks in train_sessions for t in toks]
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
                      min_train_seq_len: int, favor_coord_raw=None) -> list:
    """session 粒度留一法三分（对齐论文 session-wise leave-one-out）：
       m>=3: train = S1..S(m-2) 全序列 + 去重后的 drop 变体（NTP，无 label 区）
             val   : input=S1..S(m-2), label=S(m-1)
             test  : input=S1..S(m-1), label=Sm
       m==2: 只出 val（input=S1, label=S2）——S2 给了 val（best.pt 按 val loss
             挑选）就不再当 test label，避免选择性泄漏。
       m<2 : 整体剔除。（sessions 0-indexed: S1=sessions[0] ... Sm=sessions[m-1]）
       favor_coord_raw 只挂在 val/test 评测样本上（train 不需要）。"""
    m = len(sessions)
    if m < 2:
        return []  # 留不出任何 label session 的用户整体剔除
    if m == 2:
        return [_make_eval_sample(uid, "val", list(sessions[0][1]),
                                  sessions[1][0], sessions[1][1], favor_coord_raw)]

    out = []
    val_input = [t for _, toks in sessions[:m - 2] for t in toks]
    out.append(_make_eval_sample(uid, "val", val_input,
                                 sessions[m - 2][0], sessions[m - 2][1],
                                 favor_coord_raw))
    test_input = [t for _, toks in sessions[:m - 1] for t in toks]
    out.append(_make_eval_sample(uid, "test", test_input,
                                 sessions[m - 1][0], sessions[m - 1][1],
                                 favor_coord_raw))
    # train（user-level 全序列 + 增强，只用 S1..S(m-2)）
    out += build_train_sequences(uid, sessions[:m - 2], behavior_drop_x,
                                 min_train_seq_len)
    return out


def build_incremental_samples(uid: str, sessions: list, behavior_drop_x: int,
                              min_train_seq_len: int, train_start: str, train_end: str,
                              favor_coord_raw=None) -> list:
    """增量模式（[data] is_auto=1）：不做留一法三分，只看这一条——用户最后一次
       交互的本地日期是否落在 [train_start, train_end] 窗口内：
         - 不在窗口内：这一轮跳过该用户，返回 []（历史更早、这轮没有新动作的
           用户不重复训练）；
         - 在窗口内：用户全部历史 S1..Sm（不切分、不留 val/test）整条送去当
           train 序列构造（复用 build_train_sequences，同样做 behavior-drop
           增强），因为这个用户"有新东西"，让模型完整地重新学一遍它的序列。
       train_start/train_end 是 YYYYMMDD 字符串，跟 sessions 里的本地日期
       （YYYY-MM-DD）做字符串比较前先去掉连字符对齐格式。"""
    if not sessions:
        return []
    last_date = sessions[-1][1][-1]["date"]          # 最后一个 session 最后一条交互的本地日期
    if not last_date:
        return []
    last_dt8 = last_date.replace("-", "")             # "YYYY-MM-DD" -> "YYYYMMDD"
    if not (train_start <= last_dt8 <= train_end):
        return []
    return build_train_sequences(uid, sessions, behavior_drop_x, min_train_seq_len)


def iter_user_samples(cfg: dict, conf_path: str, part_filter=None,
                      id2sid: dict = None, verbose: bool = True):
    """流式产出 (uid, sessions, samples)：step1 取数 -> step2 映射 -> 按天切 session ->
       build_all_samples（[data] is_auto=0，默认，留一法三分 train/val/test）
       或 build_incremental_samples（is_auto=1，增量：只挑最后一次交互落在
       train_start~train_end 的用户，全部历史整条进 train，不出 val/test）。
       max_num 早停在此处理。
       本文件 main()（诊断/落盘）与 train/dataset/stream_dataset（流式训练）共用这一份逻辑。
       part_filter 透传给 stream_rows 做多 worker 的 part 文件分片。"""
    if id2sid is None:
        id2sid = load_item_sid_map(get_item_map_path(conf_path))
    seq_fields = cfg["seq_fields"]
    tz_offset = cfg["tz_offset_hours"]
    favor_coord_field = cfg.get("favor_coord_field", "user_favor_coor_top3")
    is_auto = cfg.get("is_auto", 0)
    unlimited = cfg["max_num"] == -1

    # is_auto=1 时，取数只读 train_end 这一天的快照，不要按 stream_rows 原有
    # 逻辑把 [train_start, train_end] 逐天全部读一遍——底层表每天都是"截至
    # 当天的全量累积快照"，train_end 这天天然已经包含每个用户的完整历史
    # （含他们最后一次交互是哪天），逐天重复读只会让同一个用户被扫到
    # (train_end-train_start+1) 次、生成一模一样的训练序列，纯浪费。
    # 过滤条件（判断"最后一次交互是否落在 train_start~train_end 区间"）
    # 仍然用 cfg 里原本配置的完整区间，只是取数窗口收窄，语义不变。
    stream_cfg = cfg
    if is_auto:
        stream_cfg = dict(cfg)
        stream_cfg["train_start"] = cfg["train_end"]

    n = 0
    for _dt, _hdfs, _schema, row in stream_rows(stream_cfg, part_filter=part_filter,
                                                verbose=verbose):
        mapped = map_sample(row, id2sid, seq_fields)
        timeline = build_timeline(mapped, seq_fields, tz_offset)
        sessions = sessionize_by_day(timeline)
        uid = row.get("uid")
        if is_auto:
            samples = build_incremental_samples(
                uid, sessions, cfg["behavior_drop_x"], cfg["min_train_seq_len"],
                cfg["train_start"], cfg["train_end"], row.get(favor_coord_field))
        else:
            samples = build_all_samples(
                uid, sessions, cfg["behavior_drop_x"], cfg["min_train_seq_len"],
                row.get(favor_coord_field))
        yield uid, sessions, samples
        n += 1
        if not unlimited and n >= cfg["max_num"]:
            break
        if verbose and unlimited and n % 100000 == 0:
            print(f"[INFO] 已处理 {n} 行")


def print_user_samples(uid: str, sessions: list, samples: list):
    m = len(sessions)
    print(f"\n========== 用户 uid={uid}  共 {m} 个 session ==========")
    for i, (date, toks) in enumerate(sessions):
        if m == 2:
            role = "[val-label]" if i == 1 else "[val-input]"
        elif i == m - 1:
            role = "[test-label]"
        elif i == m - 2:
            role = "[val-label]"
        else:
            role = "[train区域]"
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
            print(f"    [{s['split']}] input_len={len(inp)} tail={tail} "
                  f"-> label区@{s['label_date']} {len(s['label_tokens'])} 交互(有序): "
                  f"{[_fmt_tok(t) for t in s['label_tokens']]}")
            print(f"          positives_grouped: "
                  f"{ {g: len(v) for g, v in s['positives_grouped'].items()} }"
                  f"  favor_coord_raw={s['favor_coord_raw']!r}")
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
          f"留一法三分: train=S1..S(m-2) val=S(m-1) test=Sm"
          f"（m=2 只出 val；m<2 剔除）")
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
                   for sp in ("train", "val", "test")}
        print(f"[INFO] 样本落盘: {out_dir}/{{train,val,test}}.jsonl")
    else:
        print("[INFO] samples_out_dir 未配置，仅打印诊断不落盘")
    print(f"[INFO] 窗口: {cfg['train_start']} ~ {cfg['train_end']}，期望处理 {max_desc} 行\n")

    n_users = 0
    n_users_kept = 0                  # m>=2 被保留的用户
    session_hist = Counter()          # 每用户 session 数分布
    split_samples = Counter()         # 各 split 样本数
    total_pos = Counter()             # val label 区交互数累计，估平均 label 区长度
    train_seq_len = Counter()         # {aug_r: 累计 token 数}，估平均序列长度
    train_seq_cnt = Counter()         # {aug_r: 序列条数}
    printed = 0

    for uid, sessions, samples in iter_user_samples(cfg, conf_path, id2sid=id2sid):
        n_users += 1
        session_hist[min(len(sessions), 5)] += 1
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
                       "label_tokens": _slim_label_tokens(s["label_tokens"]),
                       "positives_grouped": s["positives_grouped"],
                       "favor_coord_raw": s["favor_coord_raw"],
                       "label_date": s["label_date"]}
            if writers:
                writers[s["split"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
        if samples and printed < cfg["log_sample_count"]:
            print_user_samples(uid, sessions, samples)
            printed += 1

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
    print("val/test（每用户各一条，label_tokens 为原始有序交互）:")
    for sp in ("val", "test"):
        cnt = split_samples.get(sp, 0)
        avg = (total_pos.get(sp, 0) / cnt) if cnt else 0.0
        print(f"  {sp:<5}: {cnt} 条  (平均 label 区交互数 {avg:.2f})")
    if writers:
        for sp, w in writers.items():
            w.close()
        print(f"已落盘: {out_dir}/{{train,val,test}}.jsonl")
    print("=====================================")


if __name__ == "__main__":
    main()
