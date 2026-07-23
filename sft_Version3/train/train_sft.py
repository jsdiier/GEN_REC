#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_sft: GAMER 从零训练循环（对齐论文 4.1.4 的训练配置）。

流程：
  1. 读 common.conf [train]；加载 vocab.json 建 SIDTokenizer；
  2. 数据接入按 data_mode 二选一：
       jsonl : GAMERJsonlDataset 读 step3 落盘文件（小规模调试）
       stream: GAMERStreamingTrainDataset 在 DataLoader worker 内直连 HDFS
               流式生成（TB 级不落盘）；val 启动时抽样收集一次常驻内存；
               lr 调度总步数 epoch 1 结束后按实测步数自动校准（见 build_scheduler）；
  3. build_model() 按 [train] model_type 二选一（配置由 tokenizer 推导：vocab_size /
     tokens_per_item / behavior_levels 等）：
       gamer : 论文对齐的手写 8 层 GAMERModel，从零训练；
       qwen3 : 原生 Qwen3 结构（transformers 库）做 backbone，从 qwen3_path 的预训练
               权重初始化，只替换词表（modeling_qwen3.py）；
  4. AdamW + 线性 warmup(4%) + 余弦衰减到 min_lr；
  5. 逐 epoch：train 全 token 监督 NTP；val（teacher-forcing，只算 label 区，按有效
     token 数加权聚合）在每个 epoch 末必跑，eval_every>0 时每隔该步数再加跑一次
     （全量数据单 epoch 很长，epoch 末一次太稀）；val loss 创新低即存 best.pt，
     每个 epoch 末更新 last.pt；
  6. patience>0 时按 val loss 早停；否则训满 epochs（论文做法，最终用 best.pt）。

说明：
  - GAMERModel 仅 ~40M，单卡即可；qwen3 model_type 参数量取决于 qwen3_path 的权重
    （官方 Qwen3-0.6B 是 28 层）；CUDA 下自动用 bf16 autocast；
  - checkpoint 含 model state_dict + config 字段 + model_type + epoch/val_loss，
    推理端用 load_checkpoint() 按 model_type 还原对应结构；
  - [data] is_auto=1（增量训练，目前只支持 data_mode=stream）：只加载
    output_dir/latest/best.pt 的权重（optimizer/epoch/lr 调度全新开始），
    没有 val split（内部另抽一小撮当监控集选 best.pt，见
    collect_incremental_monitor_samples）；训完才把 latest 改指向本次结果
    （见 update_latest_and_prune），并按 max_ckpt_keep 轮换旧 checkpoint。

用法:
    python train_sft.py [common.conf]
"""

import math
import os
import sys
import time
import configparser
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer_sid import SIDTokenizer            # noqa: E402
from modeling import GAMERConfig, GAMERModel      # noqa: E402
from modeling_qwen3 import Qwen3NTPConfig, Qwen3NTPModel  # noqa: E402
from dataset import GAMERJsonlDataset, GAMERCollator  # noqa: E402


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------
def load_train_config(conf_path: str) -> dict:
    cp = configparser.ConfigParser()
    if not cp.read(conf_path, encoding="utf-8"):
        raise FileNotFoundError(f"找不到配置文件: {conf_path}")
    root = os.path.dirname(os.path.abspath(conf_path))

    def path(key, default):
        p = cp.get("train", key, fallback=default)
        return p if os.path.isabs(p) else os.path.join(root, p)

    return {
        "model_type": cp.get("train", "model_type", fallback="gamer").strip(),
        # 支持相对路径（相对 common.conf 所在目录，即每次训练的启动根目录）；
        # 留空（gamer 场景）不做拼接，保持空字符串
        "qwen3_path": (lambda v: "" if not v else path("qwen3_path", v))(
            cp.get("train", "qwen3_path", fallback="").strip()),
        "data_mode": cp.get("train", "data_mode", fallback="jsonl"),
        "shuffle_buffer": cp.getint("train", "shuffle_buffer", fallback=10000),
        "val_sample_rate": cp.getfloat("train", "val_sample_rate", fallback=0.05),
        "max_val_users": cp.getint("train", "max_val_users", fallback=20000),
        "samples_dir": path("samples_dir", "outputs/samples"),
        "vocab_path": path("vocab_path", "outputs/vocab.json"),
        "output_dir": path("output_dir", "outputs/ckpt"),
        "epochs": cp.getint("train", "epochs", fallback=200),
        "batch_size": cp.getint("train", "batch_size", fallback=256),
        "grad_accum": cp.getint("train", "grad_accum", fallback=1),
        "lr": cp.getfloat("train", "lr", fallback=5e-4),
        "min_lr": cp.getfloat("train", "min_lr", fallback=1e-6),
        "warmup_ratio": cp.getfloat("train", "warmup_ratio", fallback=0.04),
        "weight_decay": cp.getfloat("train", "weight_decay", fallback=0.01),
        "max_len": cp.getint("train", "max_len", fallback=512),
        "patience": cp.getint("train", "patience", fallback=0),
        "seed": cp.getint("train", "seed", fallback=42),
        "num_workers": cp.getint("train", "num_workers", fallback=2),
        "log_every": cp.getint("train", "log_every", fallback=50),
        "eval_every": cp.getint("train", "eval_every", fallback=0),
        "wandb_project": cp.get("train", "wandb_project", fallback="").strip(),
        "wandb_run_name": cp.get("train", "wandb_run_name", fallback="").strip(),
        "wandb_api_key": cp.get("train", "wandb_api_key", fallback="").strip(),
        "wandb_init_retries": cp.getint("train", "wandb_init_retries", fallback=3),
        "resume_from": (lambda v: "" if not v else path("resume_from", v))(
            cp.get("train", "resume_from", fallback="").strip()),
        # [data] is_auto=1（增量训练）：只加载 output_dir/latest/best.pt 的权重
        # 当初始化（optimizer/epoch 全新开始），训完按"最多保留 3 个 ckpt"轮换、
        # 全部完成才把 latest 改指向新结果——跟 resume_from（同一个 run 目录里
        # 接着上次没走完的 epoch/optimizer 状态续训）是两回事，不能同时配
        "is_auto": cp.getint("data", "is_auto", fallback=0),
        # 只用来给 checkpoint 目录命名加个数据覆盖日期前缀（见 run_stamp 拼接处），
        # 不影响取数逻辑本身（取数窗口仍由 data/step1_get_user_action.load_config
        # 自己读 [data] train_start/train_end）
        "train_end": cp.get("data", "train_end", fallback=""),
        "max_ckpt_keep": cp.getint("train", "max_ckpt_keep", fallback=3),
    }


def init_wandb(tc: dict, model_cfg) -> "object":
    """wandb_project 配置了才开启；wandb 未安装或未登录时降级为警告，不阻塞训练。
       API key 优先读 conf 的 wandb_api_key（本地/平台容器通用），
       留空则回退 WANDB_API_KEY 环境变量 / ~/.netrc（wandb login）。
       平台节点出网情况不稳定，wandb.init() 偶发 ConnectTimeout；失败按
       wandb_init_retries 重试几次（间隔 5s），重试次数用完才放弃监控。"""
    if not tc["wandb_project"]:
        return None
    if tc["wandb_api_key"]:
        os.environ["WANDB_API_KEY"] = tc["wandb_api_key"]
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb 未安装（pip install wandb），跳过监控")
        return None
    retries = max(tc["wandb_init_retries"], 1)
    for attempt in range(1, retries + 1):
        try:
            run = wandb.init(
                project=tc["wandb_project"],
                name=tc["wandb_run_name"] or
                f"gamer-{tc['data_mode']}-{time.strftime('%m%d-%H%M')}",
                config={**{k: v for k, v in tc.items() if not k.startswith("wandb")},
                        "model": asdict(model_cfg)},
            )
            print(f"[INFO] wandb 已开启: {run.url}")
            return run
        except Exception as e:                      # 未登录 / 网络不通等
            print(f"[WARN] wandb 初始化第 {attempt}/{retries} 次失败: {e}")
            if attempt < retries:
                time.sleep(5)
    print(f"[WARN] wandb 初始化重试 {retries} 次均失败，跳过监控")
    return None


# stream 模式 epoch 1 校准前的临时 warmup 步数（升到峰值 lr 后平顶等待校准）
PROVISIONAL_WARMUP = 100


def build_scheduler(optimizer, epochs: int, warmup_ratio: float,
                    lr: float, min_lr: float, total_steps: int = None):
    """线性 warmup + 余弦衰减到 min_lr（论文 4.1.4）。

    总步数的两种来源：
      - jsonl 模式：启动就知道数据量，传 total_steps 直接精确调度；
      - stream 模式：启动时不知道每 epoch 步数（取决于 max_num/增强/过滤），
        epoch 1 先用临时调度（前 PROVISIONAL_WARMUP 步线性升温到峰值后平顶），
        epoch 1 结束调 scheduler.calibrate(实测 steps/epoch) 后按精确总步数走。
        校准后若正式 warmup（warmup_ratio*total）尚未走完（epochs > 1/ratio 时），
        lr 会落回修正后的 warmup 斜率继续升温，属正常衔接。
    """
    state = {"total": None, "warmup": None}

    def lr_lambda(step):
        if state["total"] is None:                     # 校准前：升温 + 平顶
            return min((step + 1) / PROVISIONAL_WARMUP, 1.0)
        if step < state["warmup"]:
            return (step + 1) / state["warmup"]
        progress = (step - state["warmup"]) / max(state["total"] - state["warmup"], 1)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return (min_lr + (lr - min_lr) * cos) / lr

    def calibrate(steps_per_epoch: int):
        state["total"] = steps_per_epoch * epochs
        state["warmup"] = max(int(state["total"] * warmup_ratio), 1)

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    sched.calibrate = calibrate
    sched.calibrated = lambda: state["total"] is not None
    sched.export_state = lambda: dict(state)       # 断点续训：校准状态随 last.pt 存取
    sched.import_state = state.update
    if total_steps is not None:
        state["total"] = total_steps
        state["warmup"] = max(int(total_steps * warmup_ratio), 1)
    return sched


# ------------------------------------------------------------------
# 评估：val loss（teacher-forcing，按有效 token 数加权）+ 按 token 位分解
# ------------------------------------------------------------------
def build_slot_defs(tok, device) -> dict:
    """token 位定义：{位名: (token_types 取值, 该位合法 token id 张量)}。
       behavior=行为位；sid1..sidl=SID 各层（sid1=geo, sid2=a, ...）；
       period=时段位（词表含时段时才有，type=num_levels+1）。"""
    defs = {"behavior": (0, torch.tensor(
        [tok.token2id[f"<{b}>"] for b in tok.behaviors], device=device))}
    for j, ids in enumerate(tok.level_token_ids, start=1):
        defs[f"sid{j}"] = (j, torch.tensor(ids, device=device))
    if getattr(tok, "periods", []):
        defs["period"] = (tok.period_type, torch.tensor(
            [tok.token2id[f"<{p}>"] for p in tok.periods], device=device))
    return defs


@torch.no_grad()
def evaluate(model, val_dl, device, autocast_ctx, slot_defs=None):
    """返回 (总 val loss, {位名: (loss, class_mass)})。
       - 分位 loss：只有该位的语义预测能力，剥离"结构学习"的水分
         （sid2/3/4 贴着 ln(类内大小)≈5.55 不动 = 没学到 item 语义，只在学结构）；
       - class_mass：模型放在该位【正确类别】上的概率质量，->1 表示结构已学完。"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    slot_stats = {n: [0.0, 0.0, 0] for n in (slot_defs or {})}   # [loss和, mass和, 数量]
    for batch in val_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"][:, 1:]                          # 与模型内部 shift 对齐
        valid = labels != -100
        if not valid.any():
            continue
        with autocast_ctx():
            _, logits = model(**batch)
        logits = logits[:, :-1].float()                          # (B, L-1, V)
        ce = F.cross_entropy(logits.flatten(0, 1), labels.clamp(min=0).flatten(),
                             reduction="none").view_as(labels)
        total_loss += ce[valid].sum().item()
        total_tokens += valid.sum().item()

        if slot_defs:
            types = batch["token_types"][:, 1:]                  # 目标 token 的位类型
            for name, (tval, ids) in slot_defs.items():
                m = valid & (types == tval)
                if not m.any():
                    continue
                st = slot_stats[name]
                st[0] += ce[m].sum().item()
                lg = logits[m]                                   # (n, V)
                mass = (lg[:, ids].logsumexp(-1) - lg.logsumexp(-1)).exp()
                st[1] += mass.sum().item()
                st[2] += m.sum().item()
    model.train()
    per_slot = {n: (s[0] / s[2], s[1] / s[2]) for n, s in slot_stats.items() if s[2]}
    return total_loss / max(total_tokens, 1), per_slot


def build_model(model_type: str, tok, tc: dict):
    """按 [train] model_type 二选一构造模型：
       gamer  -> 论文对齐的手写 8 层 GAMERModel，从零训练；
       qwen3  -> 原生 Qwen3 结构（transformers 库）做 backbone，从 qwen3_path 的
                 预训练权重初始化，只替换词表（resize 到 SID 小词表）。"""
    if model_type == "qwen3":
        cfg = Qwen3NTPConfig(vocab_size=tok.vocab_size,
                             qwen3_path=tc["qwen3_path"],
                             pad_token_id=tok.pad_id,
                             max_position_embeddings=tc["max_len"],
                             tokens_per_item=tok.num_levels,
                             num_behaviors=len(tok.behaviors),
                             num_periods=len(getattr(tok, "periods", [])),
                             behavior_levels=tok.behavior_levels)
        model = Qwen3NTPModel(cfg)
    elif model_type == "gamer":
        cfg = GAMERConfig(vocab_size=tok.vocab_size,
                          tokens_per_item=tok.num_levels,
                          num_behaviors=len(tok.behaviors),
                          num_periods=len(getattr(tok, "periods", [])),
                          behavior_levels=tok.behavior_levels,
                          max_position_embeddings=tc["max_len"],
                          pad_token_id=tok.pad_id)
        model = GAMERModel(cfg)
    else:
        raise ValueError(f"未知 model_type: {model_type}（支持 gamer / qwen3）")
    return model, cfg


def save_checkpoint(path: str, model, cfg, model_type: str, epoch: int, val_loss: float,
                    extra: dict = None):
    """best.pt 只存权重（推理用，体积小）；last.pt 由调用方传 extra
       （optimizer/scheduler/global_step 等）支持断点续训。
       model_type 随 checkpoint 落盘，load_checkpoint 据此还原对应模型结构，
       不依赖加载时 common.conf 恰好是哪个配置。
       qwen3 额外存一份 hf_config（架构 dict，几个数字，不是权重）：eval/inference
       重建架构骨架时直接从这份 dict 建，不用回头读 qwen3_path 的 config.json——
       qwen3_path 是相对路径，训练（platform，NFS 挂载）和 eval/inference（本地
       home 挂载）解析出来的绝对路径不是同一个物理位置，checkpoint 一旦跨语境
       加载，按 qwen3_path 重新读文件就会读到不存在的路径。"""
    d = {
        "model": model.state_dict(),
        "config": asdict(cfg),
        "model_type": model_type,
        "epoch": epoch,
        "val_loss": val_loss,
    }
    if model_type == "qwen3":
        d["hf_config"] = model.hf_config.to_dict()
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_checkpoint(path: str, map_location="cpu", qwen3_path: str = None):
    """推理端还原：返回 (model, config, meta)。
       qwen3_path：仅在 checkpoint 没存 hf_config（这个字段上线前存的旧 checkpoint）
       时才用得上——传了就覆盖 checkpoint 里存的那份（训练时按平台 NFS 语境解析出
       来的，换个语境大概率是错的路径），改用调用方当前环境自己解析出的路径。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model_type = ckpt.get("model_type", "gamer")   # 旧 checkpoint 没有这个字段，默认 gamer
    cfg_d = dict(ckpt["config"])
    if model_type == "qwen3":
        hf_config_dict = ckpt.get("hf_config")
        if hf_config_dict is None and qwen3_path:
            cfg_d["qwen3_path"] = qwen3_path
        cfg = Qwen3NTPConfig(**cfg_d)
        # load_pretrained=False：只建架构、随机初始化，马上被下面的 load_state_dict
        # 整体覆盖，加载真实预训练权重纯属浪费 I/O。hf_config 优先从 checkpoint 里
        # 存的那份 dict 建架构（不碰 qwen3_path，跨语境也不会读错路径）；没有的话
        # 才退回按 cfg.qwen3_path 读 config.json（此时才要求这个路径当前可达）。
        model = Qwen3NTPModel(cfg, load_pretrained=False, hf_config_dict=hf_config_dict)
    else:
        cfg_d["behavior_levels"] = tuple(cfg_d["behavior_levels"])
        cfg = GAMERConfig(**cfg_d)
        model = GAMERModel(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg, {"epoch": ckpt["epoch"], "val_loss": ckpt["val_loss"]}


def update_latest_and_prune(output_dir: str, run_dir: str, run_stamp: str,
                            max_keep: int) -> None:
    """训练循环正常跑完之后才调用（不是训练开始时）：
       1. 确认 run_dir 下确实有 best.pt（真的训出了可用于推理的权重）才动 latest，
          没有就只告警、不改 latest——保留原来那个仍然可用的 checkpoint；
       2. latest 软链原子性地改指向 run_stamp（先建临时软链再 os.replace 过去，
          不是先删再建——"先删再建"中间会有一个 latest 完全不存在的窗口，
          推理进程如果恰好在那个窗口读 latest/best.pt 会撞上瞬时
          FileNotFoundError；os.replace 底层是 POSIX rename，全程 latest
          要么是旧值要么已经是新值，不存在中间态）；
       3. 轮换：output_dir 下所有含 best.pt 的子目录（不分是全量训练还是增量
          训练产出的，一视同仁）按目录 mtime 排序，只留最新 max_keep 个，
          其余整目录删除。没有 best.pt 的目录（比如某次训练崩了留下的半成品）
          不参与排序和删除，原样留着，避免误删还没查清楚状况的东西。
          （删除旧目录本身也是安全的：已经打开在读旧 checkpoint 的进程手上
          持有的文件描述符不受目录项删除影响，读得到完整数据，这是
          Linux/NFS 的标准语义，不需要额外处理。）"""
    best_path = os.path.join(run_dir, "best.pt")
    if not os.path.exists(best_path):
        print(f"[WARN] {run_dir} 下没有 best.pt，不更新 latest（保留原有 checkpoint 可用）")
        return

    latest = os.path.join(output_dir, "latest")
    if os.path.exists(latest) and not os.path.islink(latest):
        print(f"[WARN] {latest} 已存在且不是软链（是个真目录），跳过更新")
        return
    tmp_link = f"{latest}.tmp{os.getpid()}"
    if os.path.lexists(tmp_link):
        os.remove(tmp_link)
    os.symlink(run_stamp, tmp_link)
    os.replace(tmp_link, latest)          # POSIX rename，原子替换，无中间态
    print(f"[INFO] 训练完成，latest -> {run_stamp}")

    import shutil
    candidates = []
    for name in os.listdir(output_dir):
        p = os.path.join(output_dir, name)
        if name == "latest" or not os.path.isdir(p) or os.path.islink(p):
            continue
        if os.path.exists(os.path.join(p, "best.pt")):
            candidates.append((os.path.getmtime(p), name, p))
    candidates.sort(key=lambda x: x[0], reverse=True)
    for _mtime, name, p in candidates[max_keep:]:
        shutil.rmtree(p)
        print(f"[INFO] 超过保留上限 {max_keep} 个，已删除旧 checkpoint: {name}")


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "common.conf")
    tc = load_train_config(conf_path)
    torch.manual_seed(tc["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    if use_bf16:
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext
    print(f"[INFO] device={device}  bf16={use_bf16}")

    # tokenizer / 数据
    tok = SIDTokenizer.load(tc["vocab_path"])
    print(f"[INFO] vocab_size={tok.vocab_size}  num_levels={tok.num_levels}  "
          f"behaviors={tok.behaviors}")
    coll = GAMERCollator(pad_id=tok.pad_id)
    if tc["is_auto"] and tc["data_mode"] != "stream":
        raise ValueError("[data] is_auto=1（增量训练）目前只支持 [train] data_mode=stream")
    if tc["is_auto"]:
        # 训练正式开始前先完整流一遍窗口数据，数出这一轮增量实际覆盖多少
        # 用户/多少条 train 序列——多花一次跟训练单个 epoch 同量级的扫描时间，
        # 换来"开跑前就知道这轮增量有多大"，方便发现配置错误（比如窗口选空了）
        from dataset import count_incremental_samples
        print("[INFO] 增量训练：训练前先统计本轮样本数 ...")
        count_incremental_samples(conf_path)
    if tc["data_mode"] == "stream":
        # 流式：train 直连 HDFS（IterableDataset，worker 按 part 分片）。
        # is_auto=0：val 启动时抽样收集一次，常驻内存；
        # is_auto=1（增量）：没有 val split，改用 collect_incremental_monitor_samples
        # 从"本轮该训练的用户"里划一小撮只当监控集（不参与训练，train_ds 传
        # 同样的 sample_rate 当 monitor_sample_rate 用于跳过这批 uid）
        from dataset import (GAMERStreamingTrainDataset, collect_val_samples,
                             collect_incremental_monitor_samples)
        train_ds = GAMERStreamingTrainDataset(
            conf_path, tok, max_len=tc["max_len"], shuffle_buffer=tc["shuffle_buffer"],
            seed=tc["seed"],
            monitor_sample_rate=tc["val_sample_rate"] if tc["is_auto"] else 0.0)
        if tc["is_auto"]:
            monitor_samples = collect_incremental_monitor_samples(
                conf_path, tc["val_sample_rate"], tc["max_val_users"])
            val_ds = GAMERJsonlDataset(None, tok, "train", max_len=tc["max_len"],
                                       samples=monitor_samples)
            print(f"[INFO] data_mode=stream  is_auto=1（增量）  "
                  f"监控集={len(val_ds)} 条（内存，不参与训练）  train 条数未知（流式）")
        else:
            val_ds = GAMERJsonlDataset(None, tok, "val", max_len=tc["max_len"],
                                       samples=collect_val_samples(
                                           conf_path, tc["val_sample_rate"],
                                           tc["max_val_users"]))
            print(f"[INFO] data_mode=stream  val={len(val_ds)} 条（内存）  "
                  f"train 条数未知（流式）")
        train_dl = DataLoader(train_ds, batch_size=tc["batch_size"],
                              collate_fn=coll, num_workers=tc["num_workers"],
                              pin_memory=(device == "cuda"))
        steps_per_epoch = None                  # epoch 1 实测后校准 lr 调度
    else:
        train_ds = GAMERJsonlDataset(os.path.join(tc["samples_dir"], "train.jsonl"),
                                     tok, "train", max_len=tc["max_len"])
        val_ds = GAMERJsonlDataset(os.path.join(tc["samples_dir"], "val.jsonl"),
                                   tok, "val", max_len=tc["max_len"])
        print(f"[INFO] data_mode=jsonl  train={len(train_ds)} 条序列  val={len(val_ds)} 条")
        train_dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                              collate_fn=coll, num_workers=tc["num_workers"],
                              pin_memory=(device == "cuda"), drop_last=False)
        steps_per_epoch = math.ceil(len(train_dl) / tc["grad_accum"])
    val_dl = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                        collate_fn=coll, num_workers=0,
                        pin_memory=(device == "cuda"))

    # 模型：[train] model_type 二选一（gamer=从零训练手写模型 / qwen3=Qwen3 预训练 backbone）
    model, cfg = build_model(tc["model_type"], tok, tc)
    model = model.to(device)
    print(f"[INFO] model_type={tc['model_type']}  模型参数量: {model.num_parameters() / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"],
                                  weight_decay=tc["weight_decay"])
    total_steps = steps_per_epoch * tc["epochs"] if steps_per_epoch else None
    scheduler = build_scheduler(optimizer, tc["epochs"], tc["warmup_ratio"],
                                tc["lr"], tc["min_lr"], total_steps=total_steps)
    steps_desc = (f"steps/epoch={steps_per_epoch}  总步数={total_steps}"
                  if steps_per_epoch else
                  "steps/epoch=未知（epoch 1 实测后校准 lr 调度）")
    print(f"[INFO] epochs={tc['epochs']}  {steps_desc}  "
          f"有效batch={tc['batch_size'] * tc['grad_accum']}")

    # ---- 断点续训（[train] resume_from = run 目录或 last.pt 路径；留空 = 全新训练）----
    start_epoch, resume_dir = 1, None
    best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    global_step = 0
    if tc["is_auto"] and tc["resume_from"]:
        raise ValueError("[data] is_auto=1（增量训练）与 [train] resume_from 不能同时配置："
                         "增量训练总是基于 latest 的权重重新起一轮全新训练（optimizer/epoch"
                         "全新开始），resume_from 是接着某次没训完的 run 继续，两者语义冲突")
    if tc["is_auto"]:
        base_ckpt = os.path.join(tc["output_dir"], "latest", "best.pt")
        if not os.path.exists(base_ckpt):
            raise FileNotFoundError(f"增量训练找不到基础权重: {base_ckpt}——"
                                    f"outputs/ckpt/latest 得先指向一个含 best.pt 的完整 "
                                    f"checkpoint 目录（全量训练产出的，或上一轮增量训练"
                                    f"产出的）")
        ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
        ck_model_type = ck.get("model_type", "gamer")
        if ck_model_type != tc["model_type"]:
            raise ValueError(f"增量训练基础 checkpoint 的 model_type={ck_model_type} 与"
                             f"当前 common.conf 的 model_type={tc['model_type']} 不一致")
        model.load_state_dict(ck["model"])
        print(f"[INFO] 增量训练：从 {base_ckpt} 加载权重（epoch={ck['epoch']} "
              f"val_loss={ck['val_loss']:.4f}），optimizer/epoch/lr 调度全新开始")
    if tc["resume_from"]:
        ckpt_file = tc["resume_from"]
        if not ckpt_file.endswith(".pt"):
            ckpt_file = os.path.join(ckpt_file, "last.pt")
        ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        ck_model_type = ck.get("model_type", "gamer")
        if ck_model_type != tc["model_type"]:
            raise ValueError(f"续训 checkpoint 的 model_type={ck_model_type} 与当前 "
                             f"common.conf 的 model_type={tc['model_type']} 不一致")
        model.load_state_dict(ck["model"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", ck["val_loss"])
        best_epoch = ck.get("best_epoch", ck["epoch"])
        if "optimizer" in ck:                      # 新格式：optimizer/调度器完整恢复
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.last_epoch = ck["sched_last_epoch"]   # LambdaLR 的位置即全部状态
            scheduler.import_state(ck["sched_calib"])
            global_step = ck["global_step"]
        else:                                      # 旧格式（只有权重）：动量冷启动，
            print("[WARN] checkpoint 无 optimizer/调度器状态（旧格式）：动量冷启动；"
                  "lr 在首个续训 epoch 走峰值平顶，epoch 末校准并快进到真实进度")
        resume_dir = os.path.dirname(os.path.abspath(ckpt_file))
        print(f"[INFO] 断点续训: {ckpt_file}  从 epoch {start_epoch}/{tc['epochs']} 继续"
              f"（已有 best_val={best_val:.4f} @ epoch {best_epoch}）")
        if start_epoch > tc["epochs"]:
            print(f"[WARN] 已训满 epochs={tc['epochs']}，无事可做")
            return

    # checkpoint 目录：全新训练建「启动时间」子目录，续训沿用原目录。
    # 注意：latest 软链不在这里更新——训练途中 latest 必须继续指向上一个【已
    # 训完、可用于推理】的 checkpoint，不能指向正在写的这个半成品目录（不然
    # 推理/下一轮增量训练读 latest 会读到还没训完甚至崩溃后残缺的权重）。
    # latest 的更新挪到 main() 末尾训练真正跑完之后，见 update_latest_and_prune。
    if resume_dir:
        run_dir = resume_dir
        run_stamp = os.path.basename(run_dir.rstrip("/"))
    else:
        # {train_end}_{启动日期}_{启动时间}：train_end 是这次训练数据覆盖到
        # 哪天，一眼就能看出这个 checkpoint 训的是哪个窗口的数据，不用再去
        # 翻 common.conf 或猜测；train_end 留空（理论上不该发生，容错）则退回
        # 纯时间戳
        start_date, start_time = time.strftime("%Y%m%d"), time.strftime("%H%M%S")
        run_stamp = (f"{tc['train_end']}_{start_date}_{start_time}" if tc["train_end"]
                    else f"{start_date}_{start_time}")
        run_dir = os.path.join(tc["output_dir"], run_stamp)
        os.makedirs(run_dir, exist_ok=True)
    if not tc["wandb_run_name"]:                   # wandb run 名与目录共用时间戳，便于对应
        suffix = f"-r{start_epoch}" if resume_dir else ""
        tc["wandb_run_name"] = f"gamer-{tc['data_mode']}-{run_stamp}{suffix}"
    print(f"[INFO] checkpoint 目录: {run_dir}  （训完才会把 latest 指过来）")

    slot_defs = build_slot_defs(tok, device)
    wb = init_wandb(tc, cfg)
    model.train()

    def _run_val(tag: str) -> float:
        """跑一遍完整 val（同一批常驻内存样本）并打印/上报 wandb；
           创新低即存 best.pt。返回本次 val loss。"""
        nonlocal best_val, best_epoch
        vl, per_slot = evaluate(model, val_dl, device, autocast_ctx, slot_defs)
        improved = vl < best_val
        if improved:
            best_val, best_epoch = vl, epoch
            save_checkpoint(os.path.join(run_dir, "best.pt"), model, cfg,
                            tc["model_type"], epoch, vl)
        if wb:
            log = {"val/loss": vl, "val/best_loss": best_val, "epoch": epoch}
            for name, (sl, sm) in per_slot.items():
                log[f"val/loss_{name}"] = sl
                log[f"val/mass_{name}"] = sm
            wb.log(log, step=global_step)
        slot_str = "  ".join(f"{n} {sl:.2f}/{sm:.2f}"
                             for n, (sl, sm) in per_slot.items())
        print(f"{tag} val_loss={vl:.4f}{'  <- best' if improved else ''}")
        print(f"  分位 loss/mass: {slot_str}")
        return vl

    for epoch in range(start_epoch, tc["epochs"] + 1):
        t0 = time.time()
        best_before = best_val                       # patience 按「整个 epoch 有无创新低」算
        epoch_start_step = global_step
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)      # stream 模式：洗牌顺序随 epoch 变化
        epoch_loss, epoch_tokens = 0.0, 0
        pending = 0                                  # 未 step 的累积 micro-batch 数
        optimizer.zero_grad(set_to_none=True)

        def _optim_step():
            nonlocal global_step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        for batch in train_dl:                       # 不依赖 len()，兼容 IterableDataset
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast_ctx():
                loss, _ = model(**batch)
            (loss / tc["grad_accum"]).backward()
            n_valid = (batch["labels"][:, 1:] != -100).sum().item()
            epoch_loss += loss.item() * n_valid
            epoch_tokens += n_valid

            pending += 1
            if pending == tc["grad_accum"]:
                pending = 0
                _optim_step()
                if wb:
                    wb.log({"train/loss": loss.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "epoch": epoch}, step=global_step)
                if global_step % tc["log_every"] == 0:
                    print(f"  epoch {epoch} step {global_step} "
                          f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                if tc["eval_every"] > 0 and global_step % tc["eval_every"] == 0:
                    _run_val(f"  [VAL@step {global_step}]")
        if pending:                                  # epoch 末尾不足 grad_accum 的余量
            _optim_step()
        if not scheduler.calibrated():               # stream：首个完整 epoch 实测步数定调度
            steps_this = global_step - epoch_start_step
            scheduler.calibrate(steps_this)
            if epoch > 1:                            # 旧格式续训：快进调度器到真实进度
                scheduler.last_epoch = steps_this * epoch
                global_step = steps_this * epoch
                print(f"[INFO] 调度器快进到 step {global_step}（补齐续训前的 {epoch - 1} 个 epoch）")
            print(f"[INFO] lr 调度校准: 实测 steps/epoch={steps_this}  "
                  f"总步数={steps_this * tc['epochs']}  "
                  f"warmup={max(int(steps_this * tc['epochs'] * tc['warmup_ratio']), 1)}")

        train_loss = epoch_loss / max(epoch_tokens, 1)
        val_loss = _run_val(f"[EPOCH {epoch}/{tc['epochs']}] "
                            f"train_loss={train_loss:.4f} ({time.time() - t0:.1f}s)")
        bad_epochs = 0 if best_val < best_before else bad_epochs + 1
        save_checkpoint(os.path.join(run_dir, "last.pt"),
                        model, cfg, tc["model_type"], epoch, val_loss,
                        extra={"optimizer": optimizer.state_dict(),
                               "sched_last_epoch": scheduler.last_epoch,
                               "sched_calib": scheduler.export_state(),
                               "global_step": global_step,
                               "best_val": best_val, "best_epoch": best_epoch})
        if wb:
            wb.log({"train/epoch_loss": train_loss, "epoch": epoch},
                   step=global_step)

        if tc["patience"] > 0 and bad_epochs >= tc["patience"]:
            print(f"[INFO] val loss 连续 {tc['patience']} 个 epoch 无改善，早停")
            break

    if wb:
        wb.summary["best_val_loss"] = best_val
        wb.summary["best_epoch"] = best_epoch
        wb.finish()
    print(f"\n[DONE] best val_loss={best_val:.4f} @ epoch {best_epoch}  "
          f"checkpoint: {run_dir}/best.pt")

    update_latest_and_prune(tc["output_dir"], run_dir, run_stamp, tc["max_ckpt_keep"])


if __name__ == "__main__":
    main()
