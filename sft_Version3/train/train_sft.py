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
    推理端用 load_checkpoint() 按 model_type 还原对应结构。

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
    if tc["data_mode"] == "stream":
        # 流式：train 直连 HDFS（IterableDataset，worker 按 part 分片）；
        # val 启动时抽样收集一次，常驻内存
        from dataset import GAMERStreamingTrainDataset, collect_val_samples
        train_ds = GAMERStreamingTrainDataset(conf_path, tok, max_len=tc["max_len"],
                                              shuffle_buffer=tc["shuffle_buffer"],
                                              seed=tc["seed"])
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

    # checkpoint 目录：全新训练建「启动时间」子目录，续训沿用原目录；
    # latest 软链指向本次写入的目录，eval 默认读 latest/best.pt
    if resume_dir:
        run_dir = resume_dir
        run_stamp = os.path.basename(run_dir.rstrip("/"))
    else:
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(tc["output_dir"], run_stamp)
        os.makedirs(run_dir, exist_ok=True)
    if os.path.dirname(os.path.abspath(run_dir)) == os.path.abspath(tc["output_dir"]):
        latest = os.path.join(tc["output_dir"], "latest")
        if os.path.islink(latest):
            os.remove(latest)
        if not os.path.exists(latest):             # 同名真目录存在则不动，只告警
            os.symlink(run_stamp, latest)
        else:
            print(f"[WARN] {latest} 已存在且不是软链，跳过更新")
    if not tc["wandb_run_name"]:                   # wandb run 名与目录共用时间戳，便于对应
        suffix = f"-r{start_epoch}" if resume_dir else ""
        tc["wandb_run_name"] = f"gamer-{tc['data_mode']}-{run_stamp}{suffix}"
    print(f"[INFO] checkpoint 目录: {run_dir}  （latest -> {run_stamp}）")

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


if __name__ == "__main__":
    main()
