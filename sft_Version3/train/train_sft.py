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
               lr 调度的总步数用 est_steps_per_epoch 估算；
  3. GAMERModel（配置由 tokenizer 推导：vocab_size / tokens_per_item / behavior_levels）；
  4. AdamW + 线性 warmup(4%) + 余弦衰减到 min_lr；
  5. 逐 epoch：train 全 token 监督 NTP；epoch 末跑 val（teacher-forcing，只算 label 区，
     按有效 token 数加权聚合）；val loss 创新低则存 best.pt，每个 epoch 更新 last.pt；
  6. patience>0 时按 val loss 早停；否则训满 epochs（论文做法，最终用 best.pt）。

说明：
  - 模型仅 ~40M，单卡即可，不需要 deepspeed/ZeRO；CUDA 下自动用 bf16 autocast；
  - checkpoint 含 model state_dict + GAMERConfig 字段 + epoch/val_loss，
    推理端用 load_checkpoint() 还原。

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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer_sid import SIDTokenizer            # noqa: E402
from modeling import GAMERConfig, GAMERModel      # noqa: E402
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
        "data_mode": cp.get("train", "data_mode", fallback="jsonl"),
        "shuffle_buffer": cp.getint("train", "shuffle_buffer", fallback=10000),
        "val_sample_rate": cp.getfloat("train", "val_sample_rate", fallback=0.05),
        "max_val_users": cp.getint("train", "max_val_users", fallback=20000),
        "est_steps_per_epoch": cp.getint("train", "est_steps_per_epoch", fallback=1000),
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
        "wandb_project": cp.get("train", "wandb_project", fallback="").strip(),
        "wandb_run_name": cp.get("train", "wandb_run_name", fallback="").strip(),
    }


def init_wandb(tc: dict, model_cfg) -> "object":
    """wandb_project 配置了才开启；wandb 未安装或未登录时降级为警告，不阻塞训练。
       API key 从环境读（wandb login / WANDB_API_KEY），不进配置文件。"""
    if not tc["wandb_project"]:
        return None
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb 未安装（pip install wandb），跳过监控")
        return None
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
        print(f"[WARN] wandb 初始化失败，跳过监控: {e}")
        return None


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float,
                    lr: float, min_lr: float):
    """线性 warmup + 余弦衰减到 min_lr（论文 4.1.4）。"""
    warmup = max(int(total_steps * warmup_ratio), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return (min_lr + (lr - min_lr) * cos) / lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------
# 评估：val loss（teacher-forcing，按有效 token 数加权）
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, val_dl, device, autocast_ctx) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in val_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        n_valid = (batch["labels"][:, 1:] != -100).sum().item()  # 与模型内部 shift 一致
        if n_valid == 0:
            continue
        with autocast_ctx():
            loss, _ = model(**batch)
        total_loss += loss.item() * n_valid
        total_tokens += n_valid
    model.train()
    return total_loss / max(total_tokens, 1)


def save_checkpoint(path: str, model, cfg: GAMERConfig, epoch: int, val_loss: float):
    torch.save({
        "model": model.state_dict(),
        "config": asdict(cfg),
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def load_checkpoint(path: str, map_location="cpu"):
    """推理端还原：返回 (model, config, meta)。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg_d = dict(ckpt["config"])
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
        steps_per_epoch = tc["est_steps_per_epoch"]
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

    # 模型（结构参数由 tokenizer 推导）
    cfg = GAMERConfig(vocab_size=tok.vocab_size,
                      tokens_per_item=tok.num_levels,
                      num_behaviors=len(tok.behaviors),
                      behavior_levels=tok.behavior_levels,
                      max_position_embeddings=tc["max_len"],
                      pad_token_id=tok.pad_id)
    model = GAMERModel(cfg).to(device)
    print(f"[INFO] 模型参数量: {model.num_parameters() / 1e6:.2f}M")

    total_steps = steps_per_epoch * tc["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"],
                                  weight_decay=tc["weight_decay"])
    scheduler = build_scheduler(optimizer, total_steps, tc["warmup_ratio"],
                                tc["lr"], tc["min_lr"])
    est = "（估算，stream 模式）" if tc["data_mode"] == "stream" else ""
    print(f"[INFO] epochs={tc['epochs']}  steps/epoch={steps_per_epoch}{est}  "
          f"总步数={total_steps}  有效batch={tc['batch_size'] * tc['grad_accum']}")

    os.makedirs(tc["output_dir"], exist_ok=True)
    wb = init_wandb(tc, cfg)
    best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    global_step = 0
    model.train()

    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
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
        if pending:                                  # epoch 末尾不足 grad_accum 的余量
            _optim_step()

        train_loss = epoch_loss / max(epoch_tokens, 1)
        val_loss = evaluate(model, val_dl, device, autocast_ctx)
        mark = ""
        if val_loss < best_val:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            save_checkpoint(os.path.join(tc["output_dir"], "best.pt"),
                            model, cfg, epoch, val_loss)
            mark = "  <- best"
        else:
            bad_epochs += 1
        save_checkpoint(os.path.join(tc["output_dir"], "last.pt"),
                        model, cfg, epoch, val_loss)
        if wb:
            wb.log({"train/epoch_loss": train_loss, "val/loss": val_loss,
                    "val/best_loss": best_val, "epoch": epoch}, step=global_step)
        print(f"[EPOCH {epoch}/{tc['epochs']}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} ({time.time() - t0:.1f}s){mark}")

        if tc["patience"] > 0 and bad_epochs >= tc["patience"]:
            print(f"[INFO] val loss 连续 {tc['patience']} 个 epoch 无改善，早停")
            break

    if wb:
        wb.summary["best_val_loss"] = best_val
        wb.summary["best_epoch"] = best_epoch
        wb.finish()
    print(f"\n[DONE] best val_loss={best_val:.4f} @ epoch {best_epoch}  "
          f"checkpoint: {tc['output_dir']}/best.pt")


if __name__ == "__main__":
    main()
