#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
submit_eval: 把 test 评测任务提交到 Luban 平台，并轮询到终态。

由统一入口 eval/run_eval.sh 在 [eval] launch_mode=platform 时调用；
平台侧执行 [platform] script_path（eval_platform.sh，NFS 路径），
scriptParam 传平台容器内的 common.conf 路径——两侧共用同一份配置文件
（同一块 NFS 盘的两个挂载名），本地改完 conf 提交即生效。
跟 train/submit_train.py 是同一套提交/轮询逻辑，只是平台侧执行的脚本不同
（eval_platform.sh 不经过 run_eval.sh 的 launch_mode 分发，避免递归提交）。

用法:
    python submit_eval.py [common.conf] [--dry-run]
    --dry-run: 只打印将要提交的 payload，不真正提交（核对配置用）
"""

import os
import sys
import json
import time
import configparser


def load_platform_config(conf_path: str) -> dict:
    cp = configparser.ConfigParser()
    if not cp.read(conf_path, encoding="utf-8"):
        raise FileNotFoundError(f"找不到配置文件: {conf_path}")
    if not cp.has_section("platform"):
        raise ValueError("common.conf 缺少 [platform] 段（launch_mode=platform 需要）")
    g = lambda k, d=None: cp.get("platform", k, fallback=d)  # noqa: E731
    # 平台侧项目根 = platform_base_dir/<项目名>；项目名默认 = conf 所在文件夹名，
    # 整个项目目录复制成新实验（sft_Version3 -> sft_V3_tiger）时自动适应
    project_name = (g("project_name", "") or "").strip() or \
        os.path.basename(os.path.dirname(os.path.abspath(conf_path)))
    base_dir = (g("platform_base_dir", "") or "").rstrip("/")
    project_dir = f"{base_dir}/{project_name}"
    pc = {
        "api_url": g("api_url"),
        "project_uuid": g("project_uuid"),
        "token": g("token"),
        "user_uuid": g("user_uuid"),
        "region": g("region"),
        "image_uuid": g("image_uuid"),
        "device": g("device", "A6000"),
        "project_name": project_name,
        "platform_project_dir": project_dir,
        "script_path": f"{project_dir}/eval/eval_platform.sh",
        "poll_interval": cp.getint("platform", "poll_interval", fallback=60),
        # eval_job_name_prefix 留空就固定用 gamer_eval（不回退到训练的
        # job_name_prefix——那个默认值 gamer_train 会让评测任务在 Luban 控制台
        # 也显示成"训练"，误导性比留空更差，不如给个独立、说得通的默认值）
        "job_name_prefix": (g("eval_job_name_prefix", "") or "").strip() or "gamer_eval",
        "level": g("level", "PRO"),
        "priority": cp.getint("platform", "priority", fallback=2),
        "backoff_limit": cp.getint("platform", "backoff_limit", fallback=1),
    }
    if not base_dir:
        raise ValueError("[platform] 缺少 platform_base_dir")
    rkey = f"resource_{pc['device'].lower()}"
    pc["resource_uuid"] = g(rkey)
    if not pc["resource_uuid"]:
        raise ValueError(f"[platform] 缺少 {rkey}（device={pc['device']} 对应的资源 uuid）")
    missing = [k for k in ("api_url", "project_uuid", "token", "user_uuid",
                           "image_uuid") if not pc[k]]
    if missing:
        raise ValueError(f"[platform] 缺少必填项: {missing}")
    return pc


def _headers(pc: dict) -> dict:
    return {
        "Content-Type": "application/json",
        "JIANSHU-PROJECT-TOKEN": pc["token"],
        "JIANSHU-PROJECT-UUID": pc["project_uuid"],
    }


def build_payload(pc: dict, job_name: str) -> dict:
    # scriptParam = 平台容器内的 common.conf 路径（eval_platform.sh 的 $1）
    conf_on_platform = os.path.join(pc["platform_project_dir"], "common.conf")
    return {
        "userUuid": pc["user_uuid"],
        "projectUuid": pc["project_uuid"],
        "imageUuid": pc["image_uuid"],
        "scriptPath": pc["script_path"],
        "scriptParam": conf_on_platform,
        "scriptSourceType": "file",
        "resourceUuid": pc["resource_uuid"],
        "regionName": pc["region"],
        "name": job_name,
        "level": pc["level"],
        "backoffLimit": pc["backoff_limit"],
        "priority": pc["priority"],
        "volumeRegions": [],
    }


def submit_job(pc: dict, payload: dict) -> str:
    import requests
    resp = requests.post(pc["api_url"], headers=_headers(pc), json=payload)
    data = resp.json()
    job_uuid = data.get("data", {}).get("appId")
    if not job_uuid or job_uuid == "null":
        raise RuntimeError(f"提交失败: {data}")
    return job_uuid


def poll_job(pc: dict, job_uuid: str) -> str:
    """每 poll_interval 秒查一次，直到 SUCCEEDED / FAILED。返回终态。"""
    import requests
    url = f"{pc['api_url']}/{job_uuid}?userUuid={pc['user_uuid']}"
    n = 0
    while True:
        try:
            raw = (requests.get(url, headers=_headers(pc)).json()
                   .get("data", {}).get("status", "") or "").upper()
        except Exception as e:                      # 网络抖动不中断轮询
            print(f"[WARN] 查询状态失败（{e}），{pc['poll_interval']}s 后重试")
            raw = ""
        if raw in ("SUCCEEDED", "COMPLETED", "SUCCESS"):
            return "SUCCEEDED"
        if raw in ("FAILED", "ERROR", "STOPPED"):
            return "FAILED"
        n += 1
        if n % 10 == 1:                             # 每 ~10 个间隔报一次心跳
            print(f"[INFO] 任务运行中（状态: {raw or '未知'}，已等待 "
                  f"{n * pc['poll_interval'] // 60} 分钟）")
        time.sleep(pc["poll_interval"])


def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv[1:]
    conf_path = args[0] if args else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "common.conf")
    pc = load_platform_config(conf_path)

    job_name = (f"{pc['job_name_prefix']}_{pc['project_name']}_"
                f"{time.strftime('%m%d_%H%M%S')}")
    payload = build_payload(pc, job_name)
    print(f"[INFO] 项目: {pc['project_name']}  平台目录: {pc['platform_project_dir']}")
    print(f"[INFO] 提交目标: {pc['api_url']}  device={pc['device']}  "
          f"region={pc['region']}")
    print(f"[INFO] 任务名: {job_name}")
    print(f"[INFO] 平台执行: {payload['scriptPath']} {payload['scriptParam']}")
    if dry_run:
        print("[DRY-RUN] payload:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    job_uuid = submit_job(pc, payload)
    print(f"[INFO] 提交成功: {job_name} -> {job_uuid}")
    status = poll_job(pc, job_uuid)
    print(f"[{'DONE' if status == 'SUCCEEDED' else 'FAIL'}] "
          f"任务 {job_uuid} 终态: {status}")
    if status != "SUCCEEDED":
        sys.exit(1)


if __name__ == "__main__":
    main()
