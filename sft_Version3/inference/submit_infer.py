#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
submit_infer: 把批量推理任务提交到 Luban 平台，并轮询到终态。

由统一入口 inference/generate.sh 在 [inference] launch_mode=platform 时调用；
平台侧执行 [platform] script_path（infer_platform.sh，NFS 路径），
scriptParam 传平台容器内的 common.conf 路径——两侧共用同一份配置文件
（同一块 NFS 盘的两个挂载名），本地改完 conf 提交即生效。
跟 train/submit_train.py、eval/submit_eval.py 是同一套提交/轮询逻辑，只是
平台侧执行的脚本不同（infer_platform.sh 不经过 generate.sh 的 launch_mode
分发，避免递归提交）；轮询行为也保持一致——阻塞到任务终态才返回，即使是
全量窗口的缓存预热跑得比 train/eval 久很多也一样等（跟 train/eval 行为对齐，
不做成 fire-and-forget）。

[inference] gpu_num>1 时是数据并行模式：提交任何 GPU 任务之前，先在当前机器
（不需要 GPU）内联跑一次 plan_infer.py 的判重预处理——流式扫全量用户、跟历史
缓存比对分类，命中的直接写 rec_{行为}_part_cold.parquet，新增/过期的按负载
均衡分配到 gpu_num 个分片、写成本地任务清单，随后删掉本轮读到的旧历史文件。
再并行提交 gpu_num 个独立单 GPU 任务（每个任务的 scriptParam 在 common.conf
路径后面多带一个 shard_index 参数，对应读取自己那份任务清单），并发轮询全部
任务到终态；全部成功才清理本地任务清单文件。有任何一个失败就直接以非零码
退出（不清理任务清单）——重新执行本脚本即可安全恢复：已成功分片的输出这次
会被判重识别为「命中」，不会重复推理，但目前没有单独重试某一个失败分片的
入口，会整体重新走一遍判重预处理。

用法:
    python submit_infer.py [common.conf] [--dry-run]
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
        "script_path": f"{project_dir}/inference/infer_platform.sh",
        "poll_interval": cp.getint("platform", "poll_interval", fallback=60),
        # 留空就固定用 gamer_infer（不回退到训练的 job_name_prefix——那个默认值
        # gamer_train 会让推理任务在 Luban 控制台也显示成"训练"，误导性更大）
        "job_name_prefix": (g("infer_job_name_prefix", "") or "").strip() or "gamer_infer",
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


def build_payload(pc: dict, job_name: str, shard_index: int = None) -> dict:
    # scriptParam = 平台容器内的 common.conf 路径（infer_platform.sh 的 $1）；
    # 数据并行分片任务在后面多带一个空格分隔的 shard_index（infer_platform.sh
    # 的 $2，透传给 generate.py 作为它的第二个位置参数）
    conf_on_platform = os.path.join(pc["platform_project_dir"], "common.conf")
    script_param = conf_on_platform if shard_index is None \
        else f"{conf_on_platform} {shard_index}"
    return {
        "userUuid": pc["user_uuid"],
        "projectUuid": pc["project_uuid"],
        "imageUuid": pc["image_uuid"],
        "scriptPath": pc["script_path"],
        "scriptParam": script_param,
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


def poll_jobs(pc: dict, jobs: list) -> dict:
    """数据并行用：并发轮询多个任务（同一个 poll_interval 节奏挨个查一遍，
       而不是逐个阻塞 poll_job），直到全部到终态。jobs = [(shard_index,
       job_uuid), ...]，返回 {job_uuid: "SUCCEEDED"/"FAILED"}。"""
    import requests
    pending = {uuid for _, uuid in jobs}
    result = {}
    n = 0
    while pending:
        for uuid in list(pending):
            url = f"{pc['api_url']}/{uuid}?userUuid={pc['user_uuid']}"
            try:
                raw = (requests.get(url, headers=_headers(pc)).json()
                       .get("data", {}).get("status", "") or "").upper()
            except Exception as e:                  # 网络抖动不中断轮询
                print(f"[WARN] 查询 {uuid} 状态失败（{e}）")
                continue
            if raw in ("SUCCEEDED", "COMPLETED", "SUCCESS"):
                result[uuid] = "SUCCEEDED"
                pending.discard(uuid)
            elif raw in ("FAILED", "ERROR", "STOPPED"):
                result[uuid] = "FAILED"
                pending.discard(uuid)
        n += 1
        if pending:
            if n % 10 == 1:
                print(f"[INFO] {len(pending)}/{len(jobs)} 个分片仍在运行"
                      f"（已等待 {n * pc['poll_interval'] // 60} 分钟）")
            time.sleep(pc["poll_interval"])
    return result


def _submit_single(pc: dict, dry_run: bool) -> None:
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


def _submit_sharded(pc: dict, conf_path: str, gpu_num: int, dry_run: bool) -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import infer_common as ifc
    import hdfs_utils as hu
    import plan_infer

    ic = ifc.load_infer_config(conf_path)
    print(f"[INFO] 项目: {pc['project_name']}  平台目录: {pc['platform_project_dir']}")
    print(f"[INFO] 数据并行 gpu_num={gpu_num}  target_dt={ic['infer_end']}")

    plan_result = None
    if dry_run:
        print("[DRY-RUN] 跳过判重预处理（不真正提交/不改 HDFS/不建任务清单）")
    else:
        fs = hu.get_fs()
        hu.mark_doing(fs, ic["hdfs_output_root"], ic["infer_end"])
        print("[INFO] 判重预处理开始（读历史缓存 + 流式扫描全量用户 + 分类）...")
        plan_result = plan_infer.run_planning(conf_path, ic)
        print("[INFO] 判重预处理完成")

    jobs = []
    for shard_index in range(gpu_num):
        job_name = (f"{pc['job_name_prefix']}_{pc['project_name']}_"
                    f"shard{shard_index}_{time.strftime('%m%d_%H%M%S')}")
        payload = build_payload(pc, job_name, shard_index=shard_index)
        print(f"[INFO] shard {shard_index} 平台执行: "
              f"{payload['scriptPath']} {payload['scriptParam']}")
        if dry_run:
            print(f"[DRY-RUN] shard {shard_index} payload:")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            continue
        job_uuid = submit_job(pc, payload)
        print(f"[INFO] shard {shard_index} 提交成功: {job_name} -> {job_uuid}")
        jobs.append((shard_index, job_uuid))
    if dry_run:
        return

    print(f"[INFO] {gpu_num} 个分片任务已全部提交，开始并发轮询 ...")
    statuses = poll_jobs(pc, jobs)
    all_ok = True
    for shard_index, job_uuid in jobs:
        status = statuses.get(job_uuid, "FAILED")
        print(f"  shard {shard_index} ({job_uuid}): {status}")
        all_ok = all_ok and status == "SUCCEEDED"

    if all_ok:
        hu.mark_done(fs, ic["hdfs_output_root"], ic["infer_end"])
        print(f"[INFO] 已标记 dt={ic['infer_end']} 分区完成（.done）")
        plan_infer.summarize_final(ic, plan_result)
        rdir = ifc.run_dir(ic)
        for b in ic["behaviors"]:
            for i in range(gpu_num):
                wl_path = ifc.work_list_path(rdir, b, i)
                if os.path.exists(wl_path):
                    os.remove(wl_path)
        print("[DONE] 全部分片成功，已清理本地任务清单文件")
    else:
        print("[FAIL] 有分片失败。直接重新执行本命令即可恢复：已成功的分片这次"
              "会被判重识别为「命中」直接跳过，只有失败分片对应的那部分会"
              "重新推理——不会丢数据，但会整体重新跑一遍判重预处理（不是真正"
              "只重跑失败分片，目前没有单独指定 shard 重试的入口）")
        sys.exit(1)


def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv[1:]
    conf_path = args[0] if args else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "common.conf")
    pc = load_platform_config(conf_path)

    cp = configparser.ConfigParser()
    cp.read(conf_path, encoding="utf-8")
    gpu_num = cp.getint("inference", "gpu_num", fallback=1)

    if gpu_num <= 1:
        _submit_single(pc, dry_run)
    else:
        _submit_sharded(pc, conf_path, gpu_num, dry_run)


if __name__ == "__main__":
    main()
