#!/usr/bin/env python
"""服务器训练队列：读取 manifest.csv，每张 GPU 同时跑一个任务（训练 + 测试）。

规则：
- 新任务：train.py -n <run> --no-overwrite --analyze（训练结束后自动在 val / test 上评估）。
- 已完成（predictions/test/metrics.json 已存在）：跳过。
- 训练已结束但缺测试（summary.json 的 status 为 completed / early_stopped）：只补跑 analyze.py。
- 未结束（running / failed / interrupted）：跳过并提示，需手动删除该 run 目录后再跑。
- 始终带 --no-overwrite：train.py 不带该参数时会删除同名目录。
- 数据数量与论文口径（train 7000 / val 1500 / test 1501）不符时拒绝启动，除非加 --allow-mismatch。

用法（仓库根目录，已激活 py312 环境）：
    python experiments/detection/project2_2611/server/run_queue.py --dry-run
    python experiments/detection/project2_2611/server/run_queue.py --gpus 0
    python experiments/detection/project2_2611/server/run_queue.py --gpus 0,1 --num-workers 8
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANALYZE_PY, DONE_STATUS, LOG_DIR, REPO, TRAIN_PY, dataset_report, read_json  # noqa: E402
from trainer import build_config, model_output_dir  # noqa: E402

POLL_SECONDS = 10


@dataclass
class Job:
    name: str
    model: str
    seed: int
    epochs: int | None
    overrides: list[str]
    run_name: str
    run_dir: Path
    mode: str     # train / analyze / skip
    reason: str


@dataclass
class Running:
    job: Job
    gpu: str
    proc: subprocess.Popen
    log_fh: object
    start: float


def now() -> str:
    return datetime.now().strftime("%F %T")


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    rows = []
    for row in csv.DictReader(lines):
        if str(row.get("enabled", "1")).strip().lower() in {"1", "true", "yes", "y"}:
            rows.append(row)
    return rows


def plan(row: dict, prefix: str) -> Job:
    name = row["name"].strip()
    model = row["model"].strip()
    seed = int(row["seed"])
    raw_epochs = str(row.get("epochs") or "").strip()
    epochs = int(raw_epochs) if raw_epochs else None
    overrides = [s.strip() for s in str(row.get("overrides") or "").split(";") if s.strip()]
    run_name = f"{prefix}_{name}"
    run_dir = model_output_dir(build_config(model)) / run_name
    job = Job(name, model, seed, epochs, overrides, run_name, run_dir, "train", "新训练")
    if (run_dir / "predictions" / "test" / "metrics.json").exists():
        job.mode, job.reason = "skip", "已完成（测试指标已存在）"
    elif run_dir.exists():
        status = read_json(run_dir / "summary.json").get("status")
        if status in DONE_STATUS:
            job.mode, job.reason = "analyze", f"训练已结束（{status}），补跑测试"
        else:
            job.mode, job.reason = "skip", f"目录已存在且未完成（status={status}），确认后手动删除: {run_dir}"
    return job


def train_cmd(job: Job, num_workers: int) -> list[str]:
    cmd = [sys.executable, str(TRAIN_PY), "-m", job.model, "-n", job.run_name, "--no-overwrite", "--analyze",
           "--set", f"train.seed={job.seed}", "--set", f"dataset.num_workers={num_workers}"]
    if job.epochs is not None:
        cmd += ["--epochs", str(job.epochs)]
    for item in job.overrides:
        cmd += ["--set", item]
    return cmd


def analyze_cmd(job: Job) -> list[str]:
    return [sys.executable, str(ANALYZE_PY), "-m", job.model, "-n", job.run_name]


def command_of(job: Job, num_workers: int) -> list[str]:
    return train_cmd(job, num_workers) if job.mode == "train" else analyze_cmd(job)


def launch(job: Job, gpu: str, num_workers: int) -> Running:
    cmd = command_of(job, num_workers)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = (LOG_DIR / f"{job.run_name}.stdout.log").open("a", encoding="utf-8")
    log_fh.write(f"\n===== {now()} | GPU {gpu} | {' '.join(cmd)}\n")
    log_fh.flush()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    return Running(job, gpu, proc, log_fh, time.time())


def test_metrics(run_dir: Path) -> tuple[float | None, float | None]:
    data = read_json(run_dir / "predictions" / "test" / "metrics.json")
    return data.get("map50"), data.get("map")


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.4f}"


def run_pool(jobs: list[Job], gpus: list[str], num_workers: int) -> list[dict]:
    pending, free, running, results = list(jobs), list(gpus), [], []
    try:
        while pending or running:
            while pending and free:
                job, gpu = pending.pop(0), free.pop(0)
                running.append(launch(job, gpu, num_workers))
                print(f"[{now()}] 启动 {job.run_name}（{job.mode}，GPU {gpu}）", flush=True)
            time.sleep(POLL_SECONDS)
            for r in list(running):
                rc = r.proc.poll()
                if rc is None:
                    continue
                running.remove(r)
                r.log_fh.close()
                free.append(r.gpu)
                m50, m = test_metrics(r.job.run_dir)
                minutes = (time.time() - r.start) / 60
                results.append({
                    "finished_at": now(), "run_name": r.job.run_name, "mode": r.job.mode, "gpu": r.gpu,
                    "returncode": rc, "minutes": round(minutes, 1),
                    "test_map50": fmt(m50), "test_map50_95": fmt(m),
                    "log": str(LOG_DIR / f"{r.job.run_name}.stdout.log"),
                })
                print(f"[{now()}] 结束 {r.job.run_name} rc={rc} 用时 {minutes:.1f} min "
                      f"test mAP50={fmt(m50)} mAP50-95={fmt(m)}", flush=True)
    except KeyboardInterrupt:
        print("收到中断，正在终止子进程 ...", flush=True)
        for r in running:
            r.proc.terminate()
        raise
    return results


def append_summary(rows: list[dict]) -> None:
    if not rows:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "queue_summary.csv"
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def print_plan(jobs: list[Job]) -> None:
    print(f"{'run':<44}{'模式':<9}说明")
    for j in jobs:
        print(f"{j.run_name:<44}{j.mode:<9}{j.reason}")


def command_preview(jobs: list[Job], num_workers: int) -> None:
    for j in jobs:
        print("CMD", " ".join(command_of(j, num_workers)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(Path(__file__).resolve().parent / "manifest.csv"))
    ap.add_argument("--gpus", default="0", help="逗号分隔的 GPU 编号，每张卡同时只跑一个任务（默认 0）")
    ap.add_argument("--num-workers", type=int, default=12, help="DataLoader 进程数（覆盖 global.toml 的 24）")
    ap.add_argument("--prefix", default=None, help="run 名称前缀，默认今天的日期（YYYY-MM-DD）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划与命令，不启动训练")
    ap.add_argument("--allow-mismatch", action="store_true", help="数据数量与期望不符时仍然启动")
    args = ap.parse_args()

    prefix = args.prefix or datetime.now().strftime("%Y-%m-%d")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    rep = dataset_report()
    print(f"[{now()}] 数据目录: {rep['data_dir']}")
    for split, c in rep["counts"].items():
        print(f"  {split:<5} 图片 {c['images']}  标签 {c['labels']}")
    for msg in rep["mismatch"]:
        print(f"  [WARN] {msg}")
    if rep["mismatch"] and not args.allow_mismatch and not args.dry_run:
        print("数据不完整，拒绝启动。确认无误后可加 --allow-mismatch。")
        return 2

    jobs = [plan(r, prefix) for r in read_manifest(Path(args.manifest))]
    if not jobs:
        print("manifest 中没有启用的任务。")
        return 0
    print_plan(jobs)
    todo = [j for j in jobs if j.mode != "skip"]

    if args.dry_run:
        command_preview(todo, args.num_workers)
        return 0
    if not todo:
        print("没有待执行的任务。")
        return 0
    if not gpus:
        print("--gpus 为空。")
        return 2

    results = run_pool(todo, gpus, args.num_workers)
    append_summary(results)
    failed = [r for r in results if r["returncode"] != 0]
    print(f"[{now()}] 队列结束：{len(results) - len(failed)} 成功，{len(failed)} 失败。"
          f"汇总: {LOG_DIR / 'queue_summary.csv'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
