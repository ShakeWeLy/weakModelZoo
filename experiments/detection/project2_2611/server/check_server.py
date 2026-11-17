#!/usr/bin/env python
"""服务器自检（开训前运行一次）：环境、GPU、数据集、模型前向；可选 --smoke 跑一次 2 epoch 冒烟。

用法（仓库根目录，已激活 py312 环境）：
    python experiments/detection/project2_2611/server/check_server.py
    python experiments/detection/project2_2611/server/check_server.py --smoke --smoke-workers 4

--smoke 会执行 train.py --quick --analyze（小批次训练 2 个 epoch + val 评估），结束后删除该临时 run。
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, TRAIN_PY, dataset_report, read_json  # noqa: E402
from trainer import build_config, build_model, model_output_dir  # noqa: E402
from trainer.data import load_dataset_meta  # noqa: E402

CHECK_MODELS = ["yolov5s", "sgc_yolov5s_c3_sgdown_gbot_dw5"]
SMOKE_NAME = "_smoke_check"


def check_env() -> bool:
    print(f"Python {platform.python_version()} | torch {torch.__version__} | CUDA(build) {torch.version.cuda}")
    ok = True
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} | 显存 {p.total_memory / 2 ** 30:.1f} GiB")
    else:
        ok = False
        print("  [FAIL] CUDA 不可用：检查驱动版本，以及 setup_env.sh 中的 TORCH_CUDA 是否匹配")
    try:
        import cv2
        import torchvision

        print(f"cv2 {cv2.__version__} | torchvision {torchvision.__version__}")
    except ImportError as e:
        ok = False
        print(f"  [FAIL] 依赖缺失: {e}")
    return ok


def check_data() -> tuple[bool, int]:
    rep = dataset_report()
    print(f"数据目录: {rep['data_dir']} | 存在: {rep['exists']}")
    for split, c in rep["counts"].items():
        print(f"  {split:<5} 图片 {c['images']}  标签 {c['labels']}")
    ok = rep["exists"] and not any(m.startswith(("数据目录不存在", "缺少 data.yaml")) for m in rep["mismatch"])
    nc = 1
    if ok:
        try:
            nc, names = load_dataset_meta(Path(rep["data_dir"]))
            print(f"data.yaml: nc={nc} names={names}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [FAIL] 读取 data.yaml 失败: {e}")
    for msg in rep["mismatch"]:
        print(f"  [WARN] {msg}")
    if rep["mismatch"]:
        ok = False
    return ok, nc


def check_models(nc: int, device: torch.device) -> bool:
    ok = True
    for key in CHECK_MODELS:
        model = None
        try:
            model = build_model(build_config(key), nc).to(device).eval()
            params = sum(p.numel() for p in model.parameters()) / 1e6
            x = torch.rand(2, 3, 640, 640, device=device)
            t0 = time.time()
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred, raw = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            scales = [tuple(r.shape) for r in raw]
            print(f"[OK] {key}: {params:.2f} M 参数 | pred {tuple(pred.shape)} | 尺度 {scales} | "
                  f"前向 {time.time() - t0:.2f}s")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"[FAIL] {key}: {type(e).__name__}: {e}")
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return ok


def run_smoke(num_workers: int) -> bool:
    run_dir = model_output_dir(build_config("yolov5s")) / SMOKE_NAME
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [sys.executable, str(TRAIN_PY), "-m", "yolov5s", "--quick", "-n", SMOKE_NAME, "--no-overwrite",
           "--analyze", "--set", f"dataset.num_workers={num_workers}"]
    print("冒烟命令:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(REPO))
    status = read_json(run_dir / "summary.json").get("status")
    val_ok = (run_dir / "predictions" / "val" / "metrics.json").exists()
    print(f"冒烟结果: returncode={rc} | status={status} | 验证集评估已产出={val_ok}")
    shutil.rmtree(run_dir, ignore_errors=True)
    return rc == 0 and status == "completed" and val_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="另外跑一次 2 epoch 冒烟（train + analyze 全链路）")
    ap.add_argument("--smoke-workers", type=int, default=4, help="冒烟时的 DataLoader 进程数（Windows 本地请用 0）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, bool] = {}
    print("== 环境 ==")
    results["环境"] = check_env()
    print("== 数据集 ==")
    results["数据集"], nc = check_data()
    print("== 模型前向（随机输入，batch=2，640）==")
    results["模型"] = check_models(nc, device)
    if args.smoke:
        print("== 冒烟训练（2 epoch 小批次 + 验证集评估）==")
        results["冒烟"] = run_smoke(args.smoke_workers)
    print("== 结果 ==")
    for name, passed in results.items():
        print(f"  {name:<6} {'通过' if passed else '未通过'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
