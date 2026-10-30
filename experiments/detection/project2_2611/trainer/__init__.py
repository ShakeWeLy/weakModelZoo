"""project2_2611 通用 YOLOv5 训练 / 评估框架。"""

from .config import PROJECT_DIR, ROOT, available_models, build_config, model_output_dir
from .evaluator import evaluate
from .models import build_model, count_parameters
from .recorder import RunRecorder, unique_run_name
from .trainer import Trainer, resolve_device

__all__ = [
    "PROJECT_DIR",
    "ROOT",
    "RunRecorder",
    "Trainer",
    "available_models",
    "build_config",
    "build_model",
    "count_parameters",
    "evaluate",
    "model_output_dir",
    "resolve_device",
    "unique_run_name",
]
