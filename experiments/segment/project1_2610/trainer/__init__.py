"""project1_2610 通用训练 / 预测分析框架。"""

from .config import PROJECT_DIR, ROOT, available_models, build_config, model_output_dir
from .predictor import Predictor, format_split_summary, load_trained_model
from .recorder import RunRecorder, unique_run_name
from .trainer import Trainer, resolve_device

__all__ = [
    "PROJECT_DIR",
    "ROOT",
    "Predictor",
    "RunRecorder",
    "Trainer",
    "available_models",
    "build_config",
    "format_split_summary",
    "load_trained_model",
    "model_output_dir",
    "resolve_device",
    "unique_run_name",
]
