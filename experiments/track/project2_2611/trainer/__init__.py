"""project2_2611 跟踪训练框架（SiamRPN++ / Anti-UAV）。"""

from .config import PROJECT_DIR, ROOT, available_models, build_config, model_output_dir
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
    "model_output_dir",
    "resolve_device",
    "unique_run_name",
]
