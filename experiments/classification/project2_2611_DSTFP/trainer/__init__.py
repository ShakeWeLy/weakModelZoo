"""project2_2611_DSTFP：DSTFP 分类实验框架（初始化占位）。"""

from .config import PROJECT_DIR, ROOT, available_models, build_config, model_output_dir
from .recorder import RunRecorder, unique_run_name
from .trainer import PlaceholderTrainer, resolve_device

__all__ = [
    "PROJECT_DIR",
    "ROOT",
    "PlaceholderTrainer",
    "RunRecorder",
    "available_models",
    "build_config",
    "model_output_dir",
    "resolve_device",
    "unique_run_name",
]
