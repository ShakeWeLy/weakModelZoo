"""project1_ADNI 五折交叉验证训练框架（DFSC 等分类模型共用）。"""

from .config import PROJECT_DIR, ROOT, available_models, build_config, model_output_dir
from .recorder import RunRecorder, unique_run_name
from .trainer import CVTrainer, resolve_device

__all__ = [
    "PROJECT_DIR",
    "ROOT",
    "CVTrainer",
    "RunRecorder",
    "available_models",
    "build_config",
    "model_output_dir",
    "resolve_device",
    "unique_run_name",
]
