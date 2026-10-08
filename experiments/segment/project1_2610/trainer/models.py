"""模型注册表。

``configs/models/<x>.toml`` 中 ``[model].arch`` 选择这里的条目，``[model.params]`` 原样作为构造参数；
``in_channels`` / ``out_channels``（以及需要时的 ``image_size``）由框架注入。

新增模型：在 ``MODEL_REGISTRY`` 加一项，并新建对应 toml 即可。
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from torch import nn

from .config import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class ModelSpec:
    module_path: str
    class_name: str
    inject_image_size: bool = False
    validate: Callable[[int, Mapping[str, Any]], None] | None = None


def _validate_swin(image_size: int, params: Mapping[str, Any]) -> None:
    patch = int(params.get("patch_size", 4))
    window = int(params.get("window_size", 7))
    if image_size % 32 != 0:
        raise ValueError(f"Swin-UNet 要求 image_size 能被 32 整除，当前为 {image_size}")
    if (image_size // patch) % window != 0:
        raise ValueError(
            f"Swin-UNet 要求 image_size/patch_size 能被 window_size={window} 整除，当前 image_size={image_size}"
        )


def _validate_div16(image_size: int, params: Mapping[str, Any]) -> None:
    if image_size % 16 != 0:
        raise ValueError(f"image_size 需能被 16 整除（4 次下采样），当前为 {image_size}")


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "unet": ModelSpec("src/models/segment/unet/unet.py", "UNet", validate=_validate_div16),
    "attn_unet": ModelSpec(
        "src/models/segment/AttentionGateUnet/AttentionGateUnet.py", "AttentionGateUnet",
        validate=_validate_div16,
    ),
    "transunet": ModelSpec(
        "src/models/segment/TransUNet/TransUNet.py", "TransUNet",
        inject_image_size=True, validate=_validate_div16,
    ),
    "swin_unet": ModelSpec("src/models/segment/Swin-UNet/Swin-UNet.py", "SwinUNet", validate=_validate_swin),
    "psc_unet": ModelSpec("src/models/segment/PSC-UNet/PSC-UNet.py", "PSC_UNet"),
    "vm_unet": ModelSpec("src/models/segment/VM-UNet/VM-UNet.py", "VMUNet"),
}


def _load_class(spec: ModelSpec) -> type[nn.Module]:
    path = ROOT / spec.module_path
    module_name = "project1_2610_models." + path.stem.replace("-", "_")
    module = sys.modules.get(module_name)
    if module is None:
        import_spec = importlib.util.spec_from_file_location(module_name, path)
        if import_spec is None or import_spec.loader is None:
            raise ImportError(f"无法加载模型模块: {path}")
        module = importlib.util.module_from_spec(import_spec)
        sys.modules[module_name] = module
        import_spec.loader.exec_module(module)
    return getattr(module, spec.class_name)


def model_in_channels(model_cfg: Mapping[str, Any]) -> int:
    return 3 if model_cfg.get("repeat_gray_to_rgb", False) else 1


def build_model(cfg: Mapping[str, Any], num_classes: int) -> nn.Module:
    model_cfg = cfg["model"]
    arch = str(model_cfg["arch"])
    if arch not in MODEL_REGISTRY:
        raise KeyError(f"未注册的模型 arch={arch!r}，可选: {', '.join(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[arch]
    image_size = int(cfg["dataset"]["image_size"])
    params = dict(model_cfg.get("params", {}))
    if spec.validate is not None:
        spec.validate(image_size, params)
    params["in_channels"] = model_in_channels(model_cfg)
    params["out_channels"] = num_classes
    if spec.inject_image_size:
        params.setdefault("image_size", image_size)
    return _load_class(spec)(**params)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
