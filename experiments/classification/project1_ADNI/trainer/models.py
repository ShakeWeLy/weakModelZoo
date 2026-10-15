"""模型注册表。

``configs/models/<x>.toml`` 中 ``[model].arch`` 选择这里的条目，``[model.params]`` 原样作为构造参数；
框架注入 ``in_features``（脑区数 N，来自数据）、``num_classes`` 与 ``transformer_embed_dim``（= features_dim）。

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
    validate: Callable[[Mapping[str, Any]], None] | None = None


def _validate_dfsc(params: Mapping[str, Any]) -> None:
    features_dim = int(params.get("features_dim", 64))
    heads = int(params.get("transformer_heads", 1))
    if features_dim % heads != 0:
        raise ValueError(f"DFSC 要求 features_dim={features_dim} 能被 transformer_heads={heads} 整除")


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "dfsc": ModelSpec("src/models/classification/DFSC/DFSC.py", "DFSC", validate=_validate_dfsc),
}


def _load_class(spec: ModelSpec) -> type[nn.Module]:
    path = ROOT / spec.module_path
    module_name = "project1_ADNI_models." + path.stem.replace("-", "_")
    module = sys.modules.get(module_name)
    if module is None:
        import_spec = importlib.util.spec_from_file_location(module_name, path)
        if import_spec is None or import_spec.loader is None:
            raise ImportError(f"无法加载模型模块: {path}")
        module = importlib.util.module_from_spec(import_spec)
        sys.modules[module_name] = module
        import_spec.loader.exec_module(module)
    return getattr(module, spec.class_name)


def build_model(cfg: Mapping[str, Any], num_nodes: int, num_classes: int) -> nn.Module:
    model_cfg = cfg["model"]
    arch = str(model_cfg["arch"])
    if arch not in MODEL_REGISTRY:
        raise KeyError(f"未注册的模型 arch={arch!r}，可选: {', '.join(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[arch]
    params = dict(model_cfg.get("params", {}))
    if spec.validate is not None:
        spec.validate(params)
    params["in_features"] = num_nodes
    params["num_classes"] = num_classes
    params["transformer_embed_dim"] = int(params.get("features_dim", 64))
    return _load_class(spec)(**params)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
