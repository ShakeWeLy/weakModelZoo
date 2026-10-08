"""配置加载：全局 ``configs/global.toml`` + 模型 ``configs/models/<model>.toml`` 深度合并。

合并顺序（后者覆盖前者）::

    global.toml → models/<model>.toml → global.[quick] → model.[quick] → CLI (--epochs / --set)

``[quick]`` 仅在 ``--quick`` 时生效，合并后会从最终配置中移除。
"""

from __future__ import annotations

import copy
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT = PROJECT_DIR.parents[2]
CONFIG_DIR = PROJECT_DIR / "configs"
GLOBAL_CONFIG_PATH = CONFIG_DIR / "global.toml"
MODEL_CONFIG_DIR = CONFIG_DIR / "models"

REQUIRED_SECTIONS = ("experiments", "dataset", "hyperparameters", "model", "train", "test")


def load_toml(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def available_models() -> list[str]:
    return sorted(path.stem for path in MODEL_CONFIG_DIR.glob("*.toml"))


def resolve_model_config_path(model: str | Path) -> Path:
    path = Path(model)
    if path.suffix == ".toml" and path.exists():
        return path.resolve()
    candidate = MODEL_CONFIG_DIR / f"{model}.toml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"未找到模型配置 {model!r}，可选: {', '.join(available_models())}"
    )


def parse_override_value(raw: str) -> Any:
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """应用 ``section.key=value`` 形式的覆盖，value 按 TOML 语法解析（解析失败视为字符串）。"""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set 需要 key=value 形式，当前为 {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        keys = [k.strip() for k in dotted_key.strip().split(".") if k.strip()]
        if not keys:
            raise ValueError(f"--set 键为空: {item!r}")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set {dotted_key}: {key} 不是配置表")
        node[keys[-1]] = parse_override_value(raw_value.strip())
    return cfg


def build_config(
    model: str | Path,
    *,
    global_path: Path = GLOBAL_CONFIG_PATH,
    quick: bool = False,
    epochs: int | None = None,
    overrides: Iterable[str] = (),
) -> dict[str, Any]:
    global_cfg = load_toml(global_path)
    model_path = resolve_model_config_path(model)
    model_cfg = load_toml(model_path)

    cfg = deep_merge(global_cfg, model_cfg)
    cfg.pop("quick", None)
    if quick:
        cfg = deep_merge(cfg, global_cfg.get("quick", {}))
        cfg = deep_merge(cfg, model_cfg.get("quick", {}))
    if epochs is not None:
        cfg.setdefault("hyperparameters", {})["num_epochs"] = int(epochs)
    apply_overrides(cfg, overrides)

    for section in REQUIRED_SECTIONS:
        cfg.setdefault(section, {})
    model_section = cfg["model"]
    model_section.setdefault("arch", model_path.stem)
    model_section.setdefault("display_name", model_section["arch"])
    model_section.setdefault("params", {})
    exp = cfg["experiments"]
    if not exp.get("date"):
        exp["date"] = date.today().isoformat()
    cfg["_meta"] = {
        "global_config": str(Path(global_path).resolve()),
        "model_config": str(model_path),
        "quick": quick,
        "overrides": list(overrides),
    }
    return cfg


def resolve_path(value: str | Path | None, *, base: Path = ROOT) -> Path | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def model_output_dir(cfg: Mapping[str, Any]) -> Path:
    """``<project>/<display_name>/<runs_dir>``，与 2609 的 ``<Model>/runs`` 布局一致。"""
    runs_dir = Path(cfg["experiments"].get("runs_dir", "runs"))
    if runs_dir.is_absolute():
        return runs_dir
    return PROJECT_DIR / str(cfg["model"]["display_name"]) / runs_dir
