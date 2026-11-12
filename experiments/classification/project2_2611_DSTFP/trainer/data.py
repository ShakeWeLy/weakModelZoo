"""数据模块占位。

目标对接：
- ``DSTFP/module/utils/data_module.py``
  - ``fMRIDataModule`` / ``FA3DDataModule`` / ``MultimodalDataModule``
- ``DSTFP/module/utils/data_preprocess_and_load/datasets.py``
"""

from __future__ import annotations

from typing import Any, Mapping


def build_datamodule(cfg: Mapping[str, Any]):
    name = str(cfg.get("dataset", {}).get("dataset_name", "Multimodal"))
    raise NotImplementedError(
        f"数据管线尚未接通（dataset_name={name!r}）。"
        "请对接 DSTFP/module/utils/data_module.py 后实现本函数。"
    )
