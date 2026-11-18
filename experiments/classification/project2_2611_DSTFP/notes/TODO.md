# DSTFP 实验待办（占位）

1. 补齐上游 `DSTFP/module/models/`（`swin4d_transformer_ver7.py` / `swin3d_transformer.py` / `fusion_H_W.py` / `load_model.py`）
2. 接通 `trainer/models.py::build_model` 与 `trainer/data.py::build_datamodule`
3. 实现 `PlaceholderTrainer` → 真实训练循环（或包装 Lightning `LitMultimodalClassifier`）
4. 写入真实数据路径到 `configs/global.toml` 的 `[dataset]`
5. 用 `--quick` 冒烟后，正式跑 `001_DSTFP_V1`；复跑自动升为 `_V2`
