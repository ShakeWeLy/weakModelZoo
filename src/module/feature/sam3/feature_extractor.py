"""SAM3 预训练视觉特征提取：预处理 + ViT trunk（1024 维）或 ViTDet neck（256 维 FPN）。

典型用法::

    extractor = SAM3FeatureExtractor(
        mode="trunk",
        checkpoint_path="path/to/sam3.pt",
        freeze=True,
    )
    feats = extractor(images)  # images: [B,C,H,W]，默认 [0,1]
    bottleneck = feats.trunk    # [B, 1024, H//14, W//14]

接入 TransUNet / 分类头时，可对 ``trunk`` 做 1x1 投影或池化；``neck`` 模式返回多尺度 256 维特征。
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from sam3.model.vitdet import ViT

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SAM3_REPO_ROOT = _REPO_ROOT / "sam3"
_SAM3_PACKAGE_DIR = _SAM3_REPO_ROOT / "sam3"


def _bootstrap_sam3_imports() -> None:
    """挂载 sam3 包路径，并跳过 ``sam3/__init__.py``（避免拉取 model_builder / iopath）。"""
    if not _SAM3_PACKAGE_DIR.is_dir():
        raise ImportError(
            f"未找到 SAM3 源码目录 {_SAM3_PACKAGE_DIR}，请确认仓库内已包含 sam3 子模块。"
        )
    repo_root = str(_SAM3_REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if "sam3" not in sys.modules:
        stub = types.ModuleType("sam3")
        stub.__path__ = [str(_SAM3_PACKAGE_DIR)]  # type: ignore[attr-defined]
        sys.modules["sam3"] = stub


def _patch_sam3_vit_mlp() -> None:
    """避免 perflib ``addmm_act`` 在 CPU 上输出 bf16 导致后续 Linear dtype 不一致。"""
    mod = importlib.import_module("sam3.model.vitdet")
    if getattr(mod.Mlp, "_weakmodelzoo_patched", False):
        return

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, self.fc1.weight, self.fc1.bias)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = F.linear(x, self.fc2.weight, self.fc2.bias)
        x = self.drop2(x)
        return x

    mod.Mlp.forward = _mlp_forward  # type: ignore[method-assign]
    mod.Mlp._weakmodelzoo_patched = True  # type: ignore[attr-defined]


def _sam3_vit_class() -> type:
    _bootstrap_sam3_imports()
    mod = importlib.import_module("sam3.model.vitdet")
    _patch_sam3_vit_mlp()
    return mod.ViT


def _sam3_neck_class() -> type:
    _bootstrap_sam3_imports()
    mod = importlib.import_module("sam3.model.necks")
    return mod.Sam3DualViTDetNeck


def _sam3_position_encoding_class() -> type:
    _bootstrap_sam3_imports()
    mod = importlib.import_module("sam3.model.position_encoding")
    return mod.PositionEmbeddingSine

SAM3Mode = Literal["trunk", "neck", "both"]

# 整包 checkpoint 里 vision 部分常见前缀（按优先级尝试）
_VISION_KEY_PREFIXES: tuple[str, ...] = (
    "detector.backbone.vision_backbone.",
    "backbone.vision_backbone.",
)


def resolve_sam3_checkpoint_path(
    checkpoint_path: str | None,
    *,
    load_from_hf: bool = False,
    hf_version: Literal["sam3", "sam3.1"] = "sam3",
) -> str | None:
    if checkpoint_path is not None:
        return checkpoint_path
    if load_from_hf:
        _bootstrap_sam3_imports()
        builder = importlib.import_module("sam3.model_builder")
        return builder.download_ckpt_from_hf(version=hf_version)
    return None


def _unwrap_checkpoint(raw: object) -> dict[str, torch.Tensor]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"checkpoint 应为 dict，得到 {type(raw)}")
    sd = dict(raw)
    if "model" in sd and isinstance(sd["model"], Mapping):
        sd = dict(sd["model"])
    if "state_dict" in sd and isinstance(sd["state_dict"], Mapping):
        sd = dict(sd["state_dict"])
    return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}


def _extract_with_prefix(
    state_dict: Mapping[str, torch.Tensor],
    module_prefix: str,
) -> dict[str, torch.Tensor]:
    """从整包权重中截取 ``module_prefix`` 下的子树，并去掉该前缀。"""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(module_prefix):
            out[key[len(module_prefix) :]] = value
    return out


def _guess_trunk_prefix(state_dict: Mapping[str, torch.Tensor]) -> str | None:
    for base in _VISION_KEY_PREFIXES:
        probe = f"{base}trunk.patch_embed.proj.weight"
        if probe in state_dict:
            return f"{base}trunk."
    if "patch_embed.proj.weight" in state_dict:
        return ""
    return None


def _guess_neck_prefix(state_dict: Mapping[str, torch.Tensor]) -> str | None:
    for base in _VISION_KEY_PREFIXES:
        probe = f"{base}trunk.patch_embed.proj.weight"
        if probe in state_dict:
            return base
    if "trunk.patch_embed.proj.weight" in state_dict:
        return ""
    return None


def load_sam3_checkpoint_weights(
    module: nn.Module,
    checkpoint_path: str,
    *,
    component: Literal["trunk", "neck"] = "trunk",
    strict: bool | None = None,
) -> tuple[list[str], list[str]]:
    """将 SAM3 官方 checkpoint 中的 trunk 或 neck 权重载入已构建好的 ``module``。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _unwrap_checkpoint(ckpt)

    if component == "trunk":
        prefix = _guess_trunk_prefix(state_dict)
        if prefix is None:
            raise KeyError(
                "checkpoint 中未找到 ViT trunk 权重（期望含 "
                "'detector.backbone.vision_backbone.trunk.' 或已剥离的 patch_embed 键）"
            )
        sub = _extract_with_prefix(state_dict, prefix)
    else:
        prefix = _guess_neck_prefix(state_dict)
        if prefix is None:
            raise KeyError(
                "checkpoint 中未找到 vision_backbone 权重（期望含 "
                "'detector.backbone.vision_backbone.trunk.'）"
            )
        sub = _extract_with_prefix(state_dict, prefix)
        if not any(k.startswith("convs.") for k in sub):
            raise KeyError("neck 权重不完整：缺少 convs.*")

    if strict is None:
        strict = component == "trunk"
    missing, unexpected = module.load_state_dict(sub, strict=strict)
    return list(missing), list(unexpected)


class SAM3ImagePreprocessor(nn.Module):
    """与 SAM3 训练一致的 resize + normalize（默认 1008，mean/std=0.5）。"""

    def __init__(
        self,
        resolution: int = 1008,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
        input_range: Literal["0_1", "0_255"] = "0_1",
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.input_range = input_range
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def _to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c == 3:
            return x
        if c == 1:
            return x.repeat(1, 3, 1, 1)
        raise ValueError(f"SAM3 预处理器期望 1 或 3 通道，得到 in_channels={c}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"期望输入 [B,C,H,W]，得到 shape={tuple(x.shape)}")
        x = self._to_rgb(x.float())
        if self.input_range == "0_255":
            x = x / 255.0
        if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
            x = F.interpolate(
                x,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std


@dataclass
class SAM3Features:
    """``SAM3FeatureExtractor`` 的输出。"""

    trunk: torch.Tensor | None = None
    """末层 ViT 特征，形状 ``[B, 1024, H/14, W/14]``（H=W=resolution 时为 72×72）。"""

    neck_pyramid: list[torch.Tensor] | None = None
    """FPN 多尺度特征，每层 ``[B, 256, h_i, w_i]``，共 4 级。"""

    neck_last: torch.Tensor | None = None
    """``neck_pyramid`` 中分辨率最高的一级（与 SAM3 detector 常用的 ``vision_features`` 一致）。"""

    sam2_neck_pyramid: list[torch.Tensor] | None = None
    sam2_neck_last: torch.Tensor | None = None

    @property
    def primary(self) -> torch.Tensor:
        """默认给下游用的主特征：trunk 模式为 trunk，neck 模式为 neck_last。"""
        if self.trunk is not None:
            return self.trunk
        if self.neck_last is not None:
            return self.neck_last
        raise RuntimeError("SAM3Features 中 trunk 与 neck_last 均为空")


def _build_vit_trunk():
    """与 ``sam3.model_builder._create_vit_backbone`` 相同配置。"""
    ViT = _sam3_vit_class()
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=None,
        use_fa3=False,
        use_rope_real=False,
        use_act_checkpoint=False,
    )


def _build_vit_neck(*, enable_sam2_neck: bool):
    PositionEmbeddingSine = _sam3_position_encoding_class()
    Sam3DualViTDetNeck = _sam3_neck_class()
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )
    vit_backbone = _build_vit_trunk()
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_sam2_neck,
    )


class SAM3FeatureExtractor(nn.Module):
    """SAM3 视觉特征提取器。"""

    TRUNK_CHANNELS = 1024
    NECK_CHANNELS = 256
    PATCH_SIZE = 14

    def __init__(
        self,
        mode: SAM3Mode = "trunk",
        checkpoint_path: str | None = None,
        *,
        load_from_hf: bool = False,
        hf_version: Literal["sam3", "sam3.1"] = "sam3",
        resolution: int = 1008,
        freeze: bool = True,
        enable_sam2_neck: bool = False,
        input_range: Literal["0_1", "0_255"] = "0_1",
    ) -> None:
        super().__init__()
        if mode not in ("trunk", "neck", "both"):
            raise ValueError(f"mode 应为 trunk/neck/both，得到 {mode!r}")

        self.mode = mode
        self.resolution = resolution
        self.preprocessor = SAM3ImagePreprocessor(
            resolution=resolution,
            input_range=input_range,
        )

        self.trunk: nn.Module | None = None
        self.neck: nn.Module | None = None

        if mode in ("trunk", "both"):
            self.trunk = _build_vit_trunk().float()
        if mode in ("neck", "both"):
            self.neck = _build_vit_neck(enable_sam2_neck=enable_sam2_neck).float()

        ckpt = resolve_sam3_checkpoint_path(
            checkpoint_path,
            load_from_hf=load_from_hf,
            hf_version=hf_version,
        )
        if ckpt is not None:
            self.load_pretrained(ckpt)

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def load_pretrained(self, checkpoint_path: str) -> None:
        if self.trunk is not None:
            missing, unexpected = load_sam3_checkpoint_weights(
                self.trunk, checkpoint_path, component="trunk", strict=True
            )
            if unexpected:
                raise RuntimeError(f"加载 trunk 时出现 unexpected keys: {unexpected[:8]}...")
            if missing:
                raise RuntimeError(f"加载 trunk 缺失 keys: {missing[:8]}...")
        if self.neck is not None:
            load_sam3_checkpoint_weights(
                self.neck, checkpoint_path, component="neck", strict=False
            )

    @staticmethod
    def trunk_output_spatial_size(input_resolution: int, patch_size: int = PATCH_SIZE) -> int:
        if input_resolution % patch_size != 0:
            raise ValueError(
                f"resolution={input_resolution} 必须能被 patch_size={patch_size} 整除"
            )
        return input_resolution // patch_size

    def forward(
        self,
        images: torch.Tensor,
        *,
        skip_preprocess: bool = False,
        return_sam2_neck: bool = False,
    ) -> SAM3Features:
        # SAM3 perflib 融合 MLP 仅在 inference / no_grad 下可用
        grad_ctx = (
            torch.inference_mode()
            if (not self.training or not any(p.requires_grad for p in self.parameters()))
            else nullcontext()
        )
        with grad_ctx:
            return self._forward_features(
                images,
                skip_preprocess=skip_preprocess,
                return_sam2_neck=return_sam2_neck,
            )

    def _forward_features(
        self,
        images: torch.Tensor,
        *,
        skip_preprocess: bool,
        return_sam2_neck: bool,
    ) -> SAM3Features:
        x = images if skip_preprocess else self.preprocessor(images)
        x = x.float()

        trunk_feat: torch.Tensor | None = None
        neck_pyramid: list[torch.Tensor] | None = None
        neck_last: torch.Tensor | None = None
        sam2_pyramid: list[torch.Tensor] | None = None
        sam2_last: torch.Tensor | None = None

        if self.mode == "trunk":
            assert self.trunk is not None
            trunk_out = self.trunk(x)
            trunk_feat = trunk_out[-1]

        elif self.mode == "neck":
            assert self.neck is not None
            sam3_out, _, sam2_out, _ = self.neck(x)
            neck_pyramid = list(sam3_out)
            neck_last = sam3_out[-1]
            if return_sam2_neck and sam2_out is not None:
                sam2_pyramid = list(sam2_out)
                sam2_last = sam2_out[-1]

        else:  # both
            assert self.neck is not None and self.trunk is not None
            sam3_out, _, sam2_out, _ = self.neck(x)
            neck_pyramid = list(sam3_out)
            neck_last = sam3_out[-1]
            trunk_out = self.trunk(x)
            trunk_feat = trunk_out[-1]
            if return_sam2_neck and sam2_out is not None:
                sam2_pyramid = list(sam2_out)
                sam2_last = sam2_out[-1]

        return SAM3Features(
            trunk=trunk_feat,
            neck_pyramid=neck_pyramid,
            neck_last=neck_last,
            sam2_neck_pyramid=sam2_pyramid,
            sam2_neck_last=sam2_last,
        )


if __name__ == "__main__":
    dummy = torch.rand(2, 1, 256, 256)
    model = SAM3FeatureExtractor(mode="trunk", checkpoint_path=None, freeze=True)
    out = model(dummy)
    h = SAM3FeatureExtractor.trunk_output_spatial_size(model.resolution)
    assert out.trunk is not None
    print("trunk:", out.trunk.shape, f"(expected [2, 1024, {h}, {h}])")
