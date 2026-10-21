"""分割模型 Grad-CAM 核心工具。

与分类示例（``ClassifierOutputTarget``）不同：分割 logits 为 ``(B,C,H,W)``，
应对某一类通道（可选空间 mask）求标量再反传。

不依赖 ``pytorch_grad_cam``，仅需 PyTorch。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_module(model: nn.Module, dotted_name: str) -> nn.Module:
    """按 ``down4.conv`` / ``up1.conv.conv.4`` 解析子模块。"""
    module: nn.Module = model
    for part in dotted_name.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


@dataclass
class SegmentationCategoryTarget:
    """对 ``logits[:, category]``（可选乘 mask）求和，作为 Grad-CAM 的反传标量。"""

    category: int
    mask: torch.Tensor | None = None  # (H,W) 或 (1,H,W) / (B,H,W)，1=关注区域

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"期望分割 logits (B,C,H,W)，得到 {tuple(logits.shape)}")
        score = logits[:, self.category, ...]
        if self.mask is not None:
            m = self.mask.to(device=score.device, dtype=score.dtype)
            while m.ndim < score.ndim:
                m = m.unsqueeze(0)
            if m.shape[-2:] != score.shape[-2:]:
                m = F.interpolate(m.float(), size=score.shape[-2:], mode="nearest")
                if m.ndim == 4 and m.shape[1] == 1:
                    m = m[:, 0]
            score = score * m
        return score.sum()


class GradCAM:
    """单层 / 多层（特征相加）Grad-CAM。"""

    def __init__(self, model: nn.Module, target_layers: Sequence[nn.Module]) -> None:
        self.model = model
        self.target_layers = list(target_layers)
        self._activations: list[torch.Tensor] = []
        self._gradients: list[torch.Tensor] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer in self.target_layers:
            self._handles.append(layer.register_forward_hook(self._forward_hook))
            self._handles.append(layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module: nn.Module, inputs, output) -> None:
        if isinstance(output, torch.Tensor):
            self._activations.append(output)

    def _backward_hook(self, module: nn.Module, grad_input, grad_output) -> None:
        if grad_output and isinstance(grad_output[0], torch.Tensor):
            self._gradients.append(grad_output[0])

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @torch.enable_grad()
    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    ) -> np.ndarray:
        """返回 ``(B, H, W)`` 的灰度 CAM（已按样本归一化到 ``[0,1]``）。"""
        self.model.eval()
        self._activations.clear()
        self._gradients.clear()

        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("模型输出应为 Tensor logits")

        self.model.zero_grad(set_to_none=True)
        loss = sum(t(logits) for t in targets)
        loss.backward()

        if not self._activations or not self._gradients:
            raise RuntimeError("未捕获到激活或梯度，请检查 target_layers 是否在计算图中")

        # hook 顺序：forward 先进先出；backward 后进先出，对齐为同一层
        grads = list(reversed(self._gradients))
        acts = self._activations
        if len(grads) != len(acts):
            n = min(len(grads), len(acts))
            grads, acts = grads[:n], acts[:n]

        cams = []
        for act, grad in zip(acts, grads):
            # act/grad: (B,C,h,w)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1)  # (B,h,w)
            cam = F.relu(cam)
            cams.append(cam)

        cam = torch.stack(cams, dim=0).sum(dim=0)  # (B,h,w)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        cam_np = cam.detach().cpu().numpy()
        out = np.zeros_like(cam_np, dtype=np.float32)
        for i in range(cam_np.shape[0]):
            m = cam_np[i]
            m = m - m.min()
            denom = m.max() + 1e-8
            out[i] = (m / denom).astype(np.float32)
        return out


def tensor_to_rgb_float(image: torch.Tensor) -> np.ndarray:
    """``(C,H,W)`` 或 ``(1/3,H,W)`` → ``(H,W,3)`` float32 in ``[0,1]``。"""
    x = image.detach().cpu().float()
    if x.ndim != 3:
        raise ValueError(f"期望 (C,H,W)，得到 {tuple(x.shape)}")
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    elif x.shape[0] > 3:
        x = x[:3]
    x = x - x.min()
    denom = x.max() + 1e-8
    x = x / denom
    return x.permute(1, 2, 0).numpy().astype(np.float32)


def overlay_cam_on_image(
    rgb: np.ndarray,
    grayscale_cam: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """``rgb``/``cam`` → uint8 RGB 叠加图。"""
    import matplotlib.cm as cm

    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    heat = cm.jet(np.clip(grayscale_cam, 0, 1))[..., :3].astype(np.float32)
    blend = (1.0 - alpha) * rgb + alpha * heat
    return (np.clip(blend, 0, 1) * 255).astype(np.uint8)


def save_gradcam_figure(
    path: str | Path,
    *,
    rgb: np.ndarray,
    cam: np.ndarray,
    overlay: np.ndarray,
    title: str = "",
) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb if rgb.max() <= 1.5 else rgb / 255.0, cmap="gray")
    axes[0].set_title("input")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
