"""PPMI + src/models/classification/DSTFP 训练（可调规模 + loss 曲线）。

  python experiments/classification/project2_2611_DSTFP/train_minimal.py --device cuda
  python experiments/classification/project2_2611_DSTFP/train_minimal.py --profile medium --epochs 10 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

WMZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WMZ))

from src.models.classification.DSTFP.DSTFP import DSTFP  # noqa: E402

DEFAULT_DATA = WMZ / "data" / "PPMI_DSTFP"
PAD = (10, 9, 4, 3, 10, 9)


class ModalStem(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fmri_proj = nn.Conv3d(1, embed_dim, kernel_size=1)
        self.fa_proj = nn.Conv3d(1, embed_dim, kernel_size=1)

    def forward_fmri(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, D, H, W, T]
        b, _, d, h, w, t = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(b * t, 1, d, h, w)
        x = self.fmri_proj(x)
        e = x.shape[1]
        return x.view(b, t, e, d, h, w).permute(0, 2, 3, 4, 5, 1).contiguous()

    def forward_fa(self, x: torch.Tensor) -> torch.Tensor:
        return self.fa_proj(x)


class DSTFPTrunc(DSTFP):
    """裁剪 Swin stage 数以控制算力。"""

    def __init__(self, *args, num_stages: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        n = max(1, min(num_stages, len(self.fmri_swin_transformer)))
        self.fmri_swin_transformer = nn.ModuleList(self.fmri_swin_transformer[:n])
        self.dti_swin_transformer = nn.ModuleList(self.dti_swin_transformer[:n])


def _pad_fmri(tensor: torch.Tensor) -> torch.Tensor:
    bg = tensor.flatten()[0].item()
    t = tensor.permute(0, 4, 1, 2, 3)
    t = F.pad(t, PAD, value=bg)
    return t.permute(0, 2, 3, 4, 1)


def _pad_fa(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)
    bg = tensor[0, 0, 0, 0, 0].item()
    return F.pad(tensor, PAD, value=bg).squeeze(0).float()


class MinimalPPMIDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        manifest: list[dict],
        *,
        seq_len: int = 4,
        max_subjects: int = 0,
        spatial: int = 8,
        window_stride: int = 0,
    ):
        self.data_root = data_root
        self.seq_len = seq_len
        self.spatial = spatial
        self.items: list[dict] = []
        seen: set[str] = set()
        for row in manifest:
            if max_subjects > 0 and len(seen) >= max_subjects:
                break
            g, sid = row["group"], row["subject_id"]
            key = f"{g}/{sid}"
            if key in seen:
                continue
            fmri_dir = data_root / f"{g}_MNI_to_TRs_minmax" / sid
            fa_pt = data_root / f"{g}_FA_processed" / "img" / sid / "fa.pt"
            if not fmri_dir.is_dir() or not fa_pt.is_file():
                continue
            seen.add(key)
            n_tr = int(row.get("n_tr") or len(list(fmri_dir.glob("frame_*.pt"))))
            stride = window_stride if window_stride > 0 else max(n_tr, 1)
            for start in range(0, max(n_tr - seq_len + 1, 1), stride):
                if start + seq_len > n_tr:
                    break
                self.items.append(
                    {
                        "fmri_dir": fmri_dir,
                        "fa_pt": fa_pt,
                        "label": int(row["label"]),
                        "key": f"{g}_{sid}",
                        "start": start,
                    }
                )
                if window_stride <= 0:
                    break

    def __len__(self) -> int:
        return len(self.items)

    def _pool_fmri(self, y: torch.Tensor) -> torch.Tensor:
        # y: [1, D, H, W, T]  -> 返回 [C=1, D, H, W, T]
        y = y.squeeze(0).permute(3, 0, 1, 2)
        y = F.adaptive_avg_pool3d(y.unsqueeze(0), self.spatial).squeeze(0)
        y = y.permute(1, 2, 3, 0)
        return y.unsqueeze(0)

    def _pool_fa(self, fa: torch.Tensor) -> torch.Tensor:
        while fa.dim() > 3:
            fa = fa.squeeze(0)
        fa = F.adaptive_avg_pool3d(fa.unsqueeze(0), self.spatial).squeeze(0)
        return fa.unsqueeze(0)

    def __getitem__(self, idx: int) -> dict:
        s = self.items[idx]
        frames = []
        start = int(s.get("start", 0))
        for i in range(start, start + self.seq_len):
            tr = torch.load(s["fmri_dir"] / f"frame_{i}.pt", weights_only=True)
            frames.append(tr.unsqueeze(0).unsqueeze(-1))
        y = torch.cat(frames, dim=4)
        y = _pad_fmri(y)
        y = self._pool_fmri(y)
        fa = _pad_fa(torch.load(s["fa_pt"], weights_only=True))
        fa = self._pool_fa(fa)
        return {
            "fmri_sequence": y.float(),
            "dti_sequence": fa.float(),
            "target": s["label"],
            "subject_name": s["key"],
        }


def save_loss_plot(
    run_dir: Path,
    step_losses: list[float],
    epoch_means: list[float],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(1, len(step_losses) + 1), step_losses, linewidth=1, alpha=0.85)
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("loss")
    ax[0].set_title("train loss (per step)")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(range(1, len(epoch_means) + 1), epoch_means, marker="o", linewidth=1.5)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("mean loss")
    ax[1].set_title("train loss (per epoch)")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = run_dir / "loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def apply_profile(args: argparse.Namespace) -> None:
    if args.profile == "medium":
        args.spatial = 16
        args.seq_len = 12
        args.embed_dim = 32
        args.window_size = 4
        args.num_stages = 2
        args.max_subjects = 16
        args.window_stride = 40
        if args.epochs == 1:
            args.epochs = 10


def pick_device(prefer: str) -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--profile", choices=("tiny", "medium", "none"), default="none")
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--num-stages", type=int, default=1)
    ap.add_argument("--window-size", type=int, default=4)
    ap.add_argument("--spatial", type=int, default=8, help="池化后的 D=H=W")
    ap.add_argument("--seq-len", type=int, default=4)
    ap.add_argument("--max-subjects", type=int, default=0, help="0=全部 manifest 被试")
    ap.add_argument("--window-stride", type=int, default=0, help=">0 时每人间多段 20-TR 窗")
    ap.add_argument("--max-steps", type=int, default=0, help="0=跑满所有 epoch")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    if args.profile != "none":
        apply_profile(args)

    manifest_path = args.data / "manifest.json"
    if not manifest_path.is_file():
        print(f"缺少 {manifest_path}，先运行 export_dstfp_weakmodel.py")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ds = MinimalPPMIDataset(
        args.data,
        manifest,
        seq_len=args.seq_len,
        max_subjects=args.max_subjects,
        spatial=args.spatial,
        window_stride=args.window_stride,
    )
    if len(ds) == 0:
        print("无可用样本")
        return 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = pick_device(args.device)
    n_heads = max(2, args.embed_dim // 8)
    print(
        f"device={device} 样本={len(ds)}  spatial={args.spatial}³  TR={args.seq_len} "
        f"embed={args.embed_dim} stages={args.num_stages}"
    )

    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    stem = ModalStem(args.embed_dim).to(device)
    model = DSTFPTrunc(
        input_dim=args.embed_dim,
        embed_dim=args.embed_dim,
        num_heads=n_heads,
        window_size=args.window_size,
        num_stages=args.num_stages,
    ).to(device)
    opt = torch.optim.AdamW(list(stem.parameters()) + list(model.parameters()), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    stem.train()
    step = 0
    total_loss = 0.0
    step_losses: list[float] = []
    epoch_means: list[float] = []

    step_cap = args.max_steps if args.max_steps > 0 else None

    for ep in range(args.epochs):
        ep_loss = 0.0
        ep_steps = 0
        for batch in loader:
            if step_cap is not None and step >= step_cap:
                break
            fmri = batch["fmri_sequence"].float().to(device)
            fa = batch["dti_sequence"].float().to(device)
            y = batch["target"].to(device)

            fmri = stem.forward_fmri(fmri)
            fa = stem.forward_fa(fa)
            logits = model(fmri, fa)
            loss = crit(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            ep_steps += 1
            lv = float(loss.item())
            total_loss += lv
            ep_loss += lv
            step_losses.append(lv)
            print(f"epoch={ep+1} step={step} loss={lv:.4f} pred={logits.argmax(1).tolist()} label={y.tolist()}")

        if ep_steps:
            em = ep_loss / ep_steps
            epoch_means.append(em)
            print(f"--- epoch {ep+1} mean_loss={em:.4f} ({ep_steps} steps) ---")
        if step_cap is not None and step >= step_cap:
            break

    tag = "gpu" if device.type == "cuda" else "cpu"
    run_dir = Path(__file__).resolve().parent / "runs" / "DSTFP" / f"minimal_ppmi_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "minimal_last.pth"
    loss_json = run_dir / "loss_history.json"
    loss_hist = {"step_loss": step_losses, "epoch_mean": epoch_means}
    loss_json.write_text(json.dumps(loss_hist, indent=2), encoding="utf-8")
    plot_path = save_loss_plot(run_dir, step_losses, epoch_means)

    torch.save(
        {
            "stem": stem.state_dict(),
            "model": model.state_dict(),
            "args": vars(args),
            "mean_loss": total_loss / max(step, 1),
            "loss_history": loss_hist,
            "time": datetime.now().isoformat(timespec="seconds"),
        },
        ckpt,
    )
    print(f"完成 {step} step，mean_loss={total_loss / max(step, 1):.4f}")
    print(f"checkpoint: {ckpt}")
    print(f"loss 图: {plot_path}")
    print(f"loss 数据: {loss_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())