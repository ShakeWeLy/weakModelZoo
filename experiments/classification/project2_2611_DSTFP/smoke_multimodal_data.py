"""Multimodal 数据冒烟：PPMI_DSTFP 的 .pt → 20-TR 窗口 + DataLoader（不依赖 sklearn/pandas）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


WMZ = Path(__file__).resolve().parents[3]
DEFAULT_DATA = WMZ / "data" / "PPMI_DSTFP"
SEQ_LEN = 20
PAD = (10, 9, 4, 3, 10, 9)


def _pad_fmri(tensor: torch.Tensor) -> torch.Tensor:
    """(1, D, H, W, T) → pad 至 80³，对齐 DSTFP MultimodalDataset._pad_to_size。"""
    bg = tensor.flatten()[0].item()
    t = tensor.permute(0, 4, 1, 2, 3)
    t = torch.nn.functional.pad(t, PAD, value=bg)
    return t.permute(0, 2, 3, 4, 1)


def _pad_fa(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)
    bg = tensor[0, 0, 0, 0, 0].item()
    out = torch.nn.functional.pad(tensor, PAD, value=bg)
    return out.squeeze(0).float()


class PPMIMultimodalSmoke(Dataset):
    def __init__(self, data_root: Path, manifest: list[dict], stride_between: int = 10):
        self.samples: list[dict] = []
        for row in manifest:
            g, sid, lab = row["group"], row["subject_id"], int(row["label"])
            fmri_dir = data_root / f"{g}_MNI_to_TRs_minmax" / sid
            fa_pt = data_root / f"{g}_FA_processed" / "img" / sid / "fa.pt"
            n = row.get("n_tr") or len(list(fmri_dir.glob("frame_*.pt")))
            dur = SEQ_LEN
            for start in range(0, max(n - dur + 1, 0), stride_between):
                self.samples.append(
                    {"fmri_dir": fmri_dir, "fa_pt": fa_pt, "start": start, "label": lab, "key": f"{g}_{sid}"}
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        frames = []
        for i in range(s["start"], s["start"] + SEQ_LEN):
            tr = torch.load(s["fmri_dir"] / f"frame_{i}.pt", weights_only=True)
            frames.append(tr.unsqueeze(0).unsqueeze(-1))
        y = torch.cat(frames, dim=4)
        y = _pad_fmri(y)
        fa = _pad_fa(torch.load(s["fa_pt"], weights_only=True))
        return {"fmri_sequence": y, "dti_sequence": fa, "target": s["label"], "subject_name": s["key"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    manifest_path = args.data / "manifest.json"
    if not manifest_path.is_file():
        print(f"缺少 manifest: {manifest_path}")
        print("请先: python F:/MRI/project_5/scripts/export_dstfp_weakmodel.py")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ds = PPMIMultimodalSmoke(args.data, manifest)
    print(f"被试 {len(manifest)} 人，20-TR 序列样本 {len(ds)}")
    if len(ds) == 0:
        return 1

    sample = ds[0]
    print(f"sample0 fmri {tuple(sample['fmri_sequence'].shape)} fa {tuple(sample['dti_sequence'].shape)} label={sample['target']}")

    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"batch fmri {tuple(batch['fmri_sequence'].shape)} fa {tuple(batch['dti_sequence'].shape)}")

    models_dir = WMZ / "DSTFP" / "module" / "models"
    if not models_dir.is_dir():
        print(f"[WARN] DSTFP 网络未就位: {models_dir}")
        print("       官方 GitHub MAOmgg/DSTFP 的 module/ 下同样无 models/，完整训练需补齐 Swin4D/Swin3D/fusion")
    else:
        print("models/ 存在，可继续接 LitMultimodalClassifier 训练")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())