"""SiamSPN++ 在 Anti-UAV 上训练（默认加载 MMTracking R50 SiamRPN 整网预训练）。

用法（仓库根目录）::

    # 冒烟
    python experiments/track/siamspn/train.py --smoke

    # 正式训练
    python experiments/track/siamspn/train.py --data /path/to/Anti-UAV --modality ir \\
        --epochs 20 --batch-size 8 --freeze-backbone-epochs 10

说明：
- 默认权重：``pretrained_weights/siamprnpp/siamese_rpn_r50_1x_lasot_*.pth``
- 前 ``freeze_backbone_epochs`` 个 epoch 冻结 ResNet，只训 neck/RPN
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.track.SiamSPN import SiamSPNLoss, SiamSPN_pp  # noqa: E402
from src.models.track.SiamSPN.load_mmtrack import DEFAULT_MMTRACK_WEIGHTS  # noqa: E402
from src.utils.data.anti_uav import AntiUAVPairDataset, collate_anti_uav  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=str, default="", help="Anti-UAV 根目录")
    p.add_argument("--modality", choices=["ir", "rgb", "auto"], default="ir")
    p.add_argument("--splits", type=str, default="train", help="逗号分隔，如 train 或 train,val")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pairs-per-epoch", type=int, default=60000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1, help="解冻后主干 lr = lr * mult")
    p.add_argument("--freeze-backbone-epochs", type=int, default=10)
    p.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_MMTRACK_WEIGHTS),
        help="MMTracking R50 SiamRPN 整网权重路径",
    )
    p.add_argument("--no-pretrained", action="store_true", help="不加载预训练（随机初始化）")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="experiments/track/siamspn/runs")
    p.add_argument("--name", type=str, default="siamspn_pp")
    p.add_argument("--print-freq", type=int, default=50)
    p.add_argument("--smoke", action="store_true", help="合成 batch 冒烟，不读真实数据")
    return p.parse_args()


def _build_optimizer(model: SiamSPN_pp, lr: float, backbone_lr_mult: float, train_backbone: bool):
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in list(model.neck.parameters()) + list(model.rpn_head.parameters()) if p.requires_grad]
    groups = [{"params": head_params, "lr": lr}]
    if train_backbone and bb_params:
        groups.append({"params": bb_params, "lr": lr * backbone_lr_mult})
    return torch.optim.SGD(groups, momentum=0.9, weight_decay=1e-4)


def _build_model(pretrained: bool, weights: str | None) -> SiamSPN_pp:
    if pretrained:
        return SiamSPN_pp(pretrained=True, weights_path=weights)
    return SiamSPN_pp(pretrained=False)


def _smoke(device: torch.device, pretrained: bool, weights: str | None) -> None:
    model = _build_model(pretrained, weights).to(device)
    criterion = SiamSPNLoss()
    model.train()
    model.freeze_backbone(True)
    z = torch.randn(2, 3, 127, 127, device=device)
    x = torch.randn(2, 3, 255, 255, device=device)
    out = model(z, x)
    b, _, h, w = out["cls"].shape
    a = model.anchor_num
    label_cls = torch.zeros(b, a, h, w, dtype=torch.long, device=device)
    label_cls[:, 0, h // 2, w // 2] = 1
    label_loc = torch.zeros(b, 4, a, h, w, device=device)
    label_w = torch.zeros(b, a, h, w, device=device)
    label_w[:, 0, h // 2, w // 2] = 1
    losses = criterion(out["cls"], out["loc"], label_cls, label_loc, label_w)
    losses["loss"].backward()
    print(f"[smoke] ok loss={float(losses['loss']):.4f} cls={tuple(out['cls'].shape)}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    pretrained = not args.no_pretrained
    weights = None if args.no_pretrained else (args.weights or None)

    if args.smoke:
        _smoke(device, pretrained, weights)
        return

    if not args.data:
        raise SystemExit("请指定 --data /path/to/Anti-UAV，或使用 --smoke")

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    dataset = AntiUAVPairDataset(
        args.data,
        modality=args.modality,  # type: ignore[arg-type]
        splits=splits,
        pairs_per_epoch=args.pairs_per_epoch,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_anti_uav,
        drop_last=True,
    )
    print(f"[data] sequences={len(dataset.sequences)} pairs/epoch={len(dataset)}")

    model = _build_model(pretrained, weights).to(device)
    criterion = SiamSPNLoss()
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_bb = epoch >= args.freeze_backbone_epochs
        model.freeze_backbone(not train_bb)
        model.train()
        if not train_bb:
            model.backbone.eval()
        optimizer = _build_optimizer(model, args.lr, args.backbone_lr_mult, train_bb)

        t0 = time.time()
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            template = batch["template"].to(device, non_blocking=True)
            search = batch["search"].to(device, non_blocking=True)
            label_cls = batch["label_cls"].to(device, non_blocking=True)
            label_loc = batch["label_loc"].to(device, non_blocking=True)
            label_w = batch["label_loc_weight"].to(device, non_blocking=True)

            out = model(template, search)
            losses = criterion(out["cls"], out["loc"], label_cls, label_loc, label_w)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running += float(losses["loss"])
            if step % args.print_freq == 0 or step == 1:
                avg = running / step
                print(
                    f"epoch {epoch+1}/{args.epochs} step {step}/{len(loader)} "
                    f"loss={avg:.4f} cls={float(losses['cls']):.4f} loc={float(losses['loc']):.4f} "
                    f"bb={'on' if train_bb else 'freeze'}"
                )

        ckpt = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "args": vars(args),
        }
        path = out_dir / f"checkpoint_e{epoch+1}.pth"
        torch.save(ckpt, path)
        print(f"[save] {path}  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
