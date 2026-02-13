# train_UNext_multi_toy.py
# Multi-label segmentation training with UNeXt + BCE(pos_weight) + Soft Dice
# Fixes NHWC vs NCHW logits issue and logs debug stats.

import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image

# --- IMPORTANT: use your UNeXt implementation import here ---
from networks.UNeXt.A1218_UNeXt_binary_base import UNext


# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    img_dir: str = "toy_multiclass_dataset/images"
    mask_dir: str = "toy_multiclass_dataset/masks"

    # number of labels/classes
    num_classes: int = 9

    # mask format:
    # "labelmap": mask image where each pixel is an integer 0..C-1 (typical multi-class label map)
    # "onehot_npy": mask .npy array with shape (C,H,W) in {0,1}
    mask_format: str = "labelmap"

    size: int = 224
    val_ratio: float = 0.2
    seed: int = 42

    epochs: int = 10
    batch_size: int = 3
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2

    deep_supervision: bool = True

    out_dir: str = "runs/multi_unext"
    best_ckpt_name: str = "best.pt"
    last_ckpt_name: str = "last.pt"

    threshold: float = 0.5
    max_batches_posweight: int = 10


# -------------------------
# Utils
# -------------------------
def pick_main_logits(model_out) -> torch.Tensor:
    """Pick the main logits when deep supervision returns list/tuple."""
    if isinstance(model_out, (list, tuple)):
        logits = model_out[-1]
    else:
        logits = model_out
    return logits


def ensure_nchw(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Ensure logits are (B,C,H,W).
    Converts (B,H,W,C) -> (B,C,H,W) when needed.
    """
    if logits.ndim == 3:
        # (B,H,W) -> binary style -> (B,1,H,W)
        logits = logits.unsqueeze(1)

    if logits.ndim != 4:
        raise ValueError(f"Expected 4D logits, got shape {tuple(logits.shape)}")

    # already (B,C,H,W)
    if logits.size(1) == num_classes:
        return logits

    # channels-last (B,H,W,C)
    if logits.size(-1) == num_classes:
        return logits.permute(0, 3, 1, 2).contiguous()

    raise ValueError(
        f"Cannot infer channel dim for logits shape {tuple(logits.shape)} with num_classes={num_classes}"
    )


def one_hot_from_labelmap(mask_hw: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    mask_hw: (H,W) int64 with values in [0..C-1]
    return: (C,H,W) float one-hot
    """
    return torch.nn.functional.one_hot(mask_hw.long(), num_classes=num_classes).permute(2, 0, 1).float()


@torch.no_grad()
def multilabel_metrics_from_logits(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7
) -> Tuple[float, float, Dict[str, float]]:
    """
    logits/target: (B,C,H,W), target in {0,1}
    Returns macro Dice/IoU and debug stats.
    """
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    inter = (pred * target).sum(dim=(0, 2, 3))  # (C,)
    pred_sum = pred.sum(dim=(0, 2, 3))
    tgt_sum = target.sum(dim=(0, 2, 3))

    dice_c = (2 * inter + eps) / (pred_sum + tgt_sum + eps)  # (C,)
    union = pred_sum + tgt_sum - inter
    iou_c = (inter + eps) / (union + eps)

    present = (tgt_sum > 0)
    dice_macro = dice_c[present].mean().item() if present.any() else dice_c.mean().item()
    iou_macro = iou_c[present].mean().item() if present.any() else iou_c.mean().item()

    dbg = {
        "mask_pos_ratio_mean": float(target.mean().item()),
        "pred_pos_ratio_mean": float(pred.mean().item()),
        "probs_min": float(probs.min().item()),
        "probs_mean": float(probs.mean().item()),
        "probs_max": float(probs.max().item()),
    }
    return float(dice_macro), float(iou_macro), dbg


def estimate_pos_weight_per_class(
    loader: DataLoader, num_classes: int, device: str, max_batches: int = 10
) -> torch.Tensor:
    """
    pos_weight[c] = neg_c / pos_c estimated over a few batches.
    target expected (B,C,H,W) in {0,1}
    """
    pos = torch.zeros(num_classes, device=device)
    neg = torch.zeros(num_classes, device=device)

    seen = 0
    for _, target, _ in loader:
        target = target.to(device, non_blocking=True)
        pos += target.sum(dim=(0, 2, 3))
        neg += (1.0 - target).sum(dim=(0, 2, 3))
        seen += 1
        if seen >= max_batches:
            break

    pos = torch.clamp(pos, min=1.0)
    w = neg / pos
    return w


def save_ckpt(path: str, model, optimizer, epoch: int, best_metric: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_metric": best_metric},
        path,
    )


# -------------------------
# Dataset
# -------------------------
class MultiSegDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        num_classes: int,
        size: int = 224,
        augment: bool = False,
        mask_format: str = "labelmap",
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.size = size
        self.augment = augment
        self.mask_format = mask_format

        self.names = sorted(os.listdir(img_dir))

        self.img_base = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def __len__(self):
        return len(self.names)

    def _augment_pair(self, img: Image.Image, mask_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
            mask_img = TF.hflip(mask_img)
        if torch.rand(1).item() < 0.5:
            img = TF.vflip(img)
            mask_img = TF.vflip(mask_img)
        if torch.rand(1).item() < 0.3:
            angle = float(torch.empty(1).uniform_(-15, 15).item())
            img = TF.rotate(img, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask_img = TF.rotate(mask_img, angle=angle, interpolation=TF.InterpolationMode.NEAREST)
        return img, mask_img

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = Image.open(img_path).convert("RGB")

        if self.mask_format == "labelmap":
            mask_img = Image.open(mask_path)  # keep palette/labels
            if self.augment:
                img, mask_img = self._augment_pair(img, mask_img)

            img = self.img_base(img)

            mask_img = TF.resize(mask_img, (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)
            mask_np = np.array(mask_img, dtype=np.int64)  # (H,W) with labels
            mask_hw = torch.from_numpy(mask_np)  # (H,W)
            target = one_hot_from_labelmap(mask_hw, self.num_classes)  # (C,H,W)

        elif self.mask_format == "onehot_npy":
            base, _ = os.path.splitext(mask_path)
            mask_path = base + ".npy"
            onehot = np.load(mask_path)  # (C,H,W)
            target = torch.from_numpy(onehot).float()

            # NOTE: augmentation not implemented safely for onehot here
            img = self.img_base(img)

            target = torch.nn.functional.interpolate(
                target.unsqueeze(0), size=(self.size, self.size), mode="nearest"
            ).squeeze(0)

        else:
            raise ValueError(f"Unknown mask_format: {self.mask_format}")

        return img, target, name


# -------------------------
# Losses
# -------------------------
class SoftDiceLossMulti(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # probs/target: (B,C,H,W)
        inter = (probs * target).sum(dim=(2, 3))               # (B,C)
        denom = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) # (B,C)
        dice = (2 * inter + self.eps) / (denom + self.eps)     # (B,C)
        return 1.0 - dice.mean()


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(
    model, loader, optimizer, device, bce_loss: nn.Module, num_classes: int
) -> float:
    model.train()
    dice_loss = SoftDiceLossMulti()

    total = 0.0
    n = 0

    for imgs, target, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)  # (B,C,H,W)

        optimizer.zero_grad(set_to_none=True)

        out = model(imgs)
        logits = pick_main_logits(out)
        logits = ensure_nchw(logits, num_classes=num_classes)

        # Safety: ensure target is (B,C,H,W)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        loss_bce = bce_loss(logits, target)
        probs = torch.sigmoid(logits)
        loss_d = dice_loss(probs, target)
        loss = loss_bce + loss_d

        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model, loader, device, bce_loss: nn.Module, threshold: float, num_classes: int
) -> Dict[str, float]:
    model.eval()
    dice_loss = SoftDiceLossMulti()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0

    first_dbg = None

    for imgs, target, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        out = model(imgs)
        logits = pick_main_logits(out)
        logits = ensure_nchw(logits, num_classes=num_classes)

        if first_dbg is None:
            d, j, dbg = multilabel_metrics_from_logits(logits, target, threshold=threshold)
            first_dbg = dbg
        else:
            d, j, _ = multilabel_metrics_from_logits(logits, target, threshold=threshold)

        loss_bce = bce_loss(logits, target)
        probs = torch.sigmoid(logits)
        loss_d = dice_loss(probs, target)
        loss = loss_bce + loss_d

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_bce += loss_bce.item() * bs
        total_dice_loss += loss_d.item() * bs
        total_dice += d * bs
        total_iou += j * bs
        n += bs

    n = max(n, 1)
    out = {
        "loss": total_loss / n,
        "bce": total_bce / n,
        "dice_loss": total_dice_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
    }
    if first_dbg is not None:
        out.update({f"dbg_{k}": v for k, v in first_dbg.items()})
    return out


def run():
    cfg = CFG()
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    full = MultiSegDataset(
        cfg.img_dir, cfg.mask_dir, cfg.num_classes, size=cfg.size,
        augment=True, mask_format=cfg.mask_format
    )

    n_total = len(full)
    n_val = int(round(cfg.val_ratio * n_total))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed)
    )
    # no augmentation for val
    val_ds.dataset.augment = False

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # Estimate per-class pos_weight to fight degenerate solutions
    pos_weight = estimate_pos_weight_per_class(
        train_dl, num_classes=cfg.num_classes, device=device, max_batches=cfg.max_batches_posweight
    )
    print("Estimated pos_weight per class:", pos_weight.detach().cpu().numpy())

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = UNext(num_classes=cfg.num_classes, input_channels=3, deep_supervision=cfg.deep_supervision).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_dice = -1.0

    print(f"Device: {device}")
    print(f"Train: {n_train} | Val: {n_val}")
    print(f"Output: {cfg.out_dir}")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_dl, optimizer, device, bce_loss=bce_loss, num_classes=cfg.num_classes
        )
        val = evaluate(
            model, val_dl, device, bce_loss=bce_loss,
            threshold=cfg.threshold, num_classes=cfg.num_classes
        )

        scheduler.step(val["dice"])
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"[{epoch:03d}/{cfg.epochs}] lr={lr_now:.2e} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={val['loss']:.4f} (bce={val['bce']:.4f}, diceL={val['dice_loss']:.4f}) | "
            f"dice={val['dice']:.4f} iou={val['iou']:.4f} | "
            f"mask+={val.get('dbg_mask_pos_ratio_mean', float('nan')):.3f} "
            f"pred+={val.get('dbg_pred_pos_ratio_mean', float('nan')):.3f} "
            f"p[min/mean/max]={val.get('dbg_probs_min', float('nan')):.3f}/"
            f"{val.get('dbg_probs_mean', float('nan')):.3f}/"
            f"{val.get('dbg_probs_max', float('nan')):.3f}"
        )

        save_ckpt(os.path.join(cfg.out_dir, cfg.last_ckpt_name), model, optimizer, epoch, best_dice)

        if val["dice"] > best_dice:
            best_dice = val["dice"]
            save_ckpt(os.path.join(cfg.out_dir, cfg.best_ckpt_name), model, optimizer, epoch, best_dice)
            print(f"  -> best updated: dice={best_dice:.4f}")

    print(f"Done. Best Dice: {best_dice:.4f}")
    print("Best checkpoint:", os.path.join(cfg.out_dir, cfg.best_ckpt_name))


if __name__ == "__main__":
    run()
