import os
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from networks.UNeXt.A1218_UNeXt_binary_base import UNext
from metrics.binary_metrics import compute_binary_metrics


# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    # data
    img_dir: str = "toy_binary_dataset/images"
    mask_dir: str = "toy_binary_dataset/masks"
    size: int = 224

    # split
    val_ratio: float = 0.2
    seed: int = 42

    # train
    epochs: int = 10
    batch_size: int = 3
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2

    # model
    deep_supervision: bool = True  # si True, on gère les sorties multiples

    # output
    out_dir: str = "runs/binary_unext"
    best_ckpt_name: str = "best.pt"
    last_ckpt_name: str = "last.pt"

    # threshold for metrics
    threshold: float = 0.5


# -------------------------
# Dataset (binaire)
# -------------------------
class BinarySegDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, size: int = 224, augment: bool = False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment

        self.names = sorted(os.listdir(img_dir))

        # base transforms
        self.img_base = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),  # [0,1]
        ])
        self.mask_base = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.names)

    def _augment_pair(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # mêmes augmentations sur image + mask
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if torch.rand(1).item() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # rotation légère (mask en nearest implicite car on applique sur PIL)
        if torch.rand(1).item() < 0.3:
            angle = float(torch.empty(1).uniform_(-15, 15).item())
            img = TF.rotate(img, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle=angle, interpolation=TF.InterpolationMode.NEAREST)

        return img, mask

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            img, mask = self._augment_pair(img, mask)

        img = self.img_base(img)            # (3,H,W)
        mask = self.mask_base(mask)         # (1,H,W) in [0,1] approx
        mask = (mask > 0.5).float()         # force {0,1}

        return img, mask, name


# -------------------------
# Loss: BCE + Dice (soft)
# -------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # probs/target: (B,1,H,W) in [0,1] / {0,1}
        probs = probs.contiguous()
        target = target.contiguous()

        inter = (probs * target).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


def pick_main_logits(model_out) -> torch.Tensor:
    """
    Gère les cas où UNeXt renvoie:
      - tensor (B,1,H,W) ou (B,H,W)
      - liste/tuple de tensors (deep supervision)
    On prend la sortie "principale" = dernière par défaut.
    """
    if isinstance(model_out, (list, tuple)):
        logits = model_out[-1]
    else:
        logits = model_out

    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    return logits


# -------------------------
# Train / Eval loops
# -------------------------
def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    bce = nn.BCEWithLogitsLoss()
    dice = SoftDiceLoss()

    total = 0.0
    n = 0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out = model(imgs)
        logits = pick_main_logits(out)

        # Loss = BCE(logits) + Dice(sigmoid(logits))
        loss_bce = bce(logits, masks)
        probs = torch.sigmoid(logits)
        loss_dice = dice(probs, masks)
        loss = loss_bce + loss_dice

        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    dice_loss = SoftDiceLoss()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        out = model(imgs)
        logits = pick_main_logits(out)
        
        if n == 0:  # first batch only
            print("logits shape:", logits.shape)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            print("mask pos ratio:", masks.mean().item())
            print("pred pos ratio:", pred.mean().item())
            print("probs min/mean/max:",
                probs.min().item(), probs.mean().item(), probs.max().item())

        loss_bce = bce(logits, masks)
        probs = torch.sigmoid(logits)
        loss_d = dice_loss(probs, masks)
        loss = loss_bce + loss_d

        m = compute_binary_metrics(logits=logits, target=masks, threshold=threshold)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_bce += loss_bce.item() * bs
        total_dice_loss += loss_d.item() * bs
        total_dice += m["dice"] * bs
        total_iou += m["iou"] * bs
        n += bs

    n = max(n, 1)
    if n == 0:  # first batch only
        print("logits shape:", logits.shape)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
        print("mask pos ratio:", masks.mean().item())
        print("pred pos ratio:", pred.mean().item())
        print("probs min/mean/max:",
            probs.min().item(), probs.mean().item(), probs.max().item())

    return {
        "loss": total_loss / n,
        "bce": total_bce / n,
        "dice_loss": total_dice_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
    }


def save_ckpt(path: str, model, optimizer, epoch: int, best_metric: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
        },
        path,
    )


def run():
    cfg = CFG()
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    # Dataset + split
    full = BinarySegDataset(cfg.img_dir, cfg.mask_dir, size=cfg.size, augment=True)
    n_total = len(full)
    n_val = int(round(cfg.val_ratio * n_total))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # IMPORTANT: val sans augment
    # random_split garde la même classe, donc on force augment=False côté dataset sous-jacent
    # => on remplace le dataset de val par une version sans augment (mêmes fichiers)
    val_ds.dataset.augment = False

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = UNext(num_classes=1, input_channels=3, deep_supervision=cfg.deep_supervision).to(device)

    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_dice = -1.0

    print(f"Device: {device}")
    print(f"Train: {n_train} | Val: {n_val}")
    print(f"Output: {cfg.out_dir}")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_dl, optimizer, device)
        val = evaluate(model, val_dl, device, threshold=cfg.threshold)

        # step scheduler on val dice
        scheduler.step(val["dice"])

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"lr={lr_now:.2e} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={val['loss']:.4f} (bce={val['bce']:.4f}, diceL={val['dice_loss']:.4f}) | "
            f"dice={val['dice']:.4f} iou={val['iou']:.4f}"
        )

        # save last
        save_ckpt(os.path.join(cfg.out_dir, cfg.last_ckpt_name), model, optimizer, epoch, best_dice)

        # save best
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            save_ckpt(os.path.join(cfg.out_dir, cfg.best_ckpt_name), model, optimizer, epoch, best_dice)
            print(f"  -> best updated: dice={best_dice:.4f}")

    print(f"Done. Best Dice: {best_dice:.4f}")
    print("Best checkpoint:", os.path.join(cfg.out_dir, cfg.best_ckpt_name))


if __name__ == "__main__":
    run()