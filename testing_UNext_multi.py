import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch import nn

from networks.UNeXt.A1218_UNeXt_multilabel_base import UNext 

@torch.no_grad()
def pixel_accuracy(pred, target, ignore_index=None, eps=1e-7):
    # pred/target: (B,H,W)
    if ignore_index is None:
        correct = (pred == target).sum().item()
        total = target.numel()
    else:
        valid = (target != ignore_index)
        correct = ((pred == target) & valid).sum().item()
        total = valid.sum().item()
    return (correct + eps) / (total + eps)

@torch.no_grad()
def f1_scores(pred, target, num_classes, ignore_index=None, eps=1e-7):
    """
    Returns:
      f1_macro: mean over classes (excluding ignore_index if set)
      f1_micro: global over all classes (excluding ignore_index pixels if set)
      f1_per_class: list of per-class f1 (None for ignored class)
    """
    # mask out ignored pixels
    if ignore_index is not None:
        valid = (target != ignore_index)
        pred = pred[valid]
        target = target[valid]

    # if nothing valid
    if target.numel() == 0:
        return 0.0, 0.0, [None] * num_classes

    f1_per_class = [None] * num_classes

    # Per-class F1 (one-vs-rest)
    f1_list = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue

        pred_c = (pred == c)
        tgt_c = (target == c)

        tp = (pred_c & tgt_c).sum().item()
        fp = (pred_c & (~tgt_c)).sum().item()
        fn = ((~pred_c) & tgt_c).sum().item()

        f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        f1_per_class[c] = float(f1)
        f1_list.append(float(f1))

    f1_macro = float(np.mean(f1_list)) if len(f1_list) else 0.0

    # Micro-F1: sum TP/FP/FN over classes
    # For single-label multiclass, micro-F1 computed this way is well-defined.
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        pred_c = (pred == c)
        tgt_c = (target == c)
        tp_sum += (pred_c & tgt_c).sum().item()
        fp_sum += (pred_c & (~tgt_c)).sum().item()
        fn_sum += ((~pred_c) & tgt_c).sum().item()

    f1_micro = (2 * tp_sum + eps) / (2 * tp_sum + fp_sum + fn_sum + eps)

    return f1_macro, float(f1_micro), f1_per_class

# -------------------------
# Dataset MULTI-CLASSE segmentation (single-label/pixel)
# mask: 0..K-1 (en PNG niveau de gris)
# -------------------------
class MultiClassSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=224):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.names = sorted(os.listdir(img_dir))

        self.img_tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),  # (3,H,W) float in [0,1]
        ])

        # IMPORTANT: nearest pour ne pas interpoler les labels
        self.mask_rs = T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # valeurs 0..K-1

        img = self.img_tf(img)     # (3,H,W)
        mask = self.mask_rs(mask)  # PIL resized

        mask = torch.from_numpy(np.array(mask)).long()  # (H,W) long
        return img, mask, name


# -------------------------
# Metrics multi-classe (mIoU, mDice)
# -------------------------
@torch.no_grad()
def mean_iou_and_dice(pred, target, num_classes, ignore_index=None, eps=1e-7):
    """
    pred/target: (B,H,W) int
    """
    ious, dices = [], []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue

        pred_c = (pred == c)
        tgt_c = (target == c)

        tp = (pred_c & tgt_c).sum().item()
        fp = (pred_c & (~tgt_c)).sum().item()
        fn = ((~pred_c) & tgt_c).sum().item()

        iou = (tp + eps) / (tp + fp + fn + eps)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)

        ious.append(iou)
        dices.append(dice)

    return float(np.mean(ious)), float(np.mean(dices))


# -------------------------
# One evaluation epoch (multi-classe)
# -------------------------
def evaluate(model, loader, device, num_classes, ignore_index=None):
    model.eval()
    ce = nn.CrossEntropyLoss() if ignore_index is None else nn.CrossEntropyLoss(ignore_index=ignore_index)

    total_loss = 0.0
    total_miou = 0.0
    total_mdice = 0.0
    total_acc = 0.0
    total_f1_macro = 0.0
    total_f1_micro = 0.0
    n = 0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device)        # (B,3,H,W)
        masks = masks.to(device)      # (B,H,W) long

        logits = model(imgs)          # (B,K,H,W)
        loss = ce(logits, masks)

        pred = torch.argmax(logits, dim=1)  # (B,H,W)

        miou, mdice = mean_iou_and_dice(pred, masks, num_classes=num_classes, ignore_index=ignore_index)
        acc = pixel_accuracy(pred, masks, ignore_index=ignore_index)
        f1_macro, f1_micro, _ = f1_scores(pred, masks, num_classes=num_classes, ignore_index=ignore_index)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_miou += miou * bs
        total_mdice += mdice * bs
        total_acc += acc * bs
        total_f1_macro += f1_macro * bs
        total_f1_micro += f1_micro * bs
        n += bs

    return (
        total_loss / n,
        total_miou / n,
        total_mdice / n,
        total_acc / n,
        total_f1_macro / n,
        total_f1_micro / n
    )


# -------------------------
# Run test
# -------------------------
def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: change paths
    img_dir = "toy_multiclass_dataset/images"
    mask_dir = "toy_multiclass_dataset/masks"

    # IMPORTANT: num_classes = K (incluant background)
    num_classes = 4   # ex: 0..3
    ignore_index = None  # ou 0 si tu veux ignorer background dans les métriques (et aussi dans CE si besoin)

    ds = MultiClassSegDataset(img_dir, mask_dir, size=224)
    dl = DataLoader(ds, batch_size=5, shuffle=False, num_workers=2, pin_memory=True)

    # --- your model ---
    model = UNext(num_classes=num_classes, input_channels=3, deep_supervision=True).to(device)

    # Sanity forward on one batch
    imgs, masks, _ = next(iter(dl))
    imgs = imgs.to(device)
    with torch.no_grad():
        logits = model(imgs)
    print("logits shape:", logits.shape)  # attendu: (B,K,H,W)

    loss, miou, mdice, acc, f1_macro, f1_micro = evaluate(
    model, dl, device, num_classes=num_classes, ignore_index=ignore_index
        )
    print(
        f"Eval | loss={loss:.4f} mIoU={miou:.4f} mDice={mdice:.4f} "
        f"acc={acc:.4f} F1(macro)={f1_macro:.4f} F1(micro)={f1_micro:.4f}"
    )



if __name__ == "__main__":
    run_test()

    """
    Exemple de résultat attendu (avant entraînement):
    Eval | loss=~1.xx mIoU=~0.0x mDice=~0.0x
    """
