# metrics/binary_segmentation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class BinarySegMetrics:
    """Container for binary segmentation metrics."""
    loss: Optional[float] = None
    dice: float = 0.0
    iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is (B,1,H,W) for binary segmentation.
    Accepts (B,H,W) or (B,1,H,W).
    """
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4 and x.size(1) == 1:
        return x
    raise ValueError(f"Expected shape (B,H,W) or (B,1,H,W), got {tuple(x.shape)}")


@torch.no_grad()
def binarize_from_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    logits -> sigmoid -> threshold -> {0,1} float tensor with shape (B,1,H,W)
    """
    logits = _ensure_bchw(logits)
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


@torch.no_grad()
def binarize_from_probs(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    probs in [0,1] -> threshold -> {0,1} float tensor with shape (B,1,H,W)
    """
    probs = _ensure_bchw(probs)
    return (probs > threshold).float()


@torch.no_grad()
def dice_coef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    """
    pred/target: (B,1,H,W) in {0,1}
    """
    pred = _ensure_bchw(pred)
    target = _ensure_bchw(target)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return float(dice.mean().item())


@torch.no_grad()
def iou_coef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    """
    pred/target: (B,1,H,W) in {0,1}
    """
    pred = _ensure_bchw(pred)
    target = _ensure_bchw(target)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target - pred * target).sum(dim=(2, 3))
    iou = (inter + eps) / (union + eps)
    return float(iou.mean().item())


@torch.no_grad()
def confusion_from_binary(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (tp, fp, fn, tn) as scalars (summed over batch and pixels).
    pred/target: (B,1,H,W) in {0,1}
    """
    pred = _ensure_bchw(pred)
    target = _ensure_bchw(target)

    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    tn = ((1.0 - pred) * (1.0 - target)).sum()
    return tp, fp, fn, tn


@torch.no_grad()
def precision_recall_accuracy(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> Tuple[float, float, float]:
    """
    pred/target: (B,1,H,W) in {0,1}
    """
    tp, fp, fn, tn = confusion_from_binary(pred, target)
    precision = (tp / (tp + fp + eps)).item()
    recall = (tp / (tp + fn + eps)).item()
    acc = ((tp + tn) / (tp + fp + fn + tn + eps)).item()
    return float(precision), float(recall), float(acc)


@torch.no_grad()
def compute_binary_metrics(
    logits: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
    pred: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Convenience wrapper:
      - provide one of: logits OR probs OR pred
      - provide target
    Returns dict with dice, iou, precision, recall, accuracy
    """
    if target is None:
        raise ValueError("target must be provided")
    target = _ensure_bchw(target).float()
    target = (target > 0.5).float()

    if pred is None:
        if logits is not None:
            pred = binarize_from_logits(logits, threshold=threshold)
        elif probs is not None:
            pred = binarize_from_probs(probs, threshold=threshold)
        else:
            raise ValueError("Provide one of logits, probs, or pred")

    pred = _ensure_bchw(pred).float()
    pred = (pred > 0.5).float()

    d = dice_coef(pred, target, eps=eps)
    j = iou_coef(pred, target, eps=eps)
    p, r, a = precision_recall_accuracy(pred, target, eps=eps)

    return {"dice": d, "iou": j, "precision": p, "recall": r, "accuracy": a}