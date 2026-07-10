"""Faz 2/3: config'ten okuyarak U-Net veya DeepLabV3+ egitimi.

Kullanim:
    python src/train.py --config configs/unet_baseline.yaml
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex
from tqdm import tqdm

from augmentations import get_train_transform, get_val_transform
from dataset import build_dataloaders
from models.factory import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ComboLoss(torch.nn.Module):
    """BCE + Dice loss (class imbalance icin) - agirliklar config'ten ayarlanabilir."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = smp.losses.SoftBCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    dice_metric = BinaryF1Score().to(device)
    iou_metric = BinaryJaccardIndex().to(device)
    acc_metric = BinaryAccuracy().to(device)

    total_loss = 0.0
    n_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks, _categories in tqdm(loader, leave=False):
            images, masks = images.to(device), masks.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = loss_fn(logits, masks)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = (torch.sigmoid(logits) > 0.5).long()
            targets = masks.long()
            dice_metric.update(preds, targets)
            iou_metric.update(preds, targets)
            acc_metric.update(preds, targets)

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": dice_metric.compute().item(),
        "iou": iou_metric.compute().item(),
        "pixel_acc": acc_metric.compute().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    manifest_path = project_root / "data" / "processed" / "splits.csv"
    image_size = config["data"]["image_size"]
    loaders = build_dataloaders(
        manifest_path,
        get_train_transform(image_size),
        get_val_transform(image_size),
        batch_size=config["train"]["batch_size"],
    )

    model = build_model(
        config["model"]["name"],
        config["model"]["encoder"],
        config["model"]["encoder_weights"],
    ).to(device)

    loss_fn = ComboLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])

    run_name = config["model"]["name"]
    checkpoint_dir = project_root / config["logging"]["checkpoint_dir"]
    log_dir = project_root / config["logging"]["log_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_dice = -1.0
    n_params = sum(p.numel() for p in model.parameters())

    for epoch in range(config["train"]["epochs"]):
        t0 = time.time()
        train_metrics = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], loss_fn, device, optimizer=None)
        scheduler.step()
        epoch_time = time.time() - t0

        print(
            f"[{run_name}] epoch {epoch + 1}/{config['train']['epochs']} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"({epoch_time:.1f}s)"
        )

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "epoch_time_sec": epoch_time,
        })

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), checkpoint_dir / f"{run_name}_best.pt")

    with open(log_dir / f"{run_name}_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_name": run_name,
            "config": config,
            "n_params": n_params,
            "best_val_dice": best_val_dice,
            "history": history,
        }, f, indent=2)

    print(f"Egitim tamamlandi. En iyi val Dice: {best_val_dice:.4f}. "
          f"Checkpoint: {checkpoint_dir / f'{run_name}_best.pt'}")


if __name__ == "__main__":
    main()
