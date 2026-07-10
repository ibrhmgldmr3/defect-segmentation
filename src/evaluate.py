"""Faz 3: Egitilmis checkpoint'leri test split'inde degerlendirir ve karsilastirma
tablosu uretir (Dice, IoU, pixel accuracy, inference suresi, parametre sayisi).

Kullanim:
    python src/evaluate.py --configs configs/unet_baseline.yaml configs/deeplabv3plus.yaml
"""
import argparse
import time
from pathlib import Path

import torch
import yaml
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex

from augmentations import get_val_transform
from dataset import MVTecSegDataset
from models.factory import build_model


def evaluate_config(config_path, project_root, device):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_name = config["model"]["name"]
    checkpoint_path = project_root / config["logging"]["checkpoint_dir"] / f"{run_name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadi: {checkpoint_path}. Once `python src/train.py "
            f"--config {config_path}` calistirin."
        )

    model = build_model(config["model"]["name"], config["model"]["encoder"],
                         config["model"]["encoder_weights"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    manifest_path = project_root / "data" / "processed" / "splits.csv"
    image_size = config["data"]["image_size"]
    test_ds = MVTecSegDataset(manifest_path, "test", transform=get_val_transform(image_size))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    dice_metric = BinaryF1Score().to(device)
    iou_metric = BinaryJaccardIndex().to(device)
    acc_metric = BinaryAccuracy().to(device)

    n_params = sum(p.numel() for p in model.parameters())

    inference_times = []
    with torch.no_grad():
        for images, masks, _categories in test_loader:
            images, masks = images.to(device), masks.to(device)

            t0 = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append((time.perf_counter() - t0) * 1000)

            preds = (torch.sigmoid(logits) > 0.5).long()
            targets = masks.long()
            dice_metric.update(preds, targets)
            iou_metric.update(preds, targets)
            acc_metric.update(preds, targets)

    return {
        "run_name": run_name,
        "encoder": config["model"]["encoder"],
        "dice": dice_metric.compute().item(),
        "iou": iou_metric.compute().item(),
        "pixel_acc": acc_metric.compute().item(),
        "avg_inference_ms": sum(inference_times) / len(inference_times),
        "n_params": n_params,
        "n_test_images": len(test_ds),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/predictions/comparison.md")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = [evaluate_config(project_root / c, project_root, device) for c in args.configs]

    header = "| Model | Encoder | Dice | IoU | Pixel Acc | Inference (ms) | Parametre | Test N |"
    separator = "|---|---|---|---|---|---|---|---|"
    rows = [
        f"| {r['run_name']} | {r['encoder']} | {r['dice']:.4f} | {r['iou']:.4f} | "
        f"{r['pixel_acc']:.4f} | {r['avg_inference_ms']:.2f} | {r['n_params']:,} | {r['n_test_images']} |"
        for r in results
    ]
    table = "\n".join([header, separator, *rows])
    print(table)

    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"# Model Karsilastirma\n\n{table}\n", encoding="utf-8")
    print(f"\nKarsilastirma tablosu yazildi: {output_path}")


if __name__ == "__main__":
    main()
