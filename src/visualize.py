"""Faz 4: tahmin gorsellestirme (girdi / gercek maske / tahmin / Grad-CAM) ve
en dusuk Dice skorlu (false positive/negative) orneklerin ayri bir galeride derlenmesi.

Kullanim:
    python src/visualize.py --config configs/unet_baseline.yaml
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

from augmentations import IMAGENET_MEAN, IMAGENET_STD, get_val_transform
from dataset import MVTecSegDataset
from models.factory import build_model


def denormalize(tensor_img):
    img = tensor_img.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


def mask_to_color(mask, color):
    overlay = np.zeros((*mask.shape, 3), dtype=np.float32)
    overlay[mask > 0.5] = color
    return overlay


def per_sample_dice(pred, target, eps=1e-6):
    pred, target = pred.flatten(), target.flatten()
    intersection = (pred * target).sum()
    return float((2 * intersection + eps) / (pred.sum() + target.sum() + eps))


def save_panel(image, gt_mask, pred_mask, cam_image, dice_score, out_path):
    gt_overlay = image * 0.6 + mask_to_color(gt_mask, (0, 1, 0)) * 0.4
    pred_overlay = image * 0.6 + mask_to_color(pred_mask, (1, 0, 0)) * 0.4

    panel = np.concatenate([image, gt_overlay, pred_overlay, cam_image / 255.0], axis=1)
    panel = (np.clip(panel, 0, 1) * 255).astype(np.uint8)
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.putText(panel, f"Dice: {dice_score:.3f} | girdi / gercek maske / tahmin / Grad-CAM",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), panel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-failures", type=int, default=5)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = config["model"]["name"]
    checkpoint_path = project_root / config["logging"]["checkpoint_dir"] / f"{run_name}_best.pt"

    model = build_model(config["model"]["name"], config["model"]["encoder"],
                         config["model"]["encoder_weights"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    manifest_path = project_root / "data" / "processed" / "splits.csv"
    image_size = config["data"]["image_size"]
    test_ds = MVTecSegDataset(manifest_path, "test", transform=get_val_transform(image_size))

    target_layers = [model.encoder.layer4[-1]]

    out_gallery = project_root / "outputs" / "predictions" / run_name / "gallery"
    out_failures = project_root / "outputs" / "predictions" / run_name / "failure_gallery"
    out_gallery.mkdir(parents=True, exist_ok=True)
    out_failures.mkdir(parents=True, exist_ok=True)

    scored = []
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for idx in range(len(test_ds)):
            image_t, mask_t, category = test_ds[idx]
            input_tensor = image_t.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                pred_mask = (torch.sigmoid(logits) > 0.5).float()[0, 0].cpu().numpy()

            gt_mask = mask_t[0].numpy()
            dice_score = per_sample_dice(pred_mask, gt_mask)

            targets = [SemanticSegmentationTarget(0, pred_mask)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            rgb_img = denormalize(image_t)
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            panel_path = out_gallery / f"{idx:03d}_{category}.png"
            save_panel(rgb_img, gt_mask, pred_mask, cam_image, dice_score, panel_path)
            scored.append((idx, dice_score, category, panel_path))

    scored.sort(key=lambda x: x[1])
    worst = scored[: args.n_failures]
    print("En dusuk Dice skorlu ornekler (false positive/negative adaylari):")
    for idx, dice_score, category, panel_path in worst:
        dst = out_failures / f"{idx:03d}_{category}_dice{dice_score:.2f}.png"
        shutil.copyfile(panel_path, dst)
        print(f"  idx={idx} category={category} dice={dice_score:.4f}")

    print(f"Tum test seti galerisi ({len(scored)} gorsel): {out_gallery}")
    print(f"Basarisiz ornek galerisi ({len(worst)} gorsel): {out_failures}")


if __name__ == "__main__":
    main()
