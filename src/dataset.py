"""MVTec AD (hazelnut) icin PyTorch Dataset ve DataLoader yardimcilari.

data/processed/splits.csv dosyasini okur (bkz. prepare_data.py). Maskesi olmayan
(kusursuz) ornekler icin goruntu boyutunda sifir maske uretilir.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MVTecSegDataset(Dataset):
    def __init__(self, manifest_path, split, transform=None):
        self.transform = transform
        self.items = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                self.items.append(
                    (row["image_path"], row["mask_path"] or None, row["category"])
                )
        if not self.items:
            raise ValueError(f"'{split}' split'i icin {manifest_path} icinde kayit bulunamadi")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, mask_path, category = self.items[idx]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        mask = mask.unsqueeze(0) if mask.ndim == 2 else mask
        return image, mask.float(), category


def build_dataloaders(manifest_path, train_transform, val_transform, batch_size, num_workers=0):
    train_ds = MVTecSegDataset(manifest_path, "train", transform=train_transform)
    val_ds = MVTecSegDataset(manifest_path, "val", transform=val_transform)
    test_ds = MVTecSegDataset(manifest_path, "test", transform=val_transform)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=False),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers),
    }
    return loaders
