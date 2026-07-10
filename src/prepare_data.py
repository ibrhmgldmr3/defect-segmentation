"""Faz 1: MVTec AD (hazelnut) icin train/val/test split manifesti ve EDA istatistikleri uretir.

MVTec AD'nin resmi train split'i sadece normal (good) goruntu icerir; piksel-seviyesi
ground_truth maskeleri yalnizca test split'indeki kusurlu gorsellerde bulunur. Denetimli
segmentasyon egitimi icin resmi test split'ini (good + tum kusur tipleri, maskeli) etiketli
havuz olarak kullaniyoruz ve kendi train/val/test split'imizi kategoriye gore stratifiye
ederek olusturuyoruz. Negatif (kusursuz) ornek cesitliligini artirmak icin resmi train/good
havuzundan da kucuk bir alt kume ekliyoruz.
"""
import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image

RAW_DIR = Path("data/raw/hazelnut")
PROCESSED_DIR = Path("data/processed")

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
N_EXTRA_GOOD_FROM_TRAIN = 60
SEED = 42


def collect_labeled_pool():
    """Resmi test/ + ground_truth/ altindan (image_path, mask_path, category) ucluleri toplar."""
    items = []
    test_dir = RAW_DIR / "test"
    gt_dir = RAW_DIR / "ground_truth"
    for category_dir in sorted(test_dir.iterdir()):
        category = category_dir.name
        for img_path in sorted(category_dir.glob("*.png")):
            if category == "good":
                items.append((img_path, None, category))
            else:
                mask_path = gt_dir / category / f"{img_path.stem}_mask.png"
                items.append((img_path, mask_path, category))
    return items


def collect_extra_good(n, rng):
    """Resmi train/good havuzundan rastgele n gorsel ekler (maske yok -> tumu negatif)."""
    train_good = sorted((RAW_DIR / "train" / "good").glob("*.png"))
    sample = rng.sample(train_good, min(n, len(train_good)))
    return [(p, None, "good_train") for p in sample]


def stratified_split(items, rng):
    """Kategoriye gore grupla, her grubu ayri ayri train/val/test'e ayir."""
    by_category = {}
    for item in items:
        by_category.setdefault(item[2], []).append(item)

    splits = {"train": [], "val": [], "test": []}
    for category, group in by_category.items():
        group = group[:]
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * SPLIT_RATIOS["train"])
        n_val = round(n * SPLIT_RATIOS["val"])
        splits["train"] += group[:n_train]
        splits["val"] += group[n_train:n_train + n_val]
        splits["test"] += group[n_train + n_val:]
    return splits


def image_size(path):
    with Image.open(path) as im:
        return im.size  # (W, H)


def mask_positive_ratio(path):
    with Image.open(path) as im:
        im = im.convert("L")
        px = im.point(lambda v: 1 if v > 127 else 0)
        data = list(px.getdata())
        return sum(data) / len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--extra-good", type=int, default=N_EXTRA_GOOD_FROM_TRAIN)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    labeled_pool = collect_labeled_pool()
    extra_good = collect_extra_good(args.extra_good, rng)
    all_items = labeled_pool + extra_good

    splits = stratified_split(all_items, rng)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROCESSED_DIR / "splits.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path", "category", "split"])
        for split_name, items in splits.items():
            for img_path, mask_path, category in items:
                writer.writerow([
                    img_path.resolve().as_posix(),
                    mask_path.resolve().as_posix() if mask_path else "",
                    category,
                    split_name,
                ])

    # --- EDA istatistikleri ---
    stats = {"counts": {}, "image_sizes": {}, "mask_positive_pixel_ratio": {}}
    for split_name, items in splits.items():
        stats["counts"][split_name] = {}
        for _, _, category in items:
            stats["counts"][split_name][category] = stats["counts"][split_name].get(category, 0) + 1

    sizes = set()
    ratios = []
    for img_path, mask_path, category in all_items:
        sizes.add(image_size(img_path))
        if mask_path is not None:
            ratios.append(mask_positive_ratio(mask_path))

    stats["image_sizes"]["unique_sizes"] = [list(s) for s in sizes]
    if ratios:
        stats["mask_positive_pixel_ratio"]["mean"] = sum(ratios) / len(ratios)
        stats["mask_positive_pixel_ratio"]["min"] = min(ratios)
        stats["mask_positive_pixel_ratio"]["max"] = max(ratios)
        stats["mask_positive_pixel_ratio"]["n_defective_masks"] = len(ratios)

    stats_path = PROCESSED_DIR / "eda_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Manifest yazildi: {manifest_path} ({len(all_items)} kayit)")
    print(f"EDA istatistikleri yazildi: {stats_path}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
