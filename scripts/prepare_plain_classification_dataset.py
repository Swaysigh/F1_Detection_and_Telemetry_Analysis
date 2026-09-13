""" Convert plain folder-per-class images (NOT YOLO bbox format - just whole
photos organized as <class_name>/*.jpg) into the greyscale, letterboxed,
train/valid split format the classifier training pipeline expects.

Unlike labels_to_classifier_dataset.py, this does NOT crop using bounding
boxes - these source images are already reasonably car-focused photos, so
we just letterbox-pad to square and resize directly.

Input structure expected:
  <raw_dir>/<ClassName1>/*.jpg
  <raw_dir>/<ClassName2>/*.jpg
  ...

Output structure produced:
  <out_dir>/train/<ClassName1>/*.jpg
  <out_dir>/valid/<ClassName1>/*.jpg
  ...

Usage:
  python scripts/prepare_plain_classification_dataset.py --raw-dir data/raw/Formula_1_cars --out data/processed/classification_5team_grayscale --grayscale --val-split 0.15
"""

import argparse
import random
from pathlib import Path
import cv2


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox_square(crop, size=224, fill=(114, 114, 114), grayscale=False):
    """Pad to square (preserving aspect ratio) then resize - avoids distorting
    car proportions the way a direct stretch-resize would."""
    if grayscale:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = crop.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    padded = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill)
    return cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)


def process_class_folder(class_dir: Path, train_out: Path, valid_out: Path,
                          val_split: float, grayscale: bool, min_size: int, seed: int):
    images = [f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    random.Random(seed).shuffle(images)

    n_val = max(1, int(len(images) * val_split))
    val_images = images[:n_val]
    train_images = images[n_val:]

    train_out.mkdir(parents=True, exist_ok=True)
    valid_out.mkdir(parents=True, exist_ok=True)

    saved_train, saved_val, skipped = 0, 0, 0

    for img_path, out_dir, counter_name in (
        *[(p, train_out, "train") for p in train_images],
        *[(p, valid_out, "valid") for p in val_images],
    ):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        if min(img.shape[0], img.shape[1]) < min_size:
            skipped += 1
            continue

        processed = letterbox_square(img, grayscale=grayscale)
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), processed)

        if counter_name == "train":
            saved_train += 1
        else:
            saved_val += 1

    return saved_train, saved_val, skipped


def prepare(raw_dir: str, out_dir: str, val_split: float, grayscale: bool, min_size: int, seed: int):
    raw_root = Path(raw_dir)
    out_root = Path(out_dir)

    class_dirs = sorted(d for d in raw_root.iterdir() if d.is_dir())
    print(f"Found {len(class_dirs)} class folders: {[d.name for d in class_dirs]}")
    print(f"Grayscale: {grayscale} | Val split: {val_split}")

    summary = {}
    for class_dir in class_dirs:
        train_out = out_root / "train" / class_dir.name
        valid_out = out_root / "valid" / class_dir.name
        n_train, n_val, n_skip = process_class_folder(
            class_dir, train_out, valid_out, val_split, grayscale, min_size, seed
        )
        summary[class_dir.name] = {"train": n_train, "valid": n_val, "skipped": n_skip}
        print(f"  [{class_dir.name}] train: {n_train}, valid: {n_val}, skipped: {n_skip}")

    print("\nFinal summary:")
    for name, counts in summary.items():
        print(f"  {name}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Path to folder containing one subfolder per class")
    parser.add_argument("--out", required=True, help="Output root for processed train/valid split")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of each class held out for validation")
    parser.add_argument("--grayscale", action="store_true", help="Convert to greyscale to match production pipeline")
    parser.add_argument("--min-size", type=int, default=60, help="Skip images smaller than this (px, shorter side)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/valid split reproducibility")
    args = parser.parse_args()
    prepare(args.raw_dir, args.out, args.val_split, args.grayscale, args.min_size, args.seed)