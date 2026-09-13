""" Check the size distribution of images in a folder-per-class dataset,
to see how many fall below various resolution thresholds. Useful for
deciding what --min-size cutoff to use before training - very small
source images get blocky/pixelated after resize and may hurt more
than they help.

Usage:
  python scripts/check_image_size_distribution.py --dir "data/raw/Formula_1_cars"
"""
import argparse
from pathlib import Path
import cv2


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Root folder containing class subfolders")
    args = parser.parse_args()

    root = Path(args.dir)
    thresholds = [60, 100, 150, 200, 300]

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        sizes = []
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            sizes.append(min(h, w))

        if not sizes:
            continue

        print(f"\n{class_dir.name} ({len(sizes)} images) - shorter-side pixel size:")
        print(f"  min: {min(sizes)}, max: {max(sizes)}, avg: {sum(sizes)/len(sizes):.0f}")
        for t in thresholds:
            below = sum(1 for s in sizes if s < t)
            pct = below / len(sizes) * 100
            print(f"  below {t}px: {below} images ({pct:.1f}%)")


if __name__ == "__main__":
    main()