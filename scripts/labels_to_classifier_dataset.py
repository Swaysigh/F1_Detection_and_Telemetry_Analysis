"""
Convert a YOLO-format OBJECT DETECTION dataset (images + bbox labels with
team-name classes) into a folder-per-class CLASSIFICATION dataset, by
cropping each labeled bounding box out of its source image.

Input structure expected (Roboflow YOLO export):
  <raw_dir>/train/images/*.jpg
  <raw_dir>/train/labels/*.txt   (class_id cx cy w h, normalized)
  <raw_dir>/valid/images/*.jpg
  <raw_dir>/valid/labels/*.txt
  <raw_dir>/data.yaml            (contains 'names: [ferrari, mclaren, ...]')

Output structure produced:
  <out_dir>/train/ferrari/*.jpg
  <out_dir>/train/mclaren/*.jpg
  <out_dir>/valid/ferrari/*.jpg
  ...
  matches what src/classification/dataset.py's DriverDataset expects
  (root_dir/<class_name>/*.jpg per split)

Usage:
  python scripts/labels_to_classifier_dataset.py --raw-dir data/raw/f1-car-classification --out data/processed/classification
"""
import argparse
from pathlib import Path
import cv2
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def letterbox_square(crop, size=224, fill=(114, 114, 114), grayscale=False):
    """
    Pad crop to a square (preserving aspect ratio) then resize to `size`x`size`.
    Prevents distortion from squashing wide/tall crops directly into a square,
    which would otherwise warp car proportions and hurt classifier training.

    grayscale: convert to greyscale (3-channel, so still compatible with
    ResNet/EfficientNet input format) to match production pipeline, which
    only ever sees greyscale detector crops. A classifier trained on color
    but run on greyscale collapses to predicting a single class - training
    on greyscale from the start avoids that domain mismatch.
    """
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


def yolo_to_pixel_bbox(cx, cy, w, h, img_w, img_h):
    """Convert normalized YOLO bbox (center_x, center_y, w, h) to pixel x1,y1,x2,y2."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(y2))


def process_split(split_dir: Path, out_split_dir: Path, class_names: list, min_size: int, grayscale: bool):
    img_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    if not img_dir.exists():
        return 0, 0

    images = [f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    saved, skipped = 0, 0

    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists() or label_path.stat().st_size == 0:
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        img_h, img_w = frame.shape[:2]

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_pixel_bbox(cx, cy, w, h, img_w, img_h)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or min(crop.shape[0], crop.shape[1]) < min_size:
                skipped += 1
                continue

            crop = letterbox_square(crop, grayscale=grayscale)
            class_name = class_names[class_id]
            class_out_dir = out_split_dir / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)

            crop_name = f"{img_path.stem}_{i}.jpg"
            cv2.imwrite(str(class_out_dir / crop_name), crop)
            saved += 1

    return saved, skipped


def prepare(raw_dir: str, out_dir: str, min_size: int = 60, grayscale: bool = False):
    raw_root = Path(raw_dir)
    out_root = Path(out_dir)

    yaml_path = raw_root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No data.yaml found at {yaml_path}")

    with open(yaml_path) as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]
    print(f"Classes: {class_names}")
    print(f"Grayscale conversion: {grayscale}")

    for split_name in ("train", "valid", "test"):
        split_dir = raw_root / split_name
        if not split_dir.exists():
            continue
        out_split_dir = out_root / split_name
        saved, skipped = process_split(split_dir, out_split_dir, class_names, min_size, grayscale)
        print(f"  [{split_name}] saved {saved} crops, skipped {skipped} (too small or unreadable)")

    # Print per-class counts so you can spot severe imbalance
    print("\nPer-class counts:")
    for split_name in ("train", "valid", "test"):
        split_dir = out_root / split_name
        if not split_dir.exists():
            continue
        counts = {}
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                counts[class_dir.name] = len(list(class_dir.glob("*.jpg")))
        print(f"  {split_name}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Path to raw YOLO-format detection dataset")
    parser.add_argument("--out", default="data/processed/classification", help="Output classification dataset root")
    parser.add_argument("--min-size", type=int, default=60, help="Skip crops smaller than this (px, shorter side)")
    parser.add_argument("--grayscale", action="store_true", help="Convert crops to greyscale (3-channel) to match production pipeline, which only sees greyscale detector output")
    args = parser.parse_args()
    prepare(args.raw_dir, args.out, args.min_size, args.grayscale)