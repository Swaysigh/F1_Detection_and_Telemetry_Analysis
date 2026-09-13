"""
Prepare a Roboflow-exported YOLO dataset for training.

Assumes the raw dataset is ALREADY in YOLO format (images + normalized-bbox
.txt label files) and already preprocessed (e.g. greyscale) by Roboflow.
This script does NOT re-process pixels or re-convert label formats — it just:

  1. Copies train/valid/test images+labels from data/raw/<name>/ to data/processed/
  2. Rewrites data.yaml so paths point at data/processed/ (Roboflow's exported
     yaml assumes it's sitting in the dataset's own folder, which won't match
     this repo's layout)
  3. Validates every image has a matching label file (catches silent
     mismatches before they waste a training run)
  4. Optionally drops images with empty label files (--drop-empty) — use this
     ONLY if you've confirmed those "empty" labels are actually mislabeled
     (i.e. cars are visible but no box was drawn), not genuine background-only
     frames. Dropped files are logged so you can revisit/relabel them later.
  5. Prints the class names so you can confirm they match your configs
"""
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def copy_split(raw_split_dir: Path, processed_split_dir: Path):
    """Copy images/ and labels/ for one split (train/valid/test)."""
    for sub in ("images", "labels"):
        src = raw_split_dir / sub
        dst = processed_split_dir / sub
        if not src.exists():
            print(f"  [skip] {src} not found")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy2(f, dst / f.name)
    print(f"  copied {raw_split_dir.name} -> {processed_split_dir}")


def validate_split(split_dir: Path):
    """Check every image has a matching label file. Returns list of problems."""
    problems = []
    img_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    if not img_dir.exists():
        return problems

    images = [f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    for img in images:
        label_path = label_dir / f"{img.stem}.txt"
        if not label_path.exists():
            problems.append(f"missing label for {img.name}")
        elif label_path.stat().st_size == 0:
            problems.append(f"empty label file for {img.name} (no boxes)")

    label_files = list(label_dir.glob("*.txt")) if label_dir.exists() else []
    img_stems = {f.stem for f in images}
    for lbl in label_files:
        if lbl.stem not in img_stems:
            problems.append(f"orphan label with no matching image: {lbl.name}")

    return problems


def drop_empty_labels(split_dir: Path, log_lines: list):
    """Remove image+label pairs where the label file is empty (0 bytes)."""
    img_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    if not img_dir.exists() or not label_dir.exists():
        return 0

    dropped = 0
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    for img in images:
        label_path = label_dir / f"{img.stem}.txt"
        if label_path.exists() and label_path.stat().st_size == 0:
            log_lines.append(f"{split_dir.name}/{img.name}")
            img.unlink()
            label_path.unlink()
            dropped += 1
    return dropped


def rewrite_data_yaml(raw_yaml_path: Path, processed_root: Path, output_path: Path):
    with open(raw_yaml_path) as f:
        data = yaml.safe_load(f)

    # YOLO resolves train/val/test paths RELATIVE TO THE YAML FILE'S OWN
    # LOCATION, not the repo root. Since data.yaml lives inside processed_root
    # itself, paths must be relative to processed_root (e.g. "train/images"),
    # NOT "data/processed/train/images" - that would get processed_root
    # prepended a second time by YOLO and double the path.
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        split_img_dir = processed_root / folder / "images"
        if split_img_dir.exists():
            data[split] = f"{folder}/images"

    with open(output_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"  wrote {output_path}")
    print(f"  classes: {data.get('names')}")
    return data


def prepare(raw_dir: str, processed_dir: str = "data/processed", drop_empty: bool = False):
    raw_root = Path(raw_dir)
    processed_root = Path(processed_dir)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_root}")

    print(f"Copying dataset: {raw_root} -> {processed_root}")
    for split_name in ("train", "valid", "test"):
        raw_split = raw_root / split_name
        if raw_split.exists():
            copy_split(raw_split, processed_root / split_name)

    # data.yaml
    raw_yaml = raw_root / "data.yaml"
    if raw_yaml.exists():
        rewrite_data_yaml(raw_yaml, processed_root, processed_root / "data.yaml")
    else:
        print("  [warn] no data.yaml found in raw dataset — you'll need to create one manually")

    # Drop empty-label images if requested (BEFORE validation, so the report is clean after)
    if drop_empty:
        print("\nDropping images with empty label files (--drop-empty)...")
        log_lines = []
        total_dropped = 0
        for split_name in ("train", "valid", "test"):
            split_dir = processed_root / split_name
            if split_dir.exists():
                n = drop_empty_labels(split_dir, log_lines)
                if n:
                    print(f"  [{split_name}] dropped {n} image(s) with empty labels")
                total_dropped += n

        if total_dropped:
            log_path = processed_root / "dropped_empty_labels.log"
            with open(log_path, "w") as f:
                f.write(f"# Dropped {total_dropped} images with empty label files\n")
                f.write(f"# Run at {datetime.now().isoformat()}\n")
                f.write(f"# These were confirmed to contain visible cars with no annotation box.\n")
                f.write(f"# Revisit and hand-label these later if you want them back in training.\n\n")
                f.write("\n".join(log_lines))
            print(f"  logged {total_dropped} dropped file(s) to {log_path}")
        else:
            print("  no empty-label images found")

    # Validation pass
    print("\nValidating processed dataset...")
    all_problems = []
    for split_name in ("train", "valid", "test"):
        split_dir = processed_root / split_name
        if split_dir.exists():
            problems = validate_split(split_dir)
            if problems:
                print(f"  [{split_name}] {len(problems)} issue(s):")
                for p in problems[:10]:
                    print(f"    - {p}")
                if len(problems) > 10:
                    print(f"    ... and {len(problems) - 10} more")
            else:
                n_images = len(list((split_dir / 'images').glob('*')))
                print(f"  [{split_name}] OK — {n_images} images, all matched to labels")
            all_problems.extend(problems)

    if all_problems:
        print(f"\n{len(all_problems)} total issue(s) found — review before training.")
    else:
        print("\nDataset ready. All images have matching labels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Path to raw Roboflow export, e.g. data/raw/f1-car-dataset-kkvsm",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Output location for the processed dataset",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Remove images whose label file is empty (0 bytes). Only use after "
             "confirming these are mislabeled (cars visible, no box) rather than "
             "genuine background-only frames.",
    )
    args = parser.parse_args()
    prepare(args.raw_dir, args.processed_dir, drop_empty=args.drop_empty)