""" Merge multiple folder-per-class (train/valid split) classification datasets
into one combined dataset. Used here to combine the original 5-team dataset
with a subset of new teams from a second source, without duplicating teams
that already exist in the primary dataset.

Usage:
  python scripts/merge_classification_datasets.py \
    --primary data/processed/classification_5team_grayscale \
    --secondary data/processed/classification_new_teams_grayscale \
    --secondary-classes "Alfa Romeo" "Alpine" "Aston Martin" "Haas" \
    --out data/processed/classification_9team_grayscale
"""
import argparse
import shutil
from pathlib import Path


def copy_class_folder(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            count += 1
    return count


def merge(primary_dir: str, secondary_dir: str, secondary_classes: list, out_dir: str):
    primary_root = Path(primary_dir)
    secondary_root = Path(secondary_dir)
    out_root = Path(out_dir)

    summary = {}

    for split in ("train", "valid"):
        primary_split = primary_root / split
        if not primary_split.exists():
            print(f"  [warn] {primary_split} not found, skipping")
            continue

        for class_dir in sorted(primary_split.iterdir()):
            if not class_dir.is_dir():
                continue
            out_class_dir = out_root / split / class_dir.name
            n = copy_class_folder(class_dir, out_class_dir)
            summary.setdefault(class_dir.name, {})[split] = n

    for split in ("train", "valid"):
        secondary_split = secondary_root / split
        if not secondary_split.exists():
            print(f"  [warn] {secondary_split} not found, skipping")
            continue

        for class_name in secondary_classes:
            class_dir = secondary_split / class_name
            if not class_dir.exists():
                print(f"  [warn] class '{class_name}' not found in {secondary_split}")
                continue
            out_class_dir = out_root / split / class_name
            n = copy_class_folder(class_dir, out_class_dir)
            summary.setdefault(class_name, {})[split] = n

    print(f"\nMerged dataset written to {out_root}")
    print("\nFinal per-class counts:")
    for name, counts in summary.items():
        print(f"  {name}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, help="Primary dataset root (train/valid folders) - all classes kept")
    parser.add_argument("--secondary", required=True, help="Secondary dataset root - only specified classes taken")
    parser.add_argument("--secondary-classes", nargs="+", required=True, help="Which class names to pull from the secondary dataset")
    parser.add_argument("--out", required=True, help="Output merged dataset root")
    args = parser.parse_args()
    merge(args.primary, args.secondary, args.secondary_classes, args.out)