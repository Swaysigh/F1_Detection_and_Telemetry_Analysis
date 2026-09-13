"""
Generate a lookup list mapping each greyscale crop (from data/interim/crops/,
produced by extract_crops.py) back to its full-size SOURCE frame filename in
the colored dataset - so you can quickly find which color image to look at
to identify the team for each crop, instead of guessing from the crop alone.

Crop filenames look like:
  <descriptive_prefix>.rf.<hash>_car<N>.jpg
The colored dataset's matching frame has the SAME descriptive_prefix but a
DIFFERENT .rf.<hash> (Roboflow regenerates that hash per export). It may ALSO
be in a different train/valid/test split than your greyscale crop came from,
since splits can be re-randomized between separately generated dataset
versions - so this searches all three split folders under --colored-root.

Usage:
  python scripts/build_crop_to_source_lookup.py \
    --crops-dir data/interim/crops \
    --colored-root "F1 Car Dataset kksm colored.v8i.yolov8" \
    --out outputs/predictions/crop_lookup.csv
"""
import argparse
import csv
import re
from pathlib import Path


def strip_to_prefix(filename: str) -> str:
    """Remove _carN suffix (if present) and .rf.<hash> suffix, keep only the
    descriptive source-frame prefix for matching across dataset exports."""
    stem = Path(filename).stem
    stem = re.sub(r"_car\d+$", "", stem)
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)
    return stem


def build_colored_lookup(colored_root: Path):
    """Search train/valid/test images subfolders under colored_root, map
    stripped-prefix -> (split_name, actual filename)."""
    lookup = {}
    for split in ("train", "valid", "test"):
        split_dir = colored_root / split / "images"
        if not split_dir.exists():
            continue
        for f in split_dir.iterdir():
            if not f.is_file():
                continue
            prefix = strip_to_prefix(f.name)
            lookup[prefix] = (split, f.name)
    return lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops-dir", required=True, help="Your existing greyscale crops folder")
    parser.add_argument("--colored-root", required=True, help="Root folder of colored dataset (contains train/valid/test subfolders)")
    parser.add_argument("--out", default="outputs/predictions/crop_lookup.csv")
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    colored_root = Path(args.colored_root)

    colored_lookup = build_colored_lookup(colored_root)
    print(f"Found {len(colored_lookup)} unique colored source frames across train/valid/test")

    crop_files = [f for f in crops_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"Found {len(crop_files)} crops to look up")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matched, unmatched = 0, 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_filename", "source_frame_prefix", "colored_split", "colored_filename", "found", "team_label"])

        for crop in sorted(crop_files):
            prefix = strip_to_prefix(crop.name)
            match = colored_lookup.get(prefix)
            if match:
                split, colored_name = match
                found = True
                matched += 1
            else:
                split, colored_name = "", ""
                found = False
                unmatched += 1
            writer.writerow([crop.name, prefix, split, colored_name, found, ""])

    print(f"\nMatched: {matched} | Unmatched: {unmatched}")
    print(f"Lookup CSV written to {out_path}")
    print("\nOpen this CSV, sort/group by source_frame_prefix, and for each group:")
    print("  1. Open colored_filename in colored_root/<colored_split>/images/")
    print("  2. Identify the team for each crop from that frame")
    print("  3. Fill in the team_label column")
    print("  4. We'll write a script to auto-sort crops into folders based on this CSV")


if __name__ == "__main__":
    main()
