"""
Read the filled-in crop_lookup.csv (team_label column completed) and copy
each crop file into a folder-per-class structure ready for fine-tuning.

Usage:
  python scripts/sort_crops_by_label.py \
    --csv outputs/predictions/crop_lookup.csv \
    --crops-dir data/interim/crops \
    --out data/raw/own_footage_labeled
"""
import argparse
import csv
import shutil
from pathlib import Path
from collections import Counter


# Normalize whatever label text was typed in the CSV to a single canonical
# folder name - handles typos/capitalization variants and matches existing
# folder naming convention where applicable.
LABEL_NORMALIZATION = {
    "haas f1": "Haas",
    "alpine bwt": "Alpine",
    "ferrari": "Ferrari F1 car",
    "kick sauber": "Kick Sauber",
    "alpha tauri": "Alpha Tauri",
    "mercedes": "Mercedes F1 car",
    "mcleren": "McLaren F1 car",
    "mclaren": "McLaren F1 car",
    "william": "Williams F1 car",
    "williams": "Williams F1 car",
    "aston martin": "Aston Martin",
    "red bull f1": "Red Bull Racing F1 car",
    "red bull": "Red Bull Racing F1 car",
}


def normalize_label(raw_label: str) -> str:
    key = raw_label.strip().lower()
    return LABEL_NORMALIZATION.get(key, raw_label.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Filled-in lookup CSV with team_label column")
    parser.add_argument("--crops-dir", required=True, help="Folder containing the actual crop image files")
    parser.add_argument("--out", required=True, help="Output root - creates one subfolder per team")
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    out_root = Path(args.out)

    counts = Counter()
    missing = 0
    skipped_empty = 0

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = row.get("team_label", "").strip()
            if not raw_label:
                skipped_empty += 1
                continue

            label = normalize_label(raw_label)
            crop_filename = row["crop_filename"]
            src_path = crops_dir / crop_filename

            if not src_path.exists():
                missing += 1
                continue

            dest_dir = out_root / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_dir / crop_filename)
            counts[label] += 1

    print(f"Sorted crops into {out_root}\n")
    print("Final per-team counts:")
    for label, count in counts.most_common():
        print(f"  {label}: {count}")

    print(f"\nTotal sorted: {sum(counts.values())}")
    if skipped_empty:
        print(f"Skipped (no label): {skipped_empty}")
    if missing:
        print(f"Missing source files (couldn't find in crops-dir): {missing}")


if __name__ == "__main__":
    main()