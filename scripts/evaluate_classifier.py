"""
Evaluate a trained classifier against a labeled validation set and produce:
  - overall accuracy + per-class accuracy
  - a CSV of every image with predicted vs actual label (correct/incorrect)
  - sample folders of correct/incorrect predictions per class, for visual review

Unlike test_classifier_on_own_crops.py (which has no ground truth), this
script uses the folder-per-class structure itself as ground truth, since
each folder name IS the correct label.

Usage:
  python scripts/evaluate_classifier.py --val-dir data/processed/classification_5team_grayscale/valid --weights models/checkpoints/classifier/classifier_best.pt
"""
import argparse
import csv
import shutil
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path, device):
    model = torch.load(weights_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    return model


def predict(model, img_path, class_names, device):
    img = Image.open(img_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
    return class_names[pred_idx], confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", required=True, help="Validation directory, folder-per-class")
    parser.add_argument("--weights", required=True, help="Path to trained classifier .pt")
    parser.add_argument("--out-dir", default="outputs/predictions/classifier_eval", help="Where to save CSV + sample images")
    parser.add_argument("--sample-per-class", type=int, default=10, help="How many correct/incorrect samples to save per class for visual review")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_dir = Path(args.val_dir)
    class_names = sorted(d.name for d in val_dir.iterdir() if d.is_dir())
    print(f"Classes: {class_names}")

    model = load_model(args.weights, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "eval_predictions.csv"

    img_exts = {".jpg", ".jpeg", ".png"}
    total, correct = 0, 0
    per_class_total = {name: 0 for name in class_names}
    per_class_correct = {name: 0 for name in class_names}
    saved_correct = {name: 0 for name in class_names}
    saved_incorrect = {name: 0 for name in class_names}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "actual_class", "predicted_class", "confidence", "correct"])

        for class_dir in sorted(val_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            actual_class = class_dir.name

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in img_exts:
                    continue

                pred_class, conf = predict(model, img_path, class_names, device)
                is_correct = pred_class == actual_class

                writer.writerow([img_path.name, actual_class, pred_class, f"{conf:.4f}", is_correct])

                total += 1
                per_class_total[actual_class] += 1
                if is_correct:
                    correct += 1
                    per_class_correct[actual_class] += 1

                if is_correct and saved_correct[actual_class] < args.sample_per_class:
                    dest = out_dir / "samples" / actual_class / "correct"
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, dest / img_path.name)
                    saved_correct[actual_class] += 1
                elif not is_correct and saved_incorrect[actual_class] < args.sample_per_class:
                    dest = out_dir / "samples" / actual_class / f"incorrect_predicted_as_{pred_class}"
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, dest / img_path.name)
                    saved_incorrect[actual_class] += 1

    print(f"\nOverall accuracy: {correct}/{total} = {correct/total:.4f}")
    print("\nPer-class accuracy:")
    for name in class_names:
        n = per_class_total[name]
        c = per_class_correct[name]
        print(f"  {name}: {c}/{n} = {c/n:.4f}" if n > 0 else f"  {name}: no samples")

    print(f"\nFull results: {csv_path}")
    print(f"Visual samples (correct + incorrect): {out_dir / 'samples'}")


if __name__ == "__main__":
    main()