"""
Run the trained team classifier on your own broadcast-style crops
(data/interim/crops/, produced by extract_crops.py from your own YOLO
detector) to check whether it generalizes beyond the aerial-angle dataset
it was trained on.

This is a spot-check tool, not a full eval - it saves predictions and
confidence scores to a CSV, and copies a sample into per-predicted-class
folders so you can visually eyeball whether the predictions look right.

Usage:
  python scripts/test_classifier_on_own_crops.py --crops-dir data/interim/crops --weights models/checkpoints/classifier/classifier_best.pt
"""
import argparse
import csv
import shutil
from pathlib import Path

import yaml
import torch
from torchvision import transforms
from PIL import Image


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_class_names(config_path="configs/classifier_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["data"]["class_names"]


def load_model(weights_path, device):
    model = torch.load(weights_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    return model


def predict(model, img_path, device, class_names):
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
    parser.add_argument("--crops-dir", required=True, help="Directory of your own crops to test on")
    parser.add_argument("--weights", required=True, help="Path to trained classifier .pt")
    parser.add_argument("--config", default="configs/classifier_config.yaml", help="Path to classifier config (for class names)")
    parser.add_argument("--out-dir", default="outputs/predictions/classifier_test", help="Where to save sorted samples + CSV")
    parser.add_argument("--sample-per-class", type=int, default=15, help="How many predicted images per class to copy for visual review")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = load_class_names(args.config)
    print(f"Classes: {class_names}")

    model = load_model(args.weights, device)

    crops_dir = Path(args.crops_dir)
    img_exts = {".jpg", ".jpeg", ".png"}
    images = [f for f in crops_dir.iterdir() if f.suffix.lower() in img_exts]
    print(f"Found {len(images)} crops to test")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"

    class_counts = {name: 0 for name in class_names}
    confidences_by_class = {name: [] for name in class_names}
    samples_copied = {name: 0 for name in class_names}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_class", "confidence"])

        for img_path in images:
            pred_class, conf = predict(model, img_path, device, class_names)
            writer.writerow([img_path.name, pred_class, f"{conf:.4f}"])

            class_counts[pred_class] += 1
            confidences_by_class[pred_class].append(conf)

            if samples_copied[pred_class] < args.sample_per_class:
                sample_dir = out_dir / "samples" / pred_class
                sample_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, sample_dir / img_path.name)
                samples_copied[pred_class] += 1

    print(f"\nPredictions saved to {csv_path}")
    print(f"Sample images (for visual review) saved under {out_dir / 'samples'}\n")

    print("Prediction distribution + average confidence per class:")
    for name in class_names:
        n = class_counts[name]
        avg_conf = sum(confidences_by_class[name]) / n if n > 0 else 0
        print(f"  {name}: {n} predictions, avg confidence {avg_conf:.3f}")

    print("\nNOTE: this script has no ground-truth labels for your own crops,")
    print("so it can't compute real accuracy. Open the sample folders and")
    print("manually check: does the predicted team actually match the livery")
    print("you see in each image? That's the real test of generalization.")


if __name__ == "__main__":
    main()