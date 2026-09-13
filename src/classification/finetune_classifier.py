"""
Fine-tune the existing classifier checkpoint on real-domain (own broadcast
footage) labeled crops. Unlike train_classifier.py (trains a fresh head from
a pretrained ImageNet backbone), this LOADS the already-trained 5-team
checkpoint and continues training at a low learning rate - the model already
knows general car/livery features, it just needs adjustment toward this
specific footage's visual domain (greyscale, motion blur, broadcast angle).

Only classes with actual labeled data in the input folder are used for
fine-tuning - you don't need all 5 teams sorted before starting, this can
run on a partial set (e.g. just Mercedes) and be re-run as you add more.

Usage:
  python -m src.classification.finetune_classifier \
    --base-checkpoint models/checkpoints/classifier/classifier_best_5team.pt \
    --data-dir data/raw/own_footage_labeled \
    --epochs 15 \
    --lr 0.00001
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Full 5-team class list + index mapping, so predictions stay consistent
# with the original model regardless of which classes have new data this run
FULL_CLASS_LIST = ["Ferrari F1 car", "McLaren F1 car", "Mercedes F1 car",
                    "Red Bull Racing F1 car", "Williams F1 car"]


class OwnFootageDataset(Dataset):
    def __init__(self, data_dir, class_to_idx, img_size=224, train=True):
        self.samples = []
        self.class_to_idx = class_to_idx
        root = Path(data_dir)

        for class_dir in root.iterdir():
            if not class_dir.is_dir():
                continue
            if class_dir.name not in class_to_idx:
                print(f"  [warn] folder '{class_dir.name}' doesn't match any known class, skipping")
                continue
            label = class_to_idx[class_dir.name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((img_path, label))

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def classes_present(self):
        return sorted(set(label for _, label in self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def split_train_val(samples, val_frac, seed):
    samples = samples.copy()
    random.Random(seed).shuffle(samples)
    n_val = max(1, int(len(samples) * val_frac))
    return samples[n_val:], samples[:n_val]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True, help="Existing trained classifier .pt to fine-tune from")
    parser.add_argument("--data-dir", required=True, help="Folder-per-class real-domain labeled data (can be partial - e.g. only Mercedes)")
    parser.add_argument("--out-checkpoint", default="models/checkpoints/classifier/classifier_finetuned.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5, help="Fine-tuning LR - much lower than fresh training to avoid catastrophic forgetting")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_to_idx = {name: i for i, name in enumerate(FULL_CLASS_LIST)}
    print(f"Full class list (fixed, matches base model): {FULL_CLASS_LIST}")

    full_dataset = OwnFootageDataset(args.data_dir, class_to_idx, train=True)
    classes_present = full_dataset.classes_present()
    present_names = [FULL_CLASS_LIST[i] for i in classes_present]
    print(f"Classes with new data this run: {present_names}")

    if len(full_dataset) == 0:
        print("No labeled images found - check --data-dir folder names match the class list exactly.")
        return

    train_samples, val_samples = split_train_val(full_dataset.samples, args.val_split, args.seed)
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    train_ds = OwnFootageDataset(args.data_dir, class_to_idx, train=True)
    train_ds.samples = train_samples
    val_ds = OwnFootageDataset(args.data_dir, class_to_idx, train=False)
    val_ds.samples = val_samples

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Loading base checkpoint: {args.base_checkpoint}")
    model = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total if train_total else 0
        val_acc = val_correct / val_total if val_total else 0
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss: {train_loss/train_total:.4f} | "
              f"train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model, out_path)
            print(f"  -> saved (val_acc: {val_acc:.4f})")

    print(f"\nFine-tuning complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Saved to {out_path}")
    print("\nNOTE: val_acc here is only on the classes you've labeled so far -")
    print("run test_classifier_on_own_crops.py afterward to check real generalization,")
    print("and evaluate_classifier.py on the original 5-team val set to confirm the")
    print("model didn't forget the other classes.")


if __name__ == "__main__":
    main()