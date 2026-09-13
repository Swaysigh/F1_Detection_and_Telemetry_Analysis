"""Train ResNet50 / EfficientNet classifier for team ID."""
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path

from src.classification.dataset import DriverDataset


def load_config(path="configs/classifier_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    backbone = cfg["model"]["backbone"]
    num_classes = cfg["model"]["num_classes"]
    pretrained = cfg["model"]["pretrained"]

    if backbone == "resnet50":
        model = models.resnet50(weights="DEFAULT" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone.startswith("efficientnet"):
        model = getattr(models, backbone)(weights="DEFAULT" if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if cfg["model"]["freeze_backbone"]:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    return model


def compute_class_weights(counts, device):
    """Inverse-frequency weighting so minority classes aren't underweighted
    just because they have fewer training images (e.g. Ferrari/Redbull vs
    the larger Mercedes split here)."""
    counts_t = torch.tensor(counts, dtype=torch.float32)
    weights = counts_t.sum() / (len(counts_t) * counts_t)
    return weights.to(device)


def train(cfg):
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = DriverDataset(cfg["data"]["train_dir"], img_size=cfg["data"]["img_size"], train=True)
    val_ds = DriverDataset(cfg["data"]["val_dir"], img_size=cfg["data"]["img_size"], train=False)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Classes: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    model = build_model(cfg).to(device)

    class_weights = compute_class_weights(train_ds.class_counts(), device)
    print(f"Class weights: {dict(zip(train_ds.classes, class_weights.tolist()))}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    patience = 8  # stop if val_acc doesn't improve for this many epochs

    for epoch in range(cfg["train"]["epochs"]):
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

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | "
              f"train_loss: {train_loss/train_total:.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model, checkpoint_dir / "classifier_best.pt")
            print(f"  -> new best model saved (val_acc: {val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping: no val_acc improvement for {patience} epochs.")
                break

    torch.save(model, checkpoint_dir / "classifier_last.pt")
    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)