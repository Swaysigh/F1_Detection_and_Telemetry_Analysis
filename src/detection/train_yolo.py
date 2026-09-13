"""Train YOLO on F1 car detection dataset."""
if __name__ == "__main__":
    cfg = load_config()
    train(cfg)


import yaml
from pathlib import Path
from ultralytics import YOLO


def load_config(path="configs/yolo_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg):
    model = YOLO(cfg["model"]["arch"])

    # Ultralytics always writes under a "runs/<task>/" base unless given an
    # absolute path - resolve project_dir to absolute so results land exactly
    # where configured, not nested inside an extra runs/detect/ folder.
    project_dir = str(Path(cfg["output"]["project_dir"]).resolve())

    model.train(
        data=cfg["data"]["yaml_path"],
        epochs=cfg["train"]["epochs"],
        imgsz=cfg["train"]["img_size"],
        batch=cfg["train"]["batch_size"],
        device=cfg["train"]["device"],
        lr0=cfg["train"]["lr0"],
        patience=cfg["train"]["patience"],
        project=project_dir,
        name=cfg["output"]["run_name"],
    )


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)