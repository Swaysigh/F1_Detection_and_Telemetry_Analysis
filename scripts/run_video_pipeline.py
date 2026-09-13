"""
End-to-end pipeline test: run YOLO detection + team classification on a
real video, frame by frame, and save an annotated output video showing
bounding boxes + predicted team + confidence for each detected car.

This is the real integration test - unlike testing on isolated pre-cropped
images, this runs the full chain (detect -> crop -> classify) on genuine
video frames, which is what the deployed pipeline actually needs to do.

Usage:
  python scripts/run_video_pipeline.py \
    --video data/raw/video_samples/dutch_gp_2025.mp4 \
    --yolo-weights runs/detect/outputs/predictions/yolo/exp1/weights/best.pt \
    --classifier-weights models/checkpoints/classifier/classifier_best.pt \
    --out outputs/videos/dutch_gp_2025_annotated.mp4 \
    --max-frames 300
"""
import argparse
from pathlib import Path

import cv2
import torch
import yaml
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO


CLS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_class_names(config_path="configs/classifier_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["data"]["class_names"]


def letterbox_square(crop, size=224, fill=(114, 114, 114)):
    h, w = crop.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    padded = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill)
    return cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)


def classify_crop(model, crop_bgr, class_names, device):
    """crop_bgr: numpy array (BGR, from cv2). Returns (class_name, confidence)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    tensor = CLS_TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
    return class_names[pred_idx], confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--yolo-weights", required=True, help="Path to trained YOLO .pt weights")
    parser.add_argument("--classifier-weights", required=True, help="Path to trained classifier .pt")
    parser.add_argument("--config", default="configs/classifier_config.yaml", help="Classifier config (for class names)")
    parser.add_argument("--out", default="outputs/videos/annotated.mp4", help="Output annotated video path")
    parser.add_argument("--det-conf", type=float, default=0.4, help="YOLO detection confidence threshold")
    parser.add_argument("--cls-conf-display", type=float, default=0.0, help="Only show classifier label if confidence exceeds this (0 = always show)")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame (1 = every frame)")
    parser.add_argument("--start-frame", type=int, default=0, help="Skip ahead to this frame before processing starts")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames (for quick testing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = load_class_names(args.config)
    print(f"Classifier classes: {class_names}")

    yolo_model = YOLO(args.yolo_weights)
    cls_model = torch.load(args.classifier_weights, map_location=device, weights_only=False)
    cls_model.eval()
    cls_model.to(device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} total frames")

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        print(f"Skipped ahead to frame {args.start_frame}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    processed = 0
    total_detections = 0
    class_prediction_counts = {name: 0 for name in class_names}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.max_frames and processed >= args.max_frames:
            break

        if frame_idx % args.frame_stride == 0:
            results = yolo_model.predict(frame, conf=args.det_conf, verbose=False)
            boxes = results[0].boxes

            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_sq = letterbox_square(crop)
                pred_class, conf = classify_crop(cls_model, crop_sq, class_names, device)

                total_detections += 1
                if conf >= args.cls_conf_display:
                    class_prediction_counts[pred_class] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{pred_class} {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)

            processed += 1
            if processed % 30 == 0:
                print(f"  processed {processed} frames...")

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\nDone. Processed {processed} frames, {total_detections} total detections.")
    print(f"Annotated video saved to {out_path}")
    print("\nPrediction distribution across all detections:")
    for name, count in class_prediction_counts.items():
        pct = count / total_detections * 100 if total_detections else 0
        print(f"  {name}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()