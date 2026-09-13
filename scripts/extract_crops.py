"""
Run trained YOLO detector on images (or video frames) and save cropped car
images to disk. These crops become the raw input for classifier training -
you'll sort them into per-driver/per-team folders afterward.

Usage:
  python scripts/extract_crops.py --source data/processed/valid/images --weights runs/detect/.../best.pt --out data/interim/crops

  python scripts/extract_crops.py --source path/to/video.mp4 --weights runs/detect/.../best.pt --out data/interim/crops --video
"""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def crop_from_bbox(frame, bbox, pad=0.05):
    """Crop with a small padding margin so the box isn't razor-tight on the car."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    return frame[y1:y2, x1:x2]

def letterbox_square(crop, size=224, fill=(114, 114, 114), grayscale=True):
    """
    Pad crop to a square (preserving aspect ratio) then resize to `size`x`size`.
    Prevents distortion from squashing wide/tall crops directly into a square,
    which would otherwise warp car proportions and hurt classifier training.

    grayscale: convert to greyscale (3-channel) by default, since the
    classifier was trained on greyscale crops - feeding it color crops from
    video causes a severe domain mismatch and collapsed/wrong predictions.
    """
    if grayscale:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = crop.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    padded = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill)
    return cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)


def process_images(model, source_dir: Path, out_dir: Path, conf: float, min_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in source_dir.iterdir() if f.suffix.lower() in img_exts]

    total_crops = 0
    skipped_small = 0
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = model.predict(frame, conf=conf, verbose=False)
        boxes = results[0].boxes

        for i, box in enumerate(boxes.xyxy.cpu().numpy()):
            crop = crop_from_bbox(frame, box)
            if crop.size == 0:
                continue
            if min(crop.shape[0], crop.shape[1]) < min_size:
                skipped_small += 1
                continue
            crop = letterbox_square(crop)
            crop_name = f"{img_path.stem}_car{i}.jpg"
            cv2.imwrite(str(out_dir / crop_name), crop)
            total_crops += 1

    print(f"Saved {total_crops} crops from {len(images)} images -> {out_dir}")
    if skipped_small:
        print(f"Skipped {skipped_small} crops below min-size ({min_size}px)")


def process_video(model, video_path: Path, out_dir: Path, conf: float, frame_stride: int = 5, min_size: int = 60, max_frames: int = None):
    """frame_stride: only process every Nth frame, since consecutive frames
    are near-duplicates and you don't need crops from every single one."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_video_frames} total frames, processing every {frame_stride}th frame")

    frame_idx = 0
    processed_count = 0
    total_crops = 0
    skipped_small = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            results = model.predict(frame, conf=conf, verbose=False)
            boxes = results[0].boxes

            for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                crop = crop_from_bbox(frame, box)
                if crop.size == 0:
                    continue
                if min(crop.shape[0], crop.shape[1]) < min_size:
                    skipped_small += 1
                    continue
                crop = letterbox_square(crop)
                crop_name = f"{video_path.stem}_f{frame_idx}_car{i}.jpg"
                cv2.imwrite(str(out_dir / crop_name), crop)
                total_crops += 1

            processed_count += 1
            if processed_count % 10 == 0:
                pct = frame_idx / total_video_frames * 100 if total_video_frames else 0
                print(f"  processed {processed_count} frames (frame_idx={frame_idx}, ~{pct:.1f}% through video), {total_crops} crops so far...")

            if max_frames and processed_count >= max_frames:
                print(f"  reached --max-frames limit ({max_frames}), stopping early")
                break

        frame_idx += 1

    cap.release()
    print(f"Saved {total_crops} crops from {frame_idx} frames -> {out_dir}")
    if skipped_small:
        print(f"Skipped {skipped_small} crops below min-size ({min_size}px)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image directory or video file path")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO .pt weights")
    parser.add_argument("--out", default="data/interim/crops", help="Where to save cropped car images")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--min-size", type=int, default=60, help="Skip crops smaller than this (px, shorter side) - too small to carry useful detail for classification")
    parser.add_argument("--video", action="store_true", help="Treat --source as a video file, not an image directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing this many sampled frames")
    parser.add_argument("--frame-stride", type=int, default=5, help="For video: process every Nth frame")
    args = parser.parse_args()

    model = YOLO(args.weights)
    out_dir = Path(args.out)

    if args.video:
        process_video(model, Path(args.source), out_dir, args.conf, args.frame_stride, args.min_size, args.max_frames)
    else:
        process_images(model, Path(args.source), out_dir, args.conf, args.min_size)


if __name__ == "__main__":
    main()