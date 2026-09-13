"""CLI entrypoint to run the full detection -> classification -> telemetry pipeline on a video."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--yolo-weights", default="models/checkpoints/yolo_best.pt")
    parser.add_argument("--classifier-weights", default="models/checkpoints/classifier_best.pt")
    parser.add_argument("--output", default="outputs/videos/annotated.mp4")
    args = parser.parse_args()

    # TODO: wire up src/pipeline.py:run_pipeline
    print(f"Running pipeline on {args.video} -> {args.output}")


if __name__ == "__main__":
    main()
