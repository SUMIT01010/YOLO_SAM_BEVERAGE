"""Main orchestrator for YOLO + SAM beverage extraction experiment."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List

from prepare_dataset import prepare_yolo_dataset
from train_yolo import train_yolov8


def parse_class_order(value: str) -> List[str]:
    """Parse comma-separated class names."""
    classes = [v.strip() for v in value.split(",") if v.strip()]
    if not classes:
        raise ValueError("class-order cannot be empty.")
    return classes


def run_experiment(args: argparse.Namespace) -> Dict[str, str]:
    """Run full pipeline: prepare -> train -> YOLO+SAM -> visualize."""
    # Import heavy runtime deps only when execution starts.
    from pipeline_infer import run_pipeline_batch
    from visualization import save_clean_crops, save_example_panels, save_examples_grid

    dataset_dir = Path(args.dataset_dir)
    annotations_csv = (
        Path(args.annotations_csv)
        if args.annotations_csv
        else dataset_dir / "_annotations.csv"
    )
    yolo_dataset_dir = Path(args.yolo_dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Prepare YOLO dataset from CSV (first N images + stratified split).
    print("[1/4] Preparing YOLO dataset from CSV annotations...")
    prep_meta = prepare_yolo_dataset(
        annotations_csv=annotations_csv,
        image_dir=dataset_dir,
        output_dir=yolo_dataset_dir,
        max_images=args.max_images,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        random_state=args.random_state,
        class_order=parse_class_order(args.class_order),
    )
    print(f"data.yaml: {prep_meta['data_yaml']}")
    print(f"split summary: {prep_meta['summary_json']}")

    # 2) Train YOLO (or use existing weights).
    if args.yolo_weights:
        yolo_weights = args.yolo_weights
        print(f"[2/4] Using provided YOLO weights: {yolo_weights}")
    else:
        if args.skip_train:
            raise ValueError(
                "skip-train was enabled but no --yolo-weights path was provided."
            )
        print("[2/4] Training YOLOv8 detector...")
        yolo_weights = train_yolov8(
            data_yaml=prep_meta["data_yaml"],
            model_checkpoint=args.yolo_base,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.run_name,
            device=args.device,
            workers=args.workers,
        )
        print(f"best weights: {yolo_weights}")

    # 3) Run YOLO + SAM on test set examples.
    print("[3/4] Running YOLO + SAM inference on test samples...")
    test_filenames = prep_meta["splits"]["test"][: args.num_examples]
    test_image_paths = [str(dataset_dir / fname) for fname in test_filenames]
    results = run_pipeline_batch(
        image_paths=test_image_paths,
        yolo_weights=yolo_weights,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        use_mobile_sam=args.use_mobile_sam,
        device=args.device,
        conf=args.conf_thres,
        iou=args.iou_thres,
        max_det=args.max_det,
    )

    # 4) Save visualization artifacts (10 examples + grid + clean crops).
    print("[4/4] Saving visualization artifacts...")
    examples_dir = output_dir / "examples"
    clean_dir = output_dir / "clean_crops"
    panels = save_example_panels(results, examples_dir, max_examples=args.num_examples)
    grid_path = save_examples_grid(
        results,
        examples_dir / "pipeline_examples_grid.png",
        max_examples=args.num_examples,
    )
    clean_paths = save_clean_crops(results, clean_dir, max_examples=args.num_examples)

    # Keep one canonical single-image visualization path for convenience.
    pipeline_visual = output_dir / "pipeline_visualization.png"
    if panels:
        shutil.copy2(panels[0], pipeline_visual)

    status_counts = Counter(r.status for r in results)
    print("status counts:", dict(status_counts))
    print(f"saved {len(panels)} example panels in: {examples_dir}")
    print(f"saved combined grid: {grid_path}")
    print(f"saved {len(clean_paths)} clean crops in: {clean_dir}")
    print(f"single preview: {pipeline_visual}")

    summary = {
        "data_yaml": str(prep_meta["data_yaml"]),
        "split_summary_json": str(prep_meta["summary_json"]),
        "yolo_weights": str(yolo_weights),
        "examples_dir": str(examples_dir),
        "examples_grid": str(grid_path),
        "clean_crops_dir": str(clean_dir),
        "single_preview": str(pipeline_visual),
        "status_counts": dict(status_counts),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO+SAM bottle extraction experiment."
    )
    parser.add_argument(
        "--dataset-dir",
        default="/Users/sumitsaurabh/Desktop/DSL/DATA",
        help="Directory containing training images and _annotations.csv (default: /Users/sumitsaurabh/Desktop/DSL/DATA).",
    )
    parser.add_argument(
        "--annotations-csv",
        default=None,
        help="Path to annotations CSV (defaults to <dataset-dir>/_annotations.csv).",
    )
    parser.add_argument(
        "--yolo-dataset-dir",
        default="datasets/beverage_yolo",
        help="Output YOLO dataset directory.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Output artifacts dir.")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--class-order",
        default="coca-cola,fanta,sprite",
        help="Comma-separated class order for class ids.",
    )

    # YOLO training params
    parser.add_argument("--yolo-base", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--run-name", default="beverage_exp25")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="Example: mps, cpu, 0, cuda:0")
    parser.add_argument(
        "--yolo-weights",
        default=None,
        help="Use existing trained YOLO weights and skip training.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip YOLO training. Requires --yolo-weights.",
    )

    # YOLO + SAM inference params
    parser.add_argument(
        "--sam-checkpoint",
        default="checkpoints/mobile_sam.pt",
        help="Path to SAM/MobileSAM .pth or .pt checkpoint file.",
    )
    parser.add_argument(
        "--sam-model-type",
        default="vit_t",
        help="SAM model type (for MobileSAM typically vit_t).",
    )
    parser.add_argument(
        "--use-mobile-sam",
        dest="use_mobile_sam",
        action="store_true",
        help="Use MobileSAM backend for faster inference.",
    )
    parser.add_argument(
        "--use-standard-sam",
        dest="use_mobile_sam",
        action="store_false",
        help="Use standard SAM backend instead of MobileSAM.",
    )
    parser.set_defaults(use_mobile_sam=True)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=20)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
