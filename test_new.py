"""Lightweight YOLO + MobileSAM image extractor.

This script:
1) Finds YOLO best weights automatically (unless provided).
2) Runs YOLO bottle detection on one image.
3) Runs MobileSAM segmentation with one-time image encoding.
4) Saves per-detection extracted crops into test_check/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    try:
        import torch
    except Exception:
        return "cpu"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_best_yolo_weights(search_root: Path = Path("runs")) -> Path:
    """Find newest best.pt under runs/."""
    candidates = list(search_root.rglob("weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No best.pt found under runs/. Pass --weights explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_yolo_model(weights_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is not installed.") from exc
    return YOLO(str(weights_path))


def load_mobile_sam_predictor(
    checkpoint_path: Path,
    model_type: str = "vit_t",
    device: Optional[str] = None,
):
    try:
        from mobile_sam import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError("mobile_sam is not installed.") from exc

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MobileSAM checkpoint not found: {checkpoint_path}")

    if model_type not in sam_model_registry:
        available = ", ".join(sorted(sam_model_registry.keys()))
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=resolve_device(device))
    return SamPredictor(sam)


def detect_boxes(
    yolo_model,
    image_bgr: np.ndarray,
    device: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 20,
) -> List[Dict[str, float]]:
    pred = yolo_model.predict(
        source=image_bgr,
        device=resolve_device(device),
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )[0]

    detections: List[Dict[str, float]] = []
    if pred.boxes is None or len(pred.boxes) == 0:
        return detections

    xyxy = pred.boxes.xyxy.detach().cpu().numpy()
    confs = pred.boxes.conf.detach().cpu().numpy()
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        detections.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(confs[i]),
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def segment_objects_with_mobile_sam(
    image_bgr: np.ndarray,
    boxes: Sequence[Tuple[float, float, float, float]],
    predictor,
) -> Tuple[List[np.ndarray], List[float], float, float]:
    """Encode once and predict per box."""
    encode_start = time.perf_counter()
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    masks_start = time.perf_counter()
    masks: List[np.ndarray] = []
    scores: List[float] = []
    for box in boxes:
        input_box = np.array(box, dtype=np.float32)
        mask_set, score_set, _ = predictor.predict(box=input_box, multimask_output=True)
        best_idx = int(np.argmax(score_set))
        masks.append(mask_set[best_idx].astype(bool))
        scores.append(float(score_set[best_idx]))
    masks_ms = (time.perf_counter() - masks_start) * 1000.0
    return masks, scores, encode_ms, masks_ms


def apply_mask_and_extract(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    padding: int = 4,
) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Empty mask.")

    h, w = image_bgr.shape[:2]
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(w - 1, int(xs.max()) + padding)
    y2 = min(h - 1, int(ys.max()) + padding)

    masked = np.zeros_like(image_bgr)
    masked[mask] = image_bgr[mask]
    return masked[y1 : y2 + 1, x1 : x2 + 1]


def list_images_from_dir(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input image directory not found: {input_dir}")
    images = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in: {input_dir}")
    return images


def run(args: argparse.Namespace) -> None:
    weights_path = Path(args.weights) if args.weights else find_best_yolo_weights()
    checkpoint_path = Path(args.sam_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_path:
        image_paths = [Path(args.image_path)]
        if not image_paths[0].exists():
            raise FileNotFoundError(f"Image not found: {image_paths[0]}")
    else:
        image_paths = list_images_from_dir(Path(args.input_dir))

    print(f"Images to process: {len(image_paths)}")
    print(f"YOLO weights: {weights_path}")
    print(f"MobileSAM checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir.resolve()}")

    yolo_model = load_yolo_model(weights_path)
    sam_predictor = load_mobile_sam_predictor(
        checkpoint_path=checkpoint_path,
        model_type=args.sam_model_type,
        device=args.device,
    )

    total_detections = 0
    total_saved = 0
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Image read failed, skipping: {image_path}")
            continue

        print(f"\nImage: {image_path}")
        yolo_start = time.perf_counter()
        detections = detect_boxes(
            yolo_model=yolo_model,
            image_bgr=image_bgr,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
        )
        yolo_ms = (time.perf_counter() - yolo_start) * 1000.0
        print(f"YOLO time: {yolo_ms:.2f} ms")
        print(f"Detections: {len(detections)}")
        total_detections += len(detections)
        if not detections:
            continue

        boxes = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in detections]
        masks, scores, encode_ms, masks_ms = segment_objects_with_mobile_sam(
            image_bgr=image_bgr,
            boxes=boxes,
            predictor=sam_predictor,
        )
        print(f"SAM encode time: {encode_ms:.2f} ms")
        print(f"SAM masks time: {masks_ms:.2f} ms")

        saved = 0
        for idx, (det, mask, score) in enumerate(zip(detections, masks, scores), start=1):
            try:
                crop = apply_mask_and_extract(image_bgr, mask)
                crop = cv2.resize(
                    crop,
                    (args.crop_size, args.crop_size),
                    interpolation=cv2.INTER_AREA,
                )
                out_path = output_dir / f"{image_path.stem}_det_{idx:02d}.png"
                cv2.imwrite(str(out_path), crop)
                print(
                    f"[{idx:02d}] saved={out_path.name} "
                    f"conf={det['confidence']:.3f} sam={score:.3f}"
                )
                saved += 1
                total_saved += 1
            except Exception as exc:
                print(f"[{idx:02d}] skipped: {exc}")

        print(f"Saved crops: {saved}/{len(detections)}")

    print(
        f"\nDone. Images={len(image_paths)} detections={total_detections} "
        f"saved={total_saved} output={output_dir.resolve()}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO + MobileSAM extraction for images."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Optional single image path. If omitted, all images in --input-dir are processed.",
    )
    parser.add_argument(
        "--input-dir",
        default="/Users/sumitsaurabh/Desktop/DSL/test_image",
        help="Directory scanned when image_path is not provided.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="YOLO weights path (.pt). If omitted, newest runs/**/weights/best.pt is used.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="checkpoints/mobile_sam.pt",
        help="MobileSAM checkpoint path.",
    )
    parser.add_argument("--sam-model-type", default="vit_t", help="MobileSAM model type.")
    parser.add_argument("--device", default=None, help="Example: mps, cpu, 0, cuda:0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--output-dir", default="test_check")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
