"""YOLOv8 training utilities."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve training device with Apple MPS preference on macOS."""
    if device:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_detection_only_data_yaml(data_yaml: str | Path) -> None:
    """Ensure YOLO data config is set to single-class bottle detection."""
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}")

    text = data_yaml.read_text(encoding="utf-8")
    has_nc_one = re.search(r"^\s*nc\s*:\s*1\s*$", text, flags=re.MULTILINE) is not None
    has_bottle_name = re.search(r"\bbottle\b", text, flags=re.IGNORECASE) is not None

    if not has_nc_one:
        raise ValueError(
            f"{data_yaml} must define 'nc: 1' for detection-only training."
        )
    if not has_bottle_name:
        raise ValueError(
            f"{data_yaml} must define bottle as the only class name."
        )


def train_yolov8(
    data_yaml: str | Path,
    model_checkpoint: str = "yolov8n.pt",
    epochs: int = 12,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "beverage_exp",
    device: Optional[str] = None,
    workers: int = 8,
    enforce_single_class: bool = True,
) -> str:
    """Train a YOLOv8 detector and return best weights path."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    if enforce_single_class:
        validate_detection_only_data_yaml(data_yaml)

    model = YOLO(model_checkpoint)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "project": project,
        "name": name,
        "exist_ok": True,
        "workers": int(workers),
    }
    train_kwargs["device"] = resolve_device(device)

    results = model.train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", Path(project) / name))
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"

    if best_path.exists():
        return str(best_path)
    if last_path.exists():
        return str(last_path)
    raise FileNotFoundError(
        f"Training completed but no weights found at {best_path} or {last_path}."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on beverage dataset.")
    parser.add_argument("--data-yaml", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--model-checkpoint", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="beverage_exp")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    best_weights = train_yolov8(
        data_yaml=args.data_yaml,
        model_checkpoint=args.model_checkpoint,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
    )
    print(best_weights)


if __name__ == "__main__":
    main()
