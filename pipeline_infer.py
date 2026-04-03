"""YOLO + SAM inference pipeline for clean bottle extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


@dataclass
class InferenceResult:
    """Container for one end-to-end YOLO+SAM inference output."""

    image_path: str
    status: str
    detections: List[Dict[str, float]]
    processed_objects: List[Dict[str, object]]
    selected_detection: Optional[Dict[str, float]]
    original_bgr: Optional[np.ndarray]
    yolo_overlay_bgr: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    masked_full_bgr: Optional[np.ndarray]
    extracted_bgr: Optional[np.ndarray]
    sam_score: Optional[float]
    crop_box_xyxy: Optional[Tuple[int, int, int, int]]
    message: str


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve PyTorch device string."""
    if device:
        return device
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_yolo_model(weights_path: str):
    """Load YOLO model from weights."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics") from exc

    return YOLO(weights_path)


def load_sam_predictor(
    checkpoint_path: str,
    model_type: str = "vit_h",
    device: Optional[str] = None,
    use_mobile_sam: bool = False,
):
    """Load Segment Anything predictor (standard SAM or MobileSAM)."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    resolved_device = resolve_device(device)
    if use_mobile_sam:
        try:
            from mobile_sam import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "mobile_sam is not installed. Run: "
                "pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
            ) from exc
    else:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "segment-anything is not installed. Run: pip install segment-anything"
            ) from exc

    resolved_model_type = model_type
    if resolved_model_type not in sam_model_registry:
        if use_mobile_sam and "vit_t" in sam_model_registry:
            resolved_model_type = "vit_t"
        else:
            available = ", ".join(sorted(sam_model_registry.keys()))
            raise ValueError(
                f"Unknown SAM model_type '{model_type}'. Available: {available}"
            )

    sam = sam_model_registry[resolved_model_type](checkpoint=str(checkpoint))
    sam.to(device=resolved_device)
    return SamPredictor(sam)


def detect_with_yolo(
    yolo_model,
    image_bgr: np.ndarray,
    device: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 20,
) -> List[Dict[str, float]]:
    """Run YOLO detection and return sorted detections."""
    resolved_device = resolve_device(device)
    pred = yolo_model.predict(
        source=image_bgr,
        device=resolved_device,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )[0]

    detections: List[Dict[str, float]] = []
    boxes = pred.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        detections.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(confs[i]),
                # Detection-only pipeline: all objects are treated as "bottle".
                "class_id": 0,
            }
        )

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def draw_yolo_detections(
    image_bgr: np.ndarray,
    detections: Sequence[Dict[str, float]],
    class_names: Optional[Dict[int, str]] = None,
    top_only: bool = False,
) -> np.ndarray:
    """Draw YOLO boxes on image."""
    canvas = image_bgr.copy()
    draw_list = detections[:1] if top_only else detections
    _ = class_names  # YOLO class labels are intentionally ignored.

    for det in draw_list:
        x1, y1, x2, y2 = (
            int(round(det["x1"])),
            int(round(det["y1"])),
            int(round(det["x2"])),
            int(round(det["y2"])),
        )
        conf = det["confidence"]
        label = f"bottle {conf:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 255, 60), 2)
        cv2.putText(
            canvas,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 255, 60),
            2,
            cv2.LINE_AA,
        )
    return canvas


def segment_with_sam_box_prompt(
    predictor,
    image_bgr: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    set_image: bool = True,
) -> Tuple[np.ndarray, float]:
    """Generate SAM mask using a YOLO bounding box prompt."""
    if set_image:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

    input_box = np.array(box_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(bool)
    best_score = float(scores[best_idx])
    return best_mask, best_score


def apply_mask_and_extract_object(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 4,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Apply mask, remove background, and return full masked image + tight crop."""
    if mask.dtype != bool:
        mask = mask.astype(bool)

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Empty mask returned by SAM.")

    h, w = image_bgr.shape[:2]
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(w - 1, int(xs.max()) + padding)
    y2 = min(h - 1, int(ys.max()) + padding)

    masked_full = np.full_like(image_bgr, background_color, dtype=np.uint8)
    masked_full[mask] = image_bgr[mask]
    extracted = masked_full[y1 : y2 + 1, x1 : x2 + 1]

    return masked_full, extracted, (x1, y1, x2, y2)


def _normalize_xyxy_boxes(
    boxes: Sequence[Sequence[float] | Dict[str, float]],
) -> List[Tuple[float, float, float, float]]:
    """Normalize boxes from tuples/lists/dicts to xyxy tuples."""
    normalized: List[Tuple[float, float, float, float]] = []
    for box in boxes:
        if isinstance(box, dict):
            normalized.append(
                (
                    float(box["x1"]),
                    float(box["y1"]),
                    float(box["x2"]),
                    float(box["y2"]),
                )
            )
        else:
            if len(box) != 4:
                raise ValueError(f"Expected box length 4, got {len(box)}")
            normalized.append(
                (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            )
    return normalized


def _segment_objects_with_sam_detailed(
    image: np.ndarray,
    boxes: Sequence[Sequence[float] | Dict[str, float]],
    predictor,
):
    """Segment all boxes with a single SAM image encoding and return details."""
    normalized_boxes = _normalize_xyxy_boxes(boxes)

    encode_start = time.perf_counter()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    masks_start = time.perf_counter()
    masks_out: List[np.ndarray] = []
    scores_out: List[float] = []
    for box_xyxy in normalized_boxes:
        input_box = np.array(box_xyxy, dtype=np.float32)
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        best_idx = int(np.argmax(scores))
        masks_out.append(masks[best_idx].astype(bool))
        scores_out.append(float(scores[best_idx]))
    masks_ms = (time.perf_counter() - masks_start) * 1000.0

    return masks_out, scores_out, {"encode_ms": encode_ms, "masks_ms": masks_ms}


def segment_objects_with_sam(
    image: np.ndarray,
    boxes: Sequence[Sequence[float] | Dict[str, float]],
    predictor,
) -> List[np.ndarray]:
    """Segment all boxes with a single SAM image encoding.

    Args:
        image: input image (numpy array)
        boxes: list of bounding boxes
        predictor: SAM predictor instance

    Returns:
        List of masks (one per box)
    """
    masks_out, _, _ = _segment_objects_with_sam_detailed(
        image=image,
        boxes=boxes,
        predictor=predictor,
    )
    return masks_out


def resize_for_classification(
    crop_bgr: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize extracted crop to classifier input size."""
    if crop_bgr.size == 0:
        raise ValueError("Cannot resize empty crop.")
    return cv2.resize(crop_bgr, target_size, interpolation=cv2.INTER_AREA)


def process_all_detections(
    image_bgr: np.ndarray,
    boxes: Sequence[Dict[str, float]],
    masks: Sequence[np.ndarray],
    mask_scores: Optional[Sequence[float]] = None,
    target_size: Tuple[int, int] = (224, 224),
) -> List[Dict[str, object]]:
    """Post-process pre-segmented detections via masking/cropping/resizing."""
    if len(boxes) != len(masks):
        raise ValueError(
            f"boxes and masks length mismatch: {len(boxes)} vs {len(masks)}"
        )
    if mask_scores is not None and len(mask_scores) != len(masks):
        raise ValueError(
            f"mask_scores and masks length mismatch: {len(mask_scores)} vs {len(masks)}"
        )

    processed: List[Dict[str, object]] = []
    for idx, det in enumerate(boxes):
        box_xyxy = (det["x1"], det["y1"], det["x2"], det["y2"])
        mask = masks[idx]
        sam_score = float(mask_scores[idx]) if mask_scores is not None else None
        out: Dict[str, object] = {
            "box": [int(round(v)) for v in box_xyxy],
            "confidence": float(det["confidence"]),
            "cropped_image": None,
            "mask": None,
            "masked_full_bgr": None,
            "crop_box_xyxy": None,
            "sam_score": None,
            "status": "sam_failed",
            "message": "SAM/masking step not run.",
        }

        try:
            masked_full, extracted, crop_box = apply_mask_and_extract_object(image_bgr, mask)
            resized_crop = resize_for_classification(extracted, target_size=target_size)

            out.update(
                {
                    "cropped_image": resized_crop,
                    "mask": mask,
                    "masked_full_bgr": masked_full,
                    "crop_box_xyxy": [int(v) for v in crop_box],
                    "sam_score": sam_score,
                    "status": "ok",
                    "message": "Success",
                }
            )
        except Exception as exc:  # pragma: no cover - runtime safety path
            out["message"] = f"SAM/masking step failed: {exc}"

        processed.append(out)

    return processed


def run_yolo_sam_on_image(
    image_path: str | Path,
    yolo_model,
    sam_predictor,
    device: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 20,
    log_timings: bool = True,
) -> InferenceResult:
    """Run YOLO->SAM->clean extraction on one image for all detections."""
    image_path = str(image_path)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return InferenceResult(
            image_path=image_path,
            status="error",
            detections=[],
            processed_objects=[],
            selected_detection=None,
            original_bgr=None,
            yolo_overlay_bgr=None,
            mask=None,
            masked_full_bgr=None,
            extracted_bgr=None,
            sam_score=None,
            crop_box_xyxy=None,
            message="Failed to read image.",
        )

    yolo_start = time.perf_counter()
    detections = detect_with_yolo(
        yolo_model=yolo_model,
        image_bgr=image_bgr,
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
    )
    yolo_ms = (time.perf_counter() - yolo_start) * 1000.0
    if log_timings:
        print(f"YOLO time: {yolo_ms:.2f} ms")

    yolo_overlay = draw_yolo_detections(
        image_bgr=image_bgr,
        detections=detections,
        class_names=None,
        top_only=False,
    )

    if not detections:
        return InferenceResult(
            image_path=image_path,
            status="no_detection",
            detections=[],
            processed_objects=[],
            selected_detection=None,
            original_bgr=image_bgr,
            yolo_overlay_bgr=yolo_overlay,
            mask=None,
            masked_full_bgr=None,
            extracted_bgr=None,
            sam_score=None,
            crop_box_xyxy=None,
            message="No YOLO detections found.",
        )

    boxes_xyxy = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in detections]
    masks, mask_scores, sam_timing = _segment_objects_with_sam_detailed(
        image=image_bgr,
        boxes=boxes_xyxy,
        predictor=sam_predictor,
    )
    if log_timings:
        print(f"SAM encode time: {sam_timing['encode_ms']:.2f} ms")
        print(f"SAM masks time: {sam_timing['masks_ms']:.2f} ms")

    processed_objects = process_all_detections(
        image_bgr=image_bgr,
        boxes=detections,
        masks=masks,
        mask_scores=mask_scores,
        target_size=(224, 224),
    )
    successful_objects = [
        obj for obj in processed_objects if obj.get("status") == "ok" and obj.get("cropped_image") is not None
    ]
    selected = detections[0] if detections else None
    primary_obj = successful_objects[0] if successful_objects else processed_objects[0]

    if len(successful_objects) == len(processed_objects):
        status = "ok"
        message = f"Processed all detections ({len(processed_objects)})."
    elif successful_objects:
        status = "partial_success"
        message = (
            f"Processed {len(successful_objects)}/{len(processed_objects)} detections."
        )
    else:
        status = "sam_failed"
        message = "SAM/masking failed for all detections."

    return InferenceResult(
        image_path=image_path,
        status=status,
        detections=detections,
        processed_objects=processed_objects,
        selected_detection=selected,
        original_bgr=image_bgr,
        yolo_overlay_bgr=yolo_overlay,
        mask=primary_obj.get("mask"),
        masked_full_bgr=primary_obj.get("masked_full_bgr"),
        extracted_bgr=primary_obj.get("cropped_image"),
        sam_score=primary_obj.get("sam_score"),
        crop_box_xyxy=tuple(primary_obj["crop_box_xyxy"]) if primary_obj.get("crop_box_xyxy") else None,
        message=message,
    )


def run_pipeline_batch(
    image_paths: Sequence[str | Path],
    yolo_weights: str,
    sam_checkpoint: str,
    sam_model_type: str = "vit_t",
    use_mobile_sam: bool = True,
    device: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 20,
    log_timings: bool = True,
) -> List[InferenceResult]:
    """Load models once and run inference on multiple images."""
    resolved_device = resolve_device(device)
    yolo_model = load_yolo_model(yolo_weights)
    sam_predictor = load_sam_predictor(
        checkpoint_path=sam_checkpoint,
        model_type=sam_model_type,
        use_mobile_sam=use_mobile_sam,
        device=resolved_device,
    )

    results: List[InferenceResult] = []
    for image_path in image_paths:
        result = run_yolo_sam_on_image(
            image_path=image_path,
            yolo_model=yolo_model,
            sam_predictor=sam_predictor,
            device=resolved_device,
            conf=conf,
            iou=iou,
            max_det=max_det,
            log_timings=log_timings,
        )
        results.append(result)
    return results
