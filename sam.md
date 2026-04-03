# SAM / MobileSAM Inference Optimization Notes

## Overview
This document explains all SAM-related changes made to optimize multi-object inference in the YOLOv8 + SAM pipeline.

Goal achieved:
- Run SAM image encoding once per image.
- Reuse encoded image features for all detected bounding boxes.
- Add MobileSAM support for faster inference.
- Add timing logs to measure latency hotspots.

---

## 1) Why This Change Was Needed

### Previous inefficiency pattern
A common anti-pattern in multi-object segmentation is:

```python
for box in boxes:
    predictor.set_image(image)  # expensive image encoder run each iteration
    predictor.predict(box=box)
```

This recomputes SAM image embeddings for every box and makes inference very slow when multiple bottles are detected.

### Correct optimized pattern

```python
predictor.set_image(image)  # once per image
for box in boxes:
    predictor.predict(box=box)  # lightweight prompt-based calls
```

This is now the implemented behavior.

---

## 2) `pipeline_infer.py` Changes

### 2.1 MobileSAM-enabled SAM loader
Function updated: `load_sam_predictor(...)`

What changed:
- Added argument: `use_mobile_sam: bool = False`.
- If `use_mobile_sam=True`, imports from `mobile_sam`.
- If `use_mobile_sam=False`, imports from `segment_anything` (standard SAM).
- Supports fallback to `vit_t` when using MobileSAM and model type is not explicitly valid.

Result:
- One loader now supports both standard SAM and MobileSAM.

### 2.2 New helper function required by design
Added function:

```python
def segment_objects_with_sam(image, boxes, predictor) -> List[np.ndarray]
```

Implementation behavior:
- Calls `predictor.set_image(...)` once.
- Loops through all boxes and runs `predictor.predict(box=...)`.
- Returns one mask per box.

Internal detailed variant also added:
- `_segment_objects_with_sam_detailed(...)`
- Returns masks + scores + timing metrics (`encode_ms`, `masks_ms`).

### 2.3 Post-processing split from SAM prompting
`process_all_detections(...)` was refactored to accept precomputed masks:
- Inputs now include `masks` and optional `mask_scores`.
- No image encoding inside this function.
- Handles only:
  - background removal,
  - tight crop,
  - resize to `224x224`,
  - per-object output packaging.

### 2.4 Timing logs added
In `run_yolo_sam_on_image(...)`, timing now prints:
- `YOLO time: X ms`
- `SAM encode time: X ms`
- `SAM masks time: X ms`

This helps compare standard SAM vs MobileSAM and diagnose bottlenecks.

### 2.5 Batch defaults aligned to MobileSAM
In `run_pipeline_batch(...)`:
- Default `sam_model_type` changed to `vit_t`.
- Default `use_mobile_sam` changed to `True`.

---

## 3) `main.py` Changes

### 3.1 MobileSAM defaults and flags
SAM CLI config updated:
- `--sam-checkpoint` default: `checkpoints/mobile_sam.pt`
- `--sam-model-type` default: `vit_t`
- Added toggle flags:
  - `--use-mobile-sam`
  - `--use-standard-sam`
- Default mode set to MobileSAM (`use_mobile_sam=True`).

### 3.2 Inference call wiring
`run_pipeline_batch(...)` now receives:
- `use_mobile_sam=args.use_mobile_sam`

So experiment runs use MobileSAM by default unless explicitly overridden.

---

## 4) `test.py` Changes

### 4.1 MobileSAM flags added to interactive inference
Added CLI flags:
- `--use-mobile-sam`
- `--use-standard-sam`
- Default set to MobileSAM.

`--sam-model-type` default updated to `vit_t`.

### 4.2 Inference path wiring
`run_full_pipeline(...)` now accepts and forwards:
- `use_mobile_sam`

This ensures `test.py` can switch between standard SAM and MobileSAM at runtime.

---

## 5) Weights and Environment Setup

### 5.1 Checkpoint stored
Downloaded MobileSAM weights to:
- `checkpoints/mobile_sam.pt`

### 5.2 Package installed
Installed:
- `mobile_sam`

### 5.3 Runtime dependency note
`mobile_sam` requires `torch` in the active Python environment.
If missing, install it before running inference.

---

## 6) What Was Intentionally Not Changed

Per constraints, these were not modified:
- YOLO detection logic itself.
- Dataset preparation/training logic for this SAM optimization task.
- EfficientNet model/training.

Only SAM usage and inference orchestration were optimized.

---

## 7) Practical Usage

### Default (MobileSAM, faster)
`main.py` now defaults to MobileSAM with:
- `--sam-checkpoint checkpoints/mobile_sam.pt`
- `--sam-model-type vit_t`
- `use_mobile_sam=True`

### Force standard SAM
Use:
- `--use-standard-sam`
- compatible standard SAM checkpoint + model type (for example `vit_h`).

---

## 8) Expected Impact

You should observe:
- Faster multi-object segmentation on images with several bottles.
- Larger speed gains as number of detections increases.
- Clear timing visibility for YOLO vs SAM encode vs SAM per-box mask generation.
