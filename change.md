# YOLO + SAM + EfficientNet Refactor Plan

## 1. Problem
- YOLO currently predicts both bounding boxes and class labels.
- Classification is already handled by a downstream EfficientNet model.
- This duplicates model responsibility and creates architectural redundancy.

## 2. Solution
- Convert YOLO training to single-class detection.
- Relabel all training annotations as `bottle`.
- Force `class_id = 0` for every annotation while preserving bounding box coordinates.

## 3. Expected Behavior Change
- YOLO will only detect bottle locations.
- YOLO will no longer distinguish between beverage brands/classes (Pepsi, Sprite, etc.).
- EfficientNet will perform all brand/product classification downstream.

## 4. Pipeline Update
Old pipeline:
- YOLO (detect + classify) -> SAM -> EfficientNet (redundant classification path)

New pipeline:
- YOLO (detect bottle only) -> SAM -> clean crop -> EfficientNet (classification)

## 5. Multi-object Handling
- Inference must process all detected bounding boxes, not a single best/top box.
- Each detection box is independently segmented and cropped.
- One input image may produce multiple bottle outputs.

## 6. Output Change
- Replace single-image return with a list of per-detection outputs.

Example output shape:
```python
[
    {
        "box": [x1, y1, x2, y2],
        "cropped_image": <image_array>
    },
    {
        "box": [x3, y3, x4, y4],
        "cropped_image": <image_array>
    }
]
```
