# YOLOv8n Baseline Results

## Dataset

Roboflow Forklift Detection dataset

- Training images: 2,391
- Validation images: 89
- Classes: forklift, person
- Validation instances:
  - Forklift: 110
  - Person: 8

The validation set contains only one image with person annotations, so
person-class metrics are not considered representative.

## Training Configuration

- Model: YOLOv8n
- Pretrained weights: COCO
- Epochs: 30
- Image size: 640
- Batch size: 8
- Training device: Apple M3 Pro (MPS)

## Validation Results

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Forklift | 0.932 | 0.878 | 0.949 | 0.651 |
| Person | 0.000 | 0.000 | 0.000 | 0.000 |
| Overall | 0.466 | 0.439 | 0.475 | 0.325 |

## Baseline Observations

The model performs strongly on the primary forklift class, achieving
0.949 mAP50 and 0.932 precision.

The person-class results are not currently meaningful because the
validation split contains only 8 person instances from a single image.
Future dataset work should improve class balance and validation coverage.