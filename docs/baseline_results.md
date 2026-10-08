# Warehouse Perception — Baseline Results

## Dataset

- Source: Addverb Technologies Warehouse Dataset (Roboflow Universe, version 1)
- Classes: AMR, cart, forklift, ladder, pallet, palletjack, person
- Original Roboflow split contained confirmed train/validation leakage through augmented versions of the same image.
- Rebuilt train/validation/test splits by grouping images with the same source filename identifier.
- Cleaned dataset: 16,976 training images, 2,194 validation images, 2,222 test images.
- 4,857 unique filename-based source groups; zero group overlap across splits.

## Model and Training

- Model: YOLOv8n, initialized from pretrained weights
- Epochs: 15
- Image size: 640 × 640
- Batch size: 8
- Training device: Apple M3 Pro (MPS)
- Best checkpoint: `runs/detect/warehouse_baseline/weights/best.pt`

## Held-Out Test Results

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Overall | 0.905 | 0.925 | 0.947 | 0.794 |
| AMR | 0.953 | 0.985 | 0.982 | 0.835 |
| Cart | 0.742 | 0.675 | 0.768 | 0.629 |
| Forklift | 0.958 | 0.985 | 0.992 | 0.830 |
| Ladder | 0.881 | 1.000 | 0.985 | 0.884 |
| Pallet | 0.903 | 0.928 | 0.964 | 0.811 |
| Palletjack | 0.962 | 0.962 | 0.978 | 0.867 |
| Person | 0.934 | 0.938 | 0.962 | 0.700 |

## Observations and Limitations

- The cleaned warehouse dataset provides substantially better class coverage than the initial forklift dataset.
- Cart has the weakest test performance and the fewest examples.
- Grouping by source filename eliminated the identified augmentation leakage, but does not rule out visually duplicated scenes under different filenames.
- Test metrics measure detection accuracy, not deployment performance.
- Jetson latency, FPS, memory usage, and TensorRT performance have not yet been measured.

## Saved Artifacts

- Training metrics: `runs/detect/warehouse_baseline/results.csv`
- Training plots: `runs/detect/warehouse_baseline/results.png`
- Model checkpoint: `runs/detect/warehouse_baseline/weights/best.pt`
- Test evaluation: `runs/detect/warehouse_baseline_test/`
