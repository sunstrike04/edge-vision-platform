# Edge Vision Platform

A warehouse object detection pipeline built with YOLOv8, PyTorch, and FastAPI. I trained a model to detect seven warehouse object classes, rebuilt the dataset splits after finding overlap between augmented images, and implemented local image inference through a Python script and HTTP API.

I'm currently extending the project toward NVIDIA Jetson deployment and TensorRT inference.

## Results

Trained YOLOv8n for 15 epochs on a seven-class warehouse dataset. Evaluation on the cleaned test split produced:

| Metric | Result |
|---|---:|
| Precision | 90.46% |
| Recall | 92.46% |
| mAP@50 | 94.73% |
| mAP@50:95 | 79.36% |

**Per-class mAP@50:**

| Class | mAP@50 |
|---|---:|
| AMR | 0.982 |
| Cart | 0.768 |
| Forklift | 0.992 |
| Ladder | 0.985 |
| Pallet | 0.964 |
| Pallet Jack | 0.978 |
| Person | 0.962 |

Cart detection was noticeably weaker than the other classes. Full evaluation results are documented in `docs/baseline_results.md`.

## Dataset Preparation

I used the Addverb Technologies warehouse dataset from Roboflow, which includes AMRs, carts, forklifts, ladders, pallets, pallet jacks, and people.

While inspecting the dataset, I found that augmented versions of the same source images appeared across the original training and validation splits. This could inflate validation performance because the model may have already encountered nearly identical images during training.

I wrote `training/create_clean_split.py` to group images by their source filename identifiers and generate new splits without overlapping groups.

| Split | Images |
|---|---:|
| Train | 16,976 |
| Validation | 2,194 |
| Test | 2,222 |

I also checked that each class was represented in the validation and test sets.

The grouping prevents overlap among identified source filename groups, although it doesn't rule out visually duplicated images with different identifiers.

## Running the Project

### Setup

```bash
git clone https://github.com/sunstrike04/edge-vision-platform.git
cd edge-vision-platform

python3 -m venv venv
source venv/bin/activate

pip install ultralytics fastapi "uvicorn[standard]" python-multipart
```

The project currently uses Python 3.13.

**Model checkpoint:** The inference scripts expect the trained weights at:

```text
runs/detect/warehouse_baseline/weights/best.pt
```

Datasets and model checkpoints are excluded from Git. The trained checkpoint isn't currently distributed with the repository, so the inference commands require a local copy of the weights.

### Image Inference

```bash
python inference/predict.py path/to/image.jpg
```

This runs detection on an image, prints predicted classes and confidence scores, and saves an annotated output.

### FastAPI Inference

Start the server:

```bash
python -m uvicorn inference.api:app --host 127.0.0.1 --port 8000
```

The API provides:

- `GET /health` — Returns service status and supported model classes.
- `POST /predict` — Accepts an image and returns detected classes, confidence scores, and bounding boxes.

Test an image upload:

```bash
curl -F "file=@path/to/image.jpg" \
  http://127.0.0.1:8000/predict
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Inference currently uses Apple's MPS backend.

## Repository Structure

```text
training/
  analyze_dataset.py       # Dataset statistics and class distribution
  create_clean_split.py    # Group-based dataset splitting
  train_baseline.py        # YOLOv8n training
  evaluate.py              # Test-set evaluation

inference/
  predict.py               # Local image inference
  api.py                   # FastAPI inference service

docs/
  baseline_results.md      # Model evaluation details
  progress.md              # Development notes
```

## Next Steps

- [x] Dataset analysis and leakage investigation
- [x] YOLOv8n baseline training and evaluation
- [x] Local image inference
- [x] FastAPI inference endpoints
- [ ] Dockerize the inference service
- [ ] Add latency and throughput benchmarks
- [ ] Deploy to NVIDIA Jetson
- [ ] Export to ONNX and TensorRT
- [ ] Compare inference performance across backends

The main focus going forward is making the inference pipeline portable to NVIDIA edge hardware and measuring the performance differences between deployment backends.
