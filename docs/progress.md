# Edge Vision Platform — Progress

## Completed

- [x] Set up Python environment and Ultralytics YOLOv8
- [x] Run sample object detection
- [x] Train initial forklift-detection baseline
- [x] Evaluate initial dataset and identify class-coverage limitations
- [x] Select a multi-class warehouse dataset
- [x] Identify augmentation-related train/validation leakage
- [x] Rebuild dataset splits by source-image filename group
- [x] Audit class distributions
- [x] Complete one-epoch sanity training
- [x] Train YOLOv8n warehouse baseline for 15 epochs
- [x] Evaluate best checkpoint on held-out test split
- [x] Organize dataset and experiment folders
- [x] Document baseline results

## Next Milestones

- [ ] Implement image inference script
- [ ] Build FastAPI inference endpoints
- [ ] Containerize inference service with Docker
- [ ] Deploy inference pipeline on NVIDIA Jetson
- [ ] Export model to ONNX and optimize with TensorRT
- [ ] Benchmark latency, FPS, and memory usage
- [ ] Add monitoring and experiment tracking
- [ ] Finalize README, architecture, and portfolio documentation

## Current Model

`runs/detect/warehouse_baseline/weights/best.pt`

Test mAP50: 0.9473

Test mAP50-95: 0.7936