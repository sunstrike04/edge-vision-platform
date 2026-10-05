from ultralytics import YOLO

model = YOLO("runs/detect/runs/yolov8n_baseline/weights/best.pt")

metrics = model.val(
    data="data/warehouse/data.yaml",
    split="val",
    device="mps"
)

print("\nBaseline Evaluation")
print("-------------------")
print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")
print(f"mAP50:       {metrics.box.map50:.4f}")
print(f"mAP50-95:    {metrics.box.map:.4f}")