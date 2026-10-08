from ultralytics import YOLO

model = YOLO("runs/detect/warehouse_baseline/weights/best.pt")

metrics = model.val(
    data="data/warehouse/data.yaml",
    split="test",
    device="mps",
    project="runs/detect",
    name="warehouse_baseline_test"
)

print("\nWarehouse Baseline — Test Results")
print("---------------------------------")
print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")
print(f"mAP50:       {metrics.box.map50:.4f}")
print(f"mAP50-95:    {metrics.box.map:.4f}")