from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data/warehouse/data.yaml",
    epochs=15,
    imgsz=640,
    batch=8,
    device="mps",
    project="runs/detect",
    name="warehouse_baseline"
)