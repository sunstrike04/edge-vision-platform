from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO

import io
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="Warehouse Edge Vision API")

MODEL_PATH = Path("runs/detect/warehouse_baseline/weights/best.pt")

# Load the trained model once when the service starts.
model = YOLO(str(MODEL_PATH))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "warehouse_yolov8n",
        "classes": model.names
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image.
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run inference using our trained model.
    results = model.predict(
        source=image,
        conf=0.25,
        device="mps",
        verbose=False
    )

    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])

        detections.append({
            "class": model.names[class_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
        })

    return {
        "filename": file.filename,
        "num_detections": len(detections),
        "detections": detections
    }