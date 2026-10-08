import argparse
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "runs/detect/warehouse_baseline/weights/best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an input image")
    args = parser.parse_args()

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=args.image,
        conf=0.25,
        device="mps",
        save=True,
        project=str(Path("runs/inference").resolve()),
        name="warehouse_predictions"
    )

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            print(f"{model.names[class_id]}: {confidence:.2f}")


if __name__ == "__main__":
    main()