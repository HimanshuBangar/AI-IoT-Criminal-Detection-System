
from pathlib import Path
from ultralytics import YOLO
import cv2

from config import MODEL_DIR, WEAPON_CONFIDENCE_THRESHOLD

# Combined Gun + Knife model
CUSTOM_MODEL_PATH = MODEL_DIR / "gun_knife.pt"


def load_weapon_model():
    if not CUSTOM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Weapon model not found: {CUSTOM_MODEL_PATH}"
        )

    model = YOLO(str(CUSTOM_MODEL_PATH))
    print("Weapon model loaded:", model.names)

    return model


def detect_weapons(frame, model, confidence=0.25):

    results = model.predict(
        source=frame,
        conf=confidence,
        imgsz=640,
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            score = float(box.conf[0])

            class_name = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_name": str(class_name).capitalize(),
                "class_id": class_id,
                "confidence": score,
                "box": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )
            })

    return detections


def draw_weapon_detections(frame, detections):

    output_frame = frame.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        confidence = detection["confidence"]
        class_name = detection["class_name"]

        label = f"{class_name}: {confidence:.2f}"

        cv2.rectangle(
            output_frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            output_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return output_frame
