
from pathlib import Path
from ultralytics import YOLO

from config import MODEL_DIR, WEAPON_CONFIDENCE_THRESHOLD


# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

CUSTOM_MODEL_PATH = MODEL_DIR / "best.pt"

DEFAULT_MODEL = "yolov8n.pt"

WEAPON_KEYWORDS = [
    "gun",
    "pistol",
    "rifle",
    "firearm",
    "knife"
]


# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

def load_weapon_model():

    if CUSTOM_MODEL_PATH.exists():
        model_path = str(CUSTOM_MODEL_PATH)
        print("Loading custom weapon detection model...")
    else:
        model_path = DEFAULT_MODEL
        print("Loading default YOLO model...")

    model = YOLO(model_path)

    return model


# ==========================================
# 3. DETECT WEAPONS IN FRAME
# ==========================================

def detect_weapons(
    frame,
    model,
    confidence=WEAPON_CONFIDENCE_THRESHOLD
):

    results = model.predict(
        source=frame,
        conf=confidence,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            class_name = model.names[class_id].lower()

            # Check whether detected object is a weapon
            if not any(
                keyword in class_name
                for keyword in WEAPON_KEYWORDS
            ):
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_name": class_name,
                "confidence": float(box.conf[0]),
                "box": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )
            })

    return detections


# ==========================================
# 4. DRAW DETECTION BOXES
# ==========================================

def draw_weapon_detections(frame, detections):

    import cv2

    output_frame = frame.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        name = detection["class_name"]

        confidence = detection["confidence"]

        label = f"{name}: {confidence:.2f}"

        cv2.rectangle(
            output_frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            output_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    return output_frame
