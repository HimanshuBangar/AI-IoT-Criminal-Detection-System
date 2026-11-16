import cv2
import numpy as np
from pathlib import Path


# ==========================================
# 1. LOAD FACE DETECTOR
# ==========================================

def load_face_detector():
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Face detector could not be loaded.")

    return detector


# ==========================================
# 2. DETECT FACES IN IMAGE OR VIDEO FRAME
# ==========================================

def detect_faces(frame):
    detector = load_face_detector()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    return faces


# ==========================================
# 3. CREATE FACE RECOGNIZER
# ==========================================

def create_face_recognizer():

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face module is unavailable. "
            "Install opencv-contrib-python."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    return recognizer


# ==========================================
# 4. TRAIN MODEL USING REFERENCE PHOTOS
# ==========================================

def train_face_recognizer(reference_dir):

    reference_dir = Path(reference_dir)

    if not reference_dir.exists():
        raise FileNotFoundError(
            "Reference faces folder does not exist."
        )

    recognizer = create_face_recognizer()

    detector = load_face_detector()

    face_images = []
    face_labels = []
    label_map = {}

    current_label = 0

    for person_folder in reference_dir.iterdir():

        if not person_folder.is_dir():
            continue

        person_name = person_folder.name

        label_map[current_label] = person_name

        for image_path in person_folder.iterdir():

            if image_path.suffix.lower() not in [
                ".jpg", ".jpeg", ".png"
            ]:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )

            if len(faces) == 0:
                continue

            # Use the largest detected face
            x, y, w, h = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(
                face,
                (200, 200)
            )

            face_images.append(face)
            face_labels.append(current_label)

        current_label += 1

    if len(face_images) == 0:
        return None, {}

    recognizer.train(
        face_images,
        np.array(face_labels)
    )

    return recognizer, label_map


# ==========================================
# 5. RECOGNIZE FACES
# ==========================================

def recognize_faces(
    frame,
    recognizer,
    label_map,
    threshold=70
):

    if recognizer is None:
        return []

    detector = load_face_detector()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    results = []

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (200, 200)
        )

        label, confidence = recognizer.predict(face)

        if confidence <= threshold:
            name = label_map.get(label, "Unknown")
            matched = True
        else:
            name = "Unknown"
            matched = False

        results.append({
            "name": name,
            "confidence": float(confidence),
            "matched": matched,
            "box": (int(x), int(y), int(w), int(h))
        })

    return results