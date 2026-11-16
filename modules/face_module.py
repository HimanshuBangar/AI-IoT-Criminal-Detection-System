
import cv2
import numpy as np
from pathlib import Path


def load_face_detector():
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Face detector could not be loaded.")

    return detector


def detect_faces(frame):
    detector = load_face_detector()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )


# Custom recognizer: does not require cv2.face
class CustomFaceRecognizer:

    def __init__(self):
        self.histograms = []
        self.labels = []

    def get_histogram(self, face):
        face = cv2.equalizeHist(face)

        histogram = cv2.calcHist(
            [face], [0], None, [256], [0, 256]
        )

        cv2.normalize(histogram, histogram)

        return histogram

    def train(self, images, labels):
        self.histograms = [
            self.get_histogram(image) for image in images
        ]
        self.labels = np.asarray(labels)

    def predict(self, face):
        if not self.histograms:
            return -1, 100.0

        histogram = self.get_histogram(face)

        distances = [
            cv2.compareHist(
                histogram,
                stored,
                cv2.HISTCMP_BHATTACHARYYA
            )
            for stored in self.histograms
        ]

        best_index = int(np.argmin(distances))

        return (
            int(self.labels[best_index]),
            float(distances[best_index] * 100)
        )


def create_face_recognizer():
    return CustomFaceRecognizer()


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
        person_faces = []

        for image_path in person_folder.iterdir():

            if image_path.suffix.lower() not in [
                ".jpg", ".jpeg", ".png"
            ]:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            gray = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )

            if len(faces) == 0:
                continue

            x, y, w, h = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            person_faces.append(face)

        if person_faces:
            label_map[current_label] = person_name

            for face in person_faces:
                face_images.append(face)
                face_labels.append(current_label)

            current_label += 1

    if not face_images:
        return None, {}

    recognizer.train(
        face_images,
        np.array(face_labels)
    )

    return recognizer, label_map


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
        frame, cv2.COLOR_BGR2GRAY
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
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        matched = confidence <= threshold

        name = (
            label_map.get(label, "Unknown")
            if matched else "Unknown"
        )

        results.append({
            "name": name,
            "confidence": float(confidence),
            "matched": matched,
            "box": (int(x), int(y), int(w), int(h))
        })

    return results
