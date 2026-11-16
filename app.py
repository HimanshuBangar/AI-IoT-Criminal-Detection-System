
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from config import (
    APP_TITLE,
    APP_ICON,
    APP_LAYOUT,
    REFERENCE_DIR,
    DETECTION_DIR
)

from utils.database import (
    initialize_database,
    add_person,
    get_all_persons,
    get_all_detections
)

from modules.face_module import (
    train_face_recognizer,
    recognize_faces
)

from modules.weapon_module import (
    load_weapon_model,
    detect_weapons,
    draw_weapon_detections
)

from modules.alert_module import (
    face_match_alert,
    weapon_detection_alert
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT
)

initialize_database()


# ==========================================
# 2. LOAD WEAPON MODEL
# ==========================================

@st.cache_resource
def get_weapon_model():
    return load_weapon_model()


# ==========================================
# 3. DASHBOARD HEADER
# ==========================================

st.title("🛡️ AI & IoT Intelligent Security System")

st.write(
    "AI-based Face Recognition and Weapon Detection Dashboard"
)

st.info(
    "This system flags potential matches and detected objects "
    "for human verification. It does not determine criminal identity."
)


# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Register Reference Face",
        "Image Detection",
        "Detection History"
    ]
)


# ==========================================
# 5. DASHBOARD PAGE
# ==========================================

if menu == "Dashboard":

    st.header("Project Dashboard")

    persons = get_all_persons()
    detections = get_all_detections()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Registered Face Records",
            len(persons)
        )

    with col2:
        st.metric(
            "Total Detection Events",
            len(detections)
        )

    st.subheader("Registered Persons")

    if persons:
        for person in persons:
            st.write("•", person[1])
    else:
        st.info("No reference faces registered yet.")

    st.subheader("Recent Detection Events")

    if detections:
        for event in detections[:5]:
            st.write(
                f"**{event[1]}** — {event[2]}"
            )
            st.caption(str(event[5]))
    else:
        st.info("No detection events recorded yet.")


# ==========================================
# 6. REGISTER REFERENCE FACE
# ==========================================

elif menu == "Register Reference Face":

    st.header("Register Reference Face")

    person_name = st.text_input(
        "Enter Person Name"
    )

    uploaded_image = st.file_uploader(
        "Upload Reference Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if st.button("Register Face"):

        if not person_name.strip():
            st.error("Please enter a person name.")

        elif uploaded_image is None:
            st.error("Please upload an image.")

        else:

            safe_name = "".join(
                char for char in person_name.strip()
                if char.isalnum() or char in (" ", "_", "-")
            ).strip()

            if not safe_name:
                st.error("Please enter a valid name.")

            else:

                person_folder = REFERENCE_DIR / safe_name

                person_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                image_path = (
                    person_folder
                    / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                )

                image_bytes = uploaded_image.getvalue()

                image = cv2.imdecode(
                    np.frombuffer(image_bytes, np.uint8),
                    cv2.IMREAD_COLOR
                )

                if image is None:
                    st.error("Invalid image file.")

                else:

                    faces = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2GRAY
                    )

                    detector = cv2.CascadeClassifier(
                        cv2.data.haarcascades
                        + "haarcascade_frontalface_default.xml"
                    )

                    detected = detector.detectMultiScale(
                        faces,
                        scaleFactor=1.1,
                        minNeighbors=5
                    )

                    if len(detected) == 0:
                        st.error(
                            "No face detected. Please upload a clear face photo."
                        )

                    else:

                        cv2.imwrite(
                            str(image_path),
                            image
                        )

                        add_person(
                            safe_name,
                            str(image_path)
                        )

                        st.success(
                            f"Reference face registered for {safe_name}!"
                        )

                        st.image(
                            image,
                            channels="BGR",
                            caption="Registered Reference Image"
                        )


# ==========================================
# 7. IMAGE DETECTION PAGE
# ==========================================

elif menu == "Image Detection":

    st.header("Face and Weapon Detection")

    uploaded_file = st.file_uploader(
        "Upload an Image for Analysis",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        frame = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            st.error("Unable to read uploaded image.")

        else:

            st.image(
                frame,
                channels="BGR",
                caption="Uploaded Image"
            )

            if st.button("Start Detection"):

                output_frame = frame.copy()

                # FACE RECOGNITION

                st.subheader("Face Recognition")

                recognizer, label_map = train_face_recognizer(
                    REFERENCE_DIR
                )

                if recognizer is None:
                    st.warning(
                        "No usable reference faces found. "
                        "Register face photos first."
                    )

                else:

                    face_results = recognize_faces(
                        frame,
                        recognizer,
                        label_map
                    )

                    for face in face_results:

                        x, y, w, h = face["box"]

                        name = face["name"]

                        cv2.rectangle(
                            output_frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            output_frame,
                            name,
                            (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        if face["matched"]:

                            face_match_alert(
                                person_name=name,
                                confidence=face["confidence"]
                            )

                            st.warning(
                                f"Potential face match: {name}. "
                                "Human verification required."
                            )

                    if not face_results:
                        st.info("No face detected.")

                # WEAPON DETECTION

                st.subheader("Weapon Detection")

                with st.spinner("Loading YOLO model..."):

                    weapon_model = get_weapon_model()

                weapon_results = detect_weapons(
                    frame,
                    weapon_model
                )

                output_frame = draw_weapon_detections(
                    output_frame,
                    weapon_results
                )

                for weapon in weapon_results:

                    weapon_detection_alert(
                        weapon_name=weapon["class_name"],
                        confidence=weapon["confidence"]
                    )

                    st.warning(
                        f"Potential object detected: "
                        f"{weapon['class_name']} "
                        f"({weapon['confidence']:.2f})"
                    )

                if not weapon_results:
                    st.info("No supported weapon class detected.")

                # SAVE RESULT IMAGE

                output_path = (
                    DETECTION_DIR
                    / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    output_frame
                )

                st.subheader("Final Detection Result")

                st.image(
                    output_frame,
                    channels="BGR",
                    caption="Analyzed Image"
                )

                st.success("Image analysis completed!")


# ==========================================
# 8. DETECTION HISTORY PAGE
# ==========================================

elif menu == "Detection History":

    st.header("Detection Event History")

    records = get_all_detections()

    if records:

        for record in records:

            with st.expander(
                f"{record[1]} - {record[5]}"
            ):

                st.write("Event:", record[1])

                st.write("Description:", record[2])

                st.write("Confidence / Distance:", record[4])

                st.write("Date and Time:", record[5])

                if record[3] and Path(record[3]).exists():

                    st.image(record[3])

    else:

        st.info("No detection history available yet.")
