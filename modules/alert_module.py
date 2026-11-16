
import serial
import time

from config import (
    IOT_ENABLED,
    SERIAL_PORT,
    BAUD_RATE
)

from utils.database import add_detection


# ==========================================
# 1. DATABASE ALERT
# ==========================================

def save_alert(
    event_type,
    description="",
    image_path=None,
    confidence=None
):

    add_detection(
        event_type=event_type,
        description=description,
        image_path=image_path,
        confidence=confidence
    )

    print(f"Alert saved: {event_type}")


# ==========================================
# 2. IOT ALERT
# ==========================================

def send_iot_alert(message="ALERT"):

    if not IOT_ENABLED:
        print("IoT alert is disabled.")
        return False

    try:
        with serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=2
        ) as arduino:

            time.sleep(2)

            arduino.write(
                (message + "\n").encode("utf-8")
            )

        print("IoT alert sent successfully.")
        return True

    except serial.SerialException as error:

        print(f"IoT connection error: {error}")
        return False


# ==========================================
# 3. FACE MATCH ALERT
# ==========================================

def face_match_alert(person_name, confidence=None,
                     image_path=None):

    description = (
        f"Potential face match detected: {person_name}. "
        "Human verification required."
    )

    save_alert(
        event_type="FACE_MATCH",
        description=description,
        image_path=image_path,
        confidence=confidence
    )

    send_iot_alert("FACE_ALERT")


# ==========================================
# 4. WEAPON DETECTION ALERT
# ==========================================

def weapon_detection_alert(weapon_name,
                           confidence=None,
                           image_path=None):

    description = (
        f"Potential weapon detected: {weapon_name}. "
        "Human verification required."
    )

    save_alert(
        event_type="WEAPON_DETECTED",
        description=description,
        image_path=image_path,
        confidence=confidence
    )

    send_iot_alert("WEAPON_ALERT")


# ==========================================
# 5. GENERAL SECURITY ALERT
# ==========================================

def general_security_alert(description):

    save_alert(
        event_type="SECURITY_ALERT",
        description=description
    )
