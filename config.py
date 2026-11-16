
from pathlib import Path

# ==========================================
# 1. MAIN PROJECT DIRECTORY
# ==========================================

BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# 2. PROJECT FOLDER PATHS
# ==========================================

DATA_DIR = BASE_DIR / "data"

REFERENCE_DIR = DATA_DIR / "reference_faces"

DETECTION_DIR = DATA_DIR / "detections"

MODEL_DIR = BASE_DIR / "models"

ASSETS_DIR = BASE_DIR / "assets"


# ==========================================
# 3. DATABASE CONFIGURATION
# ==========================================

DATABASE_PATH = DATA_DIR / "project_database.db"


# ==========================================
# 4. APPLICATION SETTINGS
# ==========================================

APP_TITLE = "AI & IoT Based Intelligent Security System"

APP_ICON = "🛡️"

APP_LAYOUT = "wide"


# ==========================================
# 5. SUPPORTED IMAGE FORMATS
# ==========================================

ALLOWED_IMAGE_TYPES = [
    "jpg",
    "jpeg",
    "png"
]


# ==========================================
# 6. SUPPORTED VIDEO FORMATS
# ==========================================

ALLOWED_VIDEO_TYPES = [
    "mp4",
    "avi",
    "mov",
    "mkv"
]


# ==========================================
# 7. FACE RECOGNITION SETTINGS
# ==========================================

FACE_MATCH_THRESHOLD = 0.60


# ==========================================
# 8. WEAPON DETECTION SETTINGS
# ==========================================

WEAPON_CONFIDENCE_THRESHOLD = 0.40

WEAPON_CLASSES = [
    "gun",
    "knife"
]


# ==========================================
# 9. IOT ALERT SETTINGS
# ==========================================

IOT_ENABLED = False

SERIAL_PORT = "COM3"

BAUD_RATE = 9600


# ==========================================
# 10. AUTOMATIC FOLDER CREATION
# ==========================================

for folder in [
    DATA_DIR,
    REFERENCE_DIR,
    DETECTION_DIR,
    MODEL_DIR,
    ASSETS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)
