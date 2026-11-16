import cv2
from pathlib import Path


# ==========================================
# 1. OPEN WEBCAM OR VIDEO FILE
# ==========================================

def open_video_source(source=0):
    """
    Opens webcam or a video file.

    source = 0 for default webcam
    source = video file path for uploaded video
    """

    if isinstance(source, Path):
        source = str(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Unable to open video source: {source}"
        )

    return cap


# ==========================================
# 2. READ VIDEO FRAME
# ==========================================

def read_frame(cap):
    """
    Reads one frame from the video.
    Returns success status and frame.
    """

    success, frame = cap.read()

    if not success:
        return False, None

    return True, frame


# ==========================================
# 3. RELEASE VIDEO SOURCE
# ==========================================

def release_video(cap):
    """
    Releases webcam or video resource.
    """

    if cap is not None:
        cap.release()


# ==========================================
# 4. GET VIDEO INFORMATION
# ==========================================

def get_video_info(cap):
    """
    Returns video width, height and FPS.
    """

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    return {
        "width": width,
        "height": height,
        "fps": fps
    }


# ==========================================
# 5. CHECK VIDEO FILE
# ==========================================

def is_valid_video(video_path):
    """
    Checks whether a video file can be opened.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        return False

    cap = cv2.VideoCapture(str(video_path))

    valid = cap.isOpened()

    cap.release()

    return valid
