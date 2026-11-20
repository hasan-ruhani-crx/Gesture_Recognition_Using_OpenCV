import cv2
import numpy as np


def flip_horizontal(frame: np.ndarray) -> np.ndarray:
    """Mirror a frame left-to-right while keeping memory C-contiguous."""
    # Flip code 1 = horizontal axis (around y-axis).
    # Prefer cv2.flip over frame[:, ::-1] so C++ backends receive positive strides.
    return cv2.flip(frame, 1)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to standard RGB channel order."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def bgr_to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a 3-channel BGR frame to 1-channel ITU-R BT.601 grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def resize_to_width(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Resize a frame to target_width while strictly preserving aspect ratio."""
    if target_width <= 0:
        raise ValueError(f"target_width must be positive, got {target_width}")

    height, width = frame.shape[:2]
    if width == target_width:
        return frame

    scale = target_width / width
    target_height = max(1, round(height * scale))

    # Downscaling requires area averaging to prevent aliasing; upscaling uses bilinear.
    interpolation = cv2.INTER_AREA if target_width < width else cv2.INTER_LINEAR

    # OpenCV expects (width, height), opposite of NumPy's (height, width) shape.
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)