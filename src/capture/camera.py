import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebcamStream:
    """Resource-safe wrapper around OpenCV's VideoCapture."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480) -> None:
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> "WebcamStream":
        """Open the webcam stream and negotiate resolution with the OS driver."""
        self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            self.release()
            raise RuntimeError(
                f"Failed to open webcam at index {self.camera_index}. "
                "Check that no other application is locking the camera."
            )

        # Request target dimensions; the driver picks the closest supported hardware mode.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Webcam %d opened (requested: %dx%d, actual: %dx%d)",
            self.camera_index,
            self.requested_width,
            self.requested_height,
            actual_w,
            actual_h,
        )
        return self

    def read(self) -> np.ndarray | None:
        """Read a single BGR frame as a uint8 NumPy array of shape (H, W, 3)."""
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("WebcamStream is not open. Call start() or use a 'with' block.")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return frame

    def release(self) -> None:
        """Release the underlying OS camera handle safely."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Webcam %d released.", self.camera_index)

    def __enter__(self) -> "WebcamStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Always runs—even if an exception crashes the main loop—preventing zombie camera locks.
        self.release()