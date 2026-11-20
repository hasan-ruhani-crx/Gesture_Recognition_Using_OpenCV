import logging
import sys
import cv2

from src.capture.camera import WebcamStream
from src.capture.transforms import (
    bgr_to_gray,
    bgr_to_rgb,
    flip_horizontal,
    resize_to_width,
)
from src.utils.fps import FPSTracker, draw_hud

WINDOW_NAME = "Gesture Recognition - Transform Lab"
TARGET_WIDTH = 640


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger("main")
    logger.info("Python %s | Starting Milestone 1.2 Transform Lab.", sys.version.split()[0])

    mirror = True
    color_mode = "bgr"  # "bgr" | "gray" | "rgb_bug"
    fps_tracker = FPSTracker(alpha=0.1)

    try:
        with WebcamStream(camera_index=0, width=TARGET_WIDTH, height=480) as cam:
            while True:
                frame = cam.read()
                if frame is None:
                    logger.warning("Failed to grab frame from webcam. Exiting capture loop.")
                    break

                # Normalize width even if the camera driver ignored our resolution request.
                processed = resize_to_width(frame, TARGET_WIDTH)

                if mirror:
                    processed = flip_horizontal(processed)

                if color_mode == "gray":
                    processed = bgr_to_gray(processed)
                    color_label = "Grayscale"
                elif color_mode == "rgb_bug":
                    # Deliberately display RGB bytes in OpenCV's BGR window to observe channel swap.
                    processed = bgr_to_rgb(processed)
                    color_label = "RGB-as-BGR (Bug Demo)"
                else:
                    color_label = "BGR"

                fps, latency_ms = fps_tracker.update()
                mode_label = f"{'Mirror' if mirror else 'Raw'} | {color_label}"
                display_frame = draw_hud(processed, fps, latency_ms, mode_label)

                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("Exit key pressed. Shutting down.")
                    break
                elif key == ord("m"):
                    mirror = not mirror
                    logger.info("Mirror toggled: %s", mirror)
                elif key == ord("g"):
                    color_mode = "bgr" if color_mode == "gray" else "gray"
                    logger.info("Color mode: %s", color_mode)
                elif key == ord("r"):
                    color_mode = "bgr" if color_mode == "rgb_bug" else "rgb_bug"
                    logger.info("Color mode: %s", color_mode)

    except RuntimeError as exc:
        logger.error("Capture error: %s", exc)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        logger.info("All OpenCV windows closed.")


if __name__ == "__main__":
    main()