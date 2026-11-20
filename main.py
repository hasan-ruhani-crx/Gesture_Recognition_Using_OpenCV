import logging
import sys
import time

import cv2

from src.capture.camera import WebcamStream
from src.capture.transforms import bgr_to_rgb, flip_horizontal, resize_to_width
from src.features.finger_state import FingerStates, extract_finger_states
from src.perception.hand_tracker import HandDetection, HandTracker
from src.utils.drawing import draw_finger_states, draw_hand_detections
from src.utils.fps import FPSTracker, draw_hud

WINDOW_NAME = "Gesture Recognition - 3D Finger Geometry"
TARGET_WIDTH = 640
MODEL_PATH = "models/hand_landmarker.task"
CONTROLS_HINT = "[m] Mirror [i] IDs [a] Angles [p] Print [q] Quit"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_hand_snapshot(
    logger: logging.Logger,
    detections: list[HandDetection],
    finger_states: list[FingerStates],
) -> None:
    """Log wrist/tip coordinates plus exact 3D joint angles for calibration."""
    if not detections:
        logger.info("Snapshot: no hands currently detected.")
        return

    for idx, (det, state) in enumerate(zip(detections, finger_states)):
        a = state.angles_deg
        logger.info(
            "--- Hand %d: %s (conf=%.2f) | Count=%d %s ---",
            idx,
            det.handedness,
            det.confidence,
            state.extended_count,
            state.format_bitmask(),
        )
        logger.info(
            "  Angles(deg): T_ip=%.1f | I_pip=%.1f | M_pip=%.1f | R_pip=%.1f | P_pip=%.1f | ThumbRatio=%.2f",
            a["thumb_ip"],
            a["index_pip"],
            a["middle_pip"],
            a["ring_pip"],
            a["pinky_pip"],
            state.thumb_abduction_ratio,
        )


def main() -> None:
    configure_logging()
    logger = logging.getLogger("main")
    logger.info("Python %s | Starting Milestone 3.1 3D Finger Geometry.", sys.version.split()[0])

    mirror = True
    show_ids = False
    show_angles = True
    fps_tracker = FPSTracker(alpha=0.1)

    try:
        with (
            WebcamStream(camera_index=0, width=TARGET_WIDTH, height=480) as cam,
            HandTracker(model_path=MODEL_PATH, num_hands=2, is_mirrored=mirror) as tracker,
        ):
            while True:
                frame = cam.read()
                if frame is None:
                    logger.warning("Failed to grab frame from webcam. Exiting capture loop.")
                    break

                processed_bgr = resize_to_width(frame, TARGET_WIDTH)
                if mirror:
                    processed_bgr = flip_horizontal(processed_bgr)

                rgb_frame = bgr_to_rgb(processed_bgr)
                timestamp_ms = int(time.perf_counter() * 1000)
                detections = tracker.detect(rgb_frame, timestamp_ms)

                # Stage 3: Extract rotation- and scale-invariant 3D finger states.
                finger_states = [
                    extract_finger_states(det.landmarks_world) for det in detections
                ]

                fps, latency_ms = fps_tracker.update()
                annotated = draw_hand_detections(processed_bgr, detections, show_ids=show_ids)
                annotated = draw_finger_states(
                    annotated,
                    detections,
                    finger_states,
                    show_angles=show_angles,
                )
                mode_label = f"{'Mirror' if mirror else 'Raw'} | Hands: {len(detections)}"
                display_frame = draw_hud(
                    annotated,
                    fps,
                    latency_ms,
                    mode_label,
                    controls_text=CONTROLS_HINT,
                )

                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("Exit key pressed. Shutting down.")
                    break
                elif key == ord("m"):
                    mirror = not mirror
                    tracker.is_mirrored = mirror
                    logger.info("Mirror toggled: %s", mirror)
                elif key == ord("i"):
                    show_ids = not show_ids
                    logger.info("Show landmark IDs: %s", show_ids)
                elif key == ord("a"):
                    show_angles = not show_angles
                    logger.info("Show angle calibration HUD: %s", show_angles)
                elif key == ord("p"):
                    log_hand_snapshot(logger, detections, finger_states)

    except (RuntimeError, FileNotFoundError) as exc:
        logger.error("Startup/Runtime error: %s", exc)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        logger.info("All OpenCV windows closed.")


if __name__ == "__main__":
    main()