import logging
import sys
import time
import cv2

from src.capture.camera import WebcamStream

WINDOW_NAME = "Gesture Recognition - Raw Feed"
FPS_LOG_INTERVAL = 30


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger("main")
    logger.info("Python %s | Gesture Recognition Pipeline initialized.", sys.version.split()[0])

    first_frame_logged = False
    frame_count = 0
    window_start = time.perf_counter()

    try:
        with WebcamStream(camera_index=0, width=640, height=480) as cam:
            while True:
                frame = cam.read()
                if frame is None:
                    logger.warning("Failed to grab frame from webcam. Exiting capture loop.")
                    break

                # Inspect tensor layout once on the very first frame
                if not first_frame_logged:
                    height, width = frame.shape[:2]
                    center_bgr = frame[height // 2, width // 2]
                    logger.info(
                        "First frame metadata | shape=%s | dtype=%s | nbytes=%d | center_bgr=%s",
                        frame.shape,
                        frame.dtype,
                        frame.nbytes,
                        center_bgr.tolist(),
                    )
                    first_frame_logged = True

                # Measure rolling FPS every 30 frames to avoid terminal I/O bottlenecks
                frame_count += 1
                if frame_count % FPS_LOG_INTERVAL == 0:
                    now = time.perf_counter()
                    elapsed = now - window_start
                    fps = FPS_LOG_INTERVAL / elapsed if elapsed > 0 else 0.0
                    avg_ms = (elapsed / FPS_LOG_INTERVAL) * 1000.0
                    logger.info(
                        "Throughput (%d frames): %.2f FPS | %.2f ms/frame",
                        FPS_LOG_INTERVAL,
                        fps,
                        avg_ms,
                    )
                    window_start = now

                cv2.imshow(WINDOW_NAME, frame)

                # Poll OS GUI events for 1 ms; mask to lower 8 bits for clean ASCII matching
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or Esc
                    logger.info("Exit key pressed. Shutting down.")
                    break

    except RuntimeError as exc:
        logger.error("Capture error: %s", exc)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        logger.info("All OpenCV windows closed.")


if __name__ == "__main__":
    main()