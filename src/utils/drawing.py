import cv2
import numpy as np

from src.features.finger_state import FingerStates
from src.perception.hand_tracker import HandDetection

# Canonical 21-point skeletal bone topology.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger & palm base
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

FINGERTIP_IDS: frozenset[int] = frozenset({4, 8, 12, 16, 20})

# Distinct BGR colors per hand to make multi-hand inspection effortless.
HAND_COLORS: dict[str, tuple[int, int, int]] = {
    "Right": (80, 220, 120),   # Green
    "Left": (255, 180, 60),    # Sky-blue/cyan
}


def draw_hand_detections(
    frame: np.ndarray,
    detections: list[HandDetection],
    show_ids: bool = False,
) -> np.ndarray:
    """Return a copy of frame with hand skeletons, bounding boxes, and labels drawn."""
    canvas = frame.copy()

    for det in detections:
        pts = det.landmarks_px
        color = HAND_COLORS.get(det.handedness, (0, 255, 255))

        # 1. Draw skeletal bone connections first so joint circles render cleanly on top.
        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = (int(pts[start_idx, 0]), int(pts[start_idx, 1]))
            p2 = (int(pts[end_idx, 0]), int(pts[end_idx, 1]))
            cv2.line(canvas, p1, p2, (230, 230, 230), thickness=2, lineType=cv2.LINE_AA)

        # 2. Draw all 21 joints (highlight wrist=0 and fingertips differently).
        for idx, (u, v) in enumerate(pts):
            center = (int(u), int(v))
            if idx in FINGERTIP_IDS:
                cv2.circle(canvas, center, 6, (40, 120, 255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, center, 6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            elif idx == 0:
                cv2.circle(canvas, center, 6, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, center, 6, color, thickness=2, lineType=cv2.LINE_AA)
            else:
                cv2.circle(canvas, center, 4, color, thickness=-1, lineType=cv2.LINE_AA)

            if show_ids:
                cv2.putText(
                    canvas,
                    str(idx),
                    (center[0] + 6, center[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 3. Draw padded bounding box and handedness + confidence header.
        pad = 14
        x_min = max(0, int(pts[:, 0].min()) - pad)
        y_min = max(0, int(pts[:, 1].min()) - pad)
        x_max = min(canvas.shape[1] - 1, int(pts[:, 0].max()) + pad)
        y_max = min(canvas.shape[0] - 1, int(pts[:, 1].max()) + pad)

        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, thickness=2)

        label = f"{det.handedness} ({det.confidence:.2f})"
        label_y = max(24, y_min - 8)
        cv2.putText(
            canvas,
            label,
            (x_min, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return canvas


def draw_finger_states(
    frame: np.ndarray,
    detections: list[HandDetection],
    finger_states: list[FingerStates],
    show_angles: bool = False,
) -> np.ndarray:
    """Overlay finger counts, [T I M R P] bitmasks, and optional live calibration angles."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    for hand_idx, (det, state) in enumerate(zip(detections, finger_states)):
        pts = det.landmarks_px
        color = HAND_COLORS.get(det.handedness, (0, 255, 255))

        x_min = max(0, int(pts[:, 0].min()) - 14)
        y_max = min(height - 10, int(pts[:, 1].max()) + 32)

        summary = f"Count: {state.extended_count}  {state.format_bitmask()}"
        cv2.putText(
            canvas,
            summary,
            (x_min, y_max),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

        # Optional bottom-left calibration panel showing exact live angles and thumb ratio.
        if show_angles:
            box_y = height - 42 * (hand_idx + 1) - 8
            if box_y >= 68:
                a = state.angles_deg
                line1 = (
                    f"{det.handedness}: "
                    f"T_ip={a['thumb_ip']:5.1f}  "
                    f"I={a['index_pip']:5.1f}  "
                    f"M={a['middle_pip']:5.1f}  "
                    f"R={a['ring_pip']:5.1f}  "
                    f"P={a['pinky_pip']:5.1f}"
                )
                line2 = f"Thumb Abduction Ratio: {state.thumb_abduction_ratio:.2f}"

                cv2.rectangle(
                    canvas,
                    (8, box_y),
                    (min(width - 8, 632), box_y + 38),
                    (20, 20, 20),
                    thickness=-1,
                )
                cv2.putText(
                    canvas,
                    line1,
                    (14, box_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    line2,
                    (14, box_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

    return canvas