import time
import cv2
import numpy as np


class FPSTracker:
    """Tracks smoothed FPS and frame latency using an Exponential Moving Average."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._prev_time: float | None = None
        self._ema_dt: float | None = None

    def update(self) -> tuple[float, float]:
        """Record a new frame timestamp and return (smoothed_fps, smoothed_ms)."""
        now = time.perf_counter()

        # First frame only establishes the baseline timestamp.
        if self._prev_time is None:
            self._prev_time = now
            return 0.0, 0.0

        dt = now - self._prev_time
        self._prev_time = now

        if dt <= 0.0:
            if self._ema_dt and self._ema_dt > 0.0:
                return 1.0 / self._ema_dt, self._ema_dt * 1000.0
            return 0.0, 0.0

        if self._ema_dt is None:
            self._ema_dt = dt
        else:
            self._ema_dt = self.alpha * dt + (1.0 - self.alpha) * self._ema_dt

        fps = 1.0 / self._ema_dt
        latency_ms = self._ema_dt * 1000.0
        return fps, latency_ms


def draw_hud(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    mode_label: str,
) -> np.ndarray:
    """Return a copy of the frame with FPS, latency, and mode telemetry drawn on top."""
    # Convert 1-channel grayscale to 3-channel BGR for display so colored HUD stays legible.
    display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()

    status_text = f"FPS: {fps:5.1f} | {latency_ms:5.1f} ms | Mode: {mode_label}"
    controls_text = "[m] Mirror  [g] Gray  [r] RGB-Bug  [q/Esc] Quit"

    # Dark backdrop bar keeps text readable against bright backgrounds.
    cv2.rectangle(display, (8, 8), (520, 62), (20, 20, 20), thickness=-1)

    cv2.putText(
        display,
        status_text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        controls_text,
        (16, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return display