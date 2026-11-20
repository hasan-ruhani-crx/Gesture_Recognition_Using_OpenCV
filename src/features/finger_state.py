from dataclasses import dataclass

import numpy as np

from src.features.geometry import EPS, euclidean_distance, joint_angle_deg, palm_scale

# Anatomical joint triplets: (PIP triplet, DIP triplet) for the four non-thumb fingers.
FINGER_JOINT_TRIPLETS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "index": ((5, 6, 7), (6, 7, 8)),
    "middle": ((9, 10, 11), (10, 11, 12)),
    "ring": ((13, 14, 15), (14, 15, 16)),
    "pinky": ((17, 18, 19), (18, 19, 20)),
}


@dataclass(frozen=True)
class FingerStates:
    """Extension state and diagnostic geometry for all five digits of one hand."""

    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool
    angles_deg: dict[str, float]   # keys: "thumb_ip", "index_pip", "middle_pip", "ring_pip", "pinky_pip"
    thumb_abduction_ratio: float   # d(THUMB_TIP=4, PINKY_MCP=17) / palm_scale

    @property
    def extended_count(self) -> int:
        return sum(self.as_tuple)

    @property
    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)

    def format_bitmask(self) -> str:
        """Compact readable state string, e.g. '[T:0 I:1 M:1 R:0 P:0]'."""
        return (
            f"[T:{int(self.thumb)} I:{int(self.index)} M:{int(self.middle)} "
            f"R:{int(self.ring)} P:{int(self.pinky)}]"
        )


def extract_finger_states(
    landmarks_world: np.ndarray,
    finger_pip_threshold: float = 150.0,
    finger_dip_threshold: float = 150.0,
    thumb_ip_threshold: float = 145.0,
    thumb_abduction_threshold: float = 1.05,
) -> FingerStates:
    """Classify each digit as extended or curled from 3D metric world landmarks."""
    lm = np.asarray(landmarks_world, dtype=np.float64)
    if lm.ndim != 2 or lm.shape[0] < 21:
        raise ValueError(f"Expected landmarks_world of shape (21, 3), got {lm.shape}")

    angles_deg: dict[str, float] = {}
    extended: dict[str, bool] = {}

    # 1. Four hinged fingers: extended when both PIP and DIP joints are sufficiently straight.
    for name, (pip_ids, dip_ids) in FINGER_JOINT_TRIPLETS.items():
        pip_angle = joint_angle_deg(lm[pip_ids[0]], lm[pip_ids[1]], lm[pip_ids[2]])
        dip_angle = joint_angle_deg(lm[dip_ids[0]], lm[dip_ids[1]], lm[dip_ids[2]])
        angles_deg[f"{name}_pip"] = pip_angle
        extended[name] = (
            pip_angle >= finger_pip_threshold and dip_angle >= finger_dip_threshold
        )

    # 2. Thumb special case: combine IP joint straightness + palm-normalized abduction away from pinky knuckle.
    thumb_ip_angle = joint_angle_deg(lm[2], lm[3], lm[4])
    angles_deg["thumb_ip"] = thumb_ip_angle

    ref_scale = palm_scale(lm)
    thumb_abduction_ratio = euclidean_distance(lm[4], lm[17]) / (ref_scale + EPS)
    thumb_extended = (
        thumb_ip_angle >= thumb_ip_threshold
        and thumb_abduction_ratio >= thumb_abduction_threshold
    )

    return FingerStates(
        thumb=thumb_extended,
        index=extended["index"],
        middle=extended["middle"],
        ring=extended["ring"],
        pinky=extended["pinky"],
        angles_deg=angles_deg,
        thumb_abduction_ratio=thumb_abduction_ratio,
    )