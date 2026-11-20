import numpy as np

from src.features.finger_state import extract_finger_states
from src.features.geometry import euclidean_distance, joint_angle_deg, palm_scale


def test_euclidean_distance() -> None:
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 4.0, 0.0])
    assert abs(euclidean_distance(p1, p2) - 5.0) < 1e-6

    p3 = np.array([1.0, 2.0, 2.0])
    assert abs(euclidean_distance(p1, p3) - 3.0) < 1e-6


def test_joint_angle_canonical_cases() -> None:
    # Straight finger: 180 degrees
    straight = joint_angle_deg(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
    )
    assert abs(straight - 180.0) < 1e-6

    # Right angle: 90 degrees
    right = joint_angle_deg(
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    assert abs(right - 90.0) < 1e-6

    # Folded completely back: 0 degrees
    folded = joint_angle_deg(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    assert abs(folded - 0.0) < 1e-6

    # Degenerate zero-length vectors: must return 0.0, never NaN
    degenerate = joint_angle_deg(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    assert not np.isnan(degenerate)
    assert degenerate == 0.0


def test_joint_angle_invariance() -> None:
    """Verify joint_angle_deg is invariant to 3D rotation, translation, and uniform scaling."""
    a = np.array([0.12, -0.05, 0.03])
    b = np.array([0.18, 0.04, -0.02])
    c = np.array([0.21, 0.14, 0.06])
    orig_angle = joint_angle_deg(a, b, c)

    # 90-degree rotation around Z axis (orthogonal matrix R in SO(3))
    rot_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([10.0, -5.0, 3.0])
    scale = 3.5

    a_t = scale * (rot_z_90 @ a) + translation
    b_t = scale * (rot_z_90 @ b) + translation
    c_t = scale * (rot_z_90 @ c) + translation

    transformed_angle = joint_angle_deg(a_t, b_t, c_t)
    assert abs(transformed_angle - orig_angle) < 1e-4


def test_extract_finger_states_synthetic() -> None:
    """Verify open palm produces 5 extended digits and curled fingers produce 0."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.00, 0.00, 0.00]   # WRIST

    # Straight thumb abducted far from pinky MCP
    lm[1] = [-0.03, 0.02, 0.00]
    lm[2] = [-0.06, 0.05, 0.00]
    lm[3] = [-0.09, 0.08, 0.00]
    lm[4] = [-0.12, 0.11, 0.00]

    # Four straight upright fingers (index x=-0.03, middle x=0.0, ring x=0.03, pinky x=0.06)
    xs = [-0.03, 0.00, 0.03, 0.06]
    for finger_idx, x in enumerate(xs):
        base = 5 + finger_idx * 4
        lm[base + 0] = [x, 0.09, 0.00]  # MCP
        lm[base + 1] = [x, 0.13, 0.00]  # PIP
        lm[base + 2] = [x, 0.16, 0.00]  # DIP
        lm[base + 3] = [x, 0.19, 0.00]  # TIP

    assert abs(palm_scale(lm) - 0.09) < 1e-6

    open_states = extract_finger_states(lm)
    assert open_states.extended_count == 5
    assert open_states.as_tuple == (True, True, True, True, True)

    # Now curl all four fingers forward/down at PIP and tuck thumb toward pinky MCP
    curled_lm = lm.copy()
    curled_lm[4] = [0.03, 0.08, -0.02]  # Thumb tucked near pinky MCP (17)
    for finger_idx, x in enumerate(xs):
        base = 5 + finger_idx * 4
        curled_lm[base + 2] = [x, 0.11, -0.04]  # Bent DIP
        curled_lm[base + 3] = [x, 0.08, -0.03]  # Curled TIP

    closed_states = extract_finger_states(curled_lm)
    assert closed_states.extended_count == 0
    assert closed_states.as_tuple == (False, False, False, False, False)


if __name__ == "__main__":
    test_euclidean_distance()
    test_joint_angle_canonical_cases()
    test_joint_angle_invariance()
    test_extract_finger_states_synthetic()
    print("All 3D geometry and finger-state unit tests passed.")