import numpy as np

EPS = 1e-6


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute the L2 (Euclidean) distance between two N-dimensional points."""
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    return float(np.linalg.norm(b - a))


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the interior angle (in degrees, [0, 180]) at vertex b formed by a-b-c."""
    pa = np.asarray(a, dtype=np.float64)
    pb = np.asarray(b, dtype=np.float64)
    pc = np.asarray(c, dtype=np.float64)

    # Both bone vectors must originate from the vertex joint b.
    u = pa - pb
    v = pc - pb

    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    if norm_u < EPS or norm_v < EPS:
        return 0.0

    # Clip to [-1, 1] to guard arccos against IEEE-754 float overshoot (e.g. -1.0000001).
    cos_theta = float(np.dot(u, v) / (norm_u * norm_v))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def palm_scale(landmarks: np.ndarray) -> float:
    """Return the rigid palm reference length between WRIST (0) and MIDDLE_MCP (9)."""
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 21:
        raise ValueError(f"Expected landmarks of shape (21, D), got {pts.shape}")
    return euclidean_distance(pts[0], pts[9])