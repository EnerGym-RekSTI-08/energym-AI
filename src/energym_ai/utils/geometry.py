from __future__ import annotations
import numpy as np


def calculate_angle(
    a: tuple[float, float] | tuple[float, float, float],
    b: tuple[float, float] | tuple[float, float, float],
    c: tuple[float, float] | tuple[float, float, float],
) -> float:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def horizontal_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return abs(p1[0] - p2[0])


def body_sway(
    shoulder: tuple[float, float],
    hip: tuple[float, float],
    reference_shoulder: tuple[float, float],
    reference_hip: tuple[float, float],
) -> float:
    current = calculate_angle(
        (shoulder[0], shoulder[1] - 1),
        shoulder,
        hip,
    )
    reference = calculate_angle(
        (reference_shoulder[0], reference_shoulder[1] - 1),
        reference_shoulder,
        reference_hip,
    )
    return abs(current - reference)
