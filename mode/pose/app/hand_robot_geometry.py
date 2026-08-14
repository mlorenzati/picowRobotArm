"""Coordinate-frame helpers inspired by Brevinbanks/ur5_mediapipe_motion.

The original project uses a fixed X rotation to move MediaPipe hand-world
coordinates into a robot/world convention, then builds a tool basis from the
wrist/index/pinky landmarks.  We keep those useful ideas, but deliberately do
not copy its translation/scaling into joint-angle calculations: translation
and non-uniform scale do not belong in orientation math.
"""

from __future__ import annotations

import numpy as np


# Same fixed axis conversion used by HandWorld2RobotWorld: Rx(-90 deg).
MEDIA_PIPE_TO_ROBOT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def hand_world_to_robot_world(point: np.ndarray) -> np.ndarray:
    """Convert a MediaPipe world point to the robot/world axis convention."""
    return MEDIA_PIPE_TO_ROBOT @ np.asarray(point, dtype=float)


def transform_landmarks(landmarks):
    """Return transformed copies of MediaPipe landmarks as Nx3 numpy data."""
    return np.array(
        [hand_world_to_robot_world([p.x, p.y, p.z]) for p in landmarks],
        dtype=float,
    )


def hand_basis(hand_world):
    """Build a stable right-handed hand basis from wrist/index/pinky.

    Columns are palm-width, palm-forward and palm-normal.  This follows the
    useful frame construction used by the reference repository while using
    the midpoint of index/pinky for a less noisy palm-forward direction.
    """
    p = transform_landmarks(hand_world)
    wrist = p[0]
    index = p[5]
    pinky = p[17]

    width = normalize(pinky - index)
    forward = normalize((index + pinky) * 0.5 - wrist)

    # Make forward perpendicular to width before computing the normal.
    forward = normalize(forward - np.dot(forward, width) * width)
    normal = normalize(np.cross(width, forward))

    if np.linalg.norm(width) < 0.1 or np.linalg.norm(forward) < 0.1:
        return None

    return np.column_stack((width, forward, normal))
