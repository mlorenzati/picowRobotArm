"""Human joint movement -> robot servo angles.

This module contains calibration, neutral-relative mapping, limits, direction,
sensitivity and smoothing. It contains no MediaPipe or Qt code.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = (
    "base",
    "shoulder",
    "elbow",
    "wrist_pitch",
    "wrist_roll",
    "gripper",
)

# Robot neutral/home values requested for this project.
HOME = np.array([90.0, 180.0, 180.0, 90.0, 90.0, 100.0], dtype=float)

SERVO_MIN = np.zeros(6, dtype=float)
SERVO_MAX = np.full(6, 180.0, dtype=float)

# Human movement corresponding approximately to +/-90 robot degrees.
HUMAN_RANGE = np.array(
    [70.0, 60.0, 70.0, 70.0, 90.0, 90.0],
    dtype=float,
)

SENSITIVITY = np.ones(6, dtype=float)

DIRECTION = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=float,
)

SMOOTHING = 0.20

# Hand-space convention:
# 1.0 = open, 0.0 = closed.
GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 180.0


class RobotMapper:
    """Owns the complete neutral-relative human->servo mapping."""

    def __init__(self):
        self.neutral_human: np.ndarray | None = None
        self.neutral_gripper: float | None = None
        self.filtered = HOME.copy()

    @property
    def calibrated(self) -> bool:
        return self.neutral_human is not None

    def calibrate(self, human_angles, gripper) -> bool:
        if human_angles is None:
            return False

        self.neutral_human = np.asarray(
            human_angles, dtype=float
        ).copy()

        self.neutral_gripper = (
            None if gripper is None else float(gripper)
        )

        # Calibration starts from the known robot neutral.
        self.filtered = HOME.copy()
        return True

    def reset_home(self) -> np.ndarray:
        self.filtered = HOME.copy()
        return HOME.copy()

    def map(self, human_angles, gripper) -> np.ndarray:
        """Map human movement relative to calibration into servo space."""
        if not self.calibrated:
            return HOME.copy()

        human_angles = np.asarray(human_angles, dtype=float)

        robot = HOME.copy()

        # Each joint is mapped independently from its own calibrated neutral.
        delta = human_angles - self.neutral_human
        normalized = delta / HUMAN_RANGE[:5]
        robot[:5] = (
            HOME[:5]
            + normalized * 90.0 * SENSITIVITY[:5] * DIRECTION[:5]
        )

        # Gripper is already in the requested robot convention:
        # open=100, closed=180. Do not apply neutral-relative mapping here.
        if gripper is not None:
            robot[5] = float(gripper)

        return np.clip(robot, SERVO_MIN, SERVO_MAX)

    def update(self, human_angles, gripper) -> np.ndarray:
        target = self.map(human_angles, gripper)

        self.filtered = (
            self.filtered * (1.0 - SMOOTHING)
            + target * SMOOTHING
        )
        return self.filtered.copy()
