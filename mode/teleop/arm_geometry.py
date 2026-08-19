"""MediaPipe landmarks -> human anatomical joint angles.

This module contains geometry only. It knows nothing about robot servo
limits, calibration, HOME positions, smoothing, Qt, or publishers.

Computes:
    [base, shoulder, elbow, wrist_pitch, wrist_roll]   (5 arm angles)
    gripper  (open/close value in robot servo space)
"""

from __future__ import annotations

import numpy as np

# Pose landmarks
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# Hand landmarks
H_WRIST = 0
H_THUMB_CMC = 1
H_INDEX_MCP = 5
H_MIDDLE_MCP = 9
H_RING_MCP = 13
H_PINKY_MCP = 17


def normalize(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v)
    if length < 1e-8:
        return np.zeros_like(v)
    return v / length


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def signed_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from a to b around axis, in degrees."""
    a = normalize(a)
    b = normalize(b)
    axis = normalize(axis)
    if (
        np.linalg.norm(a) < 1e-8
        or np.linalg.norm(b) < 1e-8
        or np.linalg.norm(axis) < 1e-8
    ):
        return 0.0
    x = np.dot(a, b)
    y = np.dot(axis, np.cross(a, b))
    return float(np.degrees(np.arctan2(y, x)))


def landmark_vector(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y, landmark.z], dtype=float)


def calculate_body_frame(landmarks):
    """Return a torso coordinate frame.

    body_x: person's right
    body_y: person's up
    body_z: torso-forward reference (camera-facing when negated)
    """
    left_shoulder = landmark_vector(landmarks[LEFT_SHOULDER])
    right_shoulder = landmark_vector(landmarks[RIGHT_SHOULDER])
    left_hip = landmark_vector(landmarks[LEFT_HIP])
    right_hip = landmark_vector(landmarks[RIGHT_HIP])

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0

    body_x = normalize(right_shoulder - left_shoulder)
    body_y = normalize(shoulder_center - hip_center)
    body_z = normalize(np.cross(body_x, body_y))
    # Re-orthogonalize
    body_y = normalize(np.cross(body_z, body_x))

    return body_x, body_y, body_z, shoulder_center, hip_center


def _project_horizontal(v: np.ndarray) -> np.ndarray:
    """Project a vector onto the horizontal X-Z plane."""
    return normalize(np.array([v[0], 0.0, v[2]], dtype=float))


def calculate_arm_angles(landmarks, hand_world=None) -> np.ndarray:
    """Calculate [base, shoulder, elbow, wrist_pitch, wrist_roll].

    These are human-space anatomical angles. They are independent of the
    robot's servo HOME values.

    wrist_pitch and wrist_roll default to 90 (neutral) when the hand is
    not detected.
    """
    shoulder = landmark_vector(landmarks[RIGHT_SHOULDER])
    elbow = landmark_vector(landmarks[RIGHT_ELBOW])
    wrist = landmark_vector(landmarks[RIGHT_WRIST])

    upper_dir = normalize(elbow - shoulder)
    forearm_dir = normalize(wrist - elbow)

    body_x, body_y, body_z, _, _ = calculate_body_frame(landmarks)
    body_down = -body_y

    # ----------------------------------------------------------------
    # BASE
    # ----------------------------------------------------------------
    arm_horizontal = _project_horizontal(upper_dir)
    torso_forward = _project_horizontal(-body_z)

    if (
        np.linalg.norm(arm_horizontal) < 0.1
        or np.linalg.norm(torso_forward) < 0.1
    ):
        base_angle = 90.0
    else:
        signed = signed_angle(
            torso_forward,
            arm_horizontal,
            np.array([0.0, 1.0, 0.0]),
        )
        base_angle = 90.0 - signed
        base_angle = float(np.clip(base_angle, 0.0, 180.0))

    # ----------------------------------------------------------------
    # SHOULDER
    # ----------------------------------------------------------------
    shoulder_angle = 180.0 - angle_between(upper_dir, body_down)

    # ----------------------------------------------------------------
    # ELBOW
    # ----------------------------------------------------------------
    elbow_angle = angle_between(-upper_dir, forearm_dir)

    # ----------------------------------------------------------------
    # WRIST PITCH + WRIST ROLL  (require hand landmarks)
    # ----------------------------------------------------------------
    wrist_pitch = 90.0
    wrist_roll = 90.0

    if hand_world is not None:
        h_wrist = landmark_vector(hand_world[H_WRIST])
        h_index = landmark_vector(hand_world[H_INDEX_MCP])
        h_middle = landmark_vector(hand_world[H_MIDDLE_MCP])
        h_pinky = landmark_vector(hand_world[H_PINKY_MCP])

        hand_forward = normalize(h_middle - h_wrist)
        palm_width = normalize(h_pinky - h_index)

        # Palm normal is used for roll.
        palm_normal = normalize(np.cross(palm_width, hand_forward))

        # Wrist pitch: angle between forearm direction and hand-forward around
        # the palm-width axis. 90 = neutral.
        pitch_delta = signed_angle(forearm_dir, hand_forward, palm_width)
        wrist_pitch = float(np.clip(90.0 + pitch_delta, 0.0, 180.0))

        # Wrist roll: compare palm orientation to an upright torso reference
        # projected onto the forearm axis.
        reference = body_y - np.dot(body_y, forearm_dir) * forearm_dir
        reference = normalize(reference)

        palm_projected = (
            palm_normal - np.dot(palm_normal, forearm_dir) * forearm_dir
        )
        palm_projected = normalize(palm_projected)

        if (
            np.linalg.norm(reference) > 0.1
            and np.linalg.norm(palm_projected) > 0.1
        ):
            roll_delta = signed_angle(reference, palm_projected, forearm_dir)
            wrist_roll = float(np.clip(90.0 + roll_delta, 0.0, 180.0))

    return np.array(
        [base_angle, shoulder_angle, elbow_angle, wrist_pitch, wrist_roll],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------

def finger_extension(hand, mcp, pip, dip, tip) -> float:
    """Return a 0..1 extension score for one finger."""
    p_mcp = landmark_vector(hand[mcp])
    p_pip = landmark_vector(hand[pip])
    p_dip = landmark_vector(hand[dip])
    p_tip = landmark_vector(hand[tip])

    pip_angle = angle_between(p_mcp - p_pip, p_dip - p_pip)
    dip_angle = angle_between(p_pip - p_dip, p_tip - p_dip)

    # A straight finger is approximately 180 + 180.
    extension = (pip_angle + dip_angle) / 360.0
    return float(np.clip(extension, 0.0, 1.0))


def calculate_gripper(hand) -> float | None:
    """Return hand openness as a robot servo value.

    Robot convention: open hand = GRIPPER_OPEN, closed fist = GRIPPER_CLOSED.
    Returns None when the hand is not detected.
    """
    if hand is None:
        return None

    fingers = [
        (5, 6, 7, 8),    # index
        (9, 10, 11, 12),  # middle
        (13, 14, 15, 16), # ring
        (17, 18, 19, 20), # pinky
    ]

    extensions = [finger_extension(hand, *f) for f in fingers]
    extension = float(np.clip(np.mean(extensions), 0.0, 1.0))

    # open hand → GRIPPER_OPEN, closed fist → GRIPPER_CLOSED
    from robot_mapping import GRIPPER_OPEN, GRIPPER_CLOSED
    closure = 1.0 - extension
    return float(GRIPPER_OPEN + closure * (GRIPPER_CLOSED - GRIPPER_OPEN))


def calculate_gripper_state(gripper_value: float) -> str:
    """Return a human-readable gripper state string."""
    from robot_mapping import GRIPPER_OPEN, GRIPPER_CLOSED
    mid = (GRIPPER_OPEN + GRIPPER_CLOSED) / 2.0
    return "OPEN" if gripper_value <= mid else "CLOSED"
