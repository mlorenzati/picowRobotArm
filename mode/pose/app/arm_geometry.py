"""MediaPipe landmarks -> human anatomical joint angles.

This module contains geometry only. It knows nothing about robot servo
limits, calibration, HOME positions, smoothing, Qt, or ZMQ.
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
H_INDEX_MCP = 5
H_MIDDLE_MCP = 9
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
    body_z: torso-forward reference

    The torso frame is a reference frame. It is not used to replace
    the actual upper-arm geometry for shoulder elevation.
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

    # Re-orthogonalize body_y so the frame remains orthogonal.
    body_y = normalize(np.cross(body_z, body_x))

    return body_x, body_y, body_z, shoulder_center, hip_center


def _project_horizontal(v: np.ndarray) -> np.ndarray:
    """Project a vector onto the camera/world horizontal X-Z plane."""
    return normalize(np.array([v[0], 0.0, v[2]], dtype=float))


def calculate_arm_angles(landmarks, hand_world=None) -> np.ndarray:
    """Calculate [base, shoulder, elbow, wrist_pitch, wrist_roll].

    These are human-space anatomical/reference angles. They are deliberately
    independent of the robot's servo HOME values.
    """
    shoulder = landmark_vector(landmarks[RIGHT_SHOULDER])
    elbow = landmark_vector(landmarks[RIGHT_ELBOW])
    wrist = landmark_vector(landmarks[RIGHT_WRIST])

    upper_dir = normalize(elbow - shoulder)
    forearm_dir = normalize(wrist - elbow)

    body_x, body_y, body_z, _, _ = calculate_body_frame(landmarks)
    body_down = -body_y

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------
    # 90 = arm centered on torso-forward reference.
    # Positive/negative values represent horizontal rotation.
    arm_horizontal = _project_horizontal(upper_dir)
    torso_forward = _project_horizontal(body_z)

    if (
        np.linalg.norm(arm_horizontal) < 0.1
        or np.linalg.norm(torso_forward) < 0.1
    ):
        base_angle = 0.0
    else:
        base_angle = signed_angle(
            torso_forward,
            arm_horizontal,
            np.array([0.0, 1.0, 0.0]),
        )

    # --------------------------------------------------------
    # SHOULDER
    # --------------------------------------------------------
    # Exactly follows the agreed convention:
    #
    # shoulder = 180 - angle(shoulder->elbow, body_down)
    #
    # hanging down -> 180
    # horizontal   -> 90
    # pointing up  -> 0
    shoulder_angle = 180.0 - angle_between(upper_dir, body_down)

    # --------------------------------------------------------
    # ELBOW
    # --------------------------------------------------------
    # Elbow vertex vectors are elbow->shoulder and elbow->wrist.
    # Straight = 180, folded = 0.
    elbow_angle = angle_between(-upper_dir, forearm_dir)

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

        # Wrist pitch: 90 is straight/neutral. Positive and negative
        # deviations are preserved instead of collapsing to 0..180.
        pitch_delta = signed_angle(
            forearm_dir,
            hand_forward,
            palm_width,
        )
        wrist_pitch = float(np.clip(90.0 + pitch_delta, 0.0, 180.0))

        # Wrist roll: compare palm orientation around the forearm axis
        # against an upright torso-relative reference.
        reference = body_y - np.dot(body_y, forearm_dir) * forearm_dir
        reference = normalize(reference)

        palm_projected = (
            palm_normal
            - np.dot(palm_normal, forearm_dir) * forearm_dir
        )
        palm_projected = normalize(palm_projected)

        if (
            np.linalg.norm(reference) > 0.1
            and np.linalg.norm(palm_projected) > 0.1
        ):
            roll_delta = signed_angle(
                reference,
                palm_projected,
                forearm_dir,
            )
            wrist_roll = float(np.clip(90.0 + roll_delta, 0.0, 180.0))

    return np.array(
        [
            base_angle,
            shoulder_angle,
            elbow_angle,
            wrist_pitch,
            wrist_roll,
        ],
        dtype=float,
    )


def finger_extension(hand, mcp, pip, dip, tip) -> float:
    """Return a 0..1 extension score for one finger."""
    p_wrist = landmark_vector(hand[H_WRIST])
    p_mcp = landmark_vector(hand[mcp])
    p_pip = landmark_vector(hand[pip])
    p_dip = landmark_vector(hand[dip])
    p_tip = landmark_vector(hand[tip])

    pip_angle = angle_between(p_mcp - p_pip, p_wrist - p_pip)
    dip_angle = angle_between(p_pip - p_dip, p_tip - p_dip)

    # A straight finger is approximately 180 + 180.
    extension = (pip_angle + dip_angle) / 360.0
    return float(np.clip(extension, 0.0, 1.0))


def calculate_gripper(hand) -> float | None:
    """Return hand openness as 0..1, where 1 is fully open."""
    if hand is None:
        return None

    fingers = [
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ]

    extensions = [
        finger_extension(hand, *finger)
        for finger in fingers
    ]

    return float(np.clip(np.mean(extensions), 0.0, 1.0))
