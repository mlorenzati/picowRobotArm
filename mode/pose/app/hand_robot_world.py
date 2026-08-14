"""Hand 6-DOF diagnostic frame.

The previous implementation built the hand frame entirely from landmark
*differences*. That is good for orientation, but translation cancels out:

    (landmark + translation) - (wrist + translation) == landmark - wrist

This module therefore keeps translation and orientation separate:

POSITION
    X/Y come from the hand wrist in the camera image, centered at the image
    center. These values change when the whole hand moves left/right/up/down,
    even if the hand does not rotate.

    Z is the MediaPipe hand-world wrist depth value. MediaPipe hand-world Z is
    useful as a relative depth signal, but is intentionally labelled as such;
    it is not claimed to be an absolute camera-space position.

ORIENTATION
    A wrist-relative basis is built from wrist -> index/middle/pinky. This is
    translation-invariant and therefore responds to rotation of the hand.

The result is a diagnostic 6-DOF signal. Robot retargeting is intentionally
kept in robot_mapping.py so these measurements can be inspected before servo
mapping is tuned.
"""
from __future__ import annotations

import numpy as np

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)


def landmark_vector(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=float)


def rx_minus_90(v):
    """Convert MediaPipe hand-world axes to the reference robot convention."""
    x, y, z = np.asarray(v, dtype=float)
    return np.array([x, z, -y], dtype=float)


def _euler_xyz(rotation):
    sy = np.clip(rotation[0, 2], -1.0, 1.0)
    pitch = np.degrees(np.arcsin(-sy))
    roll = np.degrees(np.arctan2(rotation[1, 2], rotation[2, 2]))
    yaw = np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0]))
    return float(roll), float(pitch), float(yaw)


def hand2robotworld(hand_landmarks, hand_world=None):
    """Return hand position + orientation as a diagnostic 6-DOF frame.

    ``hand_landmarks`` are the normal MediaPipe normalized image landmarks and
    are used for XY translation. ``hand_world`` supplies relative hand-world
    depth and orientation.

    Position convention:
        X = (wrist.x - 0.5) * 2       left/right, center = 0
        Y = (0.5 - wrist.y) * 2       down/up, center = 0
        Z = hand-world wrist.z        relative depth signal

    Thus a pure image-plane translation changes XYZ while preserving RPY.
    The XY values are normalized camera coordinates, not metres.
    """
    if hand_landmarks is None:
        return None

    image_wrist = landmark_vector(hand_landmarks[WRIST])
    position = np.array([
        (image_wrist[0] - 0.5) * 2.0,
        (0.5 - image_wrist[1]) * 2.0,
        0.0,
    ], dtype=float)

    result = {
        "position": position,
        "position_units": "normalized_camera_xy + relative_depth_z",
        "image_position": image_wrist.copy(),
        "x_axis": None,
        "y_axis": None,
        "z_axis": None,
        "rotation": None,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }

    if hand_world is None:
        return result

    pts = [rx_minus_90(landmark_vector(p)) for p in hand_world]
    wrist = pts[WRIST]
    index = pts[INDEX_MCP]
    middle = pts[MIDDLE_MCP]
    pinky = pts[PINKY_MCP]

    # Keep depth as a separate signal. It is relative to the MediaPipe hand
    # world frame, so don't pretend it is a camera-space metre value.
    # Use the original MediaPipe hand-world Z for the relative depth signal.
    # The Rx(-90) conversion is only for the orientation frame.
    position[2] = float(hand_world[WRIST].z)

    # Build a stable hand orientation basis. Translation is removed here on
    # purpose: orientation must not change simply because the hand translates.
    x_axis = normalize(index - wrist)
    middle_dir = normalize(middle - wrist)
    pinky_dir = normalize(pinky - wrist)

    z_axis = normalize(np.cross(x_axis, middle_dir))
    if np.linalg.norm(z_axis) < 1e-8:
        z_axis = normalize(np.cross(x_axis, pinky_dir))

    if np.linalg.norm(z_axis) < 1e-8:
        result["position"] = position
        return result

    y_axis = normalize(np.cross(z_axis, x_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    roll, pitch, yaw = _euler_xyz(rotation)

    result.update({
        "position": position,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "rotation": rotation,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    })
    return result
