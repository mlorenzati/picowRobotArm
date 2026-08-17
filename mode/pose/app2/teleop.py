"""
teleop.py

MediaPipe + PySide6 hand tracking + local 6DOF IK simulator.

NO ROS / ROS2.

Pipeline:

    Camera
       |
       v
    MediaPipe Tasks
       |
       v
    HandWorld
       |
       v
    HandWorld2RobotWorld()
       |
       v
    H2R position + orientation
       |
       v
    6DOF geometric IK
       |
       v
    Servo-space angles
       |
       +---- Console
       |
       +---- PySide arm simulator

Install:

    pip install mediapipe opencv-python PySide6 numpy transforms3d

The first execution downloads:

    models/pose_landmarker_full.task
    models/hand_landmarker.task

IMPORTANT:
The arm geometry and servo zero/direction values below are deliberately
configuration values. The IK geometry is real math, but the final servo
calibration must be fitted to the physical arm.
"""

import sys
import os
import time
import threading
import urllib.request
import math

import cv2
import numpy as np
import mediapipe as mp

from transforms3d.quaternions import mat2quat, quat2axangle

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QGridLayout,
    QHBoxLayout,
    QGroupBox,
)


# ============================================================================
# Configuration
# ============================================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

GUI_WIDTH = 1050
GUI_HEIGHT = 760

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
)

POSE_MODEL = os.path.join(
    MODEL_DIR,
    "pose_landmarker_full.task",
)

HAND_MODEL = os.path.join(
    MODEL_DIR,
    "hand_landmarker.task",
)

POSE_MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

HAND_MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)


# ============================================================================
# MediaPipe indices
# ============================================================================

R_Wrist = 16
R_Shoulder = 12
R_Hip = 24

H_Wrist = 0
H_Index = 5
H_Pinky = 17
H_Index_Pip = 6
H_Middle = 9
H_Middle_Pip = 10
H_Ring = 13
H_Ring_Pip = 14


# ============================================================================
# H2R transform configuration
# ============================================================================

origin_x = "default"
origin_y = "r_hip"
origin_z = "default"

scaling = [1.4, 1.2, 0.8]


# ============================================================================
# ROBOT GEOMETRY
#
# Coordinate system used by the IK:
#
#     X = forward/back relative to robot
#     Y = left/right
#     Z = up/down
#
# Base is at (0,0,0).
#
# The first three joints solve position:
#
#     base     = yaw around Z
#     shoulder = pitch in radial/Z plane
#     elbow    = pitch in radial/Z plane
#
# Wrist pitch/roll then reproduce the requested hand orientation as far as
# this simplified 5-axis wrist model permits.
# ============================================================================

LINK_BASE_HEIGHT = 0.045   # metres
LINK_UPPER_ARM = 0.105     # metres
LINK_FOREARM = 0.105       # metres
LINK_WRIST = 0.055         # metres


# Servo-space convention requested for this project:
#
#     base        90 = centered
#     shoulder    180 = nominal horizontal/neutral
#     elbow       180 = fully stretched
#     wrist_pitch 90 = neutral
#     wrist_roll  90 = neutral
#     gripper     100=open, 180=closed
#
# These are servo output conventions, NOT geometric joint angles.

SERVO_LIMITS = {
    "base": (0.0, 180.0),
    "shoulder": (0.0, 180.0),
    "elbow": (0.0, 180.0),
    "wrist_pitch": (0.0, 180.0),
    "wrist_roll": (0.0, 180.0),
    "gripper": (0.0, 180.0),
}


# ============================================================================
# Shared state
# ============================================================================

state = {
    "frame": None,

    "hand_world": np.zeros(3),
    "translation": np.zeros(3),
    "h2r_position": np.zeros(3),

    "rotation": np.eye(3),
    "axis_angle": np.array([0.0, 0.0, 1.0, 0.0]),

    "gripper": 100.0,
    "fps": 0.0,

    "hand_detected": False,
    "pose_detected": False,

    "ik": {
        "base": 90.0,
        "shoulder": 180.0,
        "elbow": 180.0,
        "wrist_pitch": 90.0,
        "wrist_roll": 90.0,
        "gripper": 100.0,
        "reachable": False,
        "distance": 0.0,
    },

    # Simulator joint points
    "joint_points": [
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
    ],
}

state_lock = threading.Lock()


# ============================================================================
# Helpers
# ============================================================================

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def normalize(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude < 1e-8:
        return np.zeros_like(vector)
    return vector / magnitude


def download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    print(f"[MODEL] Downloading: {url}")

    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as exc:
        if os.path.exists(destination):
            os.remove(destination)
        raise RuntimeError(
            f"Could not download model:\n{url}\nError: {exc}"
        ) from exc

    print(f"[MODEL] Saved: {destination}")


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(POSE_MODEL):
        download_file(POSE_MODEL_URL, POSE_MODEL)
    else:
        print(f"[MODEL] Pose model found: {POSE_MODEL}")

    if not os.path.exists(HAND_MODEL):
        download_file(HAND_MODEL_URL, HAND_MODEL)
    else:
        print(f"[MODEL] Hand model found: {HAND_MODEL}")


def make_mp_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )


# ============================================================================
# H2R transform
# ============================================================================

def HandWorld2RobotWorld(landmark, translation, scaling):
    R = np.array([
        [1, 0, 0],
        [0, np.cos(-np.pi / 2), -np.sin(-np.pi / 2)],
        [0, np.sin(-np.pi / 2),  np.cos(-np.pi / 2)],
    ])

    landmark = np.asarray(landmark, dtype=float)

    p = R @ landmark
    p = p * np.asarray(scaling, dtype=float)
    p = p + np.asarray(translation, dtype=float)

    return p


# ============================================================================
# Rotation helpers
# ============================================================================

def rotation_matrix_to_euler_xyz(R):
    """
    Returns XYZ intrinsic-style angles in degrees.

    Used only for the wrist simulator/control mapping.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0

    return np.degrees([x, y, z])


# ============================================================================
# 6DOF IK
# ============================================================================

def solve_ik(position, rotation, gripper):
    """
    Geometric IK for the 6DOF arm.

    Position:
        base + shoulder + elbow

    Orientation:
        wrist pitch + wrist roll

    The solver deliberately does NOT silently report an impossible point as
    reachable. If the target is outside the planar arm workspace, the point
    is projected to the closest reachable location and reachable=False.

    Servo conventions:
        base        90 = center
        shoulder    180 = neutral/horizontal
        elbow       180 = fully stretched
        wrist_pitch 90 = neutral
        wrist_roll  90 = neutral

    NOTE:
        The shoulder/elbow servo mapping is the part that must eventually be
        calibrated against the physical linkage.
    """

    p = np.asarray(position, dtype=float)
    R = np.asarray(rotation, dtype=float)

    # ------------------------------------------------------------
    # Base yaw
    # ------------------------------------------------------------

    base_geom = math.degrees(math.atan2(p[1], p[0]))

    # Servo center = 90.
    base_servo = clamp(90.0 + base_geom, 0.0, 180.0)

    # ------------------------------------------------------------
    # Remove base rotation.
    # Work in the radial/Z plane.
    # ------------------------------------------------------------

    radial = math.hypot(p[0], p[1])

    # Shoulder pivot.
    z_target = p[2] - LINK_BASE_HEIGHT

    # Wrist center.
    #
    # We use the target point directly for the geometric position solve.
    # LINK_WRIST is represented visually, while orientation is handled by
    # the wrist joints.
    x = radial
    z = z_target

    distance = math.hypot(x, z)

    L1 = LINK_UPPER_ARM
    L2 = LINK_FOREARM

    reachable = (
        distance <= (L1 + L2)
        and distance >= abs(L1 - L2)
    )

    # ------------------------------------------------------------
    # Project unreachable targets to the workspace boundary.
    # ------------------------------------------------------------

    if distance < 1e-8:
        x = max(0.001, L1 + L2 - 0.001)
        z = 0.0
        distance = math.hypot(x, z)

    if distance > L1 + L2:
        scale = (L1 + L2 - 1e-6) / distance
        x *= scale
        z *= scale
        distance = math.hypot(x, z)

    elif distance < abs(L1 - L2):
        scale = (abs(L1 - L2) + 1e-6) / distance
        x *= scale
        z *= scale
        distance = math.hypot(x, z)

    # ------------------------------------------------------------
    # Law of cosines
    # ------------------------------------------------------------

    cos_elbow = (
        (L1 * L1 + L2 * L2 - distance * distance)
        / (2.0 * L1 * L2)
    )

    cos_elbow = clamp(cos_elbow, -1.0, 1.0)

    elbow_internal = math.acos(cos_elbow)

    # Elbow-down solution:
    # angle between upper arm and +radial direction.
    cos_shoulder_offset = (
        (L1 * L1 + distance * distance - L2 * L2)
        / (2.0 * L1 * distance)
    )

    cos_shoulder_offset = clamp(
        cos_shoulder_offset,
        -1.0,
        1.0,
    )

    shoulder_offset = math.acos(cos_shoulder_offset)

    target_angle = math.atan2(z, x)

    shoulder_geom = target_angle + shoulder_offset

    # ------------------------------------------------------------
    # Convert geometric angles to this project's servo convention.
    #
    # Geometric shoulder:
    #   0 deg = pointing horizontally forward
    #   +90   = pointing straight up
    #
    # Project servo:
    #   180 = horizontal/neutral
    #
    # Therefore:
    #       servo = 180 - geometric angle
    #
    # Elbow:
    #   180 = fully stretched
    #   smaller = more bent
    #
    # elbow_internal is 180 at full extension.
    # ------------------------------------------------------------

    shoulder_servo = clamp(
        180.0 - math.degrees(shoulder_geom),
        0.0,
        180.0,
    )

    elbow_servo = clamp(
        math.degrees(elbow_internal),
        0.0,
        180.0,
    )

    # ------------------------------------------------------------
    # Wrist orientation
    # ------------------------------------------------------------

    # Desired tool orientation relative to robot.
    #
    # For this simple arm, wrist pitch is estimated from the tool's local
    # Z direction projected into the radial/Z plane.
    tool_z = normalize(R[:, 2])

    tool_radial = (
        math.cos(base_geom * math.pi / 180.0) * tool_z[0]
        + math.sin(base_geom * math.pi / 180.0) * tool_z[1]
    )

    tool_vertical = tool_z[2]

    tool_pitch = math.degrees(
        math.atan2(tool_vertical, tool_radial)
    )

    # Compensate for shoulder + elbow orientation.
    arm_pitch = (
        math.degrees(shoulder_geom)
        + math.degrees(elbow_internal - math.pi)
    )

    wrist_pitch_geom = tool_pitch - arm_pitch

    wrist_pitch_servo = clamp(
        90.0 - wrist_pitch_geom,
        0.0,
        180.0,
    )

    # Wrist roll comes from the tool's local X/Y axes around its local Z.
    #
    # We use the robot radial frame as reference.
    radial_axis = np.array([
        math.cos(base_geom),
        math.sin(base_geom),
        0.0,
    ])

    tangent_axis = np.array([
        -math.sin(base_geom),
        math.cos(base_geom),
        0.0,
    ])

    tool_x = normalize(R[:, 0])

    roll_x = np.dot(tool_x, radial_axis)
    roll_y = np.dot(tool_x, tangent_axis)

    wrist_roll_geom = math.degrees(
        math.atan2(roll_y, roll_x)
    )

    wrist_roll_servo = clamp(
        90.0 + wrist_roll_geom,
        0.0,
        180.0,
    )

    # ------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------

    gripper_servo = (
        180.0
        if gripper >= 2
        else 100.0
    )

    # ------------------------------------------------------------
    # Forward-kinematic points for simulator
    # ------------------------------------------------------------

    shoulder_point = np.array([
        0.0,
        0.0,
        LINK_BASE_HEIGHT,
    ])

    elbow_point = shoulder_point + np.array([
        L1 * math.cos(shoulder_geom) * math.cos(base_geom),
        L1 * math.cos(shoulder_geom) * math.sin(base_geom),
        L1 * math.sin(shoulder_geom),
    ])

    wrist_point = elbow_point + np.array([
        L2 * math.cos(shoulder_geom - elbow_internal) * math.cos(base_geom),
        L2 * math.cos(shoulder_geom - elbow_internal) * math.sin(base_geom),
        L2 * math.sin(shoulder_geom - elbow_internal),
    ])

    tool_point = wrist_point.copy()

    return {
        "base": base_servo,
        "shoulder": shoulder_servo,
        "elbow": elbow_servo,
        "wrist_pitch": wrist_pitch_servo,
        "wrist_roll": wrist_roll_servo,
        "gripper": gripper_servo,
        "reachable": reachable,
        "distance": distance,
        "joint_points": [
            np.array([0.0, 0.0, 0.0]),
            shoulder_point,
            elbow_point,
            wrist_point,
            tool_point,
        ],
    }


# ============================================================================
# Drawing
# ============================================================================

def draw_hand(frame, landmarks):
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17),
    ]

    h, w = frame.shape[:2]
    points = []

    for landmark in landmarks:
        pt = (
            int(landmark.x * w),
            int(landmark.y * h),
        )
        points.append(pt)
        cv2.circle(frame, pt, 3, (0,255,0), -1)

    for a, b in connections:
        cv2.line(
            frame,
            points[a],
            points[b],
            (0,255,0),
            2,
        )


def draw_pose(frame, landmarks):
    connections = [
        (11,12),(11,13),(13,15),
        (12,14),(14,16),
        (11,23),(12,24),(23,24),
        (23,25),(25,27),(24,26),(26,28),
    ]

    h, w = frame.shape[:2]
    points = [
        (int(p.x*w), int(p.y*h))
        for p in landmarks
    ]

    for a, b in connections:
        cv2.line(
            frame,
            points[a],
            points[b],
            (255,0,0),
            2,
        )


# ============================================================================
# Console output
# ============================================================================

class ConsoleOutput:

    def __init__(self):
        self.last_transform = 0.0
        self.last_ik = 0.0
        self.last_gripper = None

    def publish(
        self,
        position,
        axis_angle,
        ik,
        gripper,
    ):
        now = time.monotonic()

        if now - self.last_transform >= 0.15:
            self.last_transform = now

            print(
                "[H2R] "
                f"x={position[0]:+.3f} "
                f"y={position[1]:+.3f} "
                f"z={position[2]:+.3f} "
                "| AXIS="
                f"[{axis_angle[0]:+.3f}, "
                f"{axis_angle[1]:+.3f}, "
                f"{axis_angle[2]:+.3f}] "
                f"A={axis_angle[3]:+.1f}°"
            )

        if now - self.last_ik >= 0.15:
            self.last_ik = now

            print(
                "[IK] "
                f"B={ik['base']:6.1f} "
                f"S={ik['shoulder']:6.1f} "
                f"E={ik['elbow']:6.1f} "
                f"WP={ik['wrist_pitch']:6.1f} "
                f"WR={ik['wrist_roll']:6.1f} "
                f"G={ik['gripper']:6.1f} "
                "| "
                f"{'REACHABLE' if ik['reachable'] else 'PROJECTED'}"
            )


# ============================================================================
# Vision
# ============================================================================

def run_vision(output):
    print("=" * 72)
    print("MediaPipe Vision + 6DOF IK")
    print("ROS / ROS2: DISABLED")
    print("=" * 72)

    try:
        ensure_models()
    except Exception as exc:
        print("[ERROR] Model initialization failed:")
        print(exc)
        return

    try:
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=POSE_MODEL,
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=HAND_MODEL,
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        pose_landmarker = PoseLandmarker.create_from_options(
            pose_options
        )

        hand_landmarker = HandLandmarker.create_from_options(
            hand_options
        )

    except Exception as exc:
        print("[ERROR] MediaPipe initialization failed:")
        print(exc)
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        pose_landmarker.close()
        hand_landmarker.close()
        return

    print("[CAMERA] Camera opened.")
    print("[VISION] Processing started.")

    Basis = np.eye(3)
    tool_pos = np.zeros(3)
    hand_world = np.zeros(3)
    behavior_key = 0

    fps_time = time.monotonic()
    fps_frames = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Camera frame failed.")
                break

            fps_frames += 1

            timestamp_ms = int(time.monotonic() * 1000)
            mp_image = make_mp_image(frame)

            # ------------------------------------------------------------
            # Pose
            # ------------------------------------------------------------

            pose_result = pose_landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            pose_detected = (
                len(pose_result.pose_landmarks) > 0
            )

            translation = np.zeros(3)

            if pose_detected:
                pose_landmarks = pose_result.pose_landmarks[0]
                pose_world = pose_result.pose_world_landmarks[0]

                hip_pos = pose_world[R_Hip]

                if origin_x == "r_hip":
                    translation[0] = hip_pos.x
                elif origin_x == "r_shoulder":
                    translation[0] = pose_world[R_Shoulder].x

                if origin_y == "r_hip":
                    translation[1] = hip_pos.y
                elif origin_y == "r_shoulder":
                    translation[1] = pose_world[R_Shoulder].y

                if origin_z == "r_hip":
                    translation[2] = hip_pos.z
                elif origin_z == "r_shoulder":
                    translation[2] = pose_world[R_Shoulder].z

                draw_pose(frame, pose_landmarks)

            # ------------------------------------------------------------
            # Hand
            # ------------------------------------------------------------

            hand_result = hand_landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            hand_detected = (
                len(hand_result.hand_landmarks) > 0
            )

            if hand_detected:
                for hand_index, landmarks in enumerate(
                    hand_result.hand_landmarks
                ):
                    handedness = hand_result.handedness[hand_index]

                    if not handedness:
                        continue

                    label = handedness[0].category_name

                    # Same convention as the original program:
                    # use the LEFT hand.
                    if label != "Left":
                        continue

                    world_landmarks = (
                        hand_result.hand_world_landmarks[
                            hand_index
                        ]
                    )

                    raw_wrist = world_landmarks[H_Wrist]
                    raw_index = world_landmarks[H_Index]
                    raw_pinky = world_landmarks[H_Pinky]

                    hand_world = np.array([
                        raw_wrist.x,
                        raw_wrist.y,
                        raw_wrist.z,
                    ])

                    tool_pos = HandWorld2RobotWorld(
                        hand_world,
                        translation,
                        scaling,
                    )

                    h_wrist = HandWorld2RobotWorld(
                        [raw_wrist.x, raw_wrist.y, raw_wrist.z],
                        translation,
                        [1,1,1],
                    )

                    h_index = HandWorld2RobotWorld(
                        [raw_index.x, raw_index.y, raw_index.z],
                        translation,
                        [1,1,1],
                    )

                    h_pinky = HandWorld2RobotWorld(
                        [raw_pinky.x, raw_pinky.y, raw_pinky.z],
                        translation,
                        [1,1,1],
                    )

                    index_dir = normalize(
                        h_index - h_wrist
                    )

                    pinky_dir = normalize(
                        h_pinky - h_wrist
                    )

                    palm_dir = normalize(
                        np.cross(index_dir, pinky_dir)
                    )

                    thumb_dir = normalize(
                        np.cross(index_dir, palm_dir)
                    )

                    # Basis columns are the same orientation convention
                    # used by the previous version.
                    Basis = np.stack(
                        [
                            thumb_dir,
                            index_dir,
                            palm_dir,
                        ],
                        axis=1,
                    )

                    # ----------------------------------------------------
                    # Finger state
                    # ----------------------------------------------------

                    def world_point(index):
                        p = world_landmarks[index]
                        return HandWorld2RobotWorld(
                            [p.x, p.y, p.z],
                            translation,
                            [1,1,1],
                        )

                    index_tip = world_point(H_Index)
                    index_pip = world_point(H_Index_Pip)

                    middle_tip = world_point(H_Middle)
                    middle_pip = world_point(H_Middle_Pip)

                    ring_tip = world_point(H_Ring)
                    ring_pip = world_point(H_Ring_Pip)

                    index_dir2 = normalize(
                        index_tip - index_pip
                    )
                    middle_dir2 = normalize(
                        middle_tip - middle_pip
                    )
                    ring_dir2 = normalize(
                        ring_tip - ring_pip
                    )

                    wrist_dir = normalize(
                        h_wrist - h_index
                    )

                    votes = (
                        int(np.dot(wrist_dir, index_dir2) <= 0.7)
                        + int(np.dot(wrist_dir, middle_dir2) <= 0.7)
                        + int(np.dot(wrist_dir, ring_dir2) <= 0.7)
                    )

                    behavior_key = 2 if votes >= 2 else 0

                    draw_hand(frame, landmarks)
                    break

            # ------------------------------------------------------------
            # Orientation
            # ------------------------------------------------------------

            # If hand is lost, retain last valid orientation.
            quat = mat2quat(Basis)
            axis, angle = quat2axangle(quat)

            axis_angle = np.append(
                axis,
                np.degrees(angle),
            )

            # ------------------------------------------------------------
            # IK
            # ------------------------------------------------------------

            ik = solve_ik(
                tool_pos,
                Basis,
                behavior_key,
            )

            output.publish(
                tool_pos,
                axis_angle,
                ik,
                behavior_key,
            )

            # ------------------------------------------------------------
            # FPS
            # ------------------------------------------------------------

            now = time.monotonic()
            elapsed = now - fps_time

            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_time = now

                with state_lock:
                    state["fps"] = fps

            # ------------------------------------------------------------
            # Camera overlay
            # ------------------------------------------------------------

            cv2.putText(
                frame,
                "MediaPipe + 6DOF IK",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2,
            )

            cv2.putText(
                frame,
                f"Hand: {'YES' if hand_detected else 'NO'}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2,
            )

            cv2.putText(
                frame,
                f"H2R: "
                f"{tool_pos[0]:+.3f}, "
                f"{tool_pos[1]:+.3f}, "
                f"{tool_pos[2]:+.3f}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0,255,255),
                2,
            )

            cv2.putText(
                frame,
                f"IK: "
                f"B{ik['base']:.0f} "
                f"S{ik['shoulder']:.0f} "
                f"E{ik['elbow']:.0f} "
                f"WP{ik['wrist_pitch']:.0f} "
                f"WR{ik['wrist_roll']:.0f}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255,255,0),
                2,
            )

            # ------------------------------------------------------------
            # Shared state
            # ------------------------------------------------------------

            with state_lock:
                state["frame"] = frame.copy()
                state["hand_world"] = hand_world.copy()
                state["translation"] = translation.copy()
                state["h2r_position"] = tool_pos.copy()
                state["rotation"] = Basis.copy()
                state["axis_angle"] = axis_angle.copy()
                state["gripper"] = ik["gripper"]
                state["hand_detected"] = hand_detected
                state["pose_detected"] = pose_detected
                state["ik"] = {
                    k: v
                    for k, v in ik.items()
                    if k != "joint_points"
                }
                state["joint_points"] = [
                    p.copy()
                    for p in ik["joint_points"]
                ]

    except Exception as exc:
        print("[ERROR] Vision thread stopped:")
        print(repr(exc))
        import traceback
        traceback.print_exc()

    finally:
        cap.release()
        pose_landmarker.close()
        hand_landmarker.close()
        print("[VISION] Stopped.")


# ============================================================================
# Arm simulator widget
# ============================================================================

class ArmSimulator(QWidget):
    """
    Simple side/top hybrid visualization.

    Left/right shows the arm in the radial/Z plane.
    The base angle is displayed separately, because base yaw is perpendicular
    to this view.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 430)
        self.setStyleSheet(
            "background:#101010; border:1px solid #444;"
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        cx = w * 0.50
        ground_y = h * 0.83

        scale = min(
            w * 0.42 / (LINK_UPPER_ARM + LINK_FOREARM + LINK_WRIST),
            h * 0.58 / (LINK_UPPER_ARM + LINK_FOREARM),
        )

        with state_lock:
            points = [
                p.copy()
                for p in state["joint_points"]
            ]
            ik = dict(state["ik"])
            h2r = state["h2r_position"].copy()

        def screen(p):
            # Simulator view:
            # X radial -> right
            # Z -> up
            return QPointF(
                cx + p[0] * scale,
                ground_y - p[2] * scale,
            )

        # Grid
        grid_pen = QPen(Qt.darkGray)
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)

        for i in range(-5, 6):
            x = cx + i * 0.05 * scale
            painter.drawLine(
                QPointF(x, 30),
                QPointF(x, ground_y),
            )

        for i in range(0, 7):
            z = i * 0.05
            y = ground_y - z * scale
            painter.drawLine(
                QPointF(20, y),
                QPointF(w - 20, y),
            )

        # Ground
        painter.setPen(QPen(Qt.gray, 2))
        painter.drawLine(
            QPointF(20, ground_y),
            QPointF(w - 20, ground_y),
        )

        # Arm
        if len(points) >= 5:
            arm_pen = QPen(Qt.cyan, 10)
            arm_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(arm_pen)

            for a, b in zip(points[:-1], points[1:]):
                painter.drawLine(
                    screen(a),
                    screen(b),
                )

            joint_pen = QPen(Qt.white, 3)
            painter.setPen(joint_pen)

            for p in points:
                q = screen(p)
                painter.drawEllipse(q, 8, 8)

            # Target
            target = np.array([
                math.hypot(h2r[0], h2r[1]),
                0,
                h2r[2],
            ])

            q = screen(target)

            target_pen = QPen(Qt.yellow, 2)
            painter.setPen(target_pen)

            painter.drawEllipse(q, 11, 11)
            painter.drawLine(
                QPointF(q.x() - 16, q.y()),
                QPointF(q.x() + 16, q.y()),
            )
            painter.drawLine(
                QPointF(q.x(), q.y() - 16),
                QPointF(q.x(), q.y() + 16),
            )

        painter.setPen(Qt.white)
        painter.drawText(
            15,
            25,
            "6DOF ARM — SIDE / RADIAL VIEW",
        )

        painter.drawText(
            15,
            45,
            "Yellow = H2R target   White/Cyan = IK solution",
        )

        status = (
            "REACHABLE"
            if ik.get("reachable", False)
            else "PROJECTED TO WORKSPACE"
        )

        painter.drawText(
            15,
            h - 42,
            f"{status}   "
            f"B={ik.get('base', 90):.1f}°   "
            f"S={ik.get('shoulder', 180):.1f}°   "
            f"E={ik.get('elbow', 180):.1f}°",
        )

        painter.drawText(
            15,
            h - 20,
            f"H2R radial={math.hypot(h2r[0], h2r[1]):.3f}m   "
            f"Z={h2r[2]:+.3f}m",
        )


# ============================================================================
# PySide GUI
# ============================================================================

class VisionWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "MediaPipe Hand Vision → 6DOF IK Simulator"
        )

        self.setWindowFlag(
            Qt.WindowStaysOnTopHint,
            True,
        )

        self.resize(
            GUI_WIDTH,
            GUI_HEIGHT,
        )

        self.setup_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(50)

    def setup_ui(self):

        main = QHBoxLayout(self)

        # ================================================================
        # Left
        # ================================================================

        left = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setFixedSize(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
        )
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Starting camera...")

        left.addWidget(self.video_label)

        values_box = QGroupBox("Vision / H2R")
        grid = QGridLayout(values_box)

        labels = [
            "Pose:",
            "Hand:",
            "HandWorld:",
            "Translation:",
            "H2R:",
            "Axis/Angle:",
            "Gripper:",
            "FPS:",
        ]

        self.pose_value = QLabel("NO")
        self.hand_value = QLabel("NO")
        self.hand_world_value = QLabel("0, 0, 0")
        self.translation_value = QLabel("0, 0, 0")
        self.h2r_value = QLabel("0, 0, 0")
        self.axis_value = QLabel("[0,0,1] 0°")
        self.gripper_value = QLabel("100 OPEN")
        self.fps_value = QLabel("0")

        values = [
            self.pose_value,
            self.hand_value,
            self.hand_world_value,
            self.translation_value,
            self.h2r_value,
            self.axis_value,
            self.gripper_value,
            self.fps_value,
        ]

        font = QFont()
        font.setBold(True)

        for row, (name, value) in enumerate(
            zip(labels, values)
        ):
            label = QLabel(name)
            label.setFont(font)
            grid.addWidget(label, row, 0)
            grid.addWidget(value, row, 1)

        left.addWidget(values_box)

        # ================================================================
        # Right
        # ================================================================

        right = QVBoxLayout()

        right.addWidget(
            QLabel("6DOF TRANSFORM / IK")
        )

        self.ik_value = QLabel()
        self.ik_value.setFont(font)

        right.addWidget(self.ik_value)

        self.simulator = ArmSimulator()

        right.addWidget(
            self.simulator,
            1,
        )

        note = QLabel(
            "IK geometry is active.\n"
            "Servo zero/direction calibration comes next.\n"
            "No servo commands are sent by this program."
        )

        note.setWordWrap(True)
        right.addWidget(note)

        main.addLayout(left)
        main.addLayout(right, 1)

    def update_gui(self):

        with state_lock:
            frame = (
                state["frame"].copy()
                if state["frame"] is not None
                else None
            )

            hand_world = state["hand_world"].copy()
            translation = state["translation"].copy()
            h2r = state["h2r_position"].copy()
            axis_angle = state["axis_angle"].copy()
            gripper = state["gripper"]
            fps = state["fps"]
            hand_detected = state["hand_detected"]
            pose_detected = state["pose_detected"]
            ik = dict(state["ik"])

        # Video
        if frame is not None:
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            height, width, channels = rgb.shape

            image = QImage(
                rgb.data,
                width,
                height,
                channels * width,
                QImage.Format_RGB888,
            )

            pixmap = QPixmap.fromImage(image)

            self.video_label.setPixmap(
                pixmap.scaled(
                    CAMERA_WIDTH,
                    CAMERA_HEIGHT,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        self.pose_value.setText(
            "YES" if pose_detected else "NO"
        )

        self.hand_value.setText(
            "YES" if hand_detected else "NO"
        )

        self.hand_world_value.setText(
            f"{hand_world[0]:+.4f}, "
            f"{hand_world[1]:+.4f}, "
            f"{hand_world[2]:+.4f}"
        )

        self.translation_value.setText(
            f"{translation[0]:+.4f}, "
            f"{translation[1]:+.4f}, "
            f"{translation[2]:+.4f}"
        )

        self.h2r_value.setText(
            f"{h2r[0]:+.4f}, "
            f"{h2r[1]:+.4f}, "
            f"{h2r[2]:+.4f}"
        )

        self.axis_value.setText(
            f"["
            f"{axis_angle[0]:+.3f}, "
            f"{axis_angle[1]:+.3f}, "
            f"{axis_angle[2]:+.3f}"
            f"] "
            f"{axis_angle[3]:+.1f}°"
        )

        self.gripper_value.setText(
            "180 CLOSED"
            if gripper >= 180
            else "100 OPEN"
        )

        self.fps_value.setText(
            f"{fps:.1f}"
        )

        self.ik_value.setText(
            f"B  {ik['base']:6.1f}°\n"
            f"S  {ik['shoulder']:6.1f}°\n"
            f"E  {ik['elbow']:6.1f}°\n"
            f"WP {ik['wrist_pitch']:6.1f}°\n"
            f"WR {ik['wrist_roll']:6.1f}°\n"
            f"G  {ik['gripper']:6.1f}°\n\n"
            f"{'REACHABLE' if ik['reachable'] else 'PROJECTED'}"
        )

        self.simulator.update()


# ============================================================================
# Main
# ============================================================================

def main():

    print()
    print("Starting teleop.py")
    print(f"Python: {sys.version.split()[0]}")
    print(
        f"MediaPipe: "
        f"{getattr(mp, '__version__', 'unknown')}"
    )

    output = ConsoleOutput()

    vision_thread = threading.Thread(
        target=run_vision,
        args=(output,),
        daemon=True,
        name="VisionThread",
    )

    vision_thread.start()

    app = QApplication(sys.argv)

    window = VisionWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
