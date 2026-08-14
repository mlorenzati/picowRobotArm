"""
teleop.py

MediaPipe + PySide6 hand/pose tracking + IK simulator.

NO ROS / ROS2.

Architecture:

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
    H2R Position + Orientation
       |
       v
    IK Simulator
       |
       +---- Base
       +---- Shoulder
       +---- Elbow
       +---- Wrist Pitch
       +---- Wrist Roll
       +---- Gripper
       |
       +---- Console
       +---- PySide6 GUI

Everything is contained in this single file.

Requirements:

    pip install mediapipe opencv-python PySide6 numpy transforms3d

The first execution downloads:

    models/pose_landmarker_full.task
    models/hand_landmarker.task
"""

import sys
import os
import time
import threading
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

from transforms3d.quaternions import mat2quat, quat2axangle

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QGridLayout,
)


# ============================================================================
# Configuration
# ============================================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

GUI_WIDTH = 680
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
# Original landmark indices
# ============================================================================

# Pose
R_Index = 20
R_Pinky = 18
R_Wrist = 16
R_Shoulder = 12
R_Hip = 24

# Hand
H_Wrist = 0
H_Index = 5
H_Pinky = 17
H_Index_Pip = 6
H_Middle = 9
H_Middle_Pip = 10
H_Ring = 13
H_Ring_Pip = 14


# ============================================================================
# Transformation configuration
# ============================================================================

origin_options = [
    "default",
    "r_hip",
    "r_shoulder",
]

origin_x = "default"
origin_y = "r_hip"
origin_z = "default"

scaling = [
    1.4,
    1.2,
    0.8,
]


# ============================================================================
# IK SIMULATOR CONFIGURATION
# ============================================================================

#
# IMPORTANT:
#
# These are approximate values.
#
# Replace these with your real arm measurements.
#
# L1 = base -> shoulder
# L2 = upper arm
# L3 = forearm
#
# All dimensions are meters.
#

IK_L1 = 0.080
IK_L2 = 0.120
IK_L3 = 0.120


# ---------------------------------------------------------------------------
# Servo conventions
# ---------------------------------------------------------------------------
#
# Based on the conventions you established:
#
# Base:
#     90 = center
#
# Shoulder:
#     180 = reference/down
#     lower values = elbow goes upward
#
# Elbow:
#     180 = completely stretched
#
# Wrist pitch:
#     90 = neutral
#
# Wrist roll:
#     90 = neutral
#
# Gripper:
#     100 = open
#     180 = closed
#

IK_BASE_CENTER = 90.0

IK_SHOULDER_CENTER = 180.0

IK_ELBOW_STRETCHED = 180.0

IK_WRIST_PITCH_CENTER = 90.0

IK_WRIST_ROLL_CENTER = 90.0

IK_GRIPPER_OPEN = 100.0

IK_GRIPPER_CLOSED = 180.0


# Servo limits

IK_BASE_MIN = 0.0
IK_BASE_MAX = 180.0

IK_SHOULDER_MIN = 0.0
IK_SHOULDER_MAX = 180.0

IK_ELBOW_MIN = 0.0
IK_ELBOW_MAX = 180.0

IK_WRIST_PITCH_MIN = 0.0
IK_WRIST_PITCH_MAX = 180.0

IK_WRIST_ROLL_MIN = 0.0
IK_WRIST_ROLL_MAX = 180.0


# ============================================================================
# Shared state
# ============================================================================

state = {
    "frame": None,

    "hand_world": np.zeros(3),

    "translation": np.zeros(3),

    "h2r_position": np.zeros(3),

    "rotation": np.eye(3),

    "axis_angle": np.array([
        0.0,
        0.0,
        1.0,
        0.0,
    ]),

    "gripper": 0,

    "ik": {
        "base": 90.0,
        "shoulder": 180.0,
        "elbow": 180.0,
        "wrist_pitch": 90.0,
        "wrist_roll": 90.0,
        "gripper": 100.0,
        "reachable": False,
    },

    "fps": 0.0,

    "hand_detected": False,
    "pose_detected": False,
}

state_lock = threading.Lock()


# ============================================================================
# Utility
# ============================================================================

def clamp(
    value,
    minimum,
    maximum,
):
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def normalize(vector):

    magnitude = np.linalg.norm(
        vector
    )

    if magnitude < 1e-8:

        return np.zeros_like(
            vector
        )

    return vector / magnitude


# ============================================================================
# Model download
# ============================================================================

def download_file(
    url,
    destination,
):

    os.makedirs(
        os.path.dirname(destination),
        exist_ok=True,
    )

    print()
    print(
        f"[MODEL] Downloading:\n"
        f"        {url}"
    )

    try:

        urllib.request.urlretrieve(
            url,
            destination,
        )

    except Exception as exc:

        if os.path.exists(destination):
            os.remove(destination)

        raise RuntimeError(
            f"Could not download model:\n"
            f"{url}\n\n"
            f"Error: {exc}"
        ) from exc

    print(
        f"[MODEL] Saved:\n"
        f"        {destination}"
    )


def ensure_models():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    if not os.path.exists(POSE_MODEL):

        download_file(
            POSE_MODEL_URL,
            POSE_MODEL,
        )

    else:

        print(
            f"[MODEL] Pose model found:\n"
            f"        {POSE_MODEL}"
        )

    if not os.path.exists(HAND_MODEL):

        download_file(
            HAND_MODEL_URL,
            HAND_MODEL,
        )

    else:

        print(
            f"[MODEL] Hand model found:\n"
            f"        {HAND_MODEL}"
        )


# ============================================================================
# Output interface
# ============================================================================

class VisionOutput:

    def publish_transform(
        self,
        position,
        rotation_matrix,
        axis_angle,
    ):
        raise NotImplementedError

    def publish_gripper(
        self,
        value,
    ):
        raise NotImplementedError


# ============================================================================
# Console output
# ============================================================================

class ConsoleOutput(VisionOutput):

    def __init__(self):

        self.last_gripper = None

        self.last_print = 0.0

    def publish_transform(
        self,
        position,
        rotation_matrix,
        axis_angle,
    ):

        now = time.monotonic()

        if now - self.last_print < 0.1:
            return

        self.last_print = now

        print(
            "[H2R] "
            f"x={position[0]:+.3f} "
            f"y={position[1]:+.3f} "
            f"z={position[2]:+.3f} "
            "| "
            f"AXIS="
            f"[{axis_angle[0]:+.3f}, "
            f"{axis_angle[1]:+.3f}, "
            f"{axis_angle[2]:+.3f}] "
            f"A={axis_angle[3]:+.1f}°"
        )

    def publish_gripper(
        self,
        value,
    ):

        if value == self.last_gripper:
            return

        self.last_gripper = value

        state_name = {
            0: "OPEN",
            2: "CLOSED",
        }.get(
            value,
            "UNKNOWN",
        )

        print(
            f"[GRIPPER] "
            f"{value} "
            f"({state_name})"
        )


# ============================================================================
# IK: Axis angle -> rotation matrix
# ============================================================================

def axis_angle_to_matrix(
    axis,
    angle_degrees,
):
    """
    Convert axis-angle to a 3x3 rotation matrix.
    """

    axis = normalize(
        np.asarray(
            axis,
            dtype=float,
        )
    )

    if np.linalg.norm(axis) < 1e-8:

        return np.eye(3)

    theta = np.radians(
        angle_degrees
    )

    x, y, z = axis

    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])

    I = np.eye(3)

    R = (
        I * np.cos(theta)
        +
        (1.0 - np.cos(theta))
        * np.outer(axis, axis)
        +
        np.sin(theta) * K
    )

    return R


# ============================================================================
# IK: Position solver
# ============================================================================

def solve_position_ik(
    position,
    L1,
    L2,
    L3,
):
    """
    Solve base, shoulder and elbow.

    Coordinate system used here:

        X = forward
        Y = sideways
        Z = vertical

    Returns:

        base
        shoulder
        elbow
        reachable
    """

    x, y, z = position

    # ------------------------------------------------------------------------
    # Base
    # ------------------------------------------------------------------------

    base = np.arctan2(
        y,
        x,
    )

    # ------------------------------------------------------------------------
    # Distance from shoulder
    # ------------------------------------------------------------------------

    radius = np.sqrt(
        x * x
        +
        y * y
    )

    z_relative = (
        z - L1
    )

    distance = np.sqrt(
        radius * radius
        +
        z_relative * z_relative
    )

    # ------------------------------------------------------------------------
    # Reachability
    # ------------------------------------------------------------------------

    minimum_reach = abs(
        L2 - L3
    )

    maximum_reach = (
        L2 + L3
    )

    reachable = (
        distance >= minimum_reach
        and
        distance <= maximum_reach
    )

    # ------------------------------------------------------------------------
    # Law of cosines
    # ------------------------------------------------------------------------

    denominator = (
        2.0
        * L2
        * L3
    )

    if abs(denominator) < 1e-8:

        return (
            base,
            0.0,
            0.0,
            False,
        )

    D = (
        radius * radius
        +
        z_relative * z_relative
        -
        L2 * L2
        -
        L3 * L3
    ) / denominator

    D = clamp(
        D,
        -1.0,
        1.0,
    )

    # ------------------------------------------------------------------------
    # Elbow
    #
    # This solution corresponds to one of the two possible arm
    # configurations.
    # ------------------------------------------------------------------------

    elbow = np.arccos(
        D
    )

    # ------------------------------------------------------------------------
    # Shoulder
    # ------------------------------------------------------------------------

    shoulder = (
        np.arctan2(
            z_relative,
            radius,
        )
        -
        np.arctan2(
            L3 * np.sin(elbow),
            L2 + L3 * np.cos(elbow),
        )
    )

    return (
        base,
        shoulder,
        elbow,
        reachable,
    )


# ============================================================================
# IK: Mathematical angles -> servo angles
# ============================================================================

def robot_angles_from_geometry(
    base,
    shoulder,
    elbow,
):
    """
    Convert mathematical joint angles into the servo conventions
    used by the physical arm.
    """

    # ------------------------------------------------------------------------
    # Base
    # ------------------------------------------------------------------------

    base_deg = (
        IK_BASE_CENTER
        +
        np.degrees(base)
    )

    # ------------------------------------------------------------------------
    # Shoulder
    #
    # Servo direction is inverted.
    # ------------------------------------------------------------------------

    shoulder_deg = (
        IK_SHOULDER_CENTER
        -
        np.degrees(shoulder)
    )

    # ------------------------------------------------------------------------
    # Elbow
    #
    # Geometry:
    #
    #     0   = folded
    #     180 = stretched
    #
    # Servo convention is the same.
    # ------------------------------------------------------------------------

    elbow_deg = (
        np.degrees(elbow)
    )

    return (
        clamp(
            base_deg,
            IK_BASE_MIN,
            IK_BASE_MAX,
        ),

        clamp(
            shoulder_deg,
            IK_SHOULDER_MIN,
            IK_SHOULDER_MAX,
        ),

        clamp(
            elbow_deg,
            IK_ELBOW_MIN,
            IK_ELBOW_MAX,
        ),
    )


# ============================================================================
# IK: Wrist
# ============================================================================

def solve_wrist_ik(
    target_rotation,
    base,
    shoulder,
    elbow,
):
    """
    Calculate the remaining wrist orientation.

    This is intentionally isolated because the exact signs depend
    on the physical orientation of your wrist servos.
    """

    # ------------------------------------------------------------------------
    # Base rotation
    # ------------------------------------------------------------------------

    cb = np.cos(base)
    sb = np.sin(base)

    R_base = np.array([
        [cb, -sb, 0.0],
        [sb,  cb, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # ------------------------------------------------------------------------
    # Shoulder rotation
    # ------------------------------------------------------------------------

    cs = np.cos(shoulder)
    ss = np.sin(shoulder)

    R_shoulder = np.array([
        [ cs, 0.0, ss],
        [0.0, 1.0, 0.0],
        [-ss, 0.0, cs],
    ])

    # ------------------------------------------------------------------------
    # Elbow rotation
    # ------------------------------------------------------------------------

    ce = np.cos(elbow)
    se = np.sin(elbow)

    R_elbow = np.array([
        [ ce, 0.0, se],
        [0.0, 1.0, 0.0],
        [-se, 0.0, ce],
    ])

    # ------------------------------------------------------------------------
    # Rotation up to wrist
    # ------------------------------------------------------------------------

    R03 = (
        R_base
        @
        R_shoulder
        @
        R_elbow
    )

    # ------------------------------------------------------------------------
    # Remaining wrist rotation
    # ------------------------------------------------------------------------

    R36 = (
        R03.T
        @
        target_rotation
    )

    # ------------------------------------------------------------------------
    # Wrist pitch
    # ------------------------------------------------------------------------

    wrist_pitch = np.arctan2(
        -R36[2, 0],
        np.sqrt(
            R36[0, 0] ** 2
            +
            R36[1, 0] ** 2
        ),
    )

    # ------------------------------------------------------------------------
    # Wrist roll
    # ------------------------------------------------------------------------

    wrist_roll = np.arctan2(
        R36[2, 1],
        R36[2, 2],
    )

    return (
        np.degrees(
            wrist_pitch
        ),

        np.degrees(
            wrist_roll
        ),
    )


# ============================================================================
# IK: Complete solver
# ============================================================================

def solve_ik(
    position,
    rotation,
):
    """
    Complete transform:

        H2R XYZ
             +
        H2R rotation
             |
             v
            IK
             |
             v
        Base
        Shoulder
        Elbow
        Wrist Pitch
        Wrist Roll
    """

    # ------------------------------------------------------------------------
    # Position IK
    # ------------------------------------------------------------------------

    (
        base,
        shoulder,
        elbow,
        reachable,
    ) = solve_position_ik(
        position,
        IK_L1,
        IK_L2,
        IK_L3,
    )

    # ------------------------------------------------------------------------
    # First three servo angles
    # ------------------------------------------------------------------------

    (
        base_servo,
        shoulder_servo,
        elbow_servo,
    ) = robot_angles_from_geometry(
        base,
        shoulder,
        elbow,
    )

    # ------------------------------------------------------------------------
    # Wrist IK
    # ------------------------------------------------------------------------

    (
        wrist_pitch_math,
        wrist_roll_math,
    ) = solve_wrist_ik(
        rotation,
        base,
        shoulder,
        elbow,
    )

    # ------------------------------------------------------------------------
    # Convert wrist to servo coordinates
    # ------------------------------------------------------------------------

    wrist_pitch_servo = (
        IK_WRIST_PITCH_CENTER
        +
        wrist_pitch_math
    )

    wrist_roll_servo = (
        IK_WRIST_ROLL_CENTER
        +
        wrist_roll_math
    )

    wrist_pitch_servo = clamp(
        wrist_pitch_servo,
        IK_WRIST_PITCH_MIN,
        IK_WRIST_PITCH_MAX,
    )

    wrist_roll_servo = clamp(
        wrist_roll_servo,
        IK_WRIST_ROLL_MIN,
        IK_WRIST_ROLL_MAX,
    )

    return {
        "base": base_servo,

        "shoulder": shoulder_servo,

        "elbow": elbow_servo,

        "wrist_pitch": wrist_pitch_servo,

        "wrist_roll": wrist_roll_servo,

        "gripper": IK_GRIPPER_OPEN,

        "reachable": reachable,

        "math_base": np.degrees(
            base
        ),

        "math_shoulder": np.degrees(
            shoulder
        ),

        "math_elbow": np.degrees(
            elbow
        ),

        "math_wrist_pitch": (
            wrist_pitch_math
        ),

        "math_wrist_roll": (
            wrist_roll_math
        ),
    }


# ============================================================================
# Hand World -> Robot World
# ============================================================================

def HandWorld2RobotWorld(
    landmark,
    translation,
    scaling,
):
    """
    Original transformation from the repository.
    """

    R = np.array([
        [1, 0, 0],

        [
            0,
            np.cos(-np.pi / 2),
            -np.sin(-np.pi / 2),
        ],

        [
            0,
            np.sin(-np.pi / 2),
            np.cos(-np.pi / 2),
        ],
    ])

    E = np.append(
        np.multiply(
            np.append(
                R,
                np.transpose([
                    translation
                ]),
                axis=1,
            ),
            np.transpose([
                scaling
            ]),
        ),
        np.array([
            [0, 0, 0, 1]
        ]),
        axis=0,
    )

    if len(landmark) == 3:

        landmark = np.append(
            landmark,
            1,
        )

    return np.dot(
        E,
        landmark,
    )[0:3]


# ============================================================================
# MediaPipe image
# ============================================================================

def make_mp_image(
    frame,
):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )


# ============================================================================
# Draw hand
# ============================================================================

def draw_hand(
    frame,
    landmarks,
):

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (0, 17),
    ]

    h, w = frame.shape[:2]

    points = []

    for landmark in landmarks:

        x = int(
            landmark.x * w
        )

        y = int(
            landmark.y * h
        )

        points.append(
            (x, y)
        )

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 255, 0),
            -1,
        )

    for a, b in connections:

        cv2.line(
            frame,
            points[a],
            points[b],
            (0, 255, 0),
            2,
        )


# ============================================================================
# Draw pose
# ============================================================================

def draw_pose(
    frame,
    landmarks,
):

    connections = [
        (11, 12),

        (11, 13),
        (13, 15),

        (12, 14),
        (14, 16),

        (11, 23),
        (12, 24),

        (23, 24),

        (23, 25),
        (25, 27),

        (24, 26),
        (26, 28),
    ]

    h, w = frame.shape[:2]

    points = []

    for landmark in landmarks:

        x = int(
            landmark.x * w
        )

        y = int(
            landmark.y * h
        )

        points.append(
            (x, y)
        )

    for a, b in connections:

        cv2.line(
            frame,
            points[a],
            points[b],
            (255, 0, 0),
            2,
        )


# ============================================================================
# Vision
# ============================================================================

def run_vision(
    output: VisionOutput,
):

    print()
    print("=" * 72)
    print("MediaPipe Vision + IK Simulator")
    print("MediaPipe Tasks API")
    print("ROS / ROS2: DISABLED")
    print("=" * 72)

    # ------------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------------

    try:

        ensure_models()

    except Exception as exc:

        print()
        print(
            "[ERROR] Model initialization failed:"
        )

        print(exc)

        return

    # ------------------------------------------------------------------------
    # MediaPipe
    # ------------------------------------------------------------------------

    try:

        BaseOptions = (
            mp.tasks.BaseOptions
        )

        VisionRunningMode = (
            mp.tasks.vision.RunningMode
        )

        PoseLandmarker = (
            mp.tasks.vision.PoseLandmarker
        )

        PoseLandmarkerOptions = (
            mp.tasks.vision.PoseLandmarkerOptions
        )

        HandLandmarker = (
            mp.tasks.vision.HandLandmarker
        )

        HandLandmarkerOptions = (
            mp.tasks.vision.HandLandmarkerOptions
        )

        pose_options = (
            PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=POSE_MODEL,
                ),

                running_mode=(
                    VisionRunningMode.VIDEO
                ),

                num_poses=1,

                min_pose_detection_confidence=0.5,

                min_pose_presence_confidence=0.5,

                min_tracking_confidence=0.5,
            )
        )

        hand_options = (
            HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=HAND_MODEL,
                ),

                running_mode=(
                    VisionRunningMode.VIDEO
                ),

                num_hands=2,

                min_hand_detection_confidence=0.5,

                min_hand_presence_confidence=0.5,

                min_tracking_confidence=0.5,
            )
        )

        pose_landmarker = (
            PoseLandmarker.create_from_options(
                pose_options
            )
        )

        hand_landmarker = (
            HandLandmarker.create_from_options(
                hand_options
            )
        )

    except Exception as exc:

        print()
        print(
            "[ERROR] MediaPipe initialization failed:"
        )

        print(exc)

        return

    # ------------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    if not cap.isOpened():

        print(
            "[ERROR] Cannot open camera."
        )

        pose_landmarker.close()
        hand_landmarker.close()

        return

    print()
    print(
        "[CAMERA] Camera opened."
    )

    print(
        "[VISION] Processing started."
    )

    print()

    # ------------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------------

    Basis = np.eye(3)

    tool_pos = np.zeros(3)

    hand_world = np.zeros(3)

    behavior_key = 0

    frame_index = 0

    fps_time = time.monotonic()

    fps_frames = 0

    last_ik_print = 0.0

    # ------------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------------

    try:

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:

                print(
                    "[ERROR] Camera frame failed."
                )

                break

            frame_index += 1

            fps_frames += 1

            timestamp_ms = int(
                time.monotonic() * 1000
            )

            mp_image = make_mp_image(
                frame
            )

            # ----------------------------------------------------------------
            # Pose
            # ----------------------------------------------------------------

            pose_result = (
                pose_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )

            pose_detected = (
                len(
                    pose_result.pose_landmarks
                ) > 0
            )

            translation = np.array([
                0.0,
                0.0,
                0.0,
            ])

            if pose_detected:

                pose_landmarks = (
                    pose_result
                    .pose_landmarks[0]
                )

                pose_world = (
                    pose_result
                    .pose_world_landmarks[0]
                )

                hip_pos = (
                    pose_world[R_Hip]
                )

                if origin_x == "r_hip":

                    translation[0] = (
                        hip_pos.x
                    )

                elif origin_x == "r_shoulder":

                    translation[0] = (
                        pose_world[
                            R_Shoulder
                        ].x
                    )

                if origin_y == "r_hip":

                    translation[1] = (
                        hip_pos.y
                    )

                elif origin_y == "r_shoulder":

                    translation[1] = (
                        pose_world[
                            R_Shoulder
                        ].y
                    )

                if origin_z == "r_hip":

                    translation[2] = (
                        hip_pos.z
                    )

                elif origin_z == "r_shoulder":

                    translation[2] = (
                        pose_world[
                            R_Shoulder
                        ].z
                    )

                draw_pose(
                    frame,
                    pose_landmarks,
                )

            # ----------------------------------------------------------------
            # Hand
            # ----------------------------------------------------------------

            hand_result = (
                hand_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )

            hand_detected = (
                len(
                    hand_result.hand_landmarks
                ) > 0
            )

            if hand_detected:

                for hand_index, landmarks in enumerate(
                    hand_result.hand_landmarks
                ):

                    handedness = (
                        hand_result
                        .handedness[
                            hand_index
                        ]
                    )

                    if not handedness:

                        continue

                    label = (
                        handedness[0]
                        .category_name
                    )

                    # Original code uses LEFT hand.

                    if label != "Left":

                        continue

                    world_landmarks = (
                        hand_result
                        .hand_world_landmarks[
                            hand_index
                        ]
                    )

                    raw_wrist = (
                        world_landmarks[
                            H_Wrist
                        ]
                    )

                    raw_index = (
                        world_landmarks[
                            H_Index
                        ]
                    )

                    raw_pinky = (
                        world_landmarks[
                            H_Pinky
                        ]
                    )

                    # --------------------------------------------------------
                    # Raw wrist
                    # --------------------------------------------------------

                    hand_world = np.array([
                        raw_wrist.x,
                        raw_wrist.y,
                        raw_wrist.z,
                    ])

                    # --------------------------------------------------------
                    # H2R position
                    # --------------------------------------------------------

                    tool_pos = (
                        HandWorld2RobotWorld(
                            hand_world,
                            translation,
                            scaling,
                        )
                    )

                    # --------------------------------------------------------
                    # Hand orientation points
                    # --------------------------------------------------------

                    h_wrist = (
                        HandWorld2RobotWorld(
                            [
                                raw_wrist.x,
                                raw_wrist.y,
                                raw_wrist.z,
                            ],
                            translation,
                            [1, 1, 1],
                        )
                    )

                    h_index = (
                        HandWorld2RobotWorld(
                            [
                                raw_index.x,
                                raw_index.y,
                                raw_index.z,
                            ],
                            translation,
                            [1, 1, 1],
                        )
                    )

                    h_pinky = (
                        HandWorld2RobotWorld(
                            [
                                raw_pinky.x,
                                raw_pinky.y,
                                raw_pinky.z,
                            ],
                            translation,
                            [1, 1, 1],
                        )
                    )

                    Index_Dir = normalize(
                        h_index - h_wrist
                    )

                    Pinky_Dir = normalize(
                        h_pinky - h_wrist
                    )

                    Palm_Dir = normalize(
                        np.cross(
                            Index_Dir,
                            Pinky_Dir,
                        )
                    )

                    Thumb_Dir = normalize(
                        np.cross(
                            Index_Dir,
                            Palm_Dir,
                        )
                    )

                    # --------------------------------------------------------
                    # Rotation basis
                    # --------------------------------------------------------

                    Basis = np.stack(
                        [
                            Thumb_Dir,
                            Index_Dir,
                            Palm_Dir,
                        ],
                        axis=1,
                    )

                    # --------------------------------------------------------
                    # Finger bending
                    # --------------------------------------------------------

                    def world_point(index):

                        p = (
                            world_landmarks[
                                index
                            ]
                        )

                        return HandWorld2RobotWorld(
                            [
                                p.x,
                                p.y,
                                p.z,
                            ],
                            translation,
                            [1, 1, 1],
                        )

                    index_tip = (
                        world_point(
                            H_Index
                        )
                    )

                    index_pip = (
                        world_point(
                            H_Index_Pip
                        )
                    )

                    middle_tip = (
                        world_point(
                            H_Middle
                        )
                    )

                    middle_pip = (
                        world_point(
                            H_Middle_Pip
                        )
                    )

                    ring_tip = (
                        world_point(
                            H_Ring
                        )
                    )

                    ring_pip = (
                        world_point(
                            H_Ring_Pip
                        )
                    )

                    index_dir = normalize(
                        index_tip
                        - index_pip
                    )

                    middle_dir = normalize(
                        middle_tip
                        - middle_pip
                    )

                    ring_dir = normalize(
                        ring_tip
                        - ring_pip
                    )

                    wrist_dir = normalize(
                        h_wrist
                        - h_index
                    )

                    Index_Metric = np.dot(
                        wrist_dir,
                        index_dir,
                    )

                    Middle_Metric = np.dot(
                        wrist_dir,
                        middle_dir,
                    )

                    Ring_Metric = np.dot(
                        wrist_dir,
                        ring_dir,
                    )

                    Index_Vote = int(
                        Index_Metric <= 0.7
                    )

                    Middle_Vote = int(
                        Middle_Metric <= 0.7
                    )

                    Ring_Vote = int(
                        Ring_Metric <= 0.7
                    )

                    behavior_key = (
                        0
                        if (
                            Index_Vote
                            + Middle_Vote
                            + Ring_Vote
                        ) < 2
                        else 2
                    )

                    draw_hand(
                        frame,
                        landmarks,
                    )

                    break

            # ----------------------------------------------------------------
            # Rotation -> axis angle
            # ----------------------------------------------------------------

            quat = mat2quat(
                Basis
            )

            axis, angle = (
                quat2axangle(
                    quat
                )
            )

            axis_angle = np.append(
                axis,
                np.degrees(angle),
            )

            # ----------------------------------------------------------------
            # IK
            # ----------------------------------------------------------------

            ik_result = solve_ik(
                tool_pos,
                Basis,
            )

            # Gripper

            if behavior_key == 2:

                ik_result["gripper"] = (
                    IK_GRIPPER_CLOSED
                )

            else:

                ik_result["gripper"] = (
                    IK_GRIPPER_OPEN
                )

            # ----------------------------------------------------------------
            # Output
            # ----------------------------------------------------------------

            output.publish_transform(
                tool_pos,
                Basis,
                axis_angle,
            )

            output.publish_gripper(
                behavior_key
            )

            # ----------------------------------------------------------------
            # IK console
            # ----------------------------------------------------------------

            now = time.monotonic()

            if (
                now - last_ik_print
                >= 0.1
            ):

                last_ik_print = now

                print(
                    "[IK] "
                    f"B={ik_result['base']:6.1f} "
                    f"S={ik_result['shoulder']:6.1f} "
                    f"E={ik_result['elbow']:6.1f} "
                    f"WP={ik_result['wrist_pitch']:6.1f} "
                    f"WR={ik_result['wrist_roll']:6.1f} "
                    f"G={ik_result['gripper']:6.1f} "
                    f"| "
                    f"{'REACHABLE' if ik_result['reachable'] else 'UNREACHABLE'}"
                )

            # ----------------------------------------------------------------
            # FPS
            # ----------------------------------------------------------------

            now = time.monotonic()

            elapsed = (
                now - fps_time
            )

            if elapsed >= 1.0:

                fps = (
                    fps_frames
                    / elapsed
                )

                fps_frames = 0

                fps_time = now

                with state_lock:

                    state["fps"] = fps

            # ----------------------------------------------------------------
            # Video overlay
            # ----------------------------------------------------------------

            cv2.putText(
                frame,
                "MediaPipe + IK",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Hand: "
                f"{'YES' if hand_detected else 'NO'}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"H2R: "
                f"{tool_pos[0]:+.2f}, "
                f"{tool_pos[1]:+.2f}, "
                f"{tool_pos[2]:+.2f}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"IK: "
                f"{ik_result['base']:.0f} "
                f"{ik_result['shoulder']:.0f} "
                f"{ik_result['elbow']:.0f}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

            # ----------------------------------------------------------------
            # State
            # ----------------------------------------------------------------

            with state_lock:

                state["frame"] = (
                    frame.copy()
                )

                if hand_detected:

                    state["hand_world"] = (
                        hand_world.copy()
                    )

                state["translation"] = (
                    translation.copy()
                )

                state["h2r_position"] = (
                    tool_pos.copy()
                )

                state["rotation"] = (
                    Basis.copy()
                )

                state["axis_angle"] = (
                    axis_angle.copy()
                )

                state["gripper"] = (
                    behavior_key
                )

                state["ik"] = (
                    ik_result.copy()
                )

                state["hand_detected"] = (
                    hand_detected
                )

                state["pose_detected"] = (
                    pose_detected
                )

    except Exception as exc:

        print()
        print(
            "[ERROR] Vision thread stopped:"
        )

        print(
            repr(exc)
        )

        import traceback

        traceback.print_exc()

    finally:

        cap.release()

        pose_landmarker.close()

        hand_landmarker.close()

        print()
        print(
            "[VISION] Stopped."
        )


# ============================================================================
# PySide6 GUI
# ============================================================================

class VisionWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "MediaPipe Hand Vision + IK"
        )

        self.setWindowFlag(
            Qt.WindowStaysOnTopHint,
            True,
        )

        self.setFixedSize(
            GUI_WIDTH,
            GUI_HEIGHT,
        )

        self.setup_ui()

        self.timer = QTimer(
            self
        )

        self.timer.timeout.connect(
            self.update_gui
        )

        self.timer.start(
            50
        )

    # ------------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------------

    def setup_ui(self):

        layout = QVBoxLayout(
            self
        )

        # --------------------------------------------------------------------
        # Video
        # --------------------------------------------------------------------

        self.video_label = QLabel()

        self.video_label.setFixedSize(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
        )

        self.video_label.setAlignment(
            Qt.AlignCenter
        )

        self.video_label.setText(
            "Starting camera..."
        )

        layout.addWidget(
            self.video_label
        )

        # --------------------------------------------------------------------
        # Values
        # --------------------------------------------------------------------

        grid = QGridLayout()

        font = QFont()

        font.setBold(
            True
        )

        labels = [
            "Pose detected:",
            "Hand detected:",
            "Raw HandWorld:",
            "Translation:",
            "H2R Position:",
            "Axis/Angle:",
            "IK:",
            "Gripper:",
            "FPS:",
        ]

        for row, text in enumerate(
            labels
        ):

            label = QLabel(
                text
            )

            label.setFont(
                font
            )

            grid.addWidget(
                label,
                row,
                0,
            )

        self.pose_value = QLabel(
            "NO"
        )

        self.hand_value = QLabel(
            "NO"
        )

        self.hand_world_value = QLabel(
            "0.000, 0.000, 0.000"
        )

        self.translation_value = QLabel(
            "0.000, 0.000, 0.000"
        )

        self.h2r_value = QLabel(
            "0.000, 0.000, 0.000"
        )

        self.axis_value = QLabel(
            "[0.000, 0.000, 1.000], 0.0°"
        )

        self.ik_value = QLabel(
            "B=90 S=180 E=180 WP=90 WR=90"
        )

        self.gripper_value = QLabel(
            "100 (OPEN)"
        )

        self.fps_value = QLabel(
            "0.0"
        )

        values = [
            self.pose_value,
            self.hand_value,
            self.hand_world_value,
            self.translation_value,
            self.h2r_value,
            self.axis_value,
            self.ik_value,
            self.gripper_value,
            self.fps_value,
        ]

        for row, widget in enumerate(
            values
        ):

            grid.addWidget(
                widget,
                row,
                1,
            )

        layout.addLayout(
            grid
        )

    # ------------------------------------------------------------------------
    # GUI update
    # ------------------------------------------------------------------------

    def update_gui(self):

        with state_lock:

            frame = (
                state["frame"].copy()
                if state["frame"] is not None
                else None
            )

            hand_world = (
                state["hand_world"].copy()
            )

            translation = (
                state["translation"].copy()
            )

            h2r = (
                state["h2r_position"].copy()
            )

            axis_angle = (
                state["axis_angle"].copy()
            )

            ik = (
                state["ik"].copy()
            )

            gripper = (
                state["gripper"]
            )

            fps = (
                state["fps"]
            )

            hand_detected = (
                state["hand_detected"]
            )

            pose_detected = (
                state["pose_detected"]
            )

        # --------------------------------------------------------------------
        # Video
        # --------------------------------------------------------------------

        if frame is not None:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            height, width, channels = (
                rgb.shape
            )

            bytes_per_line = (
                channels * width
            )

            image = QImage(
                rgb.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888,
            )

            pixmap = QPixmap.fromImage(
                image
            )

            pixmap = pixmap.scaled(
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            self.video_label.setPixmap(
                pixmap
            )

        # --------------------------------------------------------------------
        # Pose
        # --------------------------------------------------------------------

        self.pose_value.setText(
            "YES"
            if pose_detected
            else "NO"
        )

        # --------------------------------------------------------------------
        # Hand
        # --------------------------------------------------------------------

        self.hand_value.setText(
            "YES"
            if hand_detected
            else "NO"
        )

        # --------------------------------------------------------------------
        # HandWorld
        # --------------------------------------------------------------------

        self.hand_world_value.setText(
            f"x={hand_world[0]:+.4f}   "
            f"y={hand_world[1]:+.4f}   "
            f"z={hand_world[2]:+.4f}"
        )

        # --------------------------------------------------------------------
        # Translation
        # --------------------------------------------------------------------

        self.translation_value.setText(
            f"x={translation[0]:+.4f}   "
            f"y={translation[1]:+.4f}   "
            f"z={translation[2]:+.4f}"
        )

        # --------------------------------------------------------------------
        # H2R
        # --------------------------------------------------------------------

        self.h2r_value.setText(
            f"x={h2r[0]:+.4f}   "
            f"y={h2r[1]:+.4f}   "
            f"z={h2r[2]:+.4f}"
        )

        # --------------------------------------------------------------------
        # Axis angle
        # --------------------------------------------------------------------

        self.axis_value.setText(
            f"["
            f"{axis_angle[0]:+.3f}, "
            f"{axis_angle[1]:+.3f}, "
            f"{axis_angle[2]:+.3f}"
            f"] "
            f"{axis_angle[3]:+.1f}°"
        )

        # --------------------------------------------------------------------
        # IK
        # --------------------------------------------------------------------

        reachable = (
            "OK"
            if ik["reachable"]
            else "UNREACHABLE"
        )

        self.ik_value.setText(
            f"B={ik['base']:.1f}  "
            f"S={ik['shoulder']:.1f}  "
            f"E={ik['elbow']:.1f}  "
            f"WP={ik['wrist_pitch']:.1f}  "
            f"WR={ik['wrist_roll']:.1f}  "
            f"[{reachable}]"
        )

        # --------------------------------------------------------------------
        # Gripper
        # --------------------------------------------------------------------

        gripper_name = {
            0: "100 (OPEN)",
            2: "180 (CLOSED)",
        }.get(
            gripper,
            str(gripper),
        )

        self.gripper_value.setText(
            gripper_name
        )

        # --------------------------------------------------------------------
        # FPS
        # --------------------------------------------------------------------

        self.fps_value.setText(
            f"{fps:.1f}"
        )


# ============================================================================
# Main
# ============================================================================

def main():

    print()

    print(
        "Starting teleop.py"
    )

    print(
        f"Python: "
        f"{sys.version.split()[0]}"
    )

    print(
        f"MediaPipe: "
        f"{getattr(mp, '__version__', 'unknown')}"
    )

    print()

    print(
        "[IK] Arm dimensions:"
    )

    print(
        f"     L1 = {IK_L1:.3f} m"
    )

    print(
        f"     L2 = {IK_L2:.3f} m"
    )

    print(
        f"     L3 = {IK_L3:.3f} m"
    )

    print()

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    output = ConsoleOutput()

    # ------------------------------------------------------------------------
    # Vision thread
    # ------------------------------------------------------------------------

    vision_thread = threading.Thread(
        target=run_vision,
        args=(output,),
        daemon=True,
        name="VisionThread",
    )

    vision_thread.start()

    # ------------------------------------------------------------------------
    # Qt
    # ------------------------------------------------------------------------

    app = QApplication(
        sys.argv
    )

    window = VisionWindow()

    window.show()

    return app.exec()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )