#!/usr/bin/env python3

import sys
import time
import logging
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import zmq

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

MODEL_DIR = (
    BASE_DIR / "models"
)

POSE_MODEL = (
    MODEL_DIR /
    "pose_landmarker_full.task"
)

HAND_MODEL = (
    MODEL_DIR /
    "hand_landmarker.task"
)


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

# macOS uses AVFoundation.
# On other platforms OpenCV will use its normal backend.

if sys.platform == "darwin":
    CAMERA_BACKEND = cv2.CAP_AVFOUNDATION
else:
    CAMERA_BACKEND = cv2.CAP_ANY


CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


# ZMQ publisher
#
# The application listens on all interfaces.
#
# Subscriber example:
#
#     tcp://192.168.x.x:5555
#

ZMQ_ADDRESS = "tcp://*:5555"


# ============================================================
# MEDIAPIPE LANDMARKS
# ============================================================

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

H_WRIST = 0
H_THUMB = 4
H_INDEX = 5
H_MIDDLE = 9
H_RING = 13
H_PINKY = 17

H_INDEX_PIP = 6
H_MIDDLE_PIP = 10
H_RING_PIP = 14


# ============================================================
# ROBOT CONFIGURATION
# ============================================================

# Robot order:
#
#   0 = Base
#   1 = Shoulder
#   2 = Elbow
#   3 = Wrist Pitch
#   4 = Wrist Roll
#   5 = Gripper
#

SERVO_MIN = np.array([
    0,
    0,
    0,
    0,
    0,
    0
], dtype=float)


SERVO_MAX = np.array([
    180,
    180,
    180,
    180,
    180,
    180
], dtype=float)


# Your robot HOME position.
#
# base -> gripper

HOME = np.array([
    90,
    180,
    180,
    100,
    90,
    170
], dtype=float)


# Human movement scaling

ANGLE_SCALE = np.array([
    1.0,    # base
    1.0,    # shoulder
    1.0,    # elbow
    1.0,    # wrist pitch
    1.0,    # wrist roll
    1.0     # gripper
], dtype=float)


# Neutral robot position

ANGLE_OFFSET = HOME.copy()


# Reverse individual axes here.
#
# Example:
#
#     -1
#
# reverses that servo.

DIRECTION = np.array([
    1,
    1,
    1,
    1,
    1,
    1
], dtype=float)


# ------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------

SMOOTHING = 0.25

filtered_angles = HOME.copy()


# ------------------------------------------------------------
# Gripper
# ------------------------------------------------------------

GRIPPER_OPEN = 0

GRIPPER_CLOSED = 180


# ============================================================
# GLOBAL STATE
# ============================================================

state = {

    "frame": None,

    "angles": HOME.copy(),

    "human_angles": np.zeros(5),

    "gripper": HOME[5],

    "camera_ok": False,

    "pose_ok": False,

    "hand_ok": False,

    "running": True,

    "error": None
}


# ============================================================
# OUTPUT INTERFACE
# ============================================================

class AnglePublisher:

    def publish(self, angles):
        raise NotImplementedError

    def close(self):
        pass


# ============================================================
# ZMQ PUBLISHER
# ============================================================

class ZMQAnglePublisher(
        AnglePublisher):

    def __init__(
            self,
            address=ZMQ_ADDRESS):

        self.context = (
            zmq.Context()
        )

        self.socket = (
            self.context.socket(
                zmq.PUB
            )
        )

        self.socket.bind(
            address
        )

        print(
            f"ZMQ publisher listening "
            f"on {address}"
        )

    def publish(self, angles):

        message = {

            "timestamp": time.time(),

            "base": int(
                round(angles[0])
            ),

            "shoulder": int(
                round(angles[1])
            ),

            "elbow": int(
                round(angles[2])
            ),

            "wrist_pitch": int(
                round(angles[3])
            ),

            "wrist_roll": int(
                round(angles[4])
            ),

            "gripper": int(
                round(angles[5])
            )
        }

        self.socket.send_json(
            message
        )

    def close(self):

        self.socket.close()

        self.context.term()


# ============================================================
# LOG PUBLISHER
# ============================================================

class LogAnglePublisher(
        AnglePublisher):

    def __init__(
            self,
            filename="robot_angles.log"):

        self.logger = logging.getLogger(
            "robot_angles"
        )

        self.logger.setLevel(
            logging.INFO
        )

        # Avoid duplicate handlers
        if not self.logger.handlers:

            handler = (
                logging.FileHandler(
                    filename
                )
            )

            formatter = (
                logging.Formatter(
                    "%(asctime)s "
                    "%(message)s"
                )
            )

            handler.setFormatter(
                formatter
            )

            self.logger.addHandler(
                handler
            )

    def publish(self, angles):

        self.logger.info(
            "BASE=%3d "
            "SHOULDER=%3d "
            "ELBOW=%3d "
            "WRIST_PITCH=%3d "
            "WRIST_ROLL=%3d "
            "GRIPPER=%3d",

            round(angles[0]),
            round(angles[1]),
            round(angles[2]),
            round(angles[3]),
            round(angles[4]),
            round(angles[5])
        )


# ============================================================
# VECTOR UTILITIES
# ============================================================

def normalize(v):

    length = np.linalg.norm(v)

    if length < 1e-8:

        return np.zeros_like(v)

    return v / length


def angle_between(a, b):

    a = normalize(a)

    b = normalize(b)

    value = np.clip(
        np.dot(a, b),
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(value)
    )


def signed_angle(
        a,
        b,
        axis):

    a = normalize(a)
    b = normalize(b)
    axis = normalize(axis)

    x = np.dot(
        a,
        b
    )

    y = np.dot(
        axis,
        np.cross(a, b)
    )

    return np.degrees(
        np.arctan2(y, x)
    )


# ============================================================
# LANDMARK -> NUMPY
# ============================================================

def landmark_vector(
        landmark):

    return np.array([
        landmark.x,
        landmark.y,
        landmark.z
    ], dtype=float)


# ============================================================
# ARM ANGLE CALCULATION
# ============================================================

def calculate_arm_angles(
        shoulder,
        elbow,
        wrist,
        hand):

    upper_arm = (
        elbow - shoulder
    )

    forearm = (
        wrist - elbow
    )

    upper_dir = normalize(
        upper_arm
    )

    forearm_dir = normalize(
        forearm
    )

    # --------------------------------------------------------
    # BASE
    #
    # Rotation of the upper arm around the body.
    # --------------------------------------------------------

    base = np.degrees(
        np.arctan2(
            upper_dir[2],
            upper_dir[0]
        )
    )

    if base < 0:

        base += 360

    base = np.clip(
        base,
        0,
        180
    )


    # --------------------------------------------------------
    # SHOULDER
    #
    # Angle between upper arm and vertical.
    # --------------------------------------------------------

    vertical = np.array([
        0.0,
        -1.0,
        0.0
    ])

    shoulder_angle = (
        angle_between(
            upper_dir,
            vertical
        )
    )


    # --------------------------------------------------------
    # ELBOW
    #
    # Straight arm ~= 180 degrees.
    # Bent arm -> smaller angle.
    # --------------------------------------------------------

    elbow_angle = (
        angle_between(
            -upper_dir,
            forearm_dir
        )
    )


    # --------------------------------------------------------
    # WRIST
    # --------------------------------------------------------

    wrist_pitch = 90.0

    wrist_roll = 90.0


    if hand is not None:

        hand_wrist = (
            landmark_vector(
                hand[H_WRIST]
            )
        )

        index = (
            landmark_vector(
                hand[H_INDEX]
            )
        )

        middle = (
            landmark_vector(
                hand[H_MIDDLE]
            )
        )

        pinky = (
            landmark_vector(
                hand[H_PINKY]
            )
        )


        # ----------------------------------------------------
        # Hand forward direction
        # ----------------------------------------------------

        hand_forward = normalize(
            middle - hand_wrist
        )


        # ----------------------------------------------------
        # Palm width
        # ----------------------------------------------------

        palm_width = normalize(
            pinky - index
        )


        # ----------------------------------------------------
        # Palm normal
        # ----------------------------------------------------

        palm_normal = normalize(
            np.cross(
                palm_width,
                hand_forward
            )
        )


        # ----------------------------------------------------
        # Wrist pitch
        # ----------------------------------------------------

        wrist_pitch = (
            angle_between(
                forearm_dir,
                hand_forward
            )
        )


        # ----------------------------------------------------
        # Wrist roll
        # ----------------------------------------------------

        reference = np.array([
            0.0,
            1.0,
            0.0
        ])


        reference -= (
            np.dot(
                reference,
                forearm_dir
            )
            * forearm_dir
        )


        reference = normalize(
            reference
        )


        palm_projected = (
            palm_normal
            -
            np.dot(
                palm_normal,
                forearm_dir
            )
            *
            forearm_dir
        )


        palm_projected = normalize(
            palm_projected
        )


        if (
            np.linalg.norm(
                reference
            ) > 0.1
            and
            np.linalg.norm(
                palm_projected
            ) > 0.1
        ):

            wrist_roll = (
                signed_angle(
                    reference,
                    palm_projected,
                    forearm_dir
                )
                + 90.0
            )


    return np.array([
        base,
        shoulder_angle,
        elbow_angle,
        wrist_pitch,
        wrist_roll
    ])


# ============================================================
# GRIPPER
# ============================================================

def calculate_gripper(
        hand):

    if hand is None:

        return None


    wrist = landmark_vector(
        hand[H_WRIST]
    )


    distances = []


    for index in [
        8,
        12,
        16,
        20
    ]:

        fingertip = (
            landmark_vector(
                hand[index]
            )
        )

        distance = np.linalg.norm(
            fingertip - wrist
        )

        distances.append(
            distance
        )


    # Three or more extended fingers
    # means open hand.

    extended = sum(
        distance > 0.30
        for distance in distances
    )


    if extended >= 3:

        return GRIPPER_OPEN


    return GRIPPER_CLOSED


# ============================================================
# HUMAN -> ROBOT
# ============================================================

def human_to_robot(
        human_angles,
        gripper):

    robot = np.zeros(
        6,
        dtype=float
    )


    robot[:5] = (
        human_angles
        *
        ANGLE_SCALE[:5]
        +
        ANGLE_OFFSET[:5]
    )


    robot[:5] *= (
        DIRECTION[:5]
    )


    if gripper is None:

        robot[5] = (
            HOME[5]
        )

    else:

        robot[5] = gripper


    return np.clip(
        robot,
        SERVO_MIN,
        SERVO_MAX
    )


# ============================================================
# SMOOTHING
# ============================================================

def smooth_angles(
        angles):

    global filtered_angles


    filtered_angles = (
        filtered_angles
        *
        (1.0 - SMOOTHING)
        +
        angles
        *
        SMOOTHING
    )


    return filtered_angles.copy()


# ============================================================
# DRAW POSE
# ============================================================

def draw_pose(
        frame,
        pose_result):

    if not pose_result.pose_landmarks:

        return


    landmarks = (
        pose_result.pose_landmarks[0]
    )


    connections = [
        (11, 12),  # shoulders

        (11, 13),
        (13, 15),  # left arm

        (12, 14),
        (14, 16),  # right arm

        (23, 24),  # hips

        (11, 23),
        (12, 24),  # torso

        (23, 25),
        (25, 27),  # left leg

        (24, 26),
        (26, 28),  # right leg
    ]


    height, width = (
        frame.shape[:2]
    )


    for a, b in connections:

        p1 = landmarks[a]
        p2 = landmarks[b]


        x1 = int(
            p1.x * width
        )

        y1 = int(
            p1.y * height
        )

        x2 = int(
            p2.x * width
        )

        y2 = int(
            p2.y * height
        )


        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


    # Draw right shoulder/elbow/wrist

    for index in [
        RIGHT_SHOULDER,
        RIGHT_ELBOW,
        RIGHT_WRIST
    ]:

        p = landmarks[index]

        x = int(
            p.x * width
        )

        y = int(
            p.y * height
        )


        cv2.circle(
            frame,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(
        frame,
        hand_result):

    if not hand_result.hand_landmarks:

        return


    height, width = (
        frame.shape[:2]
    )


    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (5, 9),
        (9, 13),
        (13, 17)
    ]


    for hand in (
        hand_result.hand_landmarks
    ):

        for a, b in connections:

            p1 = hand[a]
            p2 = hand[b]


            x1 = int(
                p1.x * width
            )

            y1 = int(
                p1.y * height
            )

            x2 = int(
                p2.x * width
            )

            y2 = int(
                p2.y * height
            )


            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 180, 0),
                2
            )


        for p in hand:

            x = int(
                p.x * width
            )

            y = int(
                p.y * height
            )


            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 180, 0),
                -1
            )


# ============================================================
# VISION THREAD
# ============================================================

def run_vision(
        publishers):

    print(
        "Starting MediaPipe vision..."
    )


    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    if not POSE_MODEL.exists():

        state["error"] = (
            f"Missing model: "
            f"{POSE_MODEL}"
        )

        state["running"] = False

        print(
            state["error"]
        )

        return


    if not HAND_MODEL.exists():

        state["error"] = (
            f"Missing model: "
            f"{HAND_MODEL}"
        )

        state["running"] = False

        print(
            state["error"]
        )

        return


    # --------------------------------------------------------
    # MediaPipe Tasks API
    # --------------------------------------------------------

    from mediapipe.tasks import python

    from mediapipe.tasks.python import vision


    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    pose_options = (
        vision.PoseLandmarkerOptions(

            base_options=(
                python.BaseOptions(
                    model_asset_path=str(
                        POSE_MODEL
                    )
                )
            ),

            running_mode=(
                vision.RunningMode.VIDEO
            ),

            num_poses=1,

            min_pose_detection_confidence=0.5,

            min_pose_presence_confidence=0.5,

            min_tracking_confidence=0.5,

            output_segmentation_masks=False
        )
    )


    pose_landmarker = (
        vision.PoseLandmarker
        .create_from_options(
            pose_options
        )
    )


    # --------------------------------------------------------
    # Hand
    # --------------------------------------------------------

    hand_options = (
        vision.HandLandmarkerOptions(

            base_options=(
                python.BaseOptions(
                    model_asset_path=str(
                        HAND_MODEL
                    )
                )
            ),

            running_mode=(
                vision.RunningMode.VIDEO
            ),

            num_hands=1,

            min_hand_detection_confidence=0.5,

            min_hand_presence_confidence=0.5,

            min_tracking_confidence=0.5
        )
    )


    hand_landmarker = (
        vision.HandLandmarker
        .create_from_options(
            hand_options
        )
    )


    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    print(
        f"Opening camera "
        f"{CAMERA_INDEX}..."
    )


    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        CAMERA_BACKEND
    )


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )


    if not cap.isOpened():

        state["error"] = (
            "Could not open camera"
        )

        state["camera_ok"] = False

        print(
            state["error"]
        )

        pose_landmarker.close()
        hand_landmarker.close()

        return


    state["camera_ok"] = True


    print(
        "Camera opened successfully."
    )


    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    start_time = time.monotonic()

    last_gripper = (
        HOME[5]
    )


    try:

        while state["running"]:

            ret, frame = (
                cap.read()
            )


            if not ret:

                print(
                    "Camera frame read failed."
                )

                time.sleep(
                    0.05
                )

                continue


            # Mirror camera

            frame = cv2.flip(
                frame,
                1
            )


            # ------------------------------------------------
            # MediaPipe input
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )


            timestamp_ms = int(
                (
                    time.monotonic()
                    -
                    start_time
                )
                *
                1000
            )


            # ------------------------------------------------
            # Pose
            # ------------------------------------------------

            pose_result = (
                pose_landmarker
                .detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )


            state["pose_ok"] = (
                len(
                    pose_result.pose_landmarks
                ) > 0
            )


            # ------------------------------------------------
            # Hand
            # ------------------------------------------------

            hand_result = (
                hand_landmarker
                .detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )


            state["hand_ok"] = (
                len(
                    hand_result.hand_landmarks
                ) > 0
            )


            # ------------------------------------------------
            # Draw
            # ------------------------------------------------

            draw_pose(
                frame,
                pose_result
            )


            draw_hand(
                frame,
                hand_result
            )


            # ------------------------------------------------
            # Calculate robot angles
            # ------------------------------------------------

            if pose_result.pose_world_landmarks:

                world_landmarks = (
                    pose_result
                    .pose_world_landmarks[0]
                )


                shoulder = (
                    landmark_vector(
                        world_landmarks[
                            RIGHT_SHOULDER
                        ]
                    )
                )


                elbow = (
                    landmark_vector(
                        world_landmarks[
                            RIGHT_ELBOW
                        ]
                    )
                )


                wrist = (
                    landmark_vector(
                        world_landmarks[
                            RIGHT_WRIST
                        ]
                    )
                )


                # Hand world landmarks

                hand_world = None


                if (
                    hand_result
                    .hand_world_landmarks
                ):

                    hand_world = (
                        hand_result
                        .hand_world_landmarks[0]
                    )


                human_angles = (
                    calculate_arm_angles(
                        shoulder,
                        elbow,
                        wrist,
                        hand_world
                    )
                )


                gripper = (
                    calculate_gripper(
                        hand_world
                    )
                )


                if gripper is None:

                    gripper = (
                        last_gripper
                    )


                last_gripper = gripper


                robot_angles = (
                    human_to_robot(
                        human_angles,
                        gripper
                    )
                )


                robot_angles = (
                    smooth_angles(
                        robot_angles
                    )
                )


                # --------------------------------------------
                # Publish
                # --------------------------------------------

                for publisher in publishers:

                    try:

                        publisher.publish(
                            robot_angles
                        )

                    except Exception as exc:

                        print(
                            "Publisher error:",
                            exc
                        )


                # --------------------------------------------
                # State
                # --------------------------------------------

                state["human_angles"] = (
                    human_angles.copy()
                )

                state["angles"] = (
                    robot_angles.copy()
                )

                state["gripper"] = (
                    robot_angles[5]
                )


            # ------------------------------------------------
            # Video overlay
            # ------------------------------------------------

            angles = (
                state["angles"]
            )


            labels = [

                "BASE",

                "SHOULDER",

                "ELBOW",

                "WRIST PITCH",

                "WRIST ROLL",

                "GRIPPER"
            ]


            for i, label in enumerate(
                labels
            ):

                text = (
                    f"{label}: "
                    f"{angles[i]:.0f}"
                )


                cv2.putText(
                    frame,
                    text,
                    (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            status = (
                "POSE: OK"
                if state["pose_ok"]
                else "POSE: ---"
            )


            cv2.putText(
                frame,
                status,
                (10, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0)
                if state["pose_ok"]
                else (0, 0, 255),
                2
            )


            hand_status = (
                "HAND: OK"
                if state["hand_ok"]
                else "HAND: ---"
            )


            cv2.putText(
                frame,
                hand_status,
                (10, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0)
                if state["hand_ok"]
                else (0, 0, 255),
                2
            )


            state["frame"] = (
                frame.copy()
            )


    except Exception as exc:

        state["error"] = (
            f"Vision error: {exc}"
        )

        print(
            state["error"]
        )

        import traceback

        traceback.print_exc()


    finally:

        cap.release()

        pose_landmarker.close()

        hand_landmarker.close()

        print(
            "Vision thread stopped."
        )


# ============================================================
# PYSIDE6 GUI
# ============================================================

class TeleopWindow(QWidget):

    def __init__(self):

        super().__init__()


        self.setWindowTitle(
            "MediaPipe 6DOF Robot Teleoperation"
        )


        self.setFixedSize(
            760,
            680
        )


        # ----------------------------------------------------
        # Video
        # ----------------------------------------------------

        self.video_label = QLabel()

        self.video_label.setFixedSize(
            720,
            480
        )

        self.video_label.setAlignment(
            Qt.AlignCenter
        )

        self.video_label.setText(
            "Waiting for camera..."
        )


        # ----------------------------------------------------
        # Joint display
        # ----------------------------------------------------

        labels = [

            "Base",

            "Shoulder",

            "Elbow",

            "Wrist Pitch",

            "Wrist Roll",

            "Gripper"
        ]


        self.joint_values = []


        joint_layout = (
            QGridLayout()
        )


        for row, name in enumerate(
            labels
        ):

            label = QLabel(
                f"{name}:"
            )


            value = QLabel(
                "0°"
            )


            value.setMinimumWidth(
                70
            )


            joint_layout.addWidget(
                label,
                row,
                0
            )


            joint_layout.addWidget(
                value,
                row,
                1
            )


            self.joint_values.append(
                value
            )


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.status_label = QLabel(
            "Starting..."
        )


        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        layout = (
            QVBoxLayout()
        )


        layout.addWidget(
            self.video_label
        )


        layout.addLayout(
            joint_layout
        )


        layout.addWidget(
            self.status_label
        )


        self.setLayout(
            layout
        )


        # ----------------------------------------------------
        # GUI timer
        # ----------------------------------------------------

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_gui
        )

        self.timer.start(
            50
        )


    def update_gui(self):

        frame = state["frame"]


        if frame is not None:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
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
                QImage.Format_RGB888
            )


            pixmap = QPixmap.fromImage(
                image
            )


            self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )


        # ----------------------------------------------------
        # Joint values
        # ----------------------------------------------------

        angles = state["angles"]


        for i in range(6):

            self.joint_values[i].setText(
                f"{angles[i]:.0f}°"
            )


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if state["error"]:

            self.status_label.setText(
                state["error"]
            )

        elif not state["camera_ok"]:

            self.status_label.setText(
                "Opening camera..."
            )

        elif state["pose_ok"]:

            if state["hand_ok"]:

                self.status_label.setText(
                    "Camera OK | "
                    "Pose OK | "
                    "Hand OK | "
                    "ZMQ PUB active"
                )

            else:

                self.status_label.setText(
                    "Camera OK | "
                    "Pose OK | "
                    "Hand not detected | "
                    "ZMQ PUB active"
                )

        else:

            self.status_label.setText(
                "Camera OK | "
                "Waiting for person..."
            )


    def closeEvent(
            self,
            event):

        state["running"] = False

        event.accept()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Verify models before starting GUI
    # --------------------------------------------------------

    if not POSE_MODEL.exists():

        print(
            f"Missing: {POSE_MODEL}"
        )

        print(
            "Run:"
        )

        print(
            "    python setup.py"
        )

        return 1


    if not HAND_MODEL.exists():

        print(
            f"Missing: {HAND_MODEL}"
        )

        print(
            "Run:"
        )

        print(
            "    python setup.py"
        )

        return 1


    # --------------------------------------------------------
    # Publishers
    # --------------------------------------------------------

    publishers = [

        ZMQAnglePublisher(
            ZMQ_ADDRESS
        ),

        LogAnglePublisher(
            BASE_DIR /
            "robot_angles.log"
        )
    ]


    # --------------------------------------------------------
    # Vision thread
    # --------------------------------------------------------

    vision_thread = (
        threading.Thread(
            target=run_vision,
            args=(publishers,),
            daemon=True
        )
    )


    vision_thread.start()


    # --------------------------------------------------------
    # Qt
    # --------------------------------------------------------

    app = QApplication(
        sys.argv
    )


    window = TeleopWindow()

    window.show()


    try:

        return app.exec()


    finally:

        state["running"] = False


        for publisher in publishers:

            try:

                publisher.close()

            except Exception:
                pass


        vision_thread.join(
            timeout=2.0
        )


if __name__ == "__main__":

    sys.exit(
        main()
    )