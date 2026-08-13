"""Camera + MediaPipe processing.

This module owns camera acquisition and MediaPipe inference. It does not
know about robot servo mapping.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

from arm_geometry import calculate_arm_angles, calculate_gripper

CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


def draw_pose(frame, pose_result):
    if not pose_result.pose_landmarks:
        return

    landmarks = pose_result.pose_landmarks[0]
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (23, 24), (11, 23), (12, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
    ]

    height, width = frame.shape[:2]

    for a, b in connections:
        p1, p2 = landmarks[a], landmarks[b]
        cv2.line(
            frame,
            (int(p1.x * width), int(p1.y * height)),
            (int(p2.x * width), int(p2.y * height)),
            (0, 255, 0),
            2,
        )

    for index in [
        LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW,
        RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    ]:
        p = landmarks[index]
        cv2.circle(
            frame,
            (int(p.x * width), int(p.y * height)),
            5,
            (0, 0, 255),
            -1,
        )


def draw_hand(frame, hand_result):
    if not hand_result.hand_landmarks:
        return

    height, width = frame.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    for hand in hand_result.hand_landmarks:
        for a, b in connections:
            p1, p2 = hand[a], hand[b]
            cv2.line(
                frame,
                (int(p1.x * width), int(p1.y * height)),
                (int(p2.x * width), int(p2.y * height)),
                (255, 180, 0),
                2,
            )

        for p in hand:
            cv2.circle(
                frame,
                (int(p.x * width), int(p.y * height)),
                3,
                (255, 180, 0),
                -1,
            )


class VisionProcessor:
    def __init__(self, pose_model: Path, hand_model: Path, state):
        self.pose_model = pose_model
        self.hand_model = hand_model
        self.state = state

    def run(self, publishers, mapper):
        print("Starting MediaPipe vision...")

        if not self.pose_model.exists():
            self.state["error"] = f"Missing model: {self.pose_model}"
            print(self.state["error"])
            return

        if not self.hand_model.exists():
            self.state["error"] = f"Missing model: {self.hand_model}"
            print(self.state["error"])
            return

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.pose_model)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.hand_model)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        print(f"Opening camera {CAMERA_INDEX}...")
        cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            self.state["error"] = "Could not open camera"
            pose_landmarker.close()
            hand_landmarker.close()
            print(self.state["error"])
            return

        self.state["camera_ok"] = True
        print("Camera opened successfully.")

        start_time = time.monotonic()
        fps_start = time.monotonic()
        frame_counter = 0
        last_gripper = None

        try:
            while self.state["running"]:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue

                frame = cv2.flip(frame, 1)

                frame_counter += 1
                elapsed = time.monotonic() - fps_start
                if elapsed >= 1.0:
                    self.state["fps"] = frame_counter / elapsed
                    frame_counter = 0
                    fps_start = time.monotonic()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                pose_result = pose_landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )
                hand_result = hand_landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                self.state["pose_ok"] = bool(pose_result.pose_landmarks)
                self.state["hand_ok"] = bool(hand_result.hand_landmarks)

                draw_pose(frame, pose_result)
                draw_hand(frame, hand_result)

                if pose_result.pose_world_landmarks:
                    landmarks = pose_result.pose_world_landmarks[0]

                    hand_world = None
                    if hand_result.hand_world_landmarks:
                        hand_world = hand_result.hand_world_landmarks[0]

                    human_angles = calculate_arm_angles(
                        landmarks, hand_world
                    )

                    gripper = calculate_gripper(hand_world)
                    if gripper is None:
                        gripper = last_gripper
                    else:
                        last_gripper = gripper

                    self.state["human_angles"] = human_angles.copy()
                    self.state["gripper"] = gripper

                    robot_angles = mapper.update(
                        human_angles,
                        gripper,
                    )
                    self.state["angles"] = robot_angles.copy()

                    for publisher in publishers:
                        try:
                            publisher.publish(robot_angles)
                        except Exception as exc:
                            print("Publisher error:", exc)

                labels = [
                    "BASE", "SHOULDER", "ELBOW",
                    "WRIST PITCH", "WRIST ROLL", "GRIPPER",
                ]

                for i, label in enumerate(labels):
                    cv2.putText(
                        frame,
                        f"{label}: {self.state['angles'][i]:.0f}",
                        (10, 30 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    frame,
                    f"FPS: {self.state['fps']:.1f}",
                    (10, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "CALIBRATED" if mapper.calibrated else "CALIBRATE NEUTRAL",
                    (10, 245),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                self.state["frame"] = frame.copy()

        except Exception as exc:
            self.state["error"] = f"Vision error: {exc}"
            print(self.state["error"])
            import traceback
            traceback.print_exc()

        finally:
            cap.release()
            pose_landmarker.close()
            hand_landmarker.close()
            print("Vision thread stopped.")
