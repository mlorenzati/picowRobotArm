"""Camera acquisition + MediaPipe Holistic processing thread.

This module owns:
    - Camera source management (OpenCV webcam or DepthAI OAK-D)
    - MediaPipe HolisticLandmarker inference (VIDEO mode)
    - Pose / hand overlay drawing on the raw frame
    - Angle computation (arm_geometry) and robot mapping
    - Publisher dispatch

It does NOT know about Qt widgets. It communicates with the GUI via the
shared ``state`` dict (written from this thread, read by the GUI thread).

State keys written by VisionThread:
    frame          ndarray | None   – BGR frame with overlays for display
    angles         ndarray          – current 6-DOF robot servo angles
    human_angles   ndarray | None   – raw human anatomical angles (5-DOF)
    gripper        float  | None    – raw gripper servo value
    gripper_state  str              – "OPEN" | "CLOSED"
    camera_ok      bool
    pose_ok        bool
    hand_ok        bool
    running        bool             – set False by the GUI to stop the thread
    error          str  | None
    fps            float
    source         dict             – {"type": "camera"|"video", "value": int|str}
    source_name    str
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from arm_geometry import (
    calculate_arm_angles,
    calculate_gripper,
    calculate_gripper_state,
    LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
)
from robot_mapping import HOME

# ---------------------------------------------------------------------------
# Camera constants
# ---------------------------------------------------------------------------

CAMERA_BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ---------------------------------------------------------------------------
# Pose skeleton connections used for drawing
# ---------------------------------------------------------------------------

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 24), (11, 23), (12, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

POSE_KEY_LANDMARKS = [
    LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW,
    RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_pose(frame: np.ndarray, landmarks) -> None:
    """Draw skeleton connections and key joint circles on the frame."""
    h, w = frame.shape[:2]
    for a, b in POSE_CONNECTIONS:
        p1, p2 = landmarks[a], landmarks[b]
        cv2.line(
            frame,
            (int(p1.x * w), int(p1.y * h)),
            (int(p2.x * w), int(p2.y * h)),
            (0, 255, 0), 2,
        )
    for idx in POSE_KEY_LANDMARKS:
        p = landmarks[idx]
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 5, (0, 0, 255), -1)


def _draw_hand(frame: np.ndarray, hand_landmarks) -> None:
    """Draw hand landmark skeleton on the frame."""
    h, w = frame.shape[:2]
    for a, b in HAND_CONNECTIONS:
        p1, p2 = hand_landmarks[a], hand_landmarks[b]
        cv2.line(
            frame,
            (int(p1.x * w), int(p1.y * h)),
            (int(p2.x * w), int(p2.y * h)),
            (255, 180, 0), 2,
        )
    for p in hand_landmarks:
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (255, 180, 0), -1)


def _put_overlay(frame: np.ndarray, state: dict) -> None:
    """Render angle values, FPS and calibration status as overlay text."""
    labels = [
        "BASE", "SHOULDER", "ELBOW",
        "WRIST PITCH", "WRIST ROLL", "GRIPPER",
    ]
    angles = state["angles"]
    for i, label in enumerate(labels):
        cv2.putText(
            frame,
            f"{label}: {angles[i]:.0f}",
            (10, 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    gripper_str = state.get("gripper_state", "---")
    cv2.putText(
        frame,
        f"GRIPPER: {gripper_str}",
        (10, 30 + 6 * 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
    )

    cv2.putText(
        frame,
        f"FPS: {state['fps']:.1f}",
        (10, 30 + 7 * 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )

    cal_text = "CALIBRATED" if state.get("calibrated", False) else "NOT CALIBRATED"
    cal_color = (0, 255, 0) if state.get("calibrated", False) else (0, 80, 255)
    cv2.putText(
        frame,
        cal_text,
        (10, 30 + 8 * 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cal_color, 2,
    )

    cv2.putText(
        frame,
        state.get("source_name", ""),
        (10, 30 + 9 * 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )


# ---------------------------------------------------------------------------
# HolisticLandmarker compatibility adapter
# ---------------------------------------------------------------------------

class _LandmarkContainer:
    """Wrap a flat landmark list behind a .landmark attribute."""
    def __init__(self, landmarks):
        self.landmark = landmarks

    def __getitem__(self, idx):
        return self.landmark[idx]

    def __iter__(self):
        return iter(self.landmark)

    def __len__(self):
        return len(self.landmark)


def _as_landmark_list(value):
    if not value:
        return None
    first = value[0]
    if hasattr(first, "x") and hasattr(first, "y"):
        return value
    return first


class _HolisticAdapter:
    """Adapt HolisticLandmarker Tasks result to simple attribute access."""
    def __init__(self, result):
        pose       = _as_landmark_list(result.pose_landmarks)
        pose_world = _as_landmark_list(result.pose_world_landmarks)
        left_hand  = _as_landmark_list(result.left_hand_landmarks)
        right_hand = _as_landmark_list(result.right_hand_landmarks)

        self.pose_landmarks        = _LandmarkContainer(pose)       if pose       else None
        self.pose_world_landmarks  = _LandmarkContainer(pose_world) if pose_world else None
        self.left_hand_landmarks   = _LandmarkContainer(left_hand)  if left_hand  else None
        self.right_hand_landmarks  = _LandmarkContainer(right_hand) if right_hand else None


# ---------------------------------------------------------------------------
# Camera source helpers
# ---------------------------------------------------------------------------

def _open_opencv_source(source: dict) -> cv2.VideoCapture:
    if source["type"] == "video":
        cap = cv2.VideoCapture(source["value"])
    else:
        index = int(source["value"])
        cap = cv2.VideoCapture(index, CAMERA_BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return cap


# ---------------------------------------------------------------------------
# Main vision thread
# ---------------------------------------------------------------------------

class VisionThread:
    """Runs MediaPipe Holistic inference in a background thread.

    Parameters
    ----------
    model_path:
        Path to the holistic_landmarker.task model file.
    state:
        Shared mutable dict for communicating with the GUI thread.
    """

    def __init__(self, model_path: Path, state: dict):
        self._model_path = model_path
        self._state = state

    # ------------------------------------------------------------------

    def run(self, publishers, mapper) -> None:
        """Thread entry point. Blocks until state["running"] is False."""
        state = self._state
        print("[Vision] Starting MediaPipe Holistic thread...")

        if not self._model_path.exists():
            state["error"] = f"Missing model: {self._model_path}"
            print(state["error"])
            return

        # ----------------------------------------------------------------
        # Build HolisticLandmarker
        # ----------------------------------------------------------------
        try:
            base_opts = mp.tasks.BaseOptions(
                model_asset_path=str(self._model_path)
            )
            holistic_opts = mp.tasks.vision.HolisticLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_pose_landmarks_confidence=0.5,
                min_hand_landmarks_confidence=0.5,
            )
            holistic = mp.tasks.vision.HolisticLandmarker.create_from_options(
                holistic_opts
            )
        except Exception as exc:
            state["error"] = f"MediaPipe init error: {exc}"
            print(state["error"])
            return

        # ----------------------------------------------------------------
        # Open camera
        # ----------------------------------------------------------------
        source = dict(state.get("source", {"type": "camera", "value": 0}))

        # Support DepthAI if available and requested
        use_depthai = source.get("type") == "depthai"
        cap = None
        depthai_cam = None

        if use_depthai:
            cap, depthai_cam = self._open_depthai(source)
            if cap is None:
                # Fall back to OpenCV
                source = {"type": "camera", "value": 0}
                state["source"] = source
                use_depthai = False

        if not use_depthai:
            cap = _open_opencv_source(source)

        cap_source_key = (source["type"], str(source.get("value", "")))

        if not cap.isOpened():
            state["error"] = "Could not open camera"
            holistic.close()
            print(state["error"])
            return

        state["camera_ok"] = True
        print("[Vision] Camera opened successfully.")

        # ----------------------------------------------------------------
        # Main loop
        # ----------------------------------------------------------------
        start_time = time.monotonic()
        fps_start  = time.monotonic()
        frame_counter = 0
        last_gripper: float | None = None
        last_timestamp_ms = -1

        try:
            while state["running"]:
                # Check for source change
                requested = dict(state.get("source", source))
                requested_key = (requested["type"], str(requested.get("value", "")))
                if requested_key != cap_source_key:
                    cap.release()
                    cap = _open_opencv_source(requested)
                    cap_source_key = requested_key
                    state["camera_ok"] = cap.isOpened()
                    state["source_name"] = self._source_name(requested)
                    if not cap.isOpened():
                        state["error"] = f"Could not open {state['source_name']}"
                        time.sleep(0.1)
                        continue
                    state["error"] = None
                    start_time = time.monotonic()
                    source = requested

                ret, frame = cap.read()
                if not ret:
                    if source.get("type") == "video":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        start_time = time.monotonic()
                        continue
                    time.sleep(0.02)
                    continue

                frame = cv2.flip(frame, 1)

                # FPS counter
                frame_counter += 1
                elapsed = time.monotonic() - fps_start
                if elapsed >= 1.0:
                    state["fps"] = frame_counter / elapsed
                    frame_counter = 0
                    fps_start = time.monotonic()

                # Build MediaPipe image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                now_ms = int((time.monotonic() - start_time) * 1000)
                timestamp_ms = max(now_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                # ---- Holistic inference ----
                raw_result = holistic.detect_for_video(mp_image, timestamp_ms)
                results = _HolisticAdapter(raw_result)

                pose_ok = results.pose_landmarks is not None
                hand_ok = results.right_hand_landmarks is not None
                state["pose_ok"] = pose_ok
                state["hand_ok"] = hand_ok

                # ---- Drawing ----
                if pose_ok:
                    _draw_pose(frame, results.pose_landmarks.landmark)
                if hand_ok:
                    _draw_hand(frame, results.right_hand_landmarks.landmark)

                # ---- Angle computation ----
                if pose_ok and results.pose_world_landmarks is not None:
                    hand_world = (
                        results.right_hand_landmarks.landmark
                        if hand_ok else None
                    )

                    human_angles = calculate_arm_angles(
                        results.pose_world_landmarks.landmark,
                        hand_world,
                    )

                    gripper = calculate_gripper(hand_world)
                    if gripper is None:
                        gripper = last_gripper
                    else:
                        last_gripper = gripper

                    gripper_state = (
                        calculate_gripper_state(gripper)
                        if gripper is not None else "---"
                    )

                    state["human_angles"] = human_angles.copy()
                    state["gripper"] = gripper
                    state["gripper_state"] = gripper_state

                    robot_angles = mapper.update(human_angles, gripper)
                    state["angles"] = robot_angles.copy()
                    state["calibrated"] = mapper.calibrated

                    for pub in publishers:
                        try:
                            pub.publish(robot_angles)
                        except Exception as exc:
                            print(f"[Vision] Publisher error: {exc}")

                # ---- Overlay ----
                state["calibrated"] = mapper.calibrated
                _put_overlay(frame, state)
                state["frame"] = frame.copy()

        except Exception as exc:
            state["error"] = f"Vision error: {exc}"
            print(state["error"])
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            holistic.close()
            print("[Vision] Thread stopped.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _source_name(source: dict) -> str:
        if source["type"] == "video":
            return f"Video: {source['value']}"
        if source["type"] == "depthai":
            return "DepthAI OAK-D"
        return f"Camera {source.get('value', 0)}"

    @staticmethod
    def _open_depthai(source: dict):
        """Try to open a DepthAI camera. Returns (cap_or_compatible, cam_obj)."""
        try:
            import depthai_cam as dc
            import cv2
            cam = dc.DepthAICam(
                width=source.get("width", 1920),
                height=source.get("height", 1080),
            )
            if cam.start():
                # Wrap DepthAICam in an OpenCV-compatible adapter
                return _DepthAICapAdapter(cam), cam
            return None, None
        except Exception as exc:
            print(f"[Vision] DepthAI open failed: {exc}")
            return None, None


class _DepthAICapAdapter:
    """Minimal cv2.VideoCapture-compatible wrapper for DepthAICam."""

    def __init__(self, cam):
        self._cam = cam

    def isOpened(self) -> bool:
        return self._cam.is_opened()

    def read(self):
        return self._cam.read_frame()

    def release(self):
        self._cam.stop()

    def set(self, *args):
        pass  # resolution is fixed at construction time
