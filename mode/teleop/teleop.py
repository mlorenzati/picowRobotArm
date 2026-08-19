#!/usr/bin/env python3
"""MediaPipe 6DOF Robot Teleoperation — app3 PySide6 GUI wrapper.

Shows:
  - Live camera feed with MediaPipe overlays
  - Three planar projection debug views (XY, XZ, YZ)
  - Joint angle readout panel (Shoulder Yaw/Pitch, Elbow, Wrist Pitch/Roll, Gripper)
  - Gripper open/close indicator with fill bar
  - Serial bridge panel: connect to robot arm via the same ASCII protocol used by
    mode/serial — space-separated 6 servo values or 'D' to disable, newline-terminated.

Serial protocol (firmware expects):
    "<base> <shoulder> <elbow> <wrist_pitch> <wrist_roll> <gripper>\\n"
    values are 0-180 integer degrees, or "D" to disable that servo.
    Baud rate: 9600.

Servo channel mapping (joint_angles array → firmware channels):
    ch0 Base        = joint_angles[20]  (shoulder yaw  → base rotation)
    ch1 Shoulder    = joint_angles[19]  (shoulder pitch)
    ch2 Elbow       = joint_angles[22]  (elbow angle)
    ch3 Wrist Pitch = joint_angles[16]  (wrist pitch)
    ch4 Wrist Roll  = joint_angles[18]  (wrist roll)
    ch5 Gripper     = joint_angles[17]  (gripper)

Usage:
    python teleop.py
    python teleop.py --force-webcam
    python teleop.py --lpf-value 0.4
"""

from __future__ import annotations

import sys
import json
import time
import threading
import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cv2
import numpy as np
import mediapipe as mp

from PySide6.QtCore import Qt, QTimer, QSettings, QByteArray
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QGroupBox,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QScrollArea,
    QFrame,
    QDockWidget,
    QSizePolicy,
)

# ---------------------------------------------------------------------------
# Load pure helper functions from controller.py (without running its camera loop)
# ---------------------------------------------------------------------------

def _import_controller_helpers():
    src_path = _HERE / "controller.py"
    with open(src_path) as fh:
        source = fh.read()
    split_marker = "\n# Read command line arguments\nparser ="
    defs_only = source.split(split_marker)[0] if split_marker in source else source[:source.find("\nparser =")]
    ns = {"__file__": str(src_path), "__name__": "__controller_defs__"}
    exec(compile(defs_only, str(src_path), "exec"), ns)
    return ns


_ctrl = _import_controller_helpers()

_HolisticResultAdapter  = _ctrl["_HolisticResultAdapter"]
ensure_holistic_model   = _ctrl["ensure_holistic_model"]
MODEL_PATH              = _ctrl["MODEL_PATH"]
mp_pose                 = _ctrl["mp_pose"]
mp_hand                 = _ctrl["mp_hand"]
POSE_CONNECTIONS        = _ctrl["POSE_CONNECTIONS"]
HAND_CONNECTIONS        = _ctrl["HAND_CONNECTIONS"]
visibilityToColour      = _ctrl["visibilityToColour"]
angle                   = _ctrl["angle"]
landmark_to_np          = _ctrl["landmark_to_np"]
calculate_y_up_matrix   = _ctrl["calculate_y_up_matrix"]
calculate_pose_angles   = _ctrl["calculate_pose_angles"]
calculate_finger_angles = _ctrl["calculate_finger_angles"]
calculate_gripper       = _ctrl["calculate_gripper"]
calculate_wrist_pitch   = _ctrl["calculate_wrist_pitch"]
GRIPPER_OPEN            = _ctrl["GRIPPER_OPEN"]
GRIPPER_CLOSED          = _ctrl["GRIPPER_CLOSED"]

import opencv_cam
import depthai_cam


# ---------------------------------------------------------------------------
# Servo calibration
# ---------------------------------------------------------------------------

_CAL_FILE = _HERE / "servo_cal.json"

_CAL_DEFAULTS = [
    # Formula: servo_out = clamp(offset + scale * raw, min, max)
    # Raw angle ranges:
    #   Base, Shoulder, Elbow: computed as 0-180° joint angles
    #   Wrist Pitch: 0-180°, neutral (wrist straight) = 90°
    #   Wrist Roll: 0-180° (pre-normalised from 0-360°), palm-camera=90°
    #   Gripper: GRIPPER_OPEN (100) to GRIPPER_CLOSED (180)
    {"name": "Base",        "offset":  92,  "scale":  1.0,  "min":   0, "max": 180},
    {"name": "Shoulder",    "offset": 180,  "scale": -1.0,  "min":   0, "max": 180},
    {"name": "Elbow",       "offset": 180,  "scale": -1.0,  "min":   0, "max": 180},
    # Wrist Pitch: offset=180/scale=-1 → neutral raw=90 → servo=90°
    {"name": "Wrist Pitch", "offset": 180,  "scale": -1.0,  "min":   0, "max": 180},
    # Wrist Roll: raw is 0-180° (normalised). offset=-120/scale=2.2 maps palm-camera≈78°
    {"name": "Wrist Roll",  "offset": -120, "scale":  2.2,  "min":   0, "max": 180},
    {"name": "Gripper",     "offset":   0,  "scale":  1.0,  "min": 100, "max": 180},
]


@dataclass
class ServoCal:
    name:   str
    offset: float
    scale:  float
    min:    int
    max:    int

    def apply(self, raw: float) -> int:
        """Apply calibration: servo_out = clamp(offset + scale * raw, min, max)."""
        return int(np.clip(round(self.offset + self.scale * raw), self.min, self.max))


def load_servo_cal() -> list[ServoCal]:
    """Load calibration from servo_cal.json; fall back to defaults if missing/corrupt."""
    try:
        with open(_CAL_FILE) as fh:
            data = json.load(fh)
        servos = data.get("servos", [])
        result = []
        for i, d in enumerate(servos[:6]):
            deflt = _CAL_DEFAULTS[i]
            result.append(ServoCal(
                name   = d.get("name",   deflt["name"]),
                offset = float(d.get("offset", deflt["offset"])),
                scale  = float(d.get("scale",  deflt["scale"])),
                min    = int(d.get("min",    deflt["min"])),
                max    = int(d.get("max",    deflt["max"])),
            ))
        # Pad if fewer than 6 entries
        for i in range(len(result), 6):
            d = _CAL_DEFAULTS[i]
            result.append(ServoCal(**d))
        return result
    except Exception:
        return [ServoCal(**d) for d in _CAL_DEFAULTS]


def save_servo_cal(cals: list[ServoCal]) -> None:
    """Save calibration to servo_cal.json."""
    data = {
        "_comment": "Calibration for SerialBridge. Formula: servo_out = clamp(offset + scale * raw_angle, min, max)",
        "servos": [
            {"name": c.name, "offset": c.offset, "scale": c.scale,
             "min": c.min, "max": c.max}
            for c in cals
        ],
    }
    with open(_CAL_FILE, "w") as fh:
        json.dump(data, fh, indent=4)


# ---------------------------------------------------------------------------
# Serial bridge
# ---------------------------------------------------------------------------

# Mapping: firmware channel index (position in ASCII packet) → joint_angles[] slot
#
#  ch | Servo       | joint_angles[] slot | Computed by
#  ---+-------------+---------------------+----------------------------
#   0 | Base        | [20]  shoulder yaw  | calculate_pose_angles()
#   1 | Shoulder    | [19]  shoulder pitch| calculate_pose_angles()
#   2 | Elbow       | [22]  elbow angle   | calculate_pose_angles()
#   3 | Wrist Pitch | [16]  wrist pitch   | calculate_wrist_pitch()
#   4 | Wrist Roll  | [18]  wrist roll    | (roll from hand normal)
#   5 | Gripper     | [17]  gripper servo | calculate_gripper()
#
# Calibration (servo_cal.json) applies per-channel AFTER reading the slot:
#   servo_out = clamp(offset + scale * joint_angles[slot], min, max)
_SERVO_CHANNEL_MAP = [20, 19, 22, 16, 18, 17]

# Which channels start disabled (can be toggled in the UI)
_DEFAULT_DISABLED = [False, False, False, False, False, False]


class SerialBridge:
    """Manages the serial connection to the robot arm firmware.

    Call ``connect(port)`` / ``disconnect()`` to open/close the port.
    Call ``send(joint_angles)`` to transmit the current pose at most once
    per ``1/rate`` seconds.
    """

    def __init__(self):
        self._ser       = None
        self._lock      = threading.Lock()
        self._last_send = 0.0
        self.rate       = 20          # Hz
        self.disabled   = list(_DEFAULT_DISABLED)   # per-channel disable mask
        self.status     = "Disconnected"
        self.connected  = False
        self.cal        = load_servo_cal()   # per-channel calibration

    # ------------------------------------------------------------------
    def connect(self, port: str) -> bool:
        try:
            import serial as _serial
            ser = _serial.Serial(
                port=port,
                baudrate=9600,
                timeout=0,
            )
            with self._lock:
                self._ser      = ser
                self.connected = True
                self.status    = f"Connected to {port}"
            return True
        except Exception as exc:
            self.status    = str(exc)
            self.connected = False
            return False

    def disconnect(self, send_disable: bool = True):
        with self._lock:
            if self._ser is None:
                return
            if send_disable:
                try:
                    pkt = " ".join(["D"] * 6) + "\n"
                    self._ser.write(pkt.encode("ascii"))
                    self._ser.flush()
                except Exception:
                    pass
            self._ser.close()
            self._ser      = None
            self.connected = False
            self.status    = "Disconnected"

    # ------------------------------------------------------------------
    def send(self, joint_angles: np.ndarray):
        """Send a calibrated packet if enough time has elapsed since the last one.

        Calibration formula per channel:
            servo_out = clamp(offset + scale * raw_angle, min, max)
        """
        now = time.time()
        period = 1.0 / max(1, self.rate)
        if now - self._last_send < period:
            return
        self._last_send = now
        with self._lock:
            if self._ser is None:
                return
            tokens = []
            for ch, ja_idx in enumerate(_SERVO_CHANNEL_MAP):
                if self.disabled[ch]:
                    tokens.append("D")
                else:
                    raw = float(joint_angles[ja_idx])
                    val = self.cal[ch].apply(raw)
                    tokens.append(str(val))
            pkt = " ".join(tokens) + "\n"
            try:
                self._ser.write(pkt.encode("ascii"))
            except Exception as exc:
                self.status    = f"Send error: {exc}"
                self.connected = False
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    def send_home(self):
        """Send the home position once regardless of rate timer."""
        home = [90, 180, 180, 100, 90, 170]
        with self._lock:
            if self._ser is None:
                return
            tokens = []
            for ch, val in enumerate(home):
                tokens.append("D" if self.disabled[ch] else str(val))
            pkt = " ".join(tokens) + "\n"
            try:
                self._ser.write(pkt.encode("ascii"))
            except Exception:
                pass

    @staticmethod
    def list_ports() -> list[str]:
        try:
            import serial.tools.list_ports
            return [p.device for p in serial.tools.list_ports.comports()]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Shared state dict
# ---------------------------------------------------------------------------

def _make_state(force_webcam: bool = False):
    return {
        "joint_angles":     np.zeros(23),
        "shoulder_yaw":     0.0,
        "shoulder_pitch":   0.0,
        "elbow_angle":      0.0,
        "wrist_roll":       0.0,
        "wrist_pitch":      90.0,
        "gripper_servo":    float(GRIPPER_OPEN),
        "gripper_is_open":  True,
        "openness":         1.0,
        "pitchmode":        "---",
        "img_xy":           None,
        "img_xz":           None,
        "img_yz":           None,
        "frame":            None,
        "pose_ok":          False,
        "hand_ok":          False,
        "fps":              0.0,
        "running":          True,
        "error":            None,
        # Set by CameraPanel / QSettings, read by VisionThread before cam init
        "force_webcam":     force_webcam,
        "camera_type":      "unknown",   # filled by VisionThread after init
    }


# ---------------------------------------------------------------------------
# Vision thread
# ---------------------------------------------------------------------------

class VisionThread:

    def __init__(self, state: dict, args, bridge: SerialBridge):
        self._state  = state
        self._args   = args
        self._bridge = bridge

    def run(self):
        state  = self._state
        args   = self._args
        bridge = self._bridge
        mp_tasks    = mp.tasks.vision
        mp_holistic = mp_tasks
        mp_drawing  = mp_tasks.drawing_utils

        # ---- Camera ----
        # state["force_webcam"] can be set at runtime by the CameraPanel;
        # args.force_webcam is the command-line fallback.
        force_webcam = state.get("force_webcam", args.force_webcam)
        cam = depthai_cam.DepthAICam(
            width=args.oakd_capture_width, height=args.oakd_capture_height)
        if force_webcam or not cam.is_depthai_device_available():
            print("No DepthAI device — falling back to webcam.")
            cam = opencv_cam.OpenCVCam(
                width=args.webcam_capture_width,
                height=args.webcam_capture_height)
        state["camera_type"] = "webcam" if isinstance(cam, opencv_cam.OpenCVCam) else "depthai"

        if not cam.start():
            state["error"] = "Failed to open camera."
            return

        # ---- MediaPipe ----
        ensure_holistic_model()
        base_opts = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        holistic_opts = mp_holistic.HolisticLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_holistic.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )

        joint_angles   = np.zeros(23)
        gripper_servo  = float(GRIPPER_OPEN)
        is_valid_frame = False
        last_ts_ms     = -1
        pitchmode      = "---"

        with mp_holistic.HolisticLandmarker.create_from_options(holistic_opts) as holistic:
            while state["running"]:
                frame_time = cv2.getTickCount()

                success, image = cam.read_frame()
                if not success:
                    continue

                # ---- MediaPipe inference ----
                image.flags.writeable = False
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now_ms  = int(time.monotonic() * 1000)
                ts_ms   = max(now_ms, last_ts_ms + 1)
                last_ts_ms = ts_ms
                raw     = holistic.detect_for_video(mp_image, ts_ms)
                results = _HolisticResultAdapter(raw)

                # ---- Draw overlays ----
                image.flags.writeable = True
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks.landmark, POSE_CONNECTIONS)
                if results.right_hand_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image, results.right_hand_landmarks.landmark, HAND_CONNECTIONS)
                image = cv2.resize(image, (args.preview_width, args.preview_height))

                # ---- Angle calculations ----
                prev_joint_angles = joint_angles.astype(np.float32)

                hand_points       = None
                hand_points_norm  = np.zeros((21, 3))
                hand_points_pf    = None
                wrist_rotation    = 0.0
                wrist_pitch_angle = 90.0
                hcp  = np.zeros(3)
                hncp = np.zeros((3, 3))
                pitchmode         = "---"
                openness          = 1.0
                gripper_is_open   = True

                if results.right_hand_landmarks is not None:
                    hl = results.right_hand_landmarks
                    hand_points = np.array([
                        [hl.landmark[i].x, hl.landmark[i].y, hl.landmark[i].z]
                        for i in range(21)
                    ])

                    hand_points_norm = deepcopy(hand_points)
                    hand_points_norm -= hand_points_norm[0]

                    nup = (hand_points_norm[mp_hand.HandLandmark.WRIST]
                           - hand_points_norm[mp_hand.HandLandmark.MIDDLE_FINGER_MCP])
                    nup /= np.linalg.norm(nup)
                    Rm = calculate_y_up_matrix(nup)
                    hand_points_norm = np.matmul(hand_points_norm, Rm)

                    index = hand_points_norm[mp_hand.HandLandmark.INDEX_FINGER_MCP]
                    pinky = hand_points_norm[mp_hand.HandLandmark.PINKY_MCP]
                    zpt   = pinky + np.array([0.0, 0.0, 1.0])
                    rel   = index - pinky

                    wrist_rotation = 180.0 - angle(
                        np.array([index[0], index[2]]),
                        np.array([pinky[0], pinky[2]]),
                        np.array([zpt[0],   zpt[2]]),
                    )
                    if rel[0] < 0:
                        wrist_rotation = 360.0 - wrist_rotation

                    joint_angles = calculate_finger_angles(joint_angles, hand_points_norm)

                    gripper_raw, gripper_is_open, openness = calculate_gripper(hand_points_norm)
                    gripper_servo = ((1.0 - args.lpf_value) * gripper_servo
                                     + args.lpf_value * gripper_raw)
                    joint_angles[17] = gripper_servo

                    if results.pose_world_landmarks is not None:
                        pose_wrist = landmark_to_np(
                            results.pose_world_landmarks.landmark[
                                mp_pose.PoseLandmark.RIGHT_WRIST])
                        hand_points_pf  = deepcopy(hand_points)
                        hand_points_pf += pose_wrist - hand_points_pf[0]

                        hcp    = (hand_points_pf[0] + hand_points_pf[5]
                                  + hand_points_pf[17]) / 3.0
                        hup    = hand_points_pf[9] - hand_points_pf[0]
                        hup   /= np.linalg.norm(hup)
                        hright = hand_points_pf[5] - hand_points_pf[17]
                        hright /= np.linalg.norm(hright)
                        hn     = np.cross(hright, hup)
                        hn    /= np.linalg.norm(hn)
                        hncp   = np.array([hcp + hright*0.2,
                                           hcp + hup*0.2,
                                           hcp + hn*0.2])

                        wrist_pitch_angle = calculate_wrist_pitch(
                            hand_points_pf, results.pose_world_landmarks)
                        joint_angles[16] = wrist_pitch_angle
                        # wrist_rotation is 0-360°; normalise to 0-180° so
                        # calibration scale/offset work within the servo range.
                        joint_angles[18] = wrist_rotation * 0.5
                    else:
                        hand_points_pf = hand_points.copy()

                if results.pose_world_landmarks is not None:
                    ea, sy, sp, pitchmode = calculate_pose_angles(
                        results.pose_world_landmarks)
                    joint_angles[19] = sp
                    joint_angles[20] = sy
                    joint_angles[21] = 0.0
                    joint_angles[22] = ea

                is_valid_frame = (results.pose_landmarks is not None
                                  and results.right_hand_landmarks is not None)

                if is_valid_frame:
                    joint_angles = ((1.0 - args.lpf_value) * prev_joint_angles
                                    + args.lpf_value * joint_angles)
                    # Send to robot arm if connected
                    bridge.send(joint_angles)

                # ---- Camera overlays ----
                flipped = cv2.flip(image, 1)
                if is_valid_frame:
                    _draw_camera_overlays(
                        flipped, results,
                        joint_angles[22], joint_angles[20], joint_angles[19],
                        wrist_rotation, wrist_pitch_angle,
                        gripper_is_open, gripper_servo, openness,
                    )

                # ---- Debug plane views ----
                img_xy, img_xz, img_yz = _build_debug_views(
                    results, hand_points_pf, hcp, hncp,
                    hand_points_norm, pitchmode,
                )

                # ---- FPS overlay ----
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - frame_time)
                cv2.rectangle(flipped, (0, 0), (200, 40), (0, 0, 0), -1)
                cv2.putText(flipped, f"FPS: {fps:.1f}", (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                # ---- Push to shared state ----
                state["joint_angles"]    = joint_angles.copy()
                state["shoulder_yaw"]    = float(joint_angles[20])
                state["shoulder_pitch"]  = float(joint_angles[19])
                state["elbow_angle"]     = float(joint_angles[22])
                state["wrist_roll"]      = float(joint_angles[18])
                state["wrist_pitch"]     = float(joint_angles[16])
                state["gripper_servo"]   = float(gripper_servo)
                state["gripper_is_open"] = gripper_is_open
                state["openness"]        = openness
                state["pitchmode"]       = pitchmode
                state["frame"]           = flipped
                state["img_xy"]          = img_xy
                state["img_xz"]          = img_xz
                state["img_yz"]          = img_yz
                state["pose_ok"]         = results.pose_landmarks is not None
                state["hand_ok"]         = results.right_hand_landmarks is not None
                state["fps"]             = fps

        cam.stop()


# ---------------------------------------------------------------------------
# Camera annotations
# ---------------------------------------------------------------------------

def _draw_camera_overlays(image, results,
                           elbow_angle, sh_yaw, sh_pitch,
                           wrist_rotation, wrist_pitch_angle,
                           gripper_is_open, gripper_servo, openness):
    H, W = image.shape[:2]
    el = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    sh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    wr = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    ex = int(W - el.x * W); ey = int(el.y * H)
    cv2.rectangle(image, (ex+5, ey-15), (ex+110, ey+5), (0, 0, 0), -1)
    cv2.putText(image, f"Elb: {elbow_angle:.1f}", (ex+5, ey),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(el.visibility), 1, cv2.LINE_AA)

    wx = int(W - wr.x * W); wy = int(wr.y * H)
    cv2.rectangle(image, (wx+5, wy-30), (wx+180, wy+5), (0, 0, 0), -1)
    cv2.putText(image, f"Roll: {wrist_rotation:.1f}", (wx+5, wy-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(wr.visibility), 1, cv2.LINE_AA)
    cv2.putText(image, f"Pitch: {wrist_pitch_angle:.1f}", (wx+5, wy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(wr.visibility), 1, cv2.LINE_AA)

    sx = int(W - sh.x * W); sy = int(sh.y * H)
    cv2.rectangle(image, (sx+5, sy-15), (sx+220, sy+5), (0, 0, 0), -1)
    cv2.putText(image, f"Sh Yaw:{sh_yaw:.1f} Pit:{sh_pitch:.1f}", (sx+5, sy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(sh.visibility), 1, cv2.LINE_AA)

    lbl  = "OPEN" if gripper_is_open else "CLOSED"
    gcol = (0, 255, 120) if gripper_is_open else (0, 80, 255)
    cv2.rectangle(image, (5, 45), (280, 95), (0, 0, 0), -1)
    cv2.putText(image, f"Gripper: {lbl} ({gripper_servo:.1f})",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gcol, 2, cv2.LINE_AA)
    bar_w = int(openness * 260)
    cv2.rectangle(image, (10, 75), (10+bar_w, 88), gcol, -1)
    cv2.rectangle(image, (10, 75), (270, 88), (128, 128, 128), 1)


# ---------------------------------------------------------------------------
# Debug plane views
# ---------------------------------------------------------------------------

def _build_debug_views(results, hand_points, hcp, hncp, hand_points_norm, pitchmode):
    WIN = 256
    xaxis = np.zeros((WIN, WIN, 3), np.uint8); xaxis[:] = (0, 0, 64)
    yaxis = np.zeros((WIN, WIN, 3), np.uint8); yaxis[:] = (0, 64, 0)
    zaxis = np.zeros((WIN, WIN, 3), np.uint8); zaxis[:] = (64, 0, 0)
    yoffset = int(WIN * 0.25)

    if results.pose_world_landmarks is not None:
        joints = [
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
        ]
        names = ['Wrist', 'Elbow', 'RSho', 'RHip', 'LHip', 'LSho']
        wl = np.array([
            [results.pose_world_landmarks.landmark[i].x,
             results.pose_world_landmarks.landmark[i].y,
             results.pose_world_landmarks.landmark[i].z]
            for i in joints
        ])
        wl = ((wl + 0.5) * WIN).astype(int)

        cp  = ((wl[2].astype(float) + wl[4].astype(float)) / 2.0).astype(int)
        nrm = np.cross(wl[3] - wl[2], wl[4] - wl[2]).astype(float)
        nn  = np.linalg.norm(nrm)
        if nn > 1e-8:
            nrm /= nn
        ncp = (cp.astype(float) + nrm * 20.0).astype(int)

        last = None
        for idx, lm in enumerate(wl):
            cv2.circle(zaxis, (lm[0], lm[1]+yoffset), 2, (255,255,255), -1)
            cv2.circle(yaxis, (lm[0], lm[2]+yoffset), 2, (255,255,255), -1)
            cv2.circle(xaxis, (lm[2], lm[1]+yoffset), 2, (255,255,255), -1)
            cv2.putText(zaxis, names[idx], (lm[0], lm[1]+yoffset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(yaxis, names[idx], (lm[0], lm[2]+yoffset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(xaxis, names[idx], (lm[2], lm[1]+yoffset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            if last is not None:
                cv2.line(zaxis,(lm[0],lm[1]+yoffset),(last[0],last[1]+yoffset),(255,255,255),1)
                cv2.line(yaxis,(lm[0],lm[2]+yoffset),(last[0],last[2]+yoffset),(255,255,255),1)
                cv2.line(xaxis,(lm[2],lm[1]+yoffset),(last[2],last[1]+yoffset),(255,255,255),1)
            last = lm

        if pitchmode == "Top View":
            cv2.line(yaxis,(wl[2][0],wl[2][2]+yoffset),(wl[1][0],wl[1][2]+yoffset),(0,255,255),2)
            cv2.line(yaxis,(wl[2][0],wl[2][2]+yoffset),(wl[5][0],wl[5][2]+yoffset),(0,255,255),2)
        else:
            cv2.line(xaxis,(wl[2][2],wl[2][1]+yoffset),(wl[1][2],wl[1][1]+yoffset),(0,255,255),2)
            cv2.line(xaxis,(wl[2][2],wl[2][1]+yoffset),(wl[3][2],wl[3][1]+yoffset),(0,255,255),2)

        cv2.line(zaxis,(wl[2][0]+2,wl[2][1]+yoffset+2),(wl[1][0]+2,wl[1][1]+yoffset+2),(255,255,0),2)
        cv2.line(zaxis,(wl[1][0],  wl[1][1]+yoffset),  (wl[0][0],  wl[0][1]+yoffset),  (255,255,0),2)
        cv2.line(zaxis,(wl[2][0],  wl[2][1]+yoffset),  (wl[3][0],  wl[3][1]+yoffset),  (255,0,255),2)
        cv2.line(zaxis,(wl[2][0],  wl[2][1]+yoffset),  (wl[1][0],  wl[1][1]+yoffset),  (255,0,255),2)

        for ax, i, j in [(zaxis,0,1),(yaxis,0,2),(xaxis,2,1)]:
            cv2.circle(ax,(cp[i],cp[j]+yoffset),2,(255,255,0),-1)
            cv2.line(ax,(cp[i],cp[j]+yoffset),(ncp[i],ncp[j]+yoffset),(255,255,0),2)

        if hand_points is not None:
            hp      = hand_points.copy()
            hp_norm = hand_points_norm.copy()
            hcp2    = hcp.copy()
            hncp2   = hncp.copy()

            hp      = ((hp + 0.5) * WIN).astype(int)
            hcp2    = ((hcp2 + 0.5) * WIN).astype(int)
            hncp2   = ((hncp2 + 0.5) * WIN).astype(int)
            hp_norm = ((hp_norm * 0.5 + 0.5) * WIN).astype(int)

            for i in range(21):
                cv2.circle(zaxis,(hp[i][0],hp[i][1]+yoffset),2,(255,255,255),-1)
                cv2.circle(yaxis,(hp[i][0],hp[i][2]+yoffset),2,(255,255,255),-1)
                cv2.circle(xaxis,(hp[i][2],hp[i][1]+yoffset),2,(255,255,255),-1)
                cv2.circle(zaxis,(hp_norm[i][0]+100,hp_norm[i][1]+100),2,(0,255,255),-1)
                cv2.circle(yaxis,(hp_norm[i][0]+100,hp_norm[i][2]+100),2,(0,255,255),-1)
                cv2.circle(xaxis,(hp_norm[i][2]+100,hp_norm[i][1]+100),2,(0,255,255),-1)

            for ax, a, b in [(zaxis,0,1),(yaxis,0,2),(xaxis,2,1)]:
                cv2.circle(ax,(hcp2[a],hcp2[b]+yoffset),2,(255,255,0),-1)

            cols = [(0,0,255),(0,255,0),(255,0,0)]
            for ci, pt in enumerate(hncp2):
                cv2.line(zaxis,(hcp2[0],hcp2[1]+yoffset),(pt[0],pt[1]+yoffset),cols[ci],2)
                cv2.line(yaxis,(hcp2[0],hcp2[2]+yoffset),(pt[0],pt[2]+yoffset),cols[ci],2)
                cv2.line(xaxis,(hcp2[2],hcp2[1]+yoffset),(pt[2],pt[1]+yoffset),cols[ci],2)

    cv2.putText(zaxis,"XY (Front)",(4,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)
    cv2.putText(yaxis,"XZ (Top)",  (4,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)
    cv2.putText(xaxis,"YZ (Side)", (4,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

    return zaxis, yaxis, xaxis


# ---------------------------------------------------------------------------
# Qt helpers
# ---------------------------------------------------------------------------

def _bgr_to_pixmap(img: np.ndarray, w: int, h: int) -> QPixmap:
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rh, rw, ch = rgb.shape
    qimg = QImage(rgb.data, rw, rh, ch * rw, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ---------------------------------------------------------------------------
# Serial panel widget
# ---------------------------------------------------------------------------

_SERVO_NAMES = ["Base", "Shoulder", "Elbow", "Wrist Pitch", "Wrist Roll", "Gripper"]


class SerialPanel(QGroupBox):
    """Self-contained panel for connecting to and driving the robot arm."""

    def __init__(self, bridge: SerialBridge, parent=None):
        super().__init__("Robot Arm — Serial Bridge", parent)
        self._bridge = bridge
        self._build()

    def _build(self):
        vl = QVBoxLayout(self)

        # ---- Connection row ----
        conn_row = QHBoxLayout()
        self._port_combo = QComboBox()
        self._port_combo.setMinimumWidth(140)
        self._refresh_ports()

        refresh_btn = QPushButton("↺ Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self._refresh_ports)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setFixedWidth(90)
        self._connect_btn.clicked.connect(self._toggle_connection)

        home_btn = QPushButton("Home")
        home_btn.setFixedWidth(60)
        home_btn.clicked.connect(self._send_home)

        self._rate_spin = QSpinBox()
        self._rate_spin.setRange(1, 50)
        self._rate_spin.setValue(self._bridge.rate)
        self._rate_spin.setSuffix(" Hz")
        self._rate_spin.setFixedWidth(70)
        self._rate_spin.valueChanged.connect(self._rate_changed)

        conn_row.addWidget(QLabel("Port:"))
        conn_row.addWidget(self._port_combo, 1)
        conn_row.addWidget(refresh_btn)
        conn_row.addWidget(self._connect_btn)
        conn_row.addWidget(QLabel("Rate:"))
        conn_row.addWidget(self._rate_spin)
        conn_row.addWidget(home_btn)
        conn_row.addStretch()
        vl.addLayout(conn_row)

        # ---- Per-channel disable checkboxes ----
        ch_row = QHBoxLayout()
        self._disable_checks: list[QCheckBox] = []
        for i, name in enumerate(_SERVO_NAMES):
            cb = QCheckBox(name)
            cb.setChecked(self._bridge.disabled[i])
            idx = i  # capture
            cb.stateChanged.connect(lambda state, ch=idx: self._disable_changed(ch, state))
            self._disable_checks.append(cb)
            ch_row.addWidget(cb)
        ch_row.addStretch()
        vl.addLayout(ch_row)

        # ---- Status ----
        self._status_lbl = QLabel("Disconnected")
        self._status_lbl.setStyleSheet("color: #888;")
        vl.addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    def _refresh_ports(self):
        current = self._port_combo.currentText()
        self._port_combo.clear()
        for p in SerialBridge.list_ports():
            self._port_combo.addItem(p)
        idx = self._port_combo.findText(current)
        if idx >= 0:
            self._port_combo.setCurrentIndex(idx)

    def _toggle_connection(self):
        if self._bridge.connected:
            self._bridge.disconnect(send_disable=True)
            self._connect_btn.setText("Connect")
        else:
            port = self._port_combo.currentText()
            if not port:
                self._status_lbl.setText("No port selected.")
                return
            ok = self._bridge.connect(port)
            self._connect_btn.setText("Disconnect" if ok else "Connect")

    def _send_home(self):
        self._bridge.send_home()

    def _rate_changed(self, val: int):
        self._bridge.rate = val

    def _disable_changed(self, ch: int, state: int):
        self._bridge.disabled[ch] = (state == Qt.Checked.value if hasattr(Qt.Checked, 'value') else bool(state))

    def refresh_status(self):
        """Called by the main window's refresh timer."""
        txt = self._bridge.status
        connected = self._bridge.connected
        self._status_lbl.setText(txt)
        self._status_lbl.setStyleSheet(
            "color: #00cc44; font-weight: bold;" if connected else "color: #cc4444;")
        # Keep button label in sync (e.g. if connection dropped)
        self._connect_btn.setText("Disconnect" if connected else "Connect")


# ---------------------------------------------------------------------------
# Calibration panel widget (collapsible)
# ---------------------------------------------------------------------------

class CalibrationPanel(QGroupBox):
    """Collapsible group box for per-channel servo calibration.

    Allows live editing of offset, scale, min and max per channel.
    Changes apply immediately. Save persists to servo_cal.json.
    """

    def __init__(self, bridge: SerialBridge, parent=None):
        super().__init__("Servo Calibration  ▼", parent)
        self._bridge   = bridge
        self._expanded = True
        self._offset_spins: list[QDoubleSpinBox] = []
        self._scale_spins:  list[QDoubleSpinBox] = []
        self._min_spins:    list[QSpinBox]        = []
        self._max_spins:    list[QSpinBox]        = []
        self._save_status  = None
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # Toggle button row
        toggle_row = QHBoxLayout()
        toggle_btn = QPushButton("▼ Collapse")
        toggle_btn.setFixedWidth(100)
        toggle_btn.clicked.connect(self._toggle_collapse)
        self._toggle_btn = toggle_btn
        toggle_row.addWidget(toggle_btn)
        toggle_row.addStretch()
        outer.addLayout(toggle_row)

        # Content widget (hidden when collapsed)
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Header row
        hdr = QHBoxLayout()
        for txt, w in [("Servo", 90), ("Offset", 80), ("Scale", 80),
                       ("Min°", 60), ("Max°", 60), ("Preview →", 70)]:
            lbl = QLabel(txt)
            lbl.setFixedWidth(w)
            lbl.setStyleSheet("font-weight: bold;")
            hdr.addWidget(lbl)
        hdr.addStretch()
        content_layout.addLayout(hdr)

        # One row per servo channel
        # ja_slot: the joint_angles[] index this channel reads from
        _ja_slots = _SERVO_CHANNEL_MAP  # [20, 19, 22, 16, 18, 17]

        self._preview_labels: list[QLabel] = []
        self._raw_labels:     list[QLabel] = []
        for ch, c in enumerate(self._bridge.cal):
            row = QHBoxLayout()

            name_lbl = QLabel(c.name)
            name_lbl.setFixedWidth(90)
            row.addWidget(name_lbl)

            # Offset: -360 to 360, step 1
            offset_spin = QDoubleSpinBox()
            offset_spin.setRange(-360.0, 360.0)
            offset_spin.setSingleStep(1.0)
            offset_spin.setDecimals(1)
            offset_spin.blockSignals(True)
            offset_spin.setValue(c.offset)
            offset_spin.blockSignals(False)
            offset_spin.setFixedWidth(80)
            offset_spin.valueChanged.connect(
                lambda v, i=ch: self._on_offset_changed(i, v))
            self._offset_spins.append(offset_spin)
            row.addWidget(offset_spin)

            # Scale: -5.0 to 5.0, step 0.05
            scale_spin = QDoubleSpinBox()
            scale_spin.setRange(-5.0, 5.0)
            scale_spin.setSingleStep(0.05)
            scale_spin.setDecimals(2)
            scale_spin.blockSignals(True)
            scale_spin.setValue(c.scale)
            scale_spin.blockSignals(False)
            scale_spin.setFixedWidth(80)
            scale_spin.valueChanged.connect(
                lambda v, i=ch: self._on_scale_changed(i, v))
            self._scale_spins.append(scale_spin)
            row.addWidget(scale_spin)

            # Min
            min_spin = QSpinBox()
            min_spin.setRange(0, 180)
            min_spin.blockSignals(True)
            min_spin.setValue(c.min)
            min_spin.blockSignals(False)
            min_spin.setFixedWidth(60)
            min_spin.valueChanged.connect(
                lambda v, i=ch: self._on_min_changed(i, v))
            self._min_spins.append(min_spin)
            row.addWidget(min_spin)

            # Max
            max_spin = QSpinBox()
            max_spin.setRange(0, 180)
            max_spin.blockSignals(True)
            max_spin.setValue(c.max)
            max_spin.blockSignals(False)
            max_spin.setFixedWidth(60)
            max_spin.valueChanged.connect(
                lambda v, i=ch: self._on_max_changed(i, v))
            self._max_spins.append(max_spin)
            row.addWidget(max_spin)

            # Raw input label (live value from joint_angles[slot])
            raw_lbl = QLabel("raw:---")
            raw_lbl.setFixedWidth(65)
            raw_lbl.setStyleSheet("color: #888; font-size: 10px;")
            self._raw_labels.append(raw_lbl)
            row.addWidget(raw_lbl)

            # Preview label (shows calibrated result from live raw)
            preview = QLabel("→ ---")
            preview.setFixedWidth(60)
            preview.setStyleSheet("color: #aaa;")
            self._preview_labels.append(preview)
            row.addWidget(preview)

            row.addStretch()
            content_layout.addLayout(row)

        # Buttons row
        btn_row = QHBoxLayout()
        save_btn = QPushButton("💾 Save")
        save_btn.setFixedWidth(80)
        save_btn.clicked.connect(self._save)

        reset_btn = QPushButton("↺ Reset")
        reset_btn.setFixedWidth(80)
        reset_btn.clicked.connect(self._reset)

        self._save_status = QLabel("")
        self._save_status.setStyleSheet("color: #aaa;")

        btn_row.addWidget(save_btn)
        btn_row.addWidget(reset_btn)
        btn_row.addWidget(self._save_status)
        btn_row.addStretch()
        content_layout.addLayout(btn_row)

        outer.addWidget(self._content)
        self._update_previews()

    # ------------------------------------------------------------------
    def _toggle_collapse(self):
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._toggle_btn.setText("▼ Collapse" if self._expanded else "▶ Expand")

    def _on_offset_changed(self, ch: int, val: float):
        self._bridge.cal[ch].offset = val
        self._update_previews()

    def _on_scale_changed(self, ch: int, val: float):
        self._bridge.cal[ch].scale = val
        self._update_previews()

    def _on_min_changed(self, ch: int, val: int):
        self._bridge.cal[ch].min = val
        self._update_previews()

    def _on_max_changed(self, ch: int, val: int):
        self._bridge.cal[ch].max = val
        self._update_previews()

    def update_live(self, joint_angles: np.ndarray):
        """Called by TeleopWindow._refresh() with the current joint_angles array.

        Updates raw angle and calibrated preview columns for every servo channel.
        """
        for ch, slot in enumerate(_SERVO_CHANNEL_MAP):
            raw = float(joint_angles[slot])
            self._raw_labels[ch].setText(f"raw:{raw:.1f}°")
            val = self._bridge.cal[ch].apply(raw)
            self._preview_labels[ch].setText(f"→ {val}°")

    def _update_previews(self, raw: float = 90.0):
        """Refresh preview labels using a fixed raw angle (default 90°).

        Used when calibration values change but no live data is available.
        """
        for ch, lbl in enumerate(self._preview_labels):
            val = self._bridge.cal[ch].apply(raw)
            lbl.setText(f"→ {val}°")

    def _save(self):
        try:
            save_servo_cal(self._bridge.cal)
            self._save_status.setText("Saved ✓")
            self._save_status.setStyleSheet("color: #00cc44;")
        except Exception as exc:
            self._save_status.setText(f"Error: {exc}")
            self._save_status.setStyleSheet("color: #cc4444;")

    def _reset(self):
        """Reload calibration from file and update spinboxes."""
        self._bridge.cal = load_servo_cal()
        for ch, c in enumerate(self._bridge.cal):
            self._offset_spins[ch].blockSignals(True)
            self._scale_spins[ch].blockSignals(True)
            self._min_spins[ch].blockSignals(True)
            self._max_spins[ch].blockSignals(True)
            self._offset_spins[ch].setValue(c.offset)
            self._scale_spins[ch].setValue(c.scale)
            self._min_spins[ch].setValue(c.min)
            self._max_spins[ch].setValue(c.max)
            self._offset_spins[ch].blockSignals(False)
            self._scale_spins[ch].blockSignals(False)
            self._min_spins[ch].blockSignals(False)
            self._max_spins[ch].blockSignals(False)
        self._update_previews()
        self._save_status.setText("Reset ✓")
        self._save_status.setStyleSheet("color: #aaa;")


# ---------------------------------------------------------------------------
# Camera selection panel
# ---------------------------------------------------------------------------

class CameraPanel(QWidget):
    """Panel for selecting the camera source (DepthAI OAK-D or webcam).

    The selected camera type is persisted to QSettings and applied on the
    next vision thread start.  A status label reflects the camera currently
    active (filled by VisionThread via state["camera_type"]).
    """

    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        self._state = state
        self._build()

    def _build(self):
        vl = QVBoxLayout(self)
        vl.setContentsMargins(6, 6, 6, 6)

        # Camera type selector
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Camera source:"))

        self._cam_combo = QComboBox()
        self._cam_combo.addItem("Auto (DepthAI → Webcam fallback)", "auto")
        self._cam_combo.addItem("Force Webcam (OpenCV)", "webcam")
        self._cam_combo.addItem("Force DepthAI OAK-D",  "depthai")
        self._cam_combo.setMinimumWidth(240)
        sel_row.addWidget(self._cam_combo, 1)
        sel_row.addStretch()
        vl.addLayout(sel_row)

        info = QLabel(
            "Note: camera is initialised once at startup.\n"
            "Change the setting and restart to switch camera.")
        info.setStyleSheet("color: #888; font-size: 10px;")
        vl.addWidget(info)

        self._active_lbl = QLabel("Active camera: unknown")
        self._active_lbl.setStyleSheet("color: #aaa;")
        vl.addWidget(self._active_lbl)

        vl.addStretch()

    def load_settings(self, settings: QSettings):
        """Restore the last-saved camera selection and apply it to state."""
        val = settings.value(_KEY_CAMERA, "auto")
        idx = self._cam_combo.findData(val)
        if idx >= 0:
            self._cam_combo.setCurrentIndex(idx)
        self._apply_to_state()
        self._cam_combo.currentIndexChanged.connect(self._on_changed)

    def save_settings(self, settings: QSettings):
        settings.setValue(_KEY_CAMERA, self._cam_combo.currentData())

    def _on_changed(self, _idx: int):
        self._apply_to_state()

    def _apply_to_state(self):
        val = self._cam_combo.currentData()
        self._state["force_webcam"] = (val == "webcam")
        # "depthai" mode: force_webcam=False, and the cam will fail if absent

    def refresh_status(self):
        cam_type = self._state.get("camera_type", "unknown")
        self._active_lbl.setText(f"Active camera: {cam_type}")


# ---------------------------------------------------------------------------
# Helper: make a scrollable dock content widget
# ---------------------------------------------------------------------------

def _make_scroll(widget: QWidget) -> QScrollArea:
    """Wrap a widget in a QScrollArea so it never forces the window to grow."""
    scroll = QScrollArea()
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    return scroll


def _make_dock(title: str, widget: QWidget, object_name: str,
               parent: "TeleopWindow") -> QDockWidget:
    dock = QDockWidget(title, parent)
    dock.setObjectName(object_name)
    dock.setAllowedAreas(Qt.AllDockWidgetAreas)
    dock.setFeatures(
        QDockWidget.DockWidgetMovable
        | QDockWidget.DockWidgetFloatable
        | QDockWidget.DockWidgetClosable
    )
    dock.setWidget(widget)
    return dock


# ---------------------------------------------------------------------------
# Main window  (QMainWindow with dockable panels)
# ---------------------------------------------------------------------------

_SETTINGS_ORG  = "picowRobotArm"
_SETTINGS_APP  = "teleop_app3"
_KEY_GEOMETRY  = "mainWindow/geometry"
_KEY_STATE     = "mainWindow/dockState"
_KEY_CAMERA    = "camera/type"   # "depthai" or "webcam"


class TeleopWindow(QMainWindow):
    """Main application window.

    Each panel lives in a QDockWidget so it can be:
      - dragged to any edge (left / right / top / bottom)
      - floated as an independent window
      - closed and re-opened via the View menu
      - stacked / tabified with other docks

    Layout (geometry + dock positions) is saved to QSettings on close and
    restored on the next launch.
    """

    def __init__(self, state: dict, bridge: SerialBridge):
        super().__init__()
        self._state  = state
        self._bridge = bridge
        self.setWindowTitle("MediaPipe 6DOF Robot Teleoperation (app3)")
        self.setDockOptions(
            QMainWindow.AllowTabbedDocks
            | QMainWindow.AllowNestedDocks
            | QMainWindow.AnimatedDocks
        )
        self._build_ui()
        self._restore_layout()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(50)   # 20 Hz UI refresh

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Central widget: camera feed ----
        self._cam_label = QLabel("Waiting for camera…")
        self._cam_label.setMinimumSize(480, 320)
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setStyleSheet("background:#111; color:#888;")
        self._cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(self._cam_label)

        # ---- Status bar ----
        self._status_lbl = QLabel("Starting…")
        self.statusBar().addWidget(self._status_lbl, 1)

        # ---- Dock: Joint Angles ----
        self._dock_joints  = self._build_joint_dock()
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock_joints)

        # ---- Dock: Plane Views (XY / XZ / YZ) ----
        self._dock_planes  = self._build_planes_dock()
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock_planes)

        # ---- Dock: Serial Bridge ----
        self._serial_panel = SerialPanel(self._bridge)
        dock_serial = _make_dock("Serial Bridge", self._serial_panel,
                                 "dock_serial", self)
        self._dock_serial  = dock_serial
        self.addDockWidget(Qt.BottomDockWidgetArea, dock_serial)

        # ---- Dock: Servo Calibration ----
        self._cal_panel = CalibrationPanel(self._bridge)
        cal_scroll      = _make_scroll(self._cal_panel)
        dock_cal = _make_dock("Servo Calibration", cal_scroll,
                              "dock_cal", self)
        self._dock_cal = dock_cal
        self.addDockWidget(Qt.BottomDockWidgetArea, dock_cal)

        # ---- Dock: Camera Selection ----
        self._cam_panel = CameraPanel(self._state)
        dock_camera = _make_dock("Camera", self._cam_panel, "dock_camera", self)
        self._dock_camera = dock_camera
        self.addDockWidget(Qt.BottomDockWidgetArea, dock_camera)

        # Tabify the three bottom docks by default (saves vertical space)
        self.tabifyDockWidget(dock_serial, dock_cal)
        self.tabifyDockWidget(dock_cal, dock_camera)
        dock_serial.raise_()   # Serial tab active by default

        # ---- View menu (toggle visibility) ----
        view_menu = self.menuBar().addMenu("&View")
        for dock in (self._dock_joints, self._dock_planes,
                     self._dock_serial, self._dock_cal, self._dock_camera):
            view_menu.addAction(dock.toggleViewAction())

        view_menu.addSeparator()
        reset_act = view_menu.addAction("Reset Layout")
        reset_act.triggered.connect(self._reset_layout)

    def _build_joint_dock(self) -> QDockWidget:
        box = QWidget()
        gl  = QGridLayout(box)
        gl.setContentsMargins(6, 6, 6, 6)

        # (label, state_key, ch_for_servo_preview or None)
        # ch is the _SERVO_CHANNEL_MAP index for the corresponding servo channel
        field_defs = [
            ("Shoulder Yaw",   "shoulder_yaw",    0),
            ("Shoulder Pitch", "shoulder_pitch",   1),
            ("Elbow",          "elbow_angle",      2),
            ("Wrist Pitch",    "wrist_pitch",      3),
            ("Wrist Roll",     "wrist_roll",       4),
            ("Gripper",        "gripper_servo",    5),
        ]
        self._joint_labels:       dict[str, QLabel] = {}
        self._servo_out_labels:   dict[str, QLabel] = {}  # ch → servo output label
        for row, (label, key, ch) in enumerate(field_defs):
            gl.addWidget(QLabel(label + ":"), row, 0)
            raw_lbl = QLabel("---")
            raw_lbl.setMinimumWidth(70)
            raw_lbl.setToolTip("Raw angle from pose estimation (input to calibration)")
            gl.addWidget(raw_lbl, row, 1)
            self._joint_labels[key] = raw_lbl

            # Servo output column (calibrated)
            out_lbl = QLabel("→ ---")
            out_lbl.setMinimumWidth(60)
            out_lbl.setStyleSheet("color: #aaffaa; font-size: 10px;")
            out_lbl.setToolTip("Calibrated servo output (sent to robot)")
            gl.addWidget(out_lbl, row, 2)
            self._servo_out_labels[key] = (ch, out_lbl)

        n = len(field_defs)
        gl.addWidget(QLabel("Gripper State:"), n, 0)
        self._gripper_state_lbl = QLabel("---")
        gl.addWidget(self._gripper_state_lbl, n, 1)

        gl.addWidget(QLabel("Pitch Mode:"), n+1, 0)
        self._pitchmode_lbl = QLabel("---")
        gl.addWidget(self._pitchmode_lbl, n+1, 1)

        # Column headers
        hdr_raw = QLabel("raw")
        hdr_raw.setStyleSheet("color: #888; font-size: 9px;")
        hdr_srv = QLabel("servo")
        hdr_srv.setStyleSheet("color: #aaffaa; font-size: 9px;")
        gl.addWidget(hdr_raw, n+2, 1)
        gl.addWidget(hdr_srv, n+2, 2)

        gl.setRowStretch(n+3, 1)
        return _make_dock("Joint Angles", box, "dock_joints", self)

    def _build_planes_dock(self) -> QDockWidget:
        box = QWidget()
        vl  = QVBoxLayout(box)
        vl.setContentsMargins(4, 4, 4, 4)
        vl.setSpacing(4)

        self._xy_label = QLabel(); self._xy_label.setMinimumSize(200, 200)
        self._xz_label = QLabel(); self._xz_label.setMinimumSize(200, 200)
        self._yz_label = QLabel(); self._yz_label.setMinimumSize(200, 200)

        for title, lbl, bg in [
            ("XY — Front View", self._xy_label, "#3a0000"),
            ("XZ — Top View",   self._xz_label, "#003a00"),
            ("YZ — Side View",  self._yz_label, "#00003a"),
        ]:
            vl.addWidget(QLabel(title))
            lbl.setStyleSheet(f"background:{bg};")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            vl.addWidget(lbl, 1)

        scroll = _make_scroll(box)
        return _make_dock("Plane Views", scroll, "dock_planes", self)

    # ------------------------------------------------------------------
    # Layout persistence
    # ------------------------------------------------------------------

    def _restore_layout(self):
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        geom       = s.value(_KEY_GEOMETRY)
        dock_state = s.value(_KEY_STATE)
        if geom:
            self.restoreGeometry(geom)
        else:
            self.resize(1400, 900)
        if dock_state:
            self.restoreState(dock_state)
        # Always load camera setting (applies to next VisionThread start)
        self._cam_panel.load_settings(s)

    def _save_layout(self):
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        s.setValue(_KEY_GEOMETRY, self.saveGeometry())
        s.setValue(_KEY_STATE,    self.saveState())
        self._cam_panel.save_settings(s)

    def _reset_layout(self):
        """Remove saved layout so the next launch uses the default arrangement."""
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        s.remove(_KEY_GEOMETRY)
        s.remove(_KEY_STATE)
        # Re-add docks in default positions
        for dock in (self._dock_joints, self._dock_planes,
                     self._dock_serial, self._dock_cal, self._dock_camera):
            self.removeDockWidget(dock)
        self.addDockWidget(Qt.RightDockWidgetArea,  self._dock_joints)
        self.addDockWidget(Qt.RightDockWidgetArea,  self._dock_planes)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._dock_serial)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._dock_cal)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._dock_camera)
        self.tabifyDockWidget(self._dock_serial, self._dock_cal)
        self.tabifyDockWidget(self._dock_cal, self._dock_camera)
        self._dock_serial.raise_()
        self.resize(1400, 900)

    # ------------------------------------------------------------------
    # Refresh (called by QTimer at 20 Hz)
    # ------------------------------------------------------------------

    def _refresh(self):
        state = self._state

        frame = state.get("frame")
        if frame is not None:
            self._cam_label.setPixmap(
                _bgr_to_pixmap(frame, self._cam_label.width(), self._cam_label.height()))

        for img_key, lbl_attr in [
            ("img_xy", "_xy_label"),
            ("img_xz", "_xz_label"),
            ("img_yz", "_yz_label"),
        ]:
            img = state.get(img_key)
            if img is not None:
                lbl = getattr(self, lbl_attr)
                lbl.setPixmap(_bgr_to_pixmap(img, lbl.width(), lbl.height()))

        ja = state.get("joint_angles")
        for key, lbl in self._joint_labels.items():
            val = state.get(key)
            lbl.setText(f"{val:.1f}°" if isinstance(val, float) else "---")
        # Update servo output (calibrated) column in joint panel
        if ja is not None:
            for key, (ch, out_lbl) in self._servo_out_labels.items():
                raw = float(ja[_SERVO_CHANNEL_MAP[ch]])
                srv = self._bridge.cal[ch].apply(raw)
                out_lbl.setText(f"→ {srv}°")

        is_open = state.get("gripper_is_open", True)
        self._gripper_state_lbl.setText("OPEN" if is_open else "CLOSED")
        self._gripper_state_lbl.setStyleSheet(
            "color:#00ff80; font-weight:bold;" if is_open else "color:#ff5050; font-weight:bold;")
        self._pitchmode_lbl.setText(state.get("pitchmode", "---"))

        err = state.get("error")
        if err:
            self._status_lbl.setText(f"⛔ {err}")
        else:
            pose  = state.get("pose_ok", False)
            hand  = state.get("hand_ok", False)
            fps   = state.get("fps", 0.0)
            parts = (["POSE ✓"] if pose else ["POSE ✗"]) + (["HAND ✓"] if hand else ["HAND ✗"])
            parts.append(f"{fps:.1f} fps")
            self._status_lbl.setText("  |  ".join(parts))

        # Update calibration panel live raw/preview columns (ja already fetched above)
        if ja is not None:
            self._cal_panel.update_live(ja)

        self._serial_panel.refresh_status()
        self._cam_panel.refresh_status()

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._save_layout()
        self._bridge.disconnect(send_disable=True)
        event.accept()


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MediaPipe 6DOF Robot Teleoperation — app3 GUI")
    p.add_argument("--nodebug", action="store_true",
                   help="(unused in GUI mode; views always shown)")
    p.add_argument("--force-webcam",          action="store_true")
    p.add_argument("--oakd-capture-width",    type=int, default=3840)
    p.add_argument("--oakd-capture-height",   type=int, default=2160)
    p.add_argument("--webcam-capture-width",  type=int, default=1920)
    p.add_argument("--webcam-capture-height", type=int, default=1080)
    p.add_argument("--preview-width",         type=int, default=1280)
    p.add_argument("--preview-height",        type=int, default=720)
    p.add_argument("--lpf-value",             type=float, default=0.25,
                   help="Low-pass filter coefficient (0-1, 1=no filtering)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    state  = _make_state()
    bridge = SerialBridge()
    vision = VisionThread(state, args, bridge)

    thread = threading.Thread(target=vision.run, daemon=True, name="VisionThread")
    thread.start()

    app    = QApplication(sys.argv)
    window = TeleopWindow(state, bridge)
    window.show()

    try:
        return app.exec()
    finally:
        state["running"] = False
        bridge.disconnect(send_disable=True)
        thread.join(timeout=3.0)


if __name__ == "__main__":
    sys.exit(main())
