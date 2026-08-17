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

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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
    {"name": "Base",        "offset":  92, "scale":  1.0, "min":   0, "max": 180},
    {"name": "Shoulder",    "offset": 180, "scale": -1.0, "min":   0, "max": 180},
    {"name": "Elbow",       "offset": 180, "scale": -1.0, "min":   0, "max": 180},
    {"name": "Wrist Pitch", "offset": 101, "scale":  1.0, "min":   0, "max": 180},
    {"name": "Wrist Roll",  "offset":  75, "scale":  1.0, "min":   0, "max": 180},
    {"name": "Gripper",     "offset":   0, "scale":  1.0, "min": 100, "max": 180},
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

# Mapping from firmware channel index → joint_angles slot
# ch0 Base, ch1 Shoulder, ch2 Elbow, ch3 WristPitch, ch4 WristRoll, ch5 Gripper
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

def _make_state():
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
        cam = depthai_cam.DepthAICam(
            width=args.oakd_capture_width, height=args.oakd_capture_height)
        if args.force_webcam or not cam.is_depthai_device_available():
            print("No DepthAI device — falling back to webcam.")
            cam = opencv_cam.OpenCVCam(
                width=args.webcam_capture_width,
                height=args.webcam_capture_height)

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
                        joint_angles[18] = wrist_rotation
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
        self._preview_labels: list[QLabel] = []
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
            offset_spin.setValue(c.offset)
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
            scale_spin.setValue(c.scale)
            scale_spin.setFixedWidth(80)
            scale_spin.valueChanged.connect(
                lambda v, i=ch: self._on_scale_changed(i, v))
            self._scale_spins.append(scale_spin)
            row.addWidget(scale_spin)

            # Min
            min_spin = QSpinBox()
            min_spin.setRange(0, 180)
            min_spin.setValue(c.min)
            min_spin.setFixedWidth(60)
            min_spin.valueChanged.connect(
                lambda v, i=ch: self._on_min_changed(i, v))
            self._min_spins.append(min_spin)
            row.addWidget(min_spin)

            # Max
            max_spin = QSpinBox()
            max_spin.setRange(0, 180)
            max_spin.setValue(c.max)
            max_spin.setFixedWidth(60)
            max_spin.valueChanged.connect(
                lambda v, i=ch: self._on_max_changed(i, v))
            self._max_spins.append(max_spin)
            row.addWidget(max_spin)

            # Preview label (shows calibrated value of 0° raw)
            preview = QLabel("→ ---")
            preview.setFixedWidth(70)
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

    def _update_previews(self):
        """Show what servo value a raw angle of 90° maps to for each channel."""
        for ch, lbl in enumerate(self._preview_labels):
            val = self._bridge.cal[ch].apply(90.0)
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
# Main window
# ---------------------------------------------------------------------------

class TeleopWindow(QWidget):

    def __init__(self, state: dict, bridge: SerialBridge):
        super().__init__()
        self._state  = state
        self._bridge = bridge
        self.setWindowTitle("MediaPipe 6DOF Robot Teleoperation (app3)")
        self.resize(1280, 860)
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(50)   # 20 Hz

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ---- Top: camera + 3 planar views ----
        top = QHBoxLayout()

        # Left: camera
        left = QVBoxLayout()
        self._cam_label = QLabel("Waiting for camera…")
        self._cam_label.setMinimumSize(640, 420)
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setStyleSheet("background:#111; color:#888;")
        left.addWidget(self._cam_label, 3)

        # Joint readout
        joint_box = QGroupBox("Joint Angles")
        jl = QGridLayout()
        field_defs = [
            ("Shoulder Yaw",   "shoulder_yaw"),
            ("Shoulder Pitch", "shoulder_pitch"),
            ("Elbow",          "elbow_angle"),
            ("Wrist Pitch",    "wrist_pitch"),
            ("Wrist Roll",     "wrist_roll"),
            ("Gripper",        "gripper_servo"),
        ]
        self._joint_labels: dict[str, QLabel] = {}
        for row, (label, key) in enumerate(field_defs):
            jl.addWidget(QLabel(label + ":"), row, 0)
            lbl = QLabel("---")
            lbl.setMinimumWidth(80)
            jl.addWidget(lbl, row, 1)
            self._joint_labels[key] = lbl

        n = len(field_defs)
        jl.addWidget(QLabel("Gripper State:"), n, 0)
        self._gripper_state_lbl = QLabel("---")
        jl.addWidget(self._gripper_state_lbl, n, 1)
        jl.addWidget(QLabel("Pitch Mode:"), n+1, 0)
        self._pitchmode_lbl = QLabel("---")
        jl.addWidget(self._pitchmode_lbl, n+1, 1)
        joint_box.setLayout(jl)
        left.addWidget(joint_box, 1)

        self._status_lbl = QLabel("Starting…")
        left.addWidget(self._status_lbl)
        top.addLayout(left, 55)

        # Right: 3 planar views
        right = QVBoxLayout()
        for title, attr, bg in [
            ("XY Plane (Front View)", "_xy_label", "#400000"),
            ("XZ Plane (Top View)",   "_xz_label", "#004000"),
            ("YZ Plane (Side View)",  "_yz_label", "#000040"),
        ]:
            right.addWidget(QLabel(title))
            lbl = QLabel()
            lbl.setMinimumSize(256, 256)
            lbl.setStyleSheet(f"background:{bg};")
            right.addWidget(lbl)
            setattr(self, attr, lbl)
        top.addLayout(right, 45)

        root.addLayout(top, 10)

        # ---- Bottom: serial panel + calibration panel ----
        self._serial_panel = SerialPanel(self._bridge, self)
        root.addWidget(self._serial_panel, 0)

        self._cal_panel = CalibrationPanel(self._bridge, self)
        root.addWidget(self._cal_panel, 0)

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

        for key, lbl in self._joint_labels.items():
            val = state.get(key)
            lbl.setText(f"{val:.1f}°" if isinstance(val, float) else "---")

        is_open = state.get("gripper_is_open", True)
        self._gripper_state_lbl.setText("OPEN" if is_open else "CLOSED")
        self._gripper_state_lbl.setStyleSheet(
            "color:#00ff80;" if is_open else "color:#ff5050;")
        self._pitchmode_lbl.setText(state.get("pitchmode", "---"))

        err = state.get("error")
        if err:
            self._status_lbl.setText(f"⛔ {err}")
        else:
            pose  = state.get("pose_ok", False)
            hand  = state.get("hand_ok", False)
            fps   = state.get("fps", 0.0)
            parts = (["POSE ✓"] if pose else []) + (["HAND ✓"] if hand else [])
            parts.append(f"{fps:.1f} fps")
            self._status_lbl.setText("  |  ".join(parts))

        self._serial_panel.refresh_status()

    def closeEvent(self, event):
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
