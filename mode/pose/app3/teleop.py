#!/usr/bin/env python3
"""MediaPipe 6DOF Robot Teleoperation — app3 PySide6 GUI wrapper.

Shows:
  - Live camera feed with MediaPipe overlays (same as controller.py)
  - Three planar projection debug views (XY, XZ, YZ)
  - Joint angle readout panel (Shoulder Yaw/Pitch, Elbow, Wrist Pitch/Roll, Gripper)
  - Gripper open/close indicator with fill bar
  - Serial output (optional via --enable-serial)

No calibrate-neutral step required.

Usage:
    python teleop.py
    python teleop.py --force-webcam
    python teleop.py --enable-serial --serial-port /dev/ttyUSB0
    python teleop.py --nodebug          # still shows 3 views in Qt panel
    python teleop.py --lpf-value 0.4
"""

from __future__ import annotations

import sys
import os
import time
import threading
import argparse
from copy import deepcopy
from pathlib import Path

# Make sure the app3 directory is on the path so we can import its modules
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
)

# Import only the pure helper functions from controller — NOT the module-level
# camera/arg-parse/loop code.  We do this by importing from the module's
# namespace after adding its directory to sys.path.  Because controller.py
# runs top-level code when imported we monkey-patch sys.argv and redirect the
# camera/mediapipe startup so only the function definitions are usable here.
#
# Strategy: exec only the function/class definitions by parsing the source and
# exec-ing up to (but not including) the `parser = argparse…` line.

def _import_controller_helpers():
    """Extract pure helper functions from controller.py without running the
    camera loop or argparse code at the module level."""
    src_path = _HERE / "controller.py"
    with open(src_path) as fh:
        source = fh.read()

    # Split at the argparse section — everything before it is pure definitions
    split_marker = "\n# Read command line arguments\nparser ="
    if split_marker in source:
        defs_only = source.split(split_marker)[0]
    else:
        # Fallback: just take everything up to the first `parser =` assignment
        defs_only = source[:source.find("\nparser =")]

    ns = {"__file__": str(src_path), "__name__": "__controller_defs__"}
    exec(compile(defs_only, str(src_path), "exec"), ns)
    return ns


_ctrl = _import_controller_helpers()

# Expose helpers with short names
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
# Shared state dict (written by VisionThread, read by Qt refresh timer)
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
        "img_xy":           None,   # XY Plane  (Front View)
        "img_xz":           None,   # XZ Plane  (Top View)
        "img_yz":           None,   # YZ Plane  (Side View)
        "frame":            None,
        "pose_ok":          False,
        "hand_ok":          False,
        "fps":              0.0,
        "running":          True,
        "error":            None,
    }


# ---------------------------------------------------------------------------
# Vision thread — mirrors the controller.py main loop without the cv2.imshow
# calls; instead stores rendered images in the shared state dict.
# ---------------------------------------------------------------------------

class VisionThread:

    def __init__(self, state: dict, args):
        self._state = state
        self._args  = args

    def run(self):
        state = self._state
        args  = self._args
        mp_tasks    = mp.tasks.vision
        mp_holistic = mp_tasks
        mp_drawing  = mp_tasks.drawing_utils

        # ---- Serial ----
        ser = None
        if args.enable_serial:
            try:
                import serial as _serial
                ser = _serial.Serial(
                    port=args.serial_port, baudrate=115200,
                    parity=_serial.PARITY_NONE,
                    stopbits=_serial.STOPBITS_ONE,
                    bytesize=_serial.EIGHTBITS,
                    xonxoff=False, rtscts=False, dsrdtr=False,
                    timeout=1,
                )
            except Exception as exc:
                print(f"[Serial] Failed to open {args.serial_port}: {exc}")

        serial_timestamp = time.time()

        # ---- Camera ----
        cam = depthai_cam.DepthAICam(
            width=args.oakd_capture_width, height=args.oakd_capture_height
        )
        if args.force_webcam or not cam.is_depthai_device_available():
            print("No DepthAI device — falling back to webcam.")
            cam = opencv_cam.OpenCVCam(
                width=args.webcam_capture_width,
                height=args.webcam_capture_height,
            )

        if not cam.start():
            state["error"] = "Failed to open camera."
            return

        # ---- MediaPipe holistic ----
        ensure_holistic_model()
        base_opts = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        holistic_opts = mp_holistic.HolisticLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_holistic.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )

        joint_angles    = np.zeros(23)
        gripper_servo   = float(GRIPPER_OPEN)
        is_valid_frame  = False
        last_ts_ms      = -1
        pitchmode       = "---"

        with mp_holistic.HolisticLandmarker.create_from_options(holistic_opts) as holistic:
            while state["running"]:
                frame_time = cv2.getTickCount()

                success, image = cam.read_frame()
                if not success:
                    continue

                # ---- Serial (pre-inference) ----
                if is_valid_frame and ser:
                    period = 1.0 / args.serial_fps
                    if time.time() - serial_timestamp >= period:
                        _transmit_serial(ser, joint_angles)
                        serial_timestamp = time.time()

                # ---- MediaPipe inference ----
                image.flags.writeable = False
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now_ms = int(time.monotonic() * 1000)
                ts_ms  = max(now_ms, last_ts_ms + 1)
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

                hand_points          = None
                hand_points_norm     = np.zeros((21, 3))
                hand_points_pf       = None
                wrist_rotation       = 0.0
                wrist_pitch_angle    = 90.0
                hcp  = np.zeros(3)
                hncp = np.zeros((3, 3))
                pitchmode            = "---"
                openness             = 1.0
                gripper_is_open      = True

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
                        hand_points_pf = deepcopy(hand_points)
                        hand_points_pf += pose_wrist - hand_points_pf[0]

                        hcp = (hand_points_pf[0] + hand_points_pf[5]
                               + hand_points_pf[17]) / 3.0
                        hup = hand_points_pf[9] - hand_points_pf[0]
                        hup /= np.linalg.norm(hup)
                        hright = hand_points_pf[5] - hand_points_pf[17]
                        hright /= np.linalg.norm(hright)
                        hn = np.cross(hright, hup)
                        hn /= np.linalg.norm(hn)
                        hncp = np.array([hcp + hright*0.2,
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
                    # Serial (post-inference)
                    if ser:
                        period = 1.0 / args.serial_fps
                        if time.time() - serial_timestamp >= period:
                            _transmit_serial(ser, joint_angles)
                            serial_timestamp = time.time()

                # ---- Annotations on camera frame ----
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
# Serial helper
# ---------------------------------------------------------------------------

def _transmit_serial(ser, joint_angles):
    import struct
    ja = np.clip(joint_angles.astype(int), 0, 255)
    chk = 255 - (int(np.sum(ja)) & 0xFF)
    ser.write(b'\xFE\xFE')
    ser.write(struct.pack('23B', *ja))
    ser.write(struct.pack('B', chk))
    ser.write(b'\xFD\xFD')
    ser.flushOutput()


# ---------------------------------------------------------------------------
# Camera annotations — mirrors controller.py's annotation block
# ---------------------------------------------------------------------------

def _draw_camera_overlays(image, results,
                           elbow_angle, sh_yaw, sh_pitch,
                           wrist_rotation, wrist_pitch_angle,
                           gripper_is_open, gripper_servo, openness):
    H, W = image.shape[:2]

    el = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    sh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    wr = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Elbow
    ex = int(W - el.x * W); ey = int(el.y * H)
    cv2.rectangle(image, (ex+5, ey-15), (ex+110, ey+5), (0, 0, 0), -1)
    cv2.putText(image, f"Elb: {elbow_angle:.1f}", (ex+5, ey),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(el.visibility), 1, cv2.LINE_AA)

    # Wrist (roll + pitch)
    wx = int(W - wr.x * W); wy = int(wr.y * H)
    cv2.rectangle(image, (wx+5, wy-30), (wx+180, wy+5), (0, 0, 0), -1)
    cv2.putText(image, f"Roll: {wrist_rotation:.1f}", (wx+5, wy-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(wr.visibility), 1, cv2.LINE_AA)
    cv2.putText(image, f"Pitch: {wrist_pitch_angle:.1f}", (wx+5, wy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(wr.visibility), 1, cv2.LINE_AA)

    # Shoulder
    sx = int(W - sh.x * W); sy = int(sh.y * H)
    cv2.rectangle(image, (sx+5, sy-15), (sx+220, sy+5), (0, 0, 0), -1)
    cv2.putText(image, f"Sh Yaw:{sh_yaw:.1f} Pit:{sh_pitch:.1f}", (sx+5, sy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(sh.visibility), 1, cv2.LINE_AA)

    # Gripper (top-left)
    lbl   = "OPEN" if gripper_is_open else "CLOSED"
    gcol  = (0, 255, 120) if gripper_is_open else (0, 80, 255)
    cv2.rectangle(image, (5, 45), (280, 95), (0, 0, 0), -1)
    cv2.putText(image, f"Gripper: {lbl} ({gripper_servo:.1f})",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gcol, 2, cv2.LINE_AA)
    bar_w = int(openness * 260)
    cv2.rectangle(image, (10, 75), (10+bar_w, 88), gcol, -1)
    cv2.rectangle(image, (10, 75), (270, 88), (128, 128, 128), 1)


# ---------------------------------------------------------------------------
# Debug plane views — same rendering as drawDebugViews() in controller.py
# but returns numpy images instead of calling cv2.imshow
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
                cv2.line(zaxis, (lm[0],lm[1]+yoffset), (last[0],last[1]+yoffset), (255,255,255), 1)
                cv2.line(yaxis, (lm[0],lm[2]+yoffset), (last[0],last[2]+yoffset), (255,255,255), 1)
                cv2.line(xaxis, (lm[2],lm[1]+yoffset), (last[2],last[1]+yoffset), (255,255,255), 1)
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

        for ax, i, j in [(zaxis,0,1), (yaxis,0,2), (xaxis,2,1)]:
            cv2.circle(ax, (cp[i], cp[j]+yoffset), 2, (255,255,0), -1)
            cv2.line(ax, (cp[i],cp[j]+yoffset), (ncp[i],ncp[j]+yoffset), (255,255,0), 2)

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
                cv2.circle(ax, (hcp2[a], hcp2[b]+yoffset), 2, (255,255,0), -1)

            cols = [(0,0,255),(0,255,0),(255,0,0)]
            for ci, pt in enumerate(hncp2):
                cv2.line(zaxis,(hcp2[0],hcp2[1]+yoffset),(pt[0],pt[1]+yoffset),cols[ci],2)
                cv2.line(yaxis,(hcp2[0],hcp2[2]+yoffset),(pt[0],pt[2]+yoffset),cols[ci],2)
                cv2.line(xaxis,(hcp2[2],hcp2[1]+yoffset),(pt[2],pt[1]+yoffset),cols[ci],2)

    cv2.putText(zaxis,"XY (Front)",(4,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)
    cv2.putText(yaxis,"XZ (Top)",  (4,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)
    cv2.putText(xaxis,"YZ (Side)", (4,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

    return zaxis, yaxis, xaxis   # XY, XZ, YZ


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
# Main Qt window
# ---------------------------------------------------------------------------

class TeleopWindow(QWidget):

    def __init__(self, state: dict):
        super().__init__()
        self._state = state
        self.setWindowTitle("MediaPipe 6DOF Robot Teleoperation (app3)")
        self.resize(1200, 780)
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(50)   # 20 Hz GUI refresh

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ---- LEFT: camera + joint readout ----
        left = QVBoxLayout()

        self._cam_label = QLabel("Waiting for camera…")
        self._cam_label.setMinimumSize(640, 420)
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setStyleSheet("background:#111; color:#888;")
        left.addWidget(self._cam_label, 3)

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
        self._joint_labels = {}
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
        root.addLayout(left, 55)

        # ---- RIGHT: 3 planar views ----
        right = QVBoxLayout()
        view_specs = [
            ("XY Plane (Front View)", "_xy_label", "#400000"),
            ("XZ Plane (Top View)",   "_xz_label", "#004000"),
            ("YZ Plane (Side View)",  "_yz_label", "#000040"),
        ]
        for title, attr, bg in view_specs:
            right.addWidget(QLabel(title))
            lbl = QLabel()
            lbl.setMinimumSize(256, 256)
            lbl.setStyleSheet(f"background:{bg};")
            right.addWidget(lbl)
            setattr(self, attr, lbl)
        root.addLayout(right, 45)

    def _refresh(self):
        state = self._state

        frame = state.get("frame")
        if frame is not None:
            self._cam_label.setPixmap(
                _bgr_to_pixmap(frame,
                               self._cam_label.width(),
                               self._cam_label.height()))

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
            return

        pose  = state.get("pose_ok", False)
        hand  = state.get("hand_ok", False)
        fps   = state.get("fps", 0.0)
        parts = (["POSE ✓"] if pose else []) + (["HAND ✓"] if hand else [])
        parts.append(f"{fps:.1f} fps")
        self._status_lbl.setText("  |  ".join(parts))


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
    p.add_argument("--enable-serial",         action="store_true")
    p.add_argument("--serial-port",           type=str, default="COM15")
    p.add_argument("--serial-fps",            type=int, default=20)
    p.add_argument("--lpf-value",             type=float, default=0.25)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    state  = _make_state()
    vision = VisionThread(state, args)

    thread = threading.Thread(target=vision.run, daemon=True, name="VisionThread")
    thread.start()

    app    = QApplication(sys.argv)
    window = TeleopWindow(state)
    window.show()

    try:
        return app.exec()
    finally:
        state["running"] = False
        thread.join(timeout=3.0)


if __name__ == "__main__":
    sys.exit(main())
