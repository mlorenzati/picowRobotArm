# Teleop — MediaPipe 6-DOF Robot Teleoperation

PySide6 GUI that drives a 6-DOF serial robot arm by tracking the user's
right arm and right hand with a webcam or DepthAI OAK-D and
**MediaPipe Holistic**.

---

## Architecture

```
mode/teleop/
├── teleop.py        ← PySide6 GUI entry-point + VisionThread
│     Panels (all dockable / re-arrangeable):
│       • Camera feed (central)
│       • Joint Angles (raw + calibrated servo output per channel)
│       • Plane Views  (XY front / XZ top / YZ side debug projections)
│       • Serial Bridge (connect, rate, per-channel disable)
│       • Servo Calibration (offset / scale / min / max live editor)
│       • Camera Selection (webcam vs DepthAI, persisted to QSettings)
│
├── controller.py    ← All MediaPipe pose/hand maths (no GUI, no I/O)
│       calculate_pose_angles()   → shoulder yaw/pitch, elbow
│       calculate_wrist_pitch()   → wrist pitch (forearm↔hand angle)
│       calculate_finger_angles() → 15 finger joint angles
│       calculate_gripper()       → gripper openness → servo degrees
│
├── opencv_cam.py    ← OpenCV webcam wrapper
├── depthai_cam.py   ← DepthAI OAK-D camera wrapper
│
├── servo_cal.json   ← Persisted servo calibration
│                        (offset, scale, min, max per channel)
│
└── holistic_landmarker.task  ← MediaPipe model (auto-downloaded)
```

> `arm_geometry.py`, `robot_mapping.py`, `publishers.py`, and `vision.py`
> are earlier-iteration files kept for reference; they are **not used** by
> `teleop.py`.

---

## Servo channel mapping

The serial packet contains **6 space-separated integer degrees** followed by
`\n`.  Each position maps to a `joint_angles[]` slot computed by
`controller.py`:

| ch | Servo       | joint_angles[] slot | Raw range  | Notes                          |
|----|-------------|---------------------|------------|--------------------------------|
| 0  | Base        | [20] shoulder yaw   | 0–180°     | Left/right rotation            |
| 1  | Shoulder    | [19] shoulder pitch | 0–180°     | Forward/back arm lift          |
| 2  | Elbow       | [22] elbow angle    | 0–180°     | Upper/lower arm bend           |
| 3  | Wrist Pitch | [16] wrist pitch    | 0–180°     | 90° = neutral (wrist straight) |
| 4  | Wrist Roll  | [18] wrist roll     | 0–180°*    | 90° = palm facing camera       |
| 5  | Gripper     | [17] gripper servo  | 100–180°   | 100=open, 180=closed           |

\* Wrist roll is internally normalised from the 0–360° MediaPipe range to
0–180° (`joint_angles[18] = wrist_rotation * 0.5`) so that calibration
`offset` / `scale` values work cleanly within the servo range.

### Calibration formula

```
servo_out = clamp(offset + scale × raw, min, max)
```

Edit values live in the **Servo Calibration** dock and press 💾 **Save** to
persist to `servo_cal.json`.  Press ↺ **Reset** to reload from file.

The **Joint Angles** dock shows two columns for each servo:
- **raw°** — angle from pose estimation (input to calibration)
- **→ servo°** — calibrated output actually sent to the robot (green)

---

## Setup

```bash
cd mode/teleop
pip install -r requirements.txt
```

For DepthAI OAK-D support, uncomment the optional lines at the end of
`requirements.txt` and run `pip install` again.

---

## Usage

```bash
# Auto-detect camera (OAK-D first, webcam fallback)
python teleop.py

# Force webcam (OpenCV)
python teleop.py --force-webcam

# Tune low-pass filter (0 = no filter, 1 = frozen)
python teleop.py --lpf-value 0.15

# Custom capture resolution
python teleop.py --webcam-capture-width 1280 --webcam-capture-height 720
```

The camera source can also be changed in the **Camera** dock at runtime;
the selection is saved to `QSettings` (`picowRobotArm / teleop_app3`) and
applied on the next startup.

### Selecting which webcam to use

The **Camera** dock has two mutually exclusive options:

| Option | Description |
|--------|-------------|
| ☑ **Webcam index: [0-9]** | Use an OpenCV webcam. Spin the index to choose which camera (0 = first USB webcam, 1 = second, …). **This is the default and priority option.** |
| ☑ **DepthAI OAK-D (auto-detect)** | Use a DepthAI OAK-D camera. Falls back to webcam 0 if no OAK-D is found. |

The webcam index and camera type are persisted to `QSettings` across sessions.
Change the setting in the dock, then **restart** the app — the camera is opened
once at startup.

To list available camera indices on your system:
```bash
python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(5)]"
```

---

## Workflow

1. Run `python teleop.py`.
2. Connect the Pico W via USB, select the serial port in the
   **Serial Bridge** dock, and click **Connect**.
3. Stand in front of the camera so your full right arm is visible.
4. Move your right arm — the robot mirrors it in real-time.
5. Tune `offset` / `scale` per servo in the **Servo Calibration** dock
   until the servo positions match your arm positions, then click 💾 Save.
6. The **Camera** dock shows which camera is active.

---

## Serial protocol

Frame format (ASCII, terminated with `\n`, 9600 baud):

```
<base> <shoulder> <elbow> <wrist_pitch> <wrist_roll> <gripper>\n
```

Each value is an integer 0–180.  Sending `D` in place of a number
disables that servo (no PWM signal).  Enable/disable per channel via
the checkboxes in the **Serial Bridge** dock.

Example:
```
92 95 88 90 78 120\n
```

---

## Camera options

| CLI flag              | Description                          |
|-----------------------|--------------------------------------|
| `--force-webcam`      | Skip OAK-D detection, use webcam     |
| `--webcam-capture-width`  | Width for OpenCV capture (default 1920) |
| `--webcam-capture-height` | Height for OpenCV capture (default 1080)|
| `--oakd-capture-width`    | Width for OAK-D capture (default 3840) |
| `--oakd-capture-height`   | Height for OAK-D capture (default 2160)|
| `--preview-width`     | Width of camera feed display (default 1280) |
| `--preview-height`    | Height of camera feed display (default 720)  |

---

## Acknowledgements

This work applies knowledge and code basis from the two following projects.
All recognition and credits to their authors.

- [iotdesignshop/mediapipe-robot-arm-controller](https://github.com/iotdesignshop/mediapipe-robot-arm-controller)
- [Brevin Banks — MediaPipe UR5 robot tracking](https://github.com/Brevinbanks/ur5_mediapipe_motion)
