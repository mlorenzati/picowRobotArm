# Robot Arm Pico W Controller

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/b55575d5-f2c4-44a3-befc-79849ea85ee8" />

Control a **6-DOF robotic arm** powered by a Raspberry Pi Pico W running
MicroPython. Includes multiple host-side control modes.

[Robot Arm model](https://makerworld.com/es/models/1134925-robotic-arm-with-servo-arduino)
by Emre Kalen.

---

## Modes

| Folder | Status | Description |
|--------|--------|-------------|
| `mode/serial` | ✅ Working | Qt/PySide6 slider GUI — 6 angle sliders sent over USB serial |
| `mode/teleop` | ✅ Working | MediaPipe teleoperation — mirrors user's arm movements via camera |
| `mode/ble`    | 🔲 Pending | Bluetooth Low Energy mode |
| `mode/wifi`   | 🔲 Pending | Wi-Fi mode |

---

## mode/serial — Slider GUI

Classic desktop control panel:

- 6 angle sliders (0–180°) for each servo joint
- 7 servo outputs (arm_a1 and arm_a2 are mechanically mirrored)
- Individual servo disable toggle for testing
- Connect / Disconnect + Refresh COM ports
- Automatic packet transmission on every slider movement

### Build & Run

**Linux / macOS**
```bash
cd mode/serial/app
pip install -r requirements.txt
python serialRobotApp.py
```

**Windows**
```bat
cd mode\serial\app
build.bat
```

---

## mode/teleop — MediaPipe Teleoperation

PySide6 GUI that uses **MediaPipe Holistic** to track the user's right arm
and hand via a webcam or DepthAI OAK-D camera, then sends calibrated servo
angles to the robot arm in real-time.

### Features

- Live camera feed with MediaPipe skeleton overlay
- 3 debug plane projections (front / top / side)
- **Joint Angles** dock: raw pose angle AND calibrated servo output per channel
- **Serial Bridge** dock: port selection, send rate, per-channel disable
- **Servo Calibration** dock: live `offset / scale / min / max` editor, saved to `servo_cal.json`
- **Camera** dock: select webcam or OAK-D, persisted across sessions
- All panels are dockable, floatable, and re-arrangeable
- Low-pass filter for smooth servo motion

### Servo channel mapping

| ch | Servo       | Robot HOME |
|----|-------------|-----------|
| 0  | Base        | 90°       |
| 1  | Shoulder    | 180°      |
| 2  | Elbow       | 180°      |
| 3  | Wrist Pitch | 90°       |
| 4  | Wrist Roll  | 90°       |
| 5  | Gripper     | 100° (open) |

### Setup & Run

```bash
cd mode/teleop
pip install -r requirements.txt
python teleop.py
```

See [`mode/teleop/README.md`](mode/teleop/README.md) for full documentation.

---

## Serial protocol

Both `mode/serial` and `mode/teleop` share the same ASCII serial protocol
(9600 baud):

```
<ch0> <ch1> <ch2> <ch3> <ch4> <ch5>\n
```

- Values are **integer degrees 0–180**
- Use `D` instead of a number to disable (no PWM) that servo

Example:
```
92 95 88 90 78 120\n
```

---

## Firmware

MicroPython firmware lives in `mode/serial/fw/main.py`.  Flash it to the
Pico W, then connect with any of the host-side apps above.

Servo GPIO assignments (GP0–GP6):

| GPIO | Servo      |
|------|------------|
| GP0  | Base       |
| GP1  | Arm A1 (Shoulder — drives arm_a2 as mirror automatically) |
| GP2  | Arm A2 (mirrored, firmware-internal) |
| GP3  | Arm B (Elbow) |
| GP4  | Wrist A (Pitch) |
| GP5  | Wrist B (Roll) |
| GP6  | Gripper    |

---

## Acknowledgements

- [iotdesignshop - mediapipe-robot-arm-controller](https://github.com/iotdesignshop/mediapipe-robot-arm-controller)
- [Brevin Banks — MediaPipe UR5 robot tracking](https://github.com/Brevinbanks/ur5_mediapipe_motion)
