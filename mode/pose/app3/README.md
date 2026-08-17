# app3 — MediaPipe 6-DOF Robot Teleoperation

Unified, well-structured teleoperation program that drives a 6-DOF serial
robot arm by tracking the user's right arm and right hand with a webcam
(or DepthAI OAK-D) and MediaPipe Holistic.

## Architecture

```
teleop.py          ← PySide6 GUI entry-point
  ├── vision.py    ← MediaPipe thread (camera → landmarks → angles)
  │     └── arm_geometry.py  ← pure geometry (no robot concepts)
  │     └── robot_mapping.py ← calibration, servo limits, smoothing
  ├── publishers.py← ZMQ / Serial / Log interface implementations
  ├── opencv_cam.py← OpenCV camera wrapper
  └── depthai_cam.py← DepthAI OAK-D camera wrapper
```

### Improvements over individual apps

| Feature | app | app2 | app3 (new) |
|---|---|---|---|
| Clean class separation | ✓ | partial | ✓ |
| Publisher interfaces | ✓ ZMQ+Log | — | ✓ ZMQ+Serial+Log |
| Robot arm simulator | — | ✓ | ✓ |
| Wrist roll | — | — | ✓ (from hand world landmarks) |
| Wrist pitch | — | — | ✓ (from hand world landmarks) |
| Gripper open/close | partial | — | ✓ (finger extension detection) |
| DepthAI camera | — | — | ✓ |
| Source switching | — | — | ✓ (runtime GUI + CLI) |
| GUI interface toggles | — | — | ✓ (ZMQ/Serial/Log checkboxes) |

## Joint convention

| Index | Name | Robot HOME | Range |
|---|---|---|---|
| 0 | base | 90° | 0–180° |
| 1 | shoulder | 180° | 0–180° |
| 2 | elbow | 180° | 0–180° |
| 3 | wrist_pitch | 90° | 0–180° |
| 4 | wrist_roll | 90° | 0–180° |
| 5 | gripper | 100° (open) | 100–180° |

### Wrist pitch
Measured as the signed angle between the forearm direction and the
hand-forward (middle-MCP → wrist) direction, around the palm-width axis.
90° = forearm and hand collinear (neutral).

### Wrist roll
Measured as the signed angle between an upright body-reference vector
(projected onto the forearm axis) and the palm normal (also projected onto
the forearm axis). 90° = palm facing down (neutral).

### Gripper
Finger extension average for index/middle/ring/pinky.
- Open hand → 100° servo
- Closed fist → 180° servo

## Setup

```bash
pip install -r requirements.txt
```

For DepthAI OAK-D support, uncomment the optional lines at the end of
`requirements.txt` and run `pip install` again.

## Usage

```bash
# Default: webcam 0 + ZMQ publisher on tcp://*:5555
python teleop.py

# Serial only (no ZMQ)
python teleop.py --no-zmq --serial /dev/ttyUSB0

# Serial + log file
python teleop.py --serial /dev/ttyUSB0 --log robot_angles.log

# Force webcam index 1
python teleop.py --force-webcam --camera-index 1

# Tune smoothing (0 = no filter, 1 = frozen)
python teleop.py --lpf 0.15
```

## Workflow

1. Start the app. The camera feed appears on the left.
2. Stand in front of the camera so your full body is visible.
3. Hold your right arm in the robot's **HOME** position (arm relaxed at side).
4. Click **Calibrate Neutral** to record your neutral pose.
5. Move your arm — the robot mirrors it in real-time via the selected
   publisher(s).
6. The right panel shows:
   - **Arm simulator**: real-time 3-D side view of the robot arm.
   - **Diagnostics**: raw human angles, wrist pitch/roll, gripper state.
   - **Output Interfaces**: toggle ZMQ / Serial / Log at runtime.

## Serial protocol

Frame format (28 bytes total):
```
0xFE 0xFE | payload[23] | checksum | 0xFD 0xFD
```

Payload layout (23 bytes):
```
[0..15]  finger joints (unused, 0)
[16]     wrist_pitch
[17]     gripper
[18]     wrist_roll
[19]     shoulder
[20]     base
[21]     0 (unused)
[22]     elbow
```
`checksum = (255 - sum(payload)) & 0xFF`
