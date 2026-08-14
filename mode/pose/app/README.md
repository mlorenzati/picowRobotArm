# Hand / Arm Pose Teleoperation

This project is deliberately split into three concepts so that geometry can be
fixed without changing servo mapping:

1. **Human geometry** (`arm_geometry.py`) — MediaPipe landmarks become human
   anatomical/reference angles.
2. **Calibration** (`robot_mapping.py`) — records the human pose that means
   robot HOME.
3. **Servo mapping** (`robot_mapping.py`) — converts movement of each joint
   independently into servo angles.

`vision.py` owns camera/MediaPipe processing, `publishers.py` owns ZMQ/logging,
and `teleop.py` owns the Qt UI.

## Robot conventions

The requested robot neutral/HOME is:

```text
Base         = 90
Shoulder     = 180
Elbow        = 180
Wrist Pitch  = 90
Wrist Roll   = 90
Gripper      = 100
```

### Base

The base is measured from the person's torso forward direction to the actual
upper-arm direction. The intended range is:

```text
person's left      camera/forward      person's right
      0                    90                180
```

The torso only supplies the reference direction. The arm supplies the actual
movement.

### Shoulder

The upper arm is compared with body-down:

```text
shoulder = 180 - angle(shoulder->elbow, body_down)
```

Therefore:

```text
arm down        = 180
arm horizontal  = 90
arm up          = 0
```

### Elbow

The angle is measured directly at the elbow:

```text
angle(elbow->shoulder, elbow->wrist)
```

Therefore:

```text
straight = 180
folded   = 0
```

This is independent of shoulder and base.

### Wrist pitch / roll

These are calculated from the hand world landmarks and the forearm. They are
kept isolated because the exact physical servo axis depends on how the robot's
wrist is mounted.

The two signs are easy to tune in `arm_geometry.py`:

```python
WRIST_PITCH_SIGN = 1.0
WRIST_ROLL_SIGN = 1.0
```

Change one to `-1.0` if that axis moves backwards.

### Gripper

Finger PIP and DIP joint angles are used to estimate finger extension. The
result is converted directly to the requested robot convention:

```text
open hand  = 100
closed fist = 180
```

This is intentionally **not** neutral-relative like the five arm joints.

## Calibration

Calibration records the current human geometry as neutral. It does not replace
robot HOME values.

For the five arm joints:

```text
MediaPipe human angle
        -
calibrated human neutral
        |
        v
joint movement
        |
        v
HUMAN_RANGE / sensitivity / direction
        |
        v
robot servo angle around HOME
```

The gripper is separate and remains in its absolute `100..180` convention.

## Tuning servo response

Edit `robot_mapping.py`:

```python
HUMAN_RANGE = np.array([
    70,  # base
    60,  # shoulder
    70,  # elbow
    70,  # wrist pitch
    90,  # wrist roll
])
```

Smaller values make a joint more sensitive.

Use `DIRECTION` to reverse an individual servo.

## What to test first

Before changing sensitivity, deliberately test one movement at a time:

1. Arm left/right -> Base
2. Arm up/down -> Shoulder
3. Bend/straighten elbow -> Elbow
4. Bend wrist -> Wrist pitch
5. Rotate palm -> Wrist roll
6. Open/close hand -> Gripper

If the displayed **human** angle is wrong, fix `arm_geometry.py`. If the human
angle is correct but the servo movement is too large/small or reversed, fix
`robot_mapping.py`.


## Hand2RobotWorld diagnostics

The GUI now exposes the intermediate hand transform rather than only showing
the final servo values. The diagnostic uses the same core idea as the
Brevin Banks UR5 MediaPipe project: MediaPipe hand-world coordinates are
converted with a fixed Rx(-90°) convention and a local hand basis is built
from the wrist/index/middle landmarks. The original project uses the wrist
relative to the hip to construct a 6DOF transform and then feeds that into
robot kinematics; this project currently uses the hand transform as a
diagnostic/reference because our robot is driven by six independent servo
angles rather than UR5 inverse kinematics.

The GUI shows:

- H2R position X/Y/Z
- H2R orientation R/P/Y
- raw human angles
- captured neutral human angles
- current neutral delta
- final servo angles

This makes it possible to identify whether an error originates in MediaPipe
geometry, neutral calibration, or servo mapping before changing tuning values.

## Camera and video input

The Input selector enumerates available camera indices at startup. `Open Video...`
adds an MP4/MOV/AVI/MKV/M4V file as an input source. Changing the source
reopens the capture without restarting the application. Video files loop when
they reach the end, which makes them useful for repeatable geometry tests.

The video source is deliberately kept separate from the mapping layer, so a
recorded sequence can be used to tune the math without moving the physical
robot.

## Hand 6-DOF diagnostic revision

The Hand2RobotWorld diagnostic now separates **translation** from **orientation**.

The previous implementation constructed the position from vectors such as
`index - wrist`. Those vectors are invariant under whole-hand translation, so
XYZ stayed almost constant when the hand moved around the image.

The current diagnostic uses:

- **X** = normalized wrist image position left/right, centered at 0.
- **Y** = normalized wrist image position up/down, centered at 0.
- **Z** = MediaPipe hand-world wrist depth as a *relative* depth signal.
- **Roll/Pitch/Yaw** = wrist-relative hand orientation from the index/middle/pinky geometry.

Therefore moving the entire hand left/right/up/down should change XYZ without
necessarily changing RPY. Rotating the hand should change RPY without changing
XY. This is intentionally a diagnostic representation; the final robot servo
retargeting remains a separate stage.

The XYZ values are not metres. X/Y are normalized camera coordinates and Z is
a MediaPipe-relative depth value. A later integration step can calibrate this
6-DOF space into the robot's physical workspace.
