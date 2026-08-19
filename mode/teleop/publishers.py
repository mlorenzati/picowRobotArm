"""Robot angle publishers.

Provides a common AnglePublisher interface with three implementations:
    - ZMQAnglePublisher   : ZeroMQ PUB socket (network)
    - SerialAnglePublisher: Serial port (direct robot connection)
    - LogAnglePublisher   : File logging
    - NullAnglePublisher  : No-op, used when an output is disabled
"""

from __future__ import annotations

import logging
import struct
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class AnglePublisher:
    """Abstract base for robot angle publishers."""

    def publish(self, angles: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# ZMQ
# ---------------------------------------------------------------------------

class ZMQAnglePublisher(AnglePublisher):
    """Publish robot angles over a ZeroMQ PUB socket as JSON."""

    def __init__(self, address: str = "tcp://*:5555"):
        try:
            import zmq
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.PUB)
            self._socket.bind(address)
            self._address = address
            print(f"[ZMQ] Publisher listening on {address}")
        except ImportError:
            self._socket = None
            print("[ZMQ] pyzmq not installed – ZMQ publisher disabled.")

    def publish(self, angles: np.ndarray) -> None:
        if self._socket is None:
            return
        message = {
            "timestamp": time.time(),
            "base":        int(round(angles[0])),
            "shoulder":    int(round(angles[1])),
            "elbow":       int(round(angles[2])),
            "wrist_pitch": int(round(angles[3])),
            "wrist_roll":  int(round(angles[4])),
            "gripper":     int(round(angles[5])),
        }
        self._socket.send_json(message)

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
            self._context.term()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Serial
# ---------------------------------------------------------------------------

class SerialAnglePublisher(AnglePublisher):
    """Publish robot angles over a serial port.

    The protocol matches app3's original transmit_angles_serial():
        0xFE 0xFE <23 bytes angles> <checksum> 0xFD 0xFD

    The 6 servo angles are placed in bytes [19..24] of the 23-byte payload
    to match the original joint_angles layout:
        [0..15]  finger joint angles (unused in this app, sent as 0)
        [16]     wrist pitch
        [17]     wrist yaw (unused – sent as 0)
        [18]     wrist roll
        [19]     shoulder pitch
        [20]     shoulder yaw (base)
        [21]     shoulder roll (unused)
        [22]     elbow

    Optionally, rate-limiting is applied via serial_fps.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        serial_fps: int = 20,
    ):
        self._port = port
        self._baudrate = baudrate
        self._period = 1.0 / serial_fps
        self._last_tx = 0.0
        self._ser = None

        try:
            import serial as pyserial
            self._ser = pyserial.Serial(
                port=port,
                baudrate=baudrate,
                parity=pyserial.PARITY_NONE,
                stopbits=pyserial.STOPBITS_ONE,
                bytesize=pyserial.EIGHTBITS,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
                timeout=1,
            )
            print(f"[Serial] Port {port} opened at {baudrate} baud.")
        except ImportError:
            print("[Serial] pyserial not installed – serial publisher disabled.")
        except Exception as exc:
            print(f"[Serial] Could not open port {port}: {exc}")

    def publish(self, angles: np.ndarray) -> None:
        if self._ser is None:
            return

        now = time.time()
        if now - self._last_tx < self._period:
            return
        self._last_tx = now

        # Build a 23-byte payload (matches original joint_angles layout).
        payload = np.zeros(23, dtype=np.uint8)
        # Base maps to shoulder yaw slot [20]
        payload[20] = int(np.clip(round(angles[0]), 0, 255))
        # Shoulder maps to shoulder pitch slot [19]
        payload[19] = int(np.clip(round(angles[1]), 0, 255))
        # Elbow maps to elbow slot [22]
        payload[22] = int(np.clip(round(angles[2]), 0, 255))
        # Wrist pitch [16]
        payload[16] = int(np.clip(round(angles[3]), 0, 255))
        # Wrist roll [18]
        payload[18] = int(np.clip(round(angles[4]), 0, 255))
        # Gripper – map to a custom extension byte or reuse slot [17]
        payload[17] = int(np.clip(round(angles[5]), 0, 255))

        checksum = int(np.sum(payload)) & 0xFF
        checksum = 255 - checksum

        try:
            self._ser.write(b"\xFE\xFE")
            self._ser.write(struct.pack("23B", *payload))
            self._ser.write(struct.pack("B", checksum))
            self._ser.write(b"\xFD\xFD")
            self._ser.flushOutput()
        except Exception as exc:
            print(f"[Serial] Write error: {exc}")

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------

class LogAnglePublisher(AnglePublisher):
    """Append robot angles to a log file (one line per frame)."""

    def __init__(self, filename: str | Path):
        self._logger = logging.getLogger("robot_angles")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = logging.FileHandler(filename)
            formatter = logging.Formatter("%(asctime)s %(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        print(f"[Log] Logging robot angles to {filename}")

    def publish(self, angles: np.ndarray) -> None:
        self._logger.info(
            "BASE=%3d SHOULDER=%3d ELBOW=%3d "
            "WRIST_PITCH=%3d WRIST_ROLL=%3d GRIPPER=%3d",
            round(angles[0]),
            round(angles[1]),
            round(angles[2]),
            round(angles[3]),
            round(angles[4]),
            round(angles[5]),
        )


# ---------------------------------------------------------------------------
# Null (no-op)
# ---------------------------------------------------------------------------

class NullAnglePublisher(AnglePublisher):
    """No-op publisher used when output is disabled."""

    def publish(self, angles: np.ndarray) -> None:
        pass
