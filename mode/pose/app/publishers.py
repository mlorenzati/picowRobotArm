"""Robot angle publishers."""

from __future__ import annotations

import logging
import time

import zmq


class AnglePublisher:
    def publish(self, angles):
        raise NotImplementedError

    def close(self):
        pass


class ZMQAnglePublisher(AnglePublisher):
    def __init__(self, address):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(address)
        print(f"ZMQ publisher listening on {address}")

    def publish(self, angles):
        message = {
            "timestamp": time.time(),
            "base": int(round(angles[0])),
            "shoulder": int(round(angles[1])),
            "elbow": int(round(angles[2])),
            "wrist_pitch": int(round(angles[3])),
            "wrist_roll": int(round(angles[4])),
            "gripper": int(round(angles[5])),
        }
        self.socket.send_json(message)

    def close(self):
        self.socket.close()
        self.context.term()


class LogAnglePublisher(AnglePublisher):
    def __init__(self, filename):
        self.logger = logging.getLogger("robot_angles")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.FileHandler(filename)
            formatter = logging.Formatter("%(asctime)s %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def publish(self, angles):
        self.logger.info(
            "BASE=%3d SHOULDER=%3d ELBOW=%3d WRIST_PITCH=%3d "
            "WRIST_ROLL=%3d GRIPPER=%3d",
            round(angles[0]),
            round(angles[1]),
            round(angles[2]),
            round(angles[3]),
            round(angles[4]),
            round(angles[5]),
        )
