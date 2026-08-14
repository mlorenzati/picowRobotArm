#!/usr/bin/env python3
"""MediaPipe 6DOF robot teleoperation application."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QComboBox, QGroupBox, QFormLayout,
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
)

from publishers import ZMQAnglePublisher, LogAnglePublisher
from robot_mapping import HOME, RobotMapper
from vision import VisionProcessor


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
POSE_MODEL = MODEL_DIR / "pose_landmarker_full.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
ZMQ_ADDRESS = "tcp://*:5555"


state = {
    "frame": None,
    "angles": HOME.copy(),
    "human_angles": None,
    "gripper": None,
    "camera_ok": False,
    "pose_ok": False,
    "hand_ok": False,
    "running": True,
    "error": None,
    "fps": 0.0,
    "source": {"type": "camera", "value": 0},
    "source_name": "Camera 0",
    "hand_robot": None,
}


class TeleopWindow(QWidget):
    def __init__(self, mapper: RobotMapper):
        super().__init__()
        self.mapper = mapper

        self.setWindowTitle("MediaPipe 6DOF Robot Teleoperation")
        self.resize(900, 750)

        self.video_label = QLabel("Waiting for camera...")
        self.video_label.setMinimumSize(720, 480)
        self.video_label.setAlignment(Qt.AlignCenter)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Input:"))
        self.source_combo = QComboBox()
        self.refresh_sources()
        self.source_combo.currentIndexChanged.connect(self.source_changed)
        source_row.addWidget(self.source_combo, 1)
        self.video_button = QPushButton("Open Video...")
        self.video_button.clicked.connect(self.open_video)
        source_row.addWidget(self.video_button)

        names = [
            "Base", "Shoulder", "Elbow",
            "Wrist Pitch", "Wrist Roll", "Gripper",
        ]

        joint_layout = QGridLayout()
        self.joint_values = []

        for row, name in enumerate(names):
            name_label = QLabel(name)
            value_label = QLabel(f"{HOME[row]:.0f}°")
            value_label.setMinimumWidth(80)

            joint_layout.addWidget(name_label, row, 0)
            joint_layout.addWidget(value_label, row, 1)
            self.joint_values.append(value_label)

        diagnostic_box = QGroupBox("Hand2RobotWorld / Mapping Diagnostics")
        diagnostic_layout = QFormLayout()
        self.h2r_position = QLabel("---")
        self.h2r_orientation = QLabel("---")
        self.raw_values = QLabel("---")
        self.neutral_delta = QLabel("---")
        self.neutral_values = QLabel("---")
        diagnostic_layout.addRow("H2R position X/Y/Z", self.h2r_position)
        diagnostic_layout.addRow("H2R orientation R/P/Y", self.h2r_orientation)
        diagnostic_layout.addRow("Raw human angles", self.raw_values)
        diagnostic_layout.addRow("Neutral human", self.neutral_values)
        diagnostic_layout.addRow("Neutral delta", self.neutral_delta)
        diagnostic_box.setLayout(diagnostic_layout)

        self.calibrate_button = QPushButton("Calibrate Neutral")
        self.calibrate_button.clicked.connect(self.calibrate_clicked)

        self.home_button = QPushButton("HOME")
        self.home_button.clicked.connect(self.home_clicked)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.calibrate_button)
        button_layout.addWidget(self.home_button)

        self.status_label = QLabel("Starting...")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(source_row)
        layout.addLayout(joint_layout)
        layout.addWidget(diagnostic_box)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(50)

    def refresh_sources(self):
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for index in range(8):
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
            if cap.isOpened():
                self.source_combo.addItem(f"Camera {index}", {"type": "camera", "value": index})
            cap.release()
        if self.source_combo.count() == 0:
            self.source_combo.addItem("Camera 0 (not detected)", {"type": "camera", "value": 0})
        self.source_combo.blockSignals(False)

    def source_changed(self, index):
        source = self.source_combo.itemData(index)
        if source:
            state["source"] = source
            state["source_name"] = (
                f"Video: {source['value']}" if source["type"] == "video"
                else f"Camera {source['value']}"
            )
            self.mapper.reset_home()
            state["angles"] = HOME.copy()

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v);;All files (*)",
        )
        if not path:
            return
        self.source_combo.blockSignals(True)
        self.source_combo.addItem(f"Video: {Path(path).name}", {"type": "video", "value": path})
        self.source_combo.setCurrentIndex(self.source_combo.count() - 1)
        self.source_combo.blockSignals(False)
        self.source_changed(self.source_combo.currentIndex())

    def calibrate_clicked(self):
        if not state["pose_ok"] or state["human_angles"] is None:
            print("Calibration failed: pose not detected.")
            return

        if self.mapper.calibrate(
            state["human_angles"],
            state["gripper"],
        ):
            state["angles"] = HOME.copy()
            print("\n================================")
            print("CALIBRATION COMPLETE")
            print("================================")
            for name, value in zip(
                ("Base", "Shoulder", "Elbow", "Wrist pitch", "Wrist roll"),
                state["human_angles"],
            ):
                print(f"{name:12s}: {value:7.2f}")
            print(f"{'Gripper':12s}: {state['gripper']:7.2f}")
            print("Robot neutral:")
            print(
                "Base=90 Shoulder=180 Elbow=180 "
                "WristPitch=90 WristRoll=90 Gripper=100"
            )
            print("================================\n")

    def home_clicked(self):
        state["angles"] = self.mapper.reset_home()
        print("Robot HOME:")
        print(state["angles"])

    def update_gui(self):
        frame = state["frame"]

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape

            image = QImage(
                rgb.data,
                width,
                height,
                channels * width,
                QImage.Format_RGB888,
            )

            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        angles = state["angles"]
        for i in range(6):
            self.joint_values[i].setText(f"{angles[i]:.0f}°")

        if state["error"]:
            self.status_label.setText(state["error"])
            return

        if not state["camera_ok"]:
            self.status_label.setText("Opening camera...")
            return

        if not state["pose_ok"]:
            self.status_label.setText(
                "Camera OK | Waiting for person..."
            )
            return

        if not self.mapper.calibrated:
            self.status_label.setText(
                "POSE OK | Place arm in robot HOME position "
                "and press Calibrate Neutral"
            )
            return

        hand = "HAND OK" if state["hand_ok"] else "HAND ---"
        self.status_label.setText(
            f"POSE OK | {hand} | CALIBRATED | "
            f"{state.get('source_name', 'Source')} | FPS {state['fps']:.1f} | ZMQ :5555"
        )


def main():
    if not POSE_MODEL.exists():
        print(f"Missing model: {POSE_MODEL}")
        print("Run: python setup.py")
        return 1

    if not HAND_MODEL.exists():
        print(f"Missing model: {HAND_MODEL}")
        print("Run: python setup.py")
        return 1

    mapper = RobotMapper()

    publishers = [
        ZMQAnglePublisher(ZMQ_ADDRESS),
        LogAnglePublisher(BASE_DIR / "robot_angles.log"),
    ]

    processor = VisionProcessor(
        POSE_MODEL,
        HAND_MODEL,
        state,
    )

    vision_thread = threading.Thread(
        target=processor.run,
        args=(publishers, mapper),
        daemon=True,
    )
    vision_thread.start()

    app = QApplication(sys.argv)
    window = TeleopWindow(mapper)
    window.show()

    try:
        return app.exec()
    finally:
        state["running"] = False

        for publisher in publishers:
            try:
                publisher.close()
            except Exception:
                pass

        vision_thread.join(timeout=2.0)


if __name__ == "__main__":
    sys.exit(main())
