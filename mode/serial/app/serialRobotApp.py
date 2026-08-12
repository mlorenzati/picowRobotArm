import sys
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QHBoxLayout,
    QVBoxLayout,
)

from PySide6.QtCore import Qt

import serial
import serial.tools.list_ports

class RobotArmWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Robot Arm Controller")

        self.serial = None

        self.loading_config = True
        self.config_file = Path(__file__).with_name("robot_arm_config.json")

        self.sliders = []
        self.value_labels = []
        self.disable_checks = []
        self.left_limits = []
        self.right_limits = []

        layout = QVBoxLayout()

        # Serial controls
        serial_layout = QHBoxLayout()

        self.port_combo = QComboBox()
        self.refresh_ports()

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_ports)

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)

        self.home_button = QPushButton("Home")
        self.home_button.clicked.connect(self.set_home_position)

        self.disable_on_disconnect = QCheckBox("Disable all on disconnect")
        self.disable_on_disconnect.setChecked(True)

        serial_layout.addWidget(QLabel("Port"))
        serial_layout.addWidget(self.port_combo)
        serial_layout.addWidget(refresh_button)
        serial_layout.addWidget(self.home_button)
        serial_layout.addWidget(self.connect_button)
        serial_layout.addWidget(self.disable_on_disconnect)

        layout.addLayout(serial_layout)

        # Column headers
        header = QHBoxLayout()

        header_servo = QLabel("Servo")
        header_servo.setMinimumWidth(100)

        header_left = QLabel("Left")
        header_slider = QLabel("Slider")
        header_right = QLabel("Right")
        header_value = QLabel("Value")
        header_disable = QLabel("Disable")

        header_right.setMinimumWidth(55)
        header_value.setMinimumWidth(35)
        header_right.setAlignment(Qt.AlignCenter)
        header_value.setAlignment(Qt.AlignCenter)
        header_disable.setAlignment(Qt.AlignCenter)
        header_slider.setAlignment(Qt.AlignCenter)

        header.addWidget(header_servo, 0)
        header.addWidget(header_left, 0)
        header.addWidget(header_slider, 1)
        header.addWidget(header_right, 0)
        header.addWidget(header_value, 0)
        header.addWidget(header_disable, 0)

        layout.addLayout(header)

        # Sliders + limits + disable checkboxes
        names = [
            "Base",
            "Shoulder",
            "Elbow",
            "Wrist Pitch",
            "Wrist Roll",
            "Gripper",
        ]

        for name in names:
            row = QHBoxLayout()

            # Servo name
            label = QLabel(name)
            label.setMinimumWidth(100)

            # Left limit
            left_limit = QSpinBox()
            left_limit.setRange(0, 180)
            left_limit.setValue(0)
            left_limit.setMinimumWidth(55)

            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 180)
            slider.setValue(90)

            # Right limit
            right_limit = QSpinBox()
            right_limit.setRange(0, 180)
            right_limit.setValue(180)
            right_limit.setMinimumWidth(55)

            # Current value
            value = QLabel("90")
            value.setMinimumWidth(35)

            # Disable checkbox
            disable_check = QCheckBox("Disable")
            disable_check.setChecked(False)
            if len(self.sliders) == 0:
                header_left.setFixedWidth(left_limit.sizeHint().width())
                header_right.setFixedWidth(right_limit.sizeHint().width())
                header_value.setFixedWidth(value.sizeHint().width())
                header_disable.setFixedWidth(disable_check.sizeHint().width())

            # Connections
            slider.valueChanged.connect(self.slider_changed)

            left_limit.valueChanged.connect(
                lambda value, s=slider, r=right_limit:
                self.limit_changed(s, value, r.value())
            )

            right_limit.valueChanged.connect(
                lambda value, s=slider, l=left_limit:
                self.limit_changed(s, l.value(), value)
            )

            disable_check.stateChanged.connect(self.disable_changed)

            # Layout
            row.addWidget(label, 0)
            row.addWidget(left_limit, 0)
            row.addWidget(slider, 1)
            row.addWidget(right_limit, 0)
            row.addWidget(value, 0)
            row.addWidget(disable_check, 0)

            layout.addLayout(row)

            self.sliders.append(slider)
            self.value_labels.append(value)
            self.disable_checks.append(disable_check)
            self.left_limits.append(left_limit)
            self.right_limits.append(right_limit)

        self.status = QLabel("Disconnected")

        layout.addWidget(self.status)

        self.setLayout(layout)
        self.load_config()
        self.loading_config = False

    def closeEvent(self, event):
        self.save_config()

        if self.serial is not None:
            self.serial.close()
            self.serial = None

        event.accept()

    # -------------------------------------------------
    def save_config(self):
        config = {
            "port": self.port_combo.currentText(),
            "disable_on_disconnect": self.disable_on_disconnect.isChecked(),
            "servos": []
        }

        for slider, left, right, check in zip(
            self.sliders,
            self.left_limits,
            self.right_limits,
            self.disable_checks
        ):
            config["servos"].append({
                "left": left.value(),
                "right": right.value(),
                "value": slider.value(),
                "disabled": check.isChecked()
            })

        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)

        except Exception as e:
            self.status.setText(f"Config save error: {e}")

    # -------------------------------------------------
    def load_config(self):
        if not self.config_file.exists():
            return

        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)

            # Serial port
            saved_port = config.get("port", "")

            if self.port_combo.findText(saved_port) >= 0:
                self.port_combo.setCurrentText(saved_port)

            # Disable on disconnect
            self.disable_on_disconnect.setChecked(
                config.get("disable_on_disconnect", True)
            )

            # Servo configuration
            servos = config.get("servos", [])

            for i, servo_config in enumerate(servos):

                if i >= len(self.sliders):
                    break

                left = servo_config.get("left", 0)
                right = servo_config.get("right", 180)
                value = servo_config.get("value", 90)
                disabled = servo_config.get("disabled", False)

                # Keep configuration within valid servo limits
                left = max(0, min(180, left))
                right = max(0, min(180, right))

                if left > right:
                    left, right = right, left

                value = max(left, min(value, right))

                self.left_limits[i].setValue(left)
                self.right_limits[i].setValue(right)

                self.sliders[i].setRange(left, right)
                self.sliders[i].setValue(value)

                self.disable_checks[i].setChecked(disabled)

                self._update_row_visual(i)

        except Exception as e:
            self.status.setText(f"Config load error: {e}")

    # -------------------------------------------------
    def refresh_ports(self):
        current = self.port_combo.currentText()

        self.port_combo.clear()

        for p in serial.tools.list_ports.comports():
            self.port_combo.addItem(p.device)

        index = self.port_combo.findText(current)

        if index >= 0:
            self.port_combo.setCurrentIndex(index)

    # -------------------------------------------------
    def toggle_connection(self):
        if self.serial is None:
            port = self.port_combo.currentText()

            if port == "":
                self.status.setText("No serial ports found")
                return

            try:
                self.serial = serial.Serial(
                    port,
                    9600,
                    timeout=0,
                )

                self.connect_button.setText("Disconnect")
                self.status.setText(f"Connected to {port}")

                self.send_packet()

            except Exception as e:
                self.status.setText(str(e))
                self.serial = None

        else:

            if self.disable_on_disconnect.isChecked():
                # Send all-disable packet before closing so the firmware
                # calls deinit() on every servo (no PWM signal left active)
                try:
                    packet = " ".join(
                        ["D"] * len(self.sliders)
                    ) + "\n"

                    self.serial.write(packet.encode("ascii"))
                    self.serial.flush()

                except Exception:
                    pass

            self.serial.close()
            self.serial = None

            self.connect_button.setText("Connect")
            self.status.setText("Disconnected")

    # -------------------------------------------------

    def limit_changed(self, slider, left, right):
        """
        Update the slider range when either limit changes.

        If the new limits exclude the current position, the slider
        is automatically moved inside the new range.
        """

        # Prevent an invalid range such as Left > Right
        if left > right:
            return

        slider.setRange(left, right)

        # QSlider keeps its value inside the new range.
        # This will also trigger slider_changed() if necessary.
        slider.setValue(
            max(left, min(slider.value(), right))
        )

    # -------------------------------------------------
    def _update_row_visual(self, index):
        """Grey-out the slider when the servo is disabled."""

        disabled = self.disable_checks[index].isChecked()

        self.sliders[index].setEnabled(not disabled)
        self.value_labels[index].setEnabled(not disabled)

    # -------------------------------------------------
    def slider_changed(self):
        for slider, label in zip(
            self.sliders,
            self.value_labels
        ):
            label.setText(str(slider.value()))

        self.send_packet()

    # -------------------------------------------------
    def set_home_position(self):
        """Move all servos to the predefined home position."""

        home_position = [
            90,   # Base
            180,  # Shoulder
            180,  # Elbow
            100,  # Wrist Pitch
            90,   # Wrist Roll
            170,  # Gripper
        ]

        for slider, value in zip(self.sliders, home_position):
            # Respect the configured servo limits
            value = max(slider.minimum(), min(value, slider.maximum()))
            slider.setValue(value)

        # slider_changed() is triggered by the slider changes,
        # but send once explicitly as well.
        self.send_packet()

    # -------------------------------------------------
    def disable_changed(self):
        """Called when any Disable checkbox changes state."""

        for i in range(len(self.disable_checks)):
            self._update_row_visual(i)

        self.send_packet()

    # -------------------------------------------------
    def send_packet(self):
        if self.serial is None:
            return

        tokens = []

        for slider, check in zip(
            self.sliders,
            self.disable_checks
        ):

            if check.isChecked():
                tokens.append("D")
            else:
                tokens.append(str(slider.value()))

        packet = " ".join(tokens) + "\n"

        try:
            self.serial.write(packet.encode("ascii"))

        except Exception as e:

            self.status.setText(str(e))


app = QApplication(sys.argv)

window = RobotArmWindow()
window.resize(780, 380)
window.show()

sys.exit(app.exec())