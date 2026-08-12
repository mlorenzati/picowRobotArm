import sys

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

        self.disable_on_disconnect = QCheckBox("Disable all on disconnect")
        self.disable_on_disconnect.setChecked(True)

        serial_layout.addWidget(QLabel("Port"))
        serial_layout.addWidget(self.port_combo)
        serial_layout.addWidget(refresh_button)
        serial_layout.addWidget(self.connect_button)
        serial_layout.addWidget(self.disable_on_disconnect)

        layout.addLayout(serial_layout)

        # Column headers
        header = QHBoxLayout()

        header.addWidget(QLabel("Servo"))
        header.itemAt(0).widget().setMinimumWidth(100)

        header.addWidget(QLabel("Left"))
        header.addWidget(QLabel("Slider"))
        header.addWidget(QLabel("Right"))
        header.addWidget(QLabel("Value"))
        header.addWidget(QLabel("Disable"))

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
            row.addWidget(label)
            row.addWidget(left_limit)
            row.addWidget(slider)
            row.addWidget(right_limit)
            row.addWidget(value)
            row.addWidget(disable_check)

            layout.addLayout(row)

            self.sliders.append(slider)
            self.value_labels.append(value)
            self.disable_checks.append(disable_check)
            self.left_limits.append(left_limit)
            self.right_limits.append(right_limit)

        self.status = QLabel("Disconnected")

        layout.addWidget(self.status)

        self.setLayout(layout)

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