import sys

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
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

        layout = QVBoxLayout()

        #
        # Serial controls
        #

        serial_layout = QHBoxLayout()

        self.port_combo = QComboBox()
        self.refresh_ports()

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_ports)

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)

        serial_layout.addWidget(QLabel("Port"))
        serial_layout.addWidget(self.port_combo)
        serial_layout.addWidget(refresh_button)
        serial_layout.addWidget(self.connect_button)

        layout.addLayout(serial_layout)

        #
        # Sliders
        #

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

            label = QLabel(name)
            label.setMinimumWidth(100)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 180)
            slider.setValue(90)
            slider.valueChanged.connect(self.slider_changed)

            value = QLabel("90")
            value.setMinimumWidth(35)

            row.addWidget(label)
            row.addWidget(slider)
            row.addWidget(value)

            layout.addLayout(row)

            self.sliders.append(slider)
            self.value_labels.append(value)

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

            self.serial.close()
            self.serial = None

            self.connect_button.setText("Connect")
            self.status.setText("Disconnected")

    # -------------------------------------------------

    def slider_changed(self):

        for slider, label in zip(self.sliders, self.value_labels):
            label.setText(str(slider.value()))

        self.send_packet()

    # -------------------------------------------------

    def send_packet(self):

        if self.serial is None:
            return

        values = [str(s.value()) for s in self.sliders]

        packet = " ".join(values) + "\n"

        try:
            self.serial.write(packet.encode("ascii"))

        except Exception as e:

            self.status.setText(str(e))


app = QApplication(sys.argv)

window = RobotArmWindow()
window.resize(600, 350)
window.show()

sys.exit(app.exec())