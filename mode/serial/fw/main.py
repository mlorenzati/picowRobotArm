from machine import Pin, PWM
import sys
import uselect

# -----------------------------
# Servo class
# -----------------------------
class Servo:
    def __init__(self, pin):
        self.pwm = PWM(Pin(pin))
        self.pwm.freq(50)

    def write(self, angle):
        angle = max(0, min(180, angle))

        # 500-2500us pulse
        pulse = 500 + (angle * 2000 // 180)

        # 16-bit duty
        duty = int(pulse * 65535 / 20000)

        self.pwm.duty_u16(duty)


# -----------------------------
# Servo assignment
# GP0..GP6
# -----------------------------

servo_root     = Servo(0)
servo_arm_a1   = Servo(1)
servo_arm_a2   = Servo(2)
servo_arm_b    = Servo(3)
servo_wrist_a  = Servo(4)
servo_wrist_b  = Servo(5)
servo_gripper  = Servo(6)

poll = uselect.poll()
poll.register(sys.stdin)

buffer = ""

print("Robot Arm Ready")

while True:

    if poll.poll(0):

        c = sys.stdin.read(1)

        if c == '\n' or c == '\r':

            # Accept spaces, commas, semicolons and tabs
            for c in ",;\t":
                buffer = buffer.replace(c, " ")

            try:
                numbers = [int(x) for x in buffer.split()]
            except ValueError:
                numbers = []

            if len(numbers) >= 6:

                servo_root.write(numbers[0])

                servo_arm_a1.write(numbers[1])
                servo_arm_a2.write(180 - numbers[1])

                servo_arm_b.write(numbers[2])
                servo_wrist_a.write(numbers[3])
                servo_wrist_b.write(numbers[4])
                servo_gripper.write(numbers[5])

            buffer = ""

        else:
            buffer += c