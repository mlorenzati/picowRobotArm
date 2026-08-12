from machine import Pin, PWM
import sys
import uselect

# -----------------------------
# Servo class
# -----------------------------
class Servo:
    def __init__(self, pin):
        self.pin = pin
        self.pwm = PWM(Pin(pin))
        self.pwm.freq(50)
        self.disabled = False

    def write(self, angle):
        if self.disabled:
            return
        angle = max(0, min(180, angle))

        # 500-2500us pulse
        pulse = 500 + (angle * 2000 // 180)

        # 16-bit duty
        duty = int(pulse * 65535 / 20000)

        self.pwm.duty_u16(duty)

    def disable(self):
        """Completely stop PWM output so the servo receives no signal."""
        self.disabled = True
        self.pwm.deinit()   # de-initialise the PWM peripheral; pin goes low

    def enable(self):
        """Re-initialise PWM so the servo can be driven again."""
        if self.disabled:
            self.pwm = PWM(Pin(self.pin))
            self.pwm.freq(50)
            self.disabled = False


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

# Ordered list: indices match the 6 logical channels sent by the app.
# Note: servo_arm_a2 is a mirror of servo_arm_a1 (180 - angle),
# so it is not a separate logical channel – it is driven internally.
servos = [
    servo_root,     # 0 – Base
    servo_arm_a1,   # 1 – Shoulder (arm_a2 follows automatically)
    servo_arm_b,    # 2 – Elbow
    servo_wrist_a,  # 3 – Wrist Pitch
    servo_wrist_b,  # 4 – Wrist Roll
    servo_gripper,  # 5 – Gripper
]

poll = uselect.poll()
poll.register(sys.stdin)

buffer = ""

print("Robot Arm Ready")

while True:

    if poll.poll(0):

        c = sys.stdin.read(1)

        if c == '\n' or c == '\r':

            # Normalise separators → spaces
            for sep in ",;\t":
                buffer = buffer.replace(sep, " ")

            tokens = buffer.split()

            if len(tokens) >= 6:

                for i, token in enumerate(tokens[:6]):
                    token = token.strip().upper()

                    if token == 'D':
                        # Disable this servo
                        servos[i].disable()
                        # Also disable the mirror servo for the shoulder
                        if i == 1:
                            servo_arm_a2.disable()
                    else:
                        try:
                            angle = int(token)
                        except ValueError:
                            continue

                        servos[i].enable()
                        servos[i].write(angle)

                        # Shoulder has a mirrored servo
                        if i == 1:
                            servo_arm_a2.enable()
                            servo_arm_a2.write(180 - angle)

            buffer = ""

        else:
            buffer += c
