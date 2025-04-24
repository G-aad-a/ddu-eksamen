from pyax12.connection import Connection
import time
import threading
import RPi.GPIO as GPIO

serial_lock = threading.Lock()

class Motor:
    """Class to control Dynamixel AX12 actuators"""

    def __init__(self):
        # Initialize the connection only once during initialization
        self.sc = Connection(port="/dev/ttyUSB0", baudrate=1000000)
        #self.sc = Connection(port="COM4", baudrate=1000000)

    def start(self):
        """Method to flush the connection on startup"""
        self.sc.flush()

    def scanUnits(self):
        """Scans for connected Dynamixel motors and returns their IDs in a list."""
        try:
            ids = self.sc.scan()
            print(f"Connected motor IDs: {ids}")
            return ids
        except ValueError as e:
            print(f"Error scanning units: {e}")
            return []

    def readDxl(self, ID):
        """Prints information of the motor with specified ID"""
        self.sc.flush()
        self.sc.pretty_print_control_table(ID)

    def jointMode(self, ID):
        """Sets the motor with the given ID to joint mode"""
        self.sc.set_cw_angle_limit(ID, 0, False)
        self.sc.set_ccw_angle_limit(ID, 1023, False)

    def wheelMode(self, ID):
        """Sets the motor with the given ID to wheel mode"""
        self.sc.set_ccw_angle_limit(ID, 0, False)
        self.sc.set_cw_angle_limit(ID, 0, False)

    def moveJoint(self, ID, position, speed=512):
        """Moves the motor with the given ID to a specified position"""
        with serial_lock:
            self.sc.goto(ID, position, speed=speed, degrees=True)
        

    def moveWheel(self, ID, speed):
        """Starts a motor in wheel mode with the given speed"""
        def wheel_movement(speed):
            try:
                if speed < 0:
                    if speed < -1024:
                        speed = 2047
                    else:
                        speed = 1023 + (-speed)
                else:
                    if speed > 1023:
                        speed = 1023
                self.sc.flush()
                with serial_lock:
                    self.sc.goto(ID, 0, int(speed), degrees=False)
            except Exception as e:
                print(f"Error moving wheel: {e}")

        thread = threading.Thread(target=wheel_movement, args=(speed,))
        thread.start()

    def stop(self):
        """Closes the USB connection to the motors"""
        self.sc.close()

    def getPos(self, ID):
        """Returns the position of the motor with the specified ID"""
        position = self.sc.get_present_position(ID, True)
        return position
    
motor = Motor()
motor.start()
#1,2,6,7

#1 = højre baghjul kører baglæns
#2 = venstre forhjul kører fremad
#6 = venstre baghjul kører fremad
#7 = højre forhjul kører baglæns


#init
for i in [1,2,6,7]:
    motor.wheelMode(i)

def rotate(degrees):
    """
    Rotates the car by a specified number of degrees.
    Positive degrees = right turn (clockwise).
    Negative degrees = left turn (counter-clockwise).
    """

    ROTATION_SPEED = 512  # Adjust if needed
    SECONDS_PER_90_DEG = 2.500  # Time it takes to turn 90° on the spot
    duration = abs(degrees) / 90 * SECONDS_PER_90_DEG
    wheel_ids = [1, 2, 6, 7]  # IDs of the wheels
    
    if degrees > 0:
       # Turn right (clockwise) — reverse left wheels and forward right wheels
       for i in wheel_ids:
            motor.moveWheel(i, ROTATION_SPEED)
            #time.sleep(0.1)
    elif degrees < 0:
        # Turn left (counter-clockwise) — reverse all wheels
        for i in wheel_ids[::-1]:
            motor.moveWheel(i, -ROTATION_SPEED)
            #time.sleep(0.1)
    else:
        return  # No rotation needed

    time.sleep(duration)

    # Stop all wheels
    for i in wheel_ids:
        motor.moveWheel(i, 0)
        #stime.sleep(0.1)


def stop():
    for i in [1, 2, 6, 7]:
        motor.moveWheel(i, 0)
        #time.sleep(0.1)

def move_forward(speed=512, duration=1.0):
    motor.moveWheel(1, -speed)
    motor.moveWheel(2, speed)
    motor.moveWheel(6, speed)
    motor.moveWheel(7, -speed)
    if duration > 0:
        time.sleep(duration)
        stop()

def move_backward(speed=512, duration=1.0):
    motor.moveWheel(1, speed)
    motor.moveWheel(2, speed)
    motor.moveWheel(6, speed)
    motor.moveWheel(7, speed)
    if duration > 0:
        time.sleep(duration)
        stop()


def move_right(speed=512, duration=1.0):
    motor.moveWheel(1, speed)
    motor.moveWheel(2, speed)
    motor.moveWheel(6, speed)
    motor.moveWheel(7, speed)
    if duration > 0:
        time.sleep(duration)
        stop()


def move_left(speed=512, duration=1.0):
    motor.moveWheel(1, -speed)
    motor.moveWheel(2, -speed)
    motor.moveWheel(6, -speed)
    motor.moveWheel(7, -speed)
    if duration > 0:
        time.sleep(duration)
        stop()



def shoot():
    PUMP_PIN = 17  # GPIO nummer
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PUMP_PIN, GPIO.OUT)

    try:
        print("Tænder pumpen")
        GPIO.output(PUMP_PIN, GPIO.HIGH)  # Tænd pumpen
        time.sleep(1)  # Kør pumpen i 5 sekunder
        GPIO.output(PUMP_PIN, GPIO.LOW)  # Sluk pumpen
        print("Pumpen er slukket")
    finally:
        GPIO.cleanup()  # Rydder GPIO-indstillinger


#rotate(-180)
#move_left(512, 2)