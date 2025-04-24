import RPi.GPIO as GPIO
import time

PUMP_PIN = 17  # GPIO nummer

GPIO.setmode(GPIO.BCM)
GPIO.setup(PUMP_PIN, GPIO.OUT)

try:
    print("Tænder pumpen")
    GPIO.output(PUMP_PIN, GPIO.HIGH)  # Tænd pumpen
    time.sleep(2)  # Kør pumpen i 5 sekunder
    GPIO.output(PUMP_PIN, GPIO.LOW)  # Sluk pumpen
    print("Pumpen er slukket")
finally:
    GPIO.cleanup()  # Rydder GPIO-indstillinger
