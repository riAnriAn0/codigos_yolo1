import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM) # BCM pinagem BCM (Numeroamento GPIO)

GPIO.setup(5, GPIO.OUT)  # Configura o pino 5 como saída

for i in range(5):  # Pisca o LED 5 vezes
    GPIO.output(5, GPIO.HIGH)  # Define o pino 5 como HIGH (liga o LED)
    time.sleep(1)  # Mantém o LED ligado por 1 segundo
    GPIO.output(5, GPIO.LOW)   # Define o pino 5 como LOW
    time.sleep(1)  # Mantém o LED desligado por 1 segundo

GPIO.cleanup()  # Limpa a configuração dos pinos GPIO