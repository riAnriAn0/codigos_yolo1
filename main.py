import cv2
from PIL import ImageFile
from picamera2 import Picamera2
import time
from ultralytics import YOLO


# Permite imagens incompletas (workaround)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOv8
model = YOLO("yolov8n_ncnn_model")  # Use the appropriate model file


while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 640))

    results = model(frame, verbose=False)

    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time if inference_time > 0 else 0

    print(f"FPS: {fps:.1f}")

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls]
        print(f" - {label}: {conf:.2f}")

    print("-" * 30)
    time.sleep(0.2)
