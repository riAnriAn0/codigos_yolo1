import cv2
from PIL import ImageFile
from picamera2 import Picamera2
import time
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True

INFERENCE_SIZE = (320, 320)

picam2 = Picamera2()
picam2.preview_configuration.main.size = INFERENCE_SIZE
picam2.preview_configuration.main.format = "BGR888" 
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

model = YOLO("yolov8n_ncnn_model")

while True:
    frame = picam2.capture_array()

    results = model(frame, 0,verbose=False) 
    start_time = time.time()
    
    inference_time_ms = results[0].speed['inference']
    fps = 1000 / inference_time_ms if inference_time_ms > 0 else 0

    print(f"FPS: {fps:.1f}")

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls]
        print(f" - {label}: {conf:.2f}")

    print("-" * 30)
