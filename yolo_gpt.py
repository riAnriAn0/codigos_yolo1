from picamera2 import Picamera2
from ultralytics import YOLO
import cv2

picam2 = Picamera2()
picam2.video_configuration.main.size = (320, 320)
picam2.video_configuration.main.format = "RGB888"
picam2.configure("video")
picam2.start()

model = YOLO("yolov8n_openvino_model")  # exportado antes

while True:
    frame = picam2.capture_array()
    results = model(frame, verbose=False)

    boxes = results[0].boxes.xyxy
    classes = results[0].boxes.cls
    confs = results[0].boxes.conf

    for box, cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
