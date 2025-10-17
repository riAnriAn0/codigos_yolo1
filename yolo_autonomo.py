import cv2
from PIL import ImageFile
from picamera2 import Picamera2
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
model = YOLO("yolov5n.pt")  # Use the appropriate model file

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")  # Debug

    # Corrige possível leitura invertida de cores
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Redimensiona para evitar problemas
    frame = cv2.resize(frame, (640, 640))

    # Run YOLO model
    results = model(frame)

    # Plot results
    annotated_frame = results[0].plot()
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time
    text = f'FPS: {fps:.1f}'

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display
    cv2.imshow("Camera", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
