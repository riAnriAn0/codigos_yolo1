import cv2
from PIL import ImageFile
from picamera2 import Picamera2
import time
from ultralytics import YOLO


# Permite imagens incompletas (workaround)
ImageFile.LOAD_TRUNCATED_IMAGES = True

picam2 = Picamera2()
picam2.preview_configuration.main.size = (320,320)
picam2.preview_configuration.main.format = "BGR888" # Usa BGR para alinhar com o OpenCV
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# 2. Otimização: Tente carregar um modelo quantizado ONNX ou TFLite (se disponível)
# Exemplo com modelo padrão (substitua pelo seu modelo otimizado se usar TFLite/ONNX)
# Substitua "yolov8n.pt" pelo seu modelo TFLite quantizado para o máximo de performance.
#model = YOLO("yolov8n_int8.tflite") # Se usou TFLite/Coral

model = YOLO("yolov8n.pt")

# Pre-allocating frame array for efficiency (small gain, but good practice)
# frame = picam2.capture_array() # Pode ser removido se o INFERENCE_SIZE for o size da capture

while True:
    # 3. Otimização: A captura já está no tamanho e formato corretos (BGR)
    frame = picam2.capture_array()
    
    # 4. Otimização: Remova a linha de resize/cvtColor.
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Não é mais necessário!
    # frame = cv2.resize(frame, (320, 320)) # Não é mais necessário!

    # A inferência será rodada no frame de 320x320
    # O argumento '0' na função model() não é padrão. Geralmente é o 'conf'.
    # Aqui, passaremos apenas a imagem.
    results = model(frame, verbose=False) # Use os argumentos padrão do Ultralytics
    
    # Cálculo de FPS (usando o tempo do Python para maior precisão em sistemas lentos)
    start_time = time.time()
    
    # Acessando 'speed' para medir o tempo de inferência é o método correto do Ultralytics
    inference_time_ms = results[0].speed['inference']
    fps = 1000 / inference_time_ms if inference_time_ms > 0 else 0

    print(f"FPS: {fps:.1f}")

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls]
        print(f" - {label}: {conf:.2f}")

    print("-" * 30)
    # 5. Otimização: O 'time.sleep(0.2)' limita o FPS a um máximo de 5. 
    # Remova-o para ver o FPS real que o sistema pode atingir.