import cv2
import torch
import numpy as np

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # confianza mínima

# Función de interpolación lineal
def lerp(a, b, t):
    return a + (b - a) * t

# Inicializar interpolación
smoothed_box = None
alpha = 0.2  # suavizado normal

# Iniciar cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Inferencia con YOLOv5
    results = model(frame)
    detections = results.xyxy[0]

    found_person = False

    for *box, conf, cls in detections:
        if int(cls.item()) == 0:  # clase 0 = persona
            x1, y1, x2, y2 = map(int, box)

            # Expande ligeramente la caja
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(width, x2 + margin)
            y2 = min(height, y2 + margin)

            # Interpolación híbrida
            if smoothed_box is None:
                smoothed_box = [x1, y1, x2, y2]
            else:
                dist = sum(abs(a - b) for a, b in zip(smoothed_box, [x1, y1, x2, y2]))
                if dist > 150:  # Snap instantáneo si te moviste mucho
                    smoothed_box = [x1, y1, x2, y2]
                else:
                    smoothed_box = [
                        int(lerp(smoothed_box[0], x1, alpha)),
                        int(lerp(smoothed_box[1], y1, alpha)),
                        int(lerp(smoothed_box[2], x2, alpha)),
                        int(lerp(smoothed_box[3], y2, alpha)),
                    ]

            x1_s, y1_s, x2_s, y2_s = smoothed_box

            # Dibujar mark box verde
            cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 3)

            # Posicionar la pestañita "persona"
            label = "persona"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_w, label_h = label_size

            tag_margin = 5
            tag_x = x1_s
            tag_y = y2_s + label_h + tag_margin

            # Si no cabe debajo, colócala arriba
            if tag_y + label_h > height:
                tag_y = y1_s - tag_margin

            tag_rect_top_left = (tag_x, tag_y - label_h - tag_margin)
            tag_rect_bottom_right = (tag_x + label_w + 10, tag_y)

            cv2.rectangle(frame, tag_rect_top_left, tag_rect_bottom_right, (0, 255, 0), -1)
            cv2.putText(frame, label, (tag_x + 5, tag_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            found_person = True
            break  # Solo seguimos la primera persona detectada

    if not found_person:
        smoothed_box = None  # Reset si no hay detección

    # Mostrar frame
    cv2.imshow("Detección de persona", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
