from ultralytics import YOLO
import cv2
import numpy as np

# Cargar modelo YOLOv5
model = YOLO("yolov5n.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Resolución para la inferencia
det_width, det_height = 640, 360

# Suavizado
smoothed_box = None
alpha = 0.2  # Interpolación

def lerp(a, b, alpha):
    return a + (b - a) * alpha

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    small = cv2.resize(frame, (det_width, det_height))
    results = model(small, verbose=False)[0]

    scale_x = w / det_width
    scale_y = h / det_height

    best_person = None
    best_conf = 0

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0 and conf > best_conf:
            best_person = (x1, y1, x2, y2)
            best_conf = conf

    if best_person:
        # Escalar coordenadas
        x1, y1, x2, y2 = best_person
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

        if smoothed_box is None:
            smoothed_box = [x1, y1, x2, y2]
        else:
            smoothed_box = [
                int(lerp(smoothed_box[0], x1, alpha)),
                int(lerp(smoothed_box[1], y1, alpha)),
                int(lerp(smoothed_box[2], x2, alpha)),
                int(lerp(smoothed_box[3], y2, alpha)),
            ]

        # Dibujar caja
        x1s, y1s, x2s, y2s = smoothed_box
        cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), (0, 255, 0), 3)

        # Etiqueta dinámica
        tag_width, tag_height = 90, 30
        margin = 5

        # Verificar si hay espacio debajo
        if y2s + tag_height + margin < h:
            tag_x, tag_y = x1s, y2s + margin
        else:
            tag_x, tag_y = x1s, y1s - tag_height - margin

        # Evitar que se salga por arriba
        if tag_y < 0:
            tag_y = 0

        # Dibujar etiqueta
        cv2.rectangle(frame, (tag_x, tag_y), (tag_x + tag_width, tag_y + tag_height), (0, 255, 0), -1)
        cv2.putText(frame, "persona", (tag_x + 5, tag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Mostrar
    cv2.imshow("YOLOv5 - Persona Etiqueta Dinámica", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
