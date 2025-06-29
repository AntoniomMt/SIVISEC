import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Cargar modelo YOLOv5n
model = YOLO("yolov8s.pt")

# Inicializar DeepSort tracker
tracker = DeepSort(max_age=15)

# Sensibilidad personalizada
min_area = 10000
aspect_ratio_threshold = 0.3
min_area_percent = 0.05

# Abrir cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Padding virtual en X (20 px a cada lado)
    padded_frame = cv2.copyMakeBorder(frame, 0, 0, 20, 20, cv2.BORDER_REPLICATE)

    # Detección con YOLOv5n
    results = model(padded_frame, verbose=False, conf=0.3)

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(boxes, scores, class_ids):
            if class_id == 0 and score > 0.3:  # Clase "person"
                x1, y1, x2, y2 = box

                # Compensar padding en X
                x1 -= 20
                x2 -= 20

                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = height / width if width != 0 else 0
                area_percent = area / (frame.shape[0] * frame.shape[1])

                if (area > min_area and 
                    aspect_ratio > aspect_ratio_threshold and 
                    area_percent > min_area_percent):
                    detections.append(([x1, y1, x2, y2], score, 'person'))

    # Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Dibujar resultados
    current_id = 1
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = current_id
        current_id += 1

        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Dibujar caja verde
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Etiqueta dinámica
        label_text = f"Persona {track_id}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_width, label_height = label_size

        offset = 10
        label_bg_color = (0, 255, 0)

        # Preferencia: abajo, arriba, derecha
        if y2 + offset + label_height < frame.shape[0]:
            label_origin = (x1, y2 + offset + label_height)
            label_box = (x1, y2 + offset, x1 + label_width + 10, y2 + offset + label_height + 10)
        elif y1 - offset - label_height > 0:
            label_origin = (x1, y1 - offset)
            label_box = (x1, y1 - offset - label_height - 10, x1 + label_width + 10, y1 - offset)
        elif x2 + offset + label_width < frame.shape[1]:
            label_origin = (x2 + offset, y1 + label_height)
            label_box = (x2 + offset, y1, x2 + offset + label_width + 10, y1 + label_height + 10)
        else:
            label_origin = (x1, y1 + label_height + 10)
            label_box = (x1, y1, x1 + label_width + 10, y1 + label_height + 10)

        cv2.rectangle(frame, (label_box[0], label_box[1]), (label_box[2], label_box[3]), label_bg_color, -1)
        cv2.putText(frame, label_text, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Mostrar frame
    cv2.imshow("Fijado Exitoso Personas v2.2 Mejor Lateral", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar
cap.release()
cv2.destroyAllWindows()
