from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

# Cargar modelo YOLOv5n
model = YOLO("yolov5n.pt")

# Inicializar DeepSort tracker
tracker = DeepSort(max_age=15, nms_max_overlap=0.6)

# Configuración de cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Configuración de inferencia
detection_width, detection_height = 416, 234
frame_count = 0
DETECTION_INTERVAL = 3
min_area = 12000
aspect_ratio_threshold = 0.7
min_area_percent = 0.08

# Historial
detections = []
track_history = {}
bbox_history = {}
SMOOTHING_FACTOR = 0.25
PREDICTIVE_FACTOR = 0.2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_h, frame_w, _ = frame.shape
    frame_area = frame_h * frame_w

    if frame_count % DETECTION_INTERVAL == 0:
        small = cv2.resize(frame, (detection_width, detection_height))
        results = model(small, verbose=False)[0]

        scale_x = frame.shape[1] / detection_width
        scale_y = frame.shape[0] / detection_height

        detections = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0 and conf >= 0.4:
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                w = x2 - x1
                h = y2 - y1
                area = w * h
                aspect_ratio = h / w if w > 0 else 0
                area_percent = area / frame_area

                if area < min_area:
                    continue
                if aspect_ratio < aspect_ratio_threshold:
                    continue
                if area_percent < min_area_percent:
                    continue
                if x1 < 200 and y2 > frame.shape[0] - 100:
                    continue

                detections.append(([x1, y1, w, h], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)
    active_tracks = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id

        if track_id not in track_history:
            track_history[track_id] = {'frames': 1, 'bbox': track.to_ltrb()}
            bbox_history[track_id] = {'velocity': [0, 0, 0, 0]}
            continue
        else:
            track_history[track_id]['frames'] += 1

        if track_history[track_id]['frames'] < 3:
            continue

        l, t, r, b = track.to_ltrb()
        prev_box = track_history[track_id]['bbox']
        velocity = bbox_history[track_id]['velocity']

        velocity = [
            (l - prev_box[0]) * PREDICTIVE_FACTOR + velocity[0] * (1 - PREDICTIVE_FACTOR),
            (t - prev_box[1]) * PREDICTIVE_FACTOR + velocity[1] * (1 - PREDICTIVE_FACTOR),
            (r - prev_box[2]) * PREDICTIVE_FACTOR + velocity[2] * (1 - PREDICTIVE_FACTOR),
            (b - prev_box[3]) * PREDICTIVE_FACTOR + velocity[3] * (1 - PREDICTIVE_FACTOR),
        ]

        bbox_history[track_id]['velocity'] = velocity

        new_box = [
            prev_box[0] * (1 - SMOOTHING_FACTOR) + (l + velocity[0]) * SMOOTHING_FACTOR,
            prev_box[1] * (1 - SMOOTHING_FACTOR) + (t + velocity[1]) * SMOOTHING_FACTOR,
            prev_box[2] * (1 - SMOOTHING_FACTOR) + (r + velocity[2]) * SMOOTHING_FACTOR,
            prev_box[3] * (1 - SMOOTHING_FACTOR) + (b + velocity[3]) * SMOOTHING_FACTOR,
        ]

        track_history[track_id]['bbox'] = new_box
        active_tracks.append((track_id, new_box))

    active_tracks = sorted(active_tracks, key=lambda x: x[0])

    for display_id, (track_id, bbox) in enumerate(active_tracks, start=1):
        x1, y1, x2, y2 = map(int, bbox)

        # Dibujar caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Etiqueta dinámica
        label = f"Persona {display_id}"
        tag_width = 150
        tag_height = 30

        tag_x = x1
        tag_y = y2

        if tag_y + tag_height <= frame_h:
            # Etiqueta abajo
            cv2.rectangle(frame, (tag_x, tag_y), (tag_x + tag_width, tag_y + tag_height), (0, 255, 0), -1)
            cv2.putText(frame, label, (tag_x + 5, tag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        elif y1 - tag_height >= 0:
            # Etiqueta arriba
            tag_y = y1 - tag_height
            cv2.rectangle(frame, (tag_x, tag_y), (tag_x + tag_width, tag_y + tag_height), (0, 255, 0), -1)
            cv2.putText(frame, label, (tag_x + 5, tag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            # Etiqueta a un lado (centrada)
            tag_y = y1 + (y2 - y1) // 2 - tag_height // 2

            if x1 - tag_width >= 0:
                tag_x = x1 - tag_width
            elif x2 + tag_width <= frame_w:
                tag_x = x2
            else:
                tag_width = 100
                tag_x = max(0, x1 - tag_width)

            tag_y = max(0, min(frame_h - tag_height, tag_y))
            cv2.rectangle(frame, (tag_x, tag_y), (tag_x + tag_width, tag_y + tag_height), (0, 255, 0), -1)
            cv2.putText(frame, label, (tag_x + 5, tag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Contador de personas
    cv2.rectangle(frame, (10, 10), (270, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas detectadas: {len(active_tracks)}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.namedWindow("Sistema de Detección", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sistema de Detección", 1280, 720)
    cv2.imshow("Sistema de Detección", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
