from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

# Modelo
model = YOLO("yolov5n.pt")

# Tracker
tracker = DeepSort(max_age=15, nms_max_overlap=0.6)

# Cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detection_width, detection_height = 416, 234
frame_count = 0
DETECTION_INTERVAL = 3
min_area = 10000
confidence_threshold = 0.4

detections = []
track_history = {}  # PARA IGNORAR IDs fugaces
SMOOTHING_FACTOR = 0.2  # PARA SUAVIZAR cajas

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % DETECTION_INTERVAL == 0:
        small = cv2.resize(frame, (detection_width, detection_height))
        results = model(small, verbose=False)[0]

        scale_x = frame.shape[1] / detection_width
        scale_y = frame.shape[0] / detection_height

        detections = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0 and conf >= confidence_threshold:
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                w = x2 - x1
                h = y2 - y1
                area = w * h
                if area < min_area:
                    continue
                if x1 < 200 and y2 > frame.shape[0] - 100:
                    continue

                detections.append(([x1, y1, w, h], conf, "person"))

    # Seguimiento
    tracks = tracker.update_tracks(detections, frame=frame)
    active_tracks = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id

        # FILTRAR falsos positivos que solo aparecen 1-2 frames
        if track_id not in track_history:
            track_history[track_id] = {'frames': 1, 'bbox': track.to_ltrb()}
            continue
        else:
            track_history[track_id]['frames'] += 1

        if track_history[track_id]['frames'] < 3:
            continue  # IGNORAMOS si no tiene historia suficiente

        l, t, r, b = track.to_ltrb()

        # SMOOTHING de cajas
        prev_box = track_history[track_id]['bbox']
        new_box = [
            prev_box[0] * (1 - SMOOTHING_FACTOR) + l * SMOOTHING_FACTOR,
            prev_box[1] * (1 - SMOOTHING_FACTOR) + t * SMOOTHING_FACTOR,
            prev_box[2] * (1 - SMOOTHING_FACTOR) + r * SMOOTHING_FACTOR,
            prev_box[3] * (1 - SMOOTHING_FACTOR) + b * SMOOTHING_FACTOR,
        ]
        track_history[track_id]['bbox'] = new_box
        active_tracks.append((track_id, new_box))

    # Ordenar por ID y dibujar
    active_tracks = sorted(active_tracks, key=lambda x: x[0])

    for display_id, (track_id, bbox) in enumerate(active_tracks, start=1):
        x1, y1, x2, y2 = map(int, bbox)

        # Dibujar caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Etiquetado dinámico
        label = f"Persona {display_id}"
        tag_width = 150
        tag_height = 30
        frame_h, frame_w, _ = frame.shape

        tag_x = x1
        tag_y = y2
        if tag_y + tag_height > frame_h:
            tag_y = y1 - tag_height
            if tag_y < 0:
                tag_y = y1
                tag_x = x1 - tag_width
                if tag_x < 0:
                    tag_x = 0
                    tag_width = 100

        cv2.rectangle(frame, (tag_x, tag_y), (tag_x + tag_width, tag_y + tag_height), (0, 255, 0), -1)
        cv2.putText(frame, label, (tag_x + 5, tag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Contador
    cv2.rectangle(frame, (10, 10), (270, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas detectadas: {len(active_tracks)}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.namedWindow("Sistema Mejorado", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sistema Mejorado", 1280, 720)
    cv2.imshow("Sistema Mejorado", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
