from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Cargar modelo YOLOv5 nano
model = YOLO("yolov5n.pt")

# Inicializar DeepSORT
tracker = DeepSort(max_age=15)  # menos edad para soltar rápido IDs

# Cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Resolución solo para detección
detection_width, detection_height = 416, 234
frame_count = 0
DETECTION_INTERVAL = 3

detections = []

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
            if int(cls) == 0:
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    # Enumerar personas visibles del 1 al N
    active_tracks = [track for track in tracks if track.is_confirmed()]
    active_tracks = sorted(active_tracks, key=lambda t: t.track_id)  # opcional: orden por ID

    for display_id, track in enumerate(active_tracks, start=1):
        l, t, r, b = track.to_ltrb()
        x1, y1, x2, y2 = int(l), int(t), int(r), int(b)

        # Dibujar caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Etiqueta tipo folder
        label = f"Persona {display_id}"
        tag_width = 150
        tag_height = 30
        tag_x = x1
        tag_y = y2
        cv2.rectangle(frame, (tag_x, tag_y), (tag_x + tag_width, tag_y + tag_height), (0, 255, 0), -1)
        cv2.putText(frame, label, (tag_x + 5, tag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Mostrar conteo
    total = len(active_tracks)
    cv2.rectangle(frame, (10, 10), (240, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas detectadas: {total}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.namedWindow("Renumeración Dinámica", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Renumeración Dinámica", 960, 540)
    cv2.imshow("Renumeración Dinámica", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
