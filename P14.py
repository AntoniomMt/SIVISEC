from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Cargar YOLOv5 nano
model = YOLO("yolov5n.pt")

# Inicializar DeepSORT
tracker = DeepSort(max_age=15, nms_max_overlap=0.6)

# Captura de cámara (resolución HD)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Resolución de procesamiento (mantén más baja para velocidad)
detection_width, detection_height = 416, 234
frame_count = 0
DETECTION_INTERVAL = 3
min_area = 10000  # Filtrado de cajas pequeñas

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
                area = w * h
                if area < min_area:
                    continue
                detections.append(([x1, y1, w, h], conf, "person"))

    # Seguimiento
    tracks = tracker.update_tracks(detections, frame=frame)

    # Reasignar números consistentes por cuadro
    active_tracks = [track for track in tracks if track.is_confirmed()]
    active_tracks = sorted(active_tracks, key=lambda t: t.track_id)

    for display_id, track in enumerate(active_tracks, start=1):
        l, t, r, b = track.to_ltrb()
        x1, y1, x2, y2 = int(l), int(t), int(r), int(b)

        # Caja verde
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Etiqueta tipo folder abajo
        label = f"Persona {display_id}"
        tag_width = 150
        tag_height = 30
        tag_x = x1
        tag_y = y2
        cv2.rectangle(frame, (tag_x, tag_y), (tag_x + tag_width, tag_y + tag_height), (0, 255, 0), -1)
        cv2.putText(frame, label, (tag_x + 5, tag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Mostrar conteo
    total = len(active_tracks)
    cv2.rectangle(frame, (10, 10), (270, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas detectadas: {total}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Mostrar frame
    cv2.namedWindow("Sistema Pulido", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sistema Pulido", 1280, 720)
    cv2.imshow("Sistema Pulido", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
