from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Cargar modelo YOLOv5
model = YOLO("yolov5n.pt")

# Inicializar tracker DeepSORT
tracker = DeepSort(max_age=30)

# Cámara en alta resolución
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Resolución reducida solo para detección
detection_width, detection_height = 640, 360

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame reducido para detección
    small = cv2.resize(frame, (detection_width, detection_height))
    results = model(small, verbose=False)[0]

    scale_x = frame.shape[1] / detection_width
    scale_y = frame.shape[0] / detection_height

    detections = []

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box

        if int(cls) == 0:  # Solo clase persona
            # Reescalar coordenadas
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            bbox = [x1, y1, x2 - x1, y2 - y1]  # formato [x, y, w, h]
            detections.append((bbox, conf, "person"))

    # Aplicar seguimiento
    tracks = tracker.update_tracks(detections, frame=frame)

    count = 0
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        x1, y1, x2, y2 = int(l), int(t), int(w), int(h)

        # Dibujar caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Dibujar etiqueta tipo folder abajo
        label = f"Persona {track_id}"
        tag_width = 150
        tag_height = 30
        tag_x = x1
        tag_y = y2
        cv2.rectangle(frame, (tag_x, tag_y), (tag_x + tag_width, tag_y + tag_height), (0, 255, 0), -1)
        cv2.putText(frame, label, (tag_x + 5, tag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        count += 1

    # Mostrar total arriba a la izquierda
    cv2.rectangle(frame, (10, 10), (220, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas detectadas: {count}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Mostrar frame
    cv2.imshow("YOLOv5 + DeepSORT - Conteo con seguimiento", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
