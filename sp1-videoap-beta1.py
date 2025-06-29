import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Ruta del video a analizar
video_path = "VIDEOS DE STOCK/video1.mp4"

# Cargar modelo YOLOv5n
model = YOLO("yolov5n.pt")

# Inicializar DeepSort tracker
tracker = DeepSort(max_age=15)

# Parámetros para filtrar falsos positivos
min_area = 10000
aspect_ratio_threshold = 0.3
min_area_percent = 0.05

# Abrir video
cap = cv2.VideoCapture(video_path)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Mostrar ventana adaptable a resolución del video
cv2.namedWindow("sp1-videoap", cv2.WINDOW_NORMAL)
cv2.resizeWindow("sp1-videoap", video_width, video_height)
# (Opcional: pantalla completa)
# cv2.setWindowProperty("sp1-videoap", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Procesamiento frame a frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Padding virtual en laterales
    padded_frame = cv2.copyMakeBorder(frame, 0, 0, 20, 20, cv2.BORDER_REPLICATE)

    # Detección con YOLO
    results = model(padded_frame, verbose=False, conf=0.3)

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(boxes, scores, class_ids):
            if class_id == 0 and score > 0.3:
                x1, y1, x2, y2 = box
                x1 -= 20  # compensar padding
                x2 -= 20

                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = height / width if width != 0 else 0
                area_percent = area / (video_width * video_height)

                if area > min_area and aspect_ratio > aspect_ratio_threshold and area_percent > min_area_percent:
                    detections.append(([x1, y1, x2, y2], score, 'person'))

    # Seguimiento con DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)
    current_id = 1

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = current_id
        current_id += 1

        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Dibujar mark box verde
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Etiqueta dinámica
        label_text = f"Persona {track_id}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_width, label_height = label_size
        offset = 10

        # Posición dinámica: abajo > arriba > derecha > sobre la caja
        if y2 + offset + label_height < video_height:
            label_origin = (x1, y2 + offset + label_height)
            label_box = (x1, y2 + offset, x1 + label_width + 10, y2 + offset + label_height + 10)
        elif y1 - offset - label_height > 0:
            label_origin = (x1, y1 - offset)
            label_box = (x1, y1 - offset - label_height - 10, x1 + label_width + 10, y1 - offset)
        elif x2 + offset + label_width < video_width:
            label_origin = (x2 + offset, y1 + label_height)
            label_box = (x2 + offset, y1, x2 + offset + label_width + 10, y1 + label_height + 10)
        else:
            label_origin = (x1, y1 + label_height + 10)
            label_box = (x1, y1, x1 + label_width + 10, y1 + label_height + 10)

        cv2.rectangle(frame, (label_box[0], label_box[1]), (label_box[2], label_box[3]), (0, 255, 0), -1)
        cv2.putText(frame, label_text, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Mostrar frame completo en la ventana
    cv2.imshow("sp1-videoap", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar
cap.release()
cv2.destroyAllWindows()
