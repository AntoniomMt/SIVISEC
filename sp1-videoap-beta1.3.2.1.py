import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Ruta del video
video_path = "VIDEOS DE STOCK/video1.mp4"

# Modelo YOLOv5s
model = YOLO("yolov5s.pt")

# DeepSORT ajustado
tracker = DeepSort(
    max_age=8,
    n_init=6,
    nn_budget=50
)

# Sensibilidad más estricta
conf_threshold = 0.25
min_area = 6000
aspect_ratio_threshold = 0.35
min_area_percent = 0.015

# Captura de video
cap = cv2.VideoCapture(video_path)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("sp1-videoap", cv2.WINDOW_NORMAL)
cv2.resizeWindow("sp1-videoap", video_width, video_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize para detección precisa
    frame_small = cv2.resize(frame, (2560, 1440))
    results = model(frame_small, conf=conf_threshold, iou=0.45, verbose=False)

    scale_x = video_width / 2560
    scale_y = video_height / 1440

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(boxes, scores, class_ids):
            if class_id == 0 and score > conf_threshold:
                x1, y1, x2, y2 = box
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y

                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = height / width if width != 0 else 0
                area_percent = area / (video_width * video_height)

                if area > min_area and aspect_ratio > aspect_ratio_threshold and area_percent > min_area_percent:
                    detections.append(([x1, y1, x2, y2], score, 'person'))

    # Anti-cajas que engloban otras
    filtered_detections = []
    for i, (boxA, scoreA, labelA) in enumerate(detections):
        xa1, ya1, xa2, ya2 = boxA
        contains_other = False
        for j, (boxB, _, _) in enumerate(detections):
            if i == j:
                continue
            xb1, yb1, xb2, yb2 = boxB
            if xa1 <= xb1 and ya1 <= yb1 and xa2 >= xb2 and ya2 >= yb2:
                contains_other = True
                break
        if not contains_other:
            filtered_detections.append((boxA, scoreA, labelA))

    # Seguimiento
    tracks = tracker.update_tracks(filtered_detections, frame=frame)
    current_id = 1

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = current_id
        current_id += 1
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Caja verde
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Etiqueta dinámica
        label_text = f"Persona {track_id}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_width, label_height = label_size
        offset = 10

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

    cv2.imshow("sp1-videoap", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
