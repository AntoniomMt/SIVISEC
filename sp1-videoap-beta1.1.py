import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

video_path = "VIDEOS DE STOCK/video1.mp4"
model = YOLO("yolov5n.pt")
tracker = DeepSort(max_age=15)

# Sensibilidad y rendimiento ajustados
conf_threshold = 0.2
min_area = 3000
aspect_ratio_threshold = 0.2
min_area_percent = 0.01

cap = cv2.VideoCapture(video_path)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("sp1-videoap", cv2.WINDOW_NORMAL)
cv2.resizeWindow("sp1-videoap", video_width, video_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Escalamos para procesar más rápido
    frame_small = cv2.resize(frame, (1280, 720))
    results = model(frame_small, verbose=False, conf=conf_threshold)

    scale_x = video_width / 1280
    scale_y = video_height / 720

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

    # Tracking
    tracks = tracker.update_tracks(detections, frame=frame)
    current_id = 1

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = current_id
        current_id += 1
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

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
