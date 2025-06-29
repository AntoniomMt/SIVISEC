import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

video_path = "VIDEOS DE STOCK/video1.mp4"
yolo_interval = 10

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=8, n_init=6, nn_budget=50)

conf_threshold = 0.3
min_area = 7000
min_area_percent = 0.02
max_area_percent = 0.3
aspect_ratio_threshold = 0.4
max_aspect_ratio = 5.0
iou_overlap_threshold = 0.6

cap = cv2.VideoCapture(video_path)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

cv2.namedWindow("sp1-videoap", cv2.WINDOW_NORMAL)
cv2.resizeWindow("sp1-videoap", video_width, video_height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_ultra_v8_v2.mp4', fourcc, fps, (video_width, video_height))

frame_count = 0
last_detections = []

def calculate_iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = (xa2 - xa1) * (ya2 - ya1)
    areaB = (xb2 - xb1) * (yb2 - yb1)
    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    detections = []

    if frame_count % yolo_interval == 0:
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        candidates = []
        areas = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            if class_id == 0 and score > conf_threshold:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                area = w * h
                area_percent = area / (video_width * video_height)
                aspect_ratio = h / w if w != 0 else 0

                if (
                    area > min_area and
                    min_area_percent < area_percent < max_area_percent and
                    aspect_ratio_threshold < aspect_ratio < max_aspect_ratio
                ):
                    candidates.append(([x1, y1, x2, y2], score, 'person'))
                    areas.append(area)

        avg_area = np.mean(areas) if areas else 1

        # Eliminar cajas mucho más grandes que las otras (más de 2.5x)
        strict_filtered = []
        for box, score, label in candidates:
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < 2.5 * avg_area:
                strict_filtered.append((box, score, label))

        # Eliminar cajas que engloban otras
        no_englobadoras = []
        for i, (boxA, scoreA, labelA) in enumerate(strict_filtered):
            xa1, ya1, xa2, ya2 = boxA
            contains = False
            for j, (boxB, _, _) in enumerate(strict_filtered):
                if i == j:
                    continue
                xb1, yb1, xb2, yb2 = boxB
                if xa1 <= xb1 and ya1 <= yb1 and xa2 >= xb2 and ya2 >= yb2:
                    contains = True
                    break
            if not contains:
                no_englobadoras.append((boxA, scoreA, labelA))

        # Filtro final por múltiples solapamientos
        final = []
        for i, (boxA, scoreA, labelA) in enumerate(no_englobadoras):
            overlaps = 0
            for j, (boxB, _, _) in enumerate(no_englobadoras):
                if i != j and calculate_iou(boxA, boxB) > iou_overlap_threshold:
                    overlaps += 1
            if overlaps <= 2:
                final.append((boxA, scoreA, labelA))

        last_detections = final

    tracks = tracker.update_tracks(last_detections, frame=frame)
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
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
