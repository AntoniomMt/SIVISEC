import cv2
import numpy as np
import os
from ultralytics import YOLO
from supervision import ByteTrack, Detections

# Rutas
video_path = "VIDEOS DE STOCK/video1.mp4"
output_folder = "RESULTADOS VIDEO BYTETRACK"
os.makedirs(output_folder, exist_ok=True)

# Generar nombre de archivo no duplicado
base_name = "video_bytetrack"
ext = ".mp4"
output_path = os.path.join(output_folder, base_name + ext)
counter = 1
while os.path.exists(output_path):
    output_path = os.path.join(output_folder, f"{base_name}_{counter}{ext}")
    counter += 1

# Modelo YOLOv8n
model = YOLO("yolov8n.pt")

# Parámetros
conf_threshold = 0.3
min_area = 6000
aspect_ratio_threshold = 0.35
min_area_percent = 0.015
input_size = (2560, 1440)

# Captura de video
cap = cv2.VideoCapture(video_path)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Salida de video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))

# Tracker ByteTrack
tracker = ByteTrack()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, input_size)
    scale_x = video_width / input_size[0]
    scale_y = video_height / input_size[1]

    results = model(resized, conf=conf_threshold, iou=0.45, verbose=False)
    detections = []
    confidences = []

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
                    detections.append([x1, y1, x2, y2])
                    confidences.append(score)

    if detections:
        np_boxes = np.array(detections, dtype=np.float32)
        np_scores = np.array(confidences, dtype=np.float32)
        dets = Detections(xyxy=np_boxes, confidence=np_scores)
        tracks = tracker.update_with_detections(dets)

        for i, track in enumerate(tracks, start=1):
            x1, y1, x2, y2 = map(int, track[0])
            label = f"Persona {i}"

            # Parámetros visuales
            thickness = max(2, int(0.005 * video_width))
            font_scale = max(0.4, 0.0007 * video_width)
            font_thickness = max(1, int(0.0012 * video_width))
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_width, label_height = label_size
            offset = 5  # MÁS PEGADA LA ETIQUETA

            # Posición dinámica del texto
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.rectangle(frame, (label_box[0], label_box[1]), (label_box[2], label_box[3]), (0, 255, 0), -1)
            cv2.putText(frame, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

    out.write(frame)

cap.release()
out.release()
print(f"\n Video procesado y guardado en: {output_path}")