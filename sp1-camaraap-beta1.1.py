import cv2
import numpy as np
import os
from ultralytics import YOLO
from supervision import ByteTrack, Detections

# Inicializar cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
fps = cap.get(cv2.CAP_PROP_FPS)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# YOLOv8n
model = YOLO("yolov8n.pt")

# Tracker
tracker = ByteTrack()

# Parámetros detección
conf_threshold = 0.02
min_area = 6000
aspect_ratio_threshold = 0.35
min_area_percent = 0.015
input_size = (1280, 720)

# Grabación
recording = False
video_writer = None
live_folder = "EN VIVO"
os.makedirs(live_folder, exist_ok=True)

print("Presiona 'q' para salir, 'r' para grabar, 's' para detener grabación.")

while True:
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

            # Estética
            thickness = max(2, int(0.005 * video_width))
            font_scale = max(0.4, 0.0007 * video_width)
            font_thickness = max(1, int(0.0012 * video_width))
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_width, label_height = label_size
            offset = 5

            # Posición dinámica
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

    # Grabación activa
    if recording and video_writer:
        video_writer.write(frame)

    cv2.imshow("VideoAp Realtime", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r') and not recording:
        # Crear nombre único
        base_name = "video_live"
        ext = ".mp4"
        index = 1
        output_path = os.path.join(live_folder, f"{base_name}_{index}{ext}")
        while os.path.exists(output_path):
            index += 1
            output_path = os.path.join(live_folder, f"{base_name}_{index}{ext}")
        # Inicializar grabación
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
        recording = True
        print(f"Grabación iniciada: {output_path}")
    elif key == ord('s') and recording:
        recording = False
        video_writer.release()
        video_writer = None
        print("Grabación detenida.")

# Limpieza
if video_writer:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()
