import cv2
from ultralytics import YOLO
import numpy as np
import os

# Carpetas
input_folder = "FOTOS DE STOCK"
output_folder = "RESULTADOS FOTOS"

os.makedirs(output_folder, exist_ok=True)

# Cargar modelo
model = YOLO("yolov5su.pt")

# Parámetros de detección
conf_threshold = 0.25
min_area = 6000
aspect_ratio_threshold = 0.35
min_area_percent = 0.015

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

for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_folder, filename)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"No se pudo cargar la imagen {filename}")
        continue

    video_height, video_width = frame.shape[:2]

    # Redimensionar para YOLO
    frame_small = cv2.resize(frame, (2560, 1440))
    scale_x = video_width / 2560
    scale_y = video_height / 1440

    results = model(frame_small, conf=conf_threshold, iou=0.45, verbose=False)

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

    # Filtrar cajas que engloban otras
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

    # Filtrar cajas que invaden otras
    final_detections = []
    areas = [((box[0][2] - box[0][0]) * (box[0][3] - box[0][1])) for box in filtered_detections]
    avg_area = np.mean(areas) if areas else 1

    for i, (boxA, scoreA, labelA) in enumerate(filtered_detections):
        overlaps = 0
        for j, (boxB, _, _) in enumerate(filtered_detections):
            if i == j:
                continue
            if calculate_iou(boxA, boxB) > 0.5:
                overlaps += 1
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        if overlaps >= 2 and areaA > 1.5 * avg_area:
            continue
        final_detections.append((boxA, scoreA, labelA))

    # Dibujar las detecciones directamente, sin DeepSort
    for idx, (box, score, label) in enumerate(final_detections, start=1):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 12) #mover para el grueso  3 original
        label_text = f"Persona {idx} {score:.2f}"

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
        cv2.putText(frame, label_text, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8) #mover para el tamaño de letra 0.8, 2 original

    # Guardar imagen procesada
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, frame)
    print(f"{filename} procesada y guardada en {output_folder}")
