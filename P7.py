import cv2
import torch
import numpy as np
import time

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5

# Interpolación lineal
def lerp(a, b, t):
    return a + (b - a) * t

# Personas rastreadas
tracked_people = {}
next_person_id = 0
alpha = 0.2  # Suavizado

# Umbral de desaparición
max_missing_frames = 15

# Cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    # Inferencia
    results = model(frame)
    detections = results.xyxy[0]

    current_people = []

    # Detectar personas
    for *box, conf, cls in detections:
        if int(cls.item()) == 0:  # Persona
            x1, y1, x2, y2 = map(int, box)
            margin = 10
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(frame_width, x2 + margin), min(frame_height, y2 + margin)
            current_people.append([x1, y1, x2, y2])

    # Asociación simple por cercanía (puede mejorarse con IoU)
    assigned_ids = set()
    for person_box in current_people:
        best_id = None
        min_dist = float('inf')

        for pid, data in tracked_people.items():
            if pid in assigned_ids:
                continue
            prev_box = data['box']
            dist = sum(abs(a - b) for a, b in zip(person_box, prev_box))
            if dist < min_dist and dist < 250:  # Umbral para match
                best_id = pid
                min_dist = dist

        if best_id is not None:
            # Interpolación suave
            prev_box = tracked_people[best_id]['box']
            smoothed_box = [
                int(lerp(prev_box[0], person_box[0], alpha)),
                int(lerp(prev_box[1], person_box[1], alpha)),
                int(lerp(prev_box[2], person_box[2], alpha)),
                int(lerp(prev_box[3], person_box[3], alpha)),
            ]
            tracked_people[best_id]['box'] = smoothed_box
            tracked_people[best_id]['missing'] = 0
            assigned_ids.add(best_id)
        else:
            # Nueva persona
            tracked_people[next_person_id] = {
                'box': person_box,
                'missing': 0
            }
            assigned_ids.add(next_person_id)
            next_person_id += 1

    # Marcar desaparecidos
    for pid in list(tracked_people.keys()):
        if pid not in assigned_ids:
            tracked_people[pid]['missing'] += 1
            if tracked_people[pid]['missing'] > max_missing_frames:
                del tracked_people[pid]

    # Dibujar cajas
    for pid, data in tracked_people.items():
        x1, y1, x2, y2 = data['box']

        # Caja verde
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        label = "persona"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_w, label_h = label_size

        tag_margin = 5
        tag_x = x1
        tag_y = y2 + label_h + tag_margin

        if tag_y + label_h > frame_height:
            tag_y = y1 - tag_margin

        tag_rect_top_left = (tag_x, tag_y - label_h - tag_margin)
        tag_rect_bottom_right = (tag_x + label_w + 10, tag_y)

        cv2.rectangle(frame, tag_rect_top_left, tag_rect_bottom_right, (0, 255, 0), -1)
        cv2.putText(frame, label, (tag_x + 5, tag_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Mostrar frame
    cv2.imshow("YOLOv5 + Interpolación + MultiPersona", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
