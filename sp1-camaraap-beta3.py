import cv2
import numpy as np
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from supervision import ByteTrack, Detections

# Crear carpeta de salida
output_folder = "EN VIVO"
os.makedirs(output_folder, exist_ok=True)

# Inicializar modelos
model = YOLO("yolov8n.pt")
deepsort = DeepSort(max_age=30)
bytetrack = ByteTrack()

# Parámetros de detección
conf_threshold = 0.2
min_area = 6000
aspect_ratio_threshold = 0.35
min_area_percent = 0.015

# Inicializar cámara
cap = cv2.VideoCapture(0)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# VideoWriter dinámico
recording = False
video_writer = None
video_index = 1

def get_next_filename():
    global video_index
    base = "video_live"
    while True:
        path = os.path.join(output_folder, f"{base}_{video_index}.mp4")
        if not os.path.exists(path):
            return path
        video_index += 1

print("Presiona 'r' para grabar, 's' para detener, 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (1280, 720))
    scale_x = video_width / 1280
    scale_y = video_height / 720

    results = model(frame_resized, conf=conf_threshold, iou=0.45, verbose=False)
    detections = []
    confidences = []
    class_ids = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(boxes, scores, ids):
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
                if class_id == 0:  # persona
                    detections.append([x1, y1, x2, y2])
                    confidences.append(score)
                    class_ids.append(class_id)

    final_tracks = []
    if detections:
        np_boxes = np.array(detections, dtype=np.float32)
        np_scores = np.array(confidences, dtype=np.float32)
        dets = Detections(xyxy=np_boxes, confidence=np_scores)
        tracks = bytetrack.update_with_detections(dets)

        # Dummy detections para DeepSORT
        dummy_dets = [[[track[0][0], track[0][1], track[0][2], track[0][3]], 0.9, 0] for track in tracks]
        deepsort_tracks = deepsort.update_tracks(dummy_dets, frame=frame)

        for track, ds_track in zip(tracks, deepsort_tracks):
            if not ds_track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track[0])
            track_id = ds_track.track_id
            color = (0, 255, 0)  # verde por default

            # Verificar si hay una botella cerca de la persona
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy().astype(int)
                for b, cid in zip(boxes, cls):
                    if cid == 39:  # botella
                        bx1, by1, bx2, by2 = b * np.array([scale_x, scale_y, scale_x, scale_y])
                        iou_x1 = max(x1, bx1)
                        iou_y1 = max(y1, by1)
                        iou_x2 = min(x2, bx2)
                        iou_y2 = min(y2, by2)
                        inter_area = max(0, iou_x2 - iou_x1) * max(0, iou_y2 - iou_y1)
                        if inter_area > 0:
                            color = (173, 216, 230)  # azul claro

            # Etiqueta
            label = f"Persona {track_id}"
            thickness = max(2, int(0.005 * video_width))
            font_scale = max(0.4, 0.0007 * video_width)
            font_thickness = max(2, int(0.0015 * video_width))
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_width, label_height = label_size
            offset = 5

            # Posición dinámica de la etiqueta
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.rectangle(frame, (label_box[0], label_box[1]), (label_box[2], label_box[3]), color, -1)
            cv2.putText(frame, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

    # Punto rojo al grabar
    if recording:
        cv2.circle(frame, (video_width - 30, 30), 10, (0, 0, 255), -1)
        video_writer.write(frame)

    cv2.imshow("Camara", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r') and not recording:
        output_path = get_next_filename()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps / 1.5, (video_width, video_height))  # ralentizado
        recording = True
        print(f"Grabando: {output_path}")

    elif key == ord('s') and recording:
        recording = False
        video_writer.release()
        print("Grabación detenida.")

    elif key == ord('q'):
        break

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
