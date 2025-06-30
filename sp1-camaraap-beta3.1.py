import cv2
import numpy as np
import os
from ultralytics import YOLO
from supervision import ByteTrack, Detections
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==== CONFIGURACIÓN ====
model = YOLO("yolov8n.pt")
tracker = ByteTrack()
deepsort = DeepSort(max_age=30)

cap = cv2.VideoCapture(0)  # Usa cámara
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ==== Carpeta para guardar ====
output_folder = "EN VIVO"
os.makedirs(output_folder, exist_ok=True)

# ==== Grabación ====
recording = False
video_writer = None
video_counter = 1

def get_next_filename():
    global video_counter
    base_name = "video_live"
    ext = ".mp4"
    filename = f"{base_name}_{video_counter}{ext}"
    while os.path.exists(os.path.join(output_folder, filename)):
        video_counter += 1
        filename = f"{base_name}_{video_counter}{ext}"
    return os.path.join(output_folder, filename)

print("Presiona 'r' para grabar, 's' para detener, 'q' para salir")

# ==== Etiquetado persistente ====
active_ids = {}
id_to_number = {}
available_numbers = list(range(1, 100))
max_missed_frames = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3, iou=0.45, verbose=False)
    detections = []
    confidences = []

    for result in results:
        for box, score, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if int(class_id) == 0:  # Persona
                x1, y1, x2, y2 = box.cpu().numpy()
                detections.append([x1, y1, x2, y2])
                confidences.append(score.cpu().item())

    # YOLO detections to ByteTrack
    if detections:
        dets = Detections(xyxy=np.array(detections, dtype=np.float32), confidence=np.array(confidences, dtype=np.float32))
        tracks = tracker.update_with_detections(dets)

        # ==== DeepSort solo para etiquetas ====
        dummy_dets = [[[x1, y1, x2, y2], 0.9, 0] for (x1, y1, x2, y2), *_ in tracks]
        deepsort_tracks = deepsort.update_tracks(dummy_dets, frame=frame)

        new_ids = set()

        for track in deepsort_tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            new_ids.add(track_id)

            if track_id not in id_to_number:
                if available_numbers:
                    id_to_number[track_id] = available_numbers.pop(0)
                    active_ids[track_id] = 0

            else:
                active_ids[track_id] = 0  # Reinicia contador si sigue activo

            num = id_to_number[track_id]
            tlbr = track.to_tlbr()
            x1, y1, x2, y2 = int(tlbr[1]), int(tlbr[0]), int(tlbr[3]), int(tlbr[2])


            # Dibujo
            label = f"Persona {num}"
            thickness = max(2, int(0.005 * video_width))
            font_scale = max(0.4, 0.0007 * video_width)
            font_thickness = max(2, int(0.0015 * video_width))  # más grueso
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

            # Dibujar
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.rectangle(frame, (label_box[0], label_box[1]), (label_box[2], label_box[3]), (0, 255, 0), -1)
            cv2.putText(frame, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        # Actualiza contador de IDs inactivos
        for tid in list(active_ids.keys()):
            if tid not in new_ids:
                active_ids[tid] += 1
                if active_ids[tid] > max_missed_frames:
                    if tid in id_to_number:
                        available_numbers.append(id_to_number[tid])
                        available_numbers.sort()
                        del id_to_number[tid]
                    del active_ids[tid]

    # Mostrar punto rojo si está grabando
    if recording:
        cv2.circle(frame, (video_width - 30, 30), 10, (0, 0, 255), -1)
        video_writer.write(frame)

    cv2.imshow("sp1-camaraap-beta3.1", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("r") and not recording:
        output_path = get_next_filename()
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (video_width, video_height))
        recording = True
        print("Grabación iniciada.")
    elif key == ord("s") and recording:
        recording = False
        video_writer.release()
        print("Grabación detenida.")

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
