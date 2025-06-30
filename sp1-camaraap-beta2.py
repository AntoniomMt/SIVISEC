import cv2
import os
import numpy as np
from ultralytics import YOLO

# Crear carpeta de salida si no existe
output_folder = "EN VIVO"
os.makedirs(output_folder, exist_ok=True)

# Modelo YOLOv8n
model = YOLO("yolov8n.pt")

# Parámetros
conf_threshold = 0.25
bottle_class_id = 39  # ID de botella en COCO
input_size = (1280, 720)

# Abrir cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_size[1])
fps = cap.get(cv2.CAP_PROP_FPS) or 30
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Variables de grabación
recording = False
video_writer = None
video_index = 1

def get_unique_filename():
    global video_index
    while True:
        filename = os.path.join(output_folder, f"video_live_{video_index}.mp4")
        if not os.path.exists(filename):
            return filename
        video_index += 1

print("Presiona 'r' para iniciar grabación, 's' para detener, 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=conf_threshold, iou=0.45, verbose=False)[0]
    people = []
    bottles = []

    # Clasificar detecciones
    for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        if int(cls) == 0:  # persona
            people.append((x1, y1, x2, y2))
        elif int(cls) == bottle_class_id:
            bottles.append((x1, y1, x2, y2))

    # Analizar personas y dibujar cajas
    for idx, (x1, y1, x2, y2) in enumerate(people, start=1):
        person_box = np.array([x1, y1, x2, y2])
        has_bottle = False

        for bx1, by1, bx2, by2 in bottles:
            bottle_box = np.array([bx1, by1, bx2, by2])
            # Checar si la botella está dentro de la caja de la persona (con margen)
            if (
                bx1 > x1 - 20 and by1 > y1 - 20 and
                bx2 < x2 + 20 and by2 < y2 + 20
            ):
                has_bottle = True
                break

        color = (0, 155, 255) if has_bottle else (0, 255, 0)  # Azul si tiene botella, verde si no
        label = f"Persona {idx}"

        thickness = max(2, int(0.005 * video_width))
        font_scale = max(0.5, 0.0008 * video_width)
        font_thickness = max(2, int(0.0015 * video_width))
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        label_width, label_height = label_size
        offset = 6

        # Etiqueta dinámica
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

    # PUNTO ROJO de grabación
    if recording:
        cv2.circle(frame, (video_width - 30, 30), 10, (0, 0, 255), -1)
        video_writer.write(frame)

    cv2.imshow("Cámara - sp1-camaraap-beta2", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r") and not recording:
        output_path = get_unique_filename()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 15.0, (video_width, video_height))  # fps reducido
        recording = True
        print(f"Grabación iniciada: {output_path}")

    elif key == ord("s") and recording:
        recording = False
        video_writer.release()
        print("Grabación detenida.")

    elif key == ord("q"):
        break

# Liberar
cap.release()
if recording:
    video_writer.release()
cv2.destroyAllWindows()
