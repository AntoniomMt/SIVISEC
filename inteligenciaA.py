import cv2
import numpy as np
from ultralytics import YOLO
import time

# InteligenciaA.py
# --- Detección de anomalías por comportamiento ---
# Pipeline: detección -> tracking -> extracción de features -> buffer temporal -> anomaly score

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Parámetros para la anomalía
WINDOW = 30  # frames para análisis
THRESHOLD = 2.5

buffer_personas = {}
next_id = 1


def centro(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    personas = []

    for box in results.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            personas.append((x1, y1, x2, y2))

    nuevas_personas = {}

    for p in personas:
        c = centro(p)
        pid = None

        # match con persona previa
        for old_id, data in buffer_personas.items():
            if dist(c, data["centro_actual"]) < 80:
                pid = old_id
                break

        if pid is None:
            pid = next_id
            next_id += 1

        if pid not in buffer_personas:
            buffer_personas[pid] = {
                "centros": [],
                "vels": [],
                "centro_actual": c,
            }

        # actualizar
        data = buffer_personas[pid]
        prev = data["centro_actual"]
        velocidad = dist(c, prev)

        data["centros"].append(c)
        data["vels"].append(velocidad)
        data["centro_actual"] = c

        # mantener ventana
        if len(data["vels"]) > WINDOW:
            data["vels"].pop(0)
            data["centros"].pop(0)

        # calcular anomalía
        if len(data["vels"]) >= 10:
            media = np.mean(data["vels"])
            std = np.std(data["vels"]) + 1e-6
            zscore = (velocidad - media) / std
        else:
            zscore = 0

        # dibujar
        color = (0, 255, 0)
        label = f"ID {pid}"

        if zscore > THRESHOLD:
            color = (0, 0, 255)
            label = f"ANOMALIA ID {pid}"

        x1, y1, x2, y2 = p
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        nuevas_personas[pid] = buffer_personas[pid]

    buffer_personas = nuevas_personas

    cv2.imshow("Inteligencia-A (Anomaly Detection)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
