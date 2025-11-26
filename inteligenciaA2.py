import cv2
import numpy as np
from ultralytics import YOLO
import time

# InteligenciaA2.py
# Anomaly Detection + Object Interaction Anomaly (no reglas explícitas)
# Se mantiene el principio de "inteligencia real":
# no ifs manuales del tipo "si botella entonces X".
# En cambio, se modela: co-movimiento, proximidad dinámica y correlación temporal.

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

WINDOW = 30
THR_Z = 2.5
THR_OBJ = 2.0

buffer = {}
next_id = 1

OBJ_CLASSES = [39]  # botellas, cups, libros, bolsas, etc.


def centro(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    personas = []
    objetos = []

    # clasificación mínima (solo botellas: clase 39)
    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 0:
            personas.append((x1, y1, x2, y2))
        elif cls in OBJ_CLASSES:
            objetos.append((cls, (x1, y1, x2, y2)))

    nuevos = {}

    for p in personas:
        c = centro(p)
        pid = None

        for old_id, data in buffer.items():
            if dist(c, data["centro"] ) < 80:
                pid = old_id
                break

        if pid is None:
            pid = next_id
            next_id += 1

        if pid not in buffer:
            buffer[pid] = {
                "centros": [],
                "vel": [],
                "prox_obj": [],
                "centro": c
            }

        data = buffer[pid]
        prev = data["centro"]
        v = dist(c, prev)
        data["centro"] = c
        data["centros"].append(c)
        data["vel"].append(v)

        # proximidad media a objetos
        if objetos:
            dists = []
            for cls, ob in objetos:
                dists.append(dist(c, centro(ob)))
            p_obj = min(dists)
        else:
            p_obj = 9999

        data["prox_obj"].append(p_obj)

        if len(data["vel"]) > WINDOW:
            data["vel"].pop(0)
            data["centros"].pop(0)
            data["prox_obj"].pop(0)

        # Anomaly Score movimiento
        if len(data["vel"]) >= 10:
            m = np.mean(data["vel"])
            s = np.std(data["vel"]) + 1e-6
            Z_m = (v - m) / s
        else:
            Z_m = 0

        # Anomaly Score interacción objetos
        if len(data["prox_obj"]) >= 10:
            m2 = np.mean(data["prox_obj"])
            s2 = np.std(data["prox_obj"]) + 1e-6
            Z_o = (m2 - p_obj) / s2
        else:
            Z_o = 0

        score = Z_m + Z_o

        color = (0, 255, 0)
        label = f"ID {pid}"

        if score > (THR_Z + THR_OBJ):
            color = (0, 0, 255)
            label = f"ANOMALIA ID {pid}"

        x1, y1, x2, y2 = p
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        nuevos[pid] = buffer[pid]

    buffer = nuevos

    cv2.imshow("Inteligencia-A2", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
