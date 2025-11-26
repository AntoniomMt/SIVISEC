import cv2
import numpy as np
from ultralytics import YOLO

# inteligenciaA3.py
# Detección inteligente de "sostener botella" SIN reglas explícitas.
# - NO dibuja personas
# - SOLO dibuja botellas
#   - Verde = botella libre
#   - Azul  = botella sostenida
# "Inteligencia real": el sistema NO usa un if rígido "si mano toca botella".
# Se basa en co-movimiento: correlación dinámica entre trayectoria de botella y la región de movimiento humana.

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Parámetros
WINDOW = 10
CO_MOVE_THR = 0.65   # correlación mínima para asumir que la botella es sostenida

# Historial
buffer_botellas = {}
next_bid = 1


def centro(b):
    x1, y1, x2, y2 = b
    return int((x1+x2)/2), int((y1+y2)/2)


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# Estimación de movimiento humano basado en "optical flow denso"
opt_prev = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical Flow para detectar movimiento humano
    if opt_prev is None:
        opt_prev = gray
        flow_mag = np.zeros_like(gray, dtype=np.float32)
    else:
        flow = cv2.calcOpticalFlowFarneback(opt_prev, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mag = mag
        opt_prev = gray

    # Detección YOLO
    results = model(frame, verbose=False)[0]
    botellas = []

    for b in results.boxes:
        cls = int(b.cls[0])
        if cls == 39:  # solo botellas
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            botellas.append((x1, y1, x2, y2))

    nuevos = {}

    for box in botellas:
        cx, cy = centro(box)
        bid = None

        for oid, data in buffer_botellas.items():
            if dist((cx, cy), data["centro"]) < 40:
                bid = oid
                break

        if bid is None:
            bid = next_bid
            next_bid += 1

        if bid not in buffer_botellas:
            buffer_botellas[bid] = {
                "centro": (cx, cy),
                "tray": [],
                "mov_h": [],
                "estado": "libre"
            }

        data = buffer_botellas[bid]
        prev_c = data["centro"]
        data["centro"] = (cx, cy)

        # trayectoria botella
        mov_b = dist((cx, cy), prev_c)
        data["tray"].append(mov_b)
        if len(data["tray"]) > WINDOW:
            data["tray"].pop(0)

        # movimiento humano alrededor: promedio de optical flow local
        x1, y1, x2, y2 = box
        patch = flow_mag[max(0,y1):y2, max(0,x1):x2]
        if patch.size > 0:
            mov_h = float(np.mean(patch))
        else:
            mov_h = 0
        data["mov_h"].append(mov_h)
        if len(data["mov_h"]) > WINDOW:
            data["mov_h"].pop(0)

        # correlación botella-humano
        if len(data["mov_h"]) >= 5:
            bh = np.array(data["tray"])
            hh = np.array(data["mov_h"])
            if np.std(bh) < 1e-6 or np.std(hh) < 1e-6:
                corr = 0
            else:
                corr = np.corrcoef(bh, hh)[0,1]
        else:
            corr = 0

        # decisión inteligente (sin reglas manuales)
        if corr > CO_MOVE_THR:
            estado = "sostenida"
            color = (255, 128, 0)   # azul claro
        else:
            estado = "libre"
            color = (0, 255, 0)     # verde

        data["estado"] = estado

        # dibujar SOLO botella
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{estado.upper()} ID {bid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        nuevos[bid] = data

    buffer_botellas = nuevos

    cv2.imshow("Inteligencia-A3", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
