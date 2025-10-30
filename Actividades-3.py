import cv2
import mediapipe as mp
from ultralytics import YOLO
import math
import time

# --- Inicialización ---
model = YOLO("yolov8n.pt")  # Modelo YOLOv8
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# --- Funciones ---
def colision(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xA < xB and yA < yB

def centro(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def suavizar(pos_anterior, pos_actual, alpha=0.3):
    return int(pos_anterior * (1 - alpha) + pos_actual * alpha)

# --- Variables ---
personas_memoria = {}
DISTANCIA_MAX = 80
TIEMPO_SOSPECHOSO = 3  # segundos
FRAMES_MIN = 3         # frames consecutivos para confirmar cambio
FPS_MEDIA = 0          # para mostrar rendimiento

with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # --- YOLO detección ---
        results = model(frame, verbose=False)
        personas, botellas = [], []

        for r in results:
            for box in r.boxes:
                # 🧩 (1) Filtro de confianza mínima
                if box.conf[0] < 0.5:
                    continue

                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 🧩 (2) Filtro de tamaño de caja (evita falsos positivos grandes o muy pequeños)
                ancho, alto = x2 - x1, y2 - y1
                if ancho < 40 or alto < 80 or ancho > w * 0.9 or alto > h * 0.9:
                    continue

                if cls == 0:
                    personas.append((x1, y1, x2, y2))
                elif cls == 39:
                    botellas.append((x1, y1, x2, y2))

        # --- Manos ---
        mano_boxes = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                mano_boxes.append((min(xs), min(ys), max(xs), max(ys)))

        nuevas_memorias = {}

        # --- Analizar personas ---
        for p_box in personas:
            px, py = centro(p_box)
            persona_id = None

            # Asociar con memoria (seguimiento por cercanía)
            for pid, data in personas_memoria.items():
                ox, oy = data["centro"]
                if math.dist((px, py), (ox, oy)) < DISTANCIA_MAX:
                    persona_id = pid
                    px = suavizar(ox, px)
                    py = suavizar(oy, py)
                    break

            if persona_id is None:
                persona_id = len(personas_memoria) + 1

            data_ant = personas_memoria.get(persona_id, {
                "estado": "normal",
                "ha_escondido": False,
                "tiempo_sospechoso": 0,
                "frames_estado": 0,
                "frames_missing": 0
            })

            estado = data_ant["estado"]
            ha_escondido = data_ant.get("ha_escondido", False)
            tiempo_sospechoso = data_ant.get("tiempo_sospechoso", 0)
            frames_estado = data_ant.get("frames_estado", 0)

            # --- Colisión mano-botella ---
            sostenida = False
            for mano in mano_boxes:
                if colision(mano, p_box):
                    for botella in botellas:
                        if colision(mano, botella):
                            sostenida = True
                            break

            # --- Lógica de transición con debounce (3) ---
            nuevo_estado = estado

            if estado != "comportamiento_sospechoso":
                if estado == "posible_robo" and sostenida:
                    nuevo_estado = "comportamiento_sospechoso"
                    tiempo_sospechoso = time.time()
                    frames_estado = 0
                elif sostenida:
                    nuevo_estado = "sosteniendo"
                elif estado == "sosteniendo" and not botellas:
                    nuevo_estado = "escondiendo"
                    ha_escondido = True
                elif estado == "escondiendo":
                    if sostenida:
                        nuevo_estado = "sosteniendo"
                        frames_estado = 0
                    elif mano_boxes and not sostenida:
                        frames_estado += 1
                        if frames_estado >= FRAMES_MIN:
                            nuevo_estado = "posible_robo"
                            frames_estado = 0
                elif estado == "normal":
                    frames_estado = 0

            # Mantener tiempo del comportamiento sospechoso
            if nuevo_estado == "comportamiento_sospechoso":
                if time.time() - tiempo_sospechoso > TIEMPO_SOSPECHOSO:
                    nuevo_estado = "normal"
                    tiempo_sospechoso = 0

            # --- (4) Mantener IDs aunque desaparezcan brevemente ---
            data_ant["frames_missing"] = 0  # reset si fue detectado
            nuevas_memorias[persona_id] = {
                "estado": nuevo_estado,
                "centro": (px, py),
                "ha_escondido": ha_escondido,
                "tiempo_sospechoso": tiempo_sospechoso,
                "frames_estado": frames_estado,
                "frames_missing": 0
            }

            # --- (5) Registrar evento sospechoso ---
            if nuevo_estado == "posible_robo" and estado != "posible_robo":
                with open("eventos.log", "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} - ID {persona_id}: posible robo detectado\n")

        # Mantener IDs que desaparecieron brevemente
        for pid, data in personas_memoria.items():
            if pid not in nuevas_memorias:
                data["frames_missing"] += 1
                if data["frames_missing"] < 10:
                    nuevas_memorias[pid] = data  # mantener por unos frames

        personas_memoria = nuevas_memorias

        # --- Dibujar ---
        for pid, data in personas_memoria.items():
            estado = data["estado"]
            p_box = None
            for box in personas:
                cx, cy = centro(box)
                if math.dist((cx, cy), data["centro"]) < DISTANCIA_MAX:
                    p_box = box
                    break
            if not p_box:
                continue

            if estado == "normal":
                color, label = (0, 255, 0), "Persona"
            elif estado == "sosteniendo":
                color, label = (0, 255, 255), "Sosteniendo mercancía"
            elif estado == "escondiendo":
                color, label = (0, 165, 255), "Escondiendo mercancía"
            elif estado == "posible_robo":
                color, label = (0, 0, 255), "Posible robo"
            else:  # comportamiento sospechoso
                color, label = (128, 200, 255), "Comportamiento sospechoso"

            cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), color, 2)
            cv2.putText(frame, f"{label} (ID {pid})", (p_box[0], p_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # --- Mostrar FPS ---
        curr_time = time.time()
        FPS_MEDIA = 0.9 * FPS_MEDIA + 0.1 * (1 / (curr_time - prev_time))
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {FPS_MEDIA:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Deteccion-4 (robusto)", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
