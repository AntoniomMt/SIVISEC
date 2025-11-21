import cv2
import mediapipe as mp
from ultralytics import YOLO
import math
import time

# === CONFIGURACIÓN BASE ===
model = YOLO("yolov8n.pt")  # Modelo YOLOv8
mp_hands = mp.solutions.hands
url = "http://192.168.0.50:8080/video"  # IP de tu tablet
cap = cv2.VideoCapture(url)

# === FUNCIONES AUXILIARES ===
def colision(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xA < xB and yA < yB

def centro(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2)//2, (y1 + y2)//2)

def suavizar(pos_ant, pos_act, alpha=0.25):
    return int(pos_ant*(1-alpha) + pos_act*alpha)

# === VARIABLES DE CONTROL ===
personas_memoria = {}
DISTANCIA_MAX = 80
FRAMES_CONFIRMACION = 6
PERSISTENCIA_BOTELLA = 10  # cuántos frames se mantiene una botella "fantasma" tras desaparecer
FPS_MEDIA = 0

botellas_mem = []  # para persistencia temporal de botellas
frames_desde_botella = 0

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

        # === DETECCIÓN YOLO ===
        results = model(frame, verbose=False)
        personas, botellas_actuales = [], []

        for r in results:
            for box in r.boxes:
                if box.conf[0] < 0.5:
                    continue
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # filtro tamaño razonable
                ancho, alto = x2 - x1, y2 - y1
                if ancho < 40 or alto < 80 or ancho > w*0.9 or alto > h*0.9:
                    continue

                if cls == 0:
                    personas.append((x1, y1, x2, y2))
                elif cls == 39:
                    botellas_actuales.append((x1, y1, x2, y2))

        # === MEMORIA DE BOTELLAS (persistencia) ===
        if botellas_actuales:
            botellas_mem = botellas_actuales
            frames_desde_botella = 0
        else:
            frames_desde_botella += 1
            if frames_desde_botella < PERSISTENCIA_BOTELLA:
                botellas_actuales = botellas_mem
            else:
                botellas_mem = []

        # === DETECCIÓN DE MANOS ===
        mano_boxes = []
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in hand.landmark]
                ys = [int(lm.y * h) for lm in hand.landmark]
                mano_boxes.append((min(xs), min(ys), max(xs), max(ys)))

        nuevas_memorias = {}

        # === PROCESO POR PERSONA ===
        for p_box in personas:
            px, py = centro(p_box)
            persona_id = None

            # Buscar persona cercana a la memoria
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
                "frames_confirmacion": 0,
                "ha_escondido": False,
                "centro": (px, py)
            })

            estado = data_ant["estado"]
            frames_confirm = data_ant["frames_confirmacion"]
            ha_escondido = data_ant["ha_escondido"]

            # --- Comprobar colisión de mano con botella ---
            sostenida = False
            for mano in mano_boxes:
                if colision(mano, p_box):
                    for botella in botellas_actuales:
                        if colision(mano, botella):
                            sostenida = True
                            break

            # === LÓGICA DE TRANSICIÓN MEJORADA ===
            nuevo_estado = estado

            # Si está sosteniendo
            if sostenida:
                if estado != "sosteniendo":
                    frames_confirm += 1
                    if frames_confirm >= FRAMES_CONFIRMACION:
                        nuevo_estado = "sosteniendo"
                        frames_confirm = 0
                else:
                    frames_confirm = 0

            # Si estaba sosteniendo y botella desaparece
            elif estado == "sosteniendo" and not botellas_actuales:
                frames_confirm += 1
                if frames_confirm >= FRAMES_CONFIRMACION:
                    nuevo_estado = "escondiendo"
                    ha_escondido = True
                    frames_confirm = 0

            # Si estaba escondiendo y ahora se ven manos sin botella
            elif estado == "escondiendo" and mano_boxes and not sostenida and not botellas_actuales:
                frames_confirm += 1
                if frames_confirm >= FRAMES_CONFIRMACION:
                    nuevo_estado = "posible_robo"
                    frames_confirm = 0

            # Si ya fue posible robo y vuelve a aparecer botella con interacción
            elif estado == "posible_robo" and sostenida:
                nuevo_estado = "sosteniendo"
                frames_confirm = 0

            else:
                frames_confirm = 0

            nuevas_memorias[persona_id] = {
                "estado": nuevo_estado,
                "frames_confirmacion": frames_confirm,
                "ha_escondido": ha_escondido,
                "centro": (px, py)
            }

        personas_memoria = nuevas_memorias

        # === DIBUJAR ===
        for pid, data in personas_memoria.items():
            estado = data["estado"]
            p_box = None
            for box in personas:
                if math.dist(centro(box), data["centro"]) < DISTANCIA_MAX:
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
            else:
                color, label = (255, 255, 255), estado

            cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), color, 2)
            cv2.putText(frame, f"{label} (ID {pid})", (p_box[0], p_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # === FPS ===
        curr_time = time.time()
        FPS_MEDIA = 0.9 * FPS_MEDIA + 0.1 * (1 / (curr_time - prev_time))
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {FPS_MEDIA:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Actividades-4", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
