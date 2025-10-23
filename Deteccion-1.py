import cv2
import mediapipe as mp
from ultralytics import YOLO
import math

# --- Inicialización ---
model = YOLO("yolov5nu.pt")  # Puedes usar yolov8n.pt si prefieres
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# --- Función para colisión ---
def colision(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xA < xB and yA < yB

# --- Memoria temporal ---
personas_memoria = {}
DISTANCIA_MAX = 80  # Para saber si es la misma persona

def centro(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # --- Detección YOLO ---
        results = model(frame, verbose=False)
        personas, botellas = [], []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls == 0:  # persona
                    personas.append((x1, y1, x2, y2))
                elif cls == 39:  # botella
                    botellas.append((x1, y1, x2, y2))

        # --- Manos ---
        mano_boxes = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                mano_boxes.append((min(xs), min(ys), max(xs), max(ys)))

        nuevas_memorias = {}
        alguien_escondiendo = False

        # --- Analizar cada persona ---
        for p_box in personas:
            px, py = centro(p_box)
            persona_id = None

            for pid, data in personas_memoria.items():
                ox, oy = data["centro"]
                if math.dist((px, py), (ox, oy)) < DISTANCIA_MAX:
                    persona_id = pid
                    break

            if persona_id is None:
                persona_id = len(personas_memoria) + 1

            data_ant = personas_memoria.get(persona_id, {"estado": "normal", "ha_escondido": False})
            estado = data_ant["estado"]
            ha_escondido = data_ant.get("ha_escondido", False)

            # --- Ver colisión de mano-botella ---
            sostenida = False
            for mano in mano_boxes:
                if colision(mano, p_box):
                    for botella in botellas:
                        if colision(mano, botella):
                            sostenida = True
                            break

            # --- Transiciones de estado ---
            if sostenida:
                estado = "sosteniendo"
            elif estado == "sosteniendo" and not botellas:
                estado = "escondiendo"
                ha_escondido = True
                alguien_escondiendo = True
            elif estado == "escondiendo" and botellas:
                # Si reaparece botella, puede volver a sostener si colisiona
                if sostenida:
                    estado = "sosteniendo"
                else:
                    estado = "normal"
            elif ha_escondido and not botellas and mano_boxes and not sostenida:
                estado = "posible_robo"

            nuevas_memorias[persona_id] = {"estado": estado, "centro": (px, py), "ha_escondido": ha_escondido}

        # Si alguien está escondiendo, marcar a todos los demás también
        if alguien_escondiendo:
            for pid, d in nuevas_memorias.items():
                if d["estado"] != "escondiendo" and d["estado"] != "posible_robo":
                    d["estado"] = "escondiendo"

        personas_memoria = nuevas_memorias

        # --- Dibujar en pantalla ---
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
                color, label = (0, 255, 255), "Sosteniendo mercancia"
            elif estado == "escondiendo":
                color, label = (0, 165, 255), "Escondiendo mercancia"
            else:
                color, label = (0, 0, 255), "Posible robo"

            cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), color, 2)
            cv2.putText(frame, label, (p_box[0], p_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Deteccion-1", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
