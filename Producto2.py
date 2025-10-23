import cv2
import mediapipe as mp
from ultralytics import YOLO
import math

# --- Inicialización ---
model = YOLO("yolov5nu.pt")  # Cambia a yolov8n.pt si prefieres
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# --- Función para colisión ---
def colision(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xA < xB and yA < yB

# --- Memoria temporal de personas ---
personas_memoria = {}

# --- Función para calcular centroide ---
def centro(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# --- Umbral de distancia para identificar misma persona ---
DISTANCIA_MAX = 80

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
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

        # --- Manos (invisibles) ---
        mano_boxes = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                mano_boxes.append((min(xs), min(ys), max(xs), max(ys)))

        # --- Procesar cada persona ---
        nuevas_memorias = {}
        for p_box in personas:
            px, py = centro(p_box)
            persona_id = None

            # Buscar si ya existía una persona cercana en memoria
            for pid, data in personas_memoria.items():
                ox, oy = data["centro"]
                if math.dist((px, py), (ox, oy)) < DISTANCIA_MAX:
                    persona_id = pid
                    break

            if persona_id is None:
                persona_id = len(personas_memoria) + 1

            estado_actual = personas_memoria.get(persona_id, {"estado": "normal"})["estado"]
            sostenida = False

            # Ver si alguna mano toca una botella
            for mano in mano_boxes:
                if colision(mano, p_box):
                    for botella in botellas:
                        if colision(mano, botella):
                            sostenida = True
                            break

            # --- Actualizar estado ---
            if sostenida:
                estado_actual = "sosteniendo"
            elif estado_actual == "sosteniendo" and not botellas:
                estado_actual = "escondiendo"

            nuevas_memorias[persona_id] = {"estado": estado_actual, "centro": (px, py)}

            # --- Dibujar ---
            if estado_actual == "normal":
                color, label = (0, 255, 0), "Persona"
            elif estado_actual == "sosteniendo":
                color, label = (0, 128, 255), "Persona sosteniendo mercancia"
            else:
                color, label = (0, 0, 255), "Escondiendo mercancia"

            cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[2]), color, 2)
            cv2.putText(frame, label, (p_box[0], p_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        personas_memoria = nuevas_memorias

        cv2.imshow("Producto2", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
