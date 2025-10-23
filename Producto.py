import cv2
import mediapipe as mp
from ultralytics import YOLO

# Inicializar modelos
model = YOLO("yolov5nu.pt")  # Puedes usar "yolov8n.pt" si prefieres
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Configuración cámara ---
cap = cv2.VideoCapture(0)

# --- Función para verificar colisión entre cajas ---
def colision(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xA < xB and yA < yB  # Si se solapan

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

        # --- Extraer manos ---
        mano_boxes = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                mano_boxes.append((x1, y1, x2, y2))

        # --- Lógica de interacción ---
        for (x1, y1, x2, y2) in personas:
            sostenida = False

            # Ver si alguna mano asociada a esta persona toca una botella
            for mano in mano_boxes:
                # Mano dentro de la persona
                if colision(mano, (x1, y1, x2, y2)):
                    for botella in botellas:
                        if colision(mano, botella):
                            sostenida = True
                            break

            # Dibujar resultado final
            if sostenida:
                color, label = (0, 128, 255), "Persona sosteniendo mercancia"
            else:
                color, label = (0, 255, 0), "Persona"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        cv2.imshow("Producto", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
