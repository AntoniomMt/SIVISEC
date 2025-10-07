import cv2
import mediapipe as mp
from ultralytics import YOLO

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Cargar modelos ---
model = YOLO("yolov5nu.pt")  # Cargar YOLO una sola vez
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        # Dibujar landmarks de manos
        mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # --- Detección de botella ---
        results = model(frame, verbose=False)
        botella_boxes = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 39:  # clase 39 = botella
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    botella_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Botella", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # --- Ver si alguna mano toca una botella ---
        def mano_toca_botella(landmarks, boxes, width, height):
            if landmarks:
                for lm in landmarks.landmark:
                    x, y = int(lm.x * width), int(lm.y * height)
                    for (x1, y1, x2, y2) in boxes:
                        # Ver si el punto está dentro o cerca del área de la botella
                        if x1 - 10 < x < x2 + 10 and y1 - 10 < y < y2 + 10:
                            return True
            return False

        h, w, _ = frame.shape
        toca = False
        if mano_toca_botella(result.left_hand_landmarks, botella_boxes, w, h):
            toca = True
        if mano_toca_botella(result.right_hand_landmarks, botella_boxes, w, h):
            toca = True

        if toca:
            cv2.putText(frame, "Mano cerca de botella", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            print("🖐 Botella sostenida o tocada")

        cv2.imshow("Segmentación + Detección", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()

