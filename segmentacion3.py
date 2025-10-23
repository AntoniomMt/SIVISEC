import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Modelos ---
model = YOLO("yolov5nu.pt")
cap = cv2.VideoCapture(0)

def iou(boxA, boxB):
    # Calcula el IoU entre dos cajas (x1,y1,x2,y2)
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

with mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Reducir copia para YOLO (mejor rendimiento)
        small_frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results_yolo = model(rgb, verbose=False)

        botella_boxes = []
        for r in results_yolo:
            for box in r.boxes:
                if int(box.cls[0]) == 39:  # 39 = botella
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    scale_x = frame.shape[1] / 640
                    scale_y = frame.shape[0] / 480
                    botella_boxes.append((
                        int(x1 * scale_x), int(y1 * scale_y),
                        int(x2 * scale_x), int(y2 * scale_y)
                    ))

        # Procesar frame completo para manos
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb_full)

        mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Generar bounding boxes de las manos
        hand_boxes = []
        h, w, _ = frame.shape
        for hand_landmarks in [result.left_hand_landmarks, result.right_hand_landmarks]:
            if hand_landmarks:
                xs = [lm.x * w for lm in hand_landmarks.landmark]
                ys = [lm.y * h for lm in hand_landmarks.landmark]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                hand_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Comparar solapamiento
        toca_botella = False
        for hb in hand_boxes:
            for bb in botella_boxes:
                if iou(hb, bb) > 0.15:
                    toca_botella = True
                    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 255), 3)
                    cv2.putText(frame, "Botella sostenida", (bb[0], bb[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if toca_botella:
            cv2.putText(frame, "🖐 Mano sosteniendo botella", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Segmentacion + Deteccion Mejorada", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
