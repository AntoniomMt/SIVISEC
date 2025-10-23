import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from collections import deque

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Modelos ---
model = YOLO("yolov5nu.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# Buffers para persistencia
last_hand_boxes = deque(maxlen=5)   # guarda últimas posiciones de manos
last_detect_frame = 0               # frame en el que se detectó interacción
frame_count = 0                     # contador global

with mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        small_frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results_yolo = model(rgb, verbose=False)

        botella_boxes = []
        for r in results_yolo:
            for box in r.boxes:
                if int(box.cls[0]) == 39:  # botella
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    sx, sy = frame.shape[1] / 640, frame.shape[0] / 480
                    botella_boxes.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)))

        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb_full)

        h, w, _ = frame.shape
        hand_boxes = []
        for hand_landmarks in [result.left_hand_landmarks, result.right_hand_landmarks]:
            if hand_landmarks:
                xs = [lm.x * w for lm in hand_landmarks.landmark]
                ys = [lm.y * h for lm in hand_landmarks.landmark]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                hand_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Si se pierden las manos, usar últimas posiciones
        if hand_boxes:
            last_hand_boxes.append(hand_boxes)
        elif len(last_hand_boxes) > 0:
            hand_boxes = last_hand_boxes[-1]

        # Comparar solapamiento (IoU)
        toca_botella = False
        for hb in hand_boxes:
            for bb in botella_boxes:
                if iou(hb, bb) > 0.15:
                    toca_botella = True
                    last_detect_frame = frame_count  # registrar detección
                    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 255), 3)

        # Persistencia de la etiqueta (~1 segundo)
        if frame_count - last_detect_frame < 30:
            cv2.putText(frame, "🖐 Mano sosteniendo botella", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Segmentacion Mejorada 4.0", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
