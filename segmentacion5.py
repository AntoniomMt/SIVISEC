import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from collections import deque

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Cargar modelo YOLO ---
model = YOLO("yolov5nu.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# --- Función IoU para solapamiento ---
def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# --- Variables ---
last_detect_frame = 0
frame_count = 0
no_hand_frames = 0

with mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w, _ = frame.shape

        # --- YOLO detección de botella ---
        results_yolo = model(frame, verbose=False)
        botella_boxes = []
        for r in results_yolo:
            for box in r.boxes:
                if int(box.cls[0]) == 39:  # clase 39 = botella
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    botella_boxes.append((x1, y1, x2, y2))

        # --- Mediapipe detección de manos ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)
        hand_boxes = []

        for hand_landmarks in [result.left_hand_landmarks, result.right_hand_landmarks]:
            if hand_landmarks:
                xs = [lm.x * w for lm in hand_landmarks.landmark]
                ys = [lm.y * h for lm in hand_landmarks.landmark]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                hand_boxes.append((x1, y1, x2, y2))

        # --- Control de persistencia de manos ---
        if hand_boxes:
            no_hand_frames = 0  # se vieron manos
        else:
            no_hand_frames += 1
            if no_hand_frames > 10:  # si pasan 10 frames sin manos, vaciar lista
                hand_boxes = []

        # --- Detectar interacción ---
        toca_botella = False
        fusion_box = None
        for hb in hand_boxes:
            for bb in botella_boxes:
                if iou(hb, bb) > 0.15:
                    toca_botella = True
                    last_detect_frame = frame_count
                    fusion_box = (
                        min(hb[0], bb[0]),
                        min(hb[1], bb[1]),
                        max(hb[2], bb[2]),
                        max(hb[3], bb[3])
                    )

        # --- Dibujar ---
        if frame_count - last_detect_frame < 30 and fusion_box:
            # Interacción activa o reciente
            x1, y1, x2, y2 = fusion_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "Botella sostenida", (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # Cajas normales
            for (x1, y1, x2, y2) in botella_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Amarillo botella
                cv2.putText(frame, "Botella", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            for (x1, y1, x2, y2) in hand_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Azul mano
                cv2.putText(frame, "Mano", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Segmentacion 5.1 - Interaccion Realista", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
