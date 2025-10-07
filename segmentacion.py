import cv2
import mediapipe as mp
#importar yolo para detección de objetos
from ultralytics import YOLO

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

        # Dibujar landmarks del cuerpo
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        # Opcional: resaltar solo las manos
        mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        #Mostrar en consola cuantos puntos de referencia se detectaron de la mano unicamente
        if result.left_hand_landmarks:
            print(f"Mano izquierda: {len(result.left_hand_landmarks.landmark)} puntos detectados")
        if result.right_hand_landmarks:
            print(f"Mano derecha: {len(result.right_hand_landmarks.landmark)} puntos detectados")

        #Colocar una bounding box alrededor de las manos
        if result.left_hand_landmarks:
            h, w, _ = frame.shape
            x_min = min([landmark.x for landmark in result.left_hand_landmarks.landmark]) * w
            y_min = min([landmark.y for landmark in result.left_hand_landmarks.landmark]) * h
            x_max = max([landmark.x for landmark in result.left_hand_landmarks.landmark]) * w
            y_max = max([landmark.y for landmark in result.left_hand_landmarks.landmark]) * h
            cv2.rectangle(frame, (int(x_min)-10, int(y_min)-10), (int(x_max)+10, int(y_max)+10), (255, 0, 0), 2)
        if result.right_hand_landmarks:
            h, w, _ = frame.shape
            x_min = min([landmark.x for landmark in result.right_hand_landmarks.landmark]) * w
            y_min = min([landmark.y for landmark in result.right_hand_landmarks.landmark]) * h
            x_max = max([landmark.x for landmark in result.right_hand_landmarks.landmark]) * w
            y_max = max([landmark.y for landmark in result.right_hand_landmarks.landmark]) * h
            cv2.rectangle(frame, (int(x_min)-10, int(y_min)-10), (int(x_max)+10, int(y_max)+10), (0, 255, 0), 2)

        #Colocar un bounding box alrededor de una botella si se detecta
        model = YOLO("yolov8n.pt")
        results = model(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 39:  # Clase 39 es botella en COCO dataset
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Botella", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        #Imprimir en consola si se detecta una botella
        if any(int(box.cls[0]) == 39 for r in results for box in r.boxes):
            print("Botella detectada")

        #Mostrar resultado
        cv2.imshow("Segmentación corporal (Manos incluidas)", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
