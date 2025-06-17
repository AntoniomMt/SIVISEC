import cv2
import mediapipe as mp
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

# Variables para suavizar la detección
smooth_face_box = None
alpha = 0.2  # Suavidad (0 = muy suave, 1 = brusco)

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detección de rostro
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Seleccionamos el rostro más grande (el más cercano)
    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])

        # Suavizamos la posición
        if smooth_face_box is None:
            smooth_face_box = (x, y, w, h)
        else:
            sx, sy, sw, sh = smooth_face_box
            x = int(alpha * x + (1 - alpha) * sx)
            y = int(alpha * y + (1 - alpha) * sy)
            w = int(alpha * w + (1 - alpha) * sw)
            h = int(alpha * h + (1 - alpha) * sh)
            smooth_face_box = (x, y, w, h)

    # Procesamos manos
    results = hands.process(rgb)
    hand_present = results.multi_hand_landmarks is not None

    # Dibujar la caja si hay rostro suavizado
    if smooth_face_box is not None:
        (x, y, w, h) = smooth_face_box
        pad = 10
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])

        # Color y texto
        color = (0, 255, 0) if not hand_present else (0, 255, 255)
        label = "persona" if not hand_present else "con mano"

        # Caja gruesa
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

        # Pestaña estilo folder inferior izquierda
        tab_width = 80
        tab_height = 25
        tab_x1 = x1
        tab_y1 = y2
        tab_x2 = x1 + tab_width
        tab_y2 = y2 + tab_height
        cv2.rectangle(frame, (tab_x1, tab_y1), (tab_x2, tab_y2), color, -1)

        triangle = [(tab_x2, tab_y1), (tab_x2 - 10, tab_y1), (tab_x2, tab_y1 + 10)]
        cv2.fillPoly(frame, [np.array(triangle)], color)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (tab_x1 + 5, tab_y1 + tab_height - 7), font, 0.5, (0, 0, 0), 1)

    cv2.imshow("Caja suave de rostro", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
