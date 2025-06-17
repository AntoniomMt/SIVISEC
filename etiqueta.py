import cv2
import mediapipe as mp
import numpy as np

# Inicializar el detector de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Inicializar cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar manos
    results = hands.process(rgb)
    hand_present = results.multi_hand_landmarks is not None

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        pad = 10
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])

        # Color y etiqueta
        color = (0, 255, 0) if not hand_present else (0, 255, 255)
        label = "persona" if not hand_present else "mano"

        # Dibujar la caja gruesa
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

        # Tamaño y posición de pestaña tipo folder
        tab_width = 80
        tab_height = 25
        tab_x1 = x1
        tab_y1 = y2
        tab_x2 = x1 + tab_width
        tab_y2 = y2 + tab_height

        # Dibujar la pestaña
        cv2.rectangle(frame, (tab_x1, tab_y1), (tab_x2, tab_y2), color, -1)

        # Dibujar una especie de "esquinita redonda" o curva simulada
        triangle = [(tab_x2, tab_y1), (tab_x2 - 10, tab_y1), (tab_x2, tab_y1 + 10)]
        cv2.fillPoly(frame, [np.array(triangle)], color)

        # Escribir el texto en la pestaña
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 0)
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = tab_x1 + 5
        text_y = tab_y1 + tab_height - 7
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, font_color, thickness)

    # Mostrar imagen
    cv2.imshow("Detección de rostro y mano", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
