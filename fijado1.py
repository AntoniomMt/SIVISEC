import cv2
import mediapipe as mp
import numpy as np

# Cargar clasificador de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Inicializar cámara
cap = cv2.VideoCapture(0)
fixed_face_box = None  # <- Aquí guardaremos el rostro fijado

print("Presiona 'r' para resetear el rostro fijado.")
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

    # Si aún no hay rostro fijado, lo detectamos
    if fixed_face_box is None:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            # Tomamos solo el primer rostro detectado
            fixed_face_box = faces[0]

    if fixed_face_box is not None:
        (x, y, w, h) = fixed_face_box
        pad = 10
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])

        # Color y etiqueta
        color = (0, 255, 0) if not hand_present else (0, 255, 255)
        label = "persona" if not hand_present else "con mano"

        # Dibujar la caja gruesa
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

        # Dibujar pestaña inferior izquierda estilo folder
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

    # Mostrar el frame
    cv2.imshow("Detección de rostro y mano (fijo)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        fixed_face_box = None  # Resetear detección del rostro

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
