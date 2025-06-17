import cv2
import mediapipe as mp

# Inicializar el detector de rostros Haar Cascade
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

    # Detección de manos
    results = hands.process(rgb)
    hand_present = results.multi_hand_landmarks is not None

    # Detección de rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        pad = 10
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])

        # Color de la caja
        color = (0, 255, 0) if not hand_present else (0, 255, 255)  # Verde o Amarillo
        label = "persona" if not hand_present else "con mano"

        # Dibujar caja más gruesa
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

        # Dibujar pestaña tipo folder
        tab_width = 80
        tab_height = 30
        tab_x1 = x2
        tab_y1 = y1
        tab_x2 = x2 + tab_width
        tab_y2 = y1 + tab_height

        cv2.rectangle(frame, (tab_x1, tab_y1), (tab_x2, tab_y2), color, -1)

        # Texto dentro de la pestaña
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)  # Negro
        thickness = 2

        cv2.putText(frame, label, (tab_x1 + 5, tab_y1 + 20), font, font_scale, font_color, thickness)

    # Mostrar imagen
    cv2.imshow("Detección de rostro y mano", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
