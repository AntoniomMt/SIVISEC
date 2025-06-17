import cv2
import mediapipe as mp

# Inicializar el detector de rostros Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7)

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

    # Flip para tipo espejo
    frame = cv2.flip(frame, 1)

    # Convertir a escala de grises para la detección de rostro
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convertir a RGB para MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detección de manos
    results = hands.process(rgb)
    hand_present = results.multi_hand_landmarks is not None

    # Detección de rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Ampliar un poco el box
        pad = 10
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])

        # Dibujar caja verde o amarilla si hay mano
        color = (0, 255, 0) if not hand_present else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Mostrar imagen
    cv2.imshow("Detección de rostro y mano", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
