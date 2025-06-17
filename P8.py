from ultralytics import YOLO
import cv2

# Cargar modelo YOLOv5 Nano (rápido para CPU)
model = YOLO("yolov5n.pt")

# Inicializar la cámara en alta resolución
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Resolución reducida solo para detección
detection_width, detection_height = 640, 360

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar imagen para detección más rápida
    small = cv2.resize(frame, (detection_width, detection_height))

    # Ejecutar YOLOv5 sobre imagen pequeña
    results = model(small, verbose=False)[0]

    # Escalado entre tamaños
    scale_x = frame.shape[1] / detection_width
    scale_y = frame.shape[0] / detection_height

    person_count = 0

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box

        if int(cls) == 0:  # Clase 0 = persona
            person_count += 1

            # Escalar coordenadas al frame original
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Dibujar caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Dibujar etiqueta tipo folder abajo a la izquierda
            tag_width = 80
            tag_height = 30
            tag_x = x1
            tag_y = y2
            cv2.rectangle(frame, (tag_x, tag_y), (tag_x + tag_width, tag_y + tag_height), (0, 255, 0), -1)
            cv2.putText(frame, "persona", (tag_x + 5, tag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Mostrar conteo total arriba a la izquierda
    cv2.rectangle(frame, (10, 10), (200, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas: {person_count}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Mostrar ventana
    cv2.imshow("Conteo de Personas (YOLOv5)", frame)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
