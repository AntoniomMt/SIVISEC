from ultralytics import YOLO
import cv2

# Cargar modelo YOLOv5 Nano
model = YOLO("yolov5n.pt")  # Usa 'yolov5s.pt' si tienes más potencia

# Inicializar la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Detección con reescalado para rapidez
detection_width, detection_height = 640, 360

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Copia redimensionada para detección
    small = cv2.resize(frame, (detection_width, detection_height))
    
    # Ejecutar YOLO sobre imagen pequeña
    results = model(small, verbose=False)[0]

    # Escalado de coordenadas
    scale_x = frame.shape[1] / detection_width
    scale_y = frame.shape[0] / detection_height

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box

        # Solo personas (class 0)
        if int(cls) == 0:
            # Escalar coordenadas
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Dibujar mark box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Dibujar "pestaña" debajo a la izquierda
            tag_width = 80
            tag_height = 30
            tag_x = x1
            tag_y = y2

            cv2.rectangle(frame, (tag_x, tag_y), (tag_x + tag_width, tag_y + tag_height), (0, 255, 0), -1)
            cv2.putText(frame, "persona", (tag_x + 5, tag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Mostrar resultado
    cv2.imshow("Detección de Personas (YOLOv5)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
