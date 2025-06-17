import cv2
import numpy as np

# Cargar el modelo preentrenado MobileNet SSD
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)


# Etiquetas de clases del modelo MobileNet-SSD (solo usamos 'person')
CLASSES = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
           "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
           "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
           "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
           "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
           "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
           "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
           "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
           "toothbrush"]

cap = cv2.VideoCapture(0)
print("Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Preparar la imagen para el modelo
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    # Dibujar detecciones de personas
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if CLASSES[class_id] == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)

                # Pestaña inferior izquierda tipo folder
                tab_width = 90
                tab_height = 25
                tab_x1 = x1
                tab_y1 = y2
                tab_x2 = x1 + tab_width
                tab_y2 = y2 + tab_height
                cv2.rectangle(frame, (tab_x1, tab_y1), (tab_x2, tab_y2), (0, 255, 255), -1)

                triangle = [(tab_x2, tab_y1), (tab_x2 - 10, tab_y1), (tab_x2, tab_y1 + 10)]
                cv2.fillPoly(frame, [np.array(triangle)], (0, 255, 255))

                label = "persona"
                cv2.putText(frame, label, (tab_x1 + 5, tab_y1 + tab_height - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Detección de personas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
net