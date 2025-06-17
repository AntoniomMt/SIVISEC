import cv2
import numpy as np

# Iniciamos el detector HOG de personas
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    orig = frame.copy()

    # Redimensionamos para mejor velocidad
    resized = cv2.resize(frame, (640, 480))

    # Detectamos personas
    boxes, weights = hog.detectMultiScale(resized, winStride=(8, 8), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in boxes:
        # Escalar caja si fue redimensionado
        scale_x = orig.shape[1] / 640
        scale_y = orig.shape[0] / 480
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        # Dibujar caja y etiqueta tipo folder abajo
        color = (0, 255, 0)  # Verde
        cv2.rectangle(orig, (x, y), (x + w, y + h), color, 3)

        # Etiqueta tipo folder
        label = "persona"
        tab_width = 80
        tab_height = 25
        tab_x1 = x
        tab_y1 = y + h
        tab_x2 = x + tab_width
        tab_y2 = tab_y1 + tab_height
        cv2.rectangle(orig, (tab_x1, tab_y1), (tab_x2, tab_y2), color, -1)
        triangle = [(tab_x2, tab_y1), (tab_x2 - 10, tab_y1), (tab_x2, tab_y1 + 10)]
        cv2.fillPoly(orig, [np.array(triangle)], color)
        cv2.putText(orig, label, (tab_x1 + 5, tab_y2 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Detección de personas", orig)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
