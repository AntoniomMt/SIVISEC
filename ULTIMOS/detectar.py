import cv2
from ultralytics import YOLO

# Carga modelos reales
model_manos = YOLO("yolov8n.pt")     # modelo descargado de Roboflow
model_botellas = YOLO("yolov8n.pt")       # tu YOLO normal para botellas

def intersecta(c1, c2, umbral=0.05):
    xA = max(c1[0], c2[0])
    yA = max(c1[1], c2[1])
    xB = min(c1[2], c2[2])
    yB = min(c1[3], c2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return False

    area_min = min(
        (c1[2]-c1[0]) * (c1[3]-c1[1]),
        (c2[2]-c2[0]) * (c2[3]-c2[1])
    )
    if area_min == 0:
        return False

    return (inter / area_min) >= umbral

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    manos = []
    botellas = []

    # --- Detectar manos ---
    rm = model_manos(frame, verbose=False)[0]
    for box in rm.boxes:
        conf = float(box.conf[0])
        if conf < 0.40:
            continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])
        manos.append((x1,y1,x2,y2))

    # --- Detectar botellas ---
    rb = model_botellas(frame, verbose=False)[0]
    for box in rb.boxes:
        cls = int(box.cls[0])
        name = rb.names[cls]
        conf = float(box.conf[0])

        if name == "bottle" and conf > 0.40:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            botellas.append((x1,y1,x2,y2))

    # --- Lógica: mano sosteniendo botella ---
    for (x1,y1,x2,y2) in manos:
        color = (0,255,0)
        etiqueta = "mano"

        for (bx1,by1,bx2,by2) in botellas:
            if intersecta((x1,y1,x2,y2), (bx1,by1,bx2,by2)):
                color = (255,0,0)
                etiqueta = "mano sosteniendo"
                break

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, etiqueta, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for (bx1,by1,bx2,by2) in botellas:
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,128,255), 2)
        cv2.putText(frame, "botella", (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)

    cv2.imshow("Hands + Bottle detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
