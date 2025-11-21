# Deteccion-2.py
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import math
import time
from collections import deque

# -----------------------
# CONFIGURACIÓN
# -----------------------
YOLO_MODEL = "yolov8n.pt"
DETECT_EVERY = 3             # correr YOLO cada N frames (reduce flicker)
CONF_THRESH = 0.45           # confianza mínima en detecciones YOLO
PERSON_PAD = 0.15            # expansión relativa de bbox persona para asociar objetos
DIST_MATCH = 100             # px, distancia máxima para mantener mismo ID
CONFIRM_HITS = 3             # hits consecutivos para confirmar presencia
CONFIRM_MISS = 5             # misses consecutivos para confirmar desaparición
HAND_CONFIRM = 2             # frames para confirmar que mano está tocando botella
FPS_SMOOTH_ALPHA = 0.12

# colores
COLORS = {
    "normal": (0, 255, 0),          # verde
    "sosteniendo": (0, 255, 255),   # amarillo
    "escondiendo": (0, 165, 255),   # naranja
    "posible_robo": (0, 0, 255)     # rojo
}

# -----------------------
# UTILIDADES
# -----------------------
def centro(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    A = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    B = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    denom = A + B - inter + 1e-9
    return inter/denom

def expand_box(box, pad_rel, frame_shape):
    x1, y1, x2, y2 = box
    w = x2 - x1; h = y2 - y1
    px = int(w * pad_rel); py = int(h * pad_rel)
    fx, fy = frame_shape[1], frame_shape[0]
    nx1 = max(0, x1 - px); ny1 = max(0, y1 - py)
    nx2 = min(fx-1, x2 + px); ny2 = min(fy-1, y2 + py)
    return (nx1, ny1, nx2, ny2)

# -----------------------
# CLASE PARA TRACKING SIMPLE
# -----------------------
class PersonTrack:
    def __init__(self, pid, bbox, now):
        self.id = pid
        self.bbox = bbox                        # smoothed bbox
        self.centro = centro(bbox)
        self.state = "normal"
        self.ha_escondido = False
        # contadores para confirmación
        self.bottle_hits = 0
        self.bottle_misses = 0
        self.hand_hits = 0
        self.hand_misses = 0
        self.hand_bottle_hits = 0
        self.hand_bottle_misses = 0
        self.last_seen = now
        self.frames_missing = 0

    def update_bbox(self, new_box, alpha=0.35):
        # suaviza bbox (exponencial simple)
        x1,y1,x2,y2 = self.bbox
        nx1,ny1,nx2,ny2 = new_box
        self.bbox = (
            int(x1*(1-alpha)+nx1*alpha),
            int(y1*(1-alpha)+ny1*alpha),
            int(x2*(1-alpha)+nx2*alpha),
            int(y2*(1-alpha)+ny2*alpha)
        )
        self.centro = centro(self.bbox)

# -----------------------
# INICIALIZACIÓN MODELOS
# -----------------------
model = YOLO(YOLO_MODEL)
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# estructura de tracking
tracks = {}          # id -> PersonTrack
next_id = 1
frame_count = 0
fps_est = 0.0
prev_time = time.time()

# -----------------------
# BUCLE PRINCIPAL
# -----------------------
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    last_detected_bottles = []     # lista de cajas de botella del último momento YOLO
    last_person_boxes = []         # lista de cajas de persona del último YOLO
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        now = time.time()
        h, w, _ = frame.shape

        # ---- MEDIA PIPE HANDS (cada frame, es ligero) ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(rgb)
        mano_boxes = []
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                mano_boxes.append((min(xs), min(ys), max(xs), max(ys)))

        # ---- YOLO cada DETECT_EVERY frames (reduce flicker) ----
        if frame_count % DETECT_EVERY == 0:
            last_person_boxes = []
            last_detected_bottles = []
            yolo_results = model(frame, verbose=False)
            for r in yolo_results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < CONF_THRESH:
                        continue
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # filtro de tamaño (pequeño/extra grande)
                    bw, bh = x2 - x1, y2 - y1
                    if bw < 30 or bh < 60 or bw > w*0.9 or bh > h*0.95:
                        continue
                    if cls == 0:
                        last_person_boxes.append((x1,y1,x2,y2))
                    elif cls == 39:
                        last_detected_bottles.append((x1,y1,x2,y2))
        # si no corrió YOLO en este frame, reutilizamos last_person_boxes y last_detected_bottles

        # ---- Asociar personas detectadas a tracks por centroides ----
        used_track_ids = set()
        assigned_tracks = {}   # bbox -> track id
        # construir lista de person centroids
        det_centers = [centro(b) for b in last_person_boxes]

        # para cada detected person box, buscar track más cercano
        for i, pbox in enumerate(last_person_boxes):
            pc = det_centers[i]
            matched_id = None
            min_d = float("inf")
            for tid, tr in tracks.items():
                d = np.linalg.norm(tr.centro - pc)
                if d < min_d and d < DIST_MATCH and tid not in used_track_ids:
                    min_d = d; matched_id = tid
            if matched_id is not None:
                tracks[matched_id].update_bbox(pbox)
                tracks[matched_id].last_seen = now
                tracks[matched_id].frames_missing = 0
                used_track_ids.add(matched_id)
                assigned_tracks[i] = matched_id
            else:
                # crear nuevo track
                tracks[next_id] = PersonTrack(next_id, pbox, now)
                assigned_tracks[i] = next_id
                used_track_ids.add(next_id)
                next_id += 1

        # marcar tracks no asignados como missing (posible que salieran del cuadro)
        for tid, tr in list(tracks.items()):
            if tid not in used_track_ids:
                tr.frames_missing += 1
                if tr.frames_missing > 30:   # si desapareció por mucho, eliminar
                    del tracks[tid]

        # ---- Asociar botellas a personas (por centro en bbox expandido) ----
        # para robustez, usamos centroidos de botella y requerimos varios hits para confirmar
        bottle_centroids = [centro(b) for b in last_detected_bottles]
        # hacemos una asociación simple: para cada track, comprobamos si alguna botella cae dentro del bbox expandido
        for i, pbox in enumerate(last_person_boxes):
            tid = assigned_tracks.get(i)
            if not tid:
                continue
            tr = tracks[tid]
            exp_box = expand_box(tr.bbox, PERSON_PAD, frame.shape)
            bottle_in = False
            for j, b in enumerate(last_detected_bottles):
                bc = bottle_centroids[j]
                # si el centro de botella está dentro exp_box
                if (exp_box[0] <= bc[0] <= exp_box[2]) and (exp_box[1] <= bc[1] <= exp_box[3]):
                    bottle_in = True
                    break
            # actualizar contadores
            if bottle_in:
                tr.bottle_hits += 1
                tr.bottle_misses = 0
            else:
                tr.bottle_misses += 1
                tr.bottle_hits = 0

            # confirmar presencia/ausencia
            bottle_visible = tr.bottle_hits >= CONFIRM_HITS
            bottle_gone = tr.bottle_misses >= CONFIRM_MISS

            # ---- comprobar interacción mano<->botella (por varios frames) ----
            hand_touch = False
            for mano in mano_boxes:
                # mano dentro persona?
                if not (mano[0] > exp_box[2] or mano[2] < exp_box[0] or mano[1] > exp_box[3] or mano[3] < exp_box[1]):
                    # si hay botella(s) detectadas, revisar colisión mano-botella
                    for b in last_detected_bottles:
                        if not (mano[2] < b[0] or mano[0] > b[2] or mano[3] < b[1] or mano[1] > b[3]):
                            hand_touch = True
                            break
                if hand_touch:
                    break

            if hand_touch:
                tr.hand_bottle_hits += 1
                tr.hand_bottle_misses = 0
            else:
                tr.hand_bottle_misses = tr.hand_bottle_misses + 1 if hasattr(tr, "hand_bottle_misses") else 1
                tr.hand_bottle_hits = 0

            hand_touch_confirmed = tr.hand_bottle_hits >= HAND_CONFIRM
            hand_touch_gone = tr.hand_bottle_misses >= CONFIRM_MISS

            # ---- Máquina de estados robusta (solo afecta a la persona tr) ----
            prev_state = tr.state
            if tr.state == "normal":
                if hand_touch_confirmed and bottle_visible:
                    tr.state = "sosteniendo"
            elif tr.state == "sosteniendo":
                # si la botella ya no se ve pero antes sostenía -> esconder
                if bottle_gone and tr.bottle_hits == 0:
                    tr.state = "escondiendo"
                    tr.ha_escondido = True
                # si sigue tocando y botella visible, se mantiene sosteniendo
                elif hand_touch_confirmed and bottle_visible:
                    tr.state = "sosteniendo"
            elif tr.state == "escondiendo":
                # si reaparece y hay contacto -> volver a sosteniendo
                if hand_touch_confirmed and bottle_visible:
                    tr.state = "sosteniendo"
                # si no hay botella pero solo manos cerca y confirma -> posible_robo
                elif (not bottle_visible) and hand_touch_confirmed:
                    tr.state = "posible_robo"
            elif tr.state == "posible_robo":
                # si la botella aparece y hay contacto -> sosteniendo (harsh but fair)
                if hand_touch_confirmed and bottle_visible:
                    tr.state = "sosteniendo"

            # registrar last seen
            tr.last_seen = now

            # logging de eventos críticos
            if prev_state != tr.state and tr.state == "posible_robo":
                with open("eventos.log", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ID {tr.id} -> POSIBLE_ROBO\n")

        # ------------------------
        # DIBUJAR BOUNDS Y ESTADOS (solo visibles de persona)
        # ------------------------
        for tid, tr in tracks.items():
            # dibujar bbox persona
            x1,y1,x2,y2 = tr.bbox
            color = COLORS.get(tr.state, (255,255,255))
            label = {
                "normal":"Persona",
                "sosteniendo":"Sosteniendo mercancia",
                "escondiendo":"Escondiendo mercancia",
                "posible_robo":"Posible robo"
            }.get(tr.state, tr.state)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)   # grosor 2 (delgado)
            cv2.putText(frame, f"{label} (ID {tid})", (x1, max(12,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # mostrar FPS suavizado
        curr_time = time.time()
        inst_fps = 1.0 / max(1e-6, (curr_time - prev_time))
        fps_est = FPS_SMOOTH_ALPHA * inst_fps + (1 - FPS_SMOOTH_ALPHA) * fps_est
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps_est:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Deteccion-2 (robusta)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
