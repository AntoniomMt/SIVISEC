"""
inteligenciaB.py

Versión B (híbrida con predicción corta y persistencia inteligente).
Basado en inteligenciaA4.py pero con:
 - predicción de posición para oclusiones cortas (3-7 frames)
 - persistencia de estado condicionada por la estabilidad del embedding
 - reconciliación cuando la detección reaparece
 - sigue siendo "inteligencia real" porque la decisión de predecir y mantener
   estado se basa en medidas estadísticas (estabilidad del embedding, Mahalanobis,
   clustering online), no en reglas rígidas "si desaparece entonces X".

Nota: el proyecto original / tesis se encuentra en local: /mnt/data/tesis-ver1.docx
(este comentario surge de una instrucción del entorno; no es usado por el script)

Requisitos:
 - ultralytics (YOLOv8)
 - mediapipe
 - opencv-python
 - numpy

Ejecución:
 python inteligenciaB.py

Te recomiendo probar con cámara y hacer pequeñas oclusiones (tapando la botella
 con la mano por 3-5 frames) para ver la persistencia.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

# --- Configuración y parámetros ---
MODEL_PATH = "yolov8n.pt"
WINDOW = 20
SMOOTH_ALPHA = 0.35
MAHAL_THRESH = 4.0
CLUSTERS = 2
MIN_INIT = 8

# predicción y persistencia
PRED_MAX_STABLE = 7    # si embedding estable
PRED_MAX_UNSTABLE = 2  # si embedding inestable
MISSING_REMOVE = 30    # frames sin reaparecer para borrar definitivamente
PRED_CONF_THRESH = 0.5 # no usado como regla, reservado

# inicializar modelos
model = YOLO(MODEL_PATH)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False,
                                 max_num_hands=2,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# optical flow prev
opt_prev = None
flow_mag = None

# estructuras
buffer_botellas = {}
next_bid = 1
emb_history = []
FRAME_COUNT = 0

# --- utilities ---

def centro(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


class OnlineKMeans:
    def __init__(self, k=2, lr=0.12):
        self.k = k
        self.lr = lr
        self.centroids = []
        self.initialized = False

    def partial_fit(self, x):
        x = np.array(x, dtype=float)
        if not self.initialized:
            # add until k
            if len(self.centroids) < self.k:
                self.centroids.append(x.copy())
                if len(self.centroids) == self.k:
                    # require some separation
                    if dist(self.centroids[0], self.centroids[1]) > 1e-3:
                        self.initialized = True
                return None
            return None
        dists = [np.linalg.norm(x - c) for c in self.centroids]
        i = int(np.argmin(dists))
        self.centroids[i] = (1 - self.lr) * self.centroids[i] + self.lr * x
        return i

    def predict(self, x):
        if not self.initialized:
            return None
        x = np.array(x, dtype=float)
        dists = [np.linalg.norm(x - c) for c in self.centroids]
        return int(np.argmin(dists))


class OnlineGaussian:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, dtype=float)
        self.M2 = np.zeros((dim, dim), dtype=float)
        self.dim = dim

    def update(self, x):
        x = np.array(x, dtype=float)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)

    def cov(self):
        if self.n < 2:
            return np.eye(self.dim) * 1e-6
        return self.M2 / (self.n - 1)

    def mahalanobis(self, x):
        x = np.array(x, dtype=float)
        cov = self.cov()
        cov += np.eye(self.dim) * 1e-6
        diff = x - self.mean
        try:
            inv = np.linalg.pinv(cov)
            m = np.sqrt(float(diff.T @ inv @ diff))
        except Exception:
            m = float(np.linalg.norm(diff))
        return m


okm = OnlineKMeans(k=CLUSTERS, lr=0.12)
emb_gauss = None

# helpers de predicción simple

def predict_next_center(data):
    # usa últimos dos centros suavizados para estimar velocidad y predecir
    try:
        prev = np.array(data['centro_smooth_prev'], dtype=float)
        curr = np.array(data['centro_smooth'], dtype=float)
        velocity = curr - prev
        # predice 1 frame adelante (se puede multiplicar para N frames)
        pred = curr + velocity
        return (int(pred[0]), int(pred[1]))
    except Exception:
        return tuple(map(int, data['centro_smooth']))


# bucle principal
while True:
    ret, frame = cap.read()
    if not ret:
        break
    FRAME_COUNT += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # optical flow
    if opt_prev is None:
        opt_prev = gray
        flow_mag = np.zeros_like(gray, dtype=np.float32)
    else:
        flow = cv2.calcOpticalFlowFarneback(opt_prev, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mag = mag
        opt_prev = gray

    # manos
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_res = hands_detector.process(rgb)
    hands_list = []
    if hands_res.multi_hand_landmarks:
        for h_lm in hands_res.multi_hand_landmarks:
            xs = [lm.x * frame.shape[1] for lm in h_lm.landmark]
            ys = [lm.y * frame.shape[0] for lm in h_lm.landmark]
            cx = float(np.mean(xs)); cy = float(np.mean(ys))
            thumb = np.array([h_lm.landmark[4].x * frame.shape[1], h_lm.landmark[4].y * frame.shape[0]])
            index = np.array([h_lm.landmark[8].x * frame.shape[1], h_lm.landmark[8].y * frame.shape[0]])
            openness = float(np.linalg.norm(thumb - index))
            hands_list.append({'centro': (cx, cy), 'openness': openness})

    # detección YOLO (botellas)
    results = model(frame, verbose=False)[0]
    botellas = []
    for b in results.boxes:
        if int(b.cls[0]) == 39:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            botellas.append((x1, y1, x2, y2))

    # marcadores de tiempo para reconciliación
    seen_this_frame = set()
    nuevos = {}

    # actualizar detecciones visibles
    for box in botellas:
        x1, y1, x2, y2 = box
        cx, cy = centro(box)
        bid = None
        for oid, data in buffer_botellas.items():
            if dist((cx, cy), data['centro_smooth']) < 60:
                bid = oid
                break
        if bid is None:
            bid = next_bid
            next_bid += 1
            buffer_botellas[bid] = {
                'centro_smooth': (cx, cy),
                'centro_smooth_prev': (cx, cy),
                'centro_raw': (cx, cy),
                'tray': [],
                'hand_prox': [],
                'hand_open': [],
                'flow_patch': [],
                'embeddings': [],
                'estado': 'libre',
                'mahal': 0.0,
                'cluster': None,
                'last_seen': FRAME_COUNT,
                'missing': 0,
                'predicted': False
            }

        data = buffer_botellas[bid]

        # actualizar prev y smooth
        data['centro_smooth_prev'] = data.get('centro_smooth', (cx, cy))
        prev = np.array(data['centro_smooth_prev'], dtype=float)
        curr = np.array((cx, cy), dtype=float)
        smooth = tuple((1 - SMOOTH_ALPHA) * prev + SMOOTH_ALPHA * curr)
        data['centro_smooth'] = smooth
        data['centro_raw'] = (cx, cy)

        # movimiento relativo
        mov_b = float(np.linalg.norm(curr - prev))
        data['tray'].append(mov_b)
        if len(data['tray']) > WINDOW:
            data['tray'].pop(0)

        # manos
        proximities = []
        opens = []
        for h in hands_list:
            proximities.append(float(np.linalg.norm(np.array(h['centro']) - curr)))
            opens.append(float(h['openness']))
        if len(proximities) > 0:
            minprox = float(np.min(proximities))
            meanopen = float(np.mean(opens))
        else:
            minprox = 9999.0
            meanopen = 0.0
        data['hand_prox'].append(minprox)
        data['hand_open'].append(meanopen)
        if len(data['hand_prox']) > WINDOW:
            data['hand_prox'].pop(0); data['hand_open'].pop(0)

        # optical flow local
        h_img, w_img = frame.shape[:2]
        px1, py1, px2, py2 = max(0, x1), max(0, y1), min(w_img-1, x2), min(h_img-1, y2)
        patch = flow_mag[py1:py2, px1:px2]
        patch_mean = float(np.mean(patch)) if patch.size > 0 else 0.0
        data['flow_patch'].append(patch_mean)
        if len(data['flow_patch']) > WINDOW:
            data['flow_patch'].pop(0)

        # embedding simple
        emb = []
        emb.append(float(np.mean(data['tray']) if len(data['tray'])>0 else 0.0))
        emb.append(float(np.std(data['tray']) if len(data['tray'])>0 else 0.0))
        emb.append(float(np.mean(data['hand_prox']) if len(data['hand_prox'])>0 else 9999.0))
        emb.append(float(np.std(data['hand_prox']) if len(data['hand_prox'])>0 else 0.0))
        emb.append(float(np.mean(data['hand_open']) if len(data['hand_open'])>0 else 0.0))
        emb.append(float(np.mean(data['flow_patch']) if len(data['flow_patch'])>0 else 0.0))
        emb = np.array(emb, dtype=float)

        # normalizaciones
        emb[2] = emb[2] / np.sqrt(w_img*h_img)
        emb[0] = emb[0] / np.sqrt(w_img*h_img)
        emb[1] = emb[1] / np.sqrt(w_img*h_img)
        emb[5] = emb[5] / (np.mean(flow_mag) + 1e-6)

        data['embeddings'].append(emb)
        if len(data['embeddings']) > WINDOW:
            data['embeddings'].pop(0)

        emb_history.append(emb)
        if len(emb_history) > 2000:
            emb_history.pop(0)

        # inicializar estadística global
        if emb_gauss is None:
            emb_gauss = OnlineGaussian(dim=emb.shape[0])

        if emb_gauss is not None:
            emb_gauss.update(emb)

        okm.partial_fit(emb)
        cluster = okm.predict(emb)
        data['cluster'] = cluster

        mahal = emb_gauss.mahalanobis(emb) if emb_gauss is not None else 0.0
        data['mahal'] = mahal

        # estabilidad: evaluar std de embeddings en ventana
        emb_stack = np.stack(data['embeddings']) if len(data['embeddings'])>0 else np.zeros((1,emb.shape[0]))
        emb_std = float(np.mean(np.std(emb_stack, axis=0)))

        # decidir estado basado en cluster y novedad (no reglas duras)
        state = 'libre'; color = (0,255,0)
        if okm.initialized and cluster is not None:
            cent0 = okm.centroids[0]; cent1 = okm.centroids[1]
            score0 = -cent0[2] + cent0[0]
            score1 = -cent1[2] + cent1[0]
            sustained_cluster = 0 if score0 > score1 else 1
            if cluster == sustained_cluster:
                state = 'sostenida'; color = (200,140,40)
        if mahal > MAHAL_THRESH:
            state = 'anomalía'; color = (0,0,255)

        data['estado'] = state
        data['last_seen'] = FRAME_COUNT
        data['missing'] = 0
        data['predicted'] = False

        # dibujar caja y label
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, f"{state.upper()} ID {bid}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        nuevos[bid] = data
        seen_this_frame.add(bid)

    # manejar botellas que no se vieron: predecir o borrar
    for oid, data in list(buffer_botellas.items()):
        if oid in seen_this_frame:
            continue
        # no vista este frame
        data['missing'] = data.get('missing',0) + 1
        data['last_seen'] = data.get('last_seen', FRAME_COUNT - data['missing'])

        # estabilidad de embedding para decidir cuánto predecir
        embs = data.get('embeddings', [])
        if len(embs) >= 3:
            emb_stack = np.stack(embs)
            emb_std = float(np.mean(np.std(emb_stack, axis=0)))
        else:
            emb_std = 1e6

        pred_limit = PRED_MAX_STABLE if emb_std < 0.002 else PRED_MAX_UNSTABLE

        if data['missing'] <= pred_limit and data.get('centro_smooth_prev') is not None:
            # predecir posición
            pred_c = predict_next_center(data)
            # marcar como predicha
            data['predicted'] = True
            # mantener estado y color
            state = data.get('estado','libre')
            if state == 'libre': color = (0,255,0)
            elif state == 'sostenida': color = (200,140,40)
            elif state == 'anomalía': color = (0,0,255)
            else: color = (0,255,255)

            # dibujar marcador predictivo (con transparencia visual por grosor)
            x_pred, y_pred = pred_c
            size = 40
            x1 = int(x_pred - size//2); y1 = int(y_pred - size//2); x2 = int(x_pred + size//2); y2 = int(y_pred + size//2)
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 1)
            cv2.putText(frame, f"PRED {state.upper()} ID {oid}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # keep in buffer for possible reconciliation
            data['centro_smooth_prev'] = data.get('centro_smooth', data.get('centro_raw', (x_pred,y_pred)))
            data['centro_smooth'] = (x_pred, y_pred)
            nuevos[oid] = data
        else:
            # si excede missing tolerable, eliminar
            if data['missing'] > MISSING_REMOVE:
                buffer_botellas.pop(oid, None)
            else:
                # mantener en buffer pero no predecir visualmente
                nuevos[oid] = data

    buffer_botellas = nuevos

    cv2.imshow("Inteligencia-B (Predicción)", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
