import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

# inteligenciaA4.py
# - Híbrido: Anomaly Detection (no supervisado) + Embeddings de movimiento
# - Integra manos SIN reglas rígidas: usa features de mano en el embedding y clustering online
# - Mejora coherencia temporal: exponencial smoothing + ventanas temporales
# - Resultado: detecta botellas, mantiene solo botellas en pantalla,
#   marca ROJO para anomalía, AZUL si el embedding/clúster indica sostenida, VERDE si libre.

# Recomendaciones: ejecutar en entorno donde mediapipe y ultralytics estén instalados.

model = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False,
                                 max_num_hands=2,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Hiperparámetros
WINDOW = 20               # ventana para embedding temporal
SMOOTH_ALPHA = 0.4        # smoothing exponencial para centros
MAHAL_THRESH = 4.0        # umbral para considerar embedding novedoso (anomalía)
CLUSTERS = 2              # k para clustering online (libre vs sostenida)
MIN_INIT = 10             # frames para inicializar estadísticos

# Estructuras
buffer_botellas = {}
next_bid = 1
emb_history = []          # embeddings global (para inicialización)

# utilities

def centro(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


class OnlineKMeans:
    """KMeans online con k=2 para clustering rápido de embeddings.
    - inicializa con los dos primeros embeddings suficientemente distintos
    - actualización con momentum simple
    """
    def __init__(self, k=2, lr=0.15):
        self.k = k
        self.lr = lr
        self.centroids = []
        self.initialized = False

    def partial_fit(self, x):
        x = np.array(x, dtype=float)
        if not self.initialized:
            # agregar centroides hasta llenar k
            self.centroids.append(x.copy())
            if len(self.centroids) == self.k:
                # si los dos están muy cercanos, esperar a más datos
                if dist(self.centroids[0], self.centroids[1]) < 1e-3:
                    self.initialized = False
                else:
                    self.initialized = True
            return None

        # asignar al centroid más cercano y actualizarlo
        dists = [np.linalg.norm(x - c) for c in self.centroids]
        i = int(np.argmin(dists))
        # update centroid with momentum
        self.centroids[i] = (1 - self.lr) * self.centroids[i] + self.lr * x
        return i

    def predict(self, x):
        if not self.initialized:
            return None
        x = np.array(x, dtype=float)
        dists = [np.linalg.norm(x - c) for c in self.centroids]
        return int(np.argmin(dists))


# instancia online kmeans
okm = OnlineKMeans(k=CLUSTERS, lr=0.12)

# estadísticas para Mahalanobis (online mean & cov)
class OnlineGaussian:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, dtype=float)
        self.M2 = np.zeros((dim, dim), dtype=float)  # sum of squares for covariance
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
        # regularizar
        cov += np.eye(self.dim) * 1e-6
        diff = x - self.mean
        try:
            inv = np.linalg.pinv(cov)
            m = np.sqrt(float(diff.T @ inv @ diff))
        except Exception:
            m = float(np.linalg.norm(diff))
        return m

# emb Gaussian global (se inicializa cuando tengamos dim)
emb_gauss = None

# optical flow para movimiento
opt_prev = None
flow_mag = None

FRAME_COUNT = 0

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

    # manos (MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_res = hands_detector.process(rgb)
    hands_list = []  # lista de manos con (centro, openness, motion_est)

    if hands_res.multi_hand_landmarks:
        for h_lm in hands_res.multi_hand_landmarks:
            xs = [lm.x * frame.shape[1] for lm in h_lm.landmark]
            ys = [lm.y * frame.shape[0] for lm in h_lm.landmark]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            # openness: distancia normalizada entre tip pulgar (4) y tip indice (8)
            thumb = np.array([h_lm.landmark[4].x * frame.shape[1], h_lm.landmark[4].y * frame.shape[0]])
            index = np.array([h_lm.landmark[8].x * frame.shape[1], h_lm.landmark[8].y * frame.shape[0]])
            openness = float(np.linalg.norm(thumb - index))
            hands_list.append({
                'centro': (cx, cy),
                'openness': openness
            })

    # detección YOLO (botellas)
    results = model(frame, verbose=False)[0]
    botellas = []
    for b in results.boxes:
        if int(b.cls[0]) == 39:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            botellas.append((x1, y1, x2, y2))

    nuevos = {}

    for box in botellas:
        x1, y1, x2, y2 = box
        cx, cy = centro(box)
        bid = None
        # match por distancia a centros suavizados
        for oid, data in buffer_botellas.items():
            if dist((cx, cy), data['centro_smooth']) < 60:
                bid = oid
                break
        if bid is None:
            bid = next_bid
            next_bid += 1
            # inicializar estructura
            buffer_botellas[bid] = {
                'centro_smooth': (cx, cy),
                'centro_raw': (cx, cy),
                'tray': [],
                'hand_prox': [],
                'hand_open': [],
                'flow_patch': [],
                'embeddings': [],
                'estado': 'libre',
                'mahal': 0.0,
                'cluster': None
            }

        data = buffer_botellas[bid]
        # centro suavizado por exponential smoothing -> mejora coherencia temporal
        prev = np.array(data['centro_smooth'], dtype=float)
        curr = np.array((cx, cy), dtype=float)
        smooth = tuple((1 - SMOOTH_ALPHA) * prev + SMOOTH_ALPHA * curr)
        data['centro_smooth'] = smooth
        data['centro_raw'] = (cx, cy)

        # movimiento botella (delta de centros)
        mov_b = float(np.linalg.norm(curr - prev))
        data['tray'].append(mov_b)
        if len(data['tray']) > WINDOW:
            data['tray'].pop(0)

        # mano mas cercana: proximidad y apertura promedio
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
            data['hand_prox'].pop(0)
            data['hand_open'].pop(0)

        # optical flow local (magnitud media en parche de la botella)
        px1, py1, px2, py2 = max(0, x1), max(0, y1), min(flow_mag.shape[1]-1, x2), min(flow_mag.shape[0]-1, y2)
        patch = flow_mag[py1:py2, px1:px2]
        patch_mean = float(np.mean(patch)) if patch.size > 0 else 0.0
        data['flow_patch'].append(patch_mean)
        if len(data['flow_patch']) > WINDOW:
            data['flow_patch'].pop(0)

        # construir embedding temporal simple (estadísticos en ventana)
        emb = []
        # movimiento: mean, std
        emb.append(float(np.mean(data['tray']) if len(data['tray'])>0 else 0.0))
        emb.append(float(np.std(data['tray']) if len(data['tray'])>0 else 0.0))
        # hand prox: mean, std
        emb.append(float(np.mean(data['hand_prox']) if len(data['hand_prox'])>0 else 9999.0))
        emb.append(float(np.std(data['hand_prox']) if len(data['hand_prox'])>0 else 0.0))
        # hand openness
        emb.append(float(np.mean(data['hand_open']) if len(data['hand_open'])>0 else 0.0))
        # flow
        emb.append(float(np.mean(data['flow_patch']) if len(data['flow_patch'])>0 else 0.0))

        emb = np.array(emb, dtype=float)
        # normalización simple por escala conocida
        # proximidad grande -> dividir por frame diagonal para stabilizar
        h, w = frame.shape[:2]
        emb[2] = emb[2] / np.sqrt(w*h)  # normalized proximity
        emb[0] = emb[0] / np.sqrt(w*h)  # normalized movement
        emb[1] = emb[1] / np.sqrt(w*h)
        emb[5] = emb[5] / (np.mean(flow_mag) + 1e-6)

        data['embeddings'].append(emb)
        if len(data['embeddings']) > WINDOW:
            data['embeddings'].pop(0)

        # agregar a historial global para inicialización
        emb_history.append(emb)
        if len(emb_history) > 1000:
            emb_history.pop(0)

        # inicializar emb_gauss cuando tengamos dim
        if emb_gauss is None:
            emb_gauss = OnlineGaussian(dim=emb.shape[0])

        # actualizar gaussian global para detección de novedad solo durante fase inicial y luego continuamente
        if emb_gauss is not None:
            emb_gauss.update(emb)

        # entrenar clustering online
        okm.partial_fit(emb)
        cluster = okm.predict(emb)
        data['cluster'] = cluster

        # Mahalanobis (novedad)
        mahal = emb_gauss.mahalanobis(emb) if emb_gauss is not None else 0.0
        data['mahal'] = mahal

        # Decidir estado en base a cluster asignado y propiedades del centroid (no reglas puntuales)
        state = 'libre'
        color = (0, 255, 0)
        # si clustering activo, estimar cual cluster representa "sostener" por comparar centroides
        if okm.initialized and cluster is not None:
            # tomamos los dos centroides y evaluamos su caracteristica promedio de proximidad (index 2 emb)
            cent0 = okm.centroids[0]
            cent1 = okm.centroids[1]
            # cluster con menor proximidad (cent[*][2]) y mayor movimiento-correlativo es candidato a 'sostenida'
            score0 = -cent0[2] + cent0[0]  # menor proximidad y mayor mov -> score mayor
            score1 = -cent1[2] + cent1[0]
            sustained_cluster = 0 if score0 > score1 else 1
            if cluster == sustained_cluster:
                state = 'sostenida'
                color = (255, 160, 50)  # azul-anaranjado visual

        # override marca anomalía si mahal > thresh
        if mahal > MAHAL_THRESH:
            state = 'anomalía'
            color = (0, 0, 255)

        data['estado'] = state

        # dibujar SOLO botella
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{state.upper()} ID {bid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        nuevos[bid] = data

    buffer_botellas = nuevos

    cv2.imshow("Inteligencia-A4 (Híbrido)", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
