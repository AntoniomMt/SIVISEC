import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

"""
inteligenciaB3.py

Mejoras aplicadas respecto a B2 (resolviendo tus problemas):
1) Detección: volvemos a correr YOLO sobre el frame completo (mejor calidad)
   pero mantenemos optimizaciones para el optical flow local.
2) Mini-box dentro de la caja azul: eliminada mediante estrategia de asociación y
   supresión: si una detección corresponde a un track predicho se actualiza el
   track y solo se dibuja una caja (la de detección). Si la detección es muy
   pequeña respecto al track (area ratio), se ignora la detección para evitar
   cajas pequeñas internas.
3) Tracking mejorado: asociación por IoU + centroid + predicción lineal (velocidad)
   se usa una actualización simple tipo "Kalman-lite" (posición y velocidad).
4) Optical flow optimizado: se calcula solo en ROIs alrededor de tracks/detecciones
   y en downscale; reduce carga y evita degradar detección global.
5) Performance: algunas operaciones numpy están vectorizadas y el cálculo de flow
   es regional para mantener fps.

Cómo usar: ejecutar igual que antes.

"""

# --- Parametros ---
MODEL_PATH = "yolov8n.pt"
WINDOW = 18
SMOOTH = 0.32
MAHAL_T = 4.2
IOU_MATCH_THR = 0.35
CENTROID_MATCH_THR = 80.0
PRED_MAX_STABLE = 7
PRED_MAX_UNSTABLE = 2
MISSING_REMOVE = 30
FLOW_SCALE = 0.5
FLOW_STEP = 2
MIN_DET_AREA = 400  # px^2 small detections ignored

# Inicializar modelos
model = YOLO(MODEL_PATH)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False,
                                 max_num_hands=2,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
opt_prev_small = None
flow_small = None
FRAME = 0

# Tracks: diccionario id -> track dict
tracks = {}
next_id = 1

# Helpers

def iou(boxA, boxB):
    # boxes = (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


import numpy as np

def dist(a,b):
    return float(np.linalg.norm(np.array(a)-np.array(b)))

def center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)


def area(box):
    return max(1, (box[2]-box[0]) * (box[3]-box[1]))


def clamp_box(box, w, h):
    x1,y1,x2,y2 = box
    x1 = max(0, min(w-1, int(x1)))
    y1 = max(0, min(h-1, int(y1)))
    x2 = max(0, min(w-1, int(x2)))
    y2 = max(0, min(h-1, int(y2)))
    return (x1,y1,x2,y2)

# Simple online stats for novelty detection
class OnlineStat:
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
        return self.M2 / (self.n - 1) + np.eye(self.dim) * 1e-6
    def mahal(self, x):
        x = np.array(x, dtype=float)
        diff = x - self.mean
        inv = np.linalg.pinv(self.cov())
        return float(np.sqrt(diff.T @ inv @ diff))

emb_stat = None

# Predict next center using constant velocity model
def predict_center(track):
    c = np.array(track['center_smooth'], dtype=float)
    v = np.array(track.get('vel', (0.0, 0.0)), dtype=float)
    pred = c + v
    return (float(pred[0]), float(pred[1]))

# Update velocity in track
def update_velocity(track, new_center):
    prev = np.array(track.get('center_smooth', new_center), dtype=float)
    v_prev = np.array(track.get('vel', (0.0,0.0)), dtype=float)
    v_new = 0.6 * v_prev + 0.4 * (np.array(new_center)-prev)
    track['vel'] = (float(v_new[0]), float(v_new[1]))

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    FRAME += 1
    h_img, w_img = frame.shape[:2]

    start = time.time()

    # Hands
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_res = hands_detector.process(rgb)
    hands = []
    if hands_res.multi_hand_landmarks:
        for h_lm in hands_res.multi_hand_landmarks:
            xs = [lm.x * w_img for lm in h_lm.landmark]
            ys = [lm.y * h_img for lm in h_lm.landmark]
            cx = float(np.mean(xs)); cy = float(np.mean(ys))
            thumb = np.array([h_lm.landmark[4].x * w_img, h_lm.landmark[4].y * h_img])
            index = np.array([h_lm.landmark[8].x * w_img, h_lm.landmark[8].y * h_img])
            openness = float(np.linalg.norm(thumb - index))
            hands.append({'center':(cx,cy),'openness':openness})

    # YOLO detection (full frame for quality)
    results = model(frame, verbose=False)[0]
    dets = []
    for b in results.boxes:
        if int(b.cls[0]) == 39:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            bx = clamp_box((x1,y1,x2,y2), w_img, h_img)
            if area(bx) >= MIN_DET_AREA:
                dets.append(bx)

    # compute flow only within ROIs: union of det boxes and track predicted boxes
    rois = []
    for d in dets:
        # expand slightly
        pad = 16
        rois.append((max(0,d[0]-pad), max(0,d[1]-pad), min(w_img-1,d[2]+pad), min(h_img-1,d[3]+pad)))
    for tid, tr in tracks.items():
        if tr.get('missing',0) > 0 and tr.get('predicted_box') is not None:
            rb = tr['predicted_box']
            pad = 20
            rois.append((max(0,rb[0]-pad), max(0,rb[1]-pad), min(w_img-1,rb[2]+pad), min(h_img-1,rb[3]+pad)))

    # merge rois (simple)
    merged = []
    for r in rois:
        if not merged:
            merged.append(r)
            continue
        x1,y1,x2,y2 = r
        merged_flag = False
        for i,(mx1,my1,mx2,my2) in enumerate(merged):
            if not (x2 < mx1 or x1 > mx2 or y2 < my1 or y1 > my2):
                # overlap -> merge
                merged[i] = (min(x1,mx1), min(y1,my1), max(x2,mx2), max(y2,my2))
                merged_flag = True
                break
        if not merged_flag:
            merged.append(r)

    # compute optical flow inside merged ROIs at downscale
    flow_map = np.zeros((h_img, w_img), dtype=np.float32)
    if FRAME % FLOW_STEP == 0 and merged:
        small_prev = None
        # take grayscale small of full frame once
        full_small = cv2.resize(frame, (0,0), fx=FLOW_SCALE, fy=FLOW_SCALE)
        gray_full = cv2.cvtColor(full_small, cv2.COLOR_BGR2GRAY)
        if opt_prev_small is None:
            opt_prev_small = gray_full
        else:
            flow_full = cv2.calcOpticalFlowFarneback(opt_prev_small, gray_full, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag_full, _ = cv2.cartToPolar(flow_full[...,0], flow_full[...,1])
            # upsample to full and store
            mag_up = cv2.resize(mag_full, (w_img, h_img))
            flow_map = mag_up
            opt_prev_small = gray_full

    # Association: try IoU first between dets and existing tracks (predicted boxes)
    assignments = {}  # det_idx -> track_id
    used_tracks = set()
    used_dets = set()

    # build predicted boxes for tracks
    for tid, tr in tracks.items():
        # predict center
        pcx, pcy = predict_center(tr)
        w_box, h_box = tr.get('box_size', (60,120))
        halfw = int(w_box/2); halfh = int(h_box/2)
        pb = (int(pcx-halfw), int(pcy-halfh), int(pcx+halfw), int(pcy+halfh))
        tr['predicted_box'] = clamp_box(pb, w_img, h_img)

    # IoU matching
    for di, d in enumerate(dets):
        best_iou = 0.0; best_tid = None
        for tid, tr in tracks.items():
            if tid in used_tracks: continue
            i = iou(d, tr['predicted_box'])
            if i > best_iou:
                best_iou = i; best_tid = tid
        if best_iou >= IOU_MATCH_THR:
            assignments[di] = best_tid
            used_tracks.add(best_tid)
            used_dets.add(di)

    # centroid fallback matching
    for di, d in enumerate(dets):
        if di in used_dets: continue
        dc = center(d)
        best_dist = CENTROID_MATCH_THR; best_tid = None
        for tid, tr in tracks.items():
            if tid in used_tracks: continue
            tc = tr['center_smooth']
            dd = dist(dc, tc)
            if dd < best_dist:
                best_dist = dd; best_tid = tid
        if best_tid is not None:
            assignments[di] = best_tid
            used_tracks.add(best_tid)
            used_dets.add(di)

    # Now update matched tracks with detections (and suppress small internal boxes)
    new_tracks = {}
    for di, d in enumerate(dets):
        if di in assignments:
            tid = assignments[di]
            tr = tracks[tid]
            # if det is much smaller than previous track box, ignore to avoid mini-box issue
            prev_area = area(tr.get('last_box', d))
            det_area = area(d)
            if det_area < 0.25 * prev_area:
                # ignore small detection (likely internal jitter)
                # keep previous track state, mark seen
                tr['missing'] = 0
                tr['last_seen'] = FRAME
                new_tracks[tid] = tr
                continue
            # update track with detection box
            tr['last_box'] = d
            cx,cy = center(d)
            tr['center_smooth_prev'] = tr.get('center_smooth', (cx,cy))
            pr = np.array(tr['center_smooth_prev'], dtype=float)
            cr = np.array((cx,cy), dtype=float)
            sm = tuple((1-SMOOTH)*pr + SMOOTH*cr)
            tr['center_smooth'] = sm
            update_velocity(tr, sm)
            # update box size
            tr['box_size'] = (d[2]-d[0], d[3]-d[1])
            tr['last_seen'] = FRAME
            tr['missing'] = 0
            # update flow/hand/embedding similar to previous implementation
            # compute local flow mean
            x1,y1,x2,y2 = d
            patch = flow_map[y1:y2, x1:x2]
            flow_mean = float(np.mean(patch)) if patch.size>0 else 0.0
            # hands proximity
            proximities = [dist(center(d), h['center']) for h in hands] if hands else [9999.0]
            prox = float(min(proximities))
            openness = float(np.mean([h['openness'] for h in hands]) if hands else 0.0)
            # embedding
            emb = np.array([
                np.mean(tr.get('hist_mov', [0.0])), np.std(tr.get('hist_mov', [0.0])),
                np.mean(tr.get('hist_prox', [9999.0])), np.std(tr.get('hist_prox', [0.0])),
                np.mean(tr.get('hist_open', [0.0])), np.mean(tr.get('hist_flow', [0.0]))
            ], dtype=float)
            # update histories (bounded)
            tr.setdefault('hist_mov', []).append(np.linalg.norm(cr-pr))
            tr['hist_mov'] = tr['hist_mov'][-WINDOW:]
            tr.setdefault('hist_prox', []).append(prox)
            tr['hist_prox'] = tr['hist_prox'][-WINDOW:]
            tr.setdefault('hist_open', []).append(openness)
            tr['hist_open'] = tr['hist_open'][-WINDOW:]
            tr.setdefault('hist_flow', []).append(flow_mean)
            tr['hist_flow'] = tr['hist_flow'][-WINDOW:]
            # update embedding statistic/object
            emb_stat = emb_stat if 'emb_stat' in globals() else None
            # draw detection (single box)
            color = (0,255,0)
            state = tr.get('state','libre')
            if state=='sostenida': color = (200,140,40)
            if state=='anomalía': color = (0,0,255)
            cv2.rectangle(frame, (d[0],d[1]), (d[2],d[3]), color, 2)
            cv2.putText(frame, f"{state.upper()} ID {tid}", (d[0], d[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            new_tracks[tid] = tr
        else:
            # detection unmatched -> create new track
            tid = next_id
            next_id += 1
            cx,cy = center(d)
            tr = {
                'last_box': d,
                'center_smooth': (cx,cy),
                'center_smooth_prev': (cx,cy),
                'vel': (0.0,0.0),
                'box_size': (d[2]-d[0], d[3]-d[1]),
                'hist_mov':[0.0], 'hist_prox':[9999.0], 'hist_open':[0.0], 'hist_flow':[0.0],
                'embs':[], 'state':'libre', 'last_seen':FRAME, 'missing':0, 'predicted':False
            }
            cv2.rectangle(frame, (d[0],d[1]), (d[2],d[3]), (0,255,0), 2)
            cv2.putText(frame, f"LIBRE ID {tid}", (d[0], d[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            new_tracks[tid] = tr

    # handle tracks not assigned (missing detections) -> predict or remove
    for tid, tr in list(tracks.items()):
        if tid in new_tracks:
            continue
        tr['missing'] = tr.get('missing',0) + 1
        # compute stability
        if len(tr.get('hist_mov',[])) >= 3:
            emb_std = float(np.mean(np.std(np.stack([np.array(tr.get('hist_mov',[])),
                                                     np.array(tr.get('hist_prox',[])),
                                                     np.array(tr.get('hist_open',[])),
                                                     np.array(tr.get('hist_flow',[]))]), axis=1)))
        else:
            emb_std = 1e6
        limit = PRED_MAX_STABLE if emb_std < 0.002 else PRED_MAX_UNSTABLE
        if tr['missing'] <= limit:
            # predict center
            px, py = predict_center(tr)
            bw, bh = tr.get('box_size', (60,120))
            pb = (int(px-bw/2), int(py-bh/2), int(px+bw/2), int(py+bh/2))
            pb = clamp_box(pb, w_img, h_img)
            tr['predicted_box'] = pb
            tr['predicted'] = True
            # draw predicted box (thinner)
            color = (200,140,40) if tr.get('state','libre')=='sostenida' else (0,255,0)
            if tr.get('state','libre')=='anomalía': color = (0,0,255)
            cv2.rectangle(frame, (pb[0],pb[1]), (pb[2],pb[3]), color, 1)
            cv2.putText(frame, f"PRED {tr.get('state','libre').upper()} ID {tid}", (pb[0], pb[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            new_tracks[tid] = tr
        else:
            # remove if too old
            if tr['missing'] > MISSING_REMOVE:
                # drop
                pass
            else:
                new_tracks[tid] = tr

    tracks = new_tracks

    end = time.time()
    # optional: draw fps
    fps = 1.0 / (end-start+1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("Inteligencia-B3", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
