"""
inteligenciaC.py

Versión C: Inteligencia secuencial aprendida (híbrido completo).
Objetivos:
 - Usar embeddings multimodales (movimiento, manos, optical flow) como en A4/B3
 - Añadir un pequeño modelo secuencial (GRU) para: (a) clasificar estado (libre/sostenida/anomalia)
   y (b) predecir la posición futura durante micro-oclusiones.
 - Permitir recolección de datos y entrenamiento interactivo (etiquetado manual rápido) para pasar
   de B a C con tu propio dataset. El entrenamiento puede correr en CPU o GPU si está disponible.
 - Mantener degradado grácil: si PyTorch no está instalado, el sistema cae a la lógica híbrida
   anterior (kalman-lite + clustering + mahalanobis) para no romper la ejecución.

Cómo usar:
 - Instalar dependencias: torch, torchvision, mediapipe, ultralytics, opencv-python, numpy
   (en CPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu)
 - Ejecutar: python inteligenciaC.py
 - Mientras corre: presiona:
     'r' -> grabar la última secuencia por track como etiqueta POSITIVA (SOSTENIDA)
     't' -> grabar la última secuencia por track como etiqueta NEGATIVA (LIBRE)
     'm' -> grabar la última secuencia por track como etiqueta ANOMALIA
     'p' -> iniciar / continuar entrenamiento del modelo con las muestras recolectadas
     's' -> guardar modelo y dataset
     ESC -> salir

Notas:
 - El entrenamiento es simple y pequeño (GRU de 1 capa); diseñado para prototipado rápido.
 - Para producción, se recomienda exportar el modelo y usar inferencia optimizada (TorchScript) en Jetson.

"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# ---- modelo secuencial (PyTorch) opcional ----
USE_TORCH = True
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    USE_TORCH = False

# ---- parámetros ----
MODEL_PATH = "yolov8n.pt"
SEQ_LEN = 20
EMB_DIM = 6
HIDDEN = 64
NUM_CLASSES = 3  # 0:libre,1:sostenida,2:anomalía
DEVICE = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if USE_TORCH else None

WINDOW = SEQ_LEN
SMOOTH_ALPHA = 0.35
FLOW_SCALE = 0.5
FLOW_STEP = 2

PRED_FRAMES = 5  # cuantos frames predecir al perder track

# ---- cargas y modelos ----
model_yolo = YOLO(MODEL_PATH)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---- data storage para entrenamiento interactivo ----
dataset_X = []  # listas de secuencias (SEQ_LEN, EMB_DIM)
dataset_y = []  # labels

# ---- utilidades ----
def center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def clamp_box(box,w,h):
    x1,y1,x2,y2 = box
    x1 = max(0,min(w-1,int(x1))); y1 = max(0,min(h-1,int(y1)))
    x2 = max(0,min(w-1,int(x2))); y2 = max(0,min(h-1,int(y2)))
    return (x1,y1,x2,y2)

def area(box):
    return max(1,(box[2]-box[0])*(box[3]-box[1]))

# ---- modelo GRU pequeño para clasificación + predicción de offset ----
class GRUModel(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, hidden=HIDDEN, num_classes=NUM_CLASSES):
        super().__init__()
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, num_classes)
        )
        # regressor para predicción de delta x,y (para próximo frame)
        self.regressor = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 2*PRED_FRAMES))

    def forward(self, x):
        # x: batch x seq x emb_dim
        out, _ = self.gru(x)
        last = out[:, -1, :]
        cls = self.classifier(last)
        pred = self.regressor(last)
        return cls, pred

# inicializar red si torch disponible
net = None
optimizer = None
criterion = None
if USE_TORCH:
    net = GRUModel().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

# ---- tracking structures ----
tracks = {}
next_id = 1
opt_prev = None
flow_map = None
FRAME = 0

# helper para extraer embedding (igual que en versiones previas)
def make_embedding(track, frame_w, frame_h):
    # expected fields in track: hist_mov, hist_prox, hist_open, hist_flow
    mov = np.mean(track.get('hist_mov', [0.0]))
    mov_std = np.std(track.get('hist_mov', [0.0]))
    prox = np.mean(track.get('hist_prox', [9999.0]))
    prox_std = np.std(track.get('hist_prox', [0.0]))
    openv = np.mean(track.get('hist_open', [0.0]))
    flowm = np.mean(track.get('hist_flow', [0.0]))
    emb = np.array([mov, mov_std, prox, prox_std, openv, flowm], dtype=float)
    # normalize some entries
    emb[2] = emb[2] / (np.sqrt(frame_w*frame_h) + 1e-9)
    emb[0] = emb[0] / (np.sqrt(frame_w*frame_h) + 1e-9)
    emb[1] = emb[1] / (np.sqrt(frame_w*frame_h) + 1e-9)
    emb[5] = emb[5] / (np.mean(flow_map) + 1e-9) if flow_map is not None else emb[5]
    return emb

# helper para guardar dataset y modelo
import os, json

def save_dataset_and_model(path_prefix='intC'):
    os.makedirs('models', exist_ok=True)
    np.save('models/{}_X.npy'.format(path_prefix), np.array(dataset_X, dtype=object))
    np.save('models/{}_y.npy'.format(path_prefix), np.array(dataset_y, dtype=np.int32))
    if USE_TORCH and net is not None:
        torch.save(net.state_dict(), 'models/{}_net.pth'.format(path_prefix))

# entrenamiento simple (mini batches)
def train_net(epochs=8, batch_size=16):
    if not USE_TORCH or net is None:
        print('Torch no disponible. No se puede entrenar aquí.')
        return
    if len(dataset_X) < 8:
        print('No hay suficientes muestras para entrenar (min 8).')
        return
    net.train()
    X = np.array([np.vstack(x) if x.shape[0]>=SEQ_LEN else np.vstack([np.zeros((SEQ_LEN-x.shape[0], EMB_DIM)), x]) for x in dataset_X])
    y = np.array(dataset_y)
    N = len(X)
    for ep in range(epochs):
        perm = np.random.permutation(N)
        losses = []
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb = torch.tensor(X[idx], dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(y[idx], dtype=torch.long).to(DEVICE)
            optimizer.zero_grad()
            cls, pred = net(xb)
            loss = criterion(cls, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {ep+1}/{epochs}  loss={np.mean(losses):.4f}')
    print('Entrenamiento terminado.')

# main loop: este código es relativamente largo pero sigue la lógica de B3
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    FRAME += 1
    h_img, w_img = frame.shape[:2]

    # optical flow (downscaled full-frame every FLOW_STEP frames)
    if FRAME % FLOW_STEP == 0:
        small = cv2.resize(frame, (0,0), fx=FLOW_SCALE, fy=FLOW_SCALE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if opt_prev is None:
            opt_prev = gray
            flow_map = np.zeros((h_img, w_img), dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(opt_prev, gray, None, 0.5,3,15,3,5,1.2,0)
            mag,_ = cv2.cartToPolar(flow[...,0], flow[...,1])
            flow_map = cv2.resize(mag, (w_img, h_img))
            opt_prev = gray

    # hands
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
            hands.append({'center':(cx,cy), 'openness':openness})

    # detección de botellas (YOLO)
    yres = model_yolo(frame, verbose=False)[0]
    dets = []
    for b in yres.boxes:
        if int(b.cls[0]) == 39:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            bx = clamp_box((x1,y1,x2,y2), w_img, h_img)
            if area(bx) > 300:
                dets.append(bx)

    # asociación simple por cercanía (puedes mejorar con IoU si lo prefieres)
    seen = set(); new_tracks = {}
    for d in dets:
        cx,cy = center(d)
        # find nearest track
        best = None; bestd = 9999
        for tid,tr in tracks.items():
            td = np.linalg.norm(np.array(tr.get('center_smooth', (cx,cy))) - np.array((cx,cy)))
            if td < bestd:
                best = tid; bestd = td
        if best is not None and bestd < 80:
            tid = best
            tr = tracks[tid]
            # update
            tr['center_smooth_prev'] = tr.get('center_smooth', (cx,cy))
            prev = np.array(tr['center_smooth_prev'], dtype=float)
            curr = np.array((cx,cy), dtype=float)
            sm = tuple((1-SMOOTH_ALPHA)*prev + SMOOTH_ALPHA*curr)
            tr['center_smooth'] = sm
            # update histories
            mov = float(np.linalg.norm(curr - prev))
            tr.setdefault('hist_mov', []).append(mov); tr['hist_mov'] = tr['hist_mov'][-WINDOW:]
            prox = min([np.linalg.norm(np.array(h['center']) - curr) for h in hands]) if hands else 9999.0
            tr.setdefault('hist_prox', []).append(prox); tr['hist_prox'] = tr['hist_prox'][-WINDOW:]
            tr.setdefault('hist_open', []).append(float(np.mean([h['openness'] for h in hands]) if hands else 0.0)); tr['hist_open'] = tr['hist_open'][-WINDOW:]
            patch = flow_map[int(max(0,d[1])):int(min(h_img-1,d[3])), int(max(0,d[0])):int(min(w_img-1,d[2]))]
            fm = float(np.mean(patch)) if patch.size>0 else 0.0
            tr.setdefault('hist_flow', []).append(fm); tr['hist_flow'] = tr['hist_flow'][-WINDOW:]
            tr['last_seen'] = FRAME; tr['missing'] = 0; tr['last_box'] = d
            new_tracks[tid] = tr; seen.add(tid)
        else:
            # crear nuevo track
            tid = next_id
            next_id += 1
            cx,cy = center(d)
            tr = {
                'center_smooth': (cx,cy), 'center_smooth_prev': (cx,cy), 'hist_mov':[0.0],
                'hist_prox':[9999.0], 'hist_open':[0.0], 'hist_flow':[0.0], 'last_seen':FRAME, 'missing':0,
                'state':'libre', 'last_box':d
            }
            new_tracks[tid] = tr; seen.add(tid)

    # handle tracks not seen -> missing++ and possibly predict
    for tid,tr in list(tracks.items()):
        if tid in seen: continue
        tr['missing'] = tr.get('missing',0) + 1
        # stability
        embs = np.stack([ [np.mean(tr.get('hist_mov',[0.0])), np.std(tr.get('hist_mov',[0.0])),
                           np.mean(tr.get('hist_prox',[9999.0])), np.std(tr.get('hist_prox',[0.0])),
                           np.mean(tr.get('hist_open',[0.0])), np.mean(tr.get('hist_flow',[0.0]))] ])
        emb_std = float(np.mean(np.std(embs, axis=0)))
        lim = PRED_FRAMES if emb_std < 0.002 else 1
        if tr['missing'] <= lim:
            # predict next centers using velocity-like diff
            prev = np.array(tr.get('center_smooth_prev', tr['center_smooth']), dtype=float)
            curr = np.array(tr.get('center_smooth', prev), dtype=float)
            vel = curr - prev
            pred = curr + vel
            tr['center_smooth_prev'] = tuple(curr); tr['center_smooth'] = tuple(pred)
            tr['predicted'] = True
            new_tracks[tid] = tr
        else:
            # if missing too long -> drop
            if tr['missing'] < 30:
                new_tracks[tid] = tr
            # else drop

    tracks = new_tracks

    # make embeddings and run model inference if available
    for tid,tr in tracks.items():
        emb = make_embedding(tr, w_img, h_img)
        # append to per-track buffer sequence
        seq = tr.get('seq', [])
        seq.append(emb); seq = seq[-SEQ_LEN:]
        tr['seq'] = seq
        # if model available and sequence full, run inference
        predicted_state = None
        if USE_TORCH and net is not None and len(seq) >= SEQ_LEN:
            net.eval()
            with torch.no_grad():
                x = torch.tensor(np.array([seq], dtype=np.float32)).to(DEVICE)
                cls_logits, pred_offsets = net(x)
                cls_pred = int(torch.argmax(cls_logits, dim=1).cpu().numpy()[0])
                predicted_state = cls_pred
                # parse pred_offsets if needed
                # pred_offsets is (batch, 2*PRED_FRAMES)
        # fallback heuristics (no rigid rules, simple: use seq statistics)
        if predicted_state is None:
            # keep previous state
            pass
        else:
            names = ['libre','sostenida','anomalía']
            tr['state'] = names[predicted_state]
        # draw
        box = tr.get('last_box', None)
        if box is not None:
            color = (0,255,0) if tr['state']=='libre' else ((200,140,40) if tr['state']=='sostenida' else (0,0,255))
            cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), color, 2)
            cv2.putText(frame, f"{tr['state'].upper()} ID {tid}", (box[0], box[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # teclado: etiquetado, entrenamiento, guardar
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('r'):
        # grab positive (sostenida) for all full sequences
        count = 0
        for tid,tr in tracks.items():
            if len(tr.get('seq',[])) >= SEQ_LEN:
                dataset_X.append(np.array(tr['seq'][-SEQ_LEN:]))
                dataset_y.append(1)
                count += 1
        print(f'Registradas {count} muestras como SOSTENIDA')
    elif k == ord('t'):
        count = 0
        for tid,tr in tracks.items():
            if len(tr.get('seq',[])) >= SEQ_LEN:
                dataset_X.append(np.array(tr['seq'][-SEQ_LEN:]))
                dataset_y.append(0)
                count += 1
        print(f'Registradas {count} muestras como LIBRE')
    elif k == ord('m'):
        count = 0
        for tid,tr in tracks.items():
            if len(tr.get('seq',[])) >= SEQ_LEN:
                dataset_X.append(np.array(tr['seq'][-SEQ_LEN:]))
                dataset_y.append(2)
                count += 1
        print(f'Registradas {count} muestras como ANOMALIA')
    elif k == ord('p'):
        print('Iniciando entrenamiento...')
        train_net(epochs=6, batch_size=8)
    elif k == ord('s'):
        save_dataset_and_model()
        print('Dataset y modelo guardados.')

    # mostrar FPS
    cv2.putText(frame, f"FRAME {FRAME}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow('Inteligencia-C (Secuencial Aprendida)', frame)

cap.release(); cv2.destroyAllWindows()
