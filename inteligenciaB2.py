# inteligenciaB2.py
# Mejoras sobre inteligenciaB:
# - 1) Mayor FPS: reducción de optical flow en tamaño (downscale) y menor frecuencia.
# - 2) Eliminación de la "mini bounding box" verde interna: se prohíbe dibujar YOLO box si la predicción ya dibuja.
# - 3) Se mantiene mismo sistema híbrido A→B.

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

# --- Configuración ---
MODEL = "yolov8n.pt"
WINDOW = 20
SMOOTH = 0.35
MAHAL_T = 4.0
CLUSTERS = 2

# Downscale para optical flow (mejora FPS)
FLOW_SCALE = 0.5
FLOW_STEP = 2   # calcular flow cada n frames

PRED_MAX_STABLE = 7
PRED_MAX_UNSTABLE = 2
MISSING_REMOVE = 25

model = YOLO(MODEL)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# --- Flow ---
flow_prev = None
flow_mag_full = None

# --- Estructuras ---
buffer = {}
next_id = 1
FRAME = 0
emb_global = []

# --- Utilidades ---
def center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)//2, (y1+y2)//2)

def dist(a,b):
    return float(np.linalg.norm(np.array(a)-np.array(b)))

class OnlineKMeans:
    def __init__(self,k=2,lr=0.12):
        self.k=k; self.lr=lr
        self.cent=[]; self.init=False
    def partial_fit(self,x):
        x=np.array(x,float)
        if not self.init:
            if len(self.cent)<self.k:
                self.cent.append(x.copy())
                if len(self.cent)==self.k and dist(self.cent[0],self.cent[1])>1e-3:
                    self.init=True
            return
        d=[np.linalg.norm(x-c) for c in self.cent]
        i=int(np.argmin(d))
        self.cent[i]= (1-self.lr)*self.cent[i] + self.lr*x
    def predict(self,x):
        if not self.init: return None
        x=np.array(x,float)
        d=[np.linalg.norm(x-c) for c in self.cent]
        return int(np.argmin(d))

class OG:
    def __init__(self,dim):
        self.n=0; self.mean=np.zeros(dim); self.M2=np.zeros((dim,dim)); self.dim=dim
    def update(self,x):
        x=np.array(x,float)
        self.n+=1
        d=x-self.mean; self.mean+=d/self.n
        d2=x-self.mean; self.M2+=np.outer(d,d2)
    def cov(self):
        if self.n<2: return np.eye(self.dim)*1e-6
        return self.M2/(self.n-1)+np.eye(self.dim)*1e-6
    def mahal(self,x):
        x=np.array(x,float)
        diff=x-self.mean
        try:
            inv=np.linalg.pinv(self.cov())
            return float(np.sqrt(diff.T@inv@diff))
        except: return float(np.linalg.norm(diff))

okm=OnlineKMeans(k=CLUSTERS)
emb_gauss=None

# --- predicción ---
def predict_pos(d):
    try:
        p=np.array(d['smooth_prev']); c=np.array(d['smooth'])
        v=c-p
        pr=c+v
        return (int(pr[0]),int(pr[1]))
    except: return tuple(map(int,d['smooth']))

# --- Loop ---
while True:
    ret,frame=cap.read()
    if not ret: break
    FRAME+=1

    h,w=frame.shape[:2]

    # HANDS -------------------------------------------------
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=hands_detector.process(rgb)
    hands=[]
    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            xs=[p.x*w for p in lm.landmark]
            ys=[p.y*h for p in lm.landmark]
            cx=np.mean(xs); cy=np.mean(ys)
            th=np.array([lm.landmark[4].x*w, lm.landmark[4].y*h])
            ind=np.array([lm.landmark[8].x*w, lm.landmark[8].y*h])
            op=float(np.linalg.norm(th-ind))
            hands.append({'c':(cx,cy),'o':op})

    # FLOW (cada FLOW_STEP frames) ---------------------------
    if FRAME % FLOW_STEP==0:
        small=cv2.resize(frame,(0,0),fx=FLOW_SCALE,fy=FLOW_SCALE)
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        if flow_prev is None:
            flow_prev=gray
            flow_mag_full=np.zeros((h,w),np.float32)
        else:
            flow=cv2.calcOpticalFlowFarneback(flow_prev,gray,None,0.5,3,15,3,5,1.2,0)
            mag,_=cv2.cartToPolar(flow[...,0],flow[...,1])
            f=cv2.resize(mag,(w,h))
            flow_mag_full=f
            flow_prev=gray

    # YOLO --------------------------------------------------
    yres=model(frame,verbose=False)[0]
    det=[]
    for b in yres.boxes:
        if int(b.cls[0])==39:
            x1,y1,x2,y2=map(int,b.xyxy[0])
            det.append((x1,y1,x2,y2))

    seen=set()
    new={}

    # ACTUALIZAR DETECCIONES -------------------------------
    for box in det:
        cx,cy=center(box)
        bid=None
        for oid,d in buffer.items():
            if dist((cx,cy),d['smooth'])<60:
                bid=oid; break
        if bid is None:
            bid=next_id; next_id+=1
            buffer[bid]={
                'smooth':(cx,cy),'smooth_prev':(cx,cy),'raw':(cx,cy),
                'tray':[],'prox':[],'open':[],'flow':[],
                'emb':[],'state':'libre','ma':0,'cl':None,
                'last':FRAME,'miss':0,'pred':False
            }
        d=buffer[bid]

        d['smooth_prev']=d['smooth']
        pr=np.array(d['smooth_prev']); cr=np.array((cx,cy))
        sm=tuple((1-SMOOTH)*pr + SMOOTH*cr)
        d['smooth']=sm; d['raw']=(cx,cy)

        # movimiento
        mv=float(np.linalg.norm(cr-pr))
        d['tray'].append(mv); d['tray'][-WINDOW:]

        # manos
        prox=[]; op=[]
        for hnd in hands:
            prox.append(dist((cx,cy),hnd['c']))
            op.append(hnd['o'])
        d['prox'].append(min(prox) if prox else 9999)
        d['open'].append(float(np.mean(op)) if op else 0)
        d['prox'][-WINDOW:]; d['open'][-WINDOW:]

        # flow
        x1,y1,x2,y2=box
        patch=flow_mag_full[y1:y2,x1:x2]
        fm=float(np.mean(patch)) if patch.size>0 else 0
        d['flow'].append(fm); d['flow'][-WINDOW:]

        # embedding
        e=[
            np.mean(d['tray']), np.std(d['tray']),
            np.mean(d['prox']), np.std(d['prox']),
            np.mean(d['open']), np.mean(d['flow'])
        ]
        e=np.array(e,float)
        e[2]/=np.sqrt(w*h); e[0]/=np.sqrt(w*h); e[1]/=np.sqrt(w*h)
        e[5]/=(np.mean(flow_mag_full)+1e-6)
        d['emb'].append(e); d['emb'][-WINDOW:]

        if emb_gauss is None:
            emb_gauss=OG(dim=len(e))

        emb_gauss.update(e)
        okm.partial_fit(e)
        cl=okm.predict(e)
        d['cl']=cl

        ma=emb_gauss.mahal(e); d['ma']=ma

        # estabilidad
        est=np.mean(np.std(np.stack(d['emb']),axis=0)) if len(d['emb'])>3 else 1e6

        # ESTADO sin reglas duras
        st='libre'; col=(0,255,0)
        if okm.init and cl is not None:
            c0=okm.cent[0]; c1=okm.cent[1]
            s0=-c0[2]+c0[0]; s1=-c1[2]+c1[0]
            sc=0 if s0>s1 else 1
            if cl==sc: st='sostenida'; col=(200,140,40)
        if ma>MAHAL_T:
            st='anomalia'; col=(0,0,255)
        d['state']=st

        # DIBUJO (solo visible, evita mini-caja)
        cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),col,3)
        cv2.putText(frame,f"{st} {bid}",(box[0],box[1]-7),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)

        d['last']=FRAME; d['miss']=0; d['pred']=False

        new[bid]=d
        seen.add(bid)

    # PREDICCIÓN Y LIMPIEZA ------------------------------
    for oid,d in buffer.items():
        if oid in seen: continue
        d['miss']+=1
        # estabilidad
        if len(d['emb'])>=3:
            est=np.mean(np.std(np.stack(d['emb']),axis=0))
        else:
            est=1e6
        lim=PRED_MAX_STABLE if est<0.002 else PRED_MAX_UNSTABLE

        if d['miss']<=lim:
            # predecir y dibujar SOLO predicción
            pr=predict_pos(d)
            d['pred']=True
            d['smooth_prev']=d['smooth']
            d['smooth']=pr

            st=d.get('state','libre')
            if st=='libre': col=(0,255,0)
            elif st=='sostenida': col=(200,140,40)
            elif st=='anomalia': col=(0,0,255)
            else: col=(0,255,255)

            size=50
            x,y=pr
            cv2.rectangle(frame,(x-size//2,y-size//2),(x+size//2,y+size//2),col,2)
            cv2.putText(frame,f"PRED {st} {oid}",(x-size//2,y-size//2-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)

            new[oid]=d
        else:
            if d['miss']<MISSING_REMOVE:
                new[oid]=d
            # si excede → se elimina automáticamente

    buffer=new

    cv2.imshow("Inteligencia-B2",frame)
    if cv2.waitKey(1)==27: break

cap.release(); cv2.destroyAllWindows()
