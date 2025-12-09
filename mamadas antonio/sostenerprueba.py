import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp
from scipy.spatial.distance import euclidean
import argparse
from collections import deque, defaultdict

# ------------------ Helpers ------------------
def cxcywh_from_xyxy(xyxy):
    x1,y1,x2,y2 = xyxy
    w = max(1, x2-x1)
    h = max(1, y2-y1)
    cx = x1 + w/2
    cy = y1 + h/2
    return (cx,cy,w,h)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter <= 0:
        return 0
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = areaA + areaB - inter
    return inter/denom if denom>0 else 0

# ------------------ Simple Tracker ------------------
class SimpleTracker:
    def __init__(self, max_age=5, dist_threshold=80):
        self.next_id = 0
        self.tracks = {}  # id -> {'centroid':(x,y), 'age':0}
        self.max_age = max_age
        self.dist_th = dist_threshold

    def update(self, detections):
        assigned = {}
        remaining = set(self.tracks.keys())
        dets = list(detections)

        # assign detections
        for det in dets:
            best_id = None
            best_d = None
            for tid in list(remaining):
                d = euclidean(det, self.tracks[tid]['centroid'])
                if best_d is None or d < best_d:
                    best_d = d
                    best_id = tid
            if best_d is not None and best_d <= self.dist_th:
                self.tracks[best_id]['centroid'] = det
                self.tracks[best_id]['age'] = 0
                assigned[best_id] = det
                remaining.remove(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'centroid': det, 'age': 0}
                assigned[tid] = det

        # age old tracks
        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        return [(tid, self.tracks[tid]['centroid']) for tid in self.tracks.keys()]

# ------------------ Main ------------------
def main(args):
    model = YOLO(args.model)

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50
    )

    # trackers
    hand_tracker = SimpleTracker(max_age=6, dist_threshold=60)
    bottle_tracker = SimpleTracker(max_age=6, dist_threshold=80)

    # hold counter
    hold_counter = defaultdict(int)
    HOLD_FRAMES_REQUIRED = args.hold_frames

    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print("No se pudo abrir la fuente.")
        return

    # FPS smoother
    fps_deque = deque(maxlen=30)

    frame_id = 0
    hand_bboxes = []  # últimas manos detectadas

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------------- YOLO detect ----------------
        res = model.predict(
            source=[frame],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=50,
            verbose=False
        )[0]

        persons = []
        bottles = []

        if hasattr(res, 'boxes'):
            for box in res.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                name = model.model.names[cls].lower()

                if name == "person":
                    persons.append((xyxy, float(box.conf[0])))
                elif name == "bottle":
                    bottles.append((xyxy, float(box.conf[0])))

        # ----------- MEDIAPIPE HANDS every 3 frames ---------
        if frame_id % 3 == 0:
            hand_bboxes = []
            mp_result = hands_detector.process(frame_rgb)
            if mp_result.multi_hand_landmarks:
                for hl in mp_result.multi_hand_landmarks:
                    xs = [lm.x for lm in hl.landmark]
                    ys = [lm.y for lm in hl.landmark]
                    x_min = int(min(xs)*W); x_max = int(max(xs)*W)
                    y_min = int(min(ys)*H); y_max = int(max(ys)*H)
                    padx = int((x_max-x_min)*0.2)+2
                    pady = int((y_max-y_min)*0.2)+2
                    x1 = max(0, x_min-padx); x2 = min(W-1, x_max+padx)
                    y1 = max(0, y_min-pady); y2 = min(H-1, y_max+pady)
                    hand_bboxes.append((x1,y1,x2,y2))

        frame_id += 1

        # ---------------- Track hands / bottles ----------------
        hand_centroids = []
        for hb in hand_bboxes:
            c = cxcywh_from_xyxy(hb)
            hand_centroids.append((c[0],c[1]))

        bottle_centroids = []
        for bb, _ in bottles:
            c = cxcywh_from_xyxy(bb)
            bottle_centroids.append((c[0],c[1]))

        hand_tracker.update(hand_centroids)
        bottle_tracker.update(bottle_centroids)

        # map tracker results
        hand_map = {}
        for tid in hand_tracker.tracks.keys():
            hand_map[tid] = {
                "centroid": hand_tracker.tracks[tid]["centroid"],
                "bbox": None
            }

        # assign nearest bbox per hand ID
        for tid in hand_map.keys():
            hx,hy = hand_map[tid]["centroid"]
            best_idx = None
            best_d = None
            for i,hb in enumerate(hand_bboxes):
                c = cxcywh_from_xyxy(hb)
                d = euclidean((hx,hy),(c[0],c[1]))
                if best_d is None or d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx is not None and best_idx < len(hand_bboxes):
                hand_map[tid]["bbox"] = hand_bboxes[best_idx]

        bottle_map = {}
        for tid in bottle_tracker.tracks.keys():
            bottle_map[tid] = {"centroid": bottle_tracker.tracks[tid]["centroid"], "bbox": None}

        for tid in bottle_map.keys():
            bx,by = bottle_map[tid]["centroid"]
            best_idx = None
            best_d = None
            for i,(bb,_) in enumerate(bottles):
                c = cxcywh_from_xyxy(bb)
                d = euclidean((bx,by),(c[0],c[1]))
                if best_d is None or d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx is not None:
                bottle_map[tid]["bbox"] = bottles[best_idx][0]

        # ------------- HOLDING logic -------------
        pairs_marked = set()

        for hid, hinfo in hand_map.items():
            hcx, hcy = hinfo["centroid"]
            hbbox = hinfo["bbox"]

            # mano diagonal
            if hbbox is not None:
                hx1,hy1,hx2,hy2 = hbbox
                hdiag = np.hypot(hx2-hx1, hy2-hy1)
            else:
                hdiag = 60

            for bid, binfo in bottle_map.items():
                bcx, bcy = binfo["centroid"]
                bbbox = binfo["bbox"]

                if bbbox is not None:
                    bx1,by1,bx2,by2 = bbbox
                    bdiag = np.hypot(bx2-bx1, by2-by1)
                    bh = by2-by1
                else:
                    bdiag = 40
                    bh = 40

                dist = euclidean((hcx,hcy),(bcx,bcy))
                dist_th = max(hdiag,bdiag) * args.dist_k
                cond_dist = dist <= dist_th

                cond_overlap = False
                if hbbox is not None and bbbox is not None:
                    if (hx1<=bcx<=hx2 and hy1<=bcy<=hy2): cond_overlap = True
                    if (bx1<=hcx<=bx2 and by1<=hcy<=by2): cond_overlap = True
                    if iou(hbbox,bbbox) > 0.01: cond_overlap = True

                cond_vertical = abs(bcy-hcy) <= max(hdiag,bh)*1.5

                holding_now = cond_dist and (cond_overlap or cond_vertical)

                if holding_now:
                    hold_counter[(hid,bid)] += 1
                else:
                    hold_counter[(hid,bid)] = 0

                if hold_counter[(hid,bid)] >= HOLD_FRAMES_REQUIRED:
                    pairs_marked.add((hid,bid))

        # ------------ map holding to persons ------------
        person_holding = set()

        for hid,bid in pairs_marked:
            hcx,hcy = hand_map[hid]["centroid"]
            best_p = None
            best_d = None
            for i,(pxyxy,_) in enumerate(persons):
                pcx,pcy,_,_ = cxcywh_from_xyxy(pxyxy)
                d = euclidean((hcx,hcy),(pcx,pcy))
                if best_d is None or d<best_d:
                    best_d = d
                    best_p = i
            if best_p is not None:
                person_holding.add(best_p)

        # ------------ Visualization ------------
        vis = frame.copy()

        # persons: yellow / red
        for idx,(xyxy,conf) in enumerate(persons):
            x1,y1,x2,y2 = map(int,xyxy)

            color = (0,255,255)  # yellow
            if idx in person_holding:
                color = (0,0,255)  # RED

            cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
            cv2.putText(vis,f"P{idx}",(x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        # bottles
        for (bb,conf) in bottles:
            x1,y1,x2,y2 = map(int,bb)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,180,255),2)

        # hands
        for hb in hand_bboxes:
            x1,y1,x2,y2 = map(int,hb)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,100),2)

        # FPS
        t1 = time.time()
        fps_deque.append(1/(t1-t0+1e-6))
        fps = sum(fps_deque)/len(fps_deque)
        cv2.putText(vis,f"FPS:{fps:.1f}",(10,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

        cv2.imshow("HOLD DETECTOR",vis)
        if cv2.waitKey(1)&0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------ CLI ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default="yolov9t.pt")
    parser.add_argument("--video",type=str,default=None)
    parser.add_argument("--imgsz",type=int,default=640)
    parser.add_argument("--conf",type=float,default=0.25)
    parser.add_argument("--iou",type=float,default=0.45)
    parser.add_argument("--dist_k",type=float,default=1.2)
    parser.add_argument("--hold_frames",type=int,default=3)
    args = parser.parse_args()
    main(args)
