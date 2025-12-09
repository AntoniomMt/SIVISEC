# hold_detector.py
import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp
from scipy.spatial.distance import euclidean
import argparse
from collections import deque, defaultdict

# ---------- Helpers ----------
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
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0,(boxA[2]-boxA[0])) * max(0,(boxA[3]-boxA[1]))
    boxBArea = max(0,(boxB[2]-boxB[0])) * max(0,(boxB[3]-boxB[1]))
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom>0 else 0.0

# Simple nearest-neighbor tracker for objects (hand/bottle)
class SimpleTracker:
    def __init__(self, max_age=5, dist_threshold=80):
        self.next_id = 0
        self.tracks = {}  # id -> {'centroid':(x,y), 'age':0}
        self.max_age = max_age
        self.dist_th = dist_threshold

    def update(self, detections):  # detections: list of centroids
        assigned = {}
        remaining_ids = set(self.tracks.keys())
        # Build cost matrix (naive greedily)
        dets = list(detections)
        for di, det in enumerate(dets):
            # find nearest existing track
            best_id = None
            best_d = None
            for tid in list(remaining_ids):
                d = euclidean(det, self.tracks[tid]['centroid'])
                if best_d is None or d < best_d:
                    best_d = d
                    best_id = tid
            if best_id is not None and best_d <= self.dist_th:
                # assign
                self.tracks[best_id]['centroid'] = det
                self.tracks[best_id]['age'] = 0
                assigned[best_id] = det
                remaining_ids.remove(best_id)
            else:
                # create new track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'centroid': det, 'age': 0}
                assigned[tid] = det
        # age unassigned tracks
        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
        # return list of (id, centroid)
        return [(tid, self.tracks[tid]['centroid']) for tid in self.tracks.keys()]

# ---------- Main detector ----------
def main(args):
    # Load YOLOv9s (Ultralytics will download if needed)
    model = YOLO(args.model)  # e.g. "yolov9s.pt"
    # MediaPipe hands
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=2,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    # trackers
    hand_tracker = SimpleTracker(max_age=6, dist_threshold=60)
    bottle_tracker = SimpleTracker(max_age=6, dist_threshold=80)

    # hold counters for pairs: (hand_id, bottle_id) -> consecutive frames
    hold_counter = defaultdict(int)
    HOLD_FRAMES_REQUIRED = args.hold_frames  # e.g. 3

    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print("No se pudo abrir la fuente.")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # main loop
    fps_deque = deque(maxlen=30)
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1) YOLO detect persons and bottles
        results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=100, verbose=False)
        # results is list with one result
        res = results[0]
        persons = []
        bottles = []
        # ultralytics result boxes: res.boxes.xyxy, res.boxes.conf, res.boxes.cls
        if hasattr(res, 'boxes') and len(res.boxes) > 0:
            for box in res.boxes:
                # convert to python numbers
                xyxy = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                name = model.model.names[cls] if hasattr(model, 'model') and hasattr(model.model, 'names') else str(cls)
                # keep person and bottle
                if name.lower() in ['person','personas','persona','personaje']:
                    persons.append((xyxy, conf))
                if name.lower() in ['bottle','botella']:
                    bottles.append((xyxy, conf))

        # 2) Mediapipe detect hands (returns landmarks -> convert to bbox)
        hand_bboxes = []
        mp_results = hands_detector.process(frame_rgb)
        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                # normalize to image coords
                x_min = int(min(xs) * W)
                x_max = int(max(xs) * W)
                y_min = int(min(ys) * H)
                y_max = int(max(ys) * H)
                # expand a little
                padx = int((x_max - x_min) * 0.2) + 2
                pady = int((y_max - y_min) * 0.2) + 2
                x1 = max(0, x_min - padx)
                y1 = max(0, y_min - pady)
                x2 = min(W-1, x_max + padx)
                y2 = min(H-1, y_max + pady)
                hand_bboxes.append((x1,y1,x2,y2))

        # Convert bboxes to centroids for trackers
        hand_centroids = []
        for hb in hand_bboxes:
            c = cxcywh_from_xyxy(hb)
            hand_centroids.append((c[0],c[1]))
        bottle_centroids = []
        for bb, conf in bottles:
            c = cxcywh_from_xyxy(bb)
            bottle_centroids.append((c[0],c[1]))

        # Update trackers
        hand_assigned = hand_tracker.update(hand_centroids)  # list of (id,centroid)
        bottle_assigned = bottle_tracker.update(bottle_centroids)

        # Build maps id -> centroid and id -> bbox (approx)
        hand_map = {hid: {'centroid': hand_tracker.tracks[hid]['centroid'], 'bbox': hand_bboxes[idx] if idx < len(hand_bboxes) else None}
                    for idx,hid in enumerate(hand_tracker.tracks.keys())}
        # For bottles, we need bbox mapping - create simple mapping by nearest match
        # Make list of bottle bboxes matched to bottle_tracker.tracks by nearest centroid
        bottle_map = {}
        for bid in bottle_tracker.tracks.keys():
            c = bottle_tracker.tracks[bid]['centroid']
            # find nearest original bottle bbox
            best_idx = None
            best_d = None
            for i,(bb,conf) in enumerate(bottles):
                bcent = cxcywh_from_xyxy(bb)
                d = euclidean(c, (bcent[0], bcent[1]))
                if best_d is None or d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx is not None:
                bottle_map[bid] = {'centroid': bottle_tracker.tracks[bid]['centroid'], 'bbox': bottles[best_idx][0]}
            else:
                bottle_map[bid] = {'centroid': bottle_tracker.tracks[bid]['centroid'], 'bbox': None}

        # 3) For every hand-bottle pair evaluate geometric rules
        pairs_marked = set()
        for hid, hinfo in hand_map.items():
            hcx, hcy = hinfo['centroid']
            hbbox = hinfo.get('bbox')
            # estimate hand diagonal size
            if hbbox is not None:
                hx1,hy1,hx2,hy2 = hbbox
                hand_diag = np.hypot(hx2-hx1, hy2-hy1)
            else:
                hand_diag = 60.0
            for bid, binfo in bottle_map.items():
                bcx, bcy = binfo['centroid']
                bbbox = binfo.get('bbox')
                if bbbox is not None:
                    bx1,by1,bx2,by2 = bbbox
                    bottle_diag = np.hypot(bx2-bx1, by2-by1)
                    bottle_h = max(1, by2-by1)
                else:
                    bottle_diag = 40.0
                    bottle_h = 40.0

                # RULE A: distance threshold relative to sizes
                dist = euclidean((hcx,hcy),(bcx,bcy))
                dist_thresh = max(hand_diag, bottle_diag) * args.dist_k  # default k=1.2
                cond_dist = dist <= dist_thresh

                # RULE B: hand bbox contains bottle center OR IoU > small threshold
                cond_overlap = False
                if hbbox and bbbox is not None:
                    # bottle center inside hand bbox?
                    if (bx1 <= hcx <= bx2) and (by1 <= hcy <= by2):
                        pass # irrelevant
                    # bottle center inside hand bbox?
                    if (bx1 <= hcx <= bx2) and (by1 <= hcy <= by2):
                        cond_overlap = True
                    # check bottle center in hand bbox
                    if (hx1 <= bcx <= hx2) and (hy1 <= bcy <= hy2):
                        cond_overlap = True
                    # IoU
                    if iou(hbbox, bbbox) > 0.01:
                        cond_overlap = True

                # RULE C: vertical/lateral coherence (bottle roughly at same vertical band as hand)
                # Bottle center y should be near hand center y +/- some factor of hand height
                cond_vertical = abs(bcy - hcy) <= max(hand_diag, bottle_h) * 1.5

                # Final decision for this frame
                holding_frame = (cond_dist and (cond_overlap or cond_vertical))

                # increment temporal counter
                if holding_frame:
                    hold_counter[(hid,bid)] += 1
                else:
                    hold_counter[(hid,bid)] = 0

                if hold_counter[(hid,bid)] >= HOLD_FRAMES_REQUIRED:
                    pairs_marked.add((hid,bid))

        # 4) Visualization
        vis = frame.copy()
        # draw person boxes
        for (xyxy, conf) in persons:
            x1,y1,x2,y2 = map(int, xyxy)
            cv2.rectangle(vis, (x1,y1),(x2,y2),(200,200,200),1)
            cv2.putText(vis, f"person {conf:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1)

        # draw bottles
        for (bb, conf) in bottles:
            x1,y1,x2,y2 = map(int, bb)
            cv2.rectangle(vis, (x1,y1),(x2,y2),(0,180,255),2)
            cv2.putText(vis, f"bottle {conf:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,255),1)

        # draw hands
        for hb in hand_bboxes:
            x1,y1,x2,y2 = map(int,hb)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,100),2)
            # draw centroid
            c = cxcywh_from_xyxy(hb)
            cv2.circle(vis, (int(c[0]),int(c[1])), 3, (0,255,100), -1)

        # draw tracked ids and holding
        for hid,hinfo in hand_map.items():
            hx,hy = map(int, hinfo['centroid'])
            cv2.putText(vis, f"H{hid}", (hx+6,hy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,100),1)
        for bid,binfo in bottle_map.items():
            bx,by = map(int, binfo['centroid'])
            cv2.putText(vis, f"B{bid}", (bx+6,by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,255),1)

        # draw holding pairs
        for hid,bid in pairs_marked:
            hcent = tuple(map(int, hand_map[hid]['centroid']))
            bcent = tuple(map(int, bottle_map[bid]['centroid']))
            cv2.line(vis, hcent, bcent, (0,0,255), 2)
            cv2.putText(vis, "HOLDING", ( (hcent[0]+bcent[0])//2, (hcent[1]+bcent[1])//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        # display FPS
        t1 = time.time()
        fps_deque.append(1.0/(t1-t0+1e-6))
        fps = sum(fps_deque)/len(fps_deque)
        cv2.putText(vis, f"FPS: {fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)

        cv2.imshow("hold_detector", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov9t.pt", help="Modelo yolov9s (ultralytics).")
    parser.add_argument("--video", type=str, default=None, help="Ruta a video (opcional).")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--dist_k", type=float, default=1.2, help="Multiplicador de umbral de distancia relativo.")
    parser.add_argument("--hold_frames", type=int, default=3, help="Frames consecutivos para confirmar hold.")
    args = parser.parse_args()
    main(args)
