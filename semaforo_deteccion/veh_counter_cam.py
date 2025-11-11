#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================
#  VehCam (FlowGate-Smooth): Vehículos EN MOVIMIENTO con YOLOv8 + Optical Flow + Kalman + EMA
#  --------------------------------------------------------------------------------------------------
#  - SOLO cuenta vehículos que se mueven (gating por flujo óptico Farneback).
#  - Tracking más FLUIDO: Kalman (const-vel) por track + suavizado EMA de la caja (w,h).
#  - Matching por distancia al centroide PREDICTO del Kalman (más estable que distancia bruta).
#  - ROI opcional, ventana única, y salidas de conteo/flag.
#  - Optimizado para FPS: FP16 opcional, flujo en resolución reducida, det_every.
#
#  Ejemplo (GPU 0 potente):
#    python3 veh_counter_cam.py --src 0 --display \
#      --yolo_model yolov8m.pt --imgsz 960 --device 0 --half \
#      --veh_conf 0.50 --nms_iou 0.65 \
#      --width 1280 --height 720 --fps 30 \
#      --mask roi_lanes.png \
#      --det_every 3 --flow_thr 1.0 --flow_frac 0.05 --flow_down 2 \
#      --max_disappeared 25 --max_distance 180 \
#      --ema_alpha 0.35 \
#      --write_count /tmp/veh_moving.txt \
#      --flag_out /tmp/veh_flag.txt --flag_threshold 4
# ==================================================================================================

from __future__ import annotations
import os, cv2, time, json, argparse, math
import numpy as np
from typing import Optional, List, Dict, Tuple

cv2.setUseOptimized(True)

ALLOWED_NAME_GROUPS = {"vehiculo": {"car","motorcycle","bus","truck","bicycle","train","van"}}
COLOR = {"vehiculo": (80,220,100), None: (220,220,220)}

# ---------------------------------- Utilidades geométricas ----------------------------------
def center_in_mask(xyxy, mask_bin) -> bool:
    if mask_bin is None: return True
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cx, cy = (x1 + x2)//2, (y1 + y2)//2
    h, w = mask_bin.shape[:2]
    return 0 <= cx < w and 0 <= cy < h and mask_bin[cy, cx] > 0

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter/union if union > 0 else 0.0

def draw_box(frame, xyxy, label=None, conf=None):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    color = COLOR.get(label, (220, 220, 220))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    txt = []
    if label: txt.append(label)
    if conf is not None: txt.append(f"{conf:.2f}")
    if txt:
        txts = " | ".join(txt)
        (tw, th), _ = cv2.getTextSize(txts, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, y0), (x1 + tw + 6, y0 + th + 6), color, -1)
        cv2.putText(frame, txts, (x1 + 3, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

# ---------------------------------- YOLO wrapper ----------------------------------
class YOLODetector:
    def __init__(self, model_path="yolov8s.pt", conf=0.5, imgsz=960, classes: Optional[List[str]]=None, device=None, nms_iou=0.60, half=False):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = float(conf); self.imgsz = int(imgsz); self.device = device; self.nms_iou = float(nms_iou); self.half = bool(half)
        try:
            import torch, torch.backends.cudnn as cudnn
            if (device is not None and str(device) != "cpu") and torch.cuda.is_available():
                cudnn.benchmark = True
        except Exception:
            pass
        self.names = self.model.model.names if hasattr(self.model, "model") else self.model.names
        if classes is not None:
            if isinstance(self.names, dict):
                name_to_idx = {v: k for k, v in self.names.items()}
            else:
                name_to_idx = {self.names[i]: i for i in range(len(self.names))}
            self.allowed_indices = sorted([int(name_to_idx[n]) for n in classes if n in name_to_idx])
        else:
            self.allowed_indices = None

    def infer(self, frame_bgr):
        res = self.model.predict(frame_bgr, conf=self.conf, iou=self.nms_iou, imgsz=self.imgsz,
                                 verbose=False, classes=self.allowed_indices, device=self.device, half=self.half)
        r0 = res[0]
        xyxy = []
        if r0.boxes is not None and len(r0.boxes) > 0:
            X = r0.boxes.xyxy.cpu().numpy()
            for i in range(len(X)): xyxy.append(X[i].tolist())
        return xyxy

# ---------------------------------- Optical Flow gate ----------------------------------
class FlowGate:
    def __init__(self, thr_mag=1.0, thr_frac=0.05, down=2):
        self.prev_small = None
        self.thr_mag = float(thr_mag); self.thr_frac = float(thr_frac); self.down = max(1, int(down))

    def update_and_check(self, gray_full, boxes, mask=None):
        if self.down > 1:
            small = cv2.resize(gray_full, (gray_full.shape[1]//self.down, gray_full.shape[0]//self.down), interpolation=cv2.INTER_AREA)
            mask_small = None if mask is None else cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            small = gray_full; mask_small = mask
        moving_flags = [False]*len(boxes)
        if self.prev_small is None:
            self.prev_small = small.copy(); return moving_flags
        flow = cv2.calcOpticalFlowFarneback(self.prev_small, small, None, 0.5, 3, 19, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=False)
        Hs, Ws = small.shape[:2]
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(v)//self.down for v in b]
            x1 = max(0, min(Ws-1, x1)); y1 = max(0, min(Hs-1, y1))
            x2 = max(0, min(Ws,   x2)); y2 = max(0, min(Hs,   y2))
            if x2 <= x1 or y2 <= y1: continue
            dx = int(0.1*(x2-x1)); dy = int(0.1*(y2-y1))
            x1i, y1i = x1+dx, y1+dy; x2i, y2i = x2-dx, y2-dy
            if x2i<=x1i or y2i<=y1i: continue
            roi_mag = mag[y1i:y2i, x1i:x2i]
            if roi_mag.size == 0: continue
            if mask_small is not None:
                roi_mask = mask_small[y1i:y2i, x1i:x2i]
                if roi_mask.size == 0: continue
                roi_mag = roi_mag[roi_mask>0]
                if roi_mag.size == 0: continue
            frac = float((roi_mag > self.thr_mag).sum()) / float(roi_mag.size)
            moving_flags[i] = (frac >= self.thr_frac)
        self.prev_small = small.copy()
        return moving_flags

# ---------------------------------- Kalman (const-vel en 2D) ----------------------------------
class KalmanCV2D:
    def __init__(self, x, y, dt=1.0, q=1.0, r=6.0):
        # Estado: [x, y, vx, vy]
        self.x = np.array([[float(x)], [float(y)], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100.0
        self.q = float(q); self.r = float(r)
        self.last_t = time.time()
        self._set_mats(dt)

    def _set_mats(self, dt):
        dt = float(max(1e-3, dt))
        self.F = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1, 0],
                           [0,0,0, 1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=np.float32)
        # Ruido de proceso (proporcional a dt)
        s = self.q * np.array([[dt**4/4, 0, dt**3/2, 0],
                               [0, dt**4/4, 0, dt**3/2],
                               [dt**3/2, 0, dt**2, 0],
                               [0, dt**3/2, 0, dt**2]], dtype=np.float32)
        self.Q = s
        self.R = np.eye(2, dtype=np.float32) * self.r

    def predict(self):
        t = time.time(); dt = t - self.last_t; self.last_t = t
        self._set_mats(dt)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0,0]), float(self.x[1,0])

    def update(self, zx, zy):
        z = np.array([[float(zx)],[float(zy)]], dtype=np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

# ---------------------------------- Tracker con Kalman + EMA ----------------------------------
class SmoothTracker:
    def __init__(self, max_disappeared=25, max_distance=180.0, ema_alpha=0.35):
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}  # id -> dict(kf, bbox, flags, disappeared)
        self.max_disappeared = int(max_disappeared)
        self.max_distance = float(max_distance)
        self.alpha = float(ema_alpha)

    @staticmethod
    def _centroid(b):
        x1,y1,x2,y2 = b
        return int((x1+x2)//2), int((y1+y2)//2)

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def update(self, boxes: List[List[int]], moving_flags: List[bool]):
        # Asegura enteros
        boxes = [[int(v) for v in b] for b in boxes]
        det_cs = [self._centroid(b) for b in boxes]

        # 1) Predict paso Kalman para todos
        preds = {}
        for tid, tr in self.tracks.items():
            px, py = tr['kf'].predict()
            preds[tid] = (int(px), int(py))

        # 2) Matching greedy por distancia a predicción
        assigned = set()
        updates = {}
        track_ids = list(self.tracks.keys())
        if preds and det_cs:
            D = np.zeros((len(track_ids), len(det_cs)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                for j, dc in enumerate(det_cs):
                    D[i,j] = self._dist(preds[tid], dc)
            for _ in range(min(D.shape)):
                i, j = np.unravel_index(np.argmin(D), D.shape)
                if D[i,j] > self.max_distance: break
                tid = track_ids[i]
                updates[tid] = j
                assigned.add(j)
                D[i,:] = 1e9; D[:,j] = 1e9

        # 3) Actualizar tracks asignados
        for tid, j in updates.items():
            cx, cy = det_cs[j]
            self.tracks[tid]['kf'].update(cx, cy)
            # EMA de la caja para suavizar tamaño
            if self.tracks[tid]['bbox'] is None:
                self.tracks[tid]['bbox'] = boxes[j]
            else:
                prev = np.array(self.tracks[tid]['bbox'], dtype=np.float32)
                curr = np.array(boxes[j], dtype=np.float32)
                sm = self.alpha*curr + (1.0-self.alpha)*prev
                self.tracks[tid]['bbox'] = [int(v) for v in sm]
            self.tracks[tid]['flags'].append(moving_flags[j])
            self.tracks[tid]['disappeared'] = 0

        # 4) Crear tracks nuevos para detecciones no asignadas
        for j in range(len(boxes)):
            if j in assigned: continue
            cx, cy = det_cs[j]
            kf = KalmanCV2D(cx, cy, dt=1.0, q=1.2, r=6.0)
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {'kf': kf, 'bbox': boxes[j], 'flags': [moving_flags[j]], 'disappeared': 0}

        # 5) Marcar desaparecidos y borrar si excede umbral
        for tid in list(self.tracks.keys()):
            if tid not in updates and (tid in self.tracks):
                self.tracks[tid]['disappeared'] += 1
                if self.tracks[tid]['disappeared'] > self.max_disappeared:
                    self.tracks.pop(tid, None)

        # Salidas
        bboxes = {tid: self.tracks[tid]['bbox'] for tid in self.tracks.keys() if self.tracks[tid]['bbox'] is not None}
        return self.tracks, bboxes

    def moving_tracks_count(self, min_recent=3, require_ratio=0.5):
        count = 0
        for tid, tr in self.tracks.items():
            flags = tr['flags'][-max(min_recent,1):]
            if len(flags)==0: continue
            if (sum(1 for f in flags if f) / float(len(flags))) >= require_ratio:
                count += 1
        return count

# ---------------------------------- Main ----------------------------------
def main():
    ap = argparse.ArgumentParser(description="VehCam (FlowGate-Smooth): YOLO+OpticalFlow+Kalman+EMA")
    ap.add_argument("--src", required=True, help="Índice de cámara (0,1,...)")
    ap.add_argument("--display", action="store_true", help="Mostrar ventana única")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0, help="Forzar hilos OpenCV (0=auto)")

    ap.add_argument("--yolo_model", type=str, default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--nms_iou", type=float, default=0.60)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--half", action="store_true", help="FP16 en CUDA")
    ap.add_argument("--veh_conf", type=float, default=0.50)
    ap.add_argument("--det_every", type=int, default=3, help="Inferir cada N frames (1=cada frame)")

    ap.add_argument("--mask", type=str, default=None, help="PNG/BMP 8-bit (blanco=ROI carriles)")

    ap.add_argument("--flow_thr", type=float, default=1.0, help="Umbral de magnitud (px/frame)")
    ap.add_argument("--flow_frac", type=float, default=0.05, help="Fracción mínima de píxeles > thr en la caja")
    ap.add_argument("--flow_down", type=int, default=2, help="Factor de reducción para flujo (2 => 1/2 resolución)")

    ap.add_argument("--max_disappeared", type=int, default=25)
    ap.add_argument("--max_distance", type=float, default=180.0)
    ap.add_argument("--ema_alpha", type=float, default=0.35, help="Suavizado EMA de bbox (0.0=no suaviza, 1.0=solo nueva)")

    ap.add_argument("--write_count", type=str, default=None)
    ap.add_argument("--flag_out", type=str, default=None)
    ap.add_argument("--flag_threshold", type=int, default=4)

    args = ap.parse_args()
    if args.threads and args.threads > 0:
        try: cv2.setNumThreads(int(args.threads))
        except Exception: pass

    # ROI
    mask_bin = None
    if args.mask and os.path.exists(args.mask):
        mv = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mv is not None:
            mask_bin = cv2.threshold(mv, 127, 255, cv2.THRESH_BINARY)[1]

    # YOLO vehículos
    classes = list(ALLOWED_NAME_GROUPS["vehiculo"])
    yolo = YOLODetector(model_path=args.yolo_model, conf=args.veh_conf, imgsz=args.imgsz,
                        classes=classes, device=args.device, nms_iou=args.nms_iou, half=args.half)

    # Flow + Tracker suave
    fgate = FlowGate(thr_mag=args.flow_thr, thr_frac=args.flow_frac, down=args.flow_down)
    tracker = SmoothTracker(max_disappeared=args.max_disappeared, max_distance=args.max_distance, ema_alpha=args.ema_alpha)

    # Cámara
    src = int(args.src) if str(args.src).isdigit() else args.src
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[VehCam] No pude abrir cámara: {args.src}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass

    if args.display:
        cv2.namedWindow("VehCam", cv2.WINDOW_NORMAL)

    last_flag = None
    frame_id = 0
    cache_boxes = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mask_r = None if mask_bin is None else cv2.resize(mask_bin, (args.width, args.height), interpolation=cv2.INTER_NEAREST)

            # YOLO (cada N frames)
            do_det = (frame_id % max(1, int(args.det_every))) == 0
            if do_det or not cache_boxes:
                boxes = yolo.infer(frame)
                boxes = [b for b in boxes if center_in_mask(b, mask_r)]
                cache_boxes = boxes
            else:
                boxes = cache_boxes

            # Gating por flujo: ¿cuáles se mueven?
            moving_flags = fgate.update_and_check(gray, boxes, mask=mask_r)
            moving_boxes = [b for b, mv in zip(boxes, moving_flags) if mv]

            # Tracking suave (Kalman + EMA)
            tracks, bboxes = tracker.update(moving_boxes, [True]*len(moving_boxes))
            moving_count = tracker.moving_tracks_count(min_recent=3, require_ratio=0.5)

            # Salidas
            print(json.dumps({"ts": time.time(), "moving": int(moving_count)}), flush=True)
            if args.write_count:
                try:
                    with open(args.write_count, "w") as f:
                        f.write(str(int(moving_count)))
                except Exception as e:
                    print(f"[VehCam] No pude escribir conteo: {e}")

            if args.flag_out:
                flag_val = 1 if moving_count >= int(args.flag_threshold) else 0
                if flag_val != last_flag:
                    try:
                        with open(args.flag_out, "w") as f:
                            f.write(str(flag_val))
                        print(json.dumps({"event":"VEHICLE_FLAG","value":flag_val,"ts":time.time()}), flush=True)
                        last_flag = flag_val
                    except Exception as e:
                        print(f"[VehCam] No pude escribir flag: {e}")

            # Render
            if args.display:
                out = frame.copy()
                if mask_r is not None:
                    inv = cv2.bitwise_not(mask_r); out[inv>0] = (out[inv>0]*0.25).astype(out.dtype)
                for tid, bb in bboxes.items():
                    x1,y1,x2,y2 = [int(v) for v in bb]
                    draw_box(out, [x1,y1,x2,y2], label="vehiculo")
                    # centroide predicho (opcional visualizar)
                    cx, cy = SmoothTracker._centroid([x1,y1,x2,y2])
                    cv2.circle(out, (int(cx), int(cy)), 3, (0,0,255), -1)
                    cv2.putText(out, f"ID {int(tid)}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(out, f"ID {int(tid)}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                cv2.putText(out, f"Vehículos en movimiento: {moving_count}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(out, f"Vehículos en movimiento: {moving_count}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1, cv2.LINE_AA)
                if args.flag_out:
                    cv2.putText(out, f"FLAG: {'1' if (last_flag==1) else '0'}  (th={args.flag_threshold})",
                                (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(out, f"FLAG: {'1' if (last_flag==1) else '0'}  (th={args.flag_threshold})",
                                (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

                cv2.imshow("VehCam", out)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break

            frame_id += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.display: cv2.destroyWindow("VehCam")
        else: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
