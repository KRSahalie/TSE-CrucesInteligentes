#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_cam_tracker.py (híbrido: YOLO + MOG2)
- YOLOv8n (Ultralytics) para detectar personas y vehículos (parados o en movimiento).
- MOG2 + filtros geométricos para reforzar detecciones por movimiento (cámaras fijas).
- Tracker ligero por centroides + persistencia y etiquetado de clase por ID.

Uso típico:
  python3 multi_cam_tracker.py --src ./videos/video1.mp4 --display --detector yolo --det_every 3 --lr 0.002

Requisitos:
  pip install ultralytics
"""

import os
import cv2
import time
import argparse
import numpy as np

# ====================== CONFIG ======================
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
FPS_TARGET    = 30

# Área mínima adaptativa por perspectiva para MOG2:
MIN_AREA_FAR   = 400     # lejos (arriba)
MIN_AREA_NEAR  = 3200    # cerca (abajo)

# Filtros geométricos para MOG2
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 4.0
MIN_SOLIDITY     = 0.35

# Visual
DRAW_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Warm-up y persistencia
WARMUP_FRAMES       = 30
PERSIST_MIN_FRAMES  = 3

# Detector
DETECT_EVERY_N_FRAMES = 3   # correr YOLO cada N frames
YOLO_CONF = 0.35
YOLO_CLASSES = {            # COCO → etiquetas
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
# ====================================================


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.nextObjectID = 0
        self.objects = {}        # id -> (cX, cY)
        self.bboxes = {}         # id -> (x1,y1,x2,y2)
        self.disappeared = {}    # id -> frames desaparecido
        self.hits = {}           # id -> frames visibles
        self.labels = {}         # id -> label consolidado ("persona"/"vehiculo"/None)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, label=None):
        oid = self.nextObjectID
        self.objects[oid] = centroid
        self.bboxes[oid]  = bbox
        self.disappeared[oid] = 0
        self.hits[oid] = 1
        self.labels[oid] = label
        self.nextObjectID += 1

    def deregister(self, objectID):
        for d in (self.objects, self.bboxes, self.disappeared, self.hits, self.labels):
            if objectID in d:
                del d[objectID]

    @staticmethod
    def _centroid(b):
        x1, y1, x2, y2 = b
        return (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

    def update(self, rects, labels=None):
        """
        rects: [(x1,y1,x2,y2), ...]
        labels: lista paralela con etiquetas ("persona"/"vehiculo"/None) o None
        """
        if labels is None:
            labels = [None]*len(rects)

        if len(rects) == 0:
            to_remove = []
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    to_remove.append(oid)
            for oid in to_remove:
                self.deregister(oid)
            return self.bboxes.copy(), self.hits.copy(), self.labels.copy()

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, b in enumerate(rects):
            inputCentroids[i] = self._centroid(b)

        if len(self.objects) == 0:
            for i, b in enumerate(rects):
                self.register(tuple(inputCentroids[i]), b, labels[i])
            return self.bboxes.copy(), self.hits.copy(), self.labels.copy()

        objectIDs = list(self.objects.keys())
        objectCentroids = np.array(list(self.objects.values()))

        diff = objectCentroids[:, None, :] - inputCentroids[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=2))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows, usedCols = set(), set()
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = objectIDs[row]
            self.objects[oid] = tuple(inputCentroids[col])
            self.bboxes[oid]  = rects[col]
            self.disappeared[oid] = 0
            self.hits[oid] = min(self.hits.get(oid, 0) + 1, 1_000_000)
            # consolidar etiqueta si llega nueva
            new_label = labels[col]
            if new_label:
                self.labels[oid] = self._merge_label(self.labels.get(oid), new_label)
            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, len(rects))).difference(usedCols)

        if D.shape[0] >= len(rects):
            for row in unusedRows:
                oid = objectIDs[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
        else:
            for col in unusedCols:
                self.register(tuple(inputCentroids[col]), rects[col], labels[col])

        return self.bboxes.copy(), self.hits.copy(), self.labels.copy()

    @staticmethod
    def _merge_label(old, new):
        """
        Fusión simple: si había None, toma new. Si difieren, prioriza 'vehiculo' si new es vehiculo.
        (Para más robustez, puedes cambiarlo a voto acumulado por historial.)
        """
        if old is None:
            return new
        if old == new:
            return old
        if new == "vehiculo":
            return "vehiculo"
        return old


class YoloDetector:
    """
    Detector YOLOv8 (Ultralytics) sobre clases de interés COCO (person/vehículos).
    """
    def __init__(self, model_name="yolov8n.pt", conf=YOLO_CONF):
        from ultralytics import YOLO  # import diferido
        self.model = YOLO(model_name)
        self.conf = conf
        self.class_map = YOLO_CLASSES

    def detect(self, frame):
        """
        Devuelve: rects, labels
        rect: (x1,y1,x2,y2); label: "persona" o "vehiculo"
        """
        H, W = frame.shape[:2]
        res = self.model.predict(source=frame, conf=self.conf, verbose=False, classes=list(self.class_map.keys()))
        rects, labels = [], []
        if not res:
            return rects, labels
        boxes = res[0].boxes
        if boxes is None:
            return rects, labels
        for b in boxes:
            cls = int(b.cls.item())
            name = self.class_map.get(cls, "")
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = [int(max(0, v)) for v in xyxy]
            x1, y1 = min(x1, W-1), min(y1, H-1)
            x2, y2 = min(x2, W-1), min(y2, H-1)

            if name == "person":
                lbl = "persona"
            else:
                lbl = "vehiculo"  # bicycle, car, motorcycle, bus, truck
            rects.append((x1, y1, x2, y2))
            labels.append(lbl)
        return rects, labels


class CamStream:
    """
    Fuente de video + MOG2 como refuerzo de movimiento (opcional).
    """
    def __init__(self, src, name="Cam", learning_rate=0.002, mask_path=None):
        self.src = src
        self.name = name
        self.learning_rate = learning_rate

        parsed = self._parse_src(src)
        self.cap = cv2.VideoCapture(parsed, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{self.name}] No se pudo abrir la fuente: {src}")

        if isinstance(parsed, int) or (isinstance(parsed, str) and parsed.startswith("/dev/video")):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

        self.backsub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=32, detectShadows=True)
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_dil   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        self.mask = None
        if mask_path and os.path.isfile(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                self.mask = (m > 0).astype(np.uint8)

        self.frame_count = 0

    @staticmethod
    def _parse_src(s):
        if isinstance(s, int):
            return s
        if isinstance(s, str):
            s = s.strip()
            if s.isdigit():
                return int(s)
            return s
        return s

    @staticmethod
    def _min_area_by_y(yc, h):
        return int(MIN_AREA_FAR + (MIN_AREA_NEAR - MIN_AREA_FAR) * (yc / max(h, 1)))

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"[{self.name}] Falló la lectura del frame.")
        return frame

    def detect_mog(self, frame):
        """
        Detecciones por movimiento (bboxes) SIN etiqueta.
        """
        self.frame_count += 1
        fg = self.backsub.apply(frame, learningRate=self.learning_rate)
        fg[fg == 127] = 0  # quitar sombras

        if self.mask is not None:
            if self.mask.shape != fg.shape:
                mask_resized = cv2.resize(self.mask, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_NEAREST)
                fg = cv2.bitwise_and(fg, fg, mask=mask_resized)
            else:
                fg = cv2.bitwise_and(fg, fg, mask=self.mask)

        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel_open,  iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close, iterations=1)
        fg = cv2.dilate(fg, self.kernel_dil, iterations=1)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        H, W = fg.shape[:2]
        bboxes = []
        for c in contours:
            if len(c) < 4:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w <= 1 or h <= 1:
                continue
            yc = y + h * 0.5
            min_area = self._min_area_by_y(yc, H)
            area = w * h
            if area < min_area:
                continue
            aspect = w / float(h)
            if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
                continue
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            cnt_area  = cv2.contourArea(c)
            if hull_area <= 1.0:
                continue
            solidity = cnt_area / hull_area
            if solidity < MIN_SOLIDITY:
                continue
            bboxes.append((x, y, x + w, y + h))
        return bboxes

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


def draw_tracks(frame, tracked_dict, hits_dict, labels_dict):
    for oid, (x1, y1, x2, y2) in tracked_dict.items():
        if hits_dict.get(oid, 0) < PERSIST_MIN_FRAMES:
            continue
        lbl = labels_dict.get(oid) or "objeto"
        color = (0, 200, 0) if lbl == "persona" else (0, 150, 255)  # verde=persona, naranja=vehículo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, DRAW_THICKNESS)
        cv2.putText(frame, f"{lbl} #{oid}", (x1, max(0, y1 - 6)),
                    FONT, 0.6, color, 2, cv2.LINE_AA)
    return frame


def parse_args():
    p = argparse.ArgumentParser(description="Multi-cam tracker (YOLO + MOG2) con clasificación persona/vehículo")
    p.add_argument("--src", action="append", required=True,
                   help="Fuente de video (repetir para varias). Ej: --src 0 --src /dev/video2 --src ruta.mp4")
    p.add_argument("--display", action="store_true", help="Mostrar ventanas")
    p.add_argument("--max_disappeared", type=int, default=30, help="Frames para olvidar IDs")
    p.add_argument("--max_distance", type=int, default=80, help="Max distancia de matching (px)")
    p.add_argument("--mask", default=None, help="Ruta a máscara ROI (blanco=área válida)")
    p.add_argument("--lr", type=float, default=0.002, help="Learning rate MOG2 (0=estático)")
    p.add_argument("--detector", choices=["none","yolo"], default="yolo", help="Detector principal para objetos quietos")
    p.add_argument("--det_every", type=int, default=DETECT_EVERY_N_FRAMES, help="Correr detector cada N frames")
    p.add_argument("--yolo", default="yolov8n.pt", help="Modelo YOLOv8 a usar (n, s, etc.)")
    return p.parse_args()


def main():
    args = parse_args()

    # Fuentes y tracker por cámara
    cams, trackers, names = [], [], []
    for i, src in enumerate(args.src):
        name = f"Cam{i}:{src}"
        cam = CamStream(src, name=name, learning_rate=args.lr, mask_path=args.mask)
        cams.append(cam)
        trackers.append(CentroidTracker(max_disappeared=args.max_disappeared,
                                        max_distance=args.max_distance))
        names.append(name)
        print(f"[OK] {name} abierta")

    # Detector global (compartido) si se usa YOLO
    yolo = None
    if args.detector == "yolo":
        yolo = YoloDetector(model_name=args.yolo, conf=YOLO_CONF)
        print("[INFO] YOLO cargado:", args.yolo)

    last_time = time.time()
    try:
        frame_idx = 0
        while True:
            frame_idx += 1
            for i, cam in enumerate(cams):
                frame = cam.read()

                # 1) Detecciones por YOLO (para objetos quietos). Solo cada N frames.
                rects_yolo, labels_yolo = [], []
                if yolo is not None and (frame_idx % max(1, args.det_every) == 0):
                    rects_yolo, labels_yolo = yolo.detect(frame)

                # 2) Detecciones por MOG2 (solo cajas, sin etiqueta)
                rects_mog = cam.detect_mog(frame)

                # 3) Fusión simple: YOLO tiene prioridad (etiquetado),
                #    añadimos MOG2 donde no hay solapamiento fuerte con YOLO.
                all_rects = list(rects_yolo)
                all_labels = list(labels_yolo)

                def iou(a, b):
                    ax1, ay1, ax2, ay2 = a
                    bx1, by1, bx2, by2 = b
                    interx1, intery1 = max(ax1, bx1), max(ay1, by1)
                    interx2, intery2 = min(ax2, bx2), min(ay2, by2)
                    iw, ih = max(0, interx2 - interx1), max(0, intery2 - intery1)
                    inter = iw * ih
                    if inter == 0:
                        return 0.0
                    area_a = (ax2-ax1)*(ay2-ay1)
                    area_b = (bx2-bx1)*(by2-by1)
                    return inter / float(area_a + area_b - inter + 1e-9)

                for r in rects_mog:
                    # si aún no hay nada, agrega el primero directo
                    if len(all_rects) == 0:
                        all_rects.append(r)
                        all_labels.append(None)
                        continue
                    # evita duplicar lo que YOLO ya detectó
                    overlaps = [iou(r, ry) for ry in rects_yolo]
                    if len(overlaps) == 0 or max(overlaps) < 0.3:
                        all_rects.append(r)
                        all_labels.append(None)

                # 4) Tracking + etiquetas
                tracked, hits, labels = trackers[i].update(all_rects, all_labels)

                # Warm-up: no dibujar antes
                if frame_idx < WARMUP_FRAMES:
                    out = frame
                else:
                    out = draw_tracks(frame, tracked, hits, labels)

                if args.display:
                    cv2.imshow(names[i], out)

            if args.display and (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

            # limitar FPS
            now = time.time()
            dt = now - last_time
            if dt < 1.0 / max(FPS_TARGET, 1):
                time.sleep((1.0 / FPS_TARGET) - dt)
            last_time = now

    except KeyboardInterrupt:
        pass
    finally:
        for cam in cams:
            cam.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
