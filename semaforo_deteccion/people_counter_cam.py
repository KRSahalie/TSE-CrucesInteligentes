#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================
#  PeatonCam: Detector y contador de PERSONAS (una sola cámara) + Bandera de semáforo
#  --------------------------------------------------------------------------------
#  - Cuenta SOLO personas dentro de un ROI opcional (más estable). También dibuja animales (opcional) sin contarlos.
#  - Ventana ÚNICA ("PeatonCam") para evitar múltiples ventanas.
#  - Salidas:
#       * Consola: JSON por frame con el conteo.
#       * Archivo de conteo (--write_count): guarda el número de personas.
#       * Bandera de semáforo (--flag_out): escribe 1 si count >= --flag_threshold, si no 0.
#         (Se emite solo cuando cambia el valor para evitar escrituras constantes.)
#
#  Ejemplo (hardware potente, GPU 0):
#    python3 people_counter_cam.py --src 0 --display \
#      --yolo_model yolov8m.pt --imgsz 1152 --device 0 \
#      --ped_conf 0.55 --animal_conf 0.40 --nms_iou 0.65 \
#      --width 1920 --height 1080 --fps 30 \
#      --mask roi_crosswalk.png \
#      --write_count /tmp/ped_count.txt \
#      --flag_out /tmp/ped_flag.txt --flag_threshold 5
#
#  Dependencias mínimas (Ubuntu 24.04):
#    sudo apt install -y python3-opencv v4l-utils ffmpeg
#    python3 -m pip install --user --break-system-packages ultralytics
# ==================================================================================================

from __future__ import annotations
import os, cv2, time, json, argparse
from typing import Optional

# ------------------------------
# Utilidades
# ------------------------------
ALLOWED_NAME_GROUPS = {
    "persona":  {"person"},
    "animal":   {"dog", "cat", "bird", "horse", "sheep", "cow"}
}
COLOR = {"persona": (60,180,255), "animal": (255,120,120), None:(220,220,220)}

def map_allowed_label(name: str) -> Optional[str]:
    n = (name or "").strip().lower()
    for k, group in ALLOWED_NAME_GROUPS.items():
        if n in group: return k
    return None

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

def center_in_mask(xyxy, mask_bin) -> bool:
    if mask_bin is None: return True
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cx, cy = (x1 + x2)//2, (y1 + y2)//2
    h, w = mask_bin.shape[:2]
    return 0 <= cx < w and 0 <= cy < h and mask_bin[cy, cx] > 0

# ------------------------------
# Detector YOLO (wrapper)
# ------------------------------
class YOLODetector:
    def __init__(self, model_path="yolov8s.pt", conf=0.5, imgsz=960, classes=None, device=None, nms_iou=0.60):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = float(conf); self.imgsz = int(imgsz); self.device = device; self.nms_iou = float(nms_iou)
        self.names = self.model.model.names if hasattr(self.model, "model") else self.model.names
        if classes is not None:
            # names: dict o lista -> normalizamos a dict {idx:name}
            if isinstance(self.names, dict):
                name_to_idx = {v: k for k, v in self.names.items()}
            else:
                name_to_idx = {self.names[i]: i for i in range(len(self.names))}
            self.allowed_indices = sorted([int(name_to_idx[n]) for n in classes if n in name_to_idx])
        else:
            self.allowed_indices = None

    def infer(self, frame_bgr):
        res = self.model.predict(frame_bgr, conf=self.conf, iou=self.nms_iou, imgsz=self.imgsz,
                                 verbose=False, classes=self.allowed_indices, device=self.device)
        r0 = res[0]
        xyxy, labels, confs = [], [], []
        if r0.boxes is not None and len(r0.boxes) > 0:
            X = r0.boxes.xyxy.cpu().numpy()
            C = r0.boxes.conf.cpu().numpy() if r0.boxes.conf is not None else [None]*len(X)
            K = r0.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(X)):
                cls_id = int(K[i])
                name = self.names[cls_id] if isinstance(self.names, dict) else self.names[cls_id]
                canon = map_allowed_label(name)
                if canon is None: continue
                xyxy.append(X[i].tolist())
                labels.append(canon)
                confs.append(float(C[i]) if C is not None else None)
        return xyxy, labels, confs

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="PeatonCam: contador de personas + bandera de semáforo (una cámara)")
    ap.add_argument("--src", required=True, help="Índice de cámara (0,1,...)")
    ap.add_argument("--display", action="store_true", help="Mostrar ventana única")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--yolo_model", type=str, default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--nms_iou", type=float, default=0.60)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--ped_conf", type=float, default=0.50)
    ap.add_argument("--animal_conf", type=float, default=0.35)

    ap.add_argument("--mask", type=str, default=None, help="PNG/BMP 8-bit (blanco=ROI peatonal)")
    ap.add_argument("--write_count", type=str, default=None, help="Ruta para escribir el conteo (entero)")

    # NUEVO: bandera para semáforo
    ap.add_argument("--flag_out", type=str, default=None, help="Ruta para escribir 1/0 según conteo")
    ap.add_argument("--flag_threshold", type=int, default=5, help="Umbral de personas para escribir 1")

    args = ap.parse_args()

    # Cargar ROI
    mask_bin = None
    if args.mask and os.path.exists(args.mask):
        mv = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mv is not None:
            mask_bin = cv2.threshold(mv, 127, 255, cv2.THRESH_BINARY)[1]

    # Detector YOLO (personas + animales)
    classes = {"person"} | {"dog","cat","bird","horse","sheep","cow"}
    yolo = YOLODetector(model_path=args.yolo_model,
                        conf=min(args.ped_conf, args.animal_conf),
                        imgsz=args.imgsz,
                        classes=list(classes),
                        device=args.device,
                        nms_iou=args.nms_iou)

    # Abrir cámara
    src = int(args.src) if str(args.src).isdigit() else args.src
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[PeatonCam] No pude abrir cámara: {args.src}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass

    # Ventana única (si aplica)
    if args.display:
        cv2.namedWindow("PeatonCam", cv2.WINDOW_NORMAL)

    last_flag = None  # evita escrituras repetidas
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Asegura tamaño
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

            # Redimensionar ROI si existe
            mask_r = None
            if mask_bin is not None:
                mask_r = cv2.resize(mask_bin, (args.width, args.height), interpolation=cv2.INTER_NEAREST)

            # Inference
            xyxy, labels, confs = yolo.infer(frame)

            persons = [b for b,l,c in zip(xyxy,labels,confs)
                       if l=="persona" and (c is None or c>=args.ped_conf) and (mask_r is None or mask_r[int((b[1]+b[3])//2), int((b[0]+b[2])//2)]>0)]
            animals = [b for b,l,c in zip(xyxy,labels,confs)
                       if l=="animal" and (c is None or c>=args.animal_conf) and (mask_r is None or mask_r[int((b[1]+b[3])//2), int((b[0]+b[2])//2)]>0)]

            count = len(persons)

            # Salidas: conteo
            print(json.dumps({"ts": time.time(), "persons": count}), flush=True)
            if args.write_count:
                try:
                    with open(args.write_count, "w") as f:
                        f.write(str(count))
                except Exception as e:
                    print(f"[PeatonCam] No pude escribir conteo en {args.write_count}: {e}")

            # NUEVO: Bandera de semáforo (1 si count >= threshold; 0 en caso contrario)
            if args.flag_out:
                flag_val = 1 if count >= int(args.flag_threshold) else 0
                if flag_val != last_flag:
                    try:
                        with open(args.flag_out, "w") as f:
                            f.write(str(flag_val))
                        print(json.dumps({"event":"PEDESTRIAN_FLAG","value":flag_val,"ts":time.time()}), flush=True)
                        last_flag = flag_val
                    except Exception as e:
                        print(f"[PeatonCam] No pude escribir flag en {args.flag_out}: {e}")

            if args.display:
                out = frame.copy()
                if mask_r is not None:
                    inv = cv2.bitwise_not(mask_r)
                    out[inv>0] = (out[inv>0]*0.25).astype(out.dtype)
                for r in persons: draw_box(out, r, label="persona")
                for r in animals: draw_box(out, r, label="animal")
                cv2.putText(out, f"Personas: {count}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(out, f"Personas: {count}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1, cv2.LINE_AA)
                if args.flag_out:
                    cv2.putText(out, f"FLAG: {'1' if (last_flag==1) else '0'}  (th={args.flag_threshold})",
                                (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(out, f"FLAG: {'1' if (last_flag==1) else '0'}  (th={args.flag_threshold})",
                                (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
                cv2.imshow("PeatonCam", out)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.display:
            cv2.destroyWindow("PeatonCam")
        else:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
