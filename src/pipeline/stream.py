"""Khung pipeline đọc video — mock + RuleEngine + event log + Telegram + ảnh minh chứng."""
import argparse, sys, time, cv2, yaml, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.postprocess import RuleEngine
from storage.events import EventDB
from alerts.telegram import TelegramAlerter


_NAMES = {0: "person", 1: "helmet", 2: "no_helmet", 3: "vest", 4: "no_vest", 5: "machinery"}
_COLORS = {
    0: (80, 200, 80), 1: (255, 200, 0), 2: (0, 0, 255),
    3: (0, 255, 128), 4: (0, 140, 255), 5: (0, 255, 255),
}


def _xyxy(bbox, w, h):
    x1, y1, x2, y2 = bbox
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


_VIOLATION_CLS = {"no_helmet": 2, "no_vest": 4, "machinery": 5}


def violation_conf(dets, vtype, person_bbox, fallback):
    """Conf cao nhất của detection đúng lớp vi phạm (tâm nằm trong bbox người; máy thì lấy mọi máy).
    vtype gộp dạng 'no_vest+machinery' -> lấy max qua từng loại."""
    if "+" in vtype:
        return max(violation_conf(dets, t, person_bbox, fallback) for t in vtype.split("+"))
    cls = _VIOLATION_CLS.get(vtype)
    best = None
    for d in dets:
        if d["cls"] != cls:
            continue
        x1, y1, x2, y2 = d["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        px1, py1, px2, py2 = person_bbox
        if cls == 5 or (px1 <= cx <= px2 and py1 <= cy <= py2):
            best = d["conf"] if best is None else max(best, d["conf"])
    return best if best is not None else fallback


def save_evidence(frame, track_id, vtype, frame_idx, dets=None, person_bbox=None, out_dir="logs"):
    vis = frame.copy()
    h, w = vis.shape[:2]
    for d in dets or []:
        x1, y1, x2, y2 = _xyxy(d["bbox"], w, h)
        c = _COLORS.get(d["cls"], (180, 180, 180))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 1)
        cv2.putText(
            vis, f"{_NAMES.get(d['cls'], d['cls'])} {d['conf']:.2f}",
            (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA,
        )
    if person_bbox:
        x1, y1, x2, y2 = _xyxy(person_bbox, w, h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        vis, f"{vtype}  id={track_id}  f={frame_idx}",
        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
    )
    out = Path(out_dir) / "evidence" / f"{vtype.replace('+', '-')}_{track_id}_{frame_idx:06d}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0=webcam hoặc path video")
    ap.add_argument("--config", default="configs/safety.yaml")
    ap.add_argument("--mock", action="store_true", help="chạy mock không cần model")
    ap.add_argument("--detector", default="", help="path best.pt M1 — có thì inference thật, không thì fallback mock loop")
    ap.add_argument("--telegram-token", default="", help="token bot Telegram (để trống = mock)")
    ap.add_argument("--telegram-chat", default="", help="chat_id Telegram (để trống = mock)")
    ap.add_argument("--reset-db", action="store_true", help="xóa events.db + ảnh evidence cũ trước khi chạy")
    ap.add_argument("--out-dir", default="logs", help="thư mục ghi events.db + evidence/ (Kaggle: /kaggle/working/logs)")
    ap.add_argument("--device", default="", help="'' = tự chọn (GPU nếu có), 'cpu', '0'...")
    ap.add_argument("--imgsz", type=int, default=640, help="kích thước resize + inference (mặc định 640 khớp train M1)")
    args = ap.parse_args()
    sz = args.imgsz
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(f"Config: {cfg['thresholds']}")
    out_dir = Path(args.out_dir)
    db_path = out_dir / "events.db"
    if args.reset_db:
        if db_path.exists():
            db_path.unlink()
        for old in (out_dir / "evidence").glob("*.jpg"):
            old.unlink()
    db = EventDB(str(db_path))
    n0 = db.count()
    tg = TelegramAlerter(token=args.telegram_token, chat_id=args.telegram_chat)
    t0, n = time.time(), 0
    if args.mock:
        print("MOCK: 90 frames + RuleEngine + DB + Telegram")
        engine = RuleEngine(cfg, fps=30)
        person = {"cls": 0, "conf": 0.9, "bbox": [0.4, 0.2, 0.6, 0.8]}
        no_h = {"cls": 2, "conf": 0.9, "bbox": [0.45, 0.21, 0.55, 0.32]}
        for i in range(90):
            out = engine.step([person, no_h], i)
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            for al in out["alerts"]:
                p = save_evidence(frame, al["track_id"], al["type"], i, [person, no_h], al.get("bbox"), out_dir)
                db.insert(al["track_id"], al["type"], i, 0.90, p)
                tg.send(f"VIOLATION {al['type']} track {al['track_id']} frame {i}", p)
            if i % 30 == 0:
                print(f"  frame {i}: tracks={len(out['tracks'])} alerts={out['alerts']}")
            n += 1
        fps = n / max(1e-6, time.time() - t0)
        print(f"Done {n} frames | FPS={fps:.1f} | new_events={db.count()-n0} | evidence in {out_dir / 'evidence'}")
        db.close()
        return
    src = 0 if args.source == "0" else args.source
    detector = None
    if args.detector:
        # Import chậm để --mock không cần ultralytics/torch.
        from ultralytics import YOLO
        detector = YOLO(args.detector)
        device = args.device
        if not device:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        print(f"Detector: {args.detector} | device={device}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open source {src}")
        db.close()
        return
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not 1 <= video_fps <= 120:
        video_fps = 30  # webcam/RTSP hay trả 0 -> mặc định 30
    engine = RuleEngine(cfg, fps=video_fps)
    print(f"Video FPS={video_fps:.1f} -> cooldown {cfg['temporal']['cooldown_sec']}s = {int(cfg['temporal']['cooldown_sec']*video_fps)} frame")
    n=0
    t_read = t_infer = 0.0
    while True:
        t1 = time.time()
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (sz, sz))
        t_read += time.time() - t1
        if detector is not None:
            dets, person_conf = [], 0.0
            t2 = time.time()
            res = detector(frame, imgsz=sz, device=device, verbose=False)[0]
            t_infer += time.time() - t2
            for b in res.boxes:
                x1, y1, x2, y2 = (b.xyxy[0] / sz).tolist()
                cls, conf = int(b.cls[0]), float(b.conf[0])
                dets.append({"cls": cls, "conf": conf, "bbox": [x1, y1, x2, y2]})
                if cls == 0:
                    person_conf = max(person_conf, conf)
            for al in engine.step(dets, n)["alerts"]:
                p = save_evidence(frame, al["track_id"], al["type"], n, dets, al.get("bbox"), out_dir)
                conf = violation_conf(dets, al["type"], al["bbox"], person_conf)
                db.insert(al["track_id"], al["type"], n, conf, p)
                tg.send(f"VIOLATION {al['type']} track {al['track_id']} frame {n}", p)
                print(f"  ALERT {al['type']} track={al['track_id']} frame={n}")
        n+=1
        if n%30==0:
            print(f"frame {n}: {frame.shape}")
    cap.release()
    fps = n / max(1e-6, time.time() - t0)
    print(f"Done {n} frames | FPS={fps:.1f} | new_events={db.count()-n0}")
    if n:
        print(f"imgsz={sz} | decode+resize={1000*t_read/n:.1f} ms/frame | inference={1000*t_infer/n:.1f} ms/frame")
    db.close()

if __name__ == "__main__":
    main()
