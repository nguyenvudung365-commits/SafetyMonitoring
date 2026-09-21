"""Khung pipeline đọc video — mock + RuleEngine + event log + Telegram + ảnh minh chứng."""
import argparse, sys, time, cv2, yaml, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.postprocess import RuleEngine
from storage.events import EventDB
from alerts.telegram import TelegramAlerter


def save_evidence(frame, track_id, vtype, frame_idx):
    out = Path(f"logs/evidence/{vtype}_{track_id}_{frame_idx:06d}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0=webcam hoặc path video")
    ap.add_argument("--config", default="configs/safety.yaml")
    ap.add_argument("--mock", action="store_true", help="chạy mock không cần model")
    ap.add_argument("--detector", default="", help="path best.pt M1 — có thì inference thật, không thì fallback mock loop")
    ap.add_argument("--telegram-token", default="", help="token bot Telegram (để trống = mock)")
    ap.add_argument("--telegram-chat", default="", help="chat_id Telegram (để trống = mock)")
    ap.add_argument("--reset-db", action="store_true", help="xóa logs/events.db trước khi chạy")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(f"Config: {cfg['thresholds']}")
    engine = RuleEngine(cfg, fps=30)
    db_path = Path("logs/events.db")
    if args.reset_db and db_path.exists():
        db_path.unlink()
    db = EventDB(str(db_path))
    n0 = db.count()
    tg = TelegramAlerter(token=args.telegram_token, chat_id=args.telegram_chat)
    t0, n = time.time(), 0
    if args.mock:
        print("MOCK: 90 frames + RuleEngine + DB + Telegram")
        person = {"cls": 0, "conf": 0.9, "bbox": [0.4, 0.2, 0.6, 0.8]}
        no_h = {"cls": 2, "conf": 0.9, "bbox": [0.45, 0.21, 0.55, 0.32]}
        for i in range(90):
            out = engine.step([person, no_h], i)
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            for al in out["alerts"]:
                p = save_evidence(frame, al["track_id"], al["type"], i)
                db.insert(al["track_id"], al["type"], i, 0.90, p)
                tg.send(f"VIOLATION {al['type']} track {al['track_id']} frame {i}", p)
            if i % 30 == 0:
                print(f"  frame {i}: tracks={len(out['tracks'])} alerts={out['alerts']}")
            n += 1
        fps = n / max(1e-6, time.time() - t0)
        print(f"Done {n} frames | FPS={fps:.1f} | new_events={db.count()-n0} | evidence in logs/evidence/")
        db.close()
        return
    src = 0 if args.source == "0" else args.source
    detector = None
    if args.detector:
        # Import chậm để --mock không cần ultralytics/torch.
        from ultralytics import YOLO
        detector = YOLO(args.detector)
        print(f"Detector: {args.detector}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open source {src}")
        db.close()
        return
    n=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640,640))
        if detector is not None:
            dets, person_conf = [], 0.0
            for b in detector(frame, verbose=False)[0].boxes:
                x1, y1, x2, y2 = (b.xyxy[0] / 640).tolist()
                cls, conf = int(b.cls[0]), float(b.conf[0])
                dets.append({"cls": cls, "conf": conf, "bbox": [x1, y1, x2, y2]})
                if cls == 0:
                    person_conf = max(person_conf, conf)
            for al in engine.step(dets, n)["alerts"]:
                p = save_evidence(frame, al["track_id"], al["type"], n)
                db.insert(al["track_id"], al["type"], n, person_conf, p)
                tg.send(f"VIOLATION {al['type']} track {al['track_id']} frame {n}", p)
                print(f"  ALERT {al['type']} track={al['track_id']} frame={n}")
        n+=1
        if n%30==0:
            print(f"frame {n}: {frame.shape}")
    cap.release()
    fps = n / max(1e-6, time.time() - t0)
    print(f"Done {n} frames | FPS={fps:.1f} | new_events={db.count()-n0}")
    db.close()

if __name__ == "__main__":
    main()
