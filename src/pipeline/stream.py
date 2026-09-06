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
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(f"Config: {cfg['thresholds']}")
    engine = RuleEngine(cfg, fps=30)
    db = EventDB("logs/events.db")
    tg = TelegramAlerter(mock=True)
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
        print(f"Done {n} frames | FPS={fps:.1f} | events={db.count()} | evidence in logs/evidence/")
        db.close()
        return
    src = 0 if args.source == "0" else args.source
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
        # TODO tuần 4: thay bằng inference YOLOv8m thật sau khi nhận weights từ Hoài
        n+=1
        if n%30==0:
            print(f"frame {n}: {frame.shape}")
        if n>=90: break
    cap.release()
    fps = n / max(1e-6, time.time() - t0)
    print(f"Done {n} frames | FPS={fps:.1f} | events={db.count()}")
    db.close()

if __name__ == "__main__":
    main()
