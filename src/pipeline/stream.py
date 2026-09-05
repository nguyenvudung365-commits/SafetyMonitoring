"""Khung pipeline đọc video — mock + RuleEngine (ByteTrack + luật)."""
import argparse, sys, cv2, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.postprocess import RuleEngine

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0=webcam hoặc path video")
    ap.add_argument("--config", default="configs/safety.yaml")
    ap.add_argument("--mock", action="store_true", help="chạy mock không cần model")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(f"Config: {cfg['thresholds']}")
    engine = RuleEngine(cfg, fps=30)
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Không mở được nguồn {src} (mock OK nếu --mock)")
        if args.mock:
            print("MOCK: giả lập 90 frame + RuleEngine")
            person = {"cls": 0, "conf": 0.9, "bbox": [0.4, 0.2, 0.6, 0.8]}
            no_h = {"cls": 2, "conf": 0.9, "bbox": [0.45, 0.21, 0.55, 0.32]}
            for i in range(90):
                out = engine.step([person, no_h], i)
                if i % 30 == 0:
                    print(f"  frame {i}: tracks={len(out['tracks'])} alerts={out['alerts']}")
            print("Done 90 frames.")
            return
        return
    n=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640,640))
        n+=1
        if n%30==0:
            print(f"frame {n}: {frame.shape}")
        if n>=90: break
    cap.release()
    print(f"Done {n} frames")

if __name__ == "__main__":
    main()
