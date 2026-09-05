"""
Trích 17 keypoints bằng YOLOv8m-pose trên 1 video.
"""
import argparse, cv2, numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="data/m2/sample.npz")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO("yolov8m-pose.pt")  # tự tải lần đầu
    cap = cv2.VideoCapture(args.video)
    all_kpts = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = model(frame, verbose=False)[0]
        # lấy người đầu tiên nếu có
        if res.keypoints is not None and len(res.keypoints.xy) > 0:
            kpts = res.keypoints.xyn[0].cpu().numpy()  # (17,2) normalized
            conf = res.keypoints.conf[0].cpu().numpy()
        else:
            kpts = np.zeros((17,2)); conf = np.zeros(17)
        all_kpts.append(np.concatenate([kpts, conf[:,None]], axis=1))  # (17,3)
        idx += 1
        if args.preview and idx % 30 == 0:
            print(f"frame {idx}: {len(res.boxes)} person(s)")
    cap.release()
    arr = np.stack(all_kpts)  # (T,17,3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, keypoints=arr)
    print(f"Saved {arr.shape} -> {args.out}")

if __name__ == "__main__":
    main()
