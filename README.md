# Nhóm 9 — Giám sát an toàn công trường

YOLOv8m M1 (6 lớp) + ByteTrack + RuleEngine + SQLite + Telegram + FastAPI.

## Cài đặt
```bash
cd Nhom9_Tuan01_Skeleton
pip install -r requirements.txt
```

## 1. Chuẩn hoá nhãn M1 (6 lớp)
```bash
python tools/prepare_m1.py \
  --src-hardhat /path/to/Hard_Hat_Workers \
  --src-site    /path/to/Construction_Site_Safety \
  --out data/m1 --train-ratio 0.7 --val-ratio 0.2 --seed 42
```
`0 person | 1 helmet | 2 no_helmet | 3 vest | 4 no_vest | 5 machinery`

## 2. Augmentation thời tiết
```bash
python tools/augment_weather.py --src data/m1/train --out data/m1/train
```

## 3. Trích keypoint M2
```bash
python tools/extract_pose.py --video /path/to/video.mp4 --out data/m2/sample.npz --preview
```

## 4. Test
```bash
pytest -q
```

## 5. Pipeline
```bash
python src/pipeline/stream.py --mock
python src/pipeline/stream.py --reset-db --source data/construction_test.mp4 \
  --detector results_m1/runs/detect/train-2/weights/best.pt
```

## 6. API
```bash
uvicorn src.api.app:app --port 8000
# /health /events /events/count /evidence/{file} /docs
```

## 7. Cấu hình
`safety.yaml` — `ppe_frames` N=15, `cooldown_sec` 60, `ppe_conf` 0.50.

## Lưu ý
Validation M1: P 0.9051 / R 0.8053 / mAP50 0.8668 / mAP50-95 0.6325. `best.pt`, `*.mp4`, `logs/`, `results_m1/` không push.
