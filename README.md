# Nhóm 9 — Tuần 1 Skeleton

## Cài đặt
```bash
pip install -r requirements.txt
```

## 1. Chuẩn hoá nhãn M1 (6 lớp)
```bash
python tools/prepare_m1.py \
  --src-hardhat /path/to/Hard_Hat_Workers \
  --src-site    /path/to/Construction_Site_Safety \
  --out data/m1 --train-ratio 0.7 --val-ratio 0.2 --seed 42
# Output: data/m1/{train,val,test}/{images,labels} + mapping.yaml + stats.md
```

## 2. Trích keypoint M2
```bash
python tools/extract_pose.py --video /path/to/video.mp4 --out data/m2/sample.npz --preview
```

## 3. Test tầng luật (không cần GPU)
```bash
pytest tests/test_rules.py -v
```

## 4. Chạy pipeline mock (chưa có model thật)
```bash
python src/pipeline/stream.py --source 0 --config configs/safety.yaml --mock
```

## Schema 6 lớp
0 person | 1 helmet | 2 no_helmet | 3 vest | 4 no_vest | 5 machinery
