# Chạy pipeline M1 trên Kaggle

Code lấy từ GitHub; `best.pt` và video **không** có trên Git nên phải upload thành Kaggle Dataset.

## 1. Chuẩn bị (1 lần)

1. Kaggle → **Datasets → New Dataset**, tên ví dụ `safety-m1-assets`, upload:
   - `results_m1/runs/detect/train-2/weights/best.pt`
   - `data/13002132_3840_2160_30fps.mp4` (và các clip test khác)
2. Tạo Notebook mới → **Add Input** → chọn dataset trên.
3. Settings: **Accelerator = GPU T4 x1**, **Internet = On** (để `git clone` và `pip install`).

## 2. Cell chạy

```python
!git clone -q https://github.com/nguyenvudung365-commits/SafetyMonitoring.git
%cd SafetyMonitoring
!pip install -q ultralytics
!nvidia-smi --query-gpu=name --format=csv,noheader
```

```python
# Đổi tên dataset nếu khác; xem đường dẫn thật bằng: !ls /kaggle/input/*
IN = "/kaggle/input/safety-m1-assets"
!python src/pipeline/stream.py --reset-db \
    --source {IN}/13002132_3840_2160_30fps.mp4 \
    --detector {IN}/best.pt \
    --out-dir /kaggle/working/logs
```

So sánh imgsz (bảng FPS Tuần 8):

```python
for s in (640, 512):
    !python src/pipeline/stream.py --reset-db --imgsz {s} --source {IN}/13002132_3840_2160_30fps.mp4 --detector {IN}/best.pt --out-dir /kaggle/working/logs_{s}
```

## 3. Lấy kết quả

- DB + ảnh nằm ở `/kaggle/working/logs/` → tab **Output** để tải về.
- Xem ảnh ngay trong notebook:

```python
from IPython.display import Image, display
import glob
for f in sorted(glob.glob("/kaggle/working/logs/evidence/*.jpg")):
    print(f); display(Image(f, width=480))
```

## Lưu ý

- `/kaggle/input` chỉ đọc → luôn truyền `--out-dir /kaggle/working/...`.
- `--device` để trống = tự dùng GPU nếu có; log in `device=0` là đang chạy GPU.
- Khi báo FPS phải ghi rõ **máy đo** (Kaggle T4 vs laptop Quadro T1000), không trộn số hai máy.
- Video 4K: decode tốn ~25–30 ms/frame ở CPU dù có GPU. Muốn FPS sát thực tế camera, dùng clip 1080p.
