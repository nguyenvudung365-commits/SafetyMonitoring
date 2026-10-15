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
python src/pipeline/stream.py --reset-db --source data/13002132_3840_2160_30fps.mp4 --detector results_m1/runs/detect/train-2/weights/best.pt
```
Windows cmd: viết trên 1 dòng (hoặc xuống dòng bằng `^`, không dùng `\`).

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--source` | `0` | `0` = webcam, hoặc đường dẫn MP4/RTSP |
| `--detector` | rỗng | `best.pt` M1; rỗng = không inference |
| `--mock` | tắt | chạy 90 frame giả, không cần model |
| `--config` | `configs/safety.yaml` | file ngưỡng/vùng cấm (`configs/safety_demo.yaml` = vùng cấm giả định cho clip demo) |
| `--imgsz` | `640` | kích thước resize + inference (640 khớp lúc train) |
| `--device` | rỗng | rỗng = GPU nếu có, `cpu`, `0`... |
| `--out-dir` | `logs` | nơi ghi `events.db` + `evidence/` |
| `--reset-db` | tắt | xoá DB **và** ảnh evidence cũ trước khi chạy |
| `--telegram-token/--telegram-chat` | rỗng | rỗng = in `[TELEGRAM MOCK]` |

Log cuối in `FPS`, `new_events` và thời gian `decode+resize` / `inference` (ms/frame) để biết nút thắt.

Ảnh evidence `logs/evidence/<loại>_<track_id>_<frame>.jpg`: box đỏ dày = người vi phạm, box mỏng + nhãn = mọi detection M1. Cùng ID nhiều vi phạm trong cùng frame được gộp, ví dụ `no_helmet-no_vest_4_000120.jpg`.

## 6. API
```bash
uvicorn src.api.app:app --port 8000
# /health /events /events/count /evidence/{file} /docs
```
Nếu pipeline chạy với `--out-dir` khác `logs`, đặt biến môi trường `SAFETY_LOG_DIR` cùng đường dẫn trước khi chạy uvicorn.

## 7. Tầng luật (`configs/safety.yaml`)
- **PPE:** `no_helmet` xét trong 1/3 trên bbox người; `no_vest` xét vùng thân (1/6 → 2/3 chiều cao). Detection dưới `ppe_conf` (0.50) bị bỏ qua.
- **Vùng cấm:** điểm chân (giữa cạnh dưới bbox) nằm trong polygon `zones.forbidden`, và person conf ≥ `zone_person_conf` (0.50) để loại móc cẩu/vật nhỏ bị nhận nhầm thành người.
- **Máy móc:** khoảng cách chân người → tâm máy < `machinery_distance` (0.08, nới theo chiều cao bbox).
- **Té ngã:** `RuleEngine.step(..., fall_probs={track_id: p})` — báo khi p > `fall_conf` (0.70) đủ `fall_frames` (8) frame. Chưa có M2 thì không truyền `fall_probs`, nhánh này tắt.
- **Xác nhận + cooldown:** PPE/vùng/máy cần `ppe_frames` (15) frame liên tiếp; cùng (ID, loại) không báo lại trong `cooldown_sec` (60 s). Số frame cooldown tính theo FPS thật của video.
- **Gộp sự kiện khi tracker đổi ID:** nếu cùng loại vi phạm đã báo cho một track vừa biến mất, cách ≤ `dedup_sec` (3 s) và bbox chồng IoU ≥ `dedup_iou` (0.20) thì không báo lại.

## 8. Chạy trên Kaggle
Xem [reports/huong_dan_kaggle.md](reports/huong_dan_kaggle.md): upload `best.pt` + video thành Kaggle Dataset, `git clone` repo, chạy với `--out-dir /kaggle/working/logs`.

## Trạng thái
- M1: train xong, chạy được trong pipeline. Validation: P 0.9051 / R 0.8053 / mAP50 0.8668 / mAP50-95 0.6325.
- M2 (té ngã): chưa tích hợp — chờ `best_tcn.pt`.
- Tracker `src/tracking/bytetrack.py` là bản IoU greedy tự viết (chưa Kalman); ID có thể đổi khi người đi chéo nhau (đã giảm trùng sự kiện bằng bước gộp ở mục 7, chưa triệt để).
- Luật `no_helmet`/`machinery` mới chạy ra sự kiện, chưa kiểm chứng đúng ngữ cảnh (clip máy xúc chỉ có người ngồi trong cabin). Clip 1:36 trên laptop: 14 → 11 sự kiện sau khi thêm lọc vùng cấm + gộp sự kiện (xem tay, 1 clip).
- Lỗi đã biết: vest phản quang màu nhạt bị báo `no_vest`.
- Chưa có số FAR/h / Precision–Recall trên video thật; event demo chưa được kiểm tay hết.
- FPS phụ thuộc máy: luôn ghi kèm GPU khi báo cáo.

## Lưu ý
`best.pt`, `*.mp4`, `logs/`, `results_m1/` không push (đã có trong `.gitignore`), cũng như các thư mục `logs_*/`.
