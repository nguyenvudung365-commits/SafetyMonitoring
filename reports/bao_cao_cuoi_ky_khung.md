# Báo cáo cuối kỳ — Giám sát an toàn lao động trên công trường (Nhóm 9)

> Khung báo cáo. Mục nào chưa có số đo thật thì ghi "chưa đo", không điền số ước lượng.

## 1. Giới thiệu và phạm vi
Bài toán, 4 nhóm rủi ro (thiếu PPE, vùng cấm, té ngã, gần máy móc), phân công (Hoài M1, Năng M2, Duy dữ liệu, Dũng hệ thống).

## 2. Dữ liệu
Nguồn, chia train/val/test theo nhóm ảnh gốc, phân bố lớp (xem `reports/data_distribution.md`).

## 3. Mô hình
- M1: YOLOv8m 6 lớp, cấu hình huấn luyện, kết quả val (và test nếu có).
- M2: pose + TCN té ngã (trạng thái: ...).

## 4. Hệ thống
Kiến trúc pipeline (stream, tracker IoU, RuleEngine, SQLite, Telegram, FastAPI). Ngưỡng trong `configs/safety.yaml`.
Luật PPE theo conf cao nhất (không phụ thuộc thứ tự); luật máy móc đo tới cạnh bbox, ngưỡng `k * h`, loại trừ người trong cabin.

## 5. Thực nghiệm
### 5.1 Hiệu năng (FPS)
| Cấu hình | GPU | imgsz | FPS | decode+resize ms | inference ms |
|---|---|---|---|---|---|
| M1 | | | | | |
| M1 + Track | | | | | |

Lấy từ `run_meta.json` của từng lần chạy (có commit hash).

### 5.2 Đánh giá cấp sự kiện
Số clip, tổng thời lượng, người gán GT, tolerance. Bảng P / R / FAR/h theo loại (`tools/eval_events.py`).

| Loại | TP | FP | FN | Precision | Recall | FAR/h |
|---|---|---|---|---|---|---|

### 5.3 Quét tham số luật
Đồ thị Recall–FAR theo `ppe_frames`, `ppe_conf`, `zone_person_conf`, `machinery_k` (`tools/sweep_rules.py`). Nêu rõ clip dùng để chỉnh và clip đánh giá cuối.

### 5.4 Phân tích lỗi
Trước/sau khi sửa `ppe_flags` và `machinery_violation`: số sự kiện, ví dụ ảnh evidence.

## 6. Hạn chế
- GT do nhóm tự gán, số clip ít.
- M2 / té ngã (nếu chưa tích hợp), dashboard/WebSocket.
- Tracker IoU greedy, không Kalman; ID đổi khi người đi chéo.
- Resize méo tỉ lệ khung hình (stretch) chưa kiểm so với lúc train; ảnh bị resize vuông còn làm khoảng cách theo trục x/y trong ảnh lệch so với khoảng cách thật, ảnh hưởng luật `machinery`/`zone`.
- Rủi ro luật loại trừ cabin: người đứng TRƯỚC máy lớn mà bbox nằm >= 80% trong bbox máy (`cabin_overlap`) bị coi là ngồi trong cabin và bị bỏ sót. Có thể quét `--cabin-overlap` bằng `tools/sweep_rules.py`; chưa đo trên clip thật.
- Lỗi mô hình đã biết: vest phản quang nhạt bị báo `no_vest`.

## 7. Hướng dẫn chạy lại
`pip install -r requirements.txt`, `scripts/run_demo.bat` (hoặc `.sh`), `pytest -q`. Xem README.

## 8. Tài liệu tham khảo
