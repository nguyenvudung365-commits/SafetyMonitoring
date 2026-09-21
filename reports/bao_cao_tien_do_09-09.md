# Báo cáo tiến độ — Nhóm 9 — 09/09/2026

## 1. Tổng quan

- Đề tài: Giám sát an toàn công trường (YOLOv8m M1 6 lớp + Pose/TCN M2 + ByteTrack + RuleEngine + SQLite + Telegram + FastAPI).
- Repo chính: https://github.com/nguyenvudung365-commits/SafetyMonitoring (branch `main`).
- Trạng thái: M1 đã train đủ 50 epochs và có `best.pt`; phần tích hợp M1 → RuleEngine → SQLite/Telegram/API đã viết xong. M2 (Năng) đã có code pipeline nhưng mới train trên dữ liệu mô phỏng.

---

## 2. Đã hoàn thành

### Tuần 1 — Data pipeline M1 (Duy)

- Gộp 2 nguồn: Hard Hat Workers + Construction Site Safety về schema 6 lớp YOLO: `person / helmet / no_helmet / vest / no_vest / machinery`.
- Loại trùng bằng hash MD5, chia theo nhóm ảnh gốc (`.rf.` group key) để tránh rò rỉ augment cùng gốc sang val/test.
- Kết quả: **19.603 ảnh hợp lệ** — train 13.717 / val 3.892 / test 1.994. Tổng bbox ~93.737 (thống kê chi tiết trong `reports/data_distribution.md`).
- Artefact: `tools/prepare_m1.py`, `tools/check_yolo_labels.py`, `tools/plot_distribution.py`, `data/m1/data.yaml`, `data/m1/mapping.yaml`, `reports/week01_data_stats.md`.

### Tuần 2 — Weather augmentation (Duy)

- Chỉ augment tập train bằng Albumentations (rain / fog / dust) — val/test giữ nguyên để đánh giá sạch.
- Train tăng 13.717 → **27.450 ảnh** (bbox ~130.858). Artefact: `tools/augment_weather.py`, `reports/log_augment_weather.txt`.

### Tuần 3 — Logging & Alerts (Dũng)

- SQLite `logs/events.db`: bảng `events(id, track_id, violation_type, timestamp, frame_idx, confidence, image_path)`, API `insert/query/count/close`.
- Evidence: `logs/evidence/<type>_<track>_<frame>.jpg` khi RuleEngine bắn alert.
- Telegram: `TelegramAlerter` mock (in console) + live (`sendMessage`/`sendPhoto`).
- Pipeline mock: `python src/pipeline/stream.py --mock` chạy 90 frames → 2 events `no_helmet` (N=15, cooldown 60s), FPS ~1900 trên CPU.
- Test: `pytest` 15 passed (`tests/test_events.py`).
- Merge code tuần 1+2 của Duy vào repo chính (commit `7f64447`).

### M2 — Pose + TCN (Năng, notebook `test.ipynb`)

Đã làm được (code + smoke test), **chưa phải kết quả trên video ngã thật**:

- Feature spec 37 chiều/frame: 17 keypoints COCO chuẩn hóa theo bbox (34) + `w/h`, góc thân `theta`, vận tốc dọc `v_y` (3). Cửa sổ 30 frame, bỏ cửa sổ thiếu >30% frame, interpolate keypoint `conf < 0.3`.
- `tools/extract_pose.py`: YOLOv8m-pose → window `(N, 30, 37)` lưu `.npz`.
- `tools/prepare_m2.py`: chia **theo video** 70/20/10 + augment Gauss + time-warp 0.8–1.2× **chỉ trên train**. Hiện đang gọi `generate_synthetic_dataset()` — 40 video giả, train 1869 / val 164 / test 93 windows. **Test split không có lớp `normal`** (0/18/75) vì random 4 video — đây là lỗ hổng split, không dùng số F1=1.0 để báo cáo.
- `models/m2_models.py`: TCN 4 tầng dilated causal (d=1,2,4,8; k=3; 64ch; dropout 0.2; residual), RF = 61 ≥ 31. LSTM 1 lớp 64. Heuristic `w/h` + `v_y`.
- `train_tcn.py` + `configs/train_m2.yaml`: 50 epochs, batch 16, lr 0.001, seed 42, `conf=0.70`, N=8, cooldown 60s (khớp `safety.yaml`).
- Smoke 5 epochs trên **CPU** (không bật GPU T4): TCN/LSTM F1=1.0 trên synthetic — **không có ý nghĩa generalization**. Cell 50 epochs chưa chạy xong.
- `tools/inference_m2.py` (`M2Inference.predict_with_rules`) sẵn cho Dũng gắn vào `stream.py`.
- `eval_fall_events.py`: event-level TP/FP/FN, tolerance −1s/+3s, FAR/h — mới test trên CSV mẫu.
- Domain shift (`eval_domain_shift.py`): chạy indoor = outdoor = cùng `test.npz` → delta 0, **chưa đo lệch miền thật**.
- Dataset UR Fall / Le2i: **chưa upload** (`/kaggle/input` trống). YOLO pose smoke test trên `bus.jpg` OK (5 persons).

---

## 3. Đang triển khai — Tuần 4

### Train M1 YOLOv8m (Hoài) — đã hoàn thành

- Đã train đủ **50/50 epochs** trên Kaggle với YOLOv8m, `imgsz=640`, `batch=16`, `seed=42`.
- Trọng số: `results_m1/runs/detect/train-2/weights/best.pt` (~50 MB); không đưa `.pt` lên GitHub.
- Kết quả validation cuối: Precision **0.9051**, Recall **0.8053**, mAP50 **0.8668**, mAP50-95 **0.6325**.
- Artefact: `results.csv`, `results.png`, confusion matrix và các đường cong P/R/F1 đã nhận đủ.

---

## 4. Vấn đề & rủi ro

- Đã chạy `data/construction_test.mp4` + `best.pt`: 404 frames, ~11 FPS. **Không có alert mới** trên clip này; 3 dòng DB cũ là mock `no_helmet` frame 14.
- FastAPI `GET /events` đã viết; `uvicorn src.api.app:app --port 8000`.
- M2 train trên **synthetic**; F1=1.0 **không được báo cáo** là kết quả mô hình. Test split thiếu lớp `normal`.
- UR Fall / Le2i chưa vào Kaggle Input. Notebook Năng chạy **CPU** (`torch-2.11.0+cpu`), không phải T4.
- Code M2 nằm trong `test.ipynb` (%%writefile), **chưa merge** vào repo GitHub chính (`extract_pose.py` skeleton cũ khác bản Năng).
- Domain shift và event-level mới chạy dummy.

---

## 5. Việc tiếp theo (sau khi đã báo cáo)

Ưu tiên theo phụ thuộc, không làm song song mù:

1. **Hoài:** đã hoàn tất M1, gửi `best.pt` + `results.csv` + `results.png`; kết quả validation mAP50=0.8668, mAP50-95=0.6325.
2. **Năng:** upload UR Fall + Le2i lên Kaggle Input; sửa `prepare_m2.py` gọi `extract_pose.py` trên video thật; chia split **theo video ID** sao cho **mọi split đều có normal/falling/fallen**; train 50 epochs GPU; gửi `best_tcn.pt` + bảng P/R/F1 thật.
3. **Dũng:** đã cắm YOLO vào `stream.py`, API `/events`, chạy video Pexels 404 frames (~11 FPS). Alert thật = 0 trên clip này; dùng `--reset-db` để không lẫn mock. Chưa gắn M2.
4. **Duy:** hỗ trợ path/data.yaml nếu Hoài kẹt; `albumentations>=1.3.0` đã thêm vào `requirements.txt`.

---

## 6. Demo hiện tại (chạy được ngay)

```bash
cd Nhom9_Tuan01_Skeleton
pip install -r requirements.txt
python src/pipeline/stream.py --mock
pytest -q
```

- Kết quả mock: in `[TELEGRAM MOCK]`, lưu JPG trong `logs/evidence/`, SQLite có 2 rows.
- Luồng thật sau khi cài dependencies: `python src/pipeline/stream.py --source video.mp4 --detector results_m1/runs/detect/train-2/weights/best.pt`.
- API: `uvicorn src.api.app:app --reload --port 8000`, xem `/events`, `/events/count`, `/docs`.

## 7. Phân công báo cáo từng thành viên

Mỗi người 2–3 phút, nói theo đúng phần mình phụ trách.

### 7.1. Dũng — System Lead, tuần 3 (Logging & Alerts)

**Nội dung nói:**
- Em phụ trách phần tích hợp hệ thống: lưu sự kiện vi phạm, ảnh minh chứng, gửi cảnh báo.
- Thiết kế bảng SQLite `events(id, track_id, violation_type, timestamp, frame_idx, confidence, image_path)` — mỗi vi phạm là một dòng, có ảnh đính kèm để truy xuất sau.
- Viết `TelegramAlerter`: chế độ mock in ra console khi chưa có token; chế độ thật gọi `sendMessage` + `sendPhoto` của Telegram Bot API.
- Ghép vào pipeline: `python src/pipeline/stream.py --mock` chạy 90 frames, sinh 2 sự kiện `no_helmet` (do N=15 frame xác nhận + cooldown 60s), lưu JPG vào `logs/evidence/`.
- Kiểm thử: `pytest` 15 tests passed, gồm test insert/query DB và test Telegram mock.
- Đã merge code tuần 1–2 của Duy vào repo chính để nhóm dùng một nguồn chung.

**Nếu bị hỏi:**
- *"Vì sao mock?"* — mock dùng để kiểm chứng logic RuleEngine → DB → Telegram; weights M1 đã có, còn cần cài dependencies và chạy trên video thật để xác nhận end-to-end.
- *"Sao chỉ 2 events?"* — vì luật N=15 frame liên tiếp mới xác nhận vi phạm, và cooldown 60s chặn spam cho cùng track_id. Đây là hành vi đúng thiết kế.

---

### 7.2. Duy — Tuần 1 + 2 (Data pipeline & Augmentation)

**Nội dung nói:**
- Tuần 1: em gộp 2 bộ dữ liệu Roboflow (Hard Hat Workers + Construction Site Safety) về đúng 6 lớp của đề tài: person, helmet, no_helmet, vest, no_vest, machinery.
- Mỗi bộ có tên lớp và ID riêng, nên em đọc `data.yaml` của từng bộ để map đúng, không map nhầm theo số.
- Lớp không thuộc schema (mask, no_mask, safety_cone) bị loại bỏ có chủ đích — đã loại 7.913 dòng nhãn.
- Chống rò rỉ dữ liệu: Roboflow augment ra nhiều ảnh từ một ảnh gốc, tên có dạng `xxx_jpg.rf.<hash>.jpg`. Em gom theo `group_key` để các bản augment của cùng ảnh gốc nằm chung một split, không bị rơi vào cả train lẫn test.
- Kết quả: 19.603 ảnh hợp lệ — train 13.717 / val 3.892 / test 1.994, tổng ~93.737 bbox.
- Tuần 2: em augment thêm mưa, sương mù, bụi chỉ cho tập train (13.717 → 27.450 ảnh). Đây là biến đổi mức pixel nên toạ độ bbox giữ nguyên, chỉ copy label.
- Val/test giữ nguyên để đánh giá trên dữ liệu sạch, đúng nguyên tắc.

**Nếu bị hỏi:**
- *"Vì sao vest/no_vest ít?"* — chỉ 3.44% và 4.23% tổng bbox. Đã ghi rõ trong `data_distribution.md`, phương án là augment + cân nhắc oversampling/class weights khi train.
- *"Vì sao `head` map thành `no_helmet`?"* — bộ Hard Hat Workers dùng nhãn `head` cho vùng đầu không đội mũ, tương đương lớp 2 của schema.

---

### 7.3. Hoài — Tuần 4 (Train M1 YOLOv8m)

**Nội dung nói:**
- Em phụ trách huấn luyện mô hình phát hiện 6 lớp.
- Đã tải 2 bộ dữ liệu lên Kaggle Dataset, chạy `prepare_m1.py` ngay trên Kaggle để ra `data/m1` với `data.yaml` trỏ đúng path `/kaggle/working/...`.
- Đã kiểm tra dữ liệu: vẽ bbox lên 50 ảnh mẫu để review trực quan, xác nhận các ảnh augment cùng gốc nằm cùng split.
- Đã smoke test: `yolo predict` với weights COCO, và train thử 5 epochs trên GPU Tesla T4 — môi trường chạy ổn, không lỗi.
- **Đang làm:** train đủ 50 epochs với `yolov8m.pt`, imgsz 640, batch 16, seed 42.
- Đầu ra sẽ là `best.pt` và `results.csv` gồm mAP50, mAP50-95, precision, recall.
- **Chưa có số liệu mAP thật** vì chưa train xong — em không ước lượng trước, sẽ báo cáo đúng kết quả khi chạy xong.

**Nếu bị hỏi:**
- *"Sao chưa có mAP?"* — train 50 epochs cần khoảng 2–3 giờ GPU, em đang chạy trên Kaggle, sẽ có kết quả trong 1–2 ngày.
- *"Vì sao val ban đầu mAP chỉ 0.04?"* — đó là em test weights COCO pretrained (80 lớp: person, car, bus...) trên bộ 6 lớp của mình, class không khớp nên chỉ số vô nghĩa. Chỉ dùng để kiểm tra pipeline đọc data, không phải kết quả mô hình.

---

### 7.4. Năng — M2 (Pose + TCN, notebook `test.ipynb`)

**Nội dung nói:**
- Em phụ trách M2: YOLOv8m-pose 17 keypoints COCO → vector 37 chiều/frame (34 tọa độ chuẩn hóa bbox + w/h + góc thân + v_y) → cửa sổ 30 frame → TCN 4 tầng dilated.
- Receptive field TCN = 61 ≥ 31 (window 30). Có 2 baseline: LSTM 64 và heuristic w/h + v_y.
- Đã viết pipeline: `extract_pose.py`, `prepare_m2.py` (chia theo video, augment Gauss + time-warp train-only), `train_tcn.py`, `inference_m2.py` (conf 0.70, N=8), `eval_fall_events.py` (event-level −1s/+3s).
- Smoke: YOLO pose nhận 5 người trên `bus.jpg`. TCN/LSTM forward OK.
- **Nói thẳng:** dataset UR Fall / Le2i chưa upload. Train hiện tại trên **40 video mô phỏng**. F1=1.0 trên synthetic **không phải kết quả thật**, test split còn thiếu lớp normal. Chưa train 50 epochs trên GPU. Domain shift chưa đo vì indoor=outdoor cùng file giả.

**Nếu bị hỏi:**
- *"F1 = 1.0?"* — Data giả, class tách bằng tay (cột w/h). Không dùng số này.
- *"Khi nào có số thật?"* — Sau khi extract pose từ UR Fall/Le2i, chia video đúng, train 50 epochs T4.

---

### 7.5. Thứ tự lên báo cáo đề xuất

1. **Duy** (tuần 1 → 2): data từ đâu, xử lý thế nào, ra bao nhiêu ảnh.
2. **Hoài** (tuần 4): data đó đưa vào train ra sao, đang ở đâu, khi nào có weights.
3. **Dũng** (tuần 3): khi có detection thì hệ thống ghi log, lưu ảnh, gửi cảnh báo thế nào + demo `--mock`.
4. **Năng** (M2): pipeline Pose+TCN đã code; data còn mô phỏng; bước tiếp theo UR Fall/Le2i.

Lý do: đi theo luồng dữ liệu — thu thập → huấn luyện → vận hành → mở rộng. Người nghe dễ theo dõi hơn là báo cáo theo số tuần.

---

## 8. Phụ lục — Lệnh Kaggle cho Hoài (tuần 4)

```bash
python tools/prepare_m1.py \
  --src-hardhat "/kaggle/input/datasets/thuhoai05/dataset/Hard Hat Workers.v11-augmented3x-allclasses-fastmodel.yolov8" \
  --src-site "/kaggle/input/datasets/thuhoai05/dataset/Construction Site Safety.v25-v1-accurate-machinery-aug5xmosaic.yolov8" \
  --out data/m1

yolo train model=yolov8m.pt data=data/m1/data.yaml epochs=50 imgsz=640 batch=16 seed=42
# sau train:
# runs/detect/train/weights/best.pt  -> gửi Drive
# runs/detect/train/results.csv, results.png -> báo cáo mAP
```

---

## 8. Hướng dẫn trình bày (từng thành viên)

Mở sẵn: **VS Code** (repo `Nhom9_Tuan01_Skeleton`) + **Chrome** (GitHub + Kaggle).

### 8.1. Duy lên đầu — Data Pipeline (Tuần 1+2)

**Chrome mở:**
- https://github.com/phuongduy1102/SafetyMonitoring → xem commits `32dfcff`, `b225848`, `9975a92`

**VS Code mở:**

| File | Dòng/mục | Nội dung trình bày |
|---|---|---|
| `tools/prepare_m1.py` | 10-18 `ALIASES` | Map 6 lớp: `head`→`no_helmet`, `mask`→`None` loại bỏ |
| `tools/prepare_m1.py` | 44-54 hàm `group_key()` | Chống rò rỉ: ảnh `*.rf.*` cùng gốc không rơi vào cả train lẫn test |
| `reports/data_distribution.md` | Bảng thống kê | 19.603 ảnh: train 13.717 / val 3.892 / test 1.994 |
| `reports/week01_data_stats.md` | Phân bố lớp | `vest` 3.44%, `no_vest` 4.23% — lớp hiếm cần lưu ý |
| `tools/augment_weather.py` | 15-35 transforms | Rain/fog/dust chỉ augment train, val/test giữ nguyên |
| `reports/log_augment_weather.txt` | Cả file | 13.725 gốc → 27.450 train sau augment |

**Câu hỏi hay bị hỏi:**
- "Sao không dùng `train_test_split` mà tự code chia theo group?" → Vì Roboflow augment nhiều bản từ 1 gốc, phải group theo `.rf.` prefix mới đảm bảo không rò.
- "Lớp `vest` chỉ 3.44%, có ảnh hưởng không?" → Có. Đã dùng augment x2 train để bù, nếu mAP thấp sẽ thêm class weights khi train.
- "Sao loại `mask`/`safety_cone`?" → Không thuộc 6 lớp schema đề tài (PPE chỉ helmet/vest, machinery chỉ xe/máy).

---

### 8.2. Hoài lên thứ 2 — Train M1 (Tuần 4)

**Chrome mở:**
- Kaggle notebook `h-c-m-y-n-ng-cao.ipynb` (hoặc file local trong Downloads)

**Cells cần chỉ:**

| Cell | Nội dung trình bày |
|---|---|
| Cell 18 `find /kaggle/input/...` | 2 dataset đã upload Kaggle: Hard Hat Workers + Construction Site Safety |
| Cell 20 `!python tools/prepare_m1.py ...` | Log chạy thành công: 13717/3892/1994 ảnh, 74134 bbox tổng |
| Cell 22 `cat data/m1/data.yaml` | `path: /kaggle/working/SafetyMonitoring/data/m1` — đúng path Kaggle |
| Cell 26 vẽ bbox 50 ảnh | Review trực quan, nhóm `004229_jpg.rf.*` (3 bản augment cùng train) |
| Cell 30 `yolo train epochs=5` | Smoke test GPU T4 OK (output quá lớn chỉ thấy log đầu) |

**KHÔNG trình bày:** Cell 23 `yolo val model=yolov8m.pt` — mAP 0.04 vì test COCO pretrained (80 lớp) trên data M1 (6 lớp), không có ý nghĩa.

**Câu hỏi hay bị hỏi:**
- "Train 50 epochs xong chưa?" → Chưa, đang chạy trên Kaggle, dự kiến 2-3h. Cell 30 chỉ test 5 epochs để check GPU.
- "mAP dự kiến bao nhiêu?" → YOLOv8m COCO pretrained thường đạt mAP50 ~0.6-0.7 trên dataset tương tự. Vest/no_vest hiếm có thể thấp hơn.
- "Sao không train local?" → Không có GPU. Kaggle cho 30h/tuần T4 miễn phí.

---

### 8.3. Dũng lên thứ 3 — Logging, Alerts, Demo (Tuần 3)

**VS Code mở:**

| File | Dòng/mục | Nội dung trình bày |
|---|---|---|
| `src/storage/events.py` | 5-14 `SCHEMA` | Bảng SQLite 7 cột: `id`, `track_id`, `violation_type`, `timestamp`, `frame_idx`, `confidence`, `image_path` |
| `src/alerts/telegram.py` | 15-25 `send()` | Mock print `[TELEGRAM MOCK]` / Live `requests.post` sendMessage/sendPhoto |
| `src/pipeline/stream.py` | 29-50 `if args.mock` | 90 frames person+no_helmet → RuleEngine → DB insert + Telegram + evidence JPG |
| `configs/safety.yaml` | `thresholds` | PPE N=15, cooldown 60s, fall N=8, TCN window 30 |
| `tests/test_events.py` | Cả file | Pytest mock SQLite + Telegram, 15 passed |

**Demo live trên terminal:**
```bash
cd Nhom9_Tuan01_Skeleton
python src/pipeline/stream.py --mock
# → in [TELEGRAM MOCK] no_helmet track 1 frame 14
# → mở logs/evidence/ thấy ảnh jpg mới sinh
pytest -q  # 15 passed
```

**Chrome mở:**
- https://github.com/nguyenvudung365-commits/SafetyMonitoring → commit `7f64447` merge code Duy

**Câu hỏi hay bị hỏi:**
- "Sao chưa chạy video thật?" → `best.pt` đã có; hiện thư mục chưa có video test và máy chạy báo thiếu `ultralytics`, nên cần cài requirements rồi chạy video mẫu.
- "Telegram gửi thật được không?" → Được, có code `requests.post(TELEGRAM_API)`. Mock để test không spam bot.
- "Evidence ảnh lưu ở đâu?" → `logs/evidence/<type>_<track>_<frame>.jpg`, mỗi alert 1 ảnh (crop bbox người vi phạm nếu có).
- "ByteTrack tự code hay thư viện?" → Tự code IoU greedy matching đơn giản (line 15-40 `src/tracking/tracker.py` — chưa có trong báo cáo này nhưng có trong repo).

---

### 8.4. Năng lên cuối — M2 Pose/TCN (`test.ipynb`)

**Mở:** `test.ipynb` (local Downloads). Không mở `extract_pose.py` cũ trong skeleton — khác bản notebook.

| Cell | Nội dung trình bày |
|---|---|
| Cell 0 markdown | Vai trò M2, 17 kpt, TCN, 2 baseline |
| Cell 7 `yolo predict yolov8m-pose` | Smoke: 5 persons trên bus.jpg |
| Cell 8 `feature_spec.md` | 37 chiều, conf không đưa vào vector |
| Cell 11 `extract_pose.py` | Interpolate, window 30, stride 5, drop >30% missing |
| Cell 14–15 `prepare_m2.py` | **Nói rõ synthetic** 1869/164/93; test 0 normal |
| Cell 18–19 `m2_models.py` | RF=61, TCN/LSTM/Heuristic forward OK |
| Cell 24 train 5 epoch | F1=1.0 = overfit synthetic — **không chiếu như kết quả** |
| Cell 30 `inference_m2.py` | API cho Dũng: `predict_with_rules(window, track_id)` |

**Không chiếu:** Cell 25 (50 epoch trống), Cell 28 domain shift delta 0, Cell 37 bar chart số placeholder 0.94/0.85.

**Câu hỏi hay bị hỏi:**
- "Dataset gì?" → Thiết kế UR Fall + Le2i; hiện pipeline test bằng synthetic. Chưa có video thật trên Input.
- "Tích hợp?" → `stream.py` lấy person track → buffer 30 frame 37-d → `M2Inference` → nếu alert thì `engine.step()` giống PPE.
