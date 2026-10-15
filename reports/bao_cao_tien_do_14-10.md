# Báo cáo tiến độ — Nhóm 9 — 14/10/2026

Hệ thống giám sát an toàn lao động trên công trường xây dựng. Báo cáo gồm Data (Duy), M1 (Hoài) và System (Dũng). **Năng (M2) báo cáo riêng.**

Các ảnh sự kiện được kiểm tra bằng mắt trên **một clip**, không phải số đo độ chính xác.

---

## 1. Tổng quan

- Pipeline: video → resize → YOLOv8m (6 lớp) → ByteTracker → RuleEngine → xác nhận N=15 frame + cooldown → SQLite + ảnh bằng chứng + Telegram (mock) → FastAPI.
- Repo: https://github.com/nguyenvudung365-commits/SafetyMonitoring (`main`).
- M1 đã train xong; pipeline chạy video thật trên laptop và Kaggle T4.
- M2 (té ngã) chưa gắn vào pipeline; phần này do Năng báo cáo.

### So với buổi 23/09

| | 23/09 | 14/10 |
|---|---|---|
| Video test | 1 clip 4K, 235 frame (~8 s) | Clip 1080p, 2871 frame (1:36), nhiều người |
| FPS | Laptop 5.8–8.9 | Laptop T1000 9.4; **Kaggle T4 32.0 (640), 49.1 (512)** |
| Mục tiêu ≥15 FPS | Chưa đạt | **Đạt trên T4**, chỉ tính M1 |
| Số sự kiện | 4 | 14 (laptop) / 15 (T4, 640) / 20 (T4, 512) |
| Kiểm tra ảnh sự kiện | Chưa | Đã xem từng ảnh |
| Sửa lỗi pipeline | — | Cooldown theo FPS, vùng vest, gộp vi phạm, reset evidence |
| Chạy Kaggle | Chưa | Đã chạy (notebook `m1-result`) |
| pytest | 15 | 25 passed |
| Công cụ gán nhãn | — | `label_tool.html` của Duy |

---

## 2. Data (Duy)

### Dataset M1 (không đổi so với 23/09)
- Schema 6 lớp: `0 person | 1 helmet | 2 no_helmet | 3 vest | 4 no_vest | 5 machinery`.
- 19.603 ảnh hợp lệ: train 13.717 / val 3.892 / test 1.994; train sau weather augment 27.450.
- Chia theo nhóm ảnh gốc Roboflow (`group_key` `.rf.`), seed 42, để không rò rỉ giữa các tập.
- Lớp `head` được dùng thay `no_helmet` (proxy). Vest/no_vest hiếm (~3–4% bbox).

### Công cụ gán nhãn `label_tool.html` (mới)
- Chạy một file HTML, không cần cài đặt.
- Dùng đúng 6 lớp như dataset; nhãn lưu dạng YOLO `cls cx cy w h`.
- Xuất file zip gồm train/val/test (ảnh + nhãn), `data.yaml`, `classes.txt`, `stats.md`.
- Có gán nhãn tự động bằng mô hình ONNX để gán nhanh rồi sửa tay.
- Khác với dataset đã có:
  - Chia train/val/test theo **từng ảnh** (có seed), không theo nhóm ảnh gốc, nên nếu dùng ảnh từ cùng video thì có thể rò rỉ giữa các tập.
  - Chỉ xuất các ảnh đã được duyệt.
- Cần thống nhất quy ước gán nhãn trước khi trộn dữ liệu mới vào dataset hiện tại.

---

## 3. M1 (Hoài)

### Kết quả huấn luyện (tập val)

| Metric | Giá trị |
|---|---|
| Precision | 0.906 |
| Recall | 0.806 |
| mAP@0.5 | 0.867 |
| mAP@0.5:0.95 | 0.632 |
| Epoch | 50/50 (6.4 giờ, Kaggle T4) |
| imgsz / batch / seed | 640 / 16 / 42 |

AP@0.5 từng lớp: person 0.779, helmet 0.962, no_helmet 0.907, vest 0.814, no_vest 0.843, machinery 0.895.

- `best.pt` ~52 MB, ở `results_m1/runs/detect/train-2/weights/best.pt`; không push lên Git.
- Việc train đúng và đã hoàn thành.
- Baseline COCO (mAP 0.04) chỉ có ý nghĩa cho lớp `person`, nên không dùng làm mốc cho 6 lớp.

### Còn thiếu so với plan (Hoài cần bổ sung)
1. Đánh giá trên **test split** (hiện mới có số trên val).
2. Ablation `freeze=10` so với full-train, cùng cấu hình.
3. Bảng 3 hàng: COCO / full-train / freeze, với AP từng lớp.
4. Phân tích lớp khó, trong đó có lỗi vest phản quang màu nhạt (mục 4.4).
5. FPS M1 và M1+ByteTrack, phối hợp với Dũng.

---

## 4. System (Dũng)

### 4.1. Sửa pipeline (commit `ebab377`, 11/10)
- Cooldown tính theo FPS thật của video (60 s × fps), không cố định số frame.
- Vùng kiểm tra vest là thân người (1/6–2/3 bbox).
- Gộp nhiều vi phạm cùng lúc của một người thành một sự kiện.
- `--reset-db` xoá cả ảnh bằng chứng cũ; thêm `--out-dir`, `--device` để chạy trên Kaggle.
- Thêm test cho các thay đổi trên; thêm `reports/huong_dan_kaggle.md`, cập nhật README.

### 4.2. Kết quả tốc độ (cùng clip 1:36, cùng `best.pt`)

| Máy | imgsz | FPS | Suy luận | Decode + resize | Sự kiện |
|---|---|---|---|---|---|
| Laptop Quadro T1000 | 640 | 9.4 | 84.7 ms/frame | 15.9 ms/frame | 14 |
| Kaggle T4 | 640 | **32.0** | 25.2 ms/frame | 3.0 ms/frame | 15 |
| Kaggle T4 | 512 | 49.1 | 16.5 ms/frame | 2.7 ms/frame | 20 |

- Đây là M1 + tracker + luật, **chưa có M2**; khi gắn TCN, FPS sẽ giảm.
- Clip 23/09 (4K) và 14/10 (1080p) khác nhau, nên không so trực tiếp 5.8 FPS với 9.4 FPS.

### 4.3. Kiểm tra bằng mắt các ảnh sự kiện

| Bản chạy | Đúng | Sai | Không chắc | Tổng |
|---|---|---|---|---|
| Laptop 640 | 11 | 2 | 1 | 14 |
| Kaggle 512 | 15 | 3 | 2 | 20 |

(Bản Kaggle 640, 15 sự kiện: kiểm tra 4 ảnh mới so với laptop, trong đó 1 không chắc; chưa xem hết.)

### 4.4. Lỗi đã thấy
- **Vest phản quang bị báo no_vest** (frame 752 và 789): người mặc vest vàng-xanh nhạt bị khoanh đỏ. Cả laptop và Kaggle đều lặp lại lỗi này, nên không phải ngẫu nhiên.
- **Vùng cấm báo nhầm:** móc cẩu và vật nhỏ bị nhận là `person` với conf 0.27–0.32, rồi vào vùng giả định.
- Một vài sự kiện không chắc (người rất nhỏ, ở xa).

### 4.5. Kết luận
- **Giữ imgsz 640.** Bản 512 nhanh hơn nhưng thêm 5 sự kiện và thêm báo nhầm; chưa đo mAP ở 512.
- **Chưa dùng làm số FAR hay độ chính xác chính thức:** chỉ một clip, kiểm tay; một người có thể bị đếm nhiều lần vì tracker đổi id (id lên tới hơn 140).
- Clip không có no_helmet và machinery, nên hai luật này **chưa được kiểm tra trên video mới**.
- Vùng cấm trong demo là vùng giả định (`configs/safety_demo.yaml`).

---

## 5. Vấn đề còn lại

1. Tracker tự viết (IoU-greedy, không Kalman) đổi id nhiều → một người sinh nhiều sự kiện.
2. Luật vùng cấm chưa lọc theo conf: cần ngưỡng riêng (≥0.5) để loại móc cẩu.
3. Lỗi vest phản quang cần đưa vào phân tích lớp khó.
4. M2 chưa gắn vào `stream.py` (chờ Năng).
5. Chưa đo FAR/h.

---

## 6. Việc tiếp theo

1. **Dũng:** ngưỡng conf riêng cho vùng cấm rồi chạy lại clip; gộp sự kiện theo người; cắt 2–3 phút từ clip máy xúc để thử machinery và no_helmet; ghi FPS T4 vào bảng Tuần 8; gắn M2 khi Năng có weights.
2. **Hoài:** mục 3 phần "Còn thiếu".
3. **Duy:** thống nhất quy ước gán nhãn và cách chia tập khi dùng `label_tool.html`; giữ schema 6 lớp.
4. **Năng:** báo cáo riêng.

---

## 7. Checklist so với plan

| Mốc | 23/09 | 14/10 |
|---|---|---|
| Tuần 1 repo + mock stream | Xong | Xong |
| Tuần 2 rules + pytest | Xong | Xong (25 test) |
| Tuần 4 M1 thật + FPS | Xong (mAP val; FPS laptop) | Thêm FPS T4; còn thiếu test split, ablation, bảng 3 hàng |
| Tuần 6 M2 trên pipeline | Chưa | Chưa |
| Tuần 7 E2E | Một phần | Một phần (M1 + DB + Telegram mock; thiếu té ngã) |
| Tuần 8 ≥15 FPS, FAR/h | Chưa | FPS đạt trên T4 (M1); FAR/h chưa |
| Tuần 10 dashboard | Chưa | Chưa (mới FastAPI) |

---

## 8. Cập nhật sau buổi (15/10): ba cải tiến và kết quả

- **Ngưỡng person riêng cho vùng cấm** (`zone_person_conf: 0.5`) trong `configs/*.yaml` và `postprocess.py`.
- **Gộp sự kiện khi tracker đổi id:** nếu cùng loại vi phạm đã báo cho một track vừa biến mất, cách <= 3 s và bbox chồng IoU >= 0.2 thì không báo lại (`dedup_sec`, `dedup_iou`).
- pytest: **28 passed** (thêm 3 test).

Clip 1:36, laptop T1000, `safety.yaml`, kiểm bằng mắt (1 clip, không phải độ chính xác):

| | Trước | Sau |
|---|---|---|
| Sự kiện | 14 (13 no_vest + 1 zone) | **11** (11 no_vest, 0 zone) |
| Bị loại | | zone frame 1632 (vật nhỏ); no_vest id 38 (f684) và id 72 (f1610), đều là cùng người đã báo trước đó (id 42, id 57) |

Lưu ý: bản demo `safety_demo.yaml` có vùng giả định nên số sự kiện zone không so sánh được.

Clip máy xúc 4K cắt 150 s (`excavator_cut.mp4`, 9000 frame, FPS 15.9 laptop): 12 sự kiện = 9 machinery + 3 no_helmet.
- **Cả hai luật lần đầu chạy trên video thật** nhưng báo theo đúng luật mà không đúng ngữ cảnh: machinery báo cho người ngồi trong cabin máy xúc (người vận hành), no_helmet báo cho tài xế xe tải nhìn từ trong cabin (đầu trần, bbox người rất lớn).
- Chưa có ảnh công nhân đi bộ gần máy, nên chưa kiểm chứng ca "đúng nghĩa". Cần clip khác.
- Ý kiến: nên có luật loại trừ người ngồi trong máy (bbox person nằm gần trọn trong bbox machinery).

### Clip bổ sung (15/10): thử no_helmet / no_vest trên cảnh có công nhân thật
Tải thêm 3 clip YouTube (≤1080p), chạy laptop, `safety.yaml`, kiểm bằng mắt trên ảnh thu nhỏ (chưa phải số đo độ chính xác):

| Clip | Frame | FPS | Sự kiện |
|---|---|---|---|
| `new_CVnd99Zx2w8` (công nhân đào rãnh, ống cống) | 1883 | 15.7 | 13 (4 no_helmet, 8 no_vest, 1 zone) |
| `new_fOcHJnfMMJY` (CCTV 23 s) | 673 | 13.9 | 0 |
| `new_AJOQlO9viRc` (thiết bị nặng) | 2750 | 17.6 | 0 |

Clip rãnh cống: no_vest phần lớn đúng (công nhân áo xanh/nâu, không có vest); no_helmet khó kết luận vì người nhỏ, đứng trong ống, bbox chồng nhau, nhiều người cùng khung; có box người sai (frame 1259 box trùm hai người; frame 1361 box rất nhỏ). Luật machinery không báo gì (chỉ có box machinery conf thấp 0.3–0.5 nhầm lên đống gỗ). Zone dùng vùng giả định nên không đánh giá.
Kết luận: vẫn **chưa có ca kiểm chứng sạch** cho machinery (người đi bộ gần máy) và no_helmet (đầu trần rõ ràng).
