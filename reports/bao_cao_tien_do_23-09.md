# Báo cáo tiến độ — Nhóm 9 — 23/09/2026

So với mốc 09/09: M1 đã có số validation đầy đủ; pipeline đã chạy **video thật + `best.pt`**, không còn chỉ `--mock`. Clip `construction_test.mp4` (09/09) = 0 alert; clip `13002132_3840_2160_30fps.mp4` (22/09) = 4 alert.

---

## 1. Tổng quan

- Đề tài: Giám sát an toàn công trường (YOLOv8m M1 6 lớp + Pose/TCN M2 + ByteTrack + RuleEngine + SQLite + Telegram + FastAPI).
- Repo: https://github.com/nguyenvudung365-commits/SafetyMonitoring (`main`).
- Mốc tuần: hết Tuần 4 / đầu Tuần 5 theo kế hoạch nhóm.
- Trạng thái: M1 train xong 50 epochs, `best.pt` đã cắm vào `stream.py`. Logging/Telegram/API chạy. M2 vẫn synthetic — **không dùng F1=1.0**.

---

## 2. Đã hoàn thành (cộng dồn)

### Data M1 (Duy) — không đổi so với 09/09

- Schema 6 lớp: `0 person | 1 helmet | 2 no_helmet | 3 vest | 4 no_vest | 5 machinery`.
- 19.603 ảnh hợp lệ (train 13.717 / val 3.892 / test 1.994) → train sau weather augment **27.450**.
- Chia theo `group_key` `.rf.` (không chia theo frame). Chi tiết: `reports/data_distribution.md`, `reports/week01_data_stats.md`.

### M1 train (Hoài) — chốt số liệu

| Metric | Giá trị |
|---|---|
| Precision | 0.9051 |
| Recall | 0.8053 |
| mAP@0.5 | 0.8668 |
| mAP@0.5:0.95 | 0.6325 |
| Epochs | 50/50 |
| imgsz / batch / seed | 640 / 16 / 42 |

- Weights: `results_m1/runs/detect/train-2/weights/best.pt` (~50 MB). **Không push `.pt` lên GitHub.**
- Artefact: `results.csv`, `results.png`, confusion matrix, P/R/F1 curves.

### Hệ thống (Dũng)

- SQLite `events(id, track_id, violation_type, timestamp, frame_idx, confidence, image_path)`.
- Evidence JPG có overlay bbox: box dày đỏ = người vi phạm (`track_id`); box mỏng + nhãn lớp + conf; dòng `vtype  id=X  f=N` (`save_evidence` trong `stream.py`; alert có thêm `bbox` từ `RuleEngine`).
- FastAPI: `/health` `/events` `/events/count` `/evidence/{file}` `/docs`.
- `stream.py` nhận `--detector best.pt` (inference thật) + `--reset-db`.
- pytest: `tests/test_events.py`, `tests/test_rules.py`, `tests/test_api.py` (15 passed sau overlay).

### M2 (Năng) — code sẵn, số liệu chưa thật

- Feature 37-d / cửa sổ 30 frame / TCN RF=61. Pipeline `extract_pose.py` + `train_tcn.py` + `inference_m2.py`.
- Train hiện tại: synthetic 40 video. Test split thiếu lớp `normal`. UR Fall / Le2i **chưa** vào Input.
- **Không báo cáo F1=1.0.**

---

## 3. Kết quả mới (22/09) — demo M1 trên video công trường

Lệnh:

```bash
cd Nhom9_Tuan01_Skeleton
python src/pipeline/stream.py --reset-db \
  --source data/13002132_3840_2160_30fps.mp4 \
  --detector results_m1/runs/detect/train-2/weights/best.pt
```

| | Clip cũ `construction_test.mp4` (09/09) | Clip mới `13002132_3840_2160_30fps.mp4` (22/09) |
|---|---|---|
| Độ phân giải nguồn | (clip demo cũ) | 3840×2160 @ 30fps, ~35 MB |
| Frames xử lý | 404 | **235** |
| Resize pipeline | 640×640 | 640×640 |
| FPS đo được | ~11 | **5.8** (lần chạy overlay **6.8**) |
| Alert mới | **0** | **4** |
| Ảnh minh chứng | full frame, không bbox | overlay bbox + nhãn lớp + `vtype/id/frame` |
| DB | 3 dòng mock `no_helmet` nếu không `--reset-db` | reset rồi ghi 4 dòng thật |

4 sự kiện clip mới (ảnh **đã vẽ lại** 23/09):

| Frame | Track | Loại | Evidence |
|------:|------:|------|----------|
| 24 | 4 | no_vest | `logs/evidence/no_vest_4_000024.jpg` |
| 104 | 4 | machinery | `logs/evidence/machinery_4_000104.jpg` |
| 144 | 20 | no_vest | `logs/evidence/no_vest_20_000144.jpg` |
| 195 | 4 | no_helmet | `logs/evidence/no_helmet_4_000195.jpg` |

Confidence trong DB là conf YOLO của lớp vi phạm (không còn conf của person): 0.535 / 0.371 / 0.725 / 0.627 theo thứ tự bảng trên.

Thử `--imgsz` (cùng clip, cùng `best.pt`, máy local; FPS dao động 5.8–8.9 giữa các lần chạy):

| imgsz | FPS | decode+resize | inference | Event |
|---|---|---|---|---|
| 640 | 6.2 | 31.6 ms/frame | 79.7 ms/frame | 4 |
| 512 | 10.1 | 33.2 ms/frame | 48.0 ms/frame | 3 |

512 nhanh hơn ~1,6 lần nhưng mất 1 event và chưa đo mAP ở 512 → **chưa chuyển**, vẫn dùng 640. Decode 4K cố định ~30 ms/frame.

Telegram: chế độ mock (in console, không spam bot).

---

## 4. Vấn đề & rủi ro

- FPS ~6–9 (dao động theo lần chạy) trên clip 4K decode + YOLO < mục tiêu ≥15 FPS (Tuần 8). Đã thử `imgsz=512`: 10.1 FPS nhưng mất 1/4 event, chưa đủ 15. Phương án tiếp: YOLOv8s, chạy GPU, M2 mỗi 2 frame khi gắn TCN.
- Overlay evidence vẽ hết bbox M1 nên cảnh đông hơi rối; box đỏ dày mới là người bị alert. Chưa crop theo bbox.
- Clip 4K kéo về 640 có thể làm bbox/IoU lệch so với khung gốc — dùng để chứng minh luồng, chưa dùng để đo FAR/h chính thức.
- M2 chưa có số trên UR Fall/Le2i; domain shift chưa đo.
- `best.pt` / `*.mp4` / `logs/` / `results_m1/` không push Git.
- Nhánh M2 chưa gắn vào `stream.py`.

---

## 5. Việc tiếp theo

1. **Năng:** upload UR Fall + Le2i; extract pose theo video; split mọi tập đều có normal/falling/fallen; train 50 epochs GPU; gửi `best_tcn.pt` + P/R/F1 thật.
2. **Dũng:** khi có `best_tcn.pt` thì gắn buffer 30 frame theo `track_id` vào `stream.py`; đo FPS full pipeline.
3. **Hoài:** hỗ trợ AP per-class (đặc biệt vest/no_vest) + ngưỡng conf từng lớp nếu FAR PPE cao trên video mới.
4. **Duy:** giữ data.yaml/schema; không đổi ID lớp.

---

## 6. Demo buổi 23/09 (mở sẵn trước khi lên)

```bash
cd Nhom9_Tuan01_Skeleton
python src/pipeline/stream.py --reset-db \
  --source data/13002132_3840_2160_30fps.mp4 \
  --detector results_m1/runs/detect/train-2/weights/best.pt
# ~40s. Mở 4 JPG trong logs/evidence (đã có bbox). uvicorn src.api.app:app --port 8000 → /events /docs
```

Không demo `--mock` làm kết quả chính (chỉ fallback nếu GPU/ultralytics lỗi).

---

## 7. Phân công nói (2–3 phút / người)

Thứ tự: **Duy → Hoài → Dũng → Năng** (theo luồng data → train → vận hành → M2).

### 7.1. Duy — Data

- Gộp Hard Hat Workers + Construction Site Safety về 6 lớp; loại mask/cone.
- Chống rò rỉ `.rf.` group_key.
- 19.603 ảnh; train 27.450 sau rain/fog/dust. Val/test không augment.
- Vest/no_vest hiếm (~3–4% bbox) — giải thích nếu mAP lớp này thấp.

### 7.2. Hoài — M1

- 50 epochs YOLOv8m Kaggle T4, seed 42.
- P 0.9051 / R 0.8053 / mAP50 0.8668 / mAP50-95 0.6325.
- Mở `results.png` + confusion matrix. Không chiếu val COCO pretrained (mAP ~0.04 vô nghĩa).
- Weights gửi Dũng, không lên Git.

### 7.3. Dũng — System (phần mới tuần này)

- 09/09 mới mock + clip 0 alert. 22/09 chạy `best.pt` trên video công trường 4K; 23/09 vẽ bbox lên evidence rồi chạy lại.
- 235 frames, FPS 5.8 (lần overlay 6.8), 4 alert: 2 no_vest, 1 machinery, 1 no_helmet. Mở 4 JPG **đã overlay**: box đỏ = người vi phạm, nhãn lớp + conf.
- N=15 + cooldown 60s → không spam. Telegram mock.
- API `/events` đọc SQLite.
- Nếu hỏi 4 ảnh trông giống: camera đứng, cùng cảnh; phân biệt bằng box đỏ + chữ `vtype/id/frame`.
- Nếu hỏi FPS thấp: clip 4K decode, mục tiêu ≥15 FPS xử lý tuần 8 (512 / YOLOv8s).
- Nếu hỏi sao clip cũ 0 alert: nội dung video khác, cùng model — không phải hệ thống hỏng.

### 7.4. Năng — M2

- Spec 37 chiều, window 30, TCN RF=61, N=8, conf 0.70.
- **Nói thẳng:** synthetic; F1=1.0 không dùng; UR Fall/Le2i chưa upload.
- Interface `M2Inference.predict_with_rules` sẵn cho Dũng khi có weights.

---

## 8. File mở sẵn khi trình bày

| Người | Mở |
|---|---|
| Duy | `tools/prepare_m1.py` (`ALIASES`, `group_key`), `reports/data_distribution.md` |
| Hoài | `results_m1/runs/detect/train-2/results.png`, confusion matrix |
| Dũng | log chạy overlay, 4 JPG `logs/evidence/` (bbox đỏ + nhãn), `save_evidence` trong `stream.py`, SCHEMA `events.py` |
| Năng | `test.ipynb` cell feature spec + TCN; **không** chiếu F1=1.0 / domain-shift delta 0 |

---

## 9. Checklist nghiệm thu so với plan

| Mốc | Trạng thái 23/09 |
|---|---|
| Tuần 1 repo + mock stream | Xong |
| Tuần 2 rules pytest | Xong |
| Tuần 4 M1 thật + FPS M1 | Xong (mAP trên val; FPS 11 và 5.8 trên 2 clip) |
| Tuần 6 M2 trên pipeline | Chưa — chờ weights Năng |
| Tuần 7 E2E 4 nhánh + SQLite + Telegram | Một phần: 3 nhánh hình học M1 + DB + Telegram; thiếu té ngã |
| Tuần 8 FAR/h, ≥15 FPS | Chưa — mới có 4 event trên 1 clip ngắn |
| Tuần 10 dashboard | Chưa — mới FastAPI REST |
