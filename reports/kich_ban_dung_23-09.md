# Kịch bản nói — Nguyễn Vũ Dũng — 23/09/2026

**Người nói:** Nguyễn Vũ Dũng, System Lead, MSSV 0205968.  
**Thời lượng:** 2,5–3 phút phần chính + Q&A.  
**Thứ tự nhóm:** Duy (data) → Hoài (M1) → **Dũng** → Năng (M2).  
**Không nói:** train M1, số liệu data, F1 M2. Đó là phần người khác.

Mở sẵn trước khi lên:

1. VS Code: `src/pipeline/stream.py`, `src/storage/events.py`, `configs/safety.yaml`
2. Explorer: `Nhom9_Tuan01_Skeleton/logs/evidence/` (4 JPG đã overlay bbox, 23/09)
3. Terminal đã chạy xong lệnh demo (hoặc log copy sẵn)
4. Chrome: GitHub repo + `http://127.0.0.1:8000/docs` nếu kịp bật API

```bash
cd Nhom9_Tuan01_Skeleton
python src/pipeline/stream.py --reset-db ^
  --source data/13002132_3840_2160_30fps.mp4 ^
  --detector results_m1/runs/detect/train-2/weights/best.pt
uvicorn src.api.app:app --port 8000
```

---

## 0. Câu mở (15 giây)

Em là Dũng, phụ trách **hệ thống**: đọc video, gắn mô hình M1 của chị Hoài, tracker, tầng luật YAML, ghi SQLite, lưu ảnh minh chứng, Telegram, FastAPI.

Tuần trước em mới demo `--mock`. Tuần này em chạy **inference thật** trên video công trường.

---

## 1. Luồng hệ thống (45 giây)

Chỉ `stream.py` và nói 5 bước, không đọc code từng dòng:

1. Đọc MP4, resize **640×640**.
2. YOLOv8m `best.pt` — 6 lớp: person, helmet, no_helmet, vest, no_vest, machinery.
3. ByteTrack giữ `track_id` theo người.
4. RuleEngine đọc `configs/safety.yaml`: N=15 frame mới xác nhận PPE/máy, cooldown **60 giây** theo `(track_id, loại vi phạm)`, `ppe_conf=0.5`.
5. Có alert thì: insert SQLite + JPG overlay bbox `logs/evidence/` + Telegram. Box đỏ dày = người `track_id` vi phạm; box mỏng = mọi detection M1 (nhãn lớp + conf); chữ góc trên `vtype id= f=`.

Bảng SQLite (mở `events.py` SCHEMA):

`id, track_id, violation_type, timestamp, frame_idx, confidence, image_path`

Telegram chưa gắn token buổi này → in `[TELEGRAM MOCK]`, code live `sendMessage`/`sendPhoto` đã có.

---

## 2. Kết quả đo — đây là phần mới, nói chậm (60 giây)

So với 09/09:

| | 09/09 `construction_test.mp4` | 22/09 `13002132_3840_2160_30fps.mp4` |
|---|---|---|
| Model | `best.pt` | cùng `best.pt` |
| Frames | 404 | **235** |
| FPS | ~11 | **5.8** (lần overlay **6.8**) |
| Alert | **0** | **4** |
| Ảnh | full frame | overlay bbox + nhãn |

Bốn sự kiện — **mở 4 ảnh đã vẽ**, chỉ vào **box đỏ** và chữ góc trên:

- Frame 24, track 4 — **no_vest** — `no_vest_4_000024.jpg`
- Frame 104, track 4 — **machinery** — `machinery_4_000104.jpg`
- Frame 144, track 20 — **no_vest** — `no_vest_20_000144.jpg`
- Frame 195, track 4 — **no_helmet** — `no_helmet_4_000195.jpg`

Câu chốt: cùng weights, đổi video thì có/không alert. Bốn tấm nền giống vì camera đứng; **không phải copy** — loại vi phạm và người được khoanh đỏ khác nhau.

Nếu kịp: Chrome `/events` — JSON 4 dòng trùng bảng trên.

---

## 3. Luật N và cooldown (20 giây)

Không spam 235 dòng. Thiếu PPE phải đủ **15 frame** liên tiếp. Cùng người cùng loại phải chờ **60 giây**. Nên 235 frame chỉ ra 4 event là đúng thiết kế, không phải model yếu.

---

## 4. Việc chưa làm — nói trước khi GV hỏi (20 giây)

- **Chưa gắn M2** (té ngã) — chờ `best_tcn.pt` của Năng.
- **Chưa dashboard**; mới REST `/health` `/events` `/events/count` `/docs`.
- **FPS 5.8** dưới mục tiêu 15. Clip **4K** decode rồi kéo 640. Tuần 8 sẽ thử `imgsz=512` hoặc YOLOv8s.
- Video này để chứng minh **luồng end-to-end**, chưa phải số FAR/h chính thức.

---

## 5. Câu kết (10 giây)

Tuần này: M1 thật + video công trường + 4 vi phạm có ảnh overlay bbox + API đọc DB. Tuần sau: gắn M2 khi có weights, đo FPS full pipeline.

---

## Q&A — trả lời ngắn

**Sao tuần trước 0 alert?**  
Nội dung clip khác. Cùng `best.pt`. Clip mới có người/PPE/máy rõ hơn.

**FPS 5.8 có đạt không?**  
Chưa. Mục tiêu ≥15 ở tuần 8. Em đã đo: decode 4K ~30 ms/frame, inference ~66–80 ms/frame ở 640. Thử `imgsz=512` lên 10.1 FPS nhưng mất 1/4 event nên chưa đổi. Tiếp theo: YOLOv8s hoặc GPU.

**Telegram gửi điện thoại chưa?**  
Code có. Buổi này mock để không spam. Gắn token là gửi ảnh + chữ.

**ByteTrack thư viện hay tự viết?**  
IoU greedy theo `track_id` trong repo (`src/tracking/`). Đủ gán ID cho cooldown. Rủi ro đổi ID → cooldown lệch; sẽ log nếu gặp.

**Sao resize 640, mất chi tiết 4K?**  
Khớp train M1 `imgsz=640`. 4K full sẽ chậm hơn nữa. Đổi 512 phải báo trade-off mAP.

**Confidence trong DB là gì?**  
Conf YOLO của lớp vi phạm (no_vest, no_helmet, machinery) gắn với người đó: 0.535 / 0.371 / 0.725 / 0.627. Riêng chế độ `--mock` vẫn gán 0.90 vì là dữ liệu giả.

**Sao machinery conf 0.371 mà vẫn alert?**  
Ngưỡng `ppe_conf=0.5` chỉ áp cho mũ/áo; máy dùng khoảng cách < 0.08. Có thể thêm ngưỡng conf riêng cho máy nếu FAR cao.

**Vùng cấm?**  
Luật polygon có trong YAML. Clip này **không** demo zone. Demo: PPE + gần máy.

**M2?**  
Interface sẵn. Chưa nhét `stream.py`. Năng báo số thật rồi em gắn buffer 30 frame theo `track_id`.

**Weights trên Git?**  
Không. `best.pt` ~50MB, `.gitignore`. Demo local.

**pytest?**  
`test_events`, `test_rules`, `test_api` — logic luật/DB/API, không thay video.

**1 người 3 loại vi phạm?**  
Track 4: no_vest, machinery, no_helmet — loại khác nên cooldown riêng. Đúng thiết kế hợp nhất đa vi phạm theo ID.

**Sao 4 ảnh gần như một cảnh?**  
Camera đứng, nền ít đổi. Trước overlay trông như copy. Giờ phân biệt bằng box đỏ + chữ `vtype id= f=`. Overlay vẽ hết bbox M1 nên hơi rối — người vi phạm là box đỏ dày.

---

## Cấm nói

- Không đọc mAP (Hoài).
- Không nói 19.603 ảnh (Duy).
- Không nói F1=1.0 / synthetic như kết quả tốt (Năng).
- Không hứa dashboard tuần này.
- Không bịa FAR/h, không bịa FPS GPU nếu chạy máy local 5.8.
- Không nhận “hệ thống phát hiện 100%”.

---

## Bản nói liền (học thuộc ~2 phút 40)

Em Dũng, phần hệ thống.

Pipeline: video vào, resize 640, YOLOv8m `best.pt` sáu lớp, gắn ID, RuleEngine YAML — N=15, cooldown 60 giây — rồi ghi SQLite, lưu JPG overlay bbox, Telegram.

Tuần trước mock và một clip 404 frame, 11 FPS, không alert. Ngày 22 em đổi clip công trường 4K `13002132`, cùng `best.pt`: **235 frame, 5.8 FPS, 4 sự kiện**. Ảnh minh chứng vẽ lại 23/09: box đỏ là người vi phạm, nhãn lớp + conf.

Ảnh: track 4 thiếu áo frame 24; gần máy frame 104; track 20 thiếu áo frame 144; track 4 thiếu mũ frame 195. Nền giống vì camera đứng, không phải 4 file copy. API `/events` đọc đúng 4 dòng.

Bốn event vì luật N và cooldown, không spam. FPS 5.8–6.8 do 4K, chưa đạt 15 — tuần 8 giảm độ phân giải hoặc model nhỏ hơn. Chưa gắn M2, chưa dashboard. Phần em tuần này là **chạy thật được, có minh chứng có bbox**.
