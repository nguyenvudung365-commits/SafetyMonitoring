# Giải thích tầng luật an toàn (phần hệ thống – Nhóm 9)

Tài liệu này giải thích từng phần luật trong code, theo đúng bản hiện tại (đã gồm các sửa B.1, B.2 và luật cabin có điều kiện chân). Mọi ngưỡng nằm ở `configs/safety.yaml`, không hard-code trong `.py`.

## 1. Bức tranh tổng thể

```
Video frame
   │
   ▼
M1 (YOLOv8m) ──► danh sách detection {cls, conf, bbox}   (bbox chuẩn hoá 0–1)
   │                 6 lớp: person, helmet, no_helmet, vest, no_vest, machinery
   ▼
split_detections ──► persons | machines | others (mũ/áo)
   │
   ▼
ByteTracker (IoU) ──► gán track_id cho từng person
   │
   ▼
evaluate_person (mỗi người, mỗi frame) ──► tín hiệu thô: no_helmet / no_vest / zone / machinery
   │      (+ fall nếu M2 có xác suất)
   ▼
Xác nhận N frame liên tiếp  ──►  Cooldown theo (track_id, loại)  ──►  Chống trùng khi đổi id
   │
   ▼
merge_alerts (gộp loại cùng người, cùng frame) ──► event ghi SQLite / gửi Telegram
```

Các file chính:

| File | Vai trò |
|---|---|
| `src/rules/safety_rules.py` | Các hàm luật thuần (hình học, PPE, cooldown). Không import torch/ultralytics/cv2 nên test được nhanh. |
| `src/pipeline/postprocess.py` | `evaluate_person` (gom 4 luật), `RuleEngine` (theo thời gian), `merge_alerts`. |
| `src/tracking/bytetrack.py` | Tracker IoU gán `track_id`. |
| `configs/safety.yaml` | Toàn bộ ngưỡng và vùng cấm. |

Quy ước chung: bbox là `[x1, y1, x2, y2]` chuẩn hoá 0–1, gốc toạ độ ở góc trên trái, **y tăng xuống dưới**. Vì vậy "cạnh đáy" của bbox là `y2`, và "chân người" là `(tâm x, y2)`.

## 2. Hàm hình học dùng chung (`safety_rules.py`)

| Hàm | Làm gì |
|---|---|
| `foot_point(bbox)` | Điểm chân = tâm cạnh dưới bbox `((x1+x2)/2, y2)`. Dùng chân thay vì tâm vì chân mới nằm trên mặt đất, nên so với vùng cấm/máy móc chính xác hơn. |
| `point_in_polygon(pt, polygon)` | Ray casting: bắn tia ngang từ điểm, đếm số lần cắt cạnh đa giác; lẻ thì ở trong. Có `+1e-9` ở mẫu số để tránh chia cho 0 khi cạnh nằm ngang. |
| `bbox_edge_distance(point, bbox)` | Khoảng cách Euclid từ điểm tới **bbox gần nhất** (tới cạnh/góc); bằng 0 nếu điểm nằm trong bbox. |
| `overlap_ratio(inner, outer)` | Diện tích giao / diện tích của `inner`. Cho biết bao nhiêu phần của `inner` nằm trong `outer`. Trả 0 nếu `inner` suy biến (diện tích 0). |
| `bbox_center(bbox)` | Tâm bbox, dùng để gắn mũ/áo vào người. |

## 3. Luật 1 – Thiếu mũ / thiếu áo (`ppe_flags`)

**Ý tưởng:** mỗi người có thể có detection mũ/áo nằm gần. Chỉ tính detection nằm đúng **vùng cơ thể** của người đó, rồi quyết định có vi phạm không.

### 3.1. Gắn PPE vào người theo vị trí

- **Mũ** (`helmet`, `no_helmet`): tâm item phải nằm trong **1/3 trên** bbox người và trong chiều ngang bbox người (`_in_upper_third`).
- **Áo** (`vest`, `no_vest`): tâm item phải nằm trong **vùng thân**, từ 1/6 đến 2/3 chiều cao tính từ đỉnh bbox người (`_in_torso`). Áo ở vùng chân bị bỏ qua (có test `test_ppe_vest_at_legs_ignored`).

Làm vậy để mũ/áo của người khác đứng gần không bị gán nhầm sang người này.

### 3.2. Chọn kết luận khi có nhiều detection (B.1)

Với mỗi cặp (`helmet`/`no_helmet`) và (`vest`/`no_vest`):

1. Bỏ detection có `conf < ppe_conf` (mặc định 0.50).
2. Trong số còn lại, lấy detection có **conf cao nhất**.
3. Nếu hoà conf, **ưu tiên lớp tuân thủ** (`helmet`/`vest`) để tránh báo nhầm.
4. Vi phạm khi detection thắng là `no_helmet`/`no_vest`.

Kết quả **không phụ thuộc thứ tự** detection (có test hoán vị `test_ppe_order_independent`). Trước đây luật kiểu "helmet ghi đè no_helmet" làm kết quả phụ thuộc cách viết luật, nay đã thay bằng so sánh conf.

### 3.3. Điều cần nhớ

- **Không có detection nào** (hoặc đều dưới 0.50) thì **không báo**. Luật thiên về độ chính xác (ít báo nhầm) hơn độ bao phủ.
- Luật phụ thuộc hoàn toàn vào chất lượng M1 (mAP50 val 0.867).

## 4. Luật 2 – Vào vùng cấm (`zone_violation`)

- Vùng cấm là đa giác trong `zones.forbidden` (hiện có `crane_zone`, hình chữ nhật `[0.55–0.85] × [0.20–0.60]`).
- Vi phạm khi **điểm chân** nằm trong **bất kỳ** đa giác nào.
- Ở `evaluate_person` có thêm cửa: `person_conf >= zone_person_conf` (0.50). Lý do: móc cẩu/vật nhỏ hay bị nhận nhầm là người với conf thấp 0.27–0.32, nếu không lọc sẽ báo vùng cấm sai.

Hạn chế: đa giác cố định theo một góc camera. Đổi camera/video khác thì phải vẽ lại vùng cấm.

## 5. Luật 3 – Đứng gần máy móc (`machinery_violation`, B.2)

### 5.1. Ngưỡng theo phối cảnh

```
ngưỡng τ = machinery_k × chiều cao bbox người      (k = 0.5)
vi phạm  ⇔  khoảng cách(chân người, CẠNH bbox máy gần nhất) < τ
```

- Người ở xa trông nhỏ (h nhỏ) thì ngưỡng cũng nhỏ; người ở gần thì ngưỡng lớn. Nhờ vậy một ngưỡng tỉ lệ dùng được cho mọi khoảng cách tới camera, thay vì một số pixel cố định.
- Đo tới **cạnh** bbox máy chứ không tới tâm. Máy to nên tâm rất xa dù người đứng sát thân máy (có test `test_machinery_edge_not_center`).

### 5.2. Loại trừ người ngồi trong cabin

Người lái nằm gọn trong bbox máy thì không phải vi phạm. Luật bỏ qua máy đó khi **cả hai** điều kiện đúng:

1. `overlap_ratio(người, máy) >= cabin_overlap` (0.8): ít nhất 80% bbox người nằm trong bbox máy.
2. `y2 người <= y2 máy`: chân người còn nằm trong bbox máy.

Điều kiện 2 thêm ở lần sửa gần nhất. Lý do: người đứng ngay **trước** máy (gần camera hơn) cũng có bbox chồng lấn lớn trong ảnh 2D, nhưng chân nằm **dưới** cạnh đáy bbox máy. Chỉ xét chồng lấn thì bỏ sót ca nguy hiểm này.

Ví dụ kiểm chứng:

| Ca | Người | Máy | Kết quả |
|---|---|---|---|
| Đứng trước máy | `[0.4, 0.4, 0.45, 0.85]` | `[0.2, 0.2, 0.8, 0.8]` | chồng lấn 0.89 nhưng `y2 người (0.85) > y2 máy (0.8)`, không loại trừ; chân cách cạnh đáy 0.05 < τ = 0.225, **báo** |
| Ngồi trong cabin | nằm gọn trong máy, `y2 ≤ y2 máy` | cùng máy | **bỏ qua** |

Test: `test_machinery_cabin_requires_foot_inside_machine`.

Hạn chế còn lại: đây vẫn là heuristic 2D, không có độ sâu. Người đứng sau máy, chân bị che nhưng nằm trong bbox máy, vẫn có thể bị coi là người trong cabin.

## 6. Luật 4 – Té ngã (nhánh `fall`)

- Đầu vào là `fall_probs = {track_id: xác suất}` do **M2** (Năng) tính.
- Vi phạm thô khi xác suất `> fall_conf` (0.70), và cần **8 frame liên tiếp** (`fall_frames`, khoảng 0.27 giây).
- Nếu `fall_probs is None` (chưa chạy M2) thì **bỏ hẳn nhánh này**. Hiện M2 chưa có weights nên nhánh té ngã chưa hoạt động thật; có test `test_fall_branch_off_without_m2`.

## 7. Lớp thời gian: biến tín hiệu thô thành sự kiện (`RuleEngine`, `SafetyState`)

Mỗi frame, với mỗi track và mỗi loại vi phạm, luật chạy theo chuỗi sau.

### 7.1. Xác nhận N frame liên tiếp (`confirm`)

Giữ lịch sử `bool` của `N` frame gần nhất cho khoá `(track_id, loại)`. Chỉ khi **cả N frame đều True** mới coi là vi phạm thật.

- PPE, vùng cấm, máy móc: `ppe_frames = 15` (khoảng 0.5 giây ở 30 fps).
- Té ngã: `fall_frames = 8`.

Mục đích: lọc nhiễu detection một vài frame (nhấp nháy `no_helmet`, bbox sai thoáng qua).

### 7.2. Cooldown (`should_alert`)

Cùng `(track_id, loại)` không báo lại trong `cooldown_sec = 60` giây. Số frame cooldown = `cooldown_sec × fps` của video (có test `test_cooldown_scales_with_video_fps`).

Hệ quả khi đánh giá: một vi phạm kéo dài hơn 60 giây sẽ sinh event thứ hai; bước đánh giá coi đó là dương tính giả.

### 7.3. Chống trùng khi tracker đổi id (`_is_duplicate`)

Tracker IoU đôi khi mất rồi cấp id mới cho cùng một người. Khi một track mới vừa xác nhận vi phạm, luật coi đó là **trùng** (không tạo event) nếu tồn tại track khác đã báo **cùng loại** thoả:

- track đó không còn xuất hiện ở frame hiện tại;
- lần thấy cuối cách đây không quá `dedup_sec = 3` giây;
- IoU giữa bbox hiện tại và bbox cuối của nó `>= dedup_iou` (0.2).

Khi đó loại vi phạm vẫn được ghi nhận cho track mới (để không báo lại) nhưng không sinh event.

### 7.4. Gộp loại (`merge_alerts`)

Nếu một người cùng được xác nhận nhiều loại trong **cùng một frame**, các loại được gộp thành một event, nối bằng `+` (ví dụ `no_helmet+no_vest`). Khi đánh giá, event gộp khớp với nhãn theo từng loại thành phần.

## 8. Tracker (`ByteTracker`)

Tên là ByteTrack nhưng thực chất là tracker **IoU greedy** đơn giản:

- Tính IoU giữa mọi track hiện có và mọi bbox mới, sắp xếp giảm dần, ghép tham lam khi `IoU >= 0.3`.
- Track không được ghép tăng bộ đếm `lost`, bị xoá khi `lost > 30` frame.
- Bbox không ghép được thì tạo track mới với id tăng dần.
- Không có Kalman filter, không dùng đặc trưng hình dạng. Người đi nhanh, che khuất lâu hoặc đứng sát nhau dễ bị đổi id. Đó là lý do có lớp chống trùng ở 7.3. Thay bằng ByteTrack thật đã bị loại khỏi phạm vi vì rủi ro cao.

## 9. Bảng ngưỡng và cách chỉnh (`configs/safety.yaml`)

| Tham số | Giá trị | Tăng thì | Giảm thì |
|---|---|---|---|
| `machinery_k` | 0.5 | báo gần máy xa hơn (nhiều báo, dễ nhiễu) | chỉ báo khi sát máy |
| `cabin_overlap` | 0.8 | khó coi là người trong cabin hơn (dễ báo người lái) | dễ bỏ qua (có thể bỏ sót người đứng cạnh máy) |
| `ppe_conf` | 0.50 | ít báo PPE hơn, bỏ sót nhiều hơn | báo nhiều hơn, dễ nhiễu |
| `zone_person_conf` | 0.50 | lọc mạnh vật nhận nhầm là người | dễ báo vùng cấm sai |
| `dedup_iou` | 0.20 | khó coi là trùng (dễ nhân đôi event) | dễ gộp nhầm hai người gần nhau |
| `fall_conf` | 0.70 | ít báo té ngã hơn | báo nhiều hơn |
| `ppe_frames` | 15 | xác nhận chậm, ít nhiễu | xác nhận nhanh, nhiều nhiễu |
| `fall_frames` | 8 | như trên, cho té ngã | như trên |
| `dedup_sec` | 3 | cửa sổ gộp rộng hơn | hẹp hơn |
| `cooldown_sec` | 60 | báo lại thưa hơn | báo lại dày hơn |

Nên chỉnh theo số đo thật. Công cụ `tools/sweep_rules.py` quét tham số và `tools/eval_events.py` đo precision/recall/FAR theo giờ trên nhãn sự kiện tay.

## 10. Test bao phủ luật (`tests/test_rules.py`)

| Nhóm | Test tiêu biểu |
|---|---|
| Hình học | `test_zone_inside`, `test_bbox_edge_distance_and_overlap`, `test_iou` |
| Máy móc | `test_machinery`, `test_machinery_edge_not_center`, `test_machinery_cabin_excluded`, `test_machinery_cabin_requires_foot_inside_machine`, `test_machinery_threshold_scales_with_height` |
| Vùng cấm | `test_zone_violation_foot`, `test_zone_ignores_low_conf_person` |
| PPE | `test_ppe_no_helmet_in_upper_third`, `test_ppe_order_independent`, `test_ppe_higher_conf_violation_wins_and_tie_prefers_compliant`, `test_ppe_vest_on_torso_detected`, `test_ppe_vest_at_legs_ignored`, `test_ppe_outside_person_ignored` |
| Thời gian | `test_confirm`, `test_cooldown`, `test_cooldown_scales_with_video_fps`, `test_rule_engine_confirm_then_alert` |
| Chống trùng | `test_dedup_when_tracker_changes_id`, `test_no_dedup_for_far_person`, `test_merge_alerts_same_track` |
| Té ngã | `test_fall_alert_after_n8_frames`, `test_fall_no_alert_below_conf_or_short`, `test_fall_branch_off_without_m2`, `test_fall_cooldown_single_event` |

Chạy: `python -m pytest -q` (hiện 108 test qua).

## 11. Hạn chế đã biết

1. **Hình học 2D.** Không có độ sâu; luật cabin và khoảng cách máy móc là heuristic trên ảnh.
2. **Phụ thuộc M1.** Thiếu detection thì không báo (PPE); detection sai thì báo sai.
3. **Tracker đơn giản.** Đổi id ảnh hưởng xác nhận N frame và cooldown; lớp chống trùng chỉ giảm bớt.
4. **Vùng cấm cố định** theo một góc camera.
5. **Té ngã chưa chạy thật** vì M2 chưa có weights.
6. **Số liệu trước/sau khi sửa B.1, B.2 chưa đo lại** trên clip 1:36 và clip máy xúc, và chưa có nhãn sự kiện thật trong `data/gt_events/`. Không nên đưa precision/recall vào báo cáo trước khi đo.
