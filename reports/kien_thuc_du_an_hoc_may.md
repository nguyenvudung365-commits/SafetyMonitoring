# Giải thích dự án từ đầu: Học máy + Học máy nâng cao qua ví dụ thật

Tài liệu cho Dũng (System Lead, Nhóm 9). Đọc từ trên xuống: mỗi phần dùng lại ý của phần trước.
Mọi ví dụ số là **số thật của dự án** (ghi rõ nguồn) hoặc số giả định (ghi "giả sử").

Mục lục
1. Dự án làm gì (bài toán)
2. Học máy là gì (nền tảng)
3. Mạng nơ-ron và học sâu
4. Thị giác máy tính: CNN và phát hiện vật thể
5. YOLOv8 chi tiết
6. Cách huấn luyện M1 trong dự án
7. Cách đo mô hình tốt hay dở (Precision, Recall, AP, mAP)
8. Theo dõi (tracking)
9. Tầng luật (RuleEngine): công thức hình học
10. Xác nhận N frame, cooldown, gộp sự kiện
11. M2: phát hiện té ngã (chuỗi thời gian, TCN)
12. Hệ thống: SQLite, FastAPI, Telegram, FPS
13. Đánh giá hệ thống thật: vì sao "xem bằng mắt 1 clip" chưa phải độ chính xác
14. Lỗi đã gặp và vì sao xảy ra
15. Bảng thuật ngữ nhanh
16. Câu hỏi ôn tập (để tự kiểm tra)

---

## 1. Dự án làm gì

**Bài toán:** camera công trường quay video. Hệ thống phải tự báo khi thấy:

| Vi phạm | Ý nghĩa |
|---|---|
| `no_helmet` | người không đội mũ bảo hộ |
| `no_vest` | người không mặc áo phản quang |
| `zone` | người bước vào vùng cấm (ví dụ dưới tầm cần cẩu) |
| `machinery` | người đứng quá gần máy móc |
| `fall` | người bị ngã (nhánh M2, chưa gắn vào pipeline) |

**Luồng chạy (pipeline):**

```
video → tách từng frame → resize 640×640
      → M1 (YOLOv8m) phát hiện: người, mũ, áo, máy...
      → Tracker gán ID cho từng người qua các frame
      → RuleEngine áp luật (mũ trên đầu? chân trong vùng cấm? gần máy?)
      → xác nhận N frame liên tiếp + cooldown (chống báo trùng)
      → lưu SQLite + ảnh bằng chứng + tin Telegram
      → FastAPI cho xem lại
```

**Vì sao không dùng thẳng một mạng "nhìn là biết vi phạm"?** Vì tách việc ra thì dễ kiểm tra từng bước:
- Phần "nhìn" (M1) là **học máy**: học từ ảnh.
- Phần "luật" là **code thường**: ngưỡng và hình học, dễ giải thích, dễ sửa không cần train lại.
Đây là thiết kế lai (hybrid) điển hình: học máy cho nhận thức, luật cho quyết định.

**Phân công:** Duy (dữ liệu + công cụ gán nhãn), Hoài (train M1), Dũng (pipeline/hệ thống), Năng (M2 té ngã).

---

## 2. Học máy là gì (nền tảng)

### 2.1 Ý tưởng
Cách lập trình thường: bạn **viết luật** → máy chạy.
Học máy: bạn đưa **ví dụ có đáp án** → máy **tự tìm luật**.

Ví dụ ngoài đời: thay vì viết "mũ bảo hộ là vật tròn, màu vàng, nằm trên đầu" (sẽ sai khi mũ màu trắng, xanh, nghiêng...), ta đưa 19.603 ảnh đã khoanh sẵn mũ, để máy tự học mũ trông như thế nào.

### 2.2 Ba loại học máy
| Loại | Dữ liệu | Ví dụ |
|---|---|---|
| **Có giám sát** (supervised) | (đầu vào, đáp án) | Ảnh + khung khoanh mũ. **Dự án dùng loại này** (M1, M2) |
| Không giám sát | chỉ đầu vào | Gom nhóm công nhân theo kiểu chuyển động |
| Tăng cường (RL) | phần thưởng | Robot học đi |

### 2.3 Hồi quy và phân loại
- **Phân loại:** đầu ra là nhãn. "Ảnh này là helmet hay no_helmet?"
- **Hồi quy:** đầu ra là số. "Hộp khoanh nằm ở toạ độ nào?"
- Phát hiện vật thể (YOLO) **làm cả hai cùng lúc**: vừa hồi quy bbox `[x1,y1,x2,y2]`, vừa phân loại lớp.

### 2.4 Mô hình, tham số, hàm mất mát
- **Mô hình** = một hàm `ŷ = f(x; θ)`. `x` là ảnh, `θ` (theta) là hàng triệu số gọi là **tham số/trọng số**.
- YOLOv8m có khoảng 26 triệu tham số (file `best.pt` ~52 MB).
- **Hàm mất mát (loss)** `L(θ)` = số đo "dự đoán sai bao nhiêu". Càng nhỏ càng tốt.
- **Huấn luyện** = tìm `θ` làm `L` nhỏ nhất.

### 2.5 Gradient descent (công thức quan trọng nhất)
Ý tưởng: đứng trên đồi trong sương mù, muốn xuống thung lũng thì bước theo hướng dốc xuống nhất.

```
θ ← θ − η · ∇L(θ)
```
- `∇L(θ)` (gradient) = hướng làm `L` tăng nhanh nhất; đi ngược lại thì `L` giảm.
- `η` (eta) = **learning rate** (bước đi). Quá lớn thì nhảy qua thung lũng; quá nhỏ thì đi rất lâu.

Thực tế dùng **mini-batch SGD**: mỗi bước chỉ lấy một nhóm ảnh nhỏ (dự án: **batch = 16** ảnh), tính gradient trên nhóm đó. Rẻ hơn nhiều so với tính trên 13.717 ảnh một lúc.

**Epoch** = đã đi qua toàn bộ tập train một lượt. Dự án: 50 epoch.
Số bước mỗi epoch ≈ 13.717 / 16 ≈ 857 (giả sử không augment; thực tế train có augment 27.450 ảnh nên nhiều hơn).

### 2.6 Chia dữ liệu: train / val / test
| Tập | Dùng để | Dự án |
|---|---|---|
| train | cập nhật `θ` | 13.717 ảnh (27.450 sau weather augment) |
| val | chọn mô hình tốt nhất, chỉnh siêu tham số | 3.892 |
| test | đo cuối cùng, **chỉ nhìn một lần** | 1.994 (Hoài chưa chạy) |

**Vì sao phải tách?** Giống thi: không được cho học sinh thấy đề thi trước. Nếu đo trên tập đã học thì điểm cao giả.

**Rò rỉ dữ liệu (data leakage):** ảnh gần giống nhau (cùng video/cùng ảnh gốc Roboflow) mà rơi vào cả train lẫn val thì val "dễ" một cách giả tạo. Duy đã chia **theo nhóm ảnh gốc** (`group_key` dạng `.rf.`), không chia từng ảnh, để tránh việc này. Công cụ `label_tool.html` hiện chia theo **từng ảnh** nên nếu dùng ảnh từ cùng video thì có nguy cơ rò rỉ.

### 2.7 Overfitting và underfitting
- **Underfitting:** mô hình quá đơn giản, cả train lẫn val đều tệ.
- **Overfitting:** học thuộc lòng train, val tệ. Dấu hiệu: loss train giảm mãi nhưng loss val tăng lại.
- Cách chống: thêm dữ liệu, **augmentation**, dừng sớm, regularization, mô hình nhỏ hơn.

**Data augmentation:** biến đổi ảnh có sẵn (lật, đổi màu, thêm mưa/sương/bụi). Dự án thêm hiệu ứng thời tiết (rain/fog/dust) cho tập train: 13.717 → 27.450 ảnh. **Val/test không augment**, vì phải đo trên dữ liệu thật.

### 2.8 Mất cân bằng lớp (class imbalance)
Trong dataset, vest/no_vest chỉ chiếm ~3–4% số bbox, còn person/helmet nhiều hơn rất nhiều. Mô hình thấy ít ví dụ vest nên học kém hơn. Đó là lý do AP của vest (0.814) thấp hơn helmet (0.962), và vì sao lỗi "vest phản quang màu nhạt" xảy ra.

---

## 3. Mạng nơ-ron và học sâu

### 3.1 Một nơ-ron
```
y = σ( w₁x₁ + w₂x₂ + … + wₙxₙ + b )
```
- `x` đầu vào, `w` trọng số, `b` độ lệch, `σ` hàm kích hoạt.
- Không có `σ` phi tuyến thì xếp bao nhiêu lớp cũng chỉ là một phép tuyến tính. `σ` cho mạng khả năng học đường cong.

Hàm kích hoạt hay gặp:
| Tên | Công thức | Dùng |
|---|---|---|
| ReLU | `max(0, x)` | lớp ẩn, rất phổ biến |
| Sigmoid | `1 / (1 + e⁻ˣ)` | ra xác suất 0–1 cho một lớp |
| Softmax | `e^{zᵢ} / Σⱼ e^{zⱼ}` | ra phân phối xác suất trên nhiều lớp (tổng = 1) |
| SiLU | `x · sigmoid(x)` | YOLOv8 dùng |

Ví dụ softmax (M2, ba trạng thái normal/falling/fallen): logit `[2.0, 0.5, 0.1]` → `e² = 7.39`, `e^0.5 = 1.65`, `e^0.1 = 1.11`, tổng 10.15 → xác suất `[0.73, 0.16, 0.11]`.

### 3.2 Học sâu
Xếp nhiều lớp nơ-ron: lớp đầu học đặc trưng đơn giản (cạnh, góc), lớp sau ghép thành đặc trưng phức tạp (bánh xe, mặt người, mũ). Không cần ta tự thiết kế đặc trưng, đó là điểm mạnh so với học máy cổ điển.

### 3.3 Lan truyền ngược (backpropagation)
Để chạy gradient descent cần `∂L/∂w` cho từng trọng số. Dùng **quy tắc chuỗi**:

```
∂L/∂w = (∂L/∂ŷ) · (∂ŷ/∂z) · (∂z/∂w)
```
Tính từ lớp cuối ngược về lớp đầu, nên gọi "lan truyền ngược". PyTorch tự làm việc này (`loss.backward()`), mình không viết tay.

### 3.4 Các bộ tối ưu và chuẩn hoá
- **SGD + momentum:** có "quán tính" để bớt dao động.
- **Adam/AdamW:** tự điều chỉnh bước cho từng tham số. Ultralytics mặc định chọn tự động (`optimizer=auto`).
- **Batch Normalization:** chuẩn hoá đầu ra mỗi lớp (trung bình 0, phương sai 1) giúp train ổn định, nhanh hơn.
- **Learning rate schedule:** giảm dần learning rate theo epoch.

---

## 4. Thị giác máy tính: CNN và phát hiện vật thể

### 4.1 Ảnh là gì với máy
Ảnh = ma trận số. Ảnh 640×640 màu = tensor `640 × 640 × 3` (ba kênh R, G, B), mỗi số 0–255. Pipeline resize mọi frame về 640×640 trước khi đưa vào M1 (`--imgsz 640`).

### 4.2 Phép tích chập (convolution)
Một **bộ lọc (kernel)** nhỏ, ví dụ 3×3, trượt qua ảnh. Tại mỗi vị trí: nhân từng ô rồi cộng lại.

```
(I * K)(i,j) = Σₘ Σₙ I(i+m, j+n) · K(m,n)
```
Ví dụ bộ lọc phát hiện cạnh dọc:
```
K = [[-1, 0, 1],
     [-1, 0, 1],
     [-1, 0, 1]]
```
Vùng sáng-tối chuyển đột ngột theo chiều ngang sẽ cho giá trị lớn. Trong CNN, các số trong `K` **không do người đặt** mà do gradient descent học.

Kích thước đầu ra: `⌊(W − K + 2P) / S⌋ + 1`
- `W` rộng ảnh, `K` cỡ kernel, `P` padding, `S` stride.
- Ví dụ `W=640, K=3, P=1, S=2` → `⌊(640−3+2)/2⌋+1 = 320`. Stride 2 làm ảnh nhỏ đi một nửa.

### 4.3 Vì sao CNN hợp với ảnh
- **Chia sẻ trọng số:** cùng một bộ lọc dùng ở mọi vị trí, nên ít tham số, và "mũ ở góc trái" hay "mũ ở góc phải" đều nhận ra.
- **Tính cục bộ:** mỗi nơ-ron chỉ nhìn vùng nhỏ.
- **Vùng nhìn (receptive field)** tăng dần khi xếp nhiều lớp: lớp sau "thấy" vùng ảnh rộng hơn.

### 4.4 Từ phân loại đến phát hiện vật thể
| Bài toán | Đầu ra |
|---|---|
| Phân loại ảnh | 1 nhãn cho cả ảnh |
| **Phát hiện vật thể** | **nhiều** `(lớp, độ tin cậy, bbox)` |
| Phân đoạn | nhãn cho từng pixel |

Đầu ra phát hiện của M1 cho mỗi vật: `{cls, conf, bbox}`. Ví dụ `{cls: 1 (helmet), conf: 0.62, bbox: [0.41, 0.20, 0.45, 0.25]}`. Toạ độ **chuẩn hoá 0–1** theo kích thước ảnh, nên không phụ thuộc độ phân giải.

### 4.5 IoU (Intersection over Union)
Đo hai khung chồng nhau bao nhiêu.

```
IoU(A, B) = Diện tích(A ∩ B) / Diện tích(A ∪ B)
```
- 1 = trùng hoàn toàn, 0 = không chạm.

Ví dụ: hai hộp vuông cạnh 10, hộp B dịch sang phải 5. Giao = 5×10 = 50. Hợp = 100 + 100 − 50 = 150. IoU = 50/150 = **0.33**.

Trong dự án IoU xuất hiện ở **ba chỗ**: (1) tính loss bbox lúc train, (2) NMS loại hộp trùng, (3) tracker ghép người giữa hai frame (`iou_thresh = 0.3`), và bước gộp sự kiện (`dedup_iou = 0.2`). Hàm nằm ở [bytetrack.py](../src/tracking/bytetrack.py).

### 4.6 NMS (Non-Maximum Suppression)
Một người thường bị mô hình đề xuất nhiều hộp chồng nhau. NMS giữ hộp conf cao nhất, xoá các hộp cùng lớp có IoU với nó lớn hơn ngưỡng, rồi lặp lại. Ultralytics làm bước này tự động trong khi suy luận.

### 4.7 Ngưỡng tin cậy (confidence threshold)
Mô hình trả mọi hộp có conf ≥ ngưỡng mặc định (0.25 trong Ultralytics). Dự án đặt **thêm ngưỡng riêng theo luật**:
- `ppe_conf = 0.50` cho mũ/áo,
- `zone_person_conf = 0.50` cho vùng cấm (mới thêm).

**Đánh đổi:** ngưỡng thấp → bắt được nhiều hơn (Recall ↑) nhưng báo nhầm nhiều hơn (Precision ↓). Ví dụ thật: móc cẩu bị nhận là `person` với conf 0.27–0.32 nên báo `zone` sai. Đặt ngưỡng 0.5 cho luật vùng cấm loại được lỗi này, và sự kiện zone nhầm ở frame 1632 biến mất.

---

## 5. YOLOv8 chi tiết

**YOLO = You Only Look Once:** nhìn ảnh **một lần** là ra toàn bộ hộp và nhãn, nên nhanh (khác các cách cũ đề xuất vùng rồi mới phân loại từng vùng).

### 5.1 Ba khối
```
Ảnh 640×640×3
   ↓
Backbone  (rút đặc trưng: CSPDarknet với khối C2f)
   ↓
Neck      (trộn đặc trưng nhiều tỉ lệ: PAN/FPN)
   ↓
Head      (đầu ra: bbox + lớp, ở 3 độ phân giải)
```
- **Backbone** học "ảnh có gì".
- **Neck** ghép thông tin chi tiết (độ phân giải cao, vật nhỏ) với thông tin ngữ nghĩa (độ phân giải thấp, vật lớn).
- **Head** dự đoán ở ba mức: lưới 80×80 (vật nhỏ), 40×40 (vừa), 20×20 (lớn) với ảnh 640. Tổng số ô dự đoán `80² + 40² + 20² = 8400`.

Vì thế người xa, nhỏ (như người trong ống cống) khó: dễ bị mất ở độ phân giải thấp. Giảm `imgsz` xuống 512 làm vật nhỏ khó hơn nữa, một lý do bản 512 có nhiều báo nhầm hơn.

### 5.2 Đặc điểm của YOLOv8
- **Anchor-free:** không cần hộp mẫu định sẵn; dự đoán trực tiếp khoảng cách từ tâm ô đến bốn cạnh.
- **Decoupled head:** nhánh bbox và nhánh phân loại tách riêng.
- **Gán nhãn động (Task-Aligned Assigner):** chọn ô dự đoán nào chịu trách nhiệm cho vật nào dựa trên độ khớp giữa chất lượng phân loại và độ chính xác bbox.

### 5.3 Các cỡ model
`n < s < m < l < x`. Dự án dùng **YOLOv8m** (medium, ~26M tham số): cân bằng giữa độ chính xác và tốc độ.

### 5.4 Loss của YOLOv8
```
L = λ_box · L_box + λ_cls · L_cls + λ_dfl · L_dfl
```
- **L_box = 1 − CIoU:** phạt hộp lệch. CIoU là IoU cộng thêm khoảng cách tâm và độ lệch tỉ lệ khung:
  `CIoU = IoU − ρ²(b, b_gt)/c² − α·v`
  (`ρ` khoảng cách hai tâm, `c` đường chéo hộp bao nhỏ nhất, `v` đo lệch tỉ lệ rộng/cao.)
- **L_cls = BCE:** Binary Cross-Entropy cho từng lớp:
  `BCE = −[ y·log(p) + (1−y)·log(1−p) ]`
  Ví dụ: vật thật là helmet (`y=1`), mô hình đoán `p=0.9` → loss = `−log 0.9 = 0.105` (nhỏ). Đoán `p=0.1` → `−log 0.1 = 2.30` (lớn).
- **L_dfl:** Distribution Focal Loss, giúp định vị cạnh hộp chính xác hơn.

### 5.5 Transfer learning (học chuyển giao)
Không train từ số 0. Lấy trọng số đã học trên **COCO** (330 nghìn ảnh, 80 lớp, có `person`) rồi **tinh chỉnh (fine-tune)** trên dữ liệu công trường. Lợi ích: đặc trưng thấp (cạnh, kết cấu) dùng lại được, train nhanh và cần ít dữ liệu hơn.

**Freeze:** khoá (không cập nhật) một số lớp đầu. `freeze=10` nghĩa là khoá 10 khối đầu của backbone. Nhanh hơn, ít overfit hơn, nhưng thích nghi kém hơn. **Ablation freeze=10 là việc Hoài còn nợ:** train lại với freeze, so với full-train, để biết khoá có đáng không.

---

## 6. Cách huấn luyện M1 trong dự án

| Mục | Giá trị |
|---|---|
| Mô hình | YOLOv8m, khởi tạo từ COCO |
| Số lớp | 6 (person, helmet, no_helmet, vest, no_vest, machinery) |
| Epoch / batch / imgsz | 50 / 16 / 640 |
| Seed | 42 (để chạy lại ra cùng kết quả) |
| Phần cứng | Kaggle T4, 6.4 giờ |
| Dữ liệu | Roboflow Hard Hat Workers + Construction Site Safety gộp về 6 lớp |

**Nhãn `head` làm proxy cho `no_helmet`:** trong dataset gốc, "đầu trần" được gán nhãn `head`, được dùng thay lớp `no_helmet`. Nghĩa là mô hình học "thấy đầu không có mũ" như một vật.

**Baseline COCO** (mAP 0.04) chỉ có ý nghĩa cho lớp `person`, vì COCO không có helmet/vest. Đừng dùng 0.04 làm mốc so sánh cho 6 lớp.

**Seed và tái lập:** cùng seed + cùng dữ liệu + cùng cấu hình ≈ cùng kết quả. Rất quan trọng khi báo cáo khoa học.

---

## 7. Cách đo mô hình tốt hay dở

### 7.1 Ma trận nhầm lẫn
Với một lớp (ví dụ helmet), mỗi dự đoán rơi vào một trong:

| | Có vật thật | Không có vật |
|---|---|---|
| Mô hình báo có | **TP** (đúng) | **FP** (báo nhầm) |
| Mô hình không báo | **FN** (bỏ sót) | TN |

Một dự đoán bbox được tính **TP** nếu đúng lớp và **IoU với hộp thật ≥ 0.5**.

### 7.2 Precision, Recall, F1
```
Precision = TP / (TP + FP)     "trong số tôi báo, bao nhiêu đúng?"
Recall    = TP / (TP + FN)     "trong số có thật, tôi bắt được bao nhiêu?"
F1        = 2·P·R / (P + R)    trung bình điều hoà của hai cái trên
```
Ví dụ giả sử: 100 người thật không đội mũ. Mô hình báo 90 lần, trong đó 80 đúng. Vậy `TP=80, FP=10, FN=20`. `P = 80/90 = 0.89`, `R = 80/100 = 0.80`, `F1 = 0.84`.

**Số thật của M1 (val):** P = 0.906, R = 0.806. Hiểu: khi M1 báo một vật, khoảng 91% đúng; còn trong tất cả vật thật, M1 bắt được khoảng 81%.

**Đánh đổi P–R:** hạ ngưỡng conf thì R tăng, P giảm; nâng ngưỡng thì ngược lại. Với hệ thống an toàn, bỏ sót (FN) nguy hiểm, nhưng báo nhầm quá nhiều (FP) làm người dùng ngừng tin cảnh báo, nên phải cân.

### 7.3 AP và mAP
- Vẽ **đường cong Precision–Recall** khi quét ngưỡng conf từ cao xuống thấp.
- **AP (Average Precision)** = diện tích dưới đường cong đó, 0 đến 1.
- **mAP** = trung bình AP của các lớp.
- **mAP@0.5:** TP khi IoU ≥ 0.5. **mAP@0.5:0.95:** trung bình trên các ngưỡng IoU 0.5, 0.55, …, 0.95, khắt khe hơn nhiều vì bắt bbox phải khớp sát.

**Số thật M1 (val):** mAP50 = 0.867; mAP50-95 = 0.632.

| Lớp | AP@0.5 | Nhận xét |
|---|---|---|
| helmet | 0.962 | nhiều dữ liệu, vật rõ |
| no_helmet | 0.907 | |
| machinery | 0.895 | vật lớn |
| no_vest | 0.843 | ít dữ liệu |
| vest | 0.814 | ít dữ liệu, hay nhầm với áo màu nhạt |
| person | 0.779 | thấp nhất vì người hay bị che, nhỏ, chồng nhau |

### 7.4 Những thứ mAP **không** cho bạn biết
mAP đo trên **ảnh tĩnh của dataset**. Nó **không** cho biết:
- hệ thống chạy trên video thật có báo nhầm bao nhiêu lần/giờ,
- tracker có đổi ID không,
- luật có hợp ngữ cảnh không.
Vì vậy phải có phép đo **mức hệ thống**: FAR/h (xem mục 13).

---

## 8. Theo dõi (tracking)

### 8.1 Vì sao cần
M1 chỉ nhìn **từng frame riêng lẻ**, không biết "người ở frame 100" và "người ở frame 101" là cùng một người. Nhưng luật cần nhớ theo thời gian ("người này đã không đội mũ 15 frame liên tiếp"). Tracker gán **ID ổn định** cho mỗi người.

### 8.2 Tracker của dự án
File [bytetrack.py](../src/tracking/bytetrack.py): tracker **tự viết**, đơn giản:
1. Tính IoU giữa mọi track cũ và mọi hộp mới.
2. Sắp xếp theo IoU giảm dần, ghép greedy (tham lam) những cặp IoU ≥ 0.3.
3. Hộp mới không ghép được → tạo ID mới.
4. Track không ghép được tăng bộ đếm `lost`; quá 30 frame thì xoá.

Ví dụ: frame 100 có người A ở `[0.40,0.20,0.60,0.80]`, frame 101 hộp ở `[0.41,0.20,0.61,0.80]`. IoU ≈ 0.90 ≥ 0.3 → vẫn là A.

### 8.3 Hạn chế (và vì sao ID lên tới hơn 140)
Tracker này **không có**:
- **Bộ lọc Kalman** (dự đoán người sẽ ở đâu ở frame sau). Khi người bị che vài frame hoặc di chuyển nhanh, hộp mới không còn chồng với vị trí cũ → mất track.
- **Đặc trưng ngoại hình (Re-ID)** để nhận ra "cùng người" khi họ xuất hiện lại.
Hậu quả: một người sinh nhiều ID → có thể bị báo nhiều lần. Bản gốc ByteTrack còn dùng thêm các hộp conf thấp để giữ track; bản của mình thì không.

### 8.4 Giải pháp tạm đã làm: gộp sự kiện (dedup)
Khi track mới sắp báo vi phạm, kiểm tra: có track khác **vừa biến mất** (≤ 3 giây), đã từng báo **cùng loại**, và hộp chồng với hộp này (IoU ≥ 0.2) không? Nếu có thì coi là cùng người, **không báo lại**.
Kết quả thật (clip 1:36, laptop): 14 → 11 sự kiện; hai sự kiện no_vest của id 38 và id 72 bị loại vì là cùng người đã báo dưới id 42 và 57.

**Giới hạn:** đây là cách vá, không thay thế tracker tốt. Hai người đứng sát nhau và cùng vi phạm cùng loại có thể bị gộp nhầm thành một.

---

## 9. Tầng luật (RuleEngine): công thức hình học

File [safety_rules.py](../src/rules/safety_rules.py) và [postprocess.py](../src/pipeline/postprocess.py). Phần này **không phải học máy**, chỉ là hình học và ngưỡng, nhưng quyết định kết quả cuối.

Quy ước: bbox người `[x1, y1, x2, y2]` chuẩn hoá 0–1; gốc toạ độ ở **góc trên trái**, `y` tăng xuống dưới. Chiều cao `h = y2 − y1`.

### 9.1 Mũ bảo hộ (no_helmet)
Chỉ xét detection mũ/đầu mà **tâm nằm trong 1/3 trên** bbox người:
```
tâm (cx, cy) thoả:  x1 ≤ cx ≤ x2   và   y1 ≤ cy ≤ y1 + h/3
```
Logic (`ppe_flags`): bỏ detection conf < 0.5; trong các `helmet`/`no_helmet` có tâm nằm trong vùng đầu, lấy detection có **conf cao nhất**; nếu đó là `no_helmet` → vi phạm. Hoà conf thì ưu tiên lớp tuân thủ (`helmet`) để tránh báo nhầm. Kết quả không phụ thuộc thứ tự detection.

Ví dụ: người `[0.40,0.20,0.60,0.80]`, `h = 0.6`, vùng đầu là `y` từ 0.20 đến 0.40. Một `no_helmet` có tâm `(0.50, 0.25)` nằm trong vùng → gắn cho người này.

### 9.2 Áo phản quang (no_vest)
Vest nằm ở **thân**: `y1 + h/6 ≤ cy ≤ y1 + 2h/3`. Với người trên, thân là `y ∈ [0.30, 0.60]`. Một `no_vest` có tâm `y = 0.475` → hợp lệ; một cái có tâm `y = 0.715` (ở chân) → **bỏ qua**. Đây là bản sửa của Dũng ngày 11/10 (trước đó dùng nhầm vùng).

### 9.3 Vùng cấm (zone): điểm chân trong đa giác
**Bước 1: lấy điểm chân** = giữa cạnh dưới bbox:
```
foot = ( (x1 + x2)/2 , y2 )
```
Dùng chân vì người "đứng" ở mặt đất bằng chân, không phải tâm người (tâm có thể nằm trên không so với mặt đất khi ảnh nhìn chéo).

**Bước 2: điểm có nằm trong đa giác không?** Thuật toán **ray casting**: bắn một tia ngang từ điểm sang phải, đếm số lần cắt cạnh đa giác. **Lẻ → trong; chẵn → ngoài.**

Ví dụ: vùng hình chữ nhật `[[0.65,0.55],[1,0.55],[1,1],[0.65,1]]` (vùng giả định trong `safety_demo.yaml`). Chân người ở `(0.80, 0.90)`: tia sang phải cắt cạnh x=1 đúng 1 lần → lẻ → **trong vùng**. Chân ở `(0.30, 0.90)`: tia cắt 2 cạnh dọc (x=0.65 và x=1) → chẵn → ngoài.

**Bước 3 (mới): lọc theo conf.** Chỉ tính nếu `person conf ≥ zone_person_conf (0.5)`.

> Lưu ý: vùng cấm trong demo là **giả định**, chưa phải vùng thật của công trường.

### 9.4 Người gần máy móc (machinery)
Công thức cũ (phiên bản đầu, chỉ để minh hoạ): khoảng cách Euclid giữa chân người và tâm hộp máy:
```
d = √( (x_foot − x_m)² + (y_foot − y_m)² )
vi phạm nếu  d < τ_adj,   τ_adj = τ · (1 + (1 − h))
```
- (Đã thay) Luật hiện tại dùng `machinery_k` (mặc định 0.5): `τ = k · h`, đo từ chân người tới CẠNH bbox máy gần nhất, và bỏ qua máy mà bbox người nằm >= `cabin_overlap` (0.8) bên trong **và** có chân còn nằm trong bbox máy (y2 người ≤ y2 máy). Công thức `τ = 0.08` (`machinery_distance`) ở phía trên là phiên bản cũ, chỉ giữ để minh hoạ.
- Hệ số `(1 + (1 − h))`: người ở **xa** (hộp thấp, `h` nhỏ) cho `τ` lớn hơn, bù cho việc xa thì cùng khoảng cách thật nhìn nhỏ hơn trong ảnh.

Ví dụ: người cao `h = 0.5` → `τ_adj = 0.08 × 1.5 = 0.12`. Nếu chân cách tâm máy 0.10 → `0.10 < 0.12` → vi phạm.

**Hạn chế còn lại:** luật vẫn chỉ dựa trên ảnh 2D. Luật loại trừ cabin (đã có) bỏ qua người nằm trong bbox máy với chân không vượt cạnh đáy máy; người đứng ngay sau/trên máy theo phối cảnh nhưng chân vẫn nằm trong bbox máy vẫn có thể bị bỏ sót, và chưa đo lại trên clip thật.

### 9.5 Té ngã
Nhận xác suất `P(falling|fallen)` từ M2; báo khi `p > fall_conf (0.7)` trong `fall_frames (8)` frame liên tiếp. Chưa có M2 thật thì nhánh này **tắt**.

---

## 10. Xác nhận N frame, cooldown, gộp sự kiện

Vì M1 có thể chớp tắt từng frame, báo ngay lần đầu thấy sẽ rất nhiều nhầm. Ba cơ chế:

### 10.1 Xác nhận N frame liên tiếp
Mỗi `(track_id, loại)` giữ lịch sử `True/False` của N frame gần nhất; chỉ báo khi **cả N đều True**.
- PPE, vùng, máy: **N = 15** (~0.5 s ở 30 FPS).
- Té ngã: **N = 8** (~0.27 s).

Đây là bộ lọc thời gian đơn giản: giảm FP vì nhiễu một-vài-frame bị loại. Cái giá: chậm báo N frame, và bỏ lỡ vi phạm quá ngắn.

### 10.2 Cooldown
Sau khi báo `(ID, loại)`, không báo lại trong 60 giây:
```
cooldown_frames = cooldown_sec × FPS_video     (60 × 30 = 1800 frame)
```
Bản sửa 11/10: trước đây số frame cố định, giờ **tính theo FPS thật của video** (video 25 FPS thì 1500 frame).

### 10.3 Gộp nhiều loại trong cùng frame
Nếu một người cùng lúc vi phạm `no_helmet` và `no_vest`, gộp thành một sự kiện `no_helmet+no_vest` (hàm `merge_alerts`), không tạo hai dòng.

### 10.4 Tổng hợp thành bộ lọc nhiều tầng
```
M1 conf ≥ 0.25 → ppe_conf/zone_person_conf ≥ 0.5 → luật hình học
→ N=15 liên tiếp → cooldown 60 s → dedup khi đổi ID → ghi sự kiện
```
Mỗi tầng loại một kiểu lỗi.

---

## 11. M2: phát hiện té ngã (Năng, học máy nâng cao)

> **Lưu ý trung thực:** M2 hiện huấn luyện trên dữ liệu **synthetic** (giả lập), chưa gắn vào pipeline. F1 = 1.0 trên đó **không được dùng làm kết quả**.

### 11.1 Ý tưởng
Té ngã là một **chuỗi chuyển động theo thời gian**, không thấy được từ một ảnh tĩnh (người nằm có thể chỉ là đang làm việc). Nên:
1. Lấy **khung xương** (pose, các khớp) của từng người từng frame bằng một mô hình pose.
2. Từ khung xương tính **vector đặc trưng 37 chiều** mỗi frame.
3. Gom **cửa sổ 30 frame** (~1 giây) → đưa vào mạng chuỗi thời gian.
4. Mạng ra 3 lớp: `normal / falling / fallen`.

### 11.2 Vì sao dùng khung xương thay vì ảnh thô
Khung xương loại bỏ màu áo, nền, ánh sáng, nên mạng học **chuyển động** thay vì ngoại hình, và nhỏ hơn rất nhiều. Cái giá: phụ thuộc chất lượng pose; người bị che hay quá nhỏ thì khớp sai.

### 11.3 TCN (Temporal Convolutional Network)
Dùng **tích chập 1 chiều theo thời gian** thay vì RNN/LSTM.
- **Nhân quả (causal):** đầu ra tại thời điểm `t` chỉ phụ thuộc `≤ t`, không "nhìn trộm" tương lai.
- **Giãn (dilated):** bộ lọc nhảy cách, để vùng nhìn tăng theo cấp số nhân mà không cần quá nhiều lớp.

**Receptive field (vùng nhìn theo thời gian)** với kernel `k` và các hệ số giãn `d₁, d₂, …`:
```
RF = 1 + (k − 1) · Σ dᵢ
```
Ví dụ giả sử `k = 3`, `d = 1, 2, 4, 8, 16`: `RF = 1 + 2 × 31 = 63` frame, đủ phủ cửa sổ 30. (Cấu hình thật của Năng ghi RF = 61; xem code của Năng.)

**So với LSTM:** TCN train song song được (nhanh hơn), gradient ổn định hơn; LSTM có bộ nhớ trạng thái nhưng tuần tự.

### 11.4 Loss và chỉ số của M2
- Loss: **Cross-Entropy** đa lớp `L = −Σ yᵢ log pᵢ`.
- Đo bằng P/R/F1 từng lớp. Lưu ý với dữ liệu té ngã: lớp `falling` rất hiếm → **mất cân bằng lớp** (xem 2.8). Cần nhìn **F1 từng lớp**, không chỉ accuracy.
- **Domain shift:** train trên video giả lập/phòng thí nghiệm (UR Fall, Le2i) rồi chạy trên công trường thật thì hiệu năng tụt. Cần đo, đây là việc còn thiếu.

### 11.5 Tại sao N = 8 và conf = 0.7
Té ngã diễn ra nhanh, nên N nhỏ hơn PPE (8 so với 15) để không bỏ lỡ; conf 0.7 cao để giảm báo nhầm.

---

## 12. Hệ thống: SQLite, FastAPI, Telegram, FPS

### 12.1 Ghi sự kiện (SQLite)
Bảng `events(id, track_id, violation_type, timestamp, frame_idx, confidence, image_path)`. SQLite là cơ sở dữ liệu một file, không cần cài server, hợp cho demo. `confidence` là conf YOLO của **lớp vi phạm** (ví dụ conf của `no_vest`), không phải conf của person.

### 12.2 Ảnh bằng chứng
`save_evidence` vẽ lên frame: **box đỏ dày** = người vi phạm, box mỏng + nhãn = mọi detection M1, dòng chữ `loại id=X f=N`. Tên file `loại_trackid_frame.jpg`.

### 12.3 FastAPI
REST API chỉ đọc: `/health`, `/events`, `/events/count`, `/evidence/{file}`, `/docs`. FastAPI tự sinh trang `/docs` (Swagger) để thử.

### 12.4 Telegram
Hiện ở **chế độ mock** (in ra console, không gửi thật), để không spam khi thử. Truyền `--telegram-token`/`--telegram-chat` để gửi thật.

### 12.5 FPS và thời gian xử lý
```
FPS = số frame xử lý / tổng giây
thời gian/frame = decode + resize + suy luận + hậu xử lý
```
Số thật (clip 1:36, 2871 frame; M1 + tracker + luật, **chưa có M2**):

| Máy | imgsz | FPS | Suy luận | Decode+resize |
|---|---|---|---|---|
| Laptop Quadro T1000 | 640 | 9.4 | 84.7 ms | 15.9 ms |
| Kaggle T4 | 640 | 32.0 | 25.2 ms | 3.0 ms |
| Kaggle T4 | 512 | 49.1 | 16.5 ms | 2.7 ms |

Kiểm tra: `1/32 s = 31 ms`, khớp `25.2 + 3.0 + hậu xử lý`.
**Mục tiêu ≥ 15 FPS** đạt trên T4; trên laptop thì dao động (9.4 ở lần đo đầu, 12.9–15.9 ở các lần chạy sau tuỳ máy bận). Luôn ghi kèm máy đo. Khi gắn M2 FPS sẽ giảm.

**Vì sao T4 nhanh hơn?** GPU mạnh hơn, và CPU Kaggle decode video nhanh hơn laptop. Clip 4K còn chậm hơn vì decode 4K tốn ~30 ms/frame.

---

## 13. Đánh giá hệ thống thật: vì sao "xem bằng mắt 1 clip" chưa phải độ chính xác

Những gì dự án đã làm: chạy clip, mở từng ảnh sự kiện, tự đánh dấu đúng/sai. Ví dụ clip 1:36 (laptop): 14 sự kiện = 11 đúng, 2 sai, 1 không chắc.

**Vì sao chưa đủ để nói "độ chính xác 79%":**
1. **Chỉ một clip**, một cảnh, một góc quay → không đại diện.
2. **Không biết FN:** ta chỉ xem những cái hệ thống đã báo. Vi phạm bị bỏ sót (không có sự kiện) thì không thấy.
3. **Một người bị đếm nhiều lần** khi tracker đổi ID.
4. **Người đánh giá là người làm hệ thống** → thiên vị.

**Đo đúng cách** cần:
- Gán nhãn thật (ground truth) cho video: "từ giây 12–20 người X không đội mũ".
- Tính TP/FP/FN ở **mức sự kiện**, rồi P/R.
- **FAR/h (False Alarm Rate per hour)** = số cảnh báo sai / số giờ video. Chỉ số quan trọng nhất cho người vận hành, vì họ chịu đựng báo nhầm theo từng giờ.
```
FAR/h = FP / (tổng giờ video)
```
Ví dụ giả sử: video 2 giờ có 6 báo nhầm → FAR = 3/h. Đây là việc **Tuần 8** còn nợ.

---

## 14. Lỗi đã gặp và vì sao xảy ra

| Lỗi | Nguyên nhân | Cách xử lý / hướng |
|---|---|---|
| Móc cẩu bị báo `zone` | M1 nhận nhầm móc cẩu là `person` conf 0.27–0.32; không có ngưỡng riêng | Thêm `zone_person_conf = 0.5` (đã làm) |
| Một người, nhiều sự kiện | Tracker IoU greedy không Kalman, đổi ID | Gộp sự kiện (đã làm); lâu dài đổi tracker |
| Vest phản quang nhạt bị báo `no_vest` | Ít dữ liệu vest (3–4%), màu nhạt không giống mẫu train | Bổ sung dữ liệu; đưa vào "lớp khó" cho Hoài |
| Người ngồi trong cabin bị báo `machinery`/`no_helmet` | Luật chỉ dùng khoảng cách/vị trí 2D, không hiểu ngữ cảnh | Thêm luật loại trừ người nằm trong bbox máy (chưa làm) |
| imgsz 512 nhiều báo nhầm hơn | Vật nhỏ mất chi tiết khi giảm độ phân giải | Giữ 640 |
| Chạy Kaggle lỗi `FileNotFoundError best.pt` | Sai đường dẫn dataset | Dùng đường dẫn thật `/kaggle/input/datasets/dung68/safety-m1-assets/` |

**Bài học chung:** lỗi ở tầng nào thì sửa ở tầng đó. Lỗi nhận dạng → dữ liệu/mô hình; lỗi báo trùng → tracker/luật; lỗi ngữ cảnh → luật.

---

## 15. Bảng thuật ngữ nhanh

| Thuật ngữ | Nghĩa ngắn |
|---|---|
| bbox | hộp khoanh quanh vật `[x1,y1,x2,y2]` |
| conf | độ tin cậy 0–1 mô hình gán cho một dự đoán |
| IoU | độ chồng nhau của hai hộp |
| NMS | loại bớt hộp trùng |
| epoch | một lượt qua toàn bộ tập train |
| batch | số mẫu cho mỗi bước cập nhật |
| learning rate | độ lớn bước gradient descent |
| loss | số đo sai của mô hình |
| overfitting | học thuộc train, kém trên dữ liệu mới |
| augmentation | biến đổi ảnh để tăng dữ liệu |
| transfer learning | dùng trọng số đã học sẵn rồi tinh chỉnh |
| freeze | khoá một số lớp, không cho cập nhật |
| ablation | bỏ/đổi một thành phần để xem nó đóng góp bao nhiêu |
| TP / FP / FN | đúng / báo nhầm / bỏ sót |
| mAP | trung bình AP các lớp |
| tracking | gán ID ổn định theo thời gian |
| Kalman | bộ lọc dự đoán vị trí kế tiếp |
| TCN | mạng tích chập theo thời gian |
| receptive field | độ dài/vùng mà một đầu ra "nhìn thấy" |
| FAR/h | số báo nhầm mỗi giờ |
| domain shift | dữ liệu thật khác dữ liệu train |
| ground truth | đáp án thật do người gán |
| data leakage | rò rỉ thông tin giữa train và val/test |

---

## 16. Câu hỏi ôn tập (tự trả lời, có đáp án gợi ý)

1. **Vì sao tách M1 (học máy) và luật (code), không để mạng tự quyết định vi phạm?**
   Dễ giải thích và sửa ngưỡng không cần train lại; luật hình học không cần dữ liệu.
2. **mAP50 = 0.867 có nghĩa hệ thống báo đúng 86.7% không?**
   Không. Đó là chất lượng phát hiện trên ảnh tĩnh ở val, không phải độ chính xác cảnh báo trên video.
3. **Vì sao val tốt nhưng video thật vẫn báo nhầm?**
   Domain shift (góc quay, ánh sáng, người nhỏ), tracker đổi ID, luật không hiểu ngữ cảnh, và lỗi hiếm lớp (vest phản quang).
4. **Hạ `ppe_conf` từ 0.5 xuống 0.3 thì chuyện gì xảy ra?**
   Recall tăng (bắt thêm), Precision giảm (báo nhầm thêm).
5. **IoU hai hộp 10×10, dịch ngang 2 là bao nhiêu?**
   Giao = 8×10 = 80, hợp = 120, IoU = 0.67.
6. **Vì sao dùng điểm chân cho vùng cấm?**
   Chân chạm đất, phản ánh vị trí đứng thật; tâm hộp có thể ở trên không.
7. **Tracker không Kalman gây hại gì?**
   Mất track khi bị che/di chuyển nhanh → đổi ID → báo trùng.
8. **Vì sao M2 dùng chuỗi 30 frame thay vì một ảnh?**
   Ngã là chuyển động theo thời gian; ảnh tĩnh người nằm có thể chỉ là đang làm việc.
9. **F1 = 1.0 trên dữ liệu synthetic có đáng tin không?**
   Không; dữ liệu giả quá dễ/trùng phân phối, chưa chứng minh gì cho công trường thật.
10. **Muốn có số "độ chính xác hệ thống" đúng nghĩa cần làm gì?**
    Gán ground truth cho nhiều video, tính TP/FP/FN mức sự kiện và FAR/h.

---

## Phụ lục: ánh xạ khái niệm → file trong repo

| Khái niệm | File |
|---|---|
| Luật hình học, ray casting, ppe_flags, cooldown | [src/rules/safety_rules.py](../src/rules/safety_rules.py) |
| Lọc conf vùng cấm, dedup, RuleEngine | [src/pipeline/postprocess.py](../src/pipeline/postprocess.py) |
| Tracker, IoU | [src/tracking/bytetrack.py](../src/tracking/bytetrack.py) |
| Pipeline chạy video, ảnh bằng chứng | [src/pipeline/stream.py](../src/pipeline/stream.py) |
| Ngưỡng, vùng cấm | [configs/safety.yaml](../configs/safety.yaml) |
| Test luật | [tests/test_rules.py](../tests/test_rules.py) |
| Chạy Kaggle | [reports/huong_dan_kaggle.md](huong_dan_kaggle.md) |
