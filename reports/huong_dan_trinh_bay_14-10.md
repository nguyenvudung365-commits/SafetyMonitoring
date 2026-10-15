# Hướng dẫn trình bày — buổi 14/10 (bản nói chi tiết, dễ hiểu)

Gốc thư mục: `C:\Users\Dung\Downloads\Học máy nâng cao\` (viết tắt `[HMNC]`).
Thứ tự: **Duy → Hoài → Dũng**. Năng báo cáo riêng.
Các đoạn trong khung `>` là câu nói mẫu, bạn đọc hoặc nói lại bằng lời của mình.

**Giải thích nhanh vài từ, để bạn nói cho dễ:**
- *FPS* = số hình máy xử lý được mỗi giây. Càng cao càng mượt. Cần từ 15 trở lên.
- *mAP* = điểm chính xác của mô hình, từ 0 đến 1, càng gần 1 càng tốt.
- *Tập val / test* = ảnh dùng để chấm điểm mô hình, mô hình chưa từng học các ảnh này.
- *Sự kiện* = một lần hệ thống báo có người vi phạm.
- *Báo nhầm* = hệ thống báo vi phạm nhưng thực tế người đó không vi phạm.

---

## Chuẩn bị trước khi lên (mở sẵn các thứ sau)

| # | Mở gì | Ở đâu |
|---|---|---|
| 1 | Công cụ gán nhãn của Duy | `[HMNC]\label_tool.html` (mở bằng trình duyệt) |
| 2 | Biểu đồ huấn luyện của Hoài | `[HMNC]\Nhom9_Tuan01_Skeleton\results_m1\runs\detect\train-2\results.png` |
| 3 | Bảng nhầm lẫn giữa các lớp | cùng thư mục trên, file `confusion_matrix_normalized.png` |
| 4 | Notebook chạy trên Kaggle | `[HMNC]\kaggle\m1-result.ipynb` (kéo xuống dòng `Done 2871 frames | FPS=32.0`) |
| 5 | Ảnh hệ thống phát hiện (đúng) | `[HMNC]\kaggle\results\logs_640\evidence\` (mở 3–4 ảnh, ví dụ `no_vest_30_000595.jpg`) |
| 6 | Ảnh báo nhầm (vest) | `[HMNC]\kaggle\results\logs_640\evidence\no_vest_34_000752.jpg` |
| 7 | Ảnh báo nhầm (móc cẩu) | `[HMNC]\kaggle\results\logs_512\evidence\zone_51_001442.jpg` |
| 8 | Báo cáo đầy đủ | `[HMNC]\Nhom9_Tuan01_Skeleton\reports\bao_cao_tien_do_14-10.md` |
| 9 | Báo cáo lần trước (để so) | `[HMNC]\Nhom9_Tuan01_Skeleton\reports\bao_cao_tien_do_23-09.md` |

---

## 1. Duy — Dữ liệu và công cụ gán nhãn (khoảng 2 phút)

**Mở:** `label_tool.html`.

**Nói:**

> Em xin báo cáo phần dữ liệu. Nhóm mình có khoảng 19.600 ảnh công trường. Mỗi ảnh được đánh dấu 6 loại: người, mũ bảo hộ, người không đội mũ, áo phản quang, người không mặc áo, và máy móc.

> Em chia ảnh thành 3 phần: khoảng 13.700 ảnh để dạy máy học, 3.900 ảnh để chấm điểm trong lúc học, và 2.000 ảnh để thi cuối. Em chia theo từng nhóm ảnh gốc, để những ảnh giống nhau không bị rơi vào cả phần học lẫn phần thi, vì như vậy điểm sẽ cao giả.

> Tuần này em làm thêm một công cụ để gán nhãn nhanh, chỉ cần mở file này trên trình duyệt. *(mở công cụ cho thầy xem)* Công cụ dùng đúng 6 loại như dữ liệu cũ, nên có thể ghép dữ liệu mới vào. Có thể cho máy tự gán nhãn trước, rồi mình sửa lại bằng tay cho nhanh.

> Có một điểm em cần nói thẳng: công cụ này đang chia ảnh theo từng ảnh riêng lẻ. Nếu nhiều ảnh lấy từ cùng một video thì ảnh giống nhau có thể rơi vào cả phần học lẫn phần thi. Em sẽ sửa lại cách chia trước khi dùng dữ liệu mới.

> Còn một khó khăn nữa là áo phản quang xuất hiện ít trong dữ liệu (khoảng 3–4%), nên mô hình nhận áo kém hơn so với mũ.

**Nếu thầy hỏi "sao có nhãn `head` mà không phải không đội mũ?"**

> Dữ liệu gốc có nhãn "đầu người", em dùng nhãn này thay cho "người không đội mũ" vì không có nhãn nào sát hơn.

---

## 2. Hoài — Mô hình nhận diện (khoảng 2–3 phút)

**Mở:** `results.png`, rồi `confusion_matrix_normalized.png`.

**Nói:**

> Em xin báo cáo phần mô hình. Em dạy mô hình trên Kaggle, học 50 vòng, mất khoảng 6 tiếng rưỡi. Em đã học xong đủ 50 vòng.

> Kết quả chấm trên phần ảnh chấm điểm: điểm chính xác tổng thể là 0,867 trên thang 1. Nghĩa là mô hình nhận đúng phần lớn. Tỉ lệ báo đúng là 0,906 và tỉ lệ tìm ra được là 0,806.

> Theo từng loại: mũ bảo hộ cao nhất, 0,96. Người không đội mũ 0,91. Máy móc 0,90. Người không mặc áo 0,84. Áo phản quang 0,81. Thấp nhất là nhận ra người, 0,78.

> Em không đưa điểm của mô hình gốc ra so sánh, vì mô hình gốc chỉ biết nhận ra người, không biết mũ hay áo, nên số đó không công bằng.

**Phần Hoài phải nói thẳng còn thiếu** *(Hoài tự xác nhận trước buổi họp; nếu đã làm xong thì sửa lại)*:

> Em chưa chấm mô hình trên tập thi cuối, chưa so sánh cách dạy cả mô hình với cách chỉ dạy một phần, và chưa có bảng so sánh với mô hình gốc. Em sẽ làm trong tuần này.

**Nếu thầy hỏi "dạy lại tốt hơn mô hình gốc bao nhiêu?"**

> Dạ chưa có bảng so sánh đầy đủ nên em chưa dám nói con số. Em sẽ làm xong và gửi thầy.

---

## 3. Dũng — Hệ thống chạy thật (khoảng 3–4 phút)

**Mở:** notebook Kaggle, rồi vài ảnh trong `logs_640\evidence\`, rồi 2 ảnh báo nhầm.

**Phần 1 — Hệ thống làm gì (30 giây):**

> Em xin báo cáo phần hệ thống. Hệ thống nhận video công trường, mô hình tìm người, mũ, áo và máy móc. Sau đó hệ thống theo dõi từng người, và nếu một người thiếu mũ hoặc thiếu áo liên tục trong khoảng 15 hình thì mới báo, để tránh báo nhầm do nhiễu một hai hình. Khi báo, hệ thống lưu lại một ảnh có khoanh đỏ vào người vi phạm.

**Phần 2 — Cái mới so với lần 23/09 (1 phút):**

> So với lần trước, có 3 điểm mới. Thứ nhất, lần trước em chỉ thử một đoạn video 8 giây, lần này em thử một video công trường dài 1 phút 36 giây, nhiều người đi lại, khoảng 2.900 hình.

> Thứ hai, em sửa một số lỗi trong chương trình: thời gian chờ giữa hai lần báo cùng một người giờ tính đúng theo tốc độ của video, vùng kiểm tra áo giờ nhìn đúng phần thân người, và nếu một người vi phạm nhiều lỗi cùng lúc thì chỉ báo một lần. Số bài kiểm tra tự động của chương trình tăng từ 15 lên 25 và đều đạt.

> Thứ ba, em đã đưa chương trình lên Kaggle để chạy trên máy mạnh hơn.

**Phần 3 — Tốc độ (1 phút):**

> Về tốc độ: trên laptop của em chỉ xử lý được khoảng 9 hình mỗi giây, chưa đủ mục tiêu là 15. Nhưng trên Kaggle thì được 32 hình mỗi giây, gấp đôi mục tiêu. *(mở notebook cho thầy xem dòng FPS=32.0)*

> Em cũng thử thu nhỏ ảnh lại để chạy nhanh hơn, được khoảng 49 hình mỗi giây. Nhưng khi thu nhỏ thì hệ thống báo thêm nhiều lần hơn và có thêm báo nhầm, và em chưa đo được độ chính xác ở cỡ ảnh nhỏ. Vì vậy em vẫn giữ cỡ ảnh cũ.

> Em xin lưu ý: tốc độ 32 hình mỗi giây này mới chỉ tính phần nhận diện mũ và áo. Khi gắn thêm phần nhận diện người ngã của bạn Năng, tốc độ sẽ giảm xuống.

**Phần 4 — Kết quả thực tế và lỗi (1 phút):**

> Em đã mở từng ảnh hệ thống báo để kiểm tra bằng mắt. Với bản chạy trên laptop có 14 lần báo: 11 lần đúng, 2 lần sai, 1 lần em không chắc. *(mở 1–2 ảnh báo đúng)*

> Có hai lỗi em đã thấy. Lỗi thứ nhất: người mặc áo phản quang màu vàng nhạt bị báo là không mặc áo. *(mở `no_vest_34_000752.jpg`)* Lỗi này lặp lại ở cả hai lần chạy, nên không phải ngẫu nhiên. Em sẽ báo cho bạn Hoài để đưa vào phần phân tích.

> Lỗi thứ hai: cái móc cẩu bị nhận nhầm là người, nên hệ thống báo có người đi vào vùng cấm. *(mở `zone_51_001442.jpg`)* Cách sửa là chỉ tin vào "người" khi mô hình chắc chắn hơn. Em sẽ sửa và chạy lại.

**Phần 5 — Những điều phải nói thẳng (30 giây):**

> Em xin nói rõ những gì mình chưa làm được. Một: đây mới là kiểm tra bằng mắt trên một video, chưa phải con số độ chính xác chính thức. Hai: một người có thể bị hệ thống đếm thành nhiều người khi họ bị che khuất rồi xuất hiện lại, nên số lần báo có thể nhiều hơn số người vi phạm thật. Ba: video này không có người thiếu mũ và không có máy móc, nên hai luật đó em chưa kiểm tra được trên video mới. Bốn: vùng cấm trong lần thử này là vùng em tự đặt, chưa phải vùng thật của công trường.

**Phần 6 — Việc tiếp theo (30 giây):**

> Em sẽ sửa lỗi móc cẩu, gộp các lần báo của cùng một người, thử thêm một video có máy xúc và người thiếu mũ, rồi đo số lần báo nhầm trên mỗi giờ. Khi bạn Năng có mô hình thật thì em sẽ gắn vào.

### Nếu thầy hỏi

| Thầy hỏi | Bạn trả lời |
|---|---|
| "Độ chính xác của hệ thống bao nhiêu?" | Em chưa có con số chính thức. Mới kiểm bằng mắt trên một video, 11 trên 14 lần báo là đúng. Em sẽ đo kỹ ở tuần 8. |
| "Sao laptop chậm vậy?" | Laptop em có card đồ họa yếu, chỉ 4 GB. Máy dự định chạy thật mạnh hơn, giống máy trên Kaggle. |
| "32 hình mỗi giây có đủ không?" | Đủ nếu chỉ có phần mũ và áo. Khi thêm phần té ngã sẽ giảm. Nếu dưới 15 thì em thử mô hình nhỏ hơn hoặc cho phần té ngã chạy thưa hơn. |
| "Sao lần này khác lần 23/09?" | Video khác nhau: lần trước là video 4K dài 8 giây, lần này video dài 1 phút 36 giây. Nên không so trực tiếp tốc độ giữa hai lần. |
| "Sao có 15 lần báo mà chỉ vài người?" | Có thể một người bị đếm nhiều lần do hệ thống theo dõi mất dấu rồi gán số mới. Em sẽ sửa phần này. |
| "Sao không dùng cỡ ảnh nhỏ cho nhanh?" | Vì chạy thử thấy báo nhầm tăng và em chưa đo độ chính xác ở cỡ đó. Cỡ cũ đã đủ nhanh. |

---

## 4. Năng — báo cáo riêng

Bảo Năng tự báo cáo phần nhận diện té ngã. Nhắc bạn ấy nói thẳng: dữ liệu hiện tại là dữ liệu giả, **không nói điểm 1,0**, và dữ liệu thật (UR Fall, Le2i) chưa đưa vào học.

---

## Tin nhắn gửi Hoài (copy dùng)

> Hoài ơi, trước buổi báo cáo bạn bổ sung giúp mình: (1) chấm mô hình trên tập test, (2) so sánh cách dạy cả mô hình với cách đóng băng một phần (freeze=10) cùng cấu hình, (3) bảng 3 hàng gồm mô hình gốc / dạy cả mô hình / đóng băng một phần, có điểm từng loại, (4) nhận xét loại khó: áo phản quang màu nhạt hay bị báo nhầm là không mặc áo, và nhận ra người đang thấp nhất (0,78). Nếu chưa kịp thì cứ nói thẳng là đang làm, đừng điền số ước lượng nhé.

## Tin nhắn gửi Năng (copy dùng)

> Năng ơi, bạn tự báo cáo phần nhận diện té ngã riêng nhé. Nhớ nói thẳng là hiện đang dùng dữ liệu giả, không báo điểm 1,0, và dữ liệu thật (UR Fall / Le2i) chưa đưa vào học. Nếu có mô hình rồi thì gửi mình file `best_tcn.pt` để mình gắn vào hệ thống.

---

## Ba điều nhớ khi đứng nói

1. **Nói thẳng điều chưa làm được.** Thầy thường quý người biết mình thiếu gì hơn là người nói quá.
2. **Không đưa con số mình chưa có.** Chưa có thì nói "em sẽ đo và gửi thầy".
3. **Số nào cũng nói rõ đo ở đâu.** Tốc độ 32 hình mỗi giây là trên Kaggle, không phải laptop.
