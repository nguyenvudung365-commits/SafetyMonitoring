# Báo cáo phân bố dữ liệu — Tuần 1

Nguồn dữ liệu: Hard Hat Workers (Joseph Nelson, v11 augmented3x-AllClasses-FastModel, 16,867 ảnh) + Construction Site Safety (Roboflow Universe Projects, v25 v1-ACCURATE-Machinery-aug5xMosaic, 2,767 ảnh).

Sau hợp nhất nhãn về schema 6 lớp, loại trùng theo hash, loại các dòng nhãn không thuộc schema (Mask, NO-Mask, Safety Cone của bộ site — 7,913 dòng bị loại), chia theo nhóm ảnh gốc (7,683 nhóm, tránh rò rỉ do các bản augment cùng gốc): tổng **19,603 ảnh / 93,737 bbox**, chia 70/20/10 theo nhóm.

## M1 — 6 lớp
| Lớp | Train | Val | Test | Tổng |
|-----|-------|-----|------|------|
| person | 7897 | 2184 | 1185 | 11266 |
| helmet | 35239 | 10030 | 5446 | 50715 |
| no_helmet | 12645 | 3527 | 1778 | 17950 |
| vest | 2220 | 714 | 320 | 3254 |
| no_vest | 2817 | 690 | 408 | 3915 |
| machinery | 4611 | 1262 | 764 | 6637 |

Số ảnh mỗi split: train 13,725 / val 3,887 / test 1,991 (65,429 / 18,407 / 9,901 bbox tương ứng). Không phát hiện lỗi class ID hay toạ độ bbox (kiểm tra bằng `tools/check_yolo_labels.py`).

## Nhận xét

- **Lớp hiếm:** `vest` (3.47% tổng bbox, 3,254/93,737) và `no_vest` (4.18%, 3,915/93,737) — thấp hơn hẳn các lớp còn lại (helmet chiếm tới 54.1%). Nguyên nhân: chỉ bộ Construction Site Safety (2,767/19,603 ảnh, ~14%) có nhãn vest/no_vest, còn bộ Hard Hat Workers (86% số ảnh) chỉ có head/helmet/person.
- **Hướng xử lý:** đề xuất dùng Focal Loss hoặc class-weighted loss khi Hoài train M1 để bù mất cân bằng; có thể cân nhắc oversampling ảnh chứa vest/no_vest, hoặc thu thập thêm ảnh có vest riêng nếu AP lớp này thấp sau khi train thử (theo kế hoạch Tuần 3-4, phối hợp Hoài xử lý AP thấp).
- **Augmentation:** cả 2 bộ đều dùng augmentation có sẵn từ Roboflow (flip ngang, xoay ±10-12°, sáng/tương phản, crop; bộ site có thêm shear, Gaussian blur nhẹ). Không có flip dọc — đúng nguyên tắc đề tài. Tuy nhiên **chưa có nhiễu mưa/sương/bụi** như kế hoạch Tuần 2 yêu cầu riêng cho đặc thù công trường — cần bổ sung bằng script riêng (việc tiếp theo).

## M2 — chuỗi keypoint
- Chưa thực hiện — thuộc phạm vi Tuần 5-6 (phối hợp Năng, video UR Fall/Le2i).

