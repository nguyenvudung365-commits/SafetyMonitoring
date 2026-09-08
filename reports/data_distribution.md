# Phân bố dữ liệu M1

## Số ảnh mỗi split

| Split | Số ảnh | Tổng bbox |
|---|---|---|
| train | 27450 | 130858 |
| val | 3887 | 18407 |
| test | 1991 | 9901 |

## Số bbox theo từng lớp

| Lớp | train | val | test | Tổng |
|---|---|---|---|---|
| 0 (person) | 15794 | 2184 | 1185 | 19163 |
| 1 (helmet) | 70478 | 10030 | 5446 | 85954 |
| 2 (no_helmet) | 25290 | 3527 | 1778 | 30595 |
| 3 (vest) | 4440 | 714 | 320 | 5474 |
| 4 (no_vest) | 5634 | 690 | 408 | 6732 |
| 5 (machinery) | 9222 | 1262 | 764 | 11248 |

## Ghi chú lớp hiếm

- Lớp **3 (vest)** chỉ chiếm 3.44% tổng số bbox (5474/159166) — cần cân nhắc tăng cường dữ liệu hoặc oversampling.
- Lớp **4 (no_vest)** chỉ chiếm 4.23% tổng số bbox (6732/159166) — cần cân nhắc tăng cường dữ liệu hoặc oversampling.