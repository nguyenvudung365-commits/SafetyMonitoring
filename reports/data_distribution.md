# Phân bố dữ liệu M1

## Số ảnh mỗi split

| Split | Số ảnh | Tổng bbox |
|---|---|---|
| train | 13725 | 65429 |
| val | 3887 | 18407 |
| test | 1991 | 9901 |

## Số bbox theo từng lớp

| Lớp | train | val | test | Tổng |
|---|---|---|---|---|
| 0 (person) | 7897 | 2184 | 1185 | 11266 |
| 1 (helmet) | 35239 | 10030 | 5446 | 50715 |
| 2 (no_helmet) | 12645 | 3527 | 1778 | 17950 |
| 3 (vest) | 2220 | 714 | 320 | 3254 |
| 4 (no_vest) | 2817 | 690 | 408 | 3915 |
| 5 (machinery) | 4611 | 1262 | 764 | 6637 |

## Ghi chú lớp hiếm

- Lớp **3 (vest)** chỉ chiếm 3.47% tổng số bbox (3254/93737) — cần cân nhắc tăng cường dữ liệu hoặc oversampling.
- Lớp **4 (no_vest)** chỉ chiếm 4.18% tổng số bbox (3915/93737) — cần cân nhắc tăng cường dữ liệu hoặc oversampling.