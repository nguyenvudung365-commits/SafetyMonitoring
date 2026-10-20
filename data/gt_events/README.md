# Ground-truth sự kiện (gán tay)

Mỗi clip một file CSV `data/gt_events/<clip>.csv`, dùng với `tools/eval_events.py` và `tools/sweep_rules.py`.
File mẫu chỉ có dòng tiêu đề: `_template.csv`. **Không có nhãn thật nào được tạo sẵn.**

Cột:

| Cột | Ý nghĩa |
|---|---|
| `clip` | tên clip (khớp `--clip`) |
| `vtype` | `no_helmet`, `no_vest`, `zone`, `machinery`, `fall` |
| `start_frame`, `end_frame` | khoảng frame (theo chỉ số frame của `stream.py`, bắt đầu từ 0) mà vi phạm có thật |
| `x`, `y` | vị trí xấp xỉ của người (chuẩn hoá 0-1), có thể để trống; hiện chưa dùng khi khớp |

Quy ước khớp: event hệ thống đúng nếu `frame_idx` nằm trong `[start - tol, end + tol]` (tol mặc định 2 s)
và cùng `vtype`; mỗi dòng GT chỉ được khớp một lần (event thứ hai trên cùng GT tính là báo giả).
Khớp tham lam: event duyệt theo frame tăng dần, mỗi event gắn vào GT chưa khớp có `start_frame` nhỏ nhất thoả điều kiện (GT được sắp theo `start_frame`, không phụ thuộc thứ tự dòng trong CSV).
Dòng bắt đầu bằng `#` bị bỏ qua. Ghi ai gán nhãn và ngày gán trong báo cáo.
Nên tách clip dùng để chỉnh tham số và clip đánh giá cuối.
