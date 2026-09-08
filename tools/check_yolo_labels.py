"""
Kiểm tra thư mục nhãn YOLO: đúng định dạng, class ID trong khoảng [0, num_classes-1],
tọa độ bbox hợp lệ (0-1). Dùng để nghiệm thu sau khi chạy prepare_m1.py.
"""
import argparse
from pathlib import Path
from collections import Counter

CLASS_NAMES = ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery"]

def parse_args():
    ap = argparse.ArgumentParser(description="Kiểm tra nhãn YOLO đúng schema N lớp")
    ap.add_argument("--labels", required=True, help="Thư mục chứa file .txt nhãn YOLO")
    ap.add_argument("--num-classes", type=int, default=6)
    return ap.parse_args()

def main():
    args = parse_args()
    labels_dir = Path(args.labels)
    if not labels_dir.exists():
        raise SystemExit(f"Lỗi: không tìm thấy thư mục {labels_dir}")

    txt_files = sorted(labels_dir.glob("*.txt"))
    print(f"Tổng số file nhãn: {len(txt_files)}")

    class_counter = Counter()
    error_files = []       # (file, lý do)
    total_boxes = 0
    empty_files = 0

    for f in txt_files:
        lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        if not lines:
            empty_files += 1
            continue
        for ln_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                error_files.append((f.name, f"dòng {ln_no}: cần đúng 5 giá trị (class cx cy w h), có {len(parts)}"))
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                error_files.append((f.name, f"dòng {ln_no}: giá trị không phải số hợp lệ: {line}"))
                continue

            if cid < 0 or cid >= args.num_classes:
                error_files.append((f.name, f"dòng {ln_no}: class ID {cid} nằm ngoài [0, {args.num_classes-1}]"))
                continue

            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                error_files.append((f.name, f"dòng {ln_no}: toạ độ bbox ngoài khoảng (0,1]: {line}"))
                continue

            class_counter[cid] += 1
            total_boxes += 1

    print(f"File nhãn rỗng (không có bbox): {empty_files}")
    print(f"Tổng số bbox hợp lệ: {total_boxes}")
    print("\nPhân bố theo lớp:")
    for cid in range(args.num_classes):
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
        print(f"  {cid} ({name}): {class_counter.get(cid, 0)} bbox")

    if error_files:
        print(f"\n⚠️  CẢNH BÁO: {len(error_files)} lỗi phát hiện được (hiển thị tối đa 20 dòng đầu):")
        for fname, reason in error_files[:20]:
            print(f"  - {fname}: {reason}")
        if len(error_files) > 20:
            print(f"  ... và {len(error_files)-20} lỗi khác")
        raise SystemExit(1)
    else:
        print("\n✅ Không phát hiện lỗi. Tất cả class ID và toạ độ đều hợp lệ.")

if __name__ == "__main__":
    main()
