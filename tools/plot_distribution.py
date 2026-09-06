"""
Thống kê phân bố lớp trên data/m1 (train/val/test): số ảnh, số bbox mỗi lớp.
Xuất: reports/data_distribution.png (histogram) + reports/data_distribution.md (bảng số liệu).
"""
import argparse
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery"]
SPLITS = ["train", "val", "test"]

def parse_args():
    ap = argparse.ArgumentParser(description="Thống kê phân bố lớp data/m1")
    ap.add_argument("--data", required=True, help="Thư mục data/m1 (chứa train/val/test)")
    ap.add_argument("--out", required=True, help="Đường dẫn file .png xuất ra")
    return ap.parse_args()

def count_split(labels_dir: Path):
    counter = Counter()
    n_images = 0
    n_boxes = 0
    if not labels_dir.exists():
        return counter, 0, 0
    for f in labels_dir.glob("*.txt"):
        n_images += 1
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            counter[cid] += 1
            n_boxes += 1
    return counter, n_images, n_boxes

def main():
    args = parse_args()
    data_dir = Path(args.data)
    out_png = Path(args.out)
    out_md = out_png.with_suffix(".md")

    stats = {}
    for split in SPLITS:
        labels_dir = data_dir / split / "labels"
        counter, n_images, n_boxes = count_split(labels_dir)
        stats[split] = {"counter": counter, "n_images": n_images, "n_boxes": n_boxes}
        print(f"{split}: {n_images} ảnh, {n_boxes} bbox")

    # ----- Vẽ histogram: số bbox mỗi lớp, nhóm theo split -----
    n_classes = len(CLASS_NAMES)
    x = range(n_classes)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, split in enumerate(SPLITS):
        counts = [stats[split]["counter"].get(cid, 0) for cid in range(n_classes)]
        ax.bar([xi + i * width for xi in x], counts, width=width, label=split)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(CLASS_NAMES, rotation=20)
    ax.set_ylabel("Số lượng bbox")
    ax.set_title("Phân bố số lượng bbox theo lớp và theo split (data/m1)")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"Đã lưu biểu đồ: {out_png}")

    # ----- Bảng markdown -----
    lines = ["# Phân bố dữ liệu M1\n"]
    lines.append("## Số ảnh mỗi split\n")
    lines.append("| Split | Số ảnh | Tổng bbox |")
    lines.append("|---|---|---|")
    for split in SPLITS:
        lines.append(f"| {split} | {stats[split]['n_images']} | {stats[split]['n_boxes']} |")

    lines.append("\n## Số bbox theo từng lớp\n")
    header = "| Lớp | " + " | ".join(SPLITS) + " | Tổng |"
    sep = "|---|" + "---|" * (len(SPLITS) + 1)
    lines.append(header)
    lines.append(sep)
    for cid, name in enumerate(CLASS_NAMES):
        row = [str(stats[s]["counter"].get(cid, 0)) for s in SPLITS]
        total = sum(stats[s]["counter"].get(cid, 0) for s in SPLITS)
        lines.append(f"| {cid} ({name}) | " + " | ".join(row) + f" | {total} |")

    # ghi chú lớp hiếm (dưới 5% tổng số bbox)
    total_all = sum(sum(stats[s]["counter"].values()) for s in SPLITS)
    lines.append("\n## Ghi chú lớp hiếm\n")
    rare_found = False
    for cid, name in enumerate(CLASS_NAMES):
        total_cls = sum(stats[s]["counter"].get(cid, 0) for s in SPLITS)
        pct = (total_cls / total_all * 100) if total_all else 0
        if pct < 5:
            lines.append(f"- Lớp **{cid} ({name})** chỉ chiếm {pct:.2f}% tổng số bbox ({total_cls}/{total_all}) — cần cân nhắc tăng cường dữ liệu hoặc oversampling.")
            rare_found = True
    if not rare_found:
        lines.append("- Không có lớp nào dưới 5% tổng số bbox.")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Đã lưu bảng thống kê: {out_md}")

if __name__ == "__main__":
    main()
