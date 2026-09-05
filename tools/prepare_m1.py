"""
Gộp 2 bộ M1 (Hard Hat Workers + Construction Site Safety) về schema 6 lớp YOLO.
Chia theo VIDEO/ảnh gốc, loại trùng bằng hash.
"""
import argparse, hashlib, random, shutil, os, yaml
from pathlib import Path
import cv2

# Map tên gốc -> id schema
ALIASES = {
    "person": 0, "worker": 0, "people": 0, "man": 0, "human": 0,
    "helmet": 1, "hard_hat": 1, "with_helmet": 1, "hardhat": 1,
    "no_helmet": 2, "without_helmet": 2, "no_hard_hat": 2, "without_hardhat": 2,
    "vest": 3, "reflective_vest": 3, "safety_vest": 3, "with_vest": 3, "reflective": 3,
    "no_vest": 4, "without_vest": 4, "without_reflective": 4,
    "machinery": 5, "excavator": 5, "crane": 5, "forklift": 5, "mixer": 5,
    "truck": 5, "vehicle": 5, "construction_vehicle": 5,
}

def file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()

def parse_args():
    ap = argparse.ArgumentParser(description="Gộp 2 bộ M1 về 6 lớp YOLO")
    ap.add_argument("--src-hardhat", required=True)
    ap.add_argument("--src-site", required=True)
    ap.add_argument("--out", default="data/m1")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def collect(src: Path):
    items = []
    for split in ["train","valid","val","test",""]:
        d = src / split if split else src
        if not d.exists(): continue
        for img in list(d.rglob("*.jpg")) + list(d.rglob("*.png")):
            # tìm label tương ứng (.txt YOLO)
            label = img.with_suffix(".txt")
            if not label.exists():
                label = d / "labels" / (img.stem + ".txt")
            items.append((img, label, split or "train"))
    return items

def main():
    args = parse_args()
    random.seed(args.seed)
    out = Path(args.out)
    # thu thập
    all_items = collect(Path(args.src_hardhat)) + collect(Path(args.src_site))
    print(f"Thu thập {len(all_items)} ảnh từ 2 nguồn")
    # loại trùng
    seen = set(); uniq = []
    for img, lbl, _ in all_items:
        h = file_hash(img)
        if h in seen: continue
        seen.add(h); uniq.append((img, lbl))
    print(f"Sau loại trùng: {len(uniq)} ảnh")
    # map nhãn
    mapped = []  # (img_path, lines_mapped)
    unknown = set()
    for img, lbl in uniq:
        if not lbl.exists():
            continue
        lines = []
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if not parts: continue
            # Ultralytics export: class cx cy w h  /  hoặc  class_name cx cy w h
            try:
                cls_raw = parts[0]
                # nếu là số -> cần file data.yaml để biết tên, tạm giữ nguyên nếu 0-5
                if cls_raw.replace(".","").isdigit():
                    cid = int(float(cls_raw))
                    if cid <= 5:
                        lines.append(line)
                    else:
                        unknown.add(cls_raw)
                else:
                    key = cls_raw.lower().strip()
                    if key in ALIASES:
                        nid = ALIASES[key]
                        lines.append(f"{nid} " + " ".join(parts[1:]))
                    else:
                        unknown.add(cls_raw)
            except Exception as e:
                unknown.add(parts[0])
        if lines:
            mapped.append((img, lines))
    if unknown:
        print(f"Cảnh báo: nhãn không map được: {unknown}")
    print(f"Giữ lại {len(mapped)} ảnh có nhãn hợp lệ")
    # chia theo video (ở đây theo ảnh, nếu có video thì nhóm theo video id)
    random.shuffle(mapped)
    n = len(mapped); n_train = int(n*args.train_ratio); n_val = int(n*args.val_ratio)
    splits = {"train": mapped[:n_train], "val": mapped[n_train:n_train+n_val], "test": mapped[n_train+n_val:]}
    for k,v in splits.items():
        print(f"  {k}: {len(v)} ảnh")
        (out/k/"images").mkdir(parents=True, exist_ok=True)
        (out/k/"labels").mkdir(parents=True, exist_ok=True)
        for img, lines in v:
            shutil.copy2(img, out/k/"images"/img.name)
            (out/k/"labels"/(img.stem+".txt")).write_text("\n".join(lines))
    # mapping
    (out/"mapping.yaml").write_text(yaml.dump({"aliases": ALIASES, "classes": ["person","helmet","no_helmet","vest","no_vest","machinery"]}, allow_unicode=True))
    print(f"Done -> {out}")

if __name__ == "__main__":
    main()
