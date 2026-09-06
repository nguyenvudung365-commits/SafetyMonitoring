"""
Gộp 2 bộ M1 (Hard Hat Workers + Construction Site Safety) về schema 6 lớp YOLO.
Chia theo VIDEO/ảnh gốc, loại trùng bằng hash.
"""
import argparse, hashlib, random, shutil, os, yaml
from pathlib import Path
import cv2

# Map tên gốc (đã chuẩn hoá) -> id schema. None = không thuộc schema 6 lớp, sẽ bị loại.
ALIASES = {
    "person": 0, "worker": 0, "people": 0, "man": 0, "human": 0,
    "helmet": 1, "hard_hat": 1, "with_helmet": 1, "hardhat": 1,
    "no_helmet": 2, "without_helmet": 2, "no_hard_hat": 2, "without_hardhat": 2,
    "no_hardhat": 2, "head": 2,  # dataset "Hard Hat Workers" dùng "head" làm proxy cho no_helmet
    "vest": 3, "reflective_vest": 3, "safety_vest": 3, "with_vest": 3, "reflective": 3,
    "no_vest": 4, "without_vest": 4, "without_reflective": 4, "no_safety_vest": 4,
    "machinery": 5, "excavator": 5, "crane": 5, "forklift": 5, "mixer": 5,
    "truck": 5, "vehicle": 5, "construction_vehicle": 5,
    # Các lớp không thuộc schema 6 lớp -> loại bỏ (không gán, không xoá nhầm)
    "mask": None, "no_mask": None, "safety_cone": None,
}

def norm(name: str) -> str:
    """Chuẩn hoá tên lớp: lower, thay khoảng trắng/gạch ngang bằng gạch dưới."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")

def load_class_names(src: Path):
    """Đọc data.yaml của bộ dữ liệu gốc để lấy danh sách tên lớp theo đúng thứ tự ID số."""
    yml = src / "data.yaml"
    if not yml.exists():
        return None
    data = yaml.safe_load(yml.read_text())
    return data.get("names")

def build_id_map(names):
    """Trả về dict: id số gốc -> id schema 6 lớp (hoặc None nếu lớp đó không thuộc schema)."""
    id_map = {}
    unknown_names = []
    for i, name in enumerate(names):
        key = norm(name)
        if key in ALIASES:
            id_map[i] = ALIASES[key]
        else:
            id_map[i] = None
            unknown_names.append(name)
    if unknown_names:
        print(f"  Cảnh báo: lớp không map được (sẽ bị loại): {unknown_names}")
    return id_map

def file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()

def group_key(img: Path) -> str:
    """
    Roboflow đặt tên file dạng '<ten_goc>_jpg.rf.<hash>.jpg' khi export augmented.
    Các bản augment cùng 1 ảnh gốc có cùng '<ten_goc>' -> dùng làm group ID
    để đảm bảo không rơi vào cả train và test (tránh rò rỉ dữ liệu).
    Nếu không khớp pattern (ảnh không phải từ Roboflow), dùng chính tên file làm group.
    """
    stem = img.stem
    if ".rf." in stem:
        return stem.split(".rf.")[0]
    return stem

def parse_args():
    ap = argparse.ArgumentParser(description="Gộp 2 bộ M1 về 6 lớp YOLO")
    ap.add_argument("--src-hardhat", required=True)
    ap.add_argument("--src-site", required=True)
    ap.add_argument("--out", default="data/m1")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def collect(src: Path, source_tag: str):
    items = []
    for split in ["train","valid","val","test",""]:
        d = src / split if split else src
        if not d.exists(): continue
        for img in list((d/"images").rglob("*.jpg")) + list((d/"images").rglob("*.png")) \
                   + list(d.rglob("*.jpg")) + list(d.rglob("*.png")):
            # tìm label tương ứng (.txt YOLO)
            label = img.with_suffix(".txt")
            if not label.exists():
                label = img.parent.parent / "labels" / (img.stem + ".txt")
            if not label.exists():
                label = d / "labels" / (img.stem + ".txt")
            items.append((img, label, source_tag))
    # loại trùng ảnh có thể bị quét 2 lần do 2 pattern rglob ở trên
    seen_paths = set(); uniq_items = []
    for it in items:
        if it[0] in seen_paths: continue
        seen_paths.add(it[0]); uniq_items.append(it)
    return uniq_items

def main():
    args = parse_args()
    random.seed(args.seed)
    out = Path(args.out)
    src_hardhat = Path(args.src_hardhat)
    src_site = Path(args.src_site)

    # đọc data.yaml từng bộ để biết đúng tên lớp theo ID số của riêng bộ đó
    names_hardhat = load_class_names(src_hardhat)
    names_site = load_class_names(src_site)
    if names_hardhat is None or names_site is None:
        raise SystemExit("Lỗi: không tìm thấy data.yaml trong 1 trong 2 thư mục nguồn. "
                          "Hãy đảm bảo đã giải nén đúng, file data.yaml phải nằm ở thư mục gốc mỗi bộ.")
    print(f"Bộ hardhat - nc={len(names_hardhat)}: {names_hardhat}")
    print(f"Bộ site    - nc={len(names_site)}: {names_site}")

    id_map_hardhat = build_id_map(names_hardhat)
    id_map_site = build_id_map(names_site)

    # thu thập, gắn tag nguồn để biết dùng id_map nào
    items_hardhat = collect(src_hardhat, "hardhat")
    items_site = collect(src_site, "site")
    all_items = items_hardhat + items_site
    print(f"Thu thập {len(all_items)} ảnh từ 2 nguồn ({len(items_hardhat)} hardhat + {len(items_site)} site)")

    # loại trùng theo hash nội dung ảnh
    seen = set(); uniq = []
    for img, lbl, tag in all_items:
        h = file_hash(img)
        if h in seen: continue
        seen.add(h); uniq.append((img, lbl, tag))
    print(f"Sau loại trùng: {len(uniq)} ảnh")

    # map nhãn: dùng đúng id_map theo nguồn của ảnh đó
    mapped = []  # (img_path, lines_mapped)
    dropped_lines = 0
    for img, lbl, tag in uniq:
        if not lbl.exists():
            continue
        id_map = id_map_hardhat if tag == "hardhat" else id_map_site
        lines = []
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if not parts: continue
            try:
                cid_raw = int(float(parts[0]))
            except ValueError:
                dropped_lines += 1
                continue
            new_id = id_map.get(cid_raw)
            if new_id is None:
                dropped_lines += 1  # lớp không thuộc schema (vd Mask, Safety Cone) -> loại đúng ý đồ
                continue
            lines.append(f"{new_id} " + " ".join(parts[1:]))
        if lines:
            mapped.append((img, lines))
    print(f"Đã loại {dropped_lines} dòng nhãn không thuộc schema 6 lớp")
    print(f"Giữ lại {len(mapped)} ảnh có ít nhất 1 nhãn hợp lệ")
    # Chia theo NHÓM (ảnh gốc trước augment), không chia theo từng ảnh riêng lẻ,
    # để các bản augment của cùng 1 ảnh gốc không rơi vào cả train và test.
    groups = {}
    for img, lines in mapped:
        gk = group_key(img)
        groups.setdefault(gk, []).append((img, lines))
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    n_groups = len(group_keys)
    n_train_g = int(n_groups*args.train_ratio); n_val_g = int(n_groups*args.val_ratio)
    train_keys = group_keys[:n_train_g]
    val_keys = group_keys[n_train_g:n_train_g+n_val_g]
    test_keys = group_keys[n_train_g+n_val_g:]
    print(f"Tổng {n_groups} nhóm ảnh gốc -> train {len(train_keys)} / val {len(val_keys)} / test {len(test_keys)} nhóm")

    splits = {
        "train": [item for k in train_keys for item in groups[k]],
        "val":   [item for k in val_keys   for item in groups[k]],
        "test":  [item for k in test_keys  for item in groups[k]],
    }
    for k,v in splits.items():
        print(f"  {k}: {len(v)} ảnh")
        (out/k/"images").mkdir(parents=True, exist_ok=True)
        (out/k/"labels").mkdir(parents=True, exist_ok=True)
        for img, lines in v:
            shutil.copy2(img, out/k/"images"/img.name)
            (out/k/"labels"/(img.stem+".txt")).write_text("\n".join(lines))
    # mapping (ghi lại bảng alias đã dùng để tra cứu/audit sau này)
    (out/"mapping.yaml").write_text(yaml.dump({"aliases": ALIASES, "classes": ["person","helmet","no_helmet","vest","no_vest","machinery"]}, allow_unicode=True))

    # data.yaml chuẩn Ultralytics để Hoài trỏ vào khi train YOLOv8 (--data data/m1/data.yaml)
    data_yaml_content = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 6,
        "names": ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery"],
    }
    (out/"data.yaml").write_text(yaml.dump(data_yaml_content, allow_unicode=True, sort_keys=False))
    print(f"Đã tạo {out/'data.yaml'} (dùng cho Ultralytics YOLOv8 train)")
    print(f"Done -> {out}")

if __name__ == "__main__":
    main()
