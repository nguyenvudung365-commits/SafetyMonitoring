"""
Bổ sung augmentation đặc thù công trường: mưa, sương mù, bóng/bụi mù.
CHỈ áp dụng lên tập TRAIN — tuyệt đối không áp lên val/test để giữ tập test sạch,
đúng nguyên tắc đề tài (đánh giá phải trên dữ liệu chưa qua tăng cường bổ sung).

Đây là các phép biến đổi ở mức PIXEL (không dịch chuyển đối tượng trong ảnh),
nên toạ độ bbox trong file .txt giữ nguyên — chỉ copy label gốc sang cho ảnh mới.
"""
import argparse
import random
from pathlib import Path
import cv2
import albumentations as A

def parse_args():
    ap = argparse.ArgumentParser(description="Tăng cường nhiễu mưa/sương/bụi cho tập train")
    ap.add_argument("--images", required=True, help="Thư mục ảnh train, ví dụ data/m1/train/images")
    ap.add_argument("--labels", required=True, help="Thư mục nhãn train, ví dụ data/m1/train/labels")
    ap.add_argument("--num-augmented", type=int, default=1,
                     help="Số bản augment mưa/sương/bụi tạo thêm cho MỖI ảnh gốc (mặc định 1)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def build_transforms():
    """Mỗi bản augment chỉ áp 1 trong 3 hiệu ứng (chọn ngẫu nhiên), tránh chồng quá nhiều làm ảnh mất chi tiết."""
    rain = A.RandomRain(
        slant_range=(-10, 10), drop_length=15, drop_width=1,
        drop_color=(200, 200, 200), blur_value=3, brightness_coefficient=0.85, p=1.0,
    )
    fog = A.RandomFog(fog_coef_range=(0.2, 0.5), alpha_coef=0.08, p=1.0)
    # "bụi mù" mô phỏng bằng RandomShadow + giảm sáng/tương phản nhẹ
    dust = A.Compose([
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=1.0),  # mặc định 1-2 bóng đổ, đủ dùng
        A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.1), contrast_limit=(-0.1, 0.1), p=1.0),
    ])
    return {"rain": rain, "fog": fog, "dust": dust}

def main():
    args = parse_args()
    random.seed(args.seed)
    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    if not img_dir.exists() or not lbl_dir.exists():
        raise SystemExit(f"Lỗi: không tìm thấy {img_dir} hoặc {lbl_dir}")

    transforms = build_transforms()
    names = list(transforms.keys())

    img_files = sorted([f for f in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
                         if "_weather-" not in f.stem])
    print(f"Tìm thấy {len(img_files)} ảnh gốc trong {img_dir} (đã loại các ảnh đã augment từ lần chạy trước, nếu có)")

    created = 0
    skipped_no_label = 0
    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            skipped_no_label += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            skipped_no_label += 1
            continue

        for i in range(args.num_augmented):
            effect_name = random.choice(names)
            transform = transforms[effect_name]
            augmented = transform(image=image)["image"]

            new_stem = f"{img_path.stem}_weather-{effect_name}-{i}"
            out_img = img_dir / f"{new_stem}{img_path.suffix}"
            out_lbl = lbl_dir / f"{new_stem}.txt"

            cv2.imwrite(str(out_img), augmented)
            out_lbl.write_text(lbl_path.read_text())  # bbox không đổi, copy nguyên nhãn gốc
            created += 1

    print(f"Đã tạo {created} ảnh augment mưa/sương/bụi mới")
    print(f"Bỏ qua {skipped_no_label} ảnh (không có nhãn hoặc không đọc được ảnh)")
    print(f"Tổng ảnh trong {img_dir} sau khi thêm: {len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))}")

if __name__ == "__main__":
    main()
