"""Đánh giá cấp sự kiện: khớp event hệ thống với ground-truth (GT) -> Precision / Recall / FAR/h.

GT là file CSV do nhóm gán tay (xem data/gt_events/README.md). Script KHÔNG tạo nhãn.

Dùng:
  python tools/eval_events.py --db logs/events.db --gt data/gt_events/<clip>.csv ^
      --clip <clip> --fps 30 --n-frames 2880 [--source <chuỗi con của source>] [--out-json kq.json]
"""
import argparse, csv, json, sqlite3, sys
from pathlib import Path

VTYPES = ("no_helmet", "no_vest", "zone", "machinery", "fall")


def load_gt(csv_path):
    """CSV cột: clip, vtype, start_frame, end_frame, x, y (x,y chuẩn hoá, có thể trống).
    Dòng trống / bắt đầu bằng '#' bị bỏ qua. Trả list[{clip, vtype, start_frame, end_frame, x, y}]."""
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        lines = [(i, ln) for i, ln in enumerate(f, 1) if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return rows
    header = next(csv.reader([lines[0][1]]))
    for ln, raw in lines[1:]:  # ln = số dòng thật trong file CSV
        r = dict(zip(header, next(csv.reader([raw]))))
        def num(key):
            v = (r.get(key) or "").strip()
            return float(v) if v else None
        def frame(key):
            v = (r.get(key) or "").strip()
            if not v:
                raise ValueError(f"{csv_path}, dòng {ln}: ô '{key}' bị trống")
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"{csv_path}, dòng {ln}: '{key}'='{v}' không phải số nguyên") from None
        try:
            x, y = num("x"), num("y")
        except ValueError:
            raise ValueError(f"{csv_path}, dòng {ln}: x/y phải là số hoặc để trống") from None
        s, e = frame("start_frame"), frame("end_frame")
        if s > e:
            raise ValueError(f"{csv_path}, dòng {ln}: start_frame ({s}) > end_frame ({e})")
        rows.append({
            "clip": (r.get("clip") or "").strip(),
            "vtype": (r.get("vtype") or "").strip(),
            "start_frame": s,
            "end_frame": e,
            "x": x, "y": y,
        })
    return rows


def expand_events(pred):
    """Event gộp 'a+b' -> mỗi loại một event riêng. pred: list[{violation_type, frame_idx, ...}]."""
    out = []
    for e in pred:
        for t in str(e["violation_type"]).split("+"):
            out.append({**e, "violation_type": t})
    return out


def match_events(pred, gt, fps, tol_sec=2.0):
    """Khớp tham lam theo thời gian, cùng loại. Một event TP nếu frame_idx nằm trong
    [start - tol, end + tol] của một GT chưa được khớp. Mỗi GT chỉ khớp 1 lần: event thứ 2 trên
    cùng GT là FP (đếm trùng). GT không có event nào là FN.
    pred: list[{violation_type, frame_idx}] (loại gộp 'a+b' được tách); gt: list từ load_gt.
    Trả {tp, fp, fn, per_type: {vtype: {tp, fp, fn}}}. Vị trí x,y của GT chưa dùng (DB không lưu vị trí)."""
    tol = int(round(tol_sec * fps))
    per = {}
    def slot(t):
        return per.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
    matched = [False] * len(gt)
    order = sorted(range(len(gt)), key=lambda i: (gt[i]["start_frame"], i))  # GT sớm nhất khớp trước
    for e in sorted(expand_events(pred), key=lambda e: (e["frame_idx"], e["violation_type"])):
        t, f = e["violation_type"], e["frame_idx"]
        hit = None
        for i in order:
            g = gt[i]
            if matched[i] or g["vtype"] != t:
                continue
            if g["start_frame"] - tol <= f <= g["end_frame"] + tol:
                hit = i
                break
        if hit is None:
            slot(t)["fp"] += 1
        else:
            matched[hit] = True
            slot(t)["tp"] += 1
    for i, g in enumerate(gt):
        if not matched[i]:
            slot(g["vtype"])["fn"] += 1
    return {
        "tp": sum(v["tp"] for v in per.values()),
        "fp": sum(v["fp"] for v in per.values()),
        "fn": sum(v["fn"] for v in per.values()),
        "per_type": per,
    }


def far_per_hour(fp, n_frames, fps):
    """Báo động giả mỗi giờ video."""
    hours = n_frames / fps / 3600.0
    return fp / hours if hours > 0 else float("nan")


def _ratio(a, b):
    return a / b if b > 0 else float("nan")


def summarize(result, n_frames, fps):
    """Thêm precision, recall, far_per_h vào kết quả match_events (tổng và từng loại)."""
    def one(d):
        return {**d, "precision": _ratio(d["tp"], d["tp"] + d["fp"]),
                "recall": _ratio(d["tp"], d["tp"] + d["fn"]),
                "far_per_h": far_per_hour(d["fp"], n_frames, fps)}
    return {**one(result), "n_frames": n_frames, "fps": fps,
            "per_type": {t: one(v) for t, v in sorted(result["per_type"].items())}}


def format_table(summary):
    lines = [f"{'loai':<12}{'TP':>4}{'FP':>4}{'FN':>4}{'P':>8}{'R':>8}{'FAR/h':>9}"]
    def row(name, d):
        return f"{name:<12}{d['tp']:>4}{d['fp']:>4}{d['fn']:>4}{d['precision']:>8.3f}{d['recall']:>8.3f}{d['far_per_h']:>9.2f}"
    for t, d in summary["per_type"].items():
        lines.append(row(t, d))
    lines.append(row("TONG", summary))
    return "\n".join(lines)


def load_events_db(db_path, source=None):
    """Đọc events từ SQLite bằng sqlite3 trực tiếp (không cần module storage)."""
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"Không tìm thấy file DB: {db_path}")
    con = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(events)")}
        sel = "violation_type, frame_idx" + (", source" if "source" in cols else ", NULL")
        rows = con.execute(f"SELECT {sel} FROM events ORDER BY id").fetchall()
    finally:
        con.close()
    ev = [{"violation_type": r[0], "frame_idx": r[1], "source": r[2]} for r in rows]
    if source:
        ev = [e for e in ev if e["source"] and source in e["source"]]
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--clip", default="", help="chỉ lấy GT của clip này (cột clip)")
    ap.add_argument("--source", default="", help="chỉ lấy event có source chứa chuỗi này")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--n-frames", type=int, required=True, help="tổng số frame của clip (để tính FAR/h)")
    ap.add_argument("--tol-sec", type=float, default=2.0)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()
    try:
        gt = load_gt(args.gt)
    except ValueError as e:
        print(f"Lỗi GT: {e}", file=sys.stderr)
        sys.exit(2)
    if not args.source:
        print("Cảnh báo: không truyền --source -> tính MỌI event trong DB (kể cả các lần chạy/clip khác). "
              "Nên dùng --source <chuỗi con của video> hoặc DB sạch (--reset-db khi chạy stream.py).", file=sys.stderr)
    if args.clip:
        gt = [g for g in gt if g["clip"] == args.clip]
    if not gt:
        print("Cảnh báo: không có dòng GT nào -> Recall không xác định, mọi event tính là FP.", file=sys.stderr)
    try:
        pred = load_events_db(args.db, args.source or None)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        sys.exit(2)
    summ = summarize(match_events(pred, gt, args.fps, args.tol_sec), args.n_frames, args.fps)
    print(f"GT={len(gt)} | events={len(pred)} | {args.n_frames} frame @ {args.fps} fps | tol={args.tol_sec}s")
    print(format_table(summ))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summ, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
