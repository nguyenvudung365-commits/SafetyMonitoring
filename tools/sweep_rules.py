"""Quét tham số tầng luật trên detection đã cache (--dump-dets của stream.py), không chạy YOLO.

Không import ultralytics/torch. Kết quả xác định: cùng input + cùng lưới -> cùng output.

  python tools/sweep_rules.py --dets logs/dets.jsonl --gt data/gt_events/<clip>.csv --clip <clip> ^
      --config configs/safety.yaml --ppe-frames 5,10,15,25 --ppe-conf 0.3,0.4,0.5,0.6,0.7 ^
      --out-csv logs/sweep.csv [--plot logs/recall_far.png]
"""
import argparse, copy, csv, itertools, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml
from pipeline.postprocess import RuleEngine
from eval_events import load_gt, match_events, summarize, expand_events


def load_dets(path):
    """Trả (meta, frames) với frames = list[(frame_idx, dets)]."""
    meta, frames = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if "meta" in o:
                meta = o["meta"]
            else:
                frames.append((o["frame"], o["dets"]))
    return meta, frames


def run_rules(cfg, frames, fps):
    """Chạy lại RuleEngine, trả list[{violation_type, frame_idx}]."""
    eng = RuleEngine(cfg, fps=fps)
    events = []
    for idx, dets in frames:
        for al in eng.step(dets, idx)["alerts"]:
            events.append({"violation_type": al["type"], "frame_idx": al["frame"]})
    return events


def _floats(s, cast=float):
    return [cast(x) for x in s.split(",") if x.strip()] if s else []


def build_grid(cfg, args):
    """Danh sách (tên tham số, giá trị) -> dict ghi đè. Tham số bỏ trống giữ giá trị trong config."""
    t, th = cfg["temporal"], cfg["thresholds"]
    axes = {
        "ppe_frames": _floats(args.ppe_frames, int) or [t["ppe_frames"]],
        "ppe_conf": _floats(args.ppe_conf) or [th["ppe_conf"]],
        "zone_person_conf": _floats(args.zone_person_conf) or [th.get("zone_person_conf", 0.0)],
        "machinery_k": _floats(args.machinery_k) or [th.get("machinery_k", 0.5)],
        "cabin_overlap": _floats(getattr(args, "cabin_overlap", "")) or [th.get("cabin_overlap", 0.8)],
    }
    names = list(axes)
    return names, [dict(zip(names, vals)) for vals in itertools.product(*axes.values())]


def apply(cfg, ov):
    c = copy.deepcopy(cfg)
    c["temporal"]["ppe_frames"] = ov["ppe_frames"]
    for k in ("ppe_conf", "zone_person_conf", "machinery_k", "cabin_overlap"):
        if k in ov:
            c["thresholds"][k] = ov[k]
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--clip", default="")
    ap.add_argument("--config", default="configs/safety.yaml")
    ap.add_argument("--fps", type=float, default=0, help="mặc định lấy từ meta trong file dets")
    ap.add_argument("--tol-sec", type=float, default=2.0)
    ap.add_argument("--ppe-frames", default="")
    ap.add_argument("--ppe-conf", default="")
    ap.add_argument("--zone-person-conf", default="")
    ap.add_argument("--machinery-k", default="")
    ap.add_argument("--cabin-overlap", default="")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--plot", default="", help="PNG Recall-FAR (cần matplotlib)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    meta, frames = load_dets(args.dets)
    fps = args.fps or meta.get("fps") or 30
    n_frames = (frames[-1][0] + 1) if frames else 0
    gt = load_gt(args.gt)
    if args.clip:
        gt = [g for g in gt if g["clip"] == args.clip]
    names, grid = build_grid(cfg, args)

    rows = []
    for ov in grid:
        events = run_rules(apply(cfg, ov), frames, fps)
        s = summarize(match_events(events, gt, fps, args.tol_sec), n_frames, fps)
        rows.append({**ov, "events": len(expand_events(events)), "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
                     "precision": round(s["precision"], 4), "recall": round(s["recall"], 4),
                     "far_per_h": round(s["far_per_h"], 3)})
    # events = số event sau khi tách event gộp "a+b" (cùng đơn vị với tp/fp)
    cols = names + ["events", "tp", "fp", "fn", "precision", "recall", "far_per_h"]
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not found, skipping --plot", file=sys.stderr)
            plt = None
        if plt is not None:
            plt.figure(figsize=(5, 4))
            plt.scatter([r["far_per_h"] for r in rows], [r["recall"] for r in rows])
            plt.xlabel("FAR / h")
            plt.ylabel("Recall")
            plt.title("Recall-FAR by rule config")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
    # In sau khi ghi CSV/plot; chỉ ASCII để không crash trên console không phải UTF-8.
    print(f"{n_frames} frame @ {fps} fps | GT={len(gt)} | {len(rows)} configs")
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
