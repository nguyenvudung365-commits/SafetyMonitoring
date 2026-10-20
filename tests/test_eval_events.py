import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from eval_events import load_gt, match_events, far_per_hour, summarize, load_events_db
import sweep_rules
from storage.events import EventDB

FPS = 30


def _gt(vtype, s, e):
    return {"clip": "c", "vtype": vtype, "start_frame": s, "end_frame": e, "x": None, "y": None}


def _ev(vtype, f):
    return {"violation_type": vtype, "frame_idx": f}


def test_tp_fp_fn():
    gt = [_gt("zone", 100, 200), _gt("no_helmet", 500, 600)]
    r = match_events([_ev("zone", 150), _ev("zone", 2000)], gt, FPS)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["per_type"]["no_helmet"]["fn"] == 1


def test_duplicate_on_same_gt_is_fp():
    r = match_events([_ev("zone", 120), _ev("zone", 180)], [_gt("zone", 100, 200)], FPS)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)


def test_tolerance_and_type_must_match():
    gt = [_gt("zone", 100, 200)]
    assert match_events([_ev("zone", 255)], gt, FPS, tol_sec=2.0)["tp"] == 1   # 200 + 60 = 260
    assert match_events([_ev("zone", 270)], gt, FPS, tol_sec=2.0)["tp"] == 0
    assert match_events([_ev("no_vest", 150)], gt, FPS)["tp"] == 0


def test_merged_event_counts_each_type():
    gt = [_gt("no_vest", 100, 200), _gt("machinery", 100, 200)]
    r = match_events([_ev("no_vest+machinery", 150)], gt, FPS)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)


def test_far_per_hour():
    assert far_per_hour(3, 108000, 30) == 3.0


def test_load_gt_skips_comments(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("clip,vtype,start_frame,end_frame,x,y\n# ghi chu\nc,zone,10,20,0.5,\n", encoding="utf-8")
    g = load_gt(p)
    assert g == [{"clip": "c", "vtype": "zone", "start_frame": 10, "end_frame": 20, "x": 0.5, "y": None}]


def test_load_gt_bad_rows_report_line_number(tmp_path):
    import pytest
    head = "clip,vtype,start_frame,end_frame,x,y\n"
    p = tmp_path / "a.csv"
    p.write_text(head + "# c\nc,zone,10,20,,\nc,zone,30,25,,\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"dòng 4.*start_frame"):
        load_gt(p)
    p.write_text(head + "c,zone,,20,,\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"dòng 2.*start_frame"):
        load_gt(p)
    p.write_text(head + "c,zone,1,abc,,\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"dòng 2.*end_frame"):
        load_gt(p)


def test_summarize_and_db_loader(tmp_path):
    db = EventDB(str(tmp_path / "e.db"))
    db.insert(1, "zone", 150, source="a.mp4")
    db.insert(2, "zone", 150, source="b.mp4")
    db.close()
    ev = load_events_db(str(tmp_path / "e.db"), source="a.mp4")
    assert len(ev) == 1
    s = summarize(match_events(ev, [_gt("zone", 100, 200)], FPS), 108000, FPS)
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["far_per_h"] == 0.0


def _write_dets(path):
    person = {"cls": 0, "conf": 0.9, "bbox": [0.4, 0.2, 0.6, 0.8]}
    no_h = {"cls": 2, "conf": 0.9, "bbox": [0.45, 0.21, 0.55, 0.32]}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": {"fps": 30}}) + "\n")
        for i in range(40):
            f.write(json.dumps({"frame": i, "dets": [person, no_h]}) + "\n")


def test_sweep_replay_deterministic(tmp_path):
    p = tmp_path / "d.jsonl"
    _write_dets(p)
    meta, frames = sweep_rules.load_dets(str(p))
    cfg = {
        "thresholds": {"machinery_k": 0.5, "ppe_conf": 0.5, "fall_conf": 0.7, "zone_person_conf": 0.5},
        "temporal": {"ppe_frames": 15, "fall_frames": 8, "cooldown_sec": 60, "window_size": 30},
        "zones": {"forbidden": []},
    }
    assert meta["fps"] == 30 and len(frames) == 40
    a = sweep_rules.run_rules(cfg, frames, 30)
    b = sweep_rules.run_rules(cfg, frames, 30)
    assert a == b and [e["violation_type"] for e in a] == ["no_helmet"]
    # N lớn hơn số frame có vi phạm -> không có event
    cfg2 = sweep_rules.apply(cfg, {"ppe_frames": 50, "ppe_conf": 0.5, "zone_person_conf": 0.5, "machinery_k": 0.5})
    assert sweep_rules.run_rules(cfg2, frames, 30) == []
