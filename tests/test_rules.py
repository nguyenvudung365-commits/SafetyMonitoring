import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rules.safety_rules import (
    point_in_polygon, SafetyState, machinery_violation,
    zone_violation, ppe_flags, CLASS_NO_HELMET, CLASS_HELMET, CLASS_NO_VEST, CLASS_VEST,
)
from tracking.bytetrack import ByteTracker, iou
from pipeline.postprocess import RuleEngine, split_detections, evaluate_person, merge_alerts


def test_zone_inside():
    poly=[[0.5,0.2],[0.8,0.2],[0.8,0.6],[0.5,0.6]]
    assert point_in_polygon((0.6,0.4), poly) is True
    assert point_in_polygon((0.2,0.2), poly) is False


def test_cooldown():
    s=SafetyState(cooldown_sec=2, fps=30)
    assert s.should_alert(1,"no_helmet",0) is True
    assert s.should_alert(1,"no_helmet",30) is False
    assert s.should_alert(1,"no_helmet",61) is True
    assert s.should_alert(1,"vest",30) is True


def test_confirm():
    s=SafetyState()
    assert s.confirm([True]*15, 15) is True
    assert s.confirm([True]*14+[False], 15) is False


def test_machinery():
    person=[0.5,0.5,0.6,0.9]
    near=[0.52,0.7,0.62,0.85]
    far=[0.9,0.9,0.95,0.95]
    assert machinery_violation(person, [near], tau=0.15) is True
    assert machinery_violation(person, [far], tau=0.08) is False


def test_zone_violation_foot():
    poly=[[0.5,0.2],[0.8,0.2],[0.8,0.6],[0.5,0.6]]
    inside=[0.55,0.10,0.65,0.40]
    outside=[0.10,0.10,0.20,0.90]
    assert zone_violation(inside, [poly]) is True
    assert zone_violation(outside, [poly]) is False


def test_ppe_no_helmet_in_upper_third():
    person=[0.4,0.2,0.6,0.8]
    helmet_item={"cls": CLASS_NO_HELMET, "conf": 0.9, "bbox": [0.45,0.21,0.55,0.32]}
    flags = ppe_flags(person, [helmet_item], conf_th=0.5)
    assert flags["no_helmet"] is True
    assert flags["no_vest"] is False


def test_ppe_helmet_overrides():
    person=[0.4,0.2,0.6,0.8]
    dets=[
        {"cls": CLASS_NO_HELMET, "conf": 0.9, "bbox": [0.45,0.21,0.55,0.32]},
        {"cls": CLASS_HELMET, "conf": 0.95, "bbox": [0.46,0.22,0.54,0.31]},
    ]
    flags = ppe_flags(person, dets, conf_th=0.5)
    assert flags["no_helmet"] is False


def test_ppe_outside_person_ignored():
    person=[0.4,0.2,0.6,0.8]
    far={"cls": CLASS_NO_VEST, "conf": 0.99, "bbox": [0.90,0.10,0.95,0.20]}
    flags = ppe_flags(person, [far], conf_th=0.5)
    assert flags["no_vest"] is False


def test_bytetrack_id_stable():
    tr = ByteTracker(iou_thresh=0.3, max_lost=5)
    a = tr.update([[0.1,0.1,0.3,0.5]])
    b = tr.update([[0.12,0.12,0.32,0.52]])
    assert a[0]["track_id"] == b[0]["track_id"] == 1


def test_bytetrack_new_id():
    tr = ByteTracker(iou_thresh=0.3)
    a = tr.update([[0.1,0.1,0.2,0.3]])
    b = tr.update([[0.8,0.8,0.9,0.95]])
    assert a[0]["track_id"] != b[0]["track_id"]


def test_iou():
    assert iou([0,0,1,1],[0,0,1,1]) == 1.0
    assert iou([0,0,1,1],[2,2,3,3]) == 0.0


def test_split_detections():
    dets = [
        {"cls": 0, "conf": 0.9, "bbox": [0,0,1,1]},
        {"cls": 5, "conf": 0.8, "bbox": [0.5,0.5,0.6,0.6]},
        {"cls": 2, "conf": 0.7, "bbox": [0.1,0.1,0.2,0.2]},
    ]
    p, m, o = split_detections(dets)
    assert len(p)==1 and len(m)==1 and len(o)==1


def test_rule_engine_confirm_then_alert():
    cfg = {
        "thresholds": {"machinery_distance": 0.08, "ppe_conf": 0.5, "fall_conf": 0.7},
        "temporal": {"ppe_frames": 3, "fall_frames": 8, "cooldown_sec": 60, "window_size": 30},
        "zones": {"forbidden": [{"name": "z", "polygon": [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]}]},
    }
    eng = RuleEngine(cfg, fps=30)
    person = {"cls": 0, "conf": 0.9, "bbox": [0.4,0.2,0.6,0.8]}
    alerts = []
    for i in range(3):
        out = eng.step([person], i)
        alerts.extend(out["alerts"])
    types = {a["type"] for a in alerts}
    assert "zone" in types


def _fall_cfg():
    return {
        "thresholds": {"machinery_distance": 0.08, "ppe_conf": 0.5, "fall_conf": 0.7},
        "temporal": {"ppe_frames": 15, "fall_frames": 8, "cooldown_sec": 60, "window_size": 30},
        "zones": {"forbidden": []},
    }


def _run_fall(probs_seq):
    eng = RuleEngine(_fall_cfg(), fps=30)
    person = {"cls": 0, "conf": 0.9, "bbox": [0.4, 0.2, 0.6, 0.8]}
    alerts = []
    for i, p in enumerate(probs_seq):
        probs = None if p is None else {1: p}
        alerts.extend(eng.step([person], i, fall_probs=probs)["alerts"])
    return [a for a in alerts if a["type"] == "fall"]


def test_fall_alert_after_n8_frames():
    assert len(_run_fall([0.9] * 8)) == 1


def test_fall_no_alert_below_conf_or_short():
    assert _run_fall([0.6] * 20) == []   # duoi fall_conf 0.7
    assert _run_fall([0.9] * 7) == []    # chua du N=8


def test_fall_branch_off_without_m2():
    assert _run_fall([None] * 20) == []  # khong truyen fall_probs -> bo qua nhanh fall


def test_fall_cooldown_single_event():
    assert len(_run_fall([0.9] * 100)) == 1


def test_ppe_vest_on_torso_detected():
    person = [0.4, 0.2, 0.6, 0.8]                        # h=0.6, thân y 0.3..0.6
    no_vest = {"cls": CLASS_NO_VEST, "conf": 0.9, "bbox": [0.42, 0.40, 0.58, 0.55]}  # tâm y=0.475
    assert ppe_flags(person, [no_vest], conf_th=0.5)["no_vest"] is True


def test_ppe_vest_on_torso_overrides():
    person = [0.4, 0.2, 0.6, 0.8]
    dets = [
        {"cls": CLASS_NO_VEST, "conf": 0.9, "bbox": [0.42, 0.40, 0.58, 0.55]},
        {"cls": CLASS_VEST, "conf": 0.9, "bbox": [0.42, 0.38, 0.58, 0.56]},
    ]
    assert ppe_flags(person, dets, conf_th=0.5)["no_vest"] is False


def test_ppe_vest_at_legs_ignored():
    person = [0.4, 0.2, 0.6, 0.8]
    legs = {"cls": CLASS_NO_VEST, "conf": 0.9, "bbox": [0.42, 0.65, 0.58, 0.78]}   # tâm y=0.715 > 0.6
    assert ppe_flags(person, [legs], conf_th=0.5)["no_vest"] is False


def test_merge_alerts_same_track():
    out = merge_alerts([
        {"track_id": 4, "type": "no_helmet", "frame": 10, "bbox": [0, 0, 1, 1]},
        {"track_id": 4, "type": "no_vest", "frame": 10, "bbox": [0, 0, 1, 1]},
        {"track_id": 7, "type": "zone", "frame": 10, "bbox": [0, 0, 1, 1]},
    ])
    types = {a["track_id"]: a["type"] for a in out}
    assert types == {4: "no_helmet+no_vest", 7: "zone"}


def test_cooldown_scales_with_video_fps():
    cfg = _fall_cfg()
    assert RuleEngine(cfg, fps=25).state.cooldown_frames == 60 * 25
    assert RuleEngine(cfg, fps=30).state.cooldown_frames == 60 * 30


def _zone_cfg(**th):
    return {
        "thresholds": {"machinery_distance": 0.08, "ppe_conf": 0.5, "fall_conf": 0.7, **th},
        "temporal": {"ppe_frames": 3, "fall_frames": 8, "cooldown_sec": 60, "window_size": 30},
        "zones": {"forbidden": [{"name": "z", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]},
    }


def test_zone_ignores_low_conf_person():
    eng = RuleEngine(_zone_cfg(zone_person_conf=0.5), fps=30)
    hook = {"cls": 0, "conf": 0.3, "bbox": [0.4, 0.2, 0.45, 0.4]}
    alerts = [a for i in range(10) for a in eng.step([hook], i)["alerts"]]
    assert alerts == []
    hook["conf"] = 0.6
    eng2 = RuleEngine(_zone_cfg(zone_person_conf=0.5), fps=30)
    assert any(a["type"] == "zone" for i in range(10) for a in eng2.step([hook], i)["alerts"])


def test_dedup_when_tracker_changes_id():
    eng = RuleEngine(_zone_cfg(zone_person_conf=0.5), fps=30)
    p = {"cls": 0, "conf": 0.9, "bbox": [0.4, 0.2, 0.6, 0.8]}
    alerts = [a for i in range(5) for a in eng.step([p], i)["alerts"]]
    assert len(alerts) == 1
    # người biến mất 1 frame rồi hiện lại với id mới (ép bằng cách xoá track cũ)
    eng.tracker.tracks.clear()
    p2 = {"cls": 0, "conf": 0.9, "bbox": [0.41, 0.2, 0.61, 0.8]}
    more = [a for i in range(5, 12) for a in eng.step([p2], i)["alerts"]]
    assert more == []


def test_no_dedup_for_far_person():
    eng = RuleEngine(_zone_cfg(zone_person_conf=0.5), fps=30)
    a = {"cls": 0, "conf": 0.9, "bbox": [0.1, 0.2, 0.2, 0.8]}
    b = {"cls": 0, "conf": 0.9, "bbox": [0.7, 0.2, 0.8, 0.8]}
    alerts = [x for i in range(5) for x in eng.step([a], i)["alerts"]]
    eng.tracker.tracks.clear()
    alerts += [x for i in range(5, 12) for x in eng.step([b], i)["alerts"]]
    assert len(alerts) == 2
