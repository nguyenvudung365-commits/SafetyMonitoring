import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rules.safety_rules import (
    point_in_polygon, SafetyState, machinery_violation,
    zone_violation, ppe_flags, CLASS_NO_HELMET, CLASS_HELMET, CLASS_NO_VEST,
)
from tracking.bytetrack import ByteTracker, iou
from pipeline.postprocess import RuleEngine, split_detections, evaluate_person


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
