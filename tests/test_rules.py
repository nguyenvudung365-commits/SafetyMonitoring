import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rules.safety_rules import (
    point_in_polygon, SafetyState, machinery_violation,
    zone_violation, ppe_flags, bbox_edge_distance, overlap_ratio, CLASS_NO_HELMET, CLASS_HELMET, CLASS_NO_VEST, CLASS_VEST,
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
    person=[0.5,0.5,0.6,0.9]            # h=0.4, chân (0.55,0.9)
    near=[0.52,0.7,0.62,0.85]           # chân cách cạnh dưới 0.05 < k*h=0.2
    far=[0.9,0.9,0.95,0.95]
    assert machinery_violation(person, [near], k=0.5) is True
    assert machinery_violation(person, [far], k=0.5) is False


def test_bbox_edge_distance_and_overlap():
    box=[0.2,0.2,0.4,0.4]
    assert bbox_edge_distance((0.3,0.3), box) == 0.0
    assert abs(bbox_edge_distance((0.5,0.3), box) - 0.1) < 1e-9
    assert abs(bbox_edge_distance((0.5,0.5), box) - (0.02**0.5)) < 1e-9
    assert overlap_ratio([0.2,0.2,0.3,0.3], box) == 1.0
    assert abs(overlap_ratio([0.3,0.3,0.5,0.5], box) - 0.25) < 1e-9
    assert overlap_ratio([0.6,0.6,0.7,0.7], box) == 0.0


def test_machinery_edge_not_center():
    # máy xúc rộng: chân người sát cạnh phải nhưng cách tâm xa -> phải báo (bản cũ đo tới tâm thì bỏ sót)
    machine=[0.1,0.5,0.7,0.9]
    person=[0.72,0.5,0.76,0.8]          # h=0.3, chân (0.74,0.8), cách cạnh 0.04 < 0.15; cách tâm ~0.34
    assert machinery_violation(person, [machine], k=0.5) is True


def test_machinery_cabin_excluded():
    machine=[0.3,0.3,0.7,0.9]
    in_cabin=[0.45,0.35,0.52,0.6]       # nằm hẳn trong bbox máy
    assert overlap_ratio(in_cabin, machine) >= 0.8
    assert machinery_violation(in_cabin, [machine], k=0.5, cabin_overlap=0.8) is False
    # tắt luật loại trừ (ngưỡng > 1) thì chân nằm trong bbox máy -> báo
    assert machinery_violation(in_cabin, [machine], k=0.5, cabin_overlap=1.1) is True


def test_machinery_cabin_requires_foot_inside_machine():
    machine=[0.2,0.2,0.8,0.8]
    front=[0.4,0.4,0.45,0.85]           # overlap 0.89 nhưng chân (y=0.85) dưới đáy máy -> đứng trước máy
    assert overlap_ratio(front, machine) >= 0.8
    assert machinery_violation(front, [machine], k=0.5, cabin_overlap=0.8) is True
    seated=[0.4,0.4,0.45,0.75]          # chân y=0.75 <= 0.8 -> ngồi trong cabin, vẫn loại trừ
    assert machinery_violation(seated, [machine], k=0.5, cabin_overlap=0.8) is False


def test_machinery_threshold_scales_with_height():
    # cùng khoảng cách 0.06 tới cạnh phải của máy: người h lớn báo, người h nhỏ (xa) không
    machine=[0.0,0.0,0.3,0.3]
    tall=[0.335,0.05,0.385,0.30]        # h=0.25 -> tau=0.125 > 0.06
    short=[0.335,0.25,0.385,0.30]       # h=0.05 -> tau=0.025 < 0.06
    assert machinery_violation(tall, [machine], k=0.5, cabin_overlap=0.8) is True
    assert machinery_violation(short, [machine], k=0.5, cabin_overlap=0.8) is False


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
    # thứ tự thực tế của ultralytics: conf giảm dần (helmet 0.95 trước no_helmet 0.90)
    dets=[
        {"cls": CLASS_HELMET, "conf": 0.95, "bbox": [0.46,0.22,0.54,0.31]},
        {"cls": CLASS_NO_HELMET, "conf": 0.9, "bbox": [0.45,0.21,0.55,0.32]},
    ]
    flags = ppe_flags(person, dets, conf_th=0.5)
    assert flags["no_helmet"] is False


def test_ppe_order_independent():
    import itertools
    person=[0.4,0.2,0.6,0.8]
    dets=[
        {"cls": CLASS_HELMET, "conf": 0.95, "bbox": [0.46,0.22,0.54,0.31]},
        {"cls": CLASS_NO_HELMET, "conf": 0.90, "bbox": [0.45,0.21,0.55,0.32]},
        {"cls": CLASS_NO_VEST, "conf": 0.80, "bbox": [0.42,0.40,0.58,0.55]},
        {"cls": CLASS_VEST, "conf": 0.60, "bbox": [0.42,0.38,0.58,0.56]},
    ]
    results = {tuple(sorted(ppe_flags(person, list(p), 0.5).items())) for p in itertools.permutations(dets)}
    assert results == {(("no_helmet", False), ("no_vest", True))}   # no_vest 0.8 > vest 0.6


def test_ppe_higher_conf_violation_wins_and_tie_prefers_compliant():
    person=[0.4,0.2,0.6,0.8]
    h={"cls": CLASS_HELMET, "conf": 0.60, "bbox": [0.46,0.22,0.54,0.31]}
    nh={"cls": CLASS_NO_HELMET, "conf": 0.90, "bbox": [0.45,0.21,0.55,0.32]}
    for order in ([h, nh], [nh, h]):
        assert ppe_flags(person, order, 0.5)["no_helmet"] is True
    nh_tie={**nh, "conf": 0.60}
    for order in ([h, nh_tie], [nh_tie, h]):
        assert ppe_flags(person, order, 0.5)["no_helmet"] is False


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
        "thresholds": {"machinery_k": 0.5, "ppe_conf": 0.5, "fall_conf": 0.7},
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
        "thresholds": {"machinery_k": 0.5, "ppe_conf": 0.5, "fall_conf": 0.7},
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
        {"cls": CLASS_VEST, "conf": 0.95, "bbox": [0.42, 0.38, 0.58, 0.56]},
        {"cls": CLASS_NO_VEST, "conf": 0.9, "bbox": [0.42, 0.40, 0.58, 0.55]},
    ]
    assert ppe_flags(person, dets, conf_th=0.5)["no_vest"] is False


def test_ppe_vest_tie_conf_no_vest_first():
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
        "thresholds": {"machinery_k": 0.5, "ppe_conf": 0.5, "fall_conf": 0.7, **th},
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


def test_rule_engine_prunes_dead_track_state():
    eng = RuleEngine(_zone_cfg(zone_person_conf=0.0), fps=30)
    det = [{"cls": 0, "conf": 0.9, "bbox": [0.2, 0.2, 0.3, 0.6]}]
    for i in range(20):
        eng.step(det, i)
    tid = next(iter(eng.last_seen))
    assert any(k[0] == tid for k in eng.history)
    last = 19
    for j in range(eng.prune_after + 5):
        last += 1
        eng.step([], last)  # không còn ai: track tid chết
    assert tid not in eng.last_seen and tid not in eng.alerted
    assert not any(k[0] == tid for k in eng.history)
    assert not any(k[0] == tid for k in eng.state.last_alert)


def test_rule_engine_keeps_state_of_live_track():
    eng = RuleEngine(_zone_cfg(zone_person_conf=0.0), fps=30)
    det = [{"cls": 0, "conf": 0.9, "bbox": [0.2, 0.2, 0.3, 0.6]}]
    for i in range(eng.prune_after + 50):
        eng.step(det, i)
    assert len(eng.last_seen) == 1 and eng.history

