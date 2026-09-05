"""3 nhánh hậu xử lý output M1: PPE, vùng cấm, khoảng cách người–máy."""
from rules.safety_rules import (
    CLASS_PERSON, CLASS_MACHINERY,
    ppe_flags, zone_violation, machinery_violation, load_config,
)
from tracking.bytetrack import ByteTracker


def split_detections(dets):
    """dets: list {cls, conf, bbox}."""
    persons, machines, others = [], [], []
    for d in dets:
        if d["cls"] == CLASS_PERSON:
            persons.append(d)
        elif d["cls"] == CLASS_MACHINERY:
            machines.append(d)
        else:
            others.append(d)
    return persons, machines, others


def evaluate_person(person_bbox, others, machines, polygons, cfg):
    th = cfg["thresholds"]
    flags = ppe_flags(person_bbox, others, conf_th=th["ppe_conf"])
    in_zone = zone_violation(person_bbox, polygons)
    near_m = machinery_violation(
        person_bbox, [m["bbox"] for m in machines], tau=th["machinery_distance"]
    )
    vtypes = []
    if flags["no_helmet"]:
        vtypes.append("no_helmet")
    if flags["no_vest"]:
        vtypes.append("no_vest")
    if in_zone:
        vtypes.append("zone")
    if near_m:
        vtypes.append("machinery")
    return vtypes


class RuleEngine:
    def __init__(self, cfg, fps=30):
        self.cfg = cfg
        self.tracker = ByteTracker()
        self.state = None
        from rules.safety_rules import SafetyState
        t = cfg["temporal"]
        self.state = SafetyState(cooldown_sec=t["cooldown_sec"], fps=fps)
        self.history = {}  # (track_id, vtype) -> list[bool]
        self.n_ppe = t["ppe_frames"]
        self.polygons = [z["polygon"] for z in cfg.get("zones", {}).get("forbidden", [])]

    def step(self, detections, frame_idx):
        persons, machines, others = split_detections(detections)
        tracks = self.tracker.update([p["bbox"] for p in persons])
        alerts = []
        seen = set()
        for tr in tracks:
            vtypes = evaluate_person(tr["bbox"], others, machines, self.polygons, self.cfg)
            seen.add(tr["track_id"])
            for vt in ("no_helmet", "no_vest", "zone", "machinery"):
                key = (tr["track_id"], vt)
                hist = self.history.setdefault(key, [])
                hist.append(vt in vtypes)
                if len(hist) > self.n_ppe:
                    del hist[:-self.n_ppe]
                if self.state.confirm(hist, self.n_ppe) and self.state.should_alert(
                    tr["track_id"], vt, frame_idx
                ):
                    alerts.append({"track_id": tr["track_id"], "type": vt, "frame": frame_idx})
        return {"tracks": tracks, "alerts": alerts}
