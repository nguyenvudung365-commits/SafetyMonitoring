"""3 nhánh hậu xử lý output M1: PPE, vùng cấm, khoảng cách người–máy."""
from rules.safety_rules import (
    CLASS_PERSON, CLASS_MACHINERY,
    ppe_flags, zone_violation, machinery_violation, load_config,
)
from tracking.bytetrack import ByteTracker, iou


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


def evaluate_person(person_bbox, others, machines, polygons, cfg, person_conf=1.0):
    th = cfg["thresholds"]
    flags = ppe_flags(person_bbox, others, conf_th=th["ppe_conf"])
    # vùng cấm cần person conf cao hơn ngưỡng detect mặc định để loại móc cẩu/vật nhỏ bị nhận nhầm
    in_zone = person_conf >= th.get("zone_person_conf", 0.0) and zone_violation(person_bbox, polygons)
    near_m = machinery_violation(
        person_bbox, [m["bbox"] for m in machines],
        k=th.get("machinery_k", 0.5), cabin_overlap=th.get("cabin_overlap", 0.8),
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
        self.n_fall = t["fall_frames"]
        self.fall_conf = cfg["thresholds"]["fall_conf"]
        self.dedup_frames = int(t.get("dedup_sec", 3) * fps)
        self.dedup_iou = cfg["thresholds"].get("dedup_iou", 0.2)
        self.alerted = {}  # track_id -> {types, bbox, last_seen}: để gộp sự kiện khi tracker đổi id
        self.last_seen = {}  # track_id -> frame_idx thấy gần nhất, để dọn state của track đã chết
        # id không tái sử dụng; quá hạn này (đủ cho dedup + tracker giữ track mất) thì state vô dụng
        self.prune_after = self.dedup_frames + self.tracker.max_lost + 1
        self.polygons = [z["polygon"] for z in cfg.get("zones", {}).get("forbidden", [])]

    def step(self, detections, frame_idx, fall_probs=None):
        """fall_probs: {track_id: P(falling|fallen)} từ M2; None = không chạy M2 (bỏ qua nhánh fall)."""
        persons, machines, others = split_detections(detections)
        tracks = self.tracker.update([p["bbox"] for p in persons])
        conf_by_box = {tuple(p["bbox"]): p["conf"] for p in persons}
        alerts = []
        seen = set()
        seen_now = {t["track_id"] for t in tracks}
        for tr in tracks:
            vtypes = evaluate_person(tr["bbox"], others, machines, self.polygons, self.cfg,
                                   person_conf=conf_by_box.get(tuple(tr["bbox"]), 1.0))
            if fall_probs is not None and fall_probs.get(tr["track_id"], 0.0) > self.fall_conf:
                vtypes.append("fall")
            seen.add(tr["track_id"])
            self.last_seen[tr["track_id"]] = frame_idx
            for vt in ("no_helmet", "no_vest", "zone", "machinery", "fall"):
                if vt == "fall" and fall_probs is None:
                    continue
                n_conf = self.n_fall if vt == "fall" else self.n_ppe
                key = (tr["track_id"], vt)
                hist = self.history.setdefault(key, [])
                hist.append(vt in vtypes)
                if len(hist) > n_conf:
                    del hist[:-n_conf]
                if self.state.confirm(hist, n_conf) and self.state.should_alert(
                    tr["track_id"], vt, frame_idx
                ):
                    if self._is_duplicate(tr, vt, frame_idx, seen_now):
                        self.alerted.setdefault(tr["track_id"], {"types": set()})["types"].add(vt)
                        continue
                    alerts.append({
                        "track_id": tr["track_id"], "type": vt, "frame": frame_idx,
                        "bbox": tr["bbox"],
                    })
        for al in alerts:
            rec = self.alerted.setdefault(al["track_id"], {"types": set()})
            rec["types"].update(al["type"].split("+"))
        for tr in tracks:
            rec = self.alerted.get(tr["track_id"])
            if rec is not None:
                rec["bbox"], rec["last_seen"] = tr["bbox"], frame_idx
        self._prune(frame_idx)
        return {"tracks": tracks, "alerts": merge_alerts(alerts)}

    def _prune(self, frame_idx):
        """Xoá history/alerted/cooldown của track không còn xuất hiện, tránh rò bộ nhớ trên luồng dài."""
        dead = {tid for tid, last in self.last_seen.items() if frame_idx - last > self.prune_after}
        if not dead:
            return
        for tid in dead:
            del self.last_seen[tid]
            self.alerted.pop(tid, None)
        for key in [k for k in self.history if k[0] in dead]:
            del self.history[key]
        for key in [k for k in self.state.last_alert if k[0] in dead]:
            del self.state.last_alert[key]

    def _is_duplicate(self, tr, vt, frame_idx, seen_now):
        """Cùng loại vi phạm đã báo cho một track vừa mất id ở gần vị trí này -> cùng một người."""
        for tid, rec in self.alerted.items():
            if tid == tr["track_id"] or tid in seen_now or vt not in rec["types"]:
                continue
            if frame_idx - rec["last_seen"] <= self.dedup_frames and iou(tr["bbox"], rec["bbox"]) >= self.dedup_iou:
                return True
        return False


def merge_alerts(alerts):
    """Gộp các vi phạm cùng track_id xác nhận trong cùng frame thành 1 event (type nối bằng '+')."""
    by_id = {}
    for al in alerts:
        if al["track_id"] in by_id:
            by_id[al["track_id"]]["types"].append(al["type"])
        else:
            by_id[al["track_id"]] = {**al, "types": [al["type"]]}
    for al in by_id.values():
        al["type"] = "+".join(al["types"])
    return list(by_id.values())
