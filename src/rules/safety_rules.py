"""Tầng luật an toàn — đọc từ YAML, không phụ thuộc torch/ultralytics."""
from pathlib import Path
import yaml

CLASS_PERSON, CLASS_HELMET, CLASS_NO_HELMET = 0, 1, 2
CLASS_VEST, CLASS_NO_VEST, CLASS_MACHINERY = 3, 4, 5


def load_config(path="configs/safety.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def point_in_polygon(pt, polygon):
    """Ray casting — pt=(x,y), polygon=list[(x,y)] chuẩn hoá 0-1."""
    x,y = pt; inside=False; n=len(polygon)
    for i in range(n):
        x1,y1 = polygon[i]; x2,y2 = polygon[(i+1)%n]
        if ((y1>y) != (y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-9)+x1):
            inside = not inside
    return inside

def foot_point(bbox):
    """Tâm cạnh dưới bbox [x1,y1,x2,y2] (chuẩn hoá)."""
    x1,y1,x2,y2 = bbox
    return ((x1+x2)/2, y2)

def euclidean(p, q):
    return ((p[0]-q[0])**2 + (p[1]-q[1])**2)**0.5

def bbox_center(bbox):
    x1,y1,x2,y2 = bbox; return ((x1+x2)/2, (y1+y2)/2)

def machinery_violation(person_bbox, machinery_bboxes, tau=0.08):
    """Khoảng cách Euclid giữa chân người và tâm máy gần nhất < τ (hiệu chỉnh theo h person)."""
    foot = foot_point(person_bbox)
    h = person_bbox[3]-person_bbox[1]
    tau_adj = tau * (1 + (1-h))  # vật xa (h nhỏ) -> τ lớn hơn
    for mb in machinery_bboxes:
        if euclidean(foot, bbox_center(mb)) < tau_adj:
            return True
    return False


def zone_violation(person_bbox, polygons):
    """True nếu chân người nằm trong bất kỳ polygon vùng cấm."""
    foot = foot_point(person_bbox)
    return any(point_in_polygon(foot, poly) for poly in polygons)


def _in_upper_third(person_bbox, item_bbox):
    """PPE gắn người nếu tâm item nằm trong 1/3 trên bbox person."""
    px1, py1, px2, py2 = person_bbox
    h = py2 - py1
    cx, cy = bbox_center(item_bbox)
    return px1 <= cx <= px2 and py1 <= cy <= py1 + h / 3.0


def ppe_flags(person_bbox, detections, conf_th=0.50):
    """detections: list of {cls, conf, bbox}. Trả {no_helmet, no_vest}."""
    no_helmet = False
    no_vest = False
    for d in detections:
        if d.get("conf", 1.0) < conf_th:
            continue
        cls = d["cls"]
        if cls not in (CLASS_HELMET, CLASS_NO_HELMET, CLASS_VEST, CLASS_NO_VEST):
            continue
        if not _in_upper_third(person_bbox, d["bbox"]):
            continue
        if cls == CLASS_NO_HELMET:
            no_helmet = True
        if cls == CLASS_NO_VEST:
            no_vest = True
        if cls == CLASS_HELMET:
            no_helmet = False
        if cls == CLASS_VEST:
            no_vest = False
    return {"no_helmet": no_helmet, "no_vest": no_vest}


class SafetyState:
    """Giữ cooldown theo (track_id, violation_type)."""
    def __init__(self, cooldown_sec=60, fps=30):
        self.cooldown_frames = int(cooldown_sec*fps)
        self.last_alert = {}  # (id, type) -> frame_idx
    def should_alert(self, track_id, vtype, frame_idx):
        key=(track_id,vtype)
        last=self.last_alert.get(key, -10**9)
        if frame_idx - last >= self.cooldown_frames:
            self.last_alert[key]=frame_idx
            return True
        return False
    def confirm(self, history, N):
        """history: list[bool] tín hiệu thô N frame gần nhất."""
        return len(history)>=N and all(history[-N:])
