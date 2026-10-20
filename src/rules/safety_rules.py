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

def bbox_edge_distance(point, bbox):
    """Khoảng cách Euclid từ điểm tới bbox [x1,y1,x2,y2] (gần nhất tới cạnh); 0 nếu điểm nằm trong bbox."""
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return (dx * dx + dy * dy) ** 0.5


def overlap_ratio(inner, outer):
    """Diện tích(inner ∩ outer) / diện tích(inner); 0 nếu inner suy biến."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def machinery_violation(person_bbox, machinery_bboxes, k=0.5, cabin_overlap=0.8):
    """Chân người cách CẠNH bbox máy gần nhất < k * chiều cao người (ngưỡng tỉ lệ phối cảnh:
    người xa, h nhỏ -> ngưỡng nhỏ). Máy mà bbox người nằm >= cabin_overlap bên trong
    (người ngồi trong cabin) bị bỏ qua, với điều kiện chân người còn nằm trong bbox máy
    (y2 người <= y2 máy); chân vượt dưới cạnh đáy máy = đứng trước máy, vẫn xét."""
    foot = foot_point(person_bbox)
    h = person_bbox[3] - person_bbox[1]
    tau_px = k * h
    for mb in machinery_bboxes:
        if overlap_ratio(person_bbox, mb) >= cabin_overlap and person_bbox[3] <= mb[3]:
            continue
        if bbox_edge_distance(foot, mb) < tau_px:
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


def _in_torso(person_bbox, item_bbox):
    """Áo gắn người nếu tâm item nằm trong vùng thân (1/6 → 2/3 chiều cao từ đỉnh bbox person)."""
    px1, py1, px2, py2 = person_bbox
    h = py2 - py1
    cx, cy = bbox_center(item_bbox)
    return px1 <= cx <= px2 and py1 + h / 6.0 <= cy <= py1 + 2 * h / 3.0


def ppe_flags(person_bbox, detections, conf_th=0.50):
    """detections: list of {cls, conf, bbox}. Mũ xét 1/3 trên, áo xét vùng thân. Trả {no_helmet, no_vest}.

    Không phụ thuộc thứ tự detection: với mỗi cặp (helmet/no_helmet), (vest/no_vest) lấy detection
    conf cao nhất gắn với người; hoà conf thì ưu tiên lớp tuân thủ (helmet/vest) để tránh báo nhầm."""
    best = {}  # cặp -> (conf, is_violation)
    for d in detections:
        conf = d.get("conf", 1.0)
        if conf < conf_th:
            continue
        cls = d["cls"]
        if cls in (CLASS_HELMET, CLASS_NO_HELMET):
            if not _in_upper_third(person_bbox, d["bbox"]):
                continue
            pair = "helmet"
        elif cls in (CLASS_VEST, CLASS_NO_VEST):
            if not _in_torso(person_bbox, d["bbox"]):
                continue
            pair = "vest"
        else:
            continue
        is_viol = cls in (CLASS_NO_HELMET, CLASS_NO_VEST)
        cur = best.get(pair)
        if cur is None or conf > cur[0] or (conf == cur[0] and cur[1] and not is_viol):
            best[pair] = (conf, is_viol)
    return {
        "no_helmet": bool(best.get("helmet", (0, False))[1]),
        "no_vest": bool(best.get("vest", (0, False))[1]),
    }


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
