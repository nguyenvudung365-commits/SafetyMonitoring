"""ByteTrack-style IoU tracker — không phụ thuộc ultralytics.

Gán track_id ổn định cho bbox person theo IoU greedy matching.
"""


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ByteTracker:
    def __init__(self, iou_thresh=0.3, max_lost=30):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}  # id -> {bbox, lost}

    def update(self, detections):
        """detections: list of [x1,y1,x2,y2] (chuẩn hoá 0-1).
        Returns list of dict {track_id, bbox}.
        """
        used_det = set()
        used_tid = set()
        pairs = []
        for tid, t in self.tracks.items():
            for i, det in enumerate(detections):
                pairs.append((iou(t["bbox"], det), tid, i))
        pairs.sort(reverse=True)

        assigned = []
        for score, tid, i in pairs:
            if score < self.iou_thresh:
                break
            if tid in used_tid or i in used_det:
                continue
            used_tid.add(tid)
            used_det.add(i)
            self.tracks[tid] = {"bbox": detections[i], "lost": 0}
            assigned.append({"track_id": tid, "bbox": detections[i]})

        for tid, t in list(self.tracks.items()):
            if tid not in used_tid:
                t["lost"] += 1
                if t["lost"] > self.max_lost:
                    del self.tracks[tid]

        for i, det in enumerate(detections):
            if i in used_det:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": det, "lost": 0}
            assigned.append({"track_id": tid, "bbox": det})

        return assigned
