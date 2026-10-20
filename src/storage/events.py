"""SQLite event log — schema tuần 7: id, track_id, violation_type, timestamp, frame_idx, confidence, image_path (+ source, video_time_sec từ migrate tuần 8)."""
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    violation_type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    frame_idx INTEGER NOT NULL,
    confidence REAL,
    image_path TEXT
);
"""


_COLS = "id, track_id, violation_type, timestamp, frame_idx, confidence, image_path, source, video_time_sec"
_MIGRATE = (("source", "TEXT"), ("video_time_sec", "REAL"))


class EventDB:
    def __init__(self, db_path="logs/events.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(SCHEMA)
        # DB cũ thiếu cột mới -> ALTER TABLE
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(events)")}
        for name, typ in _MIGRATE:
            if name not in have:
                self.conn.execute(f"ALTER TABLE events ADD COLUMN {name} {typ}")
        self.conn.commit()

    def insert(self, track_id, violation_type, frame_idx, confidence=None, image_path=None,
               source=None, video_time_sec=None):
        cur = self.conn.execute(
            "INSERT INTO events (track_id, violation_type, timestamp, frame_idx, confidence, image_path, "
            "source, video_time_sec) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (track_id, violation_type, time.time(), frame_idx, confidence, image_path,
             source, video_time_sec),
        )
        self.conn.commit()
        return cur.lastrowid

    def query(self, vtype=None, limit=100, source=None):
        """vtype khớp cả event gộp (vd 'no_vest' khớp 'no_vest+machinery')."""
        where, params = [], []
        if vtype:
            where.append("instr('+' || violation_type || '+', '+' || ? || '+') > 0")
            params.append(vtype)
        if source:
            where.append("source = ?")
            params.append(source)
        sql = f"SELECT {_COLS} FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self.conn.execute(sql, (*params, limit)).fetchall()
        return [
            {"id": r[0], "track_id": r[1], "violation_type": r[2], "timestamp": r[3],
             "frame_idx": r[4], "confidence": r[5], "image_path": r[6],
             "source": r[7], "video_time_sec": r[8]}
            for r in rows
        ]

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def close(self):
        self.conn.close()
