"""SQLite event log — schema tuần 7: id, track_id, violation_type, timestamp, frame_idx, confidence, image_path."""
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


class EventDB:
    def __init__(self, db_path="logs/events.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def insert(self, track_id, violation_type, frame_idx, confidence=None, image_path=None):
        cur = self.conn.execute(
            "INSERT INTO events (track_id, violation_type, timestamp, frame_idx, confidence, image_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (track_id, violation_type, time.time(), frame_idx, confidence, image_path),
        )
        self.conn.commit()
        return cur.lastrowid

    def query(self, vtype=None, limit=100):
        if vtype:
            rows = self.conn.execute(
                "SELECT id, track_id, violation_type, timestamp, frame_idx, confidence, image_path "
                "FROM events WHERE violation_type=? ORDER BY id DESC LIMIT ?",
                (vtype, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, track_id, violation_type, timestamp, frame_idx, confidence, image_path "
                "FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": r[0], "track_id": r[1], "violation_type": r[2], "timestamp": r[3],
             "frame_idx": r[4], "confidence": r[5], "image_path": r[6]}
            for r in rows
        ]

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def close(self):
        self.conn.close()
