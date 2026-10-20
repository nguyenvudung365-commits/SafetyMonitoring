import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from storage.events import EventDB
from alerts.telegram import TelegramAlerter


def test_eventdb_insert_query(tmp_path):
    db = EventDB(str(tmp_path / "e.db"))
    rid = db.insert(3, "no_helmet", 42, 0.91, "logs/evidence/x.jpg")
    assert rid >= 1
    assert db.count() == 1
    rows = db.query()
    assert rows[0]["track_id"] == 3
    assert rows[0]["violation_type"] == "no_helmet"
    assert rows[0]["frame_idx"] == 42
    assert rows[0]["image_path"] == "logs/evidence/x.jpg"
    only = db.query(vtype="zone")
    assert only == []
    db.close()


def test_eventdb_source_and_time(tmp_path):
    db = EventDB(str(tmp_path / "e.db"))
    db.insert(1, "zone", 30, 0.8, None, source="clip_a.mp4", video_time_sec=1.0)
    db.insert(2, "zone", 60, 0.8)
    rows = db.query(source="clip_a.mp4")
    assert len(rows) == 1 and rows[0]["video_time_sec"] == 1.0 and rows[0]["source"] == "clip_a.mp4"
    assert db.query()[0]["source"] is None
    db.close()


def test_eventdb_migrates_old_schema(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, track_id INTEGER NOT NULL, "
                "violation_type TEXT NOT NULL, timestamp REAL NOT NULL, frame_idx INTEGER NOT NULL, "
                "confidence REAL, image_path TEXT)")
    con.execute("INSERT INTO events (track_id, violation_type, timestamp, frame_idx) VALUES (1, 'zone', 0, 5)")
    con.commit()
    con.close()
    db = EventDB(path)
    assert db.query()[0]["source"] is None
    db.insert(2, "no_helmet", 9, source="x", video_time_sec=0.3)
    assert db.count() == 2
    db.close()


def test_eventdb_query_matches_merged_types(tmp_path):
    db = EventDB(str(tmp_path / "e.db"))
    db.insert(1, "no_vest+machinery", 1)
    db.insert(2, "no_vest", 2)
    db.insert(3, "no_helmet+no_vest", 3)
    db.insert(4, "zone", 4)
    db.insert(5, "xno_vest", 5)
    assert {r["track_id"] for r in db.query(vtype="no_vest")} == {1, 2, 3}
    assert {r["track_id"] for r in db.query(vtype="machinery")} == {1}
    db.close()


def test_telegram_mock(capsys):
    tg = TelegramAlerter(mock=True)
    assert tg.send("VIOLATION no_helmet track 1", "a.jpg") is True
    out = capsys.readouterr().out
    assert "[TELEGRAM MOCK]" in out
    assert "no_helmet" in out
