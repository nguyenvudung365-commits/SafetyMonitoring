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


def test_telegram_mock(capsys):
    tg = TelegramAlerter(mock=True)
    assert tg.send("VIOLATION no_helmet track 1", "a.jpg") is True
    out = capsys.readouterr().out
    assert "[TELEGRAM MOCK]" in out
    assert "no_helmet" in out
