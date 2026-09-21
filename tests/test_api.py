import sys
from pathlib import Path
import pytest
fastapi = pytest.importorskip("fastapi")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from storage.events import EventDB
import api.app as appmod


def test_api_events(tmp_path, monkeypatch):
    db_path = str(tmp_path / "e.db")
    db = EventDB(db_path)
    db.insert(1, "no_helmet", 10, 0.9, "a.jpg")
    db.insert(2, "zone", 20, 0.8, "b.jpg")
    db.close()
    monkeypatch.setattr(appmod, "DB_PATH", db_path)
    assert appmod.health() == {"status": "ok"}
    r = appmod.list_events(vtype=None, limit=100)
    assert r["count"] == 2 and len(r["events"]) == 2
    r = appmod.list_events(vtype="zone", limit=100)
    assert len(r["events"]) == 1 and r["events"][0]["violation_type"] == "zone"
    assert appmod.count_events() == {"count": 2}
    with pytest.raises(Exception):
        appmod.get_evidence("missing.jpg")
