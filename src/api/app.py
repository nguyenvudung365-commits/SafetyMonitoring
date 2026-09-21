"""REST API đọc sự kiện vi phạm từ SQLite + phục vụ ảnh minh chứng."""
from pathlib import Path
import sys
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage.events import EventDB

DB_PATH = "logs/events.db"

app = FastAPI(title="SafetyMonitoring API")


def get_db():
    return EventDB(DB_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/events")
def list_events(
    vtype: str | None = Query(default=None, description="Lọc theo loại vi phạm"),
    limit: int = Query(default=100, ge=1, le=1000),
):
    db = get_db()
    try:
        return {"count": db.count(), "events": db.query(vtype=vtype, limit=limit)}
    finally:
        db.close()


@app.get("/events/count")
def count_events():
    db = get_db()
    try:
        return {"count": db.count()}
    finally:
        db.close()


@app.get("/evidence/{filename}")
def get_evidence(filename: str):
    safe = Path(filename).name
    if safe != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    path = Path("logs/evidence") / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(path))
