"""CLI sweep_rules/eval_events: console không phải UTF-8, --cabin-overlap, mã thoát 2 khi thiếu DB."""
import csv, json, os, sqlite3, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
P = [0.4, 0.2, 0.6, 0.8]
HEAD = [0.45, 0.21, 0.55, 0.31]
TORSO = [0.45, 0.40, 0.55, 0.50]
MACH = [0.4, 0.1, 0.57, 0.9]  # người nằm 85% trong máy -> cabin khi cabin_overlap <= 0.85


def _d(c, b):
    return {"cls": c, "conf": 0.9, "bbox": b}


def _run(script, args, enc):
    env = {**os.environ, "PYTHONIOENCODING": enc}
    return subprocess.run([sys.executable, "-I", str(ROOT / "tools" / script), *map(str, args)],
                          capture_output=True, env=env)


@pytest.fixture
def data(tmp_path):
    lines = [json.dumps({"meta": {"fps": 30}})]
    for i in range(60):
        lines.append(json.dumps({"frame": i, "dets": [_d(0, P), _d(2, HEAD), _d(4, TORSO), _d(5, MACH)]}))
    (tmp_path / "d.jsonl").write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / "gt.csv").write_text(
        "clip,vtype,start_frame,end_frame,x,y\nclip\u0110,no_helmet,10,50,,\n"
        "clip\u0110,machinery,10,50,,\nclip\u0110,no_vest,10,50,,\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("enc", ["cp1252", "cp932", "utf-8"])
def test_sweep_cabin_overlap_scan_writes_csv(data, enc):
    out = data / "s.csv"
    r = _run("sweep_rules.py", ["--dets", data / "d.jsonl", "--gt", data / "gt.csv", "--clip", "clip\u0110",
                                "--config", ROOT / "configs" / "safety.yaml",
                                "--cabin-overlap", "0.5,0.8,0.9,1.0", "--out-csv", out], enc)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    assert b"UnicodeEncodeError" not in r.stderr
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [x["cabin_overlap"] for x in rows] == ["0.5", "0.8", "0.9", "1.0"]
    # người nằm 85% trong máy: ngưỡng <= 0.85 -> bỏ qua máy; > 0.85 -> báo machinery
    assert [int(x["events"]) for x in rows] == [2, 2, 3, 3]
    assert [int(x["fn"]) for x in rows] == [1, 1, 0, 0]
    assert all(int(x["tp"]) + int(x["fp"]) == int(x["events"]) for x in rows)  # events cùng đơn vị với tp/fp


@pytest.mark.parametrize("enc", ["cp1252", "cp932"])
def test_eval_cli_merged_event_non_utf8(data, enc):
    con = sqlite3.connect(data / "e.db")
    con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, violation_type TEXT, frame_idx INT, source TEXT)")
    con.execute("INSERT INTO events(violation_type,frame_idx,source) VALUES ('no_helmet+no_vest',14,'x')")
    con.commit(); con.close()
    out = data / "o.json"
    r = _run("eval_events.py", ["--db", data / "e.db", "--gt", data / "gt.csv", "--clip", "clip\u0110",
                                "--fps", 30, "--n-frames", 60, "--out-json", out], enc)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    s = json.loads(out.read_text(encoding="utf-8").replace("NaN", "null"))
    assert (s["tp"], s["fp"], s["fn"]) == (2, 0, 1)


@pytest.mark.parametrize("enc", ["cp1252", "cp932"])
def test_eval_cli_missing_db_exit_2_no_traceback(data, enc):
    missing = data / "nope.db"
    out = data / "x.json"
    r = _run("eval_events.py", ["--db", missing, "--gt", data / "gt.csv", "--fps", 30, "--n-frames", 60,
                                "--out-json", out], enc)
    assert r.returncode == 2
    assert b"Traceback" not in r.stderr
    assert not missing.exists() and not out.exists()


def test_eval_gt_sorted_by_start_frame_regardless_of_csv_order():
    sys.path.insert(0, str(ROOT / "tools"))
    import eval_events as ev
    gt = [{"clip": "", "vtype": "zone", "start_frame": 100, "end_frame": 110, "x": None, "y": None},
          {"clip": "", "vtype": "zone", "start_frame": 0, "end_frame": 10, "x": None, "y": None}]
    # event f=5 chỉ thuộc GT sớm (0-10) khi tol=0; không được ăn nhầm GT sau
    r = ev.match_events([{"violation_type": "zone", "frame_idx": 5}], gt, 30, tol_sec=0)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 1)
    # tol rộng: cả 2 GT khớp được; event sớm phải lấy GT sớm, event muộn lấy GT còn lại
    r = ev.match_events([{"violation_type": "zone", "frame_idx": 50},
                         {"violation_type": "zone", "frame_idx": 105}], gt, 30, tol_sec=2)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)
