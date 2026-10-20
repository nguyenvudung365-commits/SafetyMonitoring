"""Test biên/độc lập, viết từ tiêu chí kế hoạch (không theo cách code chạy)."""
import ast, json, sqlite3, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from rules.safety_rules import (ppe_flags, machinery_violation, bbox_edge_distance, overlap_ratio,
                                load_config)
from pipeline.postprocess import evaluate_person
from storage.events import EventDB
import eval_events as ev
import sweep_rules

P = [0.4, 0.2, 0.6, 0.8]  # h = 0.6, mũ: cy <= 0.4, áo: 0.3 <= cy <= 0.6


def _d(cls, conf, bbox):
    return {"cls": cls, "conf": conf, "bbox": bbox}


HEAD = [0.45, 0.21, 0.55, 0.31]
TORSO = [0.45, 0.40, 0.55, 0.50]


# ---------- ppe_flags ----------
def test_ppe_empty_list():
    assert ppe_flags(P, []) == {"no_helmet": False, "no_vest": False}


def test_ppe_conf_exactly_threshold_counts():
    assert ppe_flags(P, [_d(2, 0.5, HEAD)], conf_th=0.5)["no_helmet"] is True
    assert ppe_flags(P, [_d(2, 0.49, HEAD)], conf_th=0.5)["no_helmet"] is False


def test_ppe_multi_person_each_own_items():
    p2 = [0.7, 0.2, 0.9, 0.8]
    head2 = [0.75, 0.21, 0.85, 0.31]
    dets = [_d(2, 0.9, HEAD), _d(1, 0.9, head2)]
    assert ppe_flags(P, dets)["no_helmet"] is True
    assert ppe_flags(p2, dets)["no_helmet"] is False


def test_ppe_order_independent_all_permutations():
    import itertools
    dets = [_d(1, 0.8, HEAD), _d(2, 0.8, HEAD), _d(4, 0.9, TORSO), _d(3, 0.6, TORSO)]
    res = {tuple(sorted(ppe_flags(P, list(p)).items())) for p in itertools.permutations(dets)}
    assert len(res) == 1
    assert ppe_flags(P, dets) == {"no_helmet": False, "no_vest": True}


def test_ppe_ignores_other_classes():
    assert ppe_flags(P, [_d(0, 0.9, HEAD), _d(5, 0.9, HEAD)]) == {"no_helmet": False, "no_vest": False}


# ---------- geometry ----------
def test_edge_distance_inside_and_corner():
    box = [0.2, 0.2, 0.4, 0.4]
    assert bbox_edge_distance((0.3, 0.3), box) == 0.0
    assert bbox_edge_distance((0.2, 0.3), box) == 0.0  # trên cạnh
    assert bbox_edge_distance((0.5, 0.5), box) == pytest.approx((0.1 ** 2 + 0.1 ** 2) ** 0.5)
    assert bbox_edge_distance((0.3, 0.1), box) == pytest.approx(0.1)


def test_overlap_ratio_degenerate_and_disjoint():
    assert overlap_ratio([0.3, 0.3, 0.3, 0.5], [0, 0, 1, 1]) == 0.0
    assert overlap_ratio([0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.9, 0.9]) == 0.0
    assert overlap_ratio([0.1, 0.1, 0.2, 0.2], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert overlap_ratio([0, 0, 0.2, 0.2], [0.1, 0, 1, 1]) == pytest.approx(0.5)


# ---------- machinery ----------
def test_machinery_no_machines():
    assert machinery_violation(P, []) is False


def test_machinery_h_zero_never_violation_unless_inside_edge():
    flat = [0.4, 0.5, 0.6, 0.5]  # h = 0 -> ngưỡng 0, so sánh < nên False
    assert machinery_violation(flat, [[0.45, 0.45, 0.55, 0.55]]) is False
    assert machinery_violation(flat, [[0.9, 0.9, 0.95, 0.95]]) is False


def test_machinery_negative_height_no_crash():
    inv = [0.4, 0.8, 0.6, 0.2]
    assert machinery_violation(inv, [[0.0, 0.0, 0.1, 0.1]]) is False


def test_machinery_degenerate_machine_bbox_point():
    # máy suy biến thành điểm gần chân: vẫn tính khoảng cách
    assert machinery_violation(P, [[0.5, 0.85, 0.5, 0.85]]) is True  # cách chân 0.05 < 0.3
    assert machinery_violation(P, [[0.5, 1.2, 0.5, 1.2]]) is False  # cách 0.4 > 0.3


def test_machinery_multiple_one_near_is_enough():
    far = [0.0, 0.0, 0.05, 0.05]
    near = [0.62, 0.7, 0.8, 0.9]
    assert machinery_violation(P, [far, near]) is True
    assert machinery_violation(P, [far]) is False


def test_machinery_cabin_one_skipped_but_other_near_counts():
    cabin = [0.0, 0.0, 1.0, 1.0]  # người nằm 100% trong
    other = [0.62, 0.7, 0.8, 0.9]
    assert machinery_violation(P, [cabin]) is False
    assert machinery_violation(P, [cabin, other]) is True


def test_machinery_cabin_overlap_boundary():
    # máy phủ đúng 80% bbox người (theo chiều ngang): >= 0.8 bị bỏ qua
    person = [0.0, 0.0, 1.0, 1.0]
    mach = [0.0, 0.0, 0.8, 1.0]
    assert overlap_ratio(person, mach) == pytest.approx(0.8)
    assert machinery_violation(person, [mach], cabin_overlap=0.8) is False
    assert machinery_violation(person, [mach], cabin_overlap=0.81) is True


def test_machinery_distance_boundary_strict():
    person = [0.4, 0.2, 0.6, 0.8]  # chân (0.5,0.8), h=0.6
    mach = [0.0, 0.0, 0.1, 0.1]
    # máy ngay dưới chân, cách d
    m = [0.45, 0.8 + 0.25, 0.55, 1.15]
    assert machinery_violation(person, [m], k=0.5) is True   # 0.25 < 0.3
    assert machinery_violation(person, [m], k=0.4) is False  # 0.25 > 0.24
    assert machinery_violation(person, [mach], k=0.0) is False


def test_machinery_k_zero_and_large():
    m = [0.62, 0.7, 0.8, 0.9]
    assert machinery_violation(P, [m], k=0.0) is False
    assert machinery_violation(P, [m], k=100) is True


def test_machinery_edge_vs_center_semantics():
    # người cạnh máy rất to: tâm máy xa, cạnh gần -> phải báo
    big = [0.0, 0.0, 0.38, 1.0]
    assert machinery_violation(P, [big]) is True


# ---------- YAML thiếu key / evaluate_person ----------
def _cfg(th):
    return {"thresholds": th}


def test_evaluate_person_missing_machinery_keys_use_defaults():
    near = {"cls": 5, "conf": 0.9, "bbox": [0.62, 0.7, 0.8, 0.9]}
    cabin = {"cls": 5, "conf": 0.9, "bbox": [0.0, 0.0, 1.0, 1.0]}
    cfg = _cfg({"ppe_conf": 0.5})  # thiếu machinery_k, cabin_overlap, zone_person_conf
    assert evaluate_person(P, [], [near], [], cfg) == ["machinery"]
    assert evaluate_person(P, [], [cabin], [], cfg) == []


def test_evaluate_person_yaml_values_respected():
    near = {"cls": 5, "conf": 0.9, "bbox": [0.62, 0.7, 0.8, 0.9]}
    assert evaluate_person(P, [], [near], [], _cfg({"ppe_conf": 0.5, "machinery_k": 0.01})) == []
    cabin = {"cls": 5, "conf": 0.9, "bbox": [0.0, 0.0, 1.0, 1.0]}
    # cabin_overlap > 1 -> không bao giờ coi là cabin; người nằm trong bbox -> khoảng cách 0 -> báo
    assert evaluate_person(P, [], [cabin], [], _cfg({"ppe_conf": 0.5, "cabin_overlap": 1.5})) == ["machinery"]


def test_shipped_configs_have_new_keys():
    for name in ("safety.yaml", "safety_demo.yaml"):
        th = load_config(str(ROOT / "configs" / name))["thresholds"]
        assert 0 < th["machinery_k"] <= 5
        assert 0 < th["cabin_overlap"] <= 1


# ---------- eval_events ----------
def _gt(v, s, e):
    return {"clip": "c", "vtype": v, "start_frame": s, "end_frame": e, "x": None, "y": None}


def test_eval_empty_both():
    r = ev.match_events([], [], 30)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 0)
    s = ev.summarize(r, 900, 30)
    assert s["precision"] != s["precision"] and s["recall"] != s["recall"]  # NaN
    assert s["far_per_h"] == 0.0
    assert isinstance(ev.format_table(s), str)


def test_eval_empty_gt_all_fp():
    r = ev.match_events([{"violation_type": "zone", "frame_idx": 5}], [], 30)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 0)


def test_eval_empty_events_all_fn():
    r = ev.match_events([], [_gt("zone", 0, 10), _gt("fall", 5, 6)], 30)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 2)


@pytest.mark.parametrize("frame,hit", [(100 - 60, True), (100 - 61, False), (200 + 60, True), (200 + 61, False)])
def test_eval_tolerance_boundary_2s(frame, hit):
    r = ev.match_events([{"violation_type": "zone", "frame_idx": frame}], [_gt("zone", 100, 200)], 30, 2.0)
    assert (r["tp"] == 1) is hit


def test_eval_tol_scales_with_fps():
    e = [{"violation_type": "zone", "frame_idx": 100 - 50}]
    assert ev.match_events(e, [_gt("zone", 100, 200)], 25, 2.0)["tp"] == 1
    assert ev.match_events(e, [_gt("zone", 100, 200)], 25, 1.9)["tp"] == 0


def test_eval_merged_type_split_and_wrong_type():
    e = [{"violation_type": "no_vest+machinery", "frame_idx": 100}]
    r = ev.match_events(e, [_gt("no_vest", 90, 110)], 30)
    assert r["tp"] == 1 and r["fp"] == 1  # machinery không có GT
    assert r["per_type"]["machinery"]["fp"] == 1


def test_eval_two_events_two_gts_each_once():
    gt = [_gt("zone", 100, 110), _gt("zone", 120, 130)]
    r = ev.match_events([{"violation_type": "zone", "frame_idx": f} for f in (105, 125)], gt, 30)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)


def test_far_per_hour_values():
    assert ev.far_per_hour(3, 30 * 3600, 30) == pytest.approx(3.0)
    assert ev.far_per_hour(1, 0, 30) != ev.far_per_hour(1, 0, 30)  # NaN


def test_load_gt_comments_blank_and_missing_xy(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("clip,vtype,start_frame,end_frame,x,y\n# comment\n\nc1,zone,10,20,,\nc1,fall,5,9,0.5,0.25\n",
                 encoding="utf-8")
    rows = ev.load_gt(str(p))
    assert len(rows) == 2 and rows[0]["x"] is None and rows[1]["y"] == 0.25 and rows[0]["start_frame"] == 10


def test_load_gt_header_only(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("clip,vtype,start_frame,end_frame,x,y\n", encoding="utf-8")
    assert ev.load_gt(str(p)) == []


def test_load_events_db_old_schema_and_source_filter(tmp_path):
    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, track_id INTEGER NOT NULL, "
                "violation_type TEXT NOT NULL, timestamp REAL NOT NULL, frame_idx INTEGER NOT NULL, "
                "confidence REAL, image_path TEXT)")
    con.execute("INSERT INTO events (track_id, violation_type, timestamp, frame_idx) VALUES (1,'zone',0,7)")
    con.commit(); con.close()
    assert ev.load_events_db(str(p)) == [{"violation_type": "zone", "frame_idx": 7, "source": None}]
    assert ev.load_events_db(str(p), source="x") == []


def test_load_events_db_source_substring(tmp_path):
    db = EventDB(str(tmp_path / "e.db"))
    db.insert(1, "zone", 1, source="/a/clip1.mp4")
    db.insert(2, "zone", 2, source="/a/clip2.mp4")
    db.insert(3, "zone", 3)
    db.close()
    got = ev.load_events_db(str(tmp_path / "e.db"), "clip1")
    assert [e["frame_idx"] for e in got] == [1]


def test_eval_cli_empty_gt_warns(tmp_path):
    db = EventDB(str(tmp_path / "e.db")); db.insert(1, "zone", 5); db.close()
    gt = tmp_path / "gt.csv"; gt.write_text("clip,vtype,start_frame,end_frame,x,y\n", encoding="utf-8")
    out = tmp_path / "o.json"
    r = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "eval_events.py"), "--db", str(tmp_path / "e.db"),
                        "--gt", str(gt), "--fps", "30", "--n-frames", "900", "--out-json", str(out)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "GT" in r.stderr
    assert json.loads(out.read_text(encoding="utf-8").replace("NaN", "null"))["fp"] == 1


def test_eval_cli_missing_db_errors_no_file(tmp_path):
    gt = tmp_path / "gt.csv"; gt.write_text("clip,vtype,start_frame,end_frame,x,y\n", encoding="utf-8")
    missing = tmp_path / "none.db"
    r = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "eval_events.py"), "--db", str(missing),
                        "--gt", str(gt), "--fps", "30", "--n-frames", "900"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode != 0 and not missing.exists()


def test_sweep_cli_non_utf8_console_no_crash(tmp_path):
    import os
    dets = tmp_path / "d.jsonl"
    dets.write_text(json.dumps({"meta": {"fps": 30}}) + "\n" + json.dumps({"frame": 0, "dets": []}) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"; gt.write_text("clip,vtype,start_frame,end_frame,x,y\n", encoding="utf-8")
    out = tmp_path / "s.csv"
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    r = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "sweep_rules.py"), "--dets", str(dets), "--gt", str(gt),
                        "--config", str(ROOT / "configs" / "safety.yaml"), "--out-csv", str(out)],
                       capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode == 0, r.stderr
    assert out.is_file()


# ---------- EventDB ----------
def _old_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, track_id INTEGER NOT NULL, "
                "violation_type TEXT NOT NULL, timestamp REAL NOT NULL, frame_idx INTEGER NOT NULL, "
                "confidence REAL, image_path TEXT)")
    con.execute("INSERT INTO events (track_id, violation_type, timestamp, frame_idx, confidence, image_path) "
                "VALUES (9,'no_helmet',1.0,3,0.7,'a.jpg')")
    con.commit(); con.close()


def test_eventdb_migrates_old_db_keeps_data(tmp_path):
    p = str(tmp_path / "old.db")
    _old_db(p)
    db = EventDB(p)
    rows = db.query()
    assert len(rows) == 1 and rows[0]["track_id"] == 9 and rows[0]["source"] is None \
        and rows[0]["video_time_sec"] is None
    db.insert(1, "zone", 5, source="s", video_time_sec=0.5)
    assert db.count() == 2
    db.close()
    db2 = EventDB(p)  # mở lần 2 không lỗi (idempotent)
    assert db2.count() == 2
    db2.close()


def test_eventdb_query_merged_vtype(tmp_path):
    db = EventDB(str(tmp_path / "e.db"))
    db.insert(1, "no_vest+machinery", 1)
    db.insert(2, "no_vest", 2)
    db.insert(3, "machinery", 3)
    db.insert(4, "no_helmet+no_vest+zone", 4)
    db.insert(5, "fall", 5)
    got = lambda v: sorted(r["track_id"] for r in db.query(vtype=v))
    assert got("no_vest") == [1, 2, 4]
    assert got("machinery") == [1, 3]
    assert got("zone") == [4]
    assert got("no_helmet") == [4]
    assert got("no_vest+machinery") == [1]   # khớp nguyên chuỗi gộp
    assert got("vest") == []                 # không khớp chuỗi con
    assert got("helmet") == []
    assert got("mach") == []
    db.close()


def test_eventdb_query_vtype_with_like_chars_and_quotes(tmp_path):
    db = EventDB(str(tmp_path / "e.db"))
    db.insert(1, "zone", 1)
    assert db.query(vtype="%") == []
    assert db.query(vtype="_one") == []
    assert db.query(vtype="x' OR '1'='1") == []
    assert db.count() == 1
    db.close()


def test_eventdb_query_source_vtype_limit_order(tmp_path):
    db = EventDB(str(tmp_path / "e.db"))
    for i in range(5):
        db.insert(i, "zone", i, source="a" if i % 2 == 0 else "b")
    r = db.query(vtype="zone", source="a", limit=2)
    assert [x["track_id"] for x in r] == [4, 2]  # mới nhất trước
    assert db.query(source="none") == []
    db.close()


def test_api_events_merged_vtype(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import api.app as appmod
    p = str(tmp_path / "e.db")
    db = EventDB(p)
    db.insert(1, "no_vest+machinery", 1)
    db.insert(2, "zone", 2)
    db.close()
    monkeypatch.setattr(appmod, "DB_PATH", p)
    r = appmod.list_events(vtype="machinery", limit=10)
    assert [e["track_id"] for e in r["events"]] == [1] and r["count"] == 2
    assert "source" in r["events"][0] and "video_time_sec" in r["events"][0]


def test_api_events_http_param_validation(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    try:
        from fastapi.testclient import TestClient
    except Exception:
        pytest.skip("no TestClient (httpx missing)")
    import api.app as appmod
    p = str(tmp_path / "e.db")
    db = EventDB(p); db.insert(1, "no_vest+machinery", 1); db.close()
    monkeypatch.setattr(appmod, "DB_PATH", p)
    c = TestClient(appmod.app)
    assert len(c.get("/events", params={"vtype": "no_vest"}).json()["events"]) == 1
    assert c.get("/events", params={"limit": 0}).status_code == 422
    assert c.get("/events", params={"limit": 1001}).status_code == 422
    assert c.get("/evidence/..%2Fx").status_code in (400, 404)


# ---------- sweep_rules ----------
def test_sweep_load_dets_and_deterministic(tmp_path):
    f = tmp_path / "d.jsonl"
    person = {"cls": 0, "conf": 0.9, "bbox": P}
    nh = {"cls": 2, "conf": 0.9, "bbox": HEAD}
    lines = [json.dumps({"meta": {"fps": 30}}), ""]
    lines += [json.dumps({"frame": i, "dets": [person, nh]}) for i in range(40)]
    f.write_text("\n".join(lines), encoding="utf-8")
    meta, frames = sweep_rules.load_dets(str(f))
    assert meta["fps"] == 30 and len(frames) == 40
    cfg = load_config(str(ROOT / "configs" / "safety.yaml"))
    a = sweep_rules.run_rules(cfg, frames, 30)
    b = sweep_rules.run_rules(cfg, frames, 30)
    assert a == b and len(a) == 1 and a[0]["violation_type"] == "no_helmet" and a[0]["frame_idx"] == 14


def test_sweep_load_dets_empty_file(tmp_path):
    f = tmp_path / "d.jsonl"; f.write_text("", encoding="utf-8")
    assert sweep_rules.load_dets(str(f)) == ({}, [])


def test_sweep_grid_defaults_and_product():
    import argparse
    cfg = load_config(str(ROOT / "configs" / "safety.yaml"))
    a = argparse.Namespace(ppe_frames="5,10", ppe_conf="0.3,0.5,0.7", zone_person_conf="", machinery_k="")
    names, grid = sweep_rules.build_grid(cfg, a)
    assert len(grid) == 6 and "machinery_k" in names
    assert all(g["machinery_k"] == cfg["thresholds"]["machinery_k"] for g in grid)
    c2 = sweep_rules.apply(cfg, grid[0])
    assert c2["temporal"]["ppe_frames"] == 5 and cfg["temporal"]["ppe_frames"] == 15
    assert cfg["thresholds"]["ppe_conf"] == 0.5  # không đổi cfg gốc


# ---------- stream mock / run_meta / architecture ----------
def test_stream_mock_writes_source_time_and_meta(tmp_path):
    r = subprocess.run([sys.executable, "-I", str(ROOT / "src" / "pipeline" / "stream.py"), "--mock",
                        "--config", str(ROOT / "configs" / "safety.yaml"), "--out-dir", str(tmp_path),
                        "--reset-db"], capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT),
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stderr[-800:]
    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    for k in ("created", "python", "platform", "commit", "gpu"):
        assert k in meta
    db = EventDB(str(tmp_path / "events.db"))
    rows = db.query()
    assert rows, "mock phải sinh ít nhất 1 event"
    assert all(x["source"] == "mock" and x["video_time_sec"] is not None for x in rows)
    assert rows[0]["video_time_sec"] == pytest.approx(rows[0]["frame_idx"] / 30)
    db.close()


def test_write_run_meta_no_git(tmp_path):
    from pipeline.run_meta import write_run_meta
    p = write_run_meta(tmp_path / "sub", mode="x", n=3)
    m = json.loads(Path(p).read_text(encoding="utf-8"))
    assert m["mode"] == "x" and m["n"] == 3


def test_architecture_rules_module_independent():
    """src/rules không import module nghiệp vụ khác (storage, api, alerts, pipeline, tracking)."""
    forbidden = {"storage", "api", "alerts", "pipeline", "tracking"}
    for f in (ROOT / "src" / "rules").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in forbidden, f"{f.name} imports {n}"


def test_architecture_storage_and_tools_no_heavy_deps():
    for f in [ROOT / "src" / "storage" / "events.py", ROOT / "tools" / "eval_events.py",
              ROOT / "tools" / "sweep_rules.py", ROOT / "src" / "rules" / "safety_rules.py"]:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else                 [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for n in names:
                assert n.split(".")[0] not in {"torch", "ultralytics", "cv2"}, f"{f.name}: {n}"


def test_requirements_pins_and_scripts_exist():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for pkg in ("ultralytics", "pyyaml", "numpy", "fastapi", "pytest", "opencv-python"):
        assert any(l.lower().startswith(pkg + "==") for l in req.splitlines()), pkg
    assert (ROOT / "scripts" / "run_demo.sh").is_file() and (ROOT / "scripts" / "run_demo.bat").is_file()
