"""Ghi run_meta.json cho mỗi lần chạy pipeline (tái lập kết quả). Không import torch/ultralytics ở mức module."""
import json, platform, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit(repo=REPO_ROOT):
    """Commit hash hiện tại (+ '-dirty' nếu có thay đổi chưa commit); None nếu không phải repo git."""
    try:
        h = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True,
                               text=True, timeout=10, check=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return None


def gpu_name():
    """Tên GPU nếu torch đã được import bởi tiến trình (chỉ khi chạy detector thật); ngược lại None."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def write_run_meta(out_dir, **info):
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu_name(),
        **info,
    }
    path = Path(out_dir) / "run_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(path)
