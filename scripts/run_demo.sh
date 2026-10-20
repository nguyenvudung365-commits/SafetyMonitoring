#!/usr/bin/env bash
# Chạy demo một lệnh.
#   scripts/run_demo.sh                       -> chạy mock (không cần model), ghi logs_demo/
#   scripts/run_demo.sh VIDEO.mp4 best.pt     -> chạy detector thật trên video
set -e
cd "$(dirname "$0")/.."
OUT=logs_demo
if [ -n "$1" ] && [ -z "$2" ]; then
  echo "Thiếu tham số. Cách dùng: $0 [VIDEO.mp4 best.pt]  (không tham số = chạy mock)" >&2
  exit 1
fi
if [ -n "$1" ] && [ -n "$2" ]; then
  python src/pipeline/stream.py --reset-db --source "$1" --detector "$2" --out-dir "$OUT" --dump-dets "$OUT/dets.jsonl"
else
  python src/pipeline/stream.py --mock --reset-db --out-dir "$OUT"
fi
echo "Xong. Xem $OUT/events.db, $OUT/evidence/, $OUT/run_meta.json"
