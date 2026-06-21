#!/usr/bin/env bash
# ParkSight BLR — one-command launch.
# Rebuilds analytics from the raw CSV (if needed) and serves the dashboard.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8765}"

if [ ! -f web/data/zones.json ]; then
  echo "› Building analytics from raw violation data…"
  python3 pipeline/build_data.py
fi

echo "› ParkSight BLR running at  http://localhost:${PORT}"
echo "  (Ctrl+C to stop)"
cd web && python3 -m http.server "$PORT"
