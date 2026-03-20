#!/bin/bash
set -e

# =============================================================================
# qkrig_ts_range_hourly.sh
#
# Dispatch qkrig_ts_hourly.py across a date range (all 24 hours per day).
# Populates per-catchment CSVs from hourly kriged .npz exports.
#
# Usage:
#   bash Scripts/qkrig_ts_range_hourly.sh
#
# Override defaults via env vars:
#   START_DATE=2024-09-24 END_DATE=2024-09-28 OUT_DIR=/my/output \
#   bash Scripts/qkrig_ts_range_hourly.sh
# =============================================================================

START_DATE="${START_DATE:-2024-09-24}"
END_DATE="${END_DATE:-2024-09-28}"
OUT_DIR="${OUT_DIR:-/mnt/disk1/qkrig/subdaily/}"
JOBS="${JOBS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT_DIR"

echo "============================================="
echo " Hourly Hydrofabric TS Extraction"
echo "============================================="
echo " Start  : $START_DATE"
echo " End    : $END_DATE"
echo " Output : $OUT_DIR"
echo " Jobs   : $JOBS"
echo "============================================="

# Generate list of hours (YYYY-MM-DD_HH) across the full date range
HOURS=$("$PYTHON_BIN" - <<PY
from datetime import datetime, timedelta
start = datetime.strptime("$START_DATE", "%Y-%m-%d")
end   = datetime.strptime("$END_DATE", "%Y-%m-%d").replace(hour=23)
dt = start
while dt <= end:
    print(dt.strftime("%Y-%m-%d_%H"))
    dt += timedelta(hours=1)
PY
)

N_HOURS=$(echo "$HOURS" | wc -l | tr -d ' ')
echo "Dispatching $N_HOURS hours across $JOBS parallel workers..."

# Run extraction in parallel
echo "$HOURS" | xargs -n 1 -P "$JOBS" -I{} \
    "$PYTHON_BIN" "$SCRIPT_DIR/qkrig_ts_hourly.py" {} "$OUT_DIR"

echo ""
echo "✅ All hourly catchment CSVs updated in $OUT_DIR"
