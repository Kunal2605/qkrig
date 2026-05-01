#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# dispatch_usgs_krig_range_subdaily.sh
#
# Hourly USGS kriging dispatcher (mirrors dispatch_usgs_krig_range.sh).
#
# Processes ONE DAY AT A TIME to avoid redundant API calls:
#   1. For each day: run hour 00 first (fetches IV data and caches all 24 hours)
#   2. Then run hours 01–23 in parallel (all hit the cache)
#   3. Repeat for the next day
#
# Usage:
#   bash Scripts/dispatch_usgs_krig_range_subdaily.sh [CONFIG] [START] [END]
#
# Examples:
#   bash Scripts/dispatch_usgs_krig_range_subdaily.sh \
#       configs/usgsgaugekrig.yaml 2023-05-01 2023-05-03
#
#   # With pre-downloaded .rdb files (skip NWIS fetch):
#   SKIP_RDB=0 RAW_DIR=/path/to/rdb METADATA=/path/to/meta.csv \
#   bash Scripts/dispatch_usgs_krig_range_subdaily.sh ...
# =============================================================================

# ---- user knobs (same pattern as daily dispatch script) ----
# Positional args take priority; env vars are the fallback (Docker-friendly).
# Docker usage: docker run -e START_DATE=2024-01-01 -e END_DATE=2024-01-31 <image>
CONFIG="${1:-${CONFIG:-configs/usgsgaugekrig.yaml}}"
START_DATE="${2:-${START_DATE:-2023-05-01}}"
END_DATE="${3:-${END_DATE:-2023-05-31}}"
PLOT_CONFIG_OVERRIDE="${4:-}"    # optional; leave empty to use config's plot_config
MAX_PROCS="${MAX_PROCS:-16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# KV directory (where hourly .kv.txt files are read/written)
KV_DIR="${KV_DIR:-usgs_hourly_retrieval_logs}"

# Optional .rdb conversion (Stage 0). Set SKIP_RDB=0 to enable.
SKIP_RDB="${SKIP_RDB:-1}"
RAW_DIR="${RAW_DIR:-}"
METADATA="${METADATA:-}"
MIN_READINGS="${MIN_READINGS:-2}"

# Skip kriging (Stage 1). Set SKIP_KRIG=1 to only do KV conversion.
SKIP_KRIG="${SKIP_KRIG:-0}"

# Ensure src/ modules (interpolation, core, loaders) are importable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../src${PYTHONPATH:+:$PYTHONPATH}"

# Avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

echo "============================================="
echo " Subdaily USGS Krig Dispatcher"
echo "============================================="
echo " Config     : $CONFIG"
echo " Start date : $START_DATE"
echo " End date   : $END_DATE"
echo " KV dir     : $KV_DIR"
echo " Max procs  : $MAX_PROCS"
echo "============================================="

# ---- Stage 0 (optional): Convert pre-downloaded .rdb → hourly KV ----
if [[ "$SKIP_RDB" != "1" ]]; then
  if [[ -z "$RAW_DIR" || -z "$METADATA" ]]; then
    echo "ERROR: SKIP_RDB=0 requires RAW_DIR and METADATA to be set."
    exit 1
  fi
  echo ""
  echo ">>> Stage 0: Converting .rdb files → hourly KV files..."

  "$PYTHON_BIN" Scripts/usgs_raw_to_hourly_kv.py \
    --raw-dir "$RAW_DIR" \
    --metadata "$METADATA" \
    --logs-dir "$KV_DIR" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --min-readings "$MIN_READINGS"

  echo ">>> Stage 0 complete."
else
  echo ""
  echo ">>> Stage 0 skipped (no .rdb conversion)."
  echo "    run_usgs_krig_hour.py will fetch from NWIS if no KV cache exists."
fi

# ---- Stage 1: Run hourly kriging (day-by-day) ----
if [[ "$SKIP_KRIG" != "1" ]]; then
  echo ""
  echo ">>> Stage 1: Running hourly kriging..."

  # Build DATE list (not hour list — we process one day at a time)
  DATE_LIST="$(
    SDATE="$START_DATE" EDATE="$END_DATE" \
    "$PYTHON_BIN" - <<'PY'
from datetime import datetime, timedelta
import os, sys
start = datetime.strptime(os.environ["SDATE"], "%Y-%m-%d").date()
end   = datetime.strptime(os.environ["EDATE"], "%Y-%m-%d").date()
if end < start:
    sys.exit("END_DATE must be >= START_DATE")
d = start
while d <= end:
    print(d.isoformat())
    d += timedelta(days=1)
PY
  )"

  N_DAYS=$(echo "$DATE_LIST" | wc -l | tr -d ' ')
  echo "Processing $N_DAYS day(s), 24 hours each."
  echo ""

  # Function to run one hour
  run_one_hour() {
    local hr_str="$1"           # e.g. 2023-05-15_14
    local y="${hr_str:0:4}"
    local m="${hr_str:5:2}"
    local d="${hr_str:8:2}"
    local h="${hr_str:11:2}"

    if [[ -n "$PLOT_CONFIG_OVERRIDE" ]]; then
      "$PYTHON_BIN" Scripts/run_usgs_krig_hour.py \
        --config "$CONFIG" --kv-dir "$KV_DIR" \
        --year "$y" --month "$m" --day "$d" --hour "$h" \
        --plot-config "$PLOT_CONFIG_OVERRIDE"
    else
      "$PYTHON_BIN" Scripts/run_usgs_krig_hour.py \
        --config "$CONFIG" --kv-dir "$KV_DIR" \
        --year "$y" --month "$m" --day "$d" --hour "$h"
    fi
  }

  export -f run_one_hour
  export CONFIG PLOT_CONFIG_OVERRIDE PYTHON_BIN KV_DIR

  # Process one day at a time:
  #   1. Run hour 00 FIRST (fetches IV data, caches all 24 hours)
  #   2. Run hours 01–23 in PARALLEL (all hit the cache)
  for DAY in $DATE_LIST; do
    echo "--- Day: $DAY ---"

    # Step 1: Fetch + cache by running hour 00 sequentially
    echo "  [fetch] Running hour 00 (fetches & caches all 24 hours)..."
    run_one_hour "${DAY}_00"

    # Step 2: Run remaining hours 01–23 in parallel
    REMAINING_HOURS="$(
      for h in $(seq -w 1 23); do
        echo "${DAY}_${h}"
      done
    )"

    echo "  [krig]  Running hours 01–23 in parallel (max $MAX_PROCS)..."
    if command -v parallel >/dev/null 2>&1; then
      echo "$REMAINING_HOURS" | parallel -j "$MAX_PROCS" --halt now,fail=1 run_one_hour {}
    else
      echo "$REMAINING_HOURS" | xargs -n1 -P "$MAX_PROCS" -I{} bash -c 'run_one_hour "$@"' _ {}
    fi

    echo "  ✓ Day $DAY complete."
    echo ""
  done

  echo ">>> Stage 1 complete."
else
  echo ""
  echo ">>> Stage 1 skipped (SKIP_KRIG=1)."
fi

echo ""
echo "============================================="
echo " Pipeline finished."
echo " KV cache : $KV_DIR"
echo "============================================="
