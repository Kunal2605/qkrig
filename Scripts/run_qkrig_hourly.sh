#!/usr/bin/env bash
# =============================================================================
# run_qkrig_hourly.sh
#
# Fetch USGS IV data and run hourly kriging → NetCDF exports.
#
# Defaults to YESTERDAY (all 24 hours) when called with no arguments.
#
# Usage:
#   bash Scripts/run_qkrig_hourly.sh                         # yesterday, all 24 h
#   bash Scripts/run_qkrig_hourly.sh --date 2024-09-26       # specific date, all 24 h
#   bash Scripts/run_qkrig_hourly.sh --date 2024-09-26 --hour 04          # single hour
#   bash Scripts/run_qkrig_hourly.sh --date 2024-09-26 --start-hour 00 --end-hour 11  # range
#
# Docker usage (env-var driven):
#   docker run -e DATE=2024-09-26 -e HOUR=04 ...
#   docker run -e DATE=2024-09-26 -e START_HOUR=00 -e END_HOUR=11 ...
# =============================================================================
set -euo pipefail

# ---- Config (override via env or CLI flags below) ----
CONFIG="${CONFIG:-configs/usgsgaugekrig.yaml}"
KV_DIR="${KV_DIR:-usgs_hourly_retrieval_logs}"
MAX_PROCS="${MAX_PROCS:-16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---- Parse CLI flags into separate vars so they don't clobber env vars ----
CLI_DATE=""
CLI_HOUR=""
CLI_START_HOUR=""
CLI_END_HOUR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --date)        CLI_DATE="$2";        shift 2 ;;
        --hour)        CLI_HOUR="$2";        shift 2 ;;
        --start-hour)  CLI_START_HOUR="$2";  shift 2 ;;
        --end-hour)    CLI_END_HOUR="$2";    shift 2 ;;
        --config)      CONFIG="$2";          shift 2 ;;
        --kv-dir)      KV_DIR="$2";          shift 2 ;;
        --max-procs)   MAX_PROCS="$2";       shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---- Final values: CLI flags take precedence over env vars ----
DATE="${CLI_DATE:-${DATE:-}}"
HOUR="${CLI_HOUR:-${HOUR:-}}"
START_HOUR="${CLI_START_HOUR:-${START_HOUR:-}}"
END_HOUR="${CLI_END_HOUR:-${END_HOUR:-}}"

# ---- Default: yesterday (computed before validation; always valid) ----
if [[ -z "$DATE" ]]; then
    DATE=$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)
fi

# ---- Validate inputs ----
PYTHON_BIN_FOR_VALIDATE="${PYTHON_BIN:-python3}"
"$PYTHON_BIN_FOR_VALIDATE" - "$DATE" "$HOUR" "$START_HOUR" "$END_HOUR" <<'PY' || exit 1
import sys
from datetime import datetime, date

date_str, hour, start_hour, end_hour = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# --- Date format & calendar validity ---
try:
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
except ValueError:
    sys.exit(f"ERROR: invalid --date '{date_str}'. Expected YYYY-MM-DD (e.g. 2024-09-26).")

today = date.today()
if dt > today:
    sys.exit(f"ERROR: --date {date_str} is in the future (today is {today}). USGS data is not available.")
if dt.year < 1900:
    sys.exit(f"ERROR: --date {date_str} is unreasonably old.")

# --- Hour parsing helper ---
def parse_hour(s, flag):
    if not s:
        return None
    try:
        v = int(s)
    except ValueError:
        sys.exit(f"ERROR: --{flag} must be an integer 0-23, got '{s}'.")
    if v < 0 or v > 23:
        sys.exit(f"ERROR: --{flag} must be 0-23, got {v}.")
    return v

h_single = parse_hour(hour,       "hour")
h_start  = parse_hour(start_hour, "start-hour")
h_end    = parse_hour(end_hour,   "end-hour")

# --- Mutual exclusion / completeness ---
if h_single is not None and (h_start is not None or h_end is not None):
    sys.exit("ERROR: --hour cannot be combined with --start-hour / --end-hour.")
if (h_start is None) != (h_end is None):
    sys.exit("ERROR: --start-hour and --end-hour must be specified together.")
if h_start is not None and h_end is not None and h_start > h_end:
    sys.exit(f"ERROR: --start-hour ({h_start:02d}) is greater than --end-hour ({h_end:02d}).")
PY

# ---- Resolve hour range ----
if [[ -n "$HOUR" ]]; then
    # Single hour mode
    H_START=$(printf "%02d" "$((10#$HOUR))")
    H_END="$H_START"
elif [[ -n "$START_HOUR" && -n "$END_HOUR" ]]; then
    H_START=$(printf "%02d" "$((10#$START_HOUR))")
    H_END=$(printf "%02d" "$((10#$END_HOUR))")
else
    # Full day
    H_START="00"
    H_END="23"
fi

# ---- Setup ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

mkdir -p "$KV_DIR"

echo "============================================="
echo " qkrig Hourly Runner"
echo "============================================="
echo " Config     : $CONFIG"
echo " Date       : $DATE"
echo " Hours      : $H_START – $H_END"
echo " KV dir     : $KV_DIR"
echo " Max procs  : $MAX_PROCS"
echo "============================================="
echo ""

# ---- Build list of hours to process ----
HOURS_LIST=$(
    START="$H_START" END="$H_END" "$PYTHON_BIN" - <<'PY'
import os
start = int(os.environ["START"])
end   = int(os.environ["END"])
for h in range(start, end + 1):
    print(f"{h:02d}")
PY
)

N_HOURS=$(echo "$HOURS_LIST" | wc -l | tr -d ' ')
echo "Processing $N_HOURS hour(s) for $DATE"
echo ""

# ---- Run hour 00 first if in range (fetches & caches all 24h of IV data) ----
run_one_hour() {
    local h="$1"
    "$PYTHON_BIN" Scripts/run_usgs_krig_hour.py \
        --config "$CONFIG" \
        --kv-dir "$KV_DIR" \
        --year  "${DATE:0:4}" \
        --month "${DATE:5:2}" \
        --day   "${DATE:8:2}" \
        --hour  "$((10#$h))"
}
export -f run_one_hour
export CONFIG KV_DIR PYTHON_BIN DATE

# If hour 00 is in range run it first (serial) to warm the IV cache for the day
if echo "$HOURS_LIST" | grep -q "^00$"; then
    echo ">>> [00] Fetching IV data and warming KV cache..."
    run_one_hour "00"
    REMAINING=$(echo "$HOURS_LIST" | grep -v "^00$" || true)
else
    REMAINING="$HOURS_LIST"
fi

# Run remaining hours in parallel
if [[ -n "$REMAINING" ]]; then
    N_REM=$(echo "$REMAINING" | wc -l | tr -d ' ')
    echo ">>> Running $N_REM remaining hour(s) in parallel (max $MAX_PROCS)..."
    if command -v parallel >/dev/null 2>&1; then
        echo "$REMAINING" | parallel -j "$MAX_PROCS" --halt now,fail=1 run_one_hour {}
    else
        echo "$REMAINING" | xargs -P "$MAX_PROCS" -I{} bash -c 'run_one_hour "$@"' _ {}
    fi
fi

echo ""
echo "============================================="
echo " Done. NC files written to exports directory."
echo "============================================="
