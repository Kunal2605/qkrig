#!/usr/bin/env bash
# =============================================================================
# run_qkrig_daily.sh
#
# Daily kriging runner. Defaults to YESTERDAY when called with no arguments.
#
# Usage:
#   bash Scripts/run_qkrig_daily.sh                                 # yesterday
#   bash Scripts/run_qkrig_daily.sh --date 2024-09-26               # one day
#   bash Scripts/run_qkrig_daily.sh --start-date 2024-09-25 --end-date 2024-09-28
#
# Docker:
#   docker run -e DATE=2024-09-26 ... --entrypoint bash IMAGE Scripts/run_qkrig_daily.sh
#   docker run -e START_DATE=2024-09-25 -e END_DATE=2024-09-28 ... --entrypoint bash IMAGE Scripts/run_qkrig_daily.sh
# =============================================================================
set -euo pipefail

CONFIG="${CONFIG:-configs/usgsgaugekrig.yaml}"
MAX_PROCS="${MAX_PROCS:-16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---- CLI flags into separate vars so they don't clobber env vars ----
CLI_DATE=""
CLI_START_DATE=""
CLI_END_DATE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --date)        CLI_DATE="$2";        shift 2 ;;
        --start-date)  CLI_START_DATE="$2";  shift 2 ;;
        --end-date)    CLI_END_DATE="$2";    shift 2 ;;
        --config)      CONFIG="$2";          shift 2 ;;
        --max-procs)   MAX_PROCS="$2";       shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# CLI takes precedence over env vars
DATE="${CLI_DATE:-${DATE:-}}"
START_DATE="${CLI_START_DATE:-${START_DATE:-}}"
END_DATE="${CLI_END_DATE:-${END_DATE:-}}"

# Default: yesterday (only when nothing else is set)
if [[ -z "$DATE" && -z "$START_DATE" && -z "$END_DATE" ]]; then
    DATE=$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)
fi

# Resolve start/end
if [[ -n "$DATE" ]]; then
    SD="$DATE"
    ED="$DATE"
elif [[ -n "$START_DATE" && -n "$END_DATE" ]]; then
    SD="$START_DATE"
    ED="$END_DATE"
else
    echo "ERROR: provide --date, OR both --start-date and --end-date (env vars work too)."
    exit 1
fi

# ---- Validate ----
"$PYTHON_BIN" - "$SD" "$ED" <<'PY' || exit 1
import sys
from datetime import datetime, date

sd_str, ed_str = sys.argv[1], sys.argv[2]

try:
    sd = datetime.strptime(sd_str, "%Y-%m-%d").date()
except ValueError:
    sys.exit(f"ERROR: invalid start date '{sd_str}'. Expected YYYY-MM-DD.")
try:
    ed = datetime.strptime(ed_str, "%Y-%m-%d").date()
except ValueError:
    sys.exit(f"ERROR: invalid end date '{ed_str}'. Expected YYYY-MM-DD.")

today = date.today()
if sd > today:
    sys.exit(f"ERROR: start date {sd_str} is in the future (today is {today}).")
if ed < sd:
    sys.exit(f"ERROR: end date {ed_str} is before start date {sd_str}.")
if sd.year < 1900:
    sys.exit(f"ERROR: start date {sd_str} is unreasonably old.")
PY

# ---- Setup ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export PYTHON_BIN MAX_PROCS

echo "============================================="
echo " qkrig Daily Runner"
echo "============================================="
echo " Config     : $CONFIG"
echo " Start date : $SD"
echo " End date   : $ED"
echo " Max procs  : $MAX_PROCS"
echo "============================================="
echo ""

# Hand off to the existing dispatcher (which calls run_usgs_krig_day.py per date)
exec bash "$SCRIPT_DIR/dispatch_usgs_krig_range.sh" "$CONFIG" "$SD" "$ED"
