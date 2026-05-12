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
#   bash Scripts/run_qkrig_hourly.sh --date 2024-09-26 --start-hour 00 --end-hour 11  # hour range
#   bash Scripts/run_qkrig_hourly.sh --start-date 2024-09-25 --end-date 2024-09-28    # date range, all 24 h each
#
# Docker usage (env-var driven):
#   docker run -e DATE=2024-09-26 -e HOUR=04 ...
#   docker run -e DATE=2024-09-26 -e START_HOUR=00 -e END_HOUR=11 ...
#   docker run -e START_DATE=2024-09-25 -e END_DATE=2024-09-28 ...
# =============================================================================
set -euo pipefail

# ---- Config (override via env or CLI flags below) ----
CONFIG="${CONFIG:-configs/usgsgaugekrig.yaml}"
KV_DIR="${KV_DIR:-usgs_hourly_retrieval_logs}"
MAX_PROCS="${MAX_PROCS:-16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Plot config + extraction (used by the post-kriging stage). All optional.
# PLOT_CONFIG is empty by default so local users keep the YAML's own
# plot_config (notebook-style show, no saves). The Dockerfile sets
# PLOT_CONFIG=../configs/plot_config_docker.yaml so containerized runs flip
# to save-mode automatically. Path is interpreted relative to the kriging
# YAML's directory, matching that YAML's own `plot_config:` convention.
PLOT_CONFIG="${PLOT_CONFIG:-}"
EXPORTS_DIR="${EXPORTS_DIR:-./exports}"
GPKG_PATH="${GPKG_PATH:-/qkrig/hydrofabric/conus_nextgen.gpkg}"
CATCHMENT_OUT_DIR="${CATCHMENT_OUT_DIR:-./exports/catchment_csv}"
SKIP_EXTRACTION="${SKIP_EXTRACTION:-0}"
SKIP_GIF="${SKIP_GIF:-0}"
GIF_FRAME_MS="${GIF_FRAME_MS:-333}"
GIF_MAX_WIDTH="${GIF_MAX_WIDTH:-1600}"

# ---- Parse CLI flags into separate vars so they don't clobber env vars ----
CLI_DATE=""
CLI_START_DATE=""
CLI_END_DATE=""
CLI_HOUR=""
CLI_START_HOUR=""
CLI_END_HOUR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --date)        CLI_DATE="$2";        shift 2 ;;
        --start-date)  CLI_START_DATE="$2";  shift 2 ;;
        --end-date)    CLI_END_DATE="$2";    shift 2 ;;
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
START_DATE="${CLI_START_DATE:-${START_DATE:-}}"
END_DATE="${CLI_END_DATE:-${END_DATE:-}}"
HOUR="${CLI_HOUR:-${HOUR:-}}"
START_HOUR="${CLI_START_HOUR:-${START_HOUR:-}}"
END_HOUR="${CLI_END_HOUR:-${END_HOUR:-}}"

# Date-range mode: iterate the range and re-enter the script per day. Children
# clear START_DATE/END_DATE to fall through to the single-day path.
if [[ -n "$START_DATE" || -n "$END_DATE" ]]; then
    if [[ -z "$START_DATE" || -z "$END_DATE" ]]; then
        echo "ERROR: --start-date and --end-date must be specified together."
        exit 1
    fi
    if [[ -n "$DATE" ]]; then
        echo "ERROR: --date cannot be combined with --start-date / --end-date."
        exit 1
    fi
    "${PYTHON_BIN:-python3}" - "$START_DATE" "$END_DATE" <<'PY' || exit 1
import sys
from datetime import datetime, date
sd_str, ed_str = sys.argv[1], sys.argv[2]
try:
    sd = datetime.strptime(sd_str, "%Y-%m-%d").date()
except ValueError:
    sys.exit(f"ERROR: invalid --start-date '{sd_str}'. Expected YYYY-MM-DD.")
try:
    ed = datetime.strptime(ed_str, "%Y-%m-%d").date()
except ValueError:
    sys.exit(f"ERROR: invalid --end-date '{ed_str}'. Expected YYYY-MM-DD.")
today = date.today()
if sd > today: sys.exit(f"ERROR: start date {sd_str} is in the future (today is {today}).")
if ed < sd:   sys.exit(f"ERROR: end date {ed_str} is before start date {sd_str}.")
if sd.year < 1900: sys.exit(f"ERROR: start date {sd_str} is unreasonably old.")
PY

    # One bulk fetch covers the whole range so per-day Stage 0 inside the
    # recursive dispatch finds KV present and skips. ~30× fewer NWIS requests
    # vs per-day fetching for a month-long run.
    RANGE_NEED_FETCH=0
    d_check="$START_DATE"
    while [[ "$d_check" < "$END_DATE" || "$d_check" == "$END_DATE" ]]; do
        for hh in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do
            if [[ ! -f "$KV_DIR/${d_check}_${hh}.kv.txt" ]]; then
                RANGE_NEED_FETCH=1; break 2
            fi
        done
        d_check=$(date -I -d "$d_check + 1 day")
    done

    if [[ "$RANGE_NEED_FETCH" == "1" && "${SKIP_RANGE_STAGE0:-0}" != "1" ]]; then
        OUTER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        echo ""
        echo "##########################################################"
        echo "##  Range Stage 0: bulk fetch $START_DATE → $END_DATE"
        echo "##########################################################"

        # Resolve metadata + site_list from the kriging YAML
        OUTER_PATHS=$("${PYTHON_BIN:-python3}" - "$CONFIG" <<'PY'
import os, sys, yaml
cfg_path = sys.argv[1]
cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
with open(cfg_path) as f:
    cfg = yaml.safe_load(f) or {}
d = cfg.get("data", {})
for k in ("metadata_file", "site_list_file"):
    v = d.get(k)
    if v and not os.path.isabs(v):
        v = os.path.normpath(os.path.join(cfg_dir, v))
    print(v or "")
PY
)
        OUTER_META=$(echo  "$OUTER_PATHS" | sed -n '1p')
        OUTER_SITES=$(echo "$OUTER_PATHS" | sed -n '2p')
        if [[ -z "$OUTER_META" || -z "$OUTER_SITES" || ! -f "$OUTER_META" || ! -f "$OUTER_SITES" ]]; then
            echo "ERROR: could not resolve metadata_file / site_list_file from $CONFIG"
            exit 1
        fi

        mkdir -p "$EXPORTS_DIR/hour_logs"
        RAW_DIR="${RAW_DIR:-$EXPORTS_DIR/raw_iv_${START_DATE}_to_${END_DATE}}"
        rm -rf "$RAW_DIR"
        mkdir -p "$RAW_DIR"

        echo " raw dir : $RAW_DIR (fresh)"
        echo " metadata: $OUTER_META"
        echo " sites   : $OUTER_SITES"
        echo ""

        # Fetch 1 day earlier so UTC hours 00-03 of START_DATE are filled by
        # the previous-day local readings (after tz_cd→UTC conversion).
        BULK_RANGE_START=$(date -I -d "$START_DATE - 1 day")
        echo ">>> Range Stage 0a: bulk download (single-pass, $BULK_RANGE_START → $END_DATE)"
        "${PYTHON_BIN:-python3}" "$OUTER_SCRIPT_DIR/usgs_raw_to_hourly_bulk.py" \
            --site-list "$OUTER_SITES" \
            --out-dir   "$RAW_DIR" \
            --start     "$BULK_RANGE_START" \
            --end       "$END_DATE" \
            --workers   "${BULK_WORKERS:-16}" \
            > "$EXPORTS_DIR/hour_logs/range_${START_DATE}_to_${END_DATE}_stage0_bulk.log" 2>&1 \
            || { echo "ERROR: range bulk download failed; see hour_logs/"; exit 1; }

        echo ">>> Range Stage 0b: raw → hourly KV (UTC-aware, all days)"
        "${PYTHON_BIN:-python3}" "$OUTER_SCRIPT_DIR/usgs_raw_to_hourly_kv.py" \
            --raw-dir   "$RAW_DIR" \
            --metadata  "$OUTER_META" \
            --site-list "$OUTER_SITES" \
            --logs-dir  "$KV_DIR" \
            --start     "$START_DATE" \
            --end       "$END_DATE" \
            --overwrite \
            > "$EXPORTS_DIR/hour_logs/range_${START_DATE}_to_${END_DATE}_stage0_kv.log" 2>&1 \
            || { echo "ERROR: range raw→KV failed; see hour_logs/"; exit 1; }

        if [[ "${KEEP_RAW:-0}" != "1" ]]; then
            rm -rf "$RAW_DIR"
        fi

        echo ">>> Range Stage 0 complete — KV cache populated for entire range"
        echo ""
    fi

    d="$START_DATE"
    while [[ "$d" < "$END_DATE" || "$d" == "$END_DATE" ]]; do
        echo ""
        echo "##########################################"
        echo "##  Processing $d"
        echo "##########################################"
        DATE="$d" START_DATE="" END_DATE="" bash "${BASH_SOURCE[0]}"
        d=$(date -I -d "$d + 1 day")
    done
    exit 0
fi

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

mkdir -p "$KV_DIR" "$EXPORTS_DIR/plots/kriging" "$CATCHMENT_OUT_DIR"

echo "============================================="
echo " qkrig Hourly Runner"
echo "============================================="
echo " Config       : $CONFIG"
echo " Date         : $DATE"
echo " Hours        : $H_START – $H_END"
echo " KV dir       : $KV_DIR"
echo " Exports dir  : $EXPORTS_DIR"
echo " Plot config  : $PLOT_CONFIG"
echo " GPKG path    : $GPKG_PATH"
echo " Catchment out: $CATCHMENT_OUT_DIR"
echo " Max procs    : $MAX_PROCS"
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

HOUR_LOG_DIR="$EXPORTS_DIR/hour_logs"
mkdir -p "$HOUR_LOG_DIR"

# Stage 0: bulk fetch + raw→KV (UTC-aware via tz_cd). Single source of truth
# for NWIS data; the previous in-script nwis.get_record path was timezone-naive.
NEED_FETCH=0
for h in $HOURS_LIST; do
    [[ ! -f "$KV_DIR/${DATE}_${h}.kv.txt" ]] && NEED_FETCH=1 && break
done

if [[ "$NEED_FETCH" == "1" ]]; then
    echo "============================================="
    echo " Stage 0: bulk-fetching IV → KV for $DATE"
    echo "============================================="

    # Paths in usgsgaugekrig.yaml are relative to the YAML's own directory.
    PATHS=$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import os, sys, yaml
cfg_path = sys.argv[1]
cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
with open(cfg_path) as f:
    cfg = yaml.safe_load(f) or {}
d = cfg.get("data", {})
for k in ("metadata_file", "site_list_file"):
    v = d.get(k)
    if v and not os.path.isabs(v):
        v = os.path.normpath(os.path.join(cfg_dir, v))
    print(v or "")
PY
)
    META=$(echo "$PATHS"   | sed -n '1p')
    SITES=$(echo "$PATHS"  | sed -n '2p')
    if [[ -z "$META" || -z "$SITES" || ! -f "$META" || ! -f "$SITES" ]]; then
        echo "ERROR: could not resolve metadata_file / site_list_file from $CONFIG"
        echo "       META='$META'  SITES='$SITES'"
        exit 1
    fi

    # Wipe-and-refetch — bulk downloader's resume logic checks file existence,
    # not date-range coverage, so a stale raw dir would be silently reused.
    RAW_DIR="${RAW_DIR:-$EXPORTS_DIR/raw_iv_${DATE}}"
    rm -rf "$RAW_DIR"
    mkdir -p "$RAW_DIR"
    echo " raw dir : $RAW_DIR (fresh)"
    echo " metadata: $META"
    echo " sites   : $SITES"
    echo ""

    # Fetch 1 day earlier so UTC hours 00-03 of DATE are filled by previous-day
    # local readings (after tz_cd→UTC conversion in the next step).
    BULK_START=$(date -I -d "$DATE - 1 day")
    echo ">>> Stage 0a: bulk download (raw .rdb, $BULK_START → $DATE)"
    "$PYTHON_BIN" "$SCRIPT_DIR/usgs_raw_to_hourly_bulk.py" \
        --site-list "$SITES" \
        --out-dir   "$RAW_DIR" \
        --start     "$BULK_START" \
        --end       "$DATE" \
        --workers   "${BULK_WORKERS:-16}" \
        > "$HOUR_LOG_DIR/${DATE}_stage0_bulk.log" 2>&1 \
        || { echo "ERROR: bulk download failed; see $HOUR_LOG_DIR/${DATE}_stage0_bulk.log"; exit 1; }

    echo ">>> Stage 0b: raw → hourly KV (UTC-aware)"
    "$PYTHON_BIN" "$SCRIPT_DIR/usgs_raw_to_hourly_kv.py" \
        --raw-dir   "$RAW_DIR" \
        --metadata  "$META" \
        --site-list "$SITES" \
        --logs-dir  "$KV_DIR" \
        --start     "$DATE" \
        --end       "$DATE" \
        --overwrite \
        > "$HOUR_LOG_DIR/${DATE}_stage0_kv.log" 2>&1 \
        || { echo "ERROR: raw→KV conversion failed; see $HOUR_LOG_DIR/${DATE}_stage0_kv.log"; exit 1; }

    MISSING_KV=""
    for h in $HOURS_LIST; do
        [[ ! -f "$KV_DIR/${DATE}_${h}.kv.txt" ]] && MISSING_KV+="$h "
    done
    if [[ -n "$MISSING_KV" ]]; then
        echo "ERROR: Stage 0 finished but KV still missing for hours: $MISSING_KV"
        exit 1
    fi

    [[ "${KEEP_RAW:-0}" != "1" ]] && rm -rf "$RAW_DIR"
    echo ">>> Stage 0 complete — KV cache ready for $DATE"
    echo ""
fi

# Per-hour MPLCONFIGDIR avoids fontList.json race when many procs cold-start
# matplotlib in parallel (random hour failures otherwise).
run_one_hour() {
    local h="$1"
    local log="$HOUR_LOG_DIR/${DATE}_${h}.log"
    local extra=()
    if [[ -n "$PLOT_CONFIG" ]]; then
        extra+=(--plot-config "$PLOT_CONFIG")
    fi
    MPLCONFIGDIR="/tmp/matplotlib_${DATE}_${h}" \
    "$PYTHON_BIN" Scripts/run_usgs_krig_hour.py \
        --config "$CONFIG" \
        --kv-dir "$KV_DIR" \
        "${extra[@]}" \
        --year  "${DATE:0:4}" \
        --month "${DATE:5:2}" \
        --day   "${DATE:8:2}" \
        --hour  "$((10#$h))" \
        > "$log" 2>&1
}
export -f run_one_hour
export CONFIG KV_DIR PYTHON_BIN DATE PLOT_CONFIG HOUR_LOG_DIR EXPORTS_DIR

# Identify which hours succeeded by NC existence, so we can retry the rest.
succeeded_hours() {
    local out=""
    for h in $HOURS_LIST; do
        [[ -f "$EXPORTS_DIR/interp_${DATE}_${h}.nc" ]] && out+="$h "
    done
    echo "$out"
}

missing_hours() {
    local input="$1"
    local out=""
    for h in $input; do
        [[ ! -f "$EXPORTS_DIR/interp_${DATE}_${h}.nc" ]] && out+="$h "
    done
    echo "$out"
}

# Pre-warm cartopy Natural Earth cache so parallel kriging procs don't race
# the shapefile download (silently produces unmasked / blue-background plots).
"$PYTHON_BIN" - <<'PY' >/dev/null 2>&1 || true
import cartopy.io.shapereader as shpreader
shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
PY

# Stage 1: parallel kriging. No --halt so single-hour failures don't abort.
echo ">>> Running $N_HOURS hour(s) in parallel (max $MAX_PROCS)..."
set +e
if command -v parallel >/dev/null 2>&1; then
    echo "$HOURS_LIST" | parallel -j "$MAX_PROCS" run_one_hour {}
else
    echo "$HOURS_LIST" | xargs -P "$MAX_PROCS" -I{} bash -c 'run_one_hour "$@"' _ {}
fi
set -e

FAILED=$(missing_hours "$HOURS_LIST" | xargs || true)
if [[ -n "$FAILED" ]]; then
    echo ">>> Retrying failed hours serially: $FAILED"
    for h in $FAILED; do
        if run_one_hour "$h"; then
            echo "  ✓ hour $h recovered on retry"
        else
            echo "  ✗ hour $h failed retry — see $HOUR_LOG_DIR/${DATE}_${h}.log"
        fi
    done
fi

DONE_HOURS=$(succeeded_hours | wc -w | tr -d ' ')
STILL_MISSING=$(missing_hours "$HOURS_LIST" | xargs || true)

echo ""
echo "============================================="
echo " Kriging done: $DONE_HOURS / $N_HOURS hours succeeded"
if [[ -n "$STILL_MISSING" ]]; then
    echo " Missing hours: $STILL_MISSING (logs in $HOUR_LOG_DIR/)"
fi
echo " NC files in $EXPORTS_DIR"
echo "============================================="

# Daily GIF: only built on full-day runs (partial runs would be misleading).
if [[ "$SKIP_GIF" != "1" && "$H_START" == "00" && "$H_END" == "23" ]]; then
    echo ""
    echo "============================================="
    echo " Building daily GIF"
    echo "============================================="
    "$PYTHON_BIN" Scripts/build_daily_gif.py \
        --pattern "$EXPORTS_DIR/plots/kriging/kriging_combo_${DATE}_*.png" \
        --output  "$EXPORTS_DIR/plots/kriging/kriging_combo_${DATE}.gif" \
        --duration "$GIF_FRAME_MS" \
        --max-width "$GIF_MAX_WIDTH" || echo "WARNING: GIF build failed; continuing."
elif [[ "$SKIP_GIF" == "1" ]]; then
    echo ""
    echo "Daily GIF skipped (SKIP_GIF=1)."
elif [[ "$H_START" != "00" || "$H_END" != "23" ]]; then
    echo ""
    echo "Daily GIF skipped (partial-hour run: $H_START–$H_END)."
fi

# Catchment extraction: full-day only (qkrig_ts_hourly day-mode reads all 24 NCs).
if [[ "$SKIP_EXTRACTION" != "1" && "$H_START" == "00" && "$H_END" == "23" ]]; then
    if [[ ! -f "$GPKG_PATH" && ! -d "$GPKG_PATH" ]]; then
        echo ""
        echo "WARNING: GPKG not found at $GPKG_PATH; skipping catchment extraction."
        echo "         Mount it via -v <host_gpkg>:$GPKG_PATH:ro or set GPKG_PATH=..."
    else
        echo ""
        echo "============================================="
        echo " Per-catchment extraction"
        echo "============================================="
        "$PYTHON_BIN" Scripts/qkrig_ts_hourly.py "$DATE" "$CATCHMENT_OUT_DIR" \
            --gpkg "$GPKG_PATH" \
            --exports-dir "$EXPORTS_DIR"
        echo "============================================="
        echo " Catchment CSVs written to $CATCHMENT_OUT_DIR."
        echo "============================================="
    fi
elif [[ "$SKIP_EXTRACTION" == "1" ]]; then
    echo ""
    echo "Catchment extraction skipped (SKIP_EXTRACTION=1)."
elif [[ "$H_START" != "00" || "$H_END" != "23" ]]; then
    echo ""
    echo "Catchment extraction skipped (partial-hour run: $H_START–$H_END)."
fi
