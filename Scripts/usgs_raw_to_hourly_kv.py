#!/usr/bin/env python3
"""
Transform raw USGS IV .rdb files (one file per site) into per-hour KV files
used by the qkrig hourly loader.

Input assumptions:
- Each site has an .rdb file (tab-delimited), e.g. 021556525_iv.rdb
- The .rdb files have:
    - Comment lines starting with '#'
    - A header row   (agency_cd  site_no  datetime  tz_cd  NNNNN_00060 ...)
    - A format row   (5s  15s  20d  6s  14n ...)
    - Data rows with tab-separated values
- A metadata CSV provides lon/lat/area for each site.

Output (per hour):
- <logs_dir>/YYYY-MM-DD_HH.kv.txt  (OK/FAIL entries)
- <logs_dir>/YYYY-MM-DD_HH.txt     (human-readable log)

The script averages all 15-minute readings within each hour to produce an
hourly-mean discharge, then converts to mm/hr.

Usage:
    python usgs_raw_to_hourly_kv.py \\
        --raw-dir /path/to/rdb_files/ \\
        --metadata /path/to/site_metadata.csv \\
        --logs-dir /path/to/output/ \\
        --start 2023-09-25 --end 2023-09-29
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Iterable

import numpy as np
import pandas as pd


# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert raw USGS per-site IV .rdb files to per-hour KV files."
    )
    p.add_argument("--raw-dir", required=True,
                   help="Directory of raw .rdb files (one per site).")
    p.add_argument("--metadata", required=True,
                   help="USGS site metadata file (CSV/TSV; must contain site_no/dec_long_va/dec_lat_va/drain_area_va).")
    p.add_argument("--logs-dir", default="usgs_hourly_retrieval_logs",
                   help="Output directory for KV + log files.")
    p.add_argument("--start", default=None,
                   help="Optional start date (YYYY-MM-DD) to restrict output.")
    p.add_argument("--end", default=None,
                   help="Optional end date (YYYY-MM-DD) to restrict output.")
    p.add_argument("--overwrite", action="store_true",
                   help="Remove existing KV/log files in date range before writing.")
    p.add_argument("--min-readings", type=int, default=1,
                   help="Minimum number of 15-min readings required to compute an hourly mean (default: 2).")

    # Metadata parsing controls
    p.add_argument("--meta-sep", default=None,
                   help="Delimiter for metadata file. Auto-detected if omitted.")
    p.add_argument("--on-bad-lines", default="error",
                   choices=["error", "skip", "warn"],
                   help="Behavior for malformed metadata rows (default: error).")
    return p.parse_args()


# ======================================================================
# Utilities
# ======================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ======================================================================
# RDB parsing
# ======================================================================
def parse_rdb_file(filepath: str) -> Optional[pd.DataFrame]:
    """
    Parse a USGS .rdb file (tab-delimited with comment/format rows).

    Skips:
      - Lines starting with '#'
      - The format row (e.g. '5s  15s  20d  6s  14n  10s ...')

    Returns a DataFrame with the parsed data, or None on failure.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Could not read {filepath}: {e}")
        return None

    # Strip comment lines
    data_lines = [l for l in lines if not l.startswith("#")]
    if len(data_lines) < 3:
        # Need at least header + format row + 1 data row
        return None

    header = data_lines[0]

    # The format row looks like "5s\t15s\t20d\t6s\t14n\t10s..."
    # Detect and skip it
    format_row = data_lines[1]
    if re.match(r"^\d+[sndi]", format_row.strip()):
        data_start = 2
    else:
        data_start = 1  # No format row found, treat as data

    # Rebuild as a single string for pd.read_csv
    clean_text = header + "".join(data_lines[data_start:])

    try:
        from io import StringIO
        df = pd.read_csv(StringIO(clean_text), sep="\t", dtype=str)
    except Exception as e:
        print(f"Could not parse {filepath}: {e}")
        return None

    return df


def find_iv_flow_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the discharge (00060) column in an IV .rdb DataFrame.
    Skips columns ending in '_cd' (quality code columns).
    """
    cols = [str(c) for c in df.columns]
    candidates = [c for c in cols if "00060" in c and not c.endswith("_cd")]
    if len(candidates) >= 1:
        return candidates[0]
    return None


# ======================================================================
# Metadata
# ======================================================================
def load_metadata(
    meta_path: str,
    sep: Optional[str] = None,
    on_bad_lines: str = "error"
) -> pd.DataFrame:
    """
    Load and normalize site metadata to columns:
      gauge_id, gauge_lon, gauge_lat, area_km2
    """
    import csv

    def _try_read(_sep: Optional[str]) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(
                meta_path, sep=_sep, dtype=str, comment="#",
                engine="python", on_bad_lines=on_bad_lines,
            )
        except Exception:
            return None

    df = None

    # 1) Explicit sep
    if sep is not None:
        df = _try_read(sep)

    # 2) Sniff
    if df is None:
        try:
            with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
                sample = "".join([next(f) for _ in range(50)])
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            df = _try_read(dialect.delimiter)
        except Exception:
            df = None

    # 3) Fallback
    if df is None:
        for cand in [",", ";", "\t", "|"]:
            df = _try_read(cand)
            if df is not None:
                break

    if df is None or df.empty:
        raise ValueError(
            f"Failed to parse metadata file: {meta_path}. "
            "Try --meta-sep ',' (or ';'/'\\t') or --on-bad-lines skip."
        )

    # Flexible renames
    rename_map = {
        "site_no": "gauge_id",
        "gauge_id": "gauge_id",
        "dec_long_va": "gauge_lon",
        "gauge_lon": "gauge_lon",
        "dec_lat_va": "gauge_lat",
        "gauge_lat": "gauge_lat",
        "drain_area_va": "area_km2",
        "area_km2": "area_km2",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    needed = {"gauge_id", "gauge_lon", "gauge_lat", "area_km2"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}. Have: {df.columns.tolist()}")

    df["gauge_id"] = df["gauge_id"].astype(str).str.strip().str.zfill(8)
    df["gauge_lon"] = pd.to_numeric(df["gauge_lon"], errors="coerce")
    df["gauge_lat"] = pd.to_numeric(df["gauge_lat"], errors="coerce")
    df["area_km2"] = pd.to_numeric(df["area_km2"], errors="coerce")
    df = df.dropna(subset=["gauge_lon", "gauge_lat", "area_km2"])
    return df.set_index("gauge_id")


# ======================================================================
# KV / Log file helpers
# ======================================================================
def kv_path(logs_dir: str, hr_str: str) -> str:
    return os.path.join(logs_dir, f"{hr_str}.kv.txt")


def log_path(logs_dir: str, hr_str: str) -> str:
    return os.path.join(logs_dir, f"{hr_str}.txt")


def write_kv_header(path: str, hr_str: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# KV cache for USGS hourly IV retrieval at {hr_str}\n")
        f.write(f"# Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def append_kv_ok(path: str, site_id: str, lon: float, lat: float, mm_hr: float) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{site_id}=OK,{lon:.8f},{lat:.8f},{mm_hr:.8f}\n")


def append_kv_fail(path: str, site_id: str, reason: str) -> None:
    reason_clean = reason.replace(",", ";")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{site_id}=FAIL,{reason_clean}\n")


def write_log_header(path: str, hr_str: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        f.write("USGS Hourly IV Retrieval Log\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Hour queried: {hr_str}\n")
        f.write(f"Total sites attempted: 0\n")
        f.write(f"Successful: 0\n")
        f.write(f"Unsuccessful: 0\n\n")
        f.write("=== Successful retrievals ===\n")
        f.write("(none)\n\n")
        f.write("=== Unsuccessful retrievals ===\n")
        f.write("(none)\n")


def update_log_counts(
    path: str, total: int, ok: int, fail: int, hr_str: str,
    successes: Iterable[Tuple[str, float, float, float]]
) -> None:
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("USGS Hourly IV Retrieval Log")
    lines.append(f"Timestamp: {ts}")
    lines.append(f"Hour queried: {hr_str}")
    lines.append(f"Total sites attempted: {total}")
    lines.append(f"Successful: {ok}")
    lines.append(f"Unsuccessful: {fail}")
    lines.append("")
    lines.append("=== Successful retrievals ===")
    if ok > 0:
        for sid, lon, lat, mm in sorted(successes, key=lambda r: r[0]):
            lines.append(
                f"OK  site={sid}  lon={lon:.6f}  lat={lat:.6f}  streamflow_mm_hr={mm:.6f}"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("=== Unsuccessful retrievals ===")
    if fail > 0:
        lines.append("(see KV file for FAIL reasons)")
    else:
        lines.append("(none)")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ======================================================================
# Timestamp helper
# ======================================================================
def datetime_to_hr_str(dt: pd.Timestamp) -> str:
    """Convert a datetime to the YYYY-MM-DD_HH key used in kv filenames."""
    return dt.strftime("%Y-%m-%d_%H")


# ======================================================================
# Main
# ======================================================================
def main() -> int:
    args = parse_args()
    ensure_dir(args.logs_dir)

    # Optional date filtering
    start = pd.to_datetime(args.start).date() if args.start else None
    end = pd.to_datetime(args.end).date() if args.end else None

    min_readings = args.min_readings

    # Load metadata (lon/lat/area)
    meta = load_metadata(args.metadata, sep=args.meta_sep, on_bad_lines=args.on_bad_lines)
    print(f"Loaded metadata for {len(meta)} sites.")

    # Find .rdb files
    raw_files = [
        f for f in os.listdir(args.raw_dir)
        if f.lower().endswith(".rdb")
    ]
    if not raw_files:
        print(f"No .rdb files found in {args.raw_dir}")
        return 2

    print(f"Found {len(raw_files)} .rdb file(s) to process.")

    # Track per-hour counts for human logs
    hr_counts: Dict[str, Dict[str, int]] = {}       # {hr_str: {"total": X, "ok": Y, "fail": Z}}
    hr_successes: Dict[str, List] = {}               # {hr_str: [(site_id, lon, lat, mm), ...]}
    hr_sites: Dict[str, set] = {}                    # {hr_str: set of site_ids already written}

    for i, fname in enumerate(raw_files, 1):
        fpath = os.path.join(args.raw_dir, fname)
        df = parse_rdb_file(fpath)
        if df is None or df.empty:
            print(f"[{i}/{len(raw_files)}] {fname}: could not parse; skipping")
            continue

        # Identify columns
        if "site_no" not in df.columns:
            print(f"[{i}/{len(raw_files)}] {fname}: missing 'site_no' column; skipping")
            continue
        if "datetime" not in df.columns:
            print(f"[{i}/{len(raw_files)}] {fname}: missing 'datetime' column; skipping")
            continue

        flow_col = find_iv_flow_column(df)
        if flow_col is None:
            print(f"[{i}/{len(raw_files)}] {fname}: could not find 00060 discharge column; skipping")
            continue

        # Normalize
        df["site_id"] = df["site_no"].astype(str).str.strip().str.zfill(8)
        df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["dt"])

        # Date filter
        if start or end:
            dates = df["dt"].dt.date
            mask = pd.Series(True, index=df.index)
            if start:
                mask = mask & (dates >= start)
            if end:
                mask = mask & (dates <= end)
            df = df[mask]
            if df.empty:
                continue

        # Parse CFS values
        df["cfs"] = pd.to_numeric(df[flow_col], errors="coerce")

        # Floor timestamps to the hour for grouping
        df["hour"] = df["dt"].dt.floor("h")
        df["hr_str"] = df["hour"].apply(datetime_to_hr_str)

        # ---- Aggregate to hourly means per site ----
        # Group by (site_id, hour) and compute mean CFS + count of readings
        hourly = (
            df.groupby(["site_id", "hr_str", "hour"])["cfs"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_cfs", "count": "n_readings"})
        )

        # Join metadata
        hourly = hourly.merge(
            meta.reset_index(), left_on="site_id", right_on="gauge_id", how="left"
        )

        # Compute mm/hr from mean CFS
        area_m2 = pd.to_numeric(hourly["area_km2"], errors="coerce") * 1e6
        mm_hr = (hourly["mean_cfs"] * 0.0283168 * 3600.0 / area_m2) * 1000.0
        hourly["mm_hr"] = mm_hr
        hourly["ok_flag"] = (
            np.isfinite(mm_hr)
            & (area_m2 > 0)
            & (hourly["n_readings"] >= min_readings)
        )

        # Emit per-hour
        for hr_str, grp in hourly.groupby("hr_str"):
            kv = kv_path(args.logs_dir, hr_str)
            lg = log_path(args.logs_dir, hr_str)

            # Create files + headers if new
            if not os.path.exists(kv):
                write_kv_header(kv, hr_str)
            if not os.path.exists(lg):
                write_log_header(lg, hr_str)

            total = 0
            ok = 0
            fail = 0
            succ_for_log = []

            for _, row in grp.iterrows():
                sid = str(row["site_id"])
                hr_sites.setdefault(hr_str, set()).add(sid)
                total += 1
                if bool(row["ok_flag"]):
                    val = float(row["mm_hr"])
                    lon = float(row["gauge_lon"])
                    lat = float(row["gauge_lat"])
                    if np.isfinite(val):
                        append_kv_ok(kv, sid, lon, lat, val)
                        ok += 1
                        succ_for_log.append((sid, lon, lat, val))
                    else:
                        append_kv_fail(kv, sid, "missing")
                        fail += 1
                else:
                    if row["n_readings"] < min_readings:
                        reason = f"insufficient_readings (n={int(row['n_readings'])} < {min_readings})"
                    elif pd.isna(row["mm_hr"]):
                        reason = "missing"
                    else:
                        reason = "invalid_area"
                    append_kv_fail(kv, sid, reason)
                    fail += 1

            counts = hr_counts.setdefault(hr_str, {"total": 0, "ok": 0, "fail": 0})
            counts["total"] += total
            counts["ok"] += ok
            counts["fail"] += fail
            hr_successes.setdefault(hr_str, []).extend(succ_for_log)

        print(f"[{i}/{len(raw_files)}] processed {fname}")

    # Rewrite human logs with final counts
    for hr, counts in hr_counts.items():
        lg = log_path(args.logs_dir, hr)
        update_log_counts(
            lg, counts["total"], counts["ok"], counts["fail"],
            hr, hr_successes.get(hr, [])
        )

    print(f"\nDone. Written KV files for {len(hr_counts)} hours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
