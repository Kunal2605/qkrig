#!/usr/bin/env python3
"""
Transform raw USGS DV CSVs (one file per site) into per-day KV files used by qkrig.

Input assumptions:
- Each site has a CSV (as saved by usgs_dump_raw.py), e.g. 01010000.csv
- Columns include:
    datetime, site_no, 00060_Mean, 00060_Mean_cd
  OR a variant like:
    datetime, site_no, 00060_00003, 00060_00003_cd
- A metadata CSV provides lon/lat/area for each site.

Output (per day):
- <logs_dir>/YYYY-MM-DD.kv.txt  (OK/FAIL entries)
- <logs_dir>/YYYY-MM-DD.txt     (human-readable log)

You can run this multiple times; it appends to existing daily files and
only writes the header when creating a file for the first time.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert raw USGS per-site CSVs to daily KV files.")
    p.add_argument("--raw-dir", required=True, help="Directory of raw CSVs (one per site).")
    p.add_argument("--metadata", required=True,
                   help="USGS site metadata file (CSV/TSV/etc.; contains site_no/dec_long_va/dec_lat_va/drain_area_va or mapped names).")
    p.add_argument("--logs-dir", default="usgs_retrieval_logs", help="Output directory for KV + log files.")
    p.add_argument("--site-col", default="site_no",
                   help="Column name in raw CSV that holds the site id (default: site_no).")
    p.add_argument("--start", default=None, help="Optional start date (YYYY-MM-DD) to restrict output.")
    p.add_argument("--end", default=None, help="Optional end date (YYYY-MM-DD) to restrict output.")
    p.add_argument("--overwrite", action="store_true",
                   help="If set, removes existing daily KV/log files within the selected date range before writing.")
    p.add_argument("--site-list", default=None,
                   help="Optional text file with one site ID per line. Only these sites will appear in output.")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                   help="Bounding box filter: min_lon min_lat max_lon max_lat")

    # NEW: metadata parsing controls
    p.add_argument("--meta-sep", default=None,
                   help="Delimiter for metadata file (e.g. ',' ';' '\\t'). If omitted, will try to auto-detect.")
    p.add_argument("--on-bad-lines", default="error", choices=["error", "skip", "warn"],
                   help="Behavior for malformed metadata rows (default: error).")
    return p.parse_args()



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_flow_column(df: pd.DataFrame) -> Optional[str]:
    """
    Return the discharge (cfs) column name. Accepts '...Mean' or '...00003'.
    Fallback: if exactly one '00060' column exists, take it.
    """
    cols = [str(c) for c in df.columns]
    for c in cols:
        if "00060" in c and "Mean" in c:
            return c
    for c in cols:
        if "00060" in c and "00003" in c:
            return c
    cand = [c for c in cols if "00060" in c]
    if len(cand) == 1:
        return cand[0]
    return None


def load_metadata(meta_path: str, sep: Optional[str] = None, on_bad_lines: str = "error") -> pd.DataFrame:
    """
    Load and normalize site metadata to columns:
      gauge_id, gauge_lon, gauge_lat, area_km2

    Tries:
      - user-provided sep (if given)
      - csv.Sniffer auto-detection
      - common fallbacks: ',', ';', '\\t', '|'
    Uses engine='python' to better handle quotes/embedded delimiters.
    """
    import csv

    def _try_read(_sep: Optional[str]) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(
                meta_path,
                sep=_sep,
                dtype=str,
                comment="#",
                engine="python",
                on_bad_lines=on_bad_lines,  # pandas>=1.3
            )
        except Exception:
            return None

    df = None

    # 1) explicit sep
    if sep is not None:
        df = _try_read(sep)

    # 2) sniff delimiter if not provided / first try failed
    if df is None:
        try:
            with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
                sample = "".join([next(f) for _ in range(50)])
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            df = _try_read(dialect.delimiter)
        except Exception:
            df = None

    # 3) fallback cycle
    if df is None:
        for cand in [",", ";", "\t", "|"]:
            df = _try_read(cand)
            if df is not None:
                break

    if df is None or df.empty:
        raise ValueError(f"Failed to parse metadata file: {meta_path}. Try --meta-sep ',' (or ';'/'\\t') or --on-bad-lines skip.")

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

    # Tidy types
    df["gauge_id"] = df["gauge_id"].astype(str).str.strip().str.zfill(8)
    df["gauge_lon"] = pd.to_numeric(df["gauge_lon"], errors="coerce")
    df["gauge_lat"] = pd.to_numeric(df["gauge_lat"], errors="coerce")
    df["area_km2"] = pd.to_numeric(df["area_km2"], errors="coerce")
    # USGS drain_area_va is in sq miles — convert to sq km
    df["area_km2"] = df["area_km2"] * 2.58999
    df = df.dropna(subset=["gauge_lon", "gauge_lat", "area_km2"])
    return df.set_index("gauge_id")



def kv_path(logs_dir: str, date_str: str) -> str:
    return os.path.join(logs_dir, f"{date_str}.kv.txt")


def log_path(logs_dir: str, date_str: str) -> str:
    return os.path.join(logs_dir, f"{date_str}.txt")


def write_kv_header(path: str, date_str: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# KV cache for USGS retrieval on {date_str}\n")
        f.write(f"# Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def append_kv_ok(path: str, site_id: str, lon: float, lat: float, mm_day: float) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{site_id}=OK,{lon:.8f},{lat:.8f},{mm_day:.8f}\n")


def append_kv_fail(path: str, site_id: str, reason: str) -> None:
    reason_clean = reason.replace(",", ";")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{site_id}=FAIL,{reason_clean}\n")


def write_log_header(path: str, date_str: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        f.write("USGS Retrieval Log\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Date queried: {date_str}\n")
        f.write(f"Total sites attempted: 0\n")
        f.write(f"Successful: 0\n")
        f.write(f"Unsuccessful: 0\n\n")
        f.write("=== Successful retrievals ===\n")
        f.write("(none)\n\n")
        f.write("=== Unsuccessful retrievals ===\n")
        f.write("(none)\n")


def update_log_counts(path: str, total: int, ok: int, fail: int, date_str: str,
                      successes: Iterable[Tuple[str, float, float, float]]) -> None:
    """
    Rewrites the human log for the date with counts and success lines.
    """
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("USGS Retrieval Log")
    lines.append(f"Timestamp: {ts}")
    lines.append(f"Date queried: {date_str}")
    lines.append(f"Total sites attempted: {total}")
    lines.append(f"Successful: {ok}")
    lines.append(f"Unsuccessful: {fail}")
    lines.append("")
    lines.append("=== Successful retrievals ===")
    if ok > 0:
        for sid, lon, lat, mm in sorted(successes, key=lambda r: r[0]):
            lines.append(f"OK  site={sid}  lon={lon:.6f}  lat={lat:.6f}  streamflow_mm_day={mm:.6f}")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("=== Unsuccessful retrievals ===")
    if fail > 0:
        lines.append("(see KV file for FAIL reasons)")  # keep it compact
    else:
        lines.append("(none)")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    ensure_dir(args.logs_dir)

    # Optional date filtering
    start = pd.to_datetime(args.start).date() if args.start else None
    end = pd.to_datetime(args.end).date() if args.end else None

    # Load metadata (lon/lat/area)
    meta = load_metadata(args.metadata, sep=args.meta_sep, on_bad_lines=args.on_bad_lines)
    print(f"Loaded metadata for {len(meta)} sites.")

    # Optional site list filter
    if args.site_list and os.path.exists(args.site_list):
        with open(args.site_list, "r") as f:
            wanted = {line.strip() for line in f if line.strip() and not line.startswith("#")}
        meta = meta[meta.index.isin(wanted)]
        print(f"Filtered to {len(meta)} sites from site list.")

    # Optional bbox filter
    if args.bbox:
        min_lon, min_lat, max_lon, max_lat = args.bbox
        before = len(meta)
        meta = meta[
            (meta["gauge_lon"] >= min_lon) & (meta["gauge_lon"] <= max_lon) &
            (meta["gauge_lat"] >= min_lat) & (meta["gauge_lat"] <= max_lat)
        ]
        print(f"Bbox filter: {before} -> {len(meta)} sites within [{min_lon},{min_lat},{max_lon},{max_lat}]")

    # (optional) cleanup existing files in range
    if args.overwrite and start and end:
        for d in pd.date_range(start=start, end=end, freq="D"):
            ds = d.strftime("%Y-%m-%d")
            for p in (kv_path(args.logs_dir, ds), log_path(args.logs_dir, ds)):
                if os.path.exists(p):
                    os.remove(p)

    # Process each raw site file one-by-one (streaming; safe on memory)
    raw_files = [f for f in os.listdir(args.raw_dir) if f.lower().endswith(".csv")]
    if not raw_files:
        print(f"No CSVs found in {args.raw_dir}")
        return 2

    # Track per-day counts and successes for human logs
    day_counts: Dict[str, Dict[str, int]] = {}          # {date: {"total": X, "ok": Y, "fail": Z}}
    day_successes: Dict[str, list] = {}                 # {date: [(site_id, lon, lat, mm), ...]}

    for i, fname in enumerate(raw_files, 1):
        path = os.path.join(args.raw_dir, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[{i}/{len(raw_files)}] {fname}: read error: {e}")
            continue

        if args.site_col not in df.columns:
            print(f"[{i}/{len(raw_files)}] {fname}: missing '{args.site_col}' column; skipping")
            continue

        if "datetime" not in df.columns:
            print(f"[{i}/{len(raw_files)}] {fname}: missing 'datetime' column; skipping")
            continue

        flow_col = find_flow_column(df)
        if flow_col is None:
            print(f"[{i}/{len(raw_files)}] {fname}: could not find 00060 daily mean column; skipping")
            continue

        # Normalize types
        df["site_id"] = df[args.site_col].astype(str).str.strip().str.zfill(8)
        # Parse datetimes, keep date only (honors timezone info; normalize to date)
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        if start or end:
            mask = True
            if start:
                mask = mask & (df["date"] >= start)
            if end:
                mask = mask & (df["date"] <= end)
            df = df[mask]
            if df.empty:
                continue

        # Join metadata
        df = df.merge(
            meta.reset_index(), left_on="site_id", right_on="gauge_id", how="left"
        )

        # Compute mm/day
        area_m2 = (pd.to_numeric(df["area_km2"], errors="coerce") * 1e6)
        cfs = pd.to_numeric(df[flow_col], errors="coerce")
        mm_day = (cfs * 0.0283168 * 86400.0 / area_m2) * 1000.0
        df["mm_day"] = mm_day
        df["ok_flag"] = np.isfinite(mm_day) & (area_m2 > 0)

        # Emit per-day
        for ds, grp in df.groupby("date"):
            ds_str = pd.Timestamp(ds).strftime("%Y-%m-%d")
            kv = kv_path(args.logs_dir, ds_str)
            lg = log_path(args.logs_dir, ds_str)

            # Make files and headers if new
            if not os.path.exists(kv):
                write_kv_header(kv, ds_str)
            if not os.path.exists(lg):
                write_log_header(lg, ds_str)

            # Append lines
            total = 0
            ok = 0
            fail = 0
            succ_for_log = []

            for _, row in grp.iterrows():
                sid = str(row["site_id"])
                total += 1
                if bool(row["ok_flag"]):
                    val = float(row["mm_day"])
                    lon = float(row["gauge_lon"])
                    lat = float(row["gauge_lat"])
                    # Optional sanity bounds
                    if np.isfinite(val) and -69 <= val <= 69:
                        append_kv_ok(kv, sid, lon, lat, val)
                        ok += 1
                        succ_for_log.append((sid, lon, lat, val))
                    else:
                        append_kv_fail(kv, sid, "Large_magnitude_flow")
                        fail += 1
                else:
                    reason = "missing" if pd.isna(row["mm_day"]) else "invalid_area"
                    append_kv_fail(kv, sid, reason)
                    fail += 1

            # Update running counts for later rewrite of human log
            counts = day_counts.setdefault(ds_str, {"total": 0, "ok": 0, "fail": 0})
            counts["total"] += total
            counts["ok"] += ok
            counts["fail"] += fail
            day_successes.setdefault(ds_str, []).extend(succ_for_log)

        print(f"[{i}/{len(raw_files)}] processed {fname}")

    # Rewrite human logs with proper counts + success lines
    for ds, counts in day_counts.items():
        lg = log_path(args.logs_dir, ds)
        update_log_counts(lg, counts["total"], counts["ok"], counts["fail"], ds, day_successes.get(ds, []))

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
