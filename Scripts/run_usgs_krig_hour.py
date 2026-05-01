#!/usr/bin/env python3
"""
Run USGS kriging for a single HOUR.

Data resolution order (mirrors the daily USGSLoader pattern):
  1. Pre-built hourly KV file  →  <kv-dir>/YYYY-MM-DD_HH.kv.txt
  2. If no KV file, fetch IV data from NWIS for ALL sites for that DATE,
     compute hourly means, write KV caches for every hour of the day,
     then proceed with kriging for the target hour.

Exports are named with the hour:
    interp_2023-05-01_14.npz   variogram_2023-05-01_14.csv

Usage:
    python Scripts/run_usgs_krig_hour.py \\
        --config configs/usgsgaugekrig.yaml \\
        --kv-dir /path/to/hourly_kv_output/ \\
        --year 2023 --month 5 --day 1 --hour 14

    # Without --kv-dir it still works: fetches from NWIS, caches to
    # the data_cache_directory in the config.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

from interpolation.usgs_krig import USGSKrig


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Run USGS kriging for a single hour (self-contained)."
    )
    p.add_argument("--config", required=True, help="Path to usgsgaugekrig.yaml")
    p.add_argument("--kv-dir", default=None,
                   help="Directory for hourly KV files. "
                        "Defaults to data.data_cache_directory from config.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--day", type=int, required=True)
    p.add_argument("--hour", type=int, required=True)
    p.add_argument("--plot-config", default=None,
                   help="Optional override: path to plot_config.yaml.")
    return p.parse_args()


# ======================================================================
# KV file I/O
# ======================================================================
def kv_file_path(kv_dir: str, hr_str: str) -> str:
    return os.path.join(kv_dir, f"{hr_str}.kv.txt")


def load_hourly_kv(kv_dir: str, hr_str: str
                   ) -> Optional[Tuple[List[Tuple[float, float, float, str]],
                                       List[Tuple[str, str]]]]:
    """Load a YYYY-MM-DD_HH.kv.txt file. Returns (successes, failures) or None."""
    path = kv_file_path(kv_dir, hr_str)
    if not os.path.exists(path):
        return None

    successes: List[Tuple[float, float, float, str]] = []
    failures: List[Tuple[str, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, val = s.split("=", 1)
            parts = val.split(",")
            if not parts:
                continue
            status = parts[0].strip().upper()
            if status == "OK" and len(parts) == 4:
                try:
                    lon = float(parts[1])
                    lat = float(parts[2])
                    mm = float(parts[3])
                    successes.append((lon, lat, mm, key))
                except Exception:
                    failures.append((key, "kv_parse_error"))
            elif status == "FAIL" and len(parts) >= 2:
                reason = ",".join(parts[1:]).strip()
                failures.append((key, reason))
            else:
                failures.append((key, "kv_bad_line"))

    return successes, failures


def write_hourly_kv(kv_dir: str, hr_str: str,
                    successes: List[Tuple[float, float, float, str]],
                    failures: List[Tuple[str, str]]) -> None:
    """Write a YYYY-MM-DD_HH.kv.txt file (same format as usgs_raw_to_hourly_kv.py)."""
    os.makedirs(kv_dir, exist_ok=True)
    path = kv_file_path(kv_dir, hr_str)
    lines = []
    lines.append(f"# KV cache for USGS hourly IV retrieval at {hr_str}")
    lines.append(f"# Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for lon, lat, mm, sid in sorted(successes, key=lambda r: r[3]):
        lines.append(f"{sid}=OK,{lon:.8f},{lat:.8f},{mm:.8f}")
    for sid, reason in sorted(failures, key=lambda r: r[0]):
        reason_clean = str(reason).replace(",", ";")
        lines.append(f"{sid}=FAIL,{reason_clean}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ======================================================================
# Metadata loading (reuses config format from USGSLoader)
# ======================================================================
def load_gauge_metadata(cfg: dict) -> pd.DataFrame:
    """
    Load gauge metadata from config, apply site-list and bbox filters.
    Returns DataFrame indexed by gauge_id with columns:
        gauge_lat, gauge_lon, area_sq_mi
    """
    dcfg = cfg.get("data", {})
    scfg = cfg.get("settings", {})

    df = pd.read_csv(dcfg["metadata_file"], comment="#", dtype={"site_no": str, "0site_no": str})
    df = df.rename(columns={
        "site_no": "gauge_id",
        "0site_no": "gauge_id",
        "dec_lat_va": "gauge_lat",
        "dec_long_va": "gauge_lon",
        "drain_area_va": "area_sq_mi",
    })
    df = df[["gauge_id", "gauge_lat", "gauge_lon", "area_sq_mi"]]
    df = df.dropna(subset=["gauge_lon", "gauge_lat"])

    # Filter by optional site list
    site_list_file = dcfg.get("site_list_file")
    if site_list_file and os.path.exists(site_list_file):
        with open(site_list_file, "r") as f:
            wanted = {line.strip().lstrip("0") for line in f if line.strip()}
        df = df[df["gauge_id"].str.lstrip("0").isin(wanted)]

    # Optional area filter (min_area is in km²; column is sq mi)
    min_area = float(scfg.get("min_area_km2", 0.0))
    if min_area > 0:
        df = df[(df["area_sq_mi"] * 2.58999) >= min_area]

    # Optional bbox filter
    bbox = scfg.get("bbox")
    if bbox and len(bbox) == 4:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox)
        df = df[
            (df["gauge_lon"] >= min_lon) & (df["gauge_lon"] <= max_lon) &
            (df["gauge_lat"] >= min_lat) & (df["gauge_lat"] <= max_lat)
        ]

    return df.set_index("gauge_id")


# ======================================================================
# NWIS IV fetch (parallel, with retries — mirrors USGSLoader pattern)
# ======================================================================
def fetch_site_iv(
    site_id: str,
    date_str: str,
    max_retries: int = 3,
    retry_backoff: float = 0.75,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Fetch instantaneous-value (IV) discharge for one site for one day.
    Returns (site_id, DataFrame-or-None).
    """
    import dataretrieval.nwis as nwis

    attempt = 0
    while True:
        try:
            df = nwis.get_record(
                sites=site_id,
                service="iv",
                start=date_str,
                end=date_str,
                parameterCd="00060",
            )
            if df is None or df.empty:
                return (site_id, None)

            # Find discharge column (00060, not the _cd quality column)
            cols = [c for c in df.columns if "00060" in c and not c.endswith("_cd")]
            if not cols:
                return (site_id, None)

            out = df[[cols[0]]].copy()
            out = out.rename(columns={cols[0]: "cfs"})
            out["cfs"] = pd.to_numeric(out["cfs"], errors="coerce")
            out.loc[out["cfs"] < 0, "cfs"] = np.nan

            if not isinstance(out.index, pd.DatetimeIndex):
                out.index = pd.to_datetime(out.index)
            return (site_id, out)

        except Exception:
            attempt += 1
            if attempt > max_retries:
                return (site_id, None)
            sleep_s = retry_backoff * (1 + random.random()) * attempt
            time.sleep(sleep_s)


def fetch_and_cache_all_hours(
    cfg: dict,
    meta: pd.DataFrame,
    kv_dir: str,
    year: int, month: int, day: int,
    min_readings: int = 1,
) -> None:
    """
    Fetch IV data from NWIS for ALL sites for one day, compute hourly means,
    and write KV files for every hour of the day.

    This avoids redundant API calls when multiple hours from the same day
    are being processed.
    """
    scfg = cfg.get("settings", {})
    concurrency = int(scfg.get("concurrency", 16))
    max_retries = int(scfg.get("max_retries", 3))
    retry_backoff = float(scfg.get("retry_backoff_seconds", 0.75))

    date_str = f"{year:04d}-{month:02d}-{day:02d}"
    sites = list(meta.index.values)

    print(f"  Fetching IV data from NWIS for {len(sites)} sites on {date_str}...")

    # Parallel fetch
    per_site: dict[str, Optional[pd.DataFrame]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(fetch_site_iv, sid, date_str, max_retries, retry_backoff): sid
            for sid in sites
        }
        done = 0
        for fut in as_completed(futures):
            sid = futures[fut]
            _, df = fut.result()
            per_site[sid] = df
            done += 1
            if done % 50 == 0 or done == len(sites):
                print(f"    fetched {done}/{len(sites)} sites")

    # Aggregate to hourly means and write KV files for each hour
    for hour in range(24):
        hr_str = f"{date_str}_{hour:02d}"
        kv_path = kv_file_path(kv_dir, hr_str)

        # Skip if already cached
        if os.path.exists(kv_path):
            continue

        successes: List[Tuple[float, float, float, str]] = []
        failures: List[Tuple[str, str]] = []

        for sid in sites:
            try:
                row = meta.loc[sid]
                lon = float(row["gauge_lon"])
                lat = float(row["gauge_lat"])
                area_sq_mi = float(row["area_sq_mi"])
            except Exception:
                failures.append((sid, "missing_metadata"))
                continue

            if not (np.isfinite(area_sq_mi) and area_sq_mi > 0):
                failures.append((sid, "missing_drainage_area"))
                continue

            df = per_site.get(sid)
            if df is None or df.empty:
                failures.append((sid, "nwis_empty"))
                continue

            # Extract readings for this hour
            mask = df.index.hour == hour
            hour_df = df[mask].dropna(subset=["cfs"])

            if len(hour_df) < min_readings:
                failures.append((sid, "missing"))
                continue

            mean_cfs = float(hour_df["cfs"].mean())
            if not np.isfinite(mean_cfs):
                failures.append((sid, "missing"))
                continue

            # Square miles -> square meters
            area_m2 = area_sq_mi * 2.58999e6
            mm_hr = (mean_cfs * 0.0283168 * 3600.0 / area_m2) * 1000.0

            if not np.isfinite(mm_hr):
                failures.append((sid, "missing"))
                continue

            successes.append((lon, lat, mm_hr, sid))

        write_hourly_kv(kv_dir, hr_str, successes, failures)

    print(f"  Cached KV files for all 24 hours of {date_str}.")


# ======================================================================
# Bbox filter
# ======================================================================
def filter_by_bbox(
    records: List[Tuple[float, float, float, str]],
    bbox: Optional[List[float]],
    pad: float = 0.0,
) -> List[Tuple[float, float, float, str]]:
    """Apply bounding box filter to (lon, lat, mm, site_id) records."""
    if not records or not bbox or len(bbox) != 4:
        return records
    min_lon, min_lat, max_lon, max_lat = map(float, bbox)
    min_lon -= pad; min_lat -= pad; max_lon += pad; max_lat += pad
    filtered = [
        (lon, lat, mm, sid)
        for (lon, lat, mm, sid) in records
        if (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)
    ]
    dropped = len(records) - len(filtered)
    if dropped > 0:
        print(f"[bbox] Dropped {dropped} record(s) outside bbox")
    return filtered


# ======================================================================
# Main
# ======================================================================
def main():
    args = parse_args()
    hr_str = f"{args.year:04d}-{args.month:02d}-{args.day:02d}_{args.hour:02d}"

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve relative paths in cfg["data"] anchored at the config file's dir
    # (matches USGSLoader._resolve_path() behavior so this script and the
    # loader interpret the same YAML the same way).
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    dcfg = cfg.setdefault("data", {})
    for key in ("metadata_file", "site_list_file", "data_dir",
                "data_cache_directory", "land_mask"):
        v = dcfg.get(key)
        if v and not os.path.isabs(v):
            dcfg[key] = os.path.normpath(os.path.join(cfg_dir, v))

    scfg = cfg.get("settings", {})
    plot_cfg_path = args.plot_config or cfg.get("plot_config")
    if plot_cfg_path and not os.path.isabs(plot_cfg_path):
        plot_cfg_path = os.path.normpath(os.path.join(cfg_dir, plot_cfg_path))

    # Resolve KV directory
    kv_dir = args.kv_dir or dcfg.get("data_cache_directory", "usgs_hourly_retrieval_logs")
    os.makedirs(kv_dir, exist_ok=True)

    # --- Try loading from cached KV file ---
    result = load_hourly_kv(kv_dir, hr_str)

    if result is not None:
        successes, failures = result
        print(f"[{hr_str}] Loaded from KV cache ({len(successes)} OK, {len(failures)} FAIL)")
    else:
        # --- No KV file: fetch from NWIS (mirrors USGSLoader behavior) ---
        print(f"[{hr_str}] No KV cache found. Fetching from NWIS...")
        meta = load_gauge_metadata(cfg)
        if meta.empty:
            print(f"[{hr_str}] No gauges after filtering. Skipping.")
            return 0

        # Fetch IV data for the full day and cache all 24 hours
        fetch_and_cache_all_hours(
            cfg, meta, kv_dir,
            args.year, args.month, args.day,
            min_readings=1,
        )

        # Now load the target hour's KV
        result = load_hourly_kv(kv_dir, hr_str)
        if result is None:
            print(f"[{hr_str}] KV still missing after fetch. Skipping.")
            return 0
        successes, failures = result
        print(f"[{hr_str}] Fetched and cached ({len(successes)} OK, {len(failures)} FAIL)")

    if not successes:
        print(f"[{hr_str}] No OK records. Skipping kriging.")
        return 0

    # Apply bbox filtering
    bbox = scfg.get("bbox")
    bbox_pad = float(scfg.get("bbox_pad_deg", 0.0))
    successes = filter_by_bbox(successes, bbox, bbox_pad)

    if not successes:
        print(f"[{hr_str}] No data within bbox. Skipping.")
        return 0

    data = [(lon, lat, mm, sid) for (lon, lat, mm, sid) in successes]
    print(f"[{hr_str}] {len(data)} observations → running kriging...")

    # Create USGSKrig with hour so filenames and attrs include HH
    krig = USGSKrig(data, args.config, args.year, args.month, args.day, hour=args.hour)
    krig.plot_config_path = plot_cfg_path

    # Run pipeline
    krig.compute_semivariogram()
    krig.compute_kriging()
    krig.plot_variogram()
    krig.map_krig_interpolation()
    interp_path, vario_path = krig.export_all()

    print(f"[{hr_str}] Exports:")
    print(f"  {interp_path}")
    print(f"  {vario_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
