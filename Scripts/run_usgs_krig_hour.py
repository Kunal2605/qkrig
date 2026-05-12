#!/usr/bin/env python3
"""
Run USGS kriging for a single HOUR — KV-cache-only.

This script no longer fetches from NWIS. The hourly KV file at
    <kv-dir>/YYYY-MM-DD_HH.kv.txt
is a hard precondition. Populate it upstream with:
    Scripts/usgs_raw_to_hourly_bulk.py  (raw .rdb per site)
    Scripts/usgs_raw_to_hourly_kv.py    (raw → hourly KV, UTC-aware)

The docker entrypoint (Scripts/run_qkrig_hourly.sh) calls those two scripts
in its Stage 0 automatically, so the bulk → KV → krig flow is end-to-end.

Exports are named with the hour:
    interp_2023-05-01_14.nc       variogram_2023-05-01_14.csv

Usage:
    python Scripts/run_usgs_krig_hour.py \\
        --config configs/usgsgaugekrig.yaml \\
        --kv-dir /path/to/hourly_kv_output/ \\
        --year 2023 --month 5 --day 1 --hour 14
"""

from __future__ import annotations

import argparse
import os
import sys
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
    df["gauge_id"] = df["gauge_id"].str.zfill(8)

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
        # Re-apply metadata filter so config changes (min_area_km2, bbox,
        # site_list) take effect on cached KV without forcing a re-fetch.
        meta_check = load_gauge_metadata(cfg)
        valid_ids = set(meta_check.index.values)
        before = len(successes)
        kept, dropped = [], 0
        for rec in successes:
            sid = rec[3]
            if sid in valid_ids or sid.zfill(8) in valid_ids:
                kept.append(rec)
            else:
                failures.append((sid, "filtered_by_metadata"))
                dropped += 1
        successes = kept
        if dropped:
            print(f"[{hr_str}] Filtered {dropped} cached sites via metadata "
                  f"(min_area_km2 / bbox / site_list)")
        print(f"[{hr_str}] Loaded from KV cache ({len(successes)} OK, {len(failures)} FAIL)")
    else:
        # KV cache is a precondition — Stage 0 in run_qkrig_hourly.sh populates
        # it via usgs_raw_to_hourly_bulk.py + usgs_raw_to_hourly_kv.py.
        print(
            f"ERROR: [{hr_str}] No KV file at {kv_file_path(kv_dir, hr_str)}. "
            f"Populate the cache first via Stage 0 of run_qkrig_hourly.sh, or "
            f"run usgs_raw_to_hourly_bulk.py + usgs_raw_to_hourly_kv.py directly.",
            file=sys.stderr,
        )
        return 2

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
    # USGSKrig.__init__ already built `variogram_plotter` and `krig_map_plotter`
    # off the kriging YAML's `plot_config:` value (the notebook config). If the
    # caller passed --plot-config (e.g. the Docker entrypoint pointing at the
    # save-mode plot_config_docker.yaml), the plotters need to be rebuilt with
    # the override before any plot method runs — otherwise their cached
    # PlotConfig keeps `save_plots: false` and no PNGs get written.
    if plot_cfg_path:
        krig.plot_config_path = plot_cfg_path
        from vis.visualizations import VariogramPlotter, KrigingMapPlotter
        krig.variogram_plotter = VariogramPlotter(krig)
        krig.krig_map_plotter = KrigingMapPlotter(krig)

    # Run pipeline
    krig.compute_semivariogram()
    krig.compute_kriging()
    # One composite PNG per hour (map + variogram side-by-side, polished style)
    # instead of the older split into kriging_interp_*.png + variogram_*.png.
    krig.plot_interpolation_with_variogram()
    interp_path, vario_path = krig.export_all()

    print(f"[{hr_str}] Exports:")
    print(f"  {interp_path}")
    print(f"  {vario_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
