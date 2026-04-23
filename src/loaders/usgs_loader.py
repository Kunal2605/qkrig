from __future__ import annotations

import io
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd

from core.base_loader import BaseLoader
import dataretrieval.nwis as nwis


# USGS timezone abbreviations → UTC offset (hours)
TZ_MAP = {
    "EST": -5, "EDT": -4,
    "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6,
    "PST": -8, "PDT": -7,
    "AKST": -9, "AKDT": -8,
    "HST": -10, "HDT": -9,
    "AST": -4, "ADT": -3,
}


class USGSLoader(BaseLoader):
    """
    Parallel USGS discharge loader with support for daily and 15-min IV data,
    plus post-load bounding box filtering.

    Modes:
      - Daily (default): get_streamflow(year, month, day)
      - Hourly IV:       get_streamflow(year, month, day, hour=H)
      - Single IV:       get_streamflow(year, month, day, hour=H, minute=M)

    Config keys:
      data:
        metadata_file, site_list_file, data_cache_directory
      settings:
        concurrency, max_retries, retry_backoff_seconds,
        min_area_km2, bbox, bbox_pad_deg
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        dcfg = self.config.get("data", {})
        scfg = self.config.get("settings", {})

        self.metadata_file: str = dcfg["metadata_file"]
        self.site_list_file: Optional[str] = dcfg.get("site_list_file")
        self.concurrency: int = int(scfg.get("concurrency", 16))
        self.max_retries: int = int(scfg.get("max_retries", 3))
        self.retry_backoff: float = float(scfg.get("retry_backoff_seconds", 0.75))
        self.min_area_km2: float = float(scfg.get("min_area_km2", 0.0))
        self.bbox: Optional[List[float]] = scfg.get("bbox")
        self.bbox_pad_deg: float = float(scfg.get("bbox_pad_deg", 0.0))
        self.logs_dir = dcfg.get("data_cache_directory") or "usgs_retrieval_logs"
        os.makedirs(self.logs_dir, exist_ok=True)

    def _load_gauge_metadata(self) -> pd.DataFrame:
        dcfg = self.config.get("data", {})
        scfg = self.config.get("settings", {})

        df = pd.read_csv(
            dcfg["metadata_file"], comment="#", dtype={"site_no": str},
            on_bad_lines="skip", engine="python",
        )
        df = df.rename(columns={
            "site_no": "gauge_id",
            "dec_lat_va": "gauge_lat",
            "dec_long_va": "gauge_lon",
            "drain_area_va": "area_sq_mi",
        })

        required = ["gauge_id", "gauge_lat", "gauge_lon", "area_sq_mi"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[USGSLoader] Missing columns: {missing}")
            return pd.DataFrame(columns=required).set_index("gauge_id")

        df = df[required].dropna()
        df["gauge_id"] = df["gauge_id"].str.zfill(8)

        site_list_file = dcfg.get("site_list_file")
        if site_list_file and os.path.exists(site_list_file):
            with open(site_list_file, "r") as f:
                wanted = {line.strip().lstrip("0") for line in f if line.strip()}
            df = df[df["gauge_id"].str.lstrip("0").isin(wanted)]

        add_random = int(scfg.get("add_random_sites", 0))
        if add_random > 0:
            all_sites = pd.read_csv(
                dcfg["metadata_file"], comment="#", dtype={"site_no": str},
                on_bad_lines="skip", engine="python",
            )
            all_sites = all_sites.rename(columns={
                "site_no": "gauge_id",
                "dec_lat_va": "gauge_lat",
                "dec_long_va": "gauge_lon",
                "drain_area_va": "area_sq_mi",
            })[required].dropna()
            current = set(df["gauge_id"])
            candidates = all_sites[~all_sites["gauge_id"].isin(current)]
            sample_n = min(add_random, len(candidates))
            if sample_n > 0:
                df = pd.concat([df, candidates.sample(n=sample_n, random_state=42)], ignore_index=True)

        min_area = float(scfg.get("min_area_km2", 0.0))
        if min_area > 0:
            df = df[(df["area_sq_mi"] * 2.58999) >= min_area]

        bbox = scfg.get("bbox")
        if bbox and len(bbox) == 4:
            min_lon, min_lat, max_lon, max_lat = map(float, bbox)
            df = df[
                (df["gauge_lon"] >= min_lon) & (df["gauge_lon"] <= max_lon)
                & (df["gauge_lat"] >= min_lat) & (df["gauge_lat"] <= max_lat)
            ]

        return df.set_index("gauge_id")

    def _log_path_for_date(self, date_str: str) -> str:
        return os.path.join(self.logs_dir, f"{date_str}.txt")

    def _kv_path_for_date(self, date_str: str) -> str:
        return os.path.join(self.logs_dir, f"{date_str}.kv.txt")

    def _save_kv_cache(self, date_str, successes, failures):
        path = self._kv_path_for_date(date_str)
        lines = [
            f"# KV cache for USGS retrieval on {date_str}",
            f"# Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        for lon, lat, mm, sid in sorted(successes, key=lambda r: r[3]):
            lines.append(f"{sid}=OK,{lon:.8f},{lat:.8f},{mm:.8f}")
        for sid, reason in sorted(failures, key=lambda r: r[0]):
            lines.append(f"{sid}=FAIL,{str(reason).replace(',', ';')}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _load_kv_cache(self, date_str):
        path = self._kv_path_for_date(date_str)
        if not os.path.exists(path):
            return None
        successes, failures = [], []
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
                        lon, lat, mm = float(parts[1]), float(parts[2]), float(parts[3])
                        successes.append((lon, lat, mm, key))
                    except Exception:
                        failures.append((key, "kv_parse_error"))
                elif status == "FAIL" and len(parts) >= 2:
                    failures.append((key, ",".join(parts[1:]).strip()))
                else:
                    failures.append((key, "kv_bad_line"))
        return successes, failures

    def _write_log(self, date_str, attempted_sites, successes, failures):
        path = self._log_path_for_date(date_str)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "USGS Retrieval Log",
            f"Timestamp: {ts}",
            f"Date queried: {date_str}",
            f"Total sites attempted: {len(attempted_sites)}",
            f"Successful: {len(successes)}",
            f"Unsuccessful: {len(failures)}",
            "",
            "=== Successful retrievals ===",
        ]
        if successes:
            for lon, lat, mm, sid in sorted(successes, key=lambda r: r[3]):
                lines.append(f"OK  site={sid}  lon={lon:.6f}  lat={lat:.6f}  streamflow={mm:.6f}")
        else:
            lines.append("(none)")
        lines += ["", "=== Unsuccessful retrievals ==="]
        if failures:
            for sid, reason in sorted(failures, key=lambda r: r[0]):
                lines.append(f"FAIL  site={sid}  reason={reason}")
        else:
            lines.append("(none)")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _filter_by_bbox(self, records):
        """Apply bbox filter to (lon, lat, mm, site_id) records."""
        if not records or not self.bbox or len(self.bbox) != 4:
            return records
        min_lon, min_lat, max_lon, max_lat = map(float, self.bbox)
        pad = float(self.bbox_pad_deg)
        min_lon -= pad; min_lat -= pad; max_lon += pad; max_lat += pad
        filtered = [
            (lon, lat, mm, sid) for (lon, lat, mm, sid) in records
            if (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)
        ]
        dropped = len(records) - len(filtered)
        if dropped > 0:
            print(f"[bbox] Dropped {dropped} records outside bbox")
        return filtered

    def _return_cached(self, cache_key):
        """Load from KV cache if available. Returns filtered list or None."""
        cached = self._load_kv_cache(cache_key)
        if cached is None:
            return None
        successes, _ = cached
        if not successes:
            print(f"[cache] No data for {cache_key}.")
            return []
        successes = self._filter_by_bbox(successes)
        if not successes:
            print(f"[cache] No data within bbox for {cache_key}.")
            return []
        arr = np.array(successes, dtype=[
            ("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U15")
        ])
        vals = arr["streamflow"]
        print(f"[cache] {cache_key}: {len(vals)} obs, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}, mean={np.mean(vals):.4f}")
        return arr.tolist()

    def _finalize_results(self, cache_key, sites, results, failures):
        """Write log/cache, apply bbox, return final list."""
        self._write_log(cache_key, attempted_sites=sites, successes=results, failures=failures)
        self._save_kv_cache(cache_key, successes=results, failures=failures)
        if not results:
            print(f"No data for {cache_key}.")
            return []
        results = self._filter_by_bbox(results)
        if not results:
            print(f"[bbox] No data within bbox for {cache_key}.")
            return []
        arr = np.array(results, dtype=[
            ("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U15")
        ])
        vals = arr["streamflow"]
        print(f"{cache_key}: {len(vals)} obs, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}, mean={np.mean(vals):.4f}")
        return arr.tolist()

    # --- Public API ---

    def get_streamflow(self, year, month, day, hour=None, minute=None):
        """
        Fetch discharge data for all gauges.

        Modes:
          - Daily:     get_streamflow(year, month, day)      -> List of (lon, lat, mm_day, id)
          - Hourly IV: get_streamflow(year, month, day, H)   -> List of (lon, lat, mm_hour, id)
          - Single IV: get_streamflow(year, month, day, H, M) -> List of (lon, lat, mm_15min, id)

        IV modes use the raw USGS REST API (same endpoint as fetch_usgs_iv_data)
        to ensure consistent timestamps with the bulk pipeline.

        For bulk IV data, use the offline pipeline:
          usgs_raw_iv.py -> usgs_raw_to_iv_kv.py -> .kv.txt cache
        """
        if hour is not None and minute is not None:
            return self._get_streamflow_iv(year, month, day, hour, minute)
        elif hour is not None:
            return self._get_streamflow_iv_hour(year, month, day, hour)
        else:
            return self._get_streamflow_daily(year, month, day)

    # --- Daily mode ---

    def _get_streamflow_daily(self, year, month, day):
        """Fetch daily mean discharge (mm/day) via nwis.get_record(service='dv')."""
        date_str = f"{year:04d}-{month:02d}-{day:02d}"

        from_cache = self._return_cached(date_str)
        if from_cache is not None:
            return from_cache

        if self.gauge_metadata.empty:
            self._write_log(date_str, [], [], [])
            self._save_kv_cache(date_str, [], [])
            print(f"No gauges to query for {date_str}.")
            return []

        sites = list(self.gauge_metadata.index.values)
        results, failures = [], []
        print(f"[DV] Fetching {date_str} ({len(sites)} sites)")

        def fetch_one(site_id):
            try:
                meta = self.gauge_metadata.loc[site_id]
            except KeyError:
                return (None, "missing_metadata", site_id)
            lon, lat, area_sq_mi = float(meta["gauge_lon"]), float(meta["gauge_lat"]), float(meta["area_sq_mi"])
            if not np.isfinite(area_sq_mi) or area_sq_mi <= 0:
                return (None, "invalid_area", site_id)
            attempt = 0
            sid = site_id.zfill(8)
            while True:
                try:
                    df = nwis.get_record(sites=sid, service="dv",
                                         start=date_str, end=date_str, parameterCd="00060")
                    if df is None or df.empty:
                        return (None, "nwis_empty", site_id)
                    cols = [c for c in df.columns if "00060" in c and "Mean" in c]
                    if not cols:
                        return (None, "missing_mean_col", site_id)
<<<<<<< Updated upstream

                    try:
                        cfs = float(df.iloc[0][cols[0]])
                    except Exception:
                        return (None, "bad_value", site_id)

=======
                    cfs = float(df.iloc[0][cols[0]])
>>>>>>> Stashed changes
                    if not np.isfinite(cfs) or cfs < 0:
                        return (None, "nonfinite_cfs", site_id)
                    # Square miles -> square meters
                    area_m2 = area_sq_mi * 2.58999e6
                    mm_day = (cfs * 0.0283168 * 86400.0 / area_m2) * 1000.0
                    if mm_day < -500 or mm_day > 500:
                        return (None, "large_magnitude_flow", site_id)
                    return ((lon, lat, mm_day, site_id), "ok", site_id)
                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        return (None, f"exception:{str(e)[:50]}", site_id)
                    time.sleep(self.retry_backoff * (1 + random.random()) * attempt)

        completed, total = 0, len(sites)
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(fetch_one, sid): sid for sid in sites}
            for fut in as_completed(futures):
                rec, status, sid = fut.result()
                completed += 1
                if rec is not None and status == "ok":
                    results.append(rec)
                else:
                    failures.append((sid, status))
                if completed % 100 == 0 or completed == total:
                    print(f"[DV] {completed}/{total}")

        return self._finalize_results(date_str, sites, results, failures)

    # --- IV helper: fetch full day via raw USGS REST API ---

    def _fetch_day_iv(self, site_id: str, date_str: str) -> Optional[pd.DataFrame]:
        """
        Fetch all IV timestamps for one site and one day using the raw USGS
        waterservices REST API (RDB format). Same endpoint as fetch_usgs_iv_data.

        Returns DataFrame with columns ['datetime_utc', 'cfs'] or None on failure.
        Timestamps are converted from local time (as returned by RDB) to UTC.
        """
        sid = site_id.zfill(8)
        url = "https://waterservices.usgs.gov/nwis/iv/"
        params = {
            "format": "rdb",
            "sites": sid,
            "parameterCd": "00060",
            "startDT": date_str,
            "endDT": date_str,
            "siteStatus": "all",
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except Exception:
            return None

        # Parse RDB: skip comment lines (#) and the format row
        lines = resp.text.splitlines()
        header = None
        data_lines = []
        for line in lines:
            if line.startswith("#"):
                continue
            if header is None:
                header = line
                continue
            # Skip format row (e.g. "5s\t15s\t20d\t6s\t14n\t10s")
            stripped = line.replace("\t", " ").strip()
            if stripped and all(c.isdigit() or c in "sdn " for c in stripped):
                continue
            data_lines.append(line)

        if not header or not data_lines:
            return None

        csv_text = header + "\n" + "\n".join(data_lines)
        try:
            df = pd.read_csv(io.StringIO(csv_text), sep="\t", dtype=str)
        except Exception:
            return None

        # Find datetime and discharge columns
        dt_col = next((c for c in df.columns if "datetime" in c.lower()), None)
        flow_cols = [c for c in df.columns if "00060" in c and "_cd" not in c]
        if not dt_col or not flow_cols:
            return None

        # RDB 'datetime' is naive local wall-clock; use tz_cd to shift to UTC
        dt_local = pd.to_datetime(df[dt_col], errors="coerce")
        if "tz_cd" in df.columns:
            tz = df["tz_cd"].astype(str).str.strip()
            offset_hrs = tz.map(TZ_MAP)
            offset_hrs = offset_hrs.fillna(0)
            df["datetime_utc"] = (dt_local - pd.to_timedelta(offset_hrs, unit="h")).dt.tz_localize("UTC")
        else:
            df["datetime_utc"] = dt_local.dt.tz_localize("UTC")

        df["cfs"] = pd.to_numeric(df[flow_cols[0]], errors="coerce")
        df.loc[df["cfs"] < 0, "cfs"] = np.nan
        df = df.dropna(subset=["datetime_utc", "cfs"])

        if df.empty:
            return None

        return df[["datetime_utc", "cfs"]].reset_index(drop=True)

    # --- IV mode: single timestamp (nearest match) ---

    def _get_streamflow_iv(self, year, month, day, hour, minute):
        """
        Fetch IV discharge (mm/15min) for a single timestamp.
        Uses raw USGS REST API, finds the nearest available timestamp
        within a 7.5-minute window of the target.
        """
        target = pd.Timestamp(year=year, month=month, day=day,
                              hour=hour, minute=minute, tz='UTC')
        ts_str = target.strftime("%Y-%m-%d_%H-%M")
        date_str = f"{year:04d}-{month:02d}-{day:02d}"

        from_cache = self._return_cached(ts_str)
        if from_cache is not None:
            return from_cache

        if self.gauge_metadata.empty:
            self._write_log(ts_str, [], [], [])
            self._save_kv_cache(ts_str, [], [])
            print(f"No gauges to query for {ts_str}.")
            return []

        sites = list(self.gauge_metadata.index.values)
        results, failures = [], []
        print(f"[IV] Fetching {ts_str} ({len(sites)} sites)")

        def fetch_one(site_id):
            try:
                meta = self.gauge_metadata.loc[site_id]
            except KeyError:
                return (None, "missing_metadata", site_id)
            lon, lat, area_sq_mi = float(meta["gauge_lon"]), float(meta["gauge_lat"]), float(meta["area_sq_mi"])
            if not np.isfinite(area_sq_mi) or area_sq_mi <= 0:
                return (None, "invalid_area", site_id)

            attempt = 0
            while True:
                try:
                    df = self._fetch_day_iv(site_id, date_str)
                    if df is None or df.empty:
                        return (None, "no_iv_data", site_id)

                    # Find nearest timestamp within 7.5-minute window
                    diffs = abs(df["datetime_utc"] - target)
                    nearest_idx = diffs.idxmin()
                    if diffs[nearest_idx] > pd.Timedelta(minutes=7, seconds=30):
                        return (None, "no_matching_time", site_id)

                    cfs = float(df.loc[nearest_idx, "cfs"])
                    if not np.isfinite(cfs) or cfs < 0:
                        return (None, "nonfinite_cfs", site_id)

                    # Square miles -> square meters
                    area_m2 = area_sq_mi * 2.58999e6
                    mm_15min = (cfs * 0.0283168 * 900.0 / area_m2) * 1000.0
                    if mm_15min < -10 or mm_15min > 100:
                        return (None, "large_magnitude_flow", site_id)
                    return ((lon, lat, mm_15min, site_id), "ok", site_id)
                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        return (None, f"exception:{str(e)[:50]}", site_id)
                    time.sleep(self.retry_backoff * (1 + random.random()) * attempt)

        completed, total = 0, len(sites)
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(fetch_one, sid): sid for sid in sites}
            for fut in as_completed(futures):
                rec, status, sid = fut.result()
                completed += 1
                if rec is not None and status == "ok":
                    results.append(rec)
                else:
                    failures.append((sid, status))
                if completed % 100 == 0 or completed == total:
                    print(f"[IV] {completed}/{total}")

        return self._finalize_results(ts_str, sites, results, failures)

    # --- IV mode: hourly mean ---

    def _get_streamflow_iv_hour(self, year, month, day, hour):
        """
        Fetch IV discharge for all timestamps in a given hour, compute
        the mean, and return a flat list of (lon, lat, mm_hour, gauge_id).

        Uses raw USGS REST API. Averages all available timestamps in the
        target hour (handles 5-min, 15-min, and offset timestamps).
        """
        ts_str = f"{year:04d}-{month:02d}-{day:02d}_H{hour:02d}"
        date_str = f"{year:04d}-{month:02d}-{day:02d}"

        from_cache = self._return_cached(ts_str)
        if from_cache is not None:
            return from_cache

        if self.gauge_metadata.empty:
            self._write_log(ts_str, [], [], [])
            self._save_kv_cache(ts_str, [], [])
            print(f"No gauges to query for {ts_str}.")
            return []

        sites = list(self.gauge_metadata.index.values)
        results, failures = [], []
        print(f"[IV] Fetching hour {hour:02d} mean ({len(sites)} sites)")

        def fetch_one(site_id):
            try:
                meta = self.gauge_metadata.loc[site_id]
            except KeyError:
                return (None, "missing_metadata", site_id)
            lon, lat, area_sq_mi = float(meta["gauge_lon"]), float(meta["gauge_lat"]), float(meta["area_sq_mi"])
            if not np.isfinite(area_sq_mi) or area_sq_mi <= 0:
                return (None, "invalid_area", site_id)

            attempt = 0
            while True:
                try:
                    df = self._fetch_day_iv(site_id, date_str)
                    if df is None or df.empty:
                        return (None, "no_iv_data", site_id)

                    # Filter to timestamps within the target hour
                    hour_mask = df["datetime_utc"].dt.hour == hour
                    hour_data = df[hour_mask]
                    if hour_data.empty:
                        return (None, "no_data_in_hour", site_id)

                    # Mean CFS across all timestamps in the hour
                    mean_cfs = hour_data["cfs"].mean()
                    if not np.isfinite(mean_cfs) or mean_cfs < 0:
                        return (None, "nonfinite_cfs", site_id)

                    # Square miles -> square meters
                    area_m2 = area_sq_mi * 2.58999e6
                    mm_hour = (mean_cfs * 0.0283168 * 3600.0 / area_m2) * 1000.0
                    if mm_hour < -40 or mm_hour > 400:
                        return (None, "large_magnitude_flow", site_id)
                    return ((lon, lat, mm_hour, site_id), "ok", site_id)
                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        return (None, f"exception:{str(e)[:50]}", site_id)
                    time.sleep(self.retry_backoff * (1 + random.random()) * attempt)

        completed, total = 0, len(sites)
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(fetch_one, sid): sid for sid in sites}
            for fut in as_completed(futures):
                rec, status, sid = fut.result()
                completed += 1
                if rec is not None and status == "ok":
                    results.append(rec)
                else:
                    failures.append((sid, status))
                if completed % 100 == 0 or completed == total:
                    print(f"[IV] {completed}/{total}")

        return self._finalize_results(ts_str, sites, results, failures)

