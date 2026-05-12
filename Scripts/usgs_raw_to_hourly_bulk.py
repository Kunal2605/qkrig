import requests
import time
import os

def read_gauge_ids_from_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    # Strip whitespace and skip empty lines
    return [line.strip() for line in lines if line.strip()]

def fetch_usgs_iv_data(site_no, start_date, end_date, format="rdb"):
    """
    Fetch USGS instantaneous values for a single site.
    """
    base_url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": format,
        "sites": site_no,
        "parameterCd": "00060,00065",  # discharge and stage
        "startDT": start_date,
        "endDT": end_date,
        "siteStatus": "all"
    }

    try:
        response = requests.get(base_url, params=params, timeout=120)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching {site_no}: {e}")
        return None

def download_all_gauges(gauge_file, output_dir, start_date, end_date, format="rdb", workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    os.makedirs(output_dir, exist_ok=True)
    gauges = read_gauge_ids_from_file(gauge_file)

    # Filter out already-downloaded sites
    to_download = []
    for site_no in gauges:
        out_path = os.path.join(output_dir, f"{site_no}_iv.{format}")
        if os.path.exists(out_path):
            print(f"⏭Skipping {site_no} (already downloaded)")
        else:
            to_download.append(site_no)

    print(f"\n{len(to_download)} sites to download ({len(gauges) - len(to_download)} already done), using {workers} workers\n")

    def _fetch_and_save(site_no):
        out_path = os.path.join(output_dir, f"{site_no}_iv.{format}")
        data = fetch_usgs_iv_data(site_no, start_date, end_date, format=format)
        if data:
            with open(out_path, "w") as f:
                f.write(data)
            return (site_no, True)
        return (site_no, False)

    done = 0
    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_and_save, sid): sid for sid in to_download}
        for fut in as_completed(futures):
            sid = futures[fut]
            _, success = fut.result()
            done += 1
            if success:
                ok += 1
                print(f"[{done}/{len(to_download)}] Saved {sid}")
            else:
                fail += 1
                print(f"[{done}/{len(to_download)}] Failed {sid}")

    print(f"\n Done: {ok} saved, {fail} failed out of {len(to_download)}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Bulk-download USGS IV (instantaneous values) as raw .rdb per site."
    )
    p.add_argument("--site-list", required=True,
                   help="Text file with one USGS site ID per line.")
    p.add_argument("--out-dir", required=True,
                   help="Directory to write <site>_iv.rdb files (created if missing).")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--workers", type=int, default=16,
                   help="Concurrent HTTP workers (default: 16)")
    p.add_argument("--format", default="rdb", choices=["rdb", "json"],
                   help="NWIS response format (default: rdb)")
    args = p.parse_args()

    download_all_gauges(
        gauge_file=args.site_list,
        output_dir=args.out_dir,
        start_date=args.start,
        end_date=args.end,
        format=args.format,
        workers=args.workers,
    )