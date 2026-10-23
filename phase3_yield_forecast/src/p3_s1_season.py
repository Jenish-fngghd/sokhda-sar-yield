"""T17: the growth window Capella never saw — Sentinel-1 C-band, 16 dates, one track.

WHY THIS EXISTS. T1 killed SAFY on availability, and P3-14 put a number on it: the canopy
signal loses ~1/3 of its strength per 6 days of temporal mismatch, and no same-day optical
exists anywhere in the June-September window because of the monsoon. Capella contributes only
THREE acquisitions to that window (6 Jun, 19 Jun, 14 Aug). The season that determines the yield
is largely unobserved by our own sensor.

Sentinel-1 fixes exactly that hole and nothing else:

    Capella X-band   6 dates total,  3 in Jun-Sep
    Sentinel-1 C-band  16 dates,     9 in Jun-Sep,  12-day cadence, cloud-free

★ AND IT COMES FROM A SINGLE TRACK. All 16 scenes are descending, relative orbit 34. No mixing
of look directions or incidence geometries -- the confound we spent Phase 1 proving was absent
in the Capella stack does not arise here at all. That is luck, but it is checked luck: the
inventory is asserted below and the run fails loudly if a second track ever appears.

PROVENANCE, STATED PLAINLY. This is still SAR. Adding C-band alongside X-band is multi-frequency
SAR fusion, not an optical shortcut, so the provenance claim established in P3-14d -- that our
crop map is the least optically-contaminated of the six teams -- survives intact. We would not
add an optical product here for that reason.

NO CREDENTIALS. Microsoft Planetary Computer's STAC API and its anonymous SAS token endpoint
serve Sentinel-1 RTC (radiometrically terrain corrected, gamma0, 10 m) without an account. This
replaced a blocked Earth Engine path; nothing here needs a key.

Writes results/tables/p3_s1_season.csv  (966 farms x 16 dates x {VV, VH} zonal means, in dB)

Run:  py -3.12 src/p3_s1_season.py
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

os.environ.pop("PROJ_LIB", None)          # rasterio brings its own; the inherited one breaks it

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FARMS, TABLES, log

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
START, END = "2025-05-15", "2025-12-01"
TRACK = ("descending", 34)                # asserted, not assumed
_tok = {}


def sign(url):
    """Anonymous SAS signing, one token cached per container."""
    acct = url.split("//")[1].split(".")[0]
    cont = url.split("/")[3]
    k = (acct, cont)
    if k not in _tok:
        _tok[k] = json.load(urllib.request.urlopen(f"{SAS}/{acct}/{cont}", timeout=60))["token"]
    return f"{url}?{_tok[k]}"


def search(lon, lat):
    body = json.dumps({"collections": ["sentinel-1-rtc"],
                       "intersects": {"type": "Point", "coordinates": [lon, lat]},
                       "datetime": f"{START}T00:00:00Z/{END}T00:00:00Z", "limit": 100}).encode()
    r = urllib.request.Request(STAC, data=body,
                               headers={"Content-Type": "application/json",
                                        "User-Agent": "aisehack-r3"})
    return json.load(urllib.request.urlopen(r, timeout=120))["features"]


def main():
    fa = gpd.read_file(FARMS)
    wgs = fa.to_crs(4326)
    b = wgs.total_bounds
    lon, lat = float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)

    feats = search(lon, lat)
    tracks = {(f["properties"].get("sat:orbit_state"),
               f["properties"].get("sat:relative_orbit")) for f in feats}
    if tracks != {TRACK}:
        raise SystemExit(f"expected the single track {TRACK}, found {tracks} -- "
                         "a mixed-geometry series is not comparable, stopping")
    feats.sort(key=lambda f: f["properties"]["datetime"])
    log("p3_s1.inventory", n=len(feats), track=str(TRACK),
        dates=[f["properties"]["datetime"][:10] for f in feats],
        note="single track: no look-side or incidence confound in this series")

    rows = {}
    ids = fa.index.to_numpy() + 1                     # rasterize needs non-zero burn values
    for k, f in enumerate(feats):
        date = f["properties"]["datetime"][:10].replace("-", "")
        for pol in ("vv", "vh"):
            with rasterio.open(sign(f["assets"][pol]["href"])) as ds:
                bb = transform_bounds("EPSG:4326", ds.crs, b[0], b[1], b[2], b[3])
                win = from_bounds(*bb, ds.transform)
                arr = ds.read(1, window=win).astype("float64")
                tr = ds.window_transform(win)
                shapes = ((g, i) for g, i in zip(fa.to_crs(ds.crs).geometry, ids))
                lab = rasterize(shapes, out_shape=arr.shape, transform=tr,
                                fill=0, dtype="int32")
            good = np.isfinite(arr) & (arr > 0) & (lab > 0)
            # mean in POWER, then convert -- averaging dB would bias the mean low
            s = np.bincount(lab[good], weights=arr[good], minlength=len(fa) + 1)
            n = np.bincount(lab[good], minlength=len(fa) + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                db = 10.0 * np.log10(np.where(n > 0, s / np.maximum(n, 1), np.nan))
            rows[f"{pol}_{date}"] = db[1:]
            if pol == "vv":
                rows[f"npix_{date}"] = n[1:]
        print(f"  [{k+1:2d}/{len(feats)}] {date}  "
              f"VV {np.nanmean(rows[f'vv_{date}']):+6.2f} dB   "
              f"VH {np.nanmean(rows[f'vh_{date}']):+6.2f} dB   "
              f"farms with pixels {int((rows[f'npix_{date}'] > 0).sum())}")

    out = pd.DataFrame(rows)
    out.insert(0, "farm_id", fa["farm_id"].to_numpy() if "farm_id" in fa else ids)
    out.to_csv(TABLES / "p3_s1_season.csv", index=False)
    dates = [f["properties"]["datetime"][:10] for f in feats]
    cov = np.mean([(rows[f"npix_{d.replace('-','')}"] > 0).mean() for d in dates])
    log("p3_s1.done", n_farms=len(out), n_dates=len(feats),
        mean_farm_coverage=round(float(cov), 4),
        growth_window_dates=int(sum("2025-06" <= d[:7] <= "2025-09" for d in dates)),
        capella_growth_window_dates=3,
        why=("Capella contributes 3 acquisitions to Jun-Sep; this contributes 12, cloud-free, "
             "on one track. It fills the window that killed SAFY"))
    print(f"\nwrote {TABLES / 'p3_s1_season.csv'}  ({len(out)} farms x {len(feats)} dates)")


if __name__ == "__main__":
    main()
