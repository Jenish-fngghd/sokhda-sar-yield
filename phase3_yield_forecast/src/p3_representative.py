"""T21: is Sokhda TYPICAL of the district whose yield statistic we borrow?

★ THE UNEXAMINED ASSUMPTION UNDER OUR DOMINANT UNCERTAINTY.

The anchor is a Vadodara DISTRICT statistic applied to ONE VILLAGE, and it carries 385.2 t of
the village-total interval -- more than everything else combined. T19's climatology tested the
YEAR adjustment (was 2025 ordinary at Sokhda? yes, z = -0.16). It never touched the assumption
underneath:

    village yield  ==  district average yield

Nobody asks this, and it is a directional error if wrong. A district average applied to a
village that is systematically poorer than its district OVERSTATES production, and no amount of
uncertainty widening fixes a bias -- it only hides it in a wider interval.

Radar can test it, because the comparison is RELATIVE and relative is the one thing X-band and
C-band can do without labels (section 4.1).

THE DESIGN, AND WHY IT IS BLOCK-MATCHED. Comparing Sokhda's mean to individual pixels in the
surrounding landscape would compare two different things -- a 447 ha average against 100 m^2
samples, whose spreads are not comparable. So the surrounding area is cut into blocks of
SOKHDA'S OWN SIZE (~2.1 km, ~450 ha), each block averaged over its CROPLAND pixels only
(ESA WorldCover 2021, class 40). Sokhda is then one draw from a distribution of like-sized,
like-land-use neighbours, and its PERCENTILE is a meaningful statement.

★ THE CONTROL THAT MAKES IT A FINDING RATHER THAN A NUMBER. One year's percentile could be
weather, noise, or a single bad scene. So the whole thing is repeated for every year 2017-2025.
If Sokhda's rank among its neighbours is STABLE across nine independent seasons, it is a
structural property of the village and it bears on the anchor. If the rank wanders, it is noise
and we say so and stop.

HONEST LIMIT, STATED UP FRONT: VH backscatter is a canopy-density proxy, not yield. A stable
offset is evidence of a systematic difference in canopy, which is suggestive about yield and not
equivalent to it. This is reported as a directional flag on the anchor, never as a correction.

Writes results/tables/p3_representative.csv.

Run:  py -3.12 src/p3_representative.py
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FARMS, TABLES, log

import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
TRACK = ("descending", 34)
PAD = 0.20                 # degrees around Sokhda, ~22 km
BLOCK_M = 2100             # Sokhda is ~447 ha; a 2.1 km block is ~441 ha
DECIM = 2                  # read at 20 m; block means do not need 10 m
MAX_SCENES = 5             # per year, evenly spaced across Jun-Sep
CROPLAND = 40              # ESA WorldCover class
_tok = {}


def sign(url, force=False):
    """Sign a Planetary Computer asset URL.

    ★ Tokens EXPIRE. Caching one for the life of the process is fine for a 90-second run and
    fatal for a nine-year one: the first attempt at this died mid-loop with
    'IReadBlock failed ... Cannot read offset/size for strile' on a URL whose se= had already
    passed. Refresh on age, and allow a forced refresh after a read failure.
    """
    import time
    acct, cont = url.split("//")[1].split(".")[0], url.split("/")[3]
    k = (acct, cont)
    tok, born = _tok.get(k, (None, 0.0))
    if tok is None or force or (time.time() - born) > 1200:
        tok = json.load(urllib.request.urlopen(f"{SAS}/{acct}/{cont}", timeout=60))["token"]
        _tok[k] = (tok, time.time())
    return f"{url}?{tok}"


def read_window(href, box, dtype="float32"):
    """Windowed read with one re-signed retry, because a long run outlives its token."""
    for attempt in (0, 1):
        try:
            with rasterio.open(sign(href, force=bool(attempt))) as ds:
                wb = transform_bounds("EPSG:4326", ds.crs, *box)
                w = from_bounds(*wb, ds.transform)
                return (ds.read(1, window=w, boundless=True, fill_value=0).astype(dtype),
                        ds.window_transform(w), ds.crs)
        except Exception as e:                                    # noqa: BLE001
            if attempt:
                raise
            print(f"    read failed ({type(e).__name__}); re-signing and retrying")


def search(coll, geom, dt=None, limit=100):
    q = {"collections": [coll], "intersects": geom, "limit": limit}
    if dt:
        q["datetime"] = dt
    r = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                               headers={"Content-Type": "application/json",
                                        "User-Agent": "aisehack-r3"})
    return json.load(urllib.request.urlopen(r, timeout=120))["features"]


def main():
    fa = gpd.read_file(FARMS)
    fb = fa.to_crs(4326).total_bounds
    box = [fb[0] - PAD, fb[1] - PAD, fb[2] + PAD, fb[3] + PAD]
    poly = {"type": "Polygon", "coordinates": [[[box[0], box[1]], [box[2], box[1]],
                                                [box[2], box[3]], [box[0], box[3]],
                                                [box[0], box[1]]]]}
    cen = {"type": "Point", "coordinates": [float((fb[0] + fb[2]) / 2),
                                            float((fb[1] + fb[3]) / 2)]}

    # ---------- reference grid: from one S1 scene, so everything shares it ----------
    ref = search("sentinel-1-rtc", cen, "2025-08-01T00:00:00Z/2025-08-31T00:00:00Z", 1)[0]
    with rasterio.open(sign(ref["assets"]["vh"]["href"])) as ds:
        bb = transform_bounds("EPSG:4326", ds.crs, *box)
        win = from_bounds(*bb, ds.transform)
        H = int(win.height) // DECIM
        W = int(win.width) // DECIM
        grid_crs, grid_tr = ds.crs, ds.window_transform(win) * rasterio.Affine.scale(DECIM)
    print(f"grid {W} x {H} at {abs(grid_tr.a):.0f} m in {grid_crs}")

    # ---------- cropland mask ----------
    wc = [f for f in search("esa-worldcover", poly) if "2021" in f["id"]]
    crop = np.zeros((H, W), dtype="uint8")
    for f in wc:
        # ★ window-read FIRST. Passing rasterio.band() to reproject pulls the whole source
        # raster over the network -- a 3-degree WorldCover tile at 10 m, and for S1 a
        # 21701 x 28512 scene. Gigabytes per read, and why the first attempt printed nothing.
        src, stf, scrs = read_window(f["assets"]["map"]["href"], box, dtype="uint8")
        dst = np.zeros((H, W), dtype="uint8")
        reproject(source=src, destination=dst, src_transform=stf, src_crs=scrs,
                  dst_transform=grid_tr, dst_crs=grid_crs, resampling=Resampling.nearest)
        crop = np.maximum(crop, (dst == CROPLAND).astype("uint8"))
    print(f"cropland covers {100*crop.mean():.1f}% of the {2*PAD:.2f}° box "
          f"({len(wc)} WorldCover tile(s))")

    # ---------- Sokhda mask, and the block grid ----------
    sok = rasterize(((g, 1) for g in fa.to_crs(grid_crs).geometry),
                    out_shape=(H, W), transform=grid_tr, fill=0, dtype="uint8")
    bpx = max(int(round(BLOCK_M / abs(grid_tr.a))), 1)
    by, bx = np.indices((H, W))
    block = (by // bpx) * (W // bpx + 1) + (bx // bpx)
    # a block is usable only if it is mostly cropland and does not touch Sokhda
    nb = int(block.max()) + 1
    n_crop = np.bincount(block.ravel(), weights=crop.ravel(), minlength=nb)
    n_all = np.bincount(block.ravel(), minlength=nb)
    n_sok = np.bincount(block.ravel(), weights=sok.ravel(), minlength=nb)
    usable = (n_all > 0.6 * bpx * bpx) & (n_crop > 0.5 * n_all) & (n_sok == 0)
    print(f"blocks {bpx*abs(grid_tr.a):.0f} m: {int(usable.sum())} usable cropland neighbours "
          f"(of {nb}), Sokhda covers {int(sok.sum())} px")

    rows = []
    for year in range(2017, 2026):
        fs = [f for f in search("sentinel-1-rtc", cen,
                                f"{year}-06-01T00:00:00Z/{year}-09-30T00:00:00Z")
              if (f["properties"].get("sat:orbit_state"),
                  f["properties"].get("sat:relative_orbit")) == TRACK]
        fs = sorted(fs, key=lambda f: f["properties"]["datetime"])
        fs = fs[:: max(1, len(fs) // MAX_SCENES)][:MAX_SCENES]
        acc = np.zeros((H, W))
        cnt = np.zeros((H, W))
        for f in fs:
            src, stf, scrs = read_window(f["assets"]["vh"]["href"], box)
            dst = np.full((H, W), np.nan, dtype="float32")
            reproject(source=src, destination=dst, src_transform=stf, src_crs=scrs,
                      dst_transform=grid_tr, dst_crs=grid_crs,
                      resampling=Resampling.average, src_nodata=0, dst_nodata=np.nan)
            good = np.isfinite(dst) & (dst > 0)
            acc[good] += dst[good]
            cnt[good] += 1
        mean_lin = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

        m = crop.astype(bool) & np.isfinite(mean_lin)
        s = sok.astype(bool) & np.isfinite(mean_lin)
        bs = np.bincount(block[m], weights=mean_lin[m], minlength=nb)
        bn = np.bincount(block[m], minlength=nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            bmean = 10 * np.log10(np.where(bn > 0, bs / np.maximum(bn, 1), np.nan))
        neigh = bmean[usable & (bn > 0)]
        sok_db = 10 * np.log10(float(np.mean(mean_lin[s])))
        pct = 100.0 * float((neigh < sok_db).mean())
        rows.append({"year": year, "n_scenes": len(fs), "sokhda_vh_db": round(sok_db, 3),
                     "neighbour_median_db": round(float(np.median(neigh)), 3),
                     "neighbour_n": int(len(neigh)),
                     "offset_db": round(sok_db - float(np.median(neigh)), 3),
                     "percentile": round(pct, 1)})
        print(f"  {year}  n={len(fs)}  Sokhda {sok_db:+6.2f}  neighbours "
              f"{np.median(neigh):+6.2f} (n={len(neigh)})  offset {sok_db-np.median(neigh):+5.2f} dB"
              f"  percentile {pct:5.1f}")

    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "p3_representative.csv", index=False)

    off_mu, off_sd = t.offset_db.mean(), t.offset_db.std(ddof=1)
    pct_mu, pct_sd = t.percentile.mean(), t.percentile.std(ddof=1)
    stable = (off_sd < 0.5) and (np.sign(t.offset_db) == np.sign(off_mu)).all()
    print(f"\n  offset across nine years   {off_mu:+.2f} ± {off_sd:.2f} dB")
    print(f"  percentile across nine years {pct_mu:.0f} ± {pct_sd:.0f}")
    print("  CONTROL " + ("HELD — the sign is the same in every one of nine seasons, so this "
                          "is structural" if stable else
                          "FAILED — the rank wanders between years; this is noise, not a "
                          "property of the village"))
    log("p3_repr.done", years=len(t), block_m=BLOCK_M, neighbours=int(t.neighbour_n.iloc[-1]),
        offset_db_mean=round(float(off_mu), 3), offset_db_sd=round(float(off_sd), 3),
        percentile_mean=round(float(pct_mu), 1), percentile_sd=round(float(pct_sd), 1),
        stable_sign_all_years=bool(stable),
        bears_on=("the anchor is a DISTRICT statistic applied to ONE VILLAGE and carries "
                  "385.2 t of the interval. If Sokhda sits consistently off its neighbours, "
                  "that is a DIRECTIONAL bias, which a wider interval hides rather than fixes"),
        honest_limit=("VH is a canopy-density proxy, not yield. Reported as a directional flag "
                      "on the anchor, never applied as a correction"))


if __name__ == "__main__":
    main()
