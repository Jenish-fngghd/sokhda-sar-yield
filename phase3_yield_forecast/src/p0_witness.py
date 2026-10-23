"""Phase 0: fetch the full-season witness stacks, per farm, THROUGH HARVEST.

Round 2's witnesses stopped on 13 October and cannot speak to either new Capella
date. A forecast that extends past 13 October needs a witness that also extends
past it. This fetches:

  * Sentinel-1 RTC, C-band VV+VH, every scene 1 May - 15 Dec 2025 on the single
    relative orbit that covers Sokhda (34, descending). Cloud-immune, 12-day.
  * Sentinel-2 L2A NDVI for every scene under 20% cloud in the same window --
    which the census showed is 20 scenes in Oct-Nov alone, including a SAME-DAY
    0.001%-cloud scene on 12 November, the last Capella date.

WITNESS DISCIPLINE. Neither stack enters a shipped number. They exist to answer
"does an unrelated satellite see what our X-band product claims?", and that answer
is only worth something while they remain unread by the model. Promoting either
requires naming and freezing a replacement witness first. [prompt 5.1]

Slow: ~30 scenes x 966 polygons read over HTTP. Run it in the background.
Writes results/tables/witness_s1.csv and witness_s2.csv.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FARMS, TABLES, VILLAGE, log
from stac import search, sign

WINDOW = "2025-05-01/2025-12-15"
MAX_CLOUD = 20.0
SCL_KEEP = {4, 5, 6, 7}      # vegetation, bare, water, unclassified -- not cloud/shadow/snow


def zonal(href, geoms, band=1):
    """Median of a COG inside each farm polygon, read straight over HTTP.

    Median, not mean: a handful of mixed edge pixels at 10 m on a 0.27 ha field
    drags a mean, and every farm here is small relative to the pixel.
    """
    out = np.full(len(geoms), np.nan)
    with rasterio.open(sign(href)) as src:
        gg = geoms.to_crs(src.crs)
        for i, geom in enumerate(gg.geometry):
            try:
                arr, _ = mask(src, [geom], crop=True, filled=True, nodata=0, indexes=band)
            except ValueError:
                continue                                  # polygon outside the tile
            v = arr.astype("float64")
            v[v == 0] = np.nan
            if np.isfinite(v).any():
                out[i] = np.nanmedian(v)
    return out


def zonal_valid_frac(href, geoms):
    """Fraction of each farm's SCL pixels in a keep class -- the per-farm cloud check.
    A village-level cloud percentage says nothing about one 0.27 ha field."""
    out = np.full(len(geoms), np.nan)
    with rasterio.open(sign(href)) as src:
        gg = geoms.to_crs(src.crs)
        for i, geom in enumerate(gg.geometry):
            try:
                arr, _ = mask(src, [geom], crop=True, filled=True, nodata=0, indexes=1)
            except ValueError:
                continue
            v = arr.ravel()
            v = v[v != 0]
            if v.size:
                out[i] = np.isin(v, list(SCL_KEEP)).mean()
    return out


def main():
    farms = gpd.read_file(FARMS).reset_index(drop=True)
    fid = farms["FID"].astype(int) if "FID" in farms else pd.RangeIndex(1, len(farms) + 1)
    bbox = gpd.read_file(VILLAGE).to_crs(4326).total_bounds
    log("p0_witness.start", farms=len(farms), bbox=list(np.round(bbox, 5)))

    # ---------------- Sentinel-1 RTC ----------------
    items = search("sentinel-1-rtc", WINDOW, bbox, limit=500)
    rel = pd.Series([i["properties"].get("sat:relative_orbit") for i in items])
    keep = rel.value_counts().idxmax()      # one orbit only: mixing orbits mixes incidence
    items = sorted((i for i in items if i["properties"].get("sat:relative_orbit") == keep),
                   key=lambda i: i["properties"]["datetime"])
    log("p0_witness.s1_scenes", relative_orbit=int(keep), n=len(items))

    s1 = {"farm_id": fid}
    for it in items:
        d = it["properties"]["datetime"][:10].replace("-", "")
        for pol in ("vv", "vh"):
            if pol not in it["assets"]:
                continue
            v = zonal(it["assets"][pol]["href"], farms)     # RTC is linear gamma0
            if np.isfinite(v).sum() < 0.5 * len(farms):
                print(f"  S1 {d} {pol}: partial tile, dropped "
                      f"({int(np.isfinite(v).sum())}/{len(farms)})")
                continue
            s1[f"s1_{pol}_db_{d}"] = 10.0 * np.log10(np.where(v > 0, v, np.nan))
        print(f"  S1 {d}  done  ({len(s1)-1} cols)")
    pd.DataFrame(s1).to_csv(TABLES / "witness_s1.csv", index=False)
    log("p0_witness.s1_done", cols=len(s1) - 1)

    # ---------------- Sentinel-2 L2A ----------------
    items = [i for i in search("sentinel-2-l2a", WINDOW, bbox, limit=500)
             if i["properties"].get("eo:cloud_cover", 100) < MAX_CLOUD]
    items.sort(key=lambda i: (i["properties"]["datetime"], i["properties"]["eo:cloud_cover"]))
    s2 = {"farm_id": fid}
    for date, grp in pd.Series(items).groupby(
            [i["properties"]["datetime"][:10] for i in items]):
        best, bestn = None, -1
        for it in grp:                       # two MGRS tiles overlap us; take the fuller one
            red = zonal(it["assets"]["B04"]["href"], farms)
            nir = zonal(it["assets"]["B08"]["href"], farms)
            nd = (nir - red) / (nir + red)
            n = int(np.isfinite(nd).sum())
            if n > bestn:
                best, bestn = (it, nd), n
            if n > 0.95 * len(farms):
                break
        it, nd = best
        d = date.replace("-", "")
        s2[f"ndvi_{d}"] = nd
        s2[f"ndvi_valid_{d}"] = zonal_valid_frac(it["assets"]["SCL"]["href"], farms)
        print(f"  S2 {date}  n={bestn}  cloud={it['properties']['eo:cloud_cover']:.3f}%  "
              f"{it['id']}")
    pd.DataFrame(s2).to_csv(TABLES / "witness_s2.csv", index=False)
    log("p0_witness.s2_done", dates=(len(s2) - 1) // 2)


if __name__ == "__main__":
    main()
