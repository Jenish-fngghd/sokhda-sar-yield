"""Stage I5 helper -- fetch the two INDEPENDENT-SENSOR witness layers.

These are WITNESSES, never inputs. Nothing here feeds `submission.csv`. Capella
X-band SAR stays the primary and only source of the deliverable; Sentinel-2 and
Sentinel-1 exist solely to answer "does an unrelated satellite see the same
pattern our SAR product claims?".

Both come from the Microsoft Planetary Computer STAC, which serves Sentinel-1 RTC
and Sentinel-2 L2A as public COGs with an anonymous SAS token -- free, no account,
open licence (Copernicus). That satisfies the open-data rule.

THE LUCKY BREAK: Sentinel-2 flew Sokhda on **2025-10-13**, the SAME DAY as our
Capella October acquisition, at 0.003% cloud. A same-day optical witness is the
strongest independent check available for a health index, and it costs us nothing
in SAR primacy because it never enters the model.

Sentinel-1 is 2025-10-10, three days off, C-band VH -- different band, different
polarisation, different geometry. Agreement there is sensor diversity, which was
Round 1's most transferable lesson [J4].

Run:  python src/witness.py      (writes results/witness.csv, one row per farm)
"""
import json
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FARMS, RESULTS, RESULTS_AUX, VILLAGE, log

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
_tokens = {}


def sign(href):
    """PC assets need a per-collection SAS token appended. Cached per collection."""
    coll = href.split("/")[3] if href.startswith("https://") else ""
    acct = href.split("//")[1].split(".")[0]
    key = f"{acct}/{coll}"
    if key not in _tokens:
        url = f"{SAS}/{acct}/{coll}"
        with urllib.request.urlopen(url, timeout=60) as r:
            _tokens[key] = json.load(r)["token"]
    return f"{href}?{_tokens[key]}"


def search(collection, datetime, bbox, query=None, limit=20):
    body = {"collections": [collection], "bbox": list(map(float, bbox)),
            "datetime": datetime, "limit": limit}
    if query:
        body["query"] = query
    req = urllib.request.Request(f"{STAC}/search", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["features"]


def zonal(href, geoms, agg=np.nanmedian):
    """Median of a COG inside each farm polygon, read straight over HTTP.

    Median not mean: a handful of mixed edge pixels at 10 m on a 0.3 ha field
    would drag a mean, and every farm here is small relative to the pixel.
    """
    out = np.full(len(geoms), np.nan)
    with rasterio.open(sign(href)) as src:
        gg = geoms.to_crs(src.crs)
        for i, geom in enumerate(gg.geometry):
            try:
                arr, _ = mask(src, [geom], crop=True, filled=True, nodata=0)
            except ValueError:
                continue                        # polygon outside the tile
            v = arr[0].astype("float64")
            v[v == 0] = np.nan
            if np.isfinite(v).any():
                out[i] = agg(v)
    return out


def main():
    log("witness.start")
    farms = gpd.read_file(FARMS).reset_index(drop=True)
    bbox = gpd.read_file(VILLAGE).to_crs(4326).total_bounds

    # --- Sentinel-2, same day as the Capella October scene ---
    items = search("sentinel-2-l2a", "2025-10-12/2025-10-14", bbox,
                   {"eo:cloud_cover": {"lt": 5}})
    items.sort(key=lambda f: f["properties"]["eo:cloud_cover"])
    ndvi = np.full(len(farms), np.nan)
    used = None
    for it in items:                            # first tile that actually covers us
        red = zonal(it["assets"]["B04"]["href"], farms)
        nir = zonal(it["assets"]["B08"]["href"], farms)
        cand = (nir - red) / (nir + red)
        if np.isfinite(cand).sum() > np.isfinite(ndvi).sum():
            ndvi, used = cand, it["id"]
        if np.isfinite(ndvi).sum() > 0.9 * len(farms):
            break
    log("witness.s2", item=used, n=int(np.isfinite(ndvi).sum()))

    # --- Sentinel-1 RTC, C-band VH ---
    s1 = search("sentinel-1-rtc", "2025-10-05/2025-10-20", bbox)
    vh = np.full(len(farms), np.nan)
    s1id = None
    for it in s1:
        if "vh" not in it["assets"]:
            continue
        cand = zonal(it["assets"]["vh"]["href"], farms)
        if np.isfinite(cand).sum() > np.isfinite(vh).sum():
            vh, s1id = cand, it["id"]
        if np.isfinite(vh).sum() > 0.9 * len(farms):
            break
    log("witness.s1", item=s1id, n=int(np.isfinite(vh).sum()))

    out = pd.DataFrame({"farm_id": farms.get("farm_id", pd.RangeIndex(1, len(farms) + 1)),
                        "s2_ndvi_20251013": ndvi,
                        "s1_vh_db": 10.0 * np.log10(np.where(vh > 0, vh, np.nan))})
    out.to_csv(RESULTS_AUX / "witness.csv", index=False)
    print(out.describe().round(3).to_string())
    log("witness.done", s2=used, s1=s1id)


if __name__ == "__main__":
    main()
