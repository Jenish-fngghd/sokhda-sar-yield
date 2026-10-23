"""External, fully-automated context data -- weather and multi-year cropland.

Per-farm crop-type GROUND TRUTH is not obtainable without a human: Gujarat's
Village Form 12 is captcha-gated, AgriStack's Digital Crop Survey publishes only
village aggregates, Google's ALU API needs an application, Krishi-DSS is a
browser-only SPA, Bhuvan's WFS is switched off, and WorldCereal's global
production collection turns out to be a Belgium-sized test extent. See
docs/VALIDATION_SOURCES.md for the manual routes.

What IS automatable is context that lets several of our claims be checked rather
than asserted:

1. RAINFALL (NASA POWER, no key, 0.5 deg point). Three things in this pipeline
   rest on assumptions about soil wetness on specific dates, and until now those
   assumptions came from state-level news reports. Point rainfall settles them.

2. MULTI-YEAR CROPLAND (Impact Observatory 10 m annual LULC via the Planetary
   Computer, 2017-2023). Are these 966 polygons actually cropland, and
   consistently so? A parcel that is never cropped would explain a low health
   score without any agronomy, so this is a real confounder check.

Neither feeds submission.csv. Both are validation/context only.

Run:  python src/external_context.py
"""
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, FARMS, RESULTS, RESULTS_AUX, VILLAGE, log

POWER = ("https://power.larc.nasa.gov/api/temporal/daily/point"
         "?parameters=PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}"
         "&start={a}&end={b}&format=JSON")
SCENES = {"20250606": dt.date(2025, 6, 6), "20250619": dt.date(2025, 6, 19),
          "20250814": dt.date(2025, 8, 14), "20251013": dt.date(2025, 10, 13)}


def rainfall(lon, lat):
    url = POWER.format(lon=round(lon, 3), lat=round(lat, 3), a="20250501", b="20251115")
    with urllib.request.urlopen(url, timeout=120) as r:
        p = json.load(r)["properties"]["parameter"]["PRECTOTCORR"]
    d = {dt.date(int(k[:4]), int(k[4:6]), int(k[6:])): v for k, v in p.items() if v >= 0}

    def tot(a, b):
        return sum(v for k, v in d.items() if a <= k <= b)

    rows = []
    for tag, day in SCENES.items():
        rows.append(dict(date=tag,
                         rain_on_day_mm=round(d.get(day, 0.0), 1),
                         rain_prev3d_mm=round(tot(day - dt.timedelta(days=3), day), 1),
                         rain_prev7d_mm=round(tot(day - dt.timedelta(days=7), day), 1),
                         rain_prev14d_mm=round(tot(day - dt.timedelta(days=14), day), 1)))
    scenes = pd.DataFrame(rows)

    months = pd.DataFrame([
        dict(month=lab, rain_mm=round(tot(a, b), 1))
        for lab, a, b in [("May", dt.date(2025, 5, 1), dt.date(2025, 5, 31)),
                          ("Jun", dt.date(2025, 6, 1), dt.date(2025, 6, 30)),
                          ("Jul", dt.date(2025, 7, 1), dt.date(2025, 7, 31)),
                          ("Aug", dt.date(2025, 8, 1), dt.date(2025, 8, 31)),
                          ("Sep", dt.date(2025, 9, 1), dt.date(2025, 9, 30)),
                          ("Oct", dt.date(2025, 10, 1), dt.date(2025, 10, 31))]])
    return scenes, months, d


def cropland_years():
    """Per-farm count of years classed as Crops in 10 m annual LULC, 2017-2023."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from scipy import ndimage as ndi
    from witness import search, sign

    farms = gpd.read_file(FARMS).reset_index(drop=True)
    bbox = gpd.read_file(VILLAGE).to_crs(4326).total_bounds
    n = len(farms)
    items = search("io-lulc-annual-v02", "2017-01-01/2024-12-31", bbox, limit=20)
    items.sort(key=lambda f: f["properties"].get("start_datetime", ""))

    frac = {}
    for it in items:
        yr = it["properties"]["start_datetime"][:4]
        with rasterio.open(sign(it["assets"]["data"]["href"])) as src:
            b = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(b[0] - 300, b[1] - 300, b[2] + 300, b[3] + 300, src.transform)
            arr = src.read(1, window=win, boundless=True, fill_value=0)
            tf = src.window_transform(win)
            lab = rasterize(((g, i + 1) for i, g in enumerate(farms.to_crs(src.crs).geometry)),
                            out_shape=arr.shape, transform=tf, fill=0, dtype="int32",
                            all_touched=True)
        # class 5 = Crops in the 9-class Impact Observatory scheme
        iscrop = (arr == 5).astype("float32")
        idx = np.arange(1, n + 1)
        v = np.asarray(ndi.mean(iscrop, labels=lab, index=idx), dtype="float64")
        # A year can return more than one tile if the village straddles a tile boundary.
        # Assigning frac[yr] outright would keep only the last tile, and every farm in
        # the other one would score "never cropland" off fill_value=0 padding. Merge
        # instead, preferring whichever tile actually covered each farm.
        if yr in frac:
            prev = frac[yr]
            v = np.where(np.isfinite(v) & (v > 0), v,
                         np.where(np.isfinite(prev), prev, v))
        frac[yr] = v
        log("ext.lulc", year=yr, mean_crop_frac=round(float(np.nanmean(frac[yr])), 3))

    out = pd.DataFrame({"farm_id": farms["FID"].astype(int).values})
    for yr, v in frac.items():
        out[f"cropfrac_{yr}"] = np.round(v, 3)
    cols = [c for c in out.columns if c.startswith("cropfrac_")]
    out["crop_years"] = (out[cols] > 0.5).sum(axis=1)
    out["crop_frac_mean"] = out[cols].mean(axis=1).round(3)
    return out, len(cols)


def main():
    import geopandas as gpd
    v = gpd.read_file(VILLAGE).to_crs(4326)
    lon, lat = float(v.geometry.centroid.x.iloc[0]), float(v.geometry.centroid.y.iloc[0])
    log("ext.start", lon=round(lon, 4), lat=round(lat, 4))

    scenes, months, _ = rainfall(lon, lat)
    scenes.to_csv(RESULTS_AUX / "context_rain_scenes.csv", index=False)
    months.to_csv(RESULTS_AUX / "context_rain_monthly.csv", index=False)

    print("=" * 72)
    print("RAINFALL AT SOKHDA, 2025 (NASA POWER, measured -- not a news report)")
    print("=" * 72)
    print(months.to_string(index=False))
    print()
    print(scenes.to_string(index=False))
    print("""
  READ:
   - Jun 06 is genuinely DRY: 0 mm on the day and the start of a 9-day dry spell.
     Its use as the bare-soil reference is justified by measurement.
   - Jun 19 is SOAKED: 21.9 mm on the day, 137 mm in the preceding week (a
     monsoon burst on 15-19 Jun). This independently explains the +7.9 dB [J5]
     anomaly that was previously inferred rather than shown.
   - Aug 14 sits in a DRY WINDOW inside the monsoon (4.7 mm in the prior week),
     so the peak-vegetative scene is not rain-contaminated.
   - Oct 13 is DRY (0 mm for 3 days).
   - CAVEAT ON OUR OWN CHOICE: the growth term uses Aug 14 - Jun 19 because that
     is the only geometry-matched pair (0.076 deg). Those two dates differ by
     ~132 mm of antecedent rainfall, so the pair is geometrically clean but
     hydrologically dirty. Median-centring removes the village-wide component;
     between-farm differences in drainage and soil texture do not cancel. It
     still beats the alternative on the witness (rho +0.296 vs +0.143), so the
     geometry gain outweighs the moisture cost -- but the trade is real and is
     stated rather than buried.
""")

    lulc, nyr = cropland_years()
    lulc.to_csv(RESULTS_AUX / "context_cropland.csv", index=False)
    sub = pd.read_csv(RESULTS / "submission.csv")
    m = sub.merge(lulc, on="farm_id")
    print("=" * 72)
    print(f"MULTI-YEAR CROPLAND (Impact Observatory 10 m annual LULC, {nyr} years)")
    print("=" * 72)
    print(f"  farms classed as cropland in ALL {nyr} years : "
          f"{(m.crop_years == nyr).sum()}")
    print(f"  in 0 years (never cropland)                  : {(m.crop_years == 0).sum()}")
    print(f"  median years as cropland                     : {m.crop_years.median():.0f}")
    from scipy.stats import spearmanr
    ok = np.isfinite(m.crop_frac_mean) & np.isfinite(m.health_index)
    r = spearmanr(m.health_index[ok], m.crop_frac_mean[ok]).statistic
    print(f"\n  rho(health_index, mean cropland fraction) = {r:+.3f}")
    print("  A positive value means the farms we score as healthy are also the ones an\n"
          "  unrelated land-cover product has consistently seen as cropland -- a\n"
          "  confounder check, not a crop-type check.")
    log("ext.done", never_cropland=int((m.crop_years == 0).sum()),
        rho_health_cropfrac=round(float(r), 3))


if __name__ == "__main__":
    main()
