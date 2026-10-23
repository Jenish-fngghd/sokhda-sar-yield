"""A SEASON-INTEGRATED witness for the season-integrated part of the yield term.

WHY THIS EXISTS. `check_yield.py` tests two per-farm claims. The completion term is
tested well: it makes a same-day claim and there is a same-day witness, and that test
already caught a sign error. The accumulation term is tested BADLY, and the mismatch is
structural rather than a matter of effort:

    season_integral  is an integral over 12 Jun - 13 Oct
    s2_ndvi_20251013 is one instant, 13 Oct
    s1_vh_db         is one instant, 10 Oct

Correlating a season integral against a single instant is the wrong comparison. It is
why test 2 reads "mixed, and honestly so": for a crop already harvested by October both
witnesses are looking at bare soil, so they cannot speak about what accumulated in July.

WHY NOT INTEGRATED OPTICAL. The obvious fix is cumulative NDVI -- the standard yield
proxy through the Monteith chain (yield ~ integral of APAR, NDVI ~ fAPAR). It is
impossible here, and `why_xband.py` already measured why: over Sokhda in 2025 the count
of Sentinel-2 scenes under 20% cloud is **0 in June, 0 in July, 0 in August, 0 in
September** -- the best July scene is 92.6% cloud. The entire accumulation period has no
optical record. There is nothing to integrate.

WHAT THIS DOES INSTEAD. Sentinel-1 C-band is cloud-immune, so it observed the season
that Sentinel-2 missed: **10 RTC scenes, 12 Jun to 10 Oct**. Every one is relative orbit
34, descending -- one geometry all season, so the integral is not contaminated by a
changing incidence angle. Cross-pol VH is the volume-scattering channel and the
conventional C-band biomass proxy.

The integral is formed EXACTLY as the model forms its own -- trapezoid over day-of-year
in LINEAR power, matching `farm_stats.py` -- so the comparison is like-for-like and not
an artefact of two different definitions of "integral".

INDEPENDENCE. Sentinel-1 is a different satellite, a different frequency (5.405 vs
9.65 GHz), a different polarisation (cross vs co) and a different geometry. It never
enters the model, and this file writes nothing that `d4_submission.py` reads. It is a
witness, exactly like the two already in `witness.csv`.

This test can REFUTE the accumulation term. That is the point of running it.

Run:  python src/witness_season.py      (writes results/witness_season.csv)
"""
import os
os.environ.pop("PROJ_LIB", None)

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FARMS, RESULTS, RESULTS_AUX, VILLAGE, log
from witness import search, zonal

SEASON = "2025-06-01/2025-10-20"
MIN_DATES = 6          # below this the trapezoid is not a season integral in any useful sense


def main():
    log("witness_season.start")
    farms = gpd.read_file(FARMS).reset_index(drop=True)
    bbox = gpd.read_file(VILLAGE).to_crs(4326).total_bounds

    items = search("sentinel-1-rtc", SEASON, bbox, limit=200)
    # One relative orbit only. Mixing orbits would mix incidence angles, and a change in
    # geometry between two dates would enter the integral as if it were a change in the
    # crop. Pick the orbit with the most scenes rather than assuming which one that is.
    rel = pd.Series([i["properties"].get("sat:relative_orbit") for i in items])
    keep = rel.value_counts().idxmax()
    items = [i for i in items if i["properties"].get("sat:relative_orbit") == keep]
    items.sort(key=lambda i: i["properties"]["datetime"])
    log("witness_season.scenes", relative_orbit=int(keep), n=len(items),
        first=items[0]["properties"]["datetime"][:10],
        last=items[-1]["properties"]["datetime"][:10])

    cols, doys, used = [], [], []
    for it in items:
        if "vh" not in it["assets"]:
            continue
        v = zonal(it["assets"]["vh"]["href"], farms)      # RTC is linear gamma0
        if np.isfinite(v).sum() < 0.5 * len(farms):
            continue                                       # partial tile, not a season sample
        d = pd.Timestamp(it["properties"]["datetime"][:10])
        cols.append(v)
        doys.append(float(d.dayofyear))
        used.append(str(d.date()))
        print(f"  {d.date()}  n={int(np.isfinite(v).sum()):4d}  "
              f"median VH {10*np.log10(np.nanmedian(v)):+.2f} dB")

    assert len(cols) >= MIN_DATES, f"only {len(cols)} usable S1 dates, need >= {MIN_DATES}"

    lin = np.column_stack(cols)                            # linear power, farms x dates
    doy = np.asarray(doys, dtype="float64")
    # Same construction as farm_stats.py: trapezoid over day-of-year in LINEAR power.
    integral = np.trapezoid(lin, doy, axis=1)
    integral[np.isnan(lin).any(axis=1)] = np.nan           # a partial series is not an integral

    out = pd.DataFrame({
        "farm_id": farms.get("farm_id", pd.RangeIndex(1, len(farms) + 1)),
        "s1_vh_season_integral": integral,
        "s1_vh_mean_db": 10.0 * np.log10(np.nanmean(lin, axis=1)),
    })
    # Keep the per-date columns too: the integral alone cannot show the phenology, and
    # the season curve is the thing that makes "C-band saw what optical missed" visible.
    for d, col in zip(used, lin.T):
        out[f"s1_vh_db_{d.replace('-', '')}"] = 10.0 * np.log10(np.where(col > 0, col, np.nan))
    out.to_csv(RESULTS_AUX / "witness_season.csv", index=False)
    log("witness_season.done", dates=len(cols), farms_with_integral=int(np.isfinite(integral).sum()),
        span_days=float(doy[-1] - doy[0]), used=",".join(used))
    print(out.describe().round(4).to_string())


def demo():
    """The integral must reproduce the model's own definition on a known series."""
    lin = np.array([[1.0, 2.0, 3.0]])
    doy = np.array([0.0, 10.0, 20.0])
    assert np.isclose(np.trapezoid(lin, doy, axis=1)[0], 40.0)
    # a gap anywhere makes the series unintegrable, and must not silently become a number
    lin2 = np.array([[1.0, np.nan, 3.0]])
    got = np.trapezoid(lin2, doy, axis=1)
    assert not np.isfinite(got[0])
    print("demo OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
