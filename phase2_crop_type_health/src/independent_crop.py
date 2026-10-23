"""An INDEPENDENT second-opinion crop map, from Sentinel-2 + Sentinel-1 only.

Why this exists. The deliverable's crop map comes from four Capella X-band HH
scenes. Its only validation so far is that the resulting classes separate on
two single-date witnesses -- real, but it does not say whether the per-farm
LABELS are right. This builds a completely separate opinion, on different
sensors, different frequencies, different physics and a far denser time series
(~6 months of S2 optical + S1 C-band VV/VH against four X-band dates), then
reports the confusion matrix against the Capella map.

It is a VALIDATOR, never an input. Nothing here touches submission.csv. Capella
remains the primary and sole source of the deliverable -- that separation is
what makes the comparison meaningful rather than circular.

What it is and is not. This is a second model, not ground truth. Agreement means
two independent methods converge; disagreement localises where the X-band map is
weak. Neither is a measurement of accuracy against reality -- for that see
docs/VALIDATION_SOURCES.md.

The crop signatures used here are the ones Round 1 MEASURED on Sentinel-1 C-band
against the leaderboard [J4]: rice by the flood dip VH(Aug)-VH(Jun), cotton by
standing-late VH(Oct), groundnut by inverse NDVI, maize by peak VH(Aug), bajra by
no signal at all. Those were fitted on C-band, which is exactly the sensor used
here -- so this map applies them in their native band, whereas the Capella map has
to read them across to X-band HH.

Run:  python src/independent_crop.py            (fetches, caches, compares)
      python src/independent_crop.py --nofetch  (use the cached feature table)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d4_submission as D4
from common import AUX, CACHE, CROPS, FARMS, RESULTS, RESULTS_AUX, VILLAGE, log
from witness import search, sign

FEAT = RESULTS_AUX / "independent_features.csv"
OUT = RESULTS_AUX / "independent_crop.csv"

# Monthly windows across the kharif season. July/August are ~100% cloud over
# Gujarat, so the optical series is genuinely absent there -- which is precisely
# why the SAR half of this validator matters.
MONTHS = [("2025-06-01", "2025-06-30", "jun"), ("2025-07-01", "2025-07-31", "jul"),
          ("2025-08-01", "2025-08-31", "aug"), ("2025-09-01", "2025-09-30", "sep"),
          ("2025-10-01", "2025-10-31", "oct"), ("2025-11-01", "2025-11-30", "nov")]


def village_window(src, bounds4326):
    """Window covering the village bbox, with a small margin."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    b = transform_bounds("EPSG:4326", src.crs, *bounds4326)
    pad = 300.0
    return from_bounds(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad, src.transform)


def zonal_fast(href, labels_for_grid, bounds4326, n):
    """Median per farm, reading ONE village-sized window instead of 966 polygons.

    The village is ~4x4 km, i.e. ~400x400 pixels at 10 m -- trivial to hold in
    memory. Reading a window once and using a prerasterised label image turns
    966 HTTP range-reads per band into one, which is the difference between this
    validator being minutes and being hours.
    """
    import rasterio
    from scipy import ndimage as ndi
    with rasterio.open(sign(href)) as src:
        win = village_window(src, bounds4326)
        arr = src.read(1, window=win, boundless=True, fill_value=0).astype("float64")
        tf = src.window_transform(win)
        lab = labels_for_grid(src.crs, tf, arr.shape)
    arr[arr == 0] = np.nan
    ok = np.isfinite(arr)
    lv = np.where(ok, lab, 0)
    idx = np.arange(1, n + 1)
    med = np.asarray(ndi.median(np.where(ok, arr, np.nan), labels=lv, index=idx),
                     dtype="float64")

    # ndi.median FABRICATES a value for any label with no pixels: out-of-farm pixels
    # that are finite land in the label-0 group, so that group is not all-NaN and the
    # estimator returns a leaked value from it rather than NaN. Verified:
    # ndi.median returns 5.0 for an empty label where ndi.mean correctly returns NaN.
    # The leaked value then passes np.isfinite, so a farm outside the scene footprint
    # would silently receive an invented reading and feed the classifier. Count the
    # valid pixels per label explicitly and blank any label that has none.
    cnt = np.bincount(lv.ravel(), minlength=n + 1)[1:]
    med[cnt == 0] = np.nan
    return med


def make_label_factory(farms):
    """Rasterise the farms onto whatever grid a given asset happens to use."""
    from rasterio.features import rasterize
    cache = {}

    def f(crs, tf, shape):
        key = (str(crs), tuple(np.round(np.asarray(tf)[:6], 3)), shape)
        if key not in cache:
            g = farms.to_crs(crs)
            cache[key] = rasterize(
                ((geom, i + 1) for i, geom in enumerate(g.geometry)),
                out_shape=shape, transform=tf, fill=0, dtype="int32", all_touched=True)
        return cache[key]
    return f


def fetch():
    import geopandas as gpd
    farms = gpd.read_file(FARMS).reset_index(drop=True)
    bbox = gpd.read_file(VILLAGE).to_crs(4326).total_bounds
    n = len(farms)
    labfac = make_label_factory(farms)
    out = pd.DataFrame({"farm_id": farms["FID"].astype(int).values})

    for a, b, tag in MONTHS:
        # ---- Sentinel-2: NDVI, cloud-masked per pixel via SCL ----------------
        # A scene-level cloud_cover filter is the wrong instrument during the
        # monsoon: the tile can be 80% cloudy while Sokhda itself is clear, and a
        # strict scene filter simply returns nothing for Jul-Sep. Take everything
        # up to 95% scene cloud and mask per pixel on SCL instead, keeping only
        # vegetation/soil/water classes (4,5,6,7,11). Scenes that end up with too
        # few clear farms are then discarded on the evidence rather than on a
        # scene-level proxy.
        items = search("sentinel-2-l2a", f"{a}/{b}", bbox,
                       {"eo:cloud_cover": {"lt": 95}}, limit=40)
        items.sort(key=lambda f: f["properties"]["eo:cloud_cover"])
        best_ndvi, best_id, best_n = np.full(n, np.nan), None, 0
        for it in items[:8]:
            try:
                red = zonal_fast(it["assets"]["B04"]["href"], labfac, bbox, n)
                nir = zonal_fast(it["assets"]["B08"]["href"], labfac, bbox, n)
                scl = (zonal_fast(it["assets"]["SCL"]["href"], labfac, bbox, n)
                       if "SCL" in it["assets"] else np.full(n, 4.0))
            except Exception as e:                       # noqa: BLE001
                log("indep.s2_skip", month=tag, item=it["id"], err=str(e)[:80]); continue
            clear = np.isin(np.round(scl), [4, 5, 6, 7, 11])
            cand = np.where(clear, (nir - red) / (nir + red), np.nan)
            if np.isfinite(cand).sum() > best_n:
                best_ndvi, best_id, best_n = cand, it["id"], int(np.isfinite(cand).sum())
            if best_n > 0.9 * n:
                break
        out[f"ndvi_{tag}"] = best_ndvi
        log("indep.s2", month=tag, item=best_id, n_clear=best_n, n_items=len(items),
            cloud=round(items[0]["properties"]["eo:cloud_cover"], 1) if items else None)

        # ---- Sentinel-1 RTC: VV and VH, median over the month ----
        s1 = search("sentinel-1-rtc", f"{a}/{b}", bbox, limit=30)
        acc = {"vv": [], "vh": []}
        for it in s1[:6]:
            for pol in ("vv", "vh"):
                if pol not in it["assets"]:
                    continue
                try:
                    v = zonal_fast(it["assets"][pol]["href"], labfac, bbox, n)
                except Exception as e:                   # noqa: BLE001
                    log("indep.s1_skip", month=tag, err=str(e)[:80]); continue
                if np.isfinite(v).sum() > 0.5 * n:
                    acc[pol].append(v)
        for pol in ("vv", "vh"):
            if acc[pol]:
                lin = np.nanmedian(np.vstack(acc[pol]), axis=0)
                out[f"{pol}_{tag}"] = 10 * np.log10(np.where(lin > 0, lin, np.nan))
            else:
                out[f"{pol}_{tag}"] = np.nan
        log("indep.s1", month=tag, n_scenes=len(acc["vh"]),
            vh_med=round(float(np.nanmedian(out[f"vh_{tag}"])), 2)
            if acc["vh"] else None)

    out.to_csv(FEAT, index=False)
    log("indep.fetch_done", path=FEAT.name, cols=out.shape[1])
    return out


def classify(g):
    """Independent crop opinion from C-band + optical phenology.

    Same STRUCTURE as the deliverable's crop step -- soft evidence, softmax,
    prior-matched marginals, argmax -- deliberately, so the comparison isolates
    the effect of the SENSOR rather than of the assignment machinery. The
    EVIDENCE is entirely different: not one Capella feature enters.
    """
    z = D4.z

    def col(name):
        return g[name].values if name in g else np.full(len(g), np.nan)

    vh_jun, vh_aug, vh_oct = col("vh_jun"), col("vh_aug"), col("vh_oct")
    vh_sep, vv_oct = col("vh_sep"), col("vv_oct")
    ndvi_oct, ndvi_sep, ndvi_nov = col("ndvi_oct"), col("ndvi_sep"), col("ndvi_nov")

    flood = vh_aug - vh_jun          # R1's biggest measured lever: paddy flood dip
    standing = vh_oct                # cotton still standing in October
    senesce = np.where(np.isfinite(ndvi_nov), ndvi_nov, ndvi_oct) - ndvi_sep

    e = {}
    # Rice -- flooded paddy depresses C-band VH between June and August, then the
    # crop is harvested by mid-October, so NDVI falls away late.
    e["Rice"] = -1.0 * z(flood) - 0.4 * z(ndvi_oct) + 0.3 * z(senesce) * -1.0
    # Cotton -- the only crop still fully standing in October: high VH and the
    # highest October NDVI of the five.
    e["Cotton"] = 0.9 * z(standing) + 0.8 * z(ndvi_oct) + 0.3 * z(vv_oct)
    # Groundnut -- R1 measured its NDVI coupling as NEGATIVE: thin canopy on the
    # drier, lighter soils. Lifted Oct-Nov.
    e["Groundnut"] = -0.7 * z(ndvi_oct) + 0.3 * z(vh_sep) - 0.2 * z(standing)
    # Maize -- tall, high biomass at the August peak, bare by October.
    e["Maize"] = 0.8 * z(vh_aug) - 0.6 * z(ndvi_oct) + 0.3 * z(senesce)
    # Bajra -- R1 found no usable signal in any free layer; let the prior place it.
    e["Bajra"] = -0.3 * z(vh_aug) - 0.3 * z(ndvi_oct)

    E = np.column_stack([e[c] for c in CROPS])
    E = np.where(np.isfinite(E), E, 0.0)
    E -= E.max(axis=1, keepdims=True)
    P = np.exp(E); P /= P.sum(axis=1, keepdims=True)

    truth = pd.read_csv(AUX / "sokhda_phase1_crop_shares.csv").set_index("crop")
    prior = np.array([float(truth.loc[c, "sokhda_share"]) for c in CROPS])
    prior = prior / prior.sum()
    area = pd.read_csv(RESULTS / "farm_features.csv").set_index("farm_id") \
             .loc[g.farm_id, "area_ha"].values
    area = np.where(np.isfinite(area) & (area > 0), area, np.nanmedian(area))
    Q, _ = D4.fit_prior(P, prior, area)
    return np.array(CROPS)[Q.argmax(axis=1)], Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nofetch", action="store_true")
    a = ap.parse_args()

    g = pd.read_csv(FEAT) if (a.nofetch and FEAT.exists()) else fetch()
    lab, Q = classify(g)
    sub = pd.read_csv(RESULTS / "submission.csv")
    cap = sub.set_index("farm_id").loc[g.farm_id, "crop_type"].values

    res = pd.DataFrame({"farm_id": g.farm_id, "capella_crop": cap,
                        "independent_crop": lab})
    for i, c in enumerate(CROPS):
        res[f"p_{c}"] = np.round(Q[:, i], 4)
    res.to_csv(OUT, index=False)

    agree = float((cap == lab).mean())
    area = pd.read_csv(RESULTS / "farm_features.csv").set_index("farm_id") \
             .loc[g.farm_id, "area_ha"].values
    agree_a = float(area[cap == lab].sum() / area.sum())

    print("\n" + "=" * 70)
    print("INDEPENDENT CROP MAP (Sentinel-2 + Sentinel-1) vs CAPELLA X-band")
    print("=" * 70)
    print(f"  agreement by farm {100*agree:.1f}%   by area {100*agree_a:.1f}%")
    print(f"  chance level (prior-matched marginals) "
          f"{100*sum((cap == c).mean() * (lab == c).mean() for c in CROPS):.1f}%")
    print("\n  rows = Capella, cols = independent:")
    print(pd.crosstab(cap, lab).to_string())
    # Cohen's kappa: raw agreement is not interpretable when both maps are pinned
    # to the SAME area prior, because that alone forces a large chance overlap.
    pe = sum((cap == c).mean() * (lab == c).mean() for c in CROPS)
    kappa = (agree - pe) / (1 - pe)
    print(f"\n  Cohen's kappa {kappa:+.3f}  "
          f"({'negligible' if kappa < 0.2 else 'fair' if kappa < 0.4 else 'moderate'})")

    print("\n  per-crop recall vs that crop's own marginal in the independent map:")
    for c in CROPS:
        m = cap == c
        if m.sum():
            rec = (lab[m] == c).mean()
            marg = (lab == c).mean()
            print(f"    {c:11s} recall {100*rec:5.1f}%  marginal {100*marg:5.1f}%  "
                  f"lift {rec/max(marg,1e-9):4.2f}x   n={m.sum()}")

    # Does the deliverable's own confidence predict where the two maps agree? If it
    # does, crop_confidence is a usable reliability flag; if not, it is decorative.
    dbg = pd.read_csv(RESULTS / "d4_debug.csv").set_index("farm_id")
    conf = dbg.loc[g.farm_id, "crop_confidence"].values
    hi_m = conf >= np.nanmedian(conf)
    print(f"\n  agreement on high-confidence half {100*(cap == lab)[hi_m].mean():.1f}%"
          f"  vs low-confidence half {100*(cap == lab)[~hi_m].mean():.1f}%")

    log("indep.compare", agree_farm=round(agree, 3), agree_area=round(agree_a, 3),
        kappa=round(kappa, 3),
        agree_hiconf=round(float((cap == lab)[hi_m].mean()), 3),
        agree_loconf=round(float((cap == lab)[~hi_m].mean()), 3))


if __name__ == "__main__":
    main()
