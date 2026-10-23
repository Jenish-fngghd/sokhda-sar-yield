"""Phase 1.2: per-farm features on the six-date stack, plus coverage and the NESZ gate.

Feature families, and what each is for:

  level        mean/median gamma0 per date          canopy + soil scattering strength
  uniformity   spatial CV                           "more uniform canopy scores higher"
  texture      K-distribution 1/alpha [speckle      heterogeneity with SPECKLE SEPARATED
               separated out], GLCM entropy on      OUT, which a plain CV cannot do
               the FINE grid, UNFILTERED
  temporal     deltas, season integral, temporal CV growth and senescence
  referenced   date minus the June bare-soil        removes each plot's static soil and
               baseline                             roughness term

Three rules carried forward from Round 2 that are not negotiable:
  * Texture is computed on UNFILTERED data. Speckle filtering destroys the very
    second-order statistics being measured.
  * The effective look count is ESTIMATED FROM THE DATA, not assumed. The K-distribution
    estimator divides by (1 + 1/L); assuming the annotated ENL of 1.0 puts every farm
    across the estimator's singularity and returns NaN for most of them.
  * NO FARM IS EVER DROPPED. Coverage is 15 rubric points this round. The interior
    extraction degrades through a negative-buffer ladder and every fallback is counted.

NEW THIS ROUND, and both matter more than the features:

  COVERAGE PER PLOT PER DATE. R2 found missingness clustered on a north-west swath edge,
  not random -- 29 farms fully missing. 29 October's swath is MIRRORED and covers 5 points
  less of the AOI, so its missing set is a different set of farms. Measured here per farm
  per date and written out as its own table.

  THE NESZ MARGIN, per farm per date, in BETA0. P1-2 established that `nesz_peak` is
  referenced to beta0 and that a gate written in gamma0 would declare a fifth of the scene
  to be below the noise floor. Dark, smooth, harvested fields late in the season are
  exactly the condition that puts a plot mean into the noise -- and the late dates are
  where a yield forecast leans hardest. The per-date distribution is reported here; no
  threshold is chosen yet, because a threshold chosen before seeing the distribution is a
  threshold chosen to get an answer.

Writes results/tables/p1_farm_features.csv, p1_coverage.csv, p1_nesz_margin.csv.

Run:  py -3.12 src/p1_features.py
"""
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi
from skimage.feature import graycomatrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATES, FARMS, TABLES, db, log, scene

import geopandas as gpd

UTM = 32643
BUFFERS = [-5.0, -2.0, 0.0]          # the negative-buffer ladder, metres
# 8 grey levels, not 32. With L levels the GLCM has L^2 bins and a plot with n pixels
# contributes ~4n pairs; when 4n << L^2 most bins stay empty and the entropy degenerates
# to log(4n) -- i.e. it measures PLOT SIZE. At 32 levels that artefact correlated with
# area at rho = 0.95. Our smallest usable plots hold ~100 fine pixels.
GLCM_LEVELS = 8
NESZ_MARGINS_DB = [0.0, 3.0, 6.0]    # report the distribution; choose nothing yet


def doy(d):
    return _date(int(d[:4]), int(d[4:6]), int(d[6:8])).timetuple().tm_yday


def load_grid(grid, quantities=("gamma0", "beta0", "incidence")):
    G = {q: {} for q in quantities}
    for q in quantities:
        for d in DATES:
            with rasterio.open(CACHE / f"{q}_{grid}_{d}.tif") as s:
                G[q][d] = s.read(1)
                tf, shape = s.transform, (s.height, s.width)
    return G, tf, shape


def farm_labels(farms, tf, shape, buffers=BUFFERS):
    """Rasterise farm interiors to a label image, degrading the buffer per farm.

    Rasterising once and using labelled statistics is ~100x faster than masking 966
    polygons one at a time, which is what makes recomputing the whole feature set cheap.
    Returns the label image (0 = background, i+1 = farm i) and the ladder level each farm
    ended on -- so a plot that only survived at zero buffer is visible as such, not hidden.
    """
    lab = np.zeros(shape, dtype="int32")
    level = np.full(len(farms), -1, dtype="int8")
    for li, buf in enumerate(buffers):
        todo = np.where(level < 0)[0]
        if not len(todo):
            break
        shapes = []
        for i in todo:
            g = farms.geometry.iloc[i]
            g = g.buffer(buf) if buf else g
            if (not g.is_empty) and g.area > 0:
                shapes.append((g, int(i) + 1))
        if not shapes:
            continue
        tmp = rasterize(shapes, out_shape=shape, transform=tf, fill=0, dtype="int32",
                        all_touched=(buf == 0.0))
        got = np.unique(tmp)
        got = got[got > 0] - 1
        lab[(tmp > 0) & (lab == 0)] = tmp[(tmp > 0) & (lab == 0)]
        level[np.isin(np.arange(len(farms)), got) & (level < 0)] = li
    return lab, level


def labelled_stats(arr, lab, n):
    """mean / median / std / count per label, skipping nodata."""
    a = np.where(np.isfinite(arr), arr, np.nan)
    lv = np.where(np.isfinite(a), lab, 0)
    cnt = np.bincount(lv.ravel(), minlength=n + 1)[1:]
    s = np.bincount(lv.ravel(), weights=np.nan_to_num(a).ravel(), minlength=n + 1)[1:]
    s2 = np.bincount(lv.ravel(), weights=np.nan_to_num(a).ravel() ** 2, minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt > 1, s2 / np.maximum(cnt, 1) - mean ** 2, np.nan)
    med = np.array(ndi.median(a, labels=lv, index=np.arange(1, n + 1)), dtype="float64")
    return mean, med, np.sqrt(np.maximum(var, 0)), cnt


def estimate_looks(cv, cnt, min_px=100, q=0.05):
    """Effective looks, estimated FROM THE DATA rather than assumed.

    The most homogeneous large plots are as close to pure speckle as anything available,
    so a low quantile of their CV gives L = 1/CV^2. Geocoding by `average` resampling is
    the multilook, so L here is a property of our own grid, not of the vendor's ENL = 1.0.
    """
    ok = (cnt >= min_px) & np.isfinite(cv) & (cv > 0)
    if ok.sum() < 20:
        return np.nan
    return float(1.0 / np.quantile(cv[ok], q) ** 2)


def kdist_texture(mean, std, cnt, looks):
    """K-distribution texture by method of moments.

    Under the multiplicative model the normalised second intensity moment factorises:
        E[I^2]/E[I]^2 = (1 + 1/L)(1 + 1/alpha)
    so 1/alpha isolates SCENE texture from SPECKLE, which a raw CV cannot do.

    We return 1/alpha, not alpha: it is bounded below at 0 (perfectly uniform) where
    alpha runs to infinity, and when sampling noise puts the measured variance BELOW the
    pure-speckle expectation the correct reading is "no resolvable texture", 1/alpha = 0,
    not missing data. Clipping at zero keeps the feature defined for every farm.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        inv_alpha = (1.0 + (std / mean) ** 2) / (1.0 + 1.0 / looks) - 1.0
    inv_alpha = np.where(np.isfinite(inv_alpha), np.maximum(inv_alpha, 0.0), np.nan)
    inv_alpha[cnt < 12] = np.nan
    return inv_alpha


def glcm_entropy(dbimg, lab, n):
    """GLCM entropy per farm on the FINE grid, unfiltered.

    GLCM needs a rectangle but a farm is not one. Outside pixels are mapped to level 0
    and the GLCM's zeroth row and column are zeroed, removing every co-occurrence pair
    that involved an outside pixel -- so the statistic is the plot interior only, with no
    contamination from the neighbouring field.
    """
    ent = np.full(n, np.nan)
    lo, hi = np.nanpercentile(dbimg, [1, 99])
    objs = ndi.find_objects(lab)
    for i in range(n):
        sl = objs[i] if i < len(objs) else None
        if sl is None:
            continue
        sub = dbimg[sl]
        m = (lab[sl] == i + 1) & np.isfinite(sub)
        if m.sum() < 16:
            continue
        q = np.zeros(sub.shape, dtype="uint8")
        q[m] = 1 + (np.clip((sub[m] - lo) / max(hi - lo, 1e-6), 0, 1)
                    * (GLCM_LEVELS - 1)).astype("uint8")
        g = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=GLCM_LEVELS + 1, symmetric=True, normed=False).astype("float64")
        g[0, :, :, :] = 0
        g[:, 0, :, :] = 0
        tot = g.sum(axis=(0, 1), keepdims=True)
        if not np.all(tot > 0):
            continue
        p = g / tot
        with np.errstate(divide="ignore", invalid="ignore"):
            ent[i] = float(np.mean(-np.nansum(p * np.log(np.where(p > 0, p, 1)), axis=(0, 1))))
    return ent


def main():
    farms = gpd.read_file(FARMS).to_crs(UTM)
    farms["geometry"] = farms.geometry.make_valid()
    n = len(farms)
    out = pd.DataFrame({"farm_id": farms["FID"].astype(int).values,
                        "village_id": farms["ID_1"].astype(int).values,
                        "area_ha": (farms.area / 1e4).values})

    G, tf, shape = load_grid("base")
    lab, level = farm_labels(farms, tf, shape)
    out["buffer_level"] = level
    log("p1_feat.labels", grid="base", **{f"lvl_{b:g}m": int((level == i).sum())
                                          for i, b in enumerate(BUFFERS)},
        failed=int((level < 0).sum()))

    cov_rows, nesz_rows = [], []
    for d in DATES:
        sc = scene(d)
        g = G["gamma0"][d]
        mean, med, std, cnt = labelled_stats(g, lab, n)
        cv = std / np.where(mean > 0, mean, np.nan)
        looks = estimate_looks(cv, cnt)
        out[f"g0_db_{d}"] = db(mean)
        out[f"g0_lin_{d}"] = mean
        out[f"g0_db_med_{d}"] = db(med)
        out[f"cv_{d}"] = cv
        out[f"ktex_{d}"] = kdist_texture(mean, std, cnt, looks)
        out[f"npix_{d}"] = cnt
        out[f"inc_{d}"] = labelled_stats(G["incidence"][d], lab, n)[0]

        # ---- coverage: pixels present, against the plot's own full-buffer capacity ----
        capacity = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        frac = np.where(capacity > 0, cnt / np.maximum(capacity, 1), np.nan)
        out[f"cov_{d}"] = frac
        cov_rows.append({"date": d, "look": sc["pointing"],
                         "farms_with_pixels": int((cnt > 0).sum()),
                         "farms_no_pixels": int((cnt == 0).sum()),
                         "farms_below_50pct": int(np.nansum(frac < 0.5)),
                         "median_coverage": round(float(np.nanmedian(frac)), 4),
                         "median_npix": int(np.median(cnt))})

        # ---- NESZ margin, in BETA0, against the vendor's own declared floor ----
        bd = db(G["beta0"][d])
        bmean, _, _, bcnt = labelled_stats(G["beta0"][d], lab, n)
        out[f"b0_db_{d}"] = db(bmean)
        out[f"nesz_margin_db_{d}"] = db(bmean) - sc["nesz_peak"]
        r = {"date": d, "nesz_peak_db": sc["nesz_peak"],
             "farm_margin_p05": round(float(np.nanpercentile(db(bmean) - sc["nesz_peak"], 5)), 2),
             "farm_margin_med": round(float(np.nanmedian(db(bmean) - sc["nesz_peak"])), 2),
             "farms_mean_below_floor": int(np.nansum(db(bmean) < sc["nesz_peak"]))}
        for m in NESZ_MARGINS_DB:
            near = np.where(np.isfinite(bd) & (bd < sc["nesz_peak"] + m), lab, 0)
            k = np.bincount(near.ravel(), minlength=n + 1)[1:]
            f = np.where(bcnt > 0, k / np.maximum(bcnt, 1), np.nan)
            out[f"nesz_frac{m:g}_{d}"] = f
            r[f"px_frac_within_{m:g}dB_med"] = round(float(np.nanmedian(f)), 4)
            r[f"farms_over_half_within_{m:g}dB"] = int(np.nansum(f > 0.5))
        nesz_rows.append(r)

        log("p1_feat.date", date=d, looks_est=round(looks, 2),
            n_with_pixels=int((cnt > 0).sum()),
            g0_db_med=round(float(np.nanmedian(db(mean))), 2),
            cv_med=round(float(np.nanmedian(cv)), 3),
            nesz_margin_med=round(r["farm_margin_med"], 2))

    # ---- temporal features, six points ----
    D = out[[f"g0_db_{d}" for d in DATES]].values
    L = out[[f"g0_lin_{d}" for d in DATES]].values
    j = {d: i for i, d in enumerate(DATES)}
    out["d_aug_jun19"] = D[:, j["20250814"]] - D[:, j["20250619"]]   # geometry-matched pair
    out["d_oct13_aug"] = D[:, j["20251013"]] - D[:, j["20250814"]]
    out["d_oct29_oct13"] = D[:, j["20251029"]] - D[:, j["20251013"]]  # WET vs dry: see P1-3
    out["d_nov_oct13"] = D[:, j["20251112"]] - D[:, j["20251013"]]
    out["d_nov_oct29"] = D[:, j["20251112"]] - D[:, j["20251029"]]
    out["d_late"] = D[:, j["20251112"]] - D[:, j["20250814"]]         # peak -> harvest
    base = np.nanmean(D[:, [j["20250606"], j["20250619"]]], axis=1)   # bare-soil baseline
    out["jun_baseline_db"] = base
    for d in DATES[2:]:
        out[f"ref_{d}"] = D[:, j[d]] - base
    out["temporal_cv"] = np.nanstd(L, axis=1) / np.nanmean(L, axis=1)
    out["temporal_range_db"] = np.nanmax(D, axis=1) - np.nanmin(D, axis=1)
    dd = np.array([doy(d) for d in DATES], dtype="float64")
    out["season_integral"] = np.trapezoid(L, dd, axis=1)              # LINEAR power
    # senescence slope: OLS on the three post-peak dates, dB per day
    late = [j["20251013"], j["20251029"], j["20251112"]]
    x = dd[late] - dd[late].mean()
    Y = D[:, late]
    out["senescence_db_per_day"] = ((Y - np.nanmean(Y, axis=1, keepdims=True)) * x).sum(1) / (x ** 2).sum()

    # ---- GLCM entropy on the FINE grid, unfiltered ----
    Gf, tff, shapef = load_grid("fine", quantities=("gamma0",))
    labf, _ = farm_labels(farms, tff, shapef)
    for d in DATES:
        ent = glcm_entropy(db(Gf["gamma0"][d]), labf, n)
        out[f"glcm_ent_{d}"] = ent
        # 8 levels cut the plot-size artefact from rho=0.95 to ~0.48, but a residual
        # dependence on sample count remains and is an estimator effect, not canopy
        # structure. Regress it out against log(npix) and keep the RESIDUAL.
        npx = out[f"npix_{d}"].values.astype("float64")
        ok = np.isfinite(ent) & (npx > 0)
        resid = np.full(n, np.nan)
        if ok.sum() > 30:
            A = np.c_[np.ones(int(ok.sum())), np.log(npx[ok])]
            resid[ok] = ent[ok] - A @ np.linalg.lstsq(A, ent[ok], rcond=None)[0]
        out[f"glcm_resid_{d}"] = resid
        log("p1_feat.glcm", date=d, n=int(np.isfinite(ent).sum()),
            med=round(float(np.nanmedian(ent)), 3))

    # ---- coverage / QC ----
    npix = out[[f"npix_{d}" for d in DATES]].values
    out["n_dates"] = (npix > 0).sum(axis=1)
    out["qc_flag"] = np.where(out.n_dates == len(DATES), "ok",
                              np.where(out.n_dates == 0, "no_sar_data", "partial"))

    out.to_csv(TABLES / "p1_farm_features.csv", index=False)
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(TABLES / "p1_coverage.csv", index=False)
    nz = pd.DataFrame(nesz_rows)
    nz.to_csv(TABLES / "p1_nesz_margin.csv", index=False)

    log("p1_feat.done", rows=len(out), cols=out.shape[1],
        ok=int((out.qc_flag == "ok").sum()),
        partial=int((out.qc_flag == "partial").sum()),
        no_data=int((out.qc_flag == "no_sar_data").sum()))

    print("\nCOVERAGE PER DATE")
    print(cov.to_string(index=False))
    print(f"\n  farms with all {len(DATES)} dates: {int((out.n_dates == len(DATES)).sum())}"
          f" / {n}   |  any date missing: {int((out.n_dates < len(DATES)).sum())}"
          f"   |  no data at all: {int((out.n_dates == 0).sum())}")
    print("\n  farms by usable-date count: "
          + "  ".join(f"{k}:{int((out.n_dates == k).sum())}" for k in range(len(DATES) + 1)))

    print("\nNESZ MARGIN PER DATE (beta0 farm mean minus declared nesz_peak)")
    print(nz.to_string(index=False))


if __name__ == "__main__":
    main()
