"""Stage I2 -- per-farm feature extraction from the calibrated gamma0 stack.

Feature families and what each is for (evidence tag in brackets):

  level        mean/median gamma0 per date         canopy + soil scattering strength
  uniformity   spatial CV, IQR, p10-p90 spread     canopy uniformity -- the rubric's own
                                                   wording, "more uniform canopy scores
                                                   higher" [B6]
  texture      K-distribution shape alpha [D3]     heterogeneity with SPECKLE SEPARATED OUT,
               GLCM entropy on FINE, unfiltered     which plain CV cannot do [B5]
  temporal     deltas, season integral, temporal CV growth/senescence; near-zero temporal CV
                                                   means nothing ever grew there [B6, B10]
  referenced   date minus June bare-soil baseline   removes each plot's static soil/roughness
                                                   term; June is pre-monsoon, pre-sowing [B12]

Two rules that are not negotiable here:
  * Texture is computed on UNFILTERED data -- speckle filtering destroys the very
    second-order statistics being measured. [B5]
  * No farm is ever dropped. Coverage is 10 rubric points and needs all 966 rows, so
    the interior extraction degrades through a documented fallback ladder and every
    fallback used is counted and reported. [F2, F4, E1]

Run:  python src/farm_stats.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATES, FARMS, RESULTS, log

import geopandas as gpd

UTM = 32643
BUFFERS = [-5.0, -2.0, 0.0]      # the negative-buffer ladder, metres
# 8 grey levels, not 32. With L levels the GLCM has L^2 bins, and a plot with n
# pixels contributes only ~4n pairs. When 4n << L^2 most bins stay empty and the
# entropy degenerates to log(4n) -- i.e. it measures PLOT SIZE, not canopy texture.
# At 32 levels that artefact dominated: entropy correlated with area at rho = 0.95.
# Our smallest usable plots hold ~100 fine pixels (~400 pairs), so L^2 must stay
# well under that: 8 levels -> 64 bins. [B5, G8]
GLCM_LEVELS = 8
# looks are ESTIMATED from the data per date (see estimate_looks); the I1 ENL was a
# lower bound and assuming it broke the K-distribution estimator. [G6, G7]


def load_grid(grid):
    G, TH = {}, {}
    for d in DATES:
        with rasterio.open(CACHE / f"gamma0_{grid}_{d}.tif") as s:
            G[d] = s.read(1); tf, shape = s.transform, (s.height, s.width)
        with rasterio.open(CACHE / f"incidence_{grid}_{d}.tif") as s:
            TH[d] = s.read(1)
    return G, TH, tf, shape


def farm_labels(farms, tf, shape, buffers=BUFFERS):
    """Rasterise farm interiors to a label image, degrading the buffer per farm.

    Rasterising once to a label image and then using ndimage's labelled statistics
    is ~100x faster than masking each of the 966 polygons separately, which is what
    makes the full feature set cheap enough to recompute freely.

    Returns the label image (0 = background, i+1 = farm i) and the ladder level
    each farm ended up on.
    """
    lab = np.zeros(shape, dtype="int32")
    level = np.full(len(farms), -1, dtype="int8")

    for li, buf in enumerate(buffers):
        todo = np.where(level < 0)[0]
        if not len(todo):
            break
        shapes, ids = [], []
        for i in todo:
            g = farms.geometry.iloc[i]
            g = g.buffer(buf) if buf else g
            if (not g.is_empty) and g.area > 0:
                shapes.append((g, int(i) + 1)); ids.append(i)
        if not shapes:
            continue
        tmp = rasterize(shapes, out_shape=shape, transform=tf, fill=0,
                        dtype="int32", all_touched=(buf == 0.0))
        got = np.unique(tmp)
        got = got[got > 0] - 1
        take = np.isin(np.arange(len(farms)), got) & (level < 0)
        lab[(tmp > 0) & (lab == 0)] = tmp[(tmp > 0) & (lab == 0)]
        level[take] = li
    return lab, level


def labelled_stats(arr, lab, n):
    """mean / median / std / p10 / p90 per label, skipping nodata."""
    idx = np.arange(1, n + 1)
    a = np.where(np.isfinite(arr), arr, np.nan)
    valid = np.isfinite(a)
    lv = np.where(valid, lab, 0)
    cnt = np.bincount(lv.ravel(), minlength=n + 1)[1:]
    s = np.bincount(lv.ravel(), weights=np.nan_to_num(a).ravel(), minlength=n + 1)[1:]
    s2 = np.bincount(lv.ravel(), weights=np.nan_to_num(a).ravel() ** 2, minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt > 1, s2 / np.maximum(cnt, 1) - mean ** 2, np.nan)
    std = np.sqrt(np.maximum(var, 0))
    med = np.array(ndi.median(a, labels=lv, index=idx), dtype="float64")
    return mean, med, std, cnt


def estimate_looks(cv, cnt, min_px=100, q=0.05):
    """Effective number of looks, estimated FROM THE DATA rather than assumed.

    The most homogeneous large plots in the scene are as close to pure speckle as
    anything available, so a low quantile of their CV gives L = 1/CV^2. This
    matters more than it looks: the K-distribution estimator below divides by
    (1 + 1/L), so a wrong L pushes every farm across the estimator's singularity.
    Assuming the I1 lower-bound ENL of 3.5 did exactly that -- 805 of 966 farms came
    back NaN. Measured L here is ~6-7. [G6, D3]
    """
    ok = (cnt >= min_px) & np.isfinite(cv) & (cv > 0)
    if ok.sum() < 20:
        return np.nan
    return float(1.0 / np.quantile(cv[ok], q) ** 2)


def kdist_texture(mean, std, cnt, looks):
    """K-distribution texture by method of moments. [D3]

    Under the multiplicative model the normalised second intensity moment
    factorises into a speckle part and a scene-texture part:

        E[I^2]/E[I]^2 = (1 + 1/L)(1 + 1/alpha)

    so 1/alpha isolates SCENE texture from SPECKLE -- which a raw CV cannot do,
    because at our look count CV is dominated by speckle rather than by the canopy.

    We return 1/alpha, not alpha. Two reasons. It is bounded below at 0 (a perfectly
    uniform field) whereas alpha runs to infinity there and spans five orders of
    magnitude across farms. And when the measured variance falls *below* the
    pure-speckle expectation -- which happens by sampling noise on small plots --
    the physically correct reading is "no resolvable scene texture", i.e. 1/alpha =
    0, NOT missing data. Clipping at zero keeps the feature defined for every farm
    instead of discarding 80% of them.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        m2 = 1.0 + (std / mean) ** 2                 # E[I^2]/E[I]^2
        inv_alpha = m2 / (1.0 + 1.0 / looks) - 1.0
    inv_alpha = np.where(np.isfinite(inv_alpha), np.maximum(inv_alpha, 0.0), np.nan)
    inv_alpha[cnt < 12] = np.nan                     # too few samples to estimate a moment
    return inv_alpha


def glcm_entropy(dbimg, lab, n, farms_idx):
    """GLCM entropy per farm on the FINE grid, unfiltered. [B5]

    GLCM needs a rectangular array but a farm is not rectangular. Pixels outside the
    farm are mapped to level 0 and the GLCM's zeroth row and column are then zeroed,
    which removes every co-occurrence pair that involved an outside pixel -- so the
    statistic is computed on the plot interior only, without contamination from the
    neighbouring field.
    """
    ent = np.full(n, np.nan)
    lo, hi = np.nanpercentile(dbimg, [1, 99])
    objs = ndi.find_objects(lab)
    for i in farms_idx:
        sl = objs[i]
        if sl is None:
            continue
        sub = dbimg[sl]; m = (lab[sl] == i + 1) & np.isfinite(sub)
        if m.sum() < 16:
            continue
        q = np.zeros(sub.shape, dtype="uint8")
        v = np.clip((sub[m] - lo) / max(hi - lo, 1e-6), 0, 1)
        q[m] = 1 + (v * (GLCM_LEVELS - 1)).astype("uint8")
        g = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=GLCM_LEVELS + 1, symmetric=True, normed=False)
        g = g.astype("float64")
        g[0, :, :, :] = 0; g[:, 0, :, :] = 0          # drop outside-pixel pairs
        tot = g.sum(axis=(0, 1), keepdims=True)
        if not np.all(tot > 0):
            continue
        p = g / tot
        with np.errstate(divide="ignore", invalid="ignore"):
            e = -np.nansum(p * np.log(np.where(p > 0, p, 1)), axis=(0, 1))
        ent[i] = float(np.mean(e))
    return ent


def main():
    farms = gpd.read_file(FARMS).to_crs(UTM)
    farms["geometry"] = farms.geometry.make_valid()
    n = len(farms)
    out = pd.DataFrame({"farm_id": farms["FID"].astype(int).values,
                        "village_id": farms["ID_1"].astype(int).values,
                        "area_ha": (farms.area / 1e4).values})

    G, TH, tf, shape = load_grid("base")
    lab, level = farm_labels(farms, tf, shape)
    out["buffer_level"] = level
    log("feat.labels", grid="base",
        lvl_neg5=int((level == 0).sum()), lvl_neg2=int((level == 1).sum()),
        lvl_none=int((level == 2).sum()), lvl_failed=int((level < 0).sum()))

    # ---- per-date level, uniformity, K-distribution texture (BASE grid) ----
    for d in DATES:
        g = G[d]
        mean, med, std, cnt = labelled_stats(g, lab, n)
        dbm = 10 * np.log10(np.where(mean > 0, mean, np.nan))
        cv = std / np.where(mean > 0, mean, np.nan)
        looks = estimate_looks(cv, cnt)
        inv_alpha = kdist_texture(mean, std, cnt, looks)
        out[f"g0_db_{d}"] = dbm
        out[f"g0_lin_{d}"] = mean
        out[f"cv_{d}"] = cv
        out[f"ktex_{d}"] = inv_alpha        # 1/alpha: 0 = uniform, larger = heterogeneous
        out[f"npix_{d}"] = cnt
        out[f"inc_{d}"] = labelled_stats(TH[d], lab, n)[0]
        log("feat.date", date=d, n_with_pixels=int((cnt > 0).sum()),
            looks_est=round(looks, 2),
            db_med=round(float(np.nanmedian(dbm)), 2),
            cv_med=round(float(np.nanmedian(cv)), 3),
            ktex_med=round(float(np.nanmedian(inv_alpha)), 4),
            ktex_nan=int(np.isnan(inv_alpha).sum()),
            ktex_zero=int((inv_alpha == 0).sum()))

    # ---- temporal features ----
    db = out[[f"g0_db_{d}" for d in DATES]].values
    lin = out[[f"g0_lin_{d}" for d in DATES]].values
    out["d_aug_jun19"] = db[:, 2] - db[:, 1]     # cleanest pair: geometry matched [G4]
    out["d_oct_aug"] = db[:, 3] - db[:, 2]
    out["d_oct_jun"] = db[:, 3] - db[:, 0]
    jun = np.nanmean(db[:, :2], axis=1)          # bare-soil baseline [B12]
    out["jun_baseline_db"] = jun
    out["ref_aug"] = db[:, 2] - jun              # vegetation-attributable component
    out["ref_oct"] = db[:, 3] - jun
    out["temporal_cv"] = np.nanstd(lin, axis=1) / np.nanmean(lin, axis=1)
    out["temporal_range_db"] = np.nanmax(db, axis=1) - np.nanmin(db, axis=1)
    # season integral: trapezoid over day-of-year, in LINEAR power [B10]
    doy = np.array([157, 170, 226, 286], dtype="float64")
    out["season_integral"] = np.trapezoid(lin, doy, axis=1)

    # ---- GLCM entropy on the FINE grid, unfiltered ----
    Gf, _, tff, shapef = load_grid("fine")
    labf, levelf = farm_labels(farms, tff, shapef)
    idx = np.arange(n)
    for d in DATES:
        dbimg = 10 * np.log10(np.where(Gf[d] > 0, Gf[d], np.nan))
        ent = glcm_entropy(dbimg, labf, n, idx)
        out[f"glcm_ent_{d}"] = ent
        # Dropping to 8 levels cut the size artefact from rho=0.95 to ~0.48, but a
        # residual dependence on sample count remains and is purely an estimator
        # effect, not canopy structure. Regress it out against log(npix) and use the
        # RESIDUAL as the texture feature. [B5, G8]
        npx = out[f"npix_{d}"].values.astype("float64")
        ok = np.isfinite(ent) & (npx > 0)
        resid = np.full(n, np.nan)
        if ok.sum() > 30:
            x = np.log(npx[ok])
            A = np.c_[np.ones(ok.sum()), x]
            c, *_ = np.linalg.lstsq(A, ent[ok], rcond=None)
            resid[ok] = ent[ok] - A @ c
        out[f"glcm_resid_{d}"] = resid
        log("feat.glcm", date=d, n=int(np.isfinite(ent).sum()),
            med=round(float(np.nanmedian(ent)), 3),
            resid_std=round(float(np.nanstd(resid)), 4))

    # ---- coverage / QC ----
    npix = out[[f"npix_{d}" for d in DATES]].values
    out["n_dates"] = (npix > 0).sum(axis=1)
    out["qc_flag"] = np.where(out["n_dates"] == 4, "ok",
                       np.where(out["n_dates"] == 0, "no_sar_data", "partial"))
    p = RESULTS / "farm_features.csv"
    out.to_csv(p, index=False)
    log("feat.done", path=p.name, rows=len(out), cols=out.shape[1],
        ok=int((out.qc_flag == "ok").sum()),
        partial=int((out.qc_flag == "partial").sum()),
        no_data=int((out.qc_flag == "no_sar_data").sum()))
    print(out[["area_ha", "g0_db_20251013", "cv_20251013", "ktex_20251013",
               "glcm_ent_20251013", "ref_aug", "ref_oct", "temporal_cv",
               "d_aug_jun19"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
