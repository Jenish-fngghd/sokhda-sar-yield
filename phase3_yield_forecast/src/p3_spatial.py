"""T22 / P3-20: is the modifier ranking the FARM, or ranking WHERE THE FARM IS?

Every approach so far -- the shipped health index, WCM, SAFY, the X+C fusion, the
climatology -- treats a parcel as an independent unit. None of them ever asked
whether the ranking they produce has SPATIAL STRUCTURE larger than a field.

That question is not cosmetic. The deliverable is "which farms need attention".
If the within-crop ranking is a smooth landscape gradient -- soil texture, water
table, canal command, distance to the drainage line -- then the advice a farmer
reads off it is "your location is poor", which no in-season action can change.
Only the part of the ranking that varies FIELD TO FIELD BETWEEN NEIGHBOURS is
something management could have caused and management could fix.

So: decompose the within-crop health residual into
    r  =  landscape (spatially smooth)  +  farm-specific (neighbour contrast)
and report the variance split, with four controls that can kill it.

CONTROLS, all of which could fail:

  C1  Permutation null.  Moran's I on r, against 999 within-crop shuffles of the
      SAME values over the SAME centroids. If observed I sits inside the null,
      there is no spatial structure and there is nothing to decompose. This test
      is the whole finding's licence to exist.

  C2  Geometry.  Incidence angle varies smoothly across a single Capella scene by
      construction, so ANY quantity carrying a residual incidence term inherits a
      landscape-scale gradient for free. Moran's I is therefore recomputed on r
      after regressing out incidence. If it collapses, the "landscape" is the
      radar's viewing geometry, not the land.

  C3  Static ground.  The June bare-soil baseline is soil brightness and roughness
      before a canopy exists. Regressing it out too separates "this ground is
      permanently bright" from "this season's canopy is doing well here".

  C2+C3 together are the honest test: what survives BOTH is spatial structure in
  the SEASON, not in the sensor and not in the dirt.

  C4  Invariance.  The decomposition must not touch the forecast. The village total
      is recomputed from the untouched forecast and reported. This is a diagnostic,
      not a change.

Range is compared against the mean field size (sqrt(447.5 ha / 966) ~ 68 m). A
variogram range at 68 m means the ranking is field-scale. A range at 500 m+ means
one field in a bright neighbourhood scores well because of the neighbourhood.

Writes results/tables/p3_spatial.csv, p3_spatial_stats.csv, p3_spatial_variogram.csv.

Run:  py -3.12 -u src/p3_spatial.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, FARMS, TABLES, log

import geopandas as gpd

sys.stdout.reconfigure(encoding="utf8")

UTM = 32643
RNG = np.random.default_rng(20260831)
NPERM = 999
LAGS = np.array([50, 100, 150, 200, 300, 400, 500, 700, 900, 1200, 1600], float)
SMOOTH_R = 400.0      # landscape radius; reported against the variogram range below


def moran(v, W):
    """Moran's I with a row-normalised weight matrix. v must already be mean-zero."""
    n = len(v)
    num = float(v @ (W @ v))
    return (n / W.sum()) * num / float(v @ v)


def perm_p(v, W, groups, nperm=NPERM):
    """Within-group permutation null. Shuffling INSIDE crop keeps crop composition
    fixed, so the null tests location and not crop identity."""
    obs = moran(v, W)
    null = np.empty(nperm)
    idx = np.arange(len(v))
    gi = [idx[groups == g] for g in np.unique(groups)]
    for k in range(nperm):
        p = idx.copy()
        for g in gi:
            p[g] = RNG.permutation(g)
        null[k] = moran(v[p], W)
    return obs, float(null.mean()), float(null.std()), (1 + (null >= obs).sum()) / (nperm + 1)


def variogram(v, D, lags=LAGS):
    """Classical semivariance by distance bin. Returns rows of (centre, gamma, npair)."""
    iu = np.triu_indices(len(v), 1)
    d, sq = D[iu], 0.5 * (v[iu[0]] - v[iu[1]]) ** 2
    edges = np.r_[0.0, lags]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (d >= a) & (d < b)
        out.append((0.5 * (a + b), float(sq[m].mean()) if m.sum() else np.nan, int(m.sum())))
    return np.array(out)


def resid(y, X):
    """Least-squares residual of y on [1, X]. X is (n,) or (n, k)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def main():
    sub = pd.read_csv(TABLES.parent / "submission.csv")
    feat = pd.read_csv(TABLES / "p1_farm_features.csv")
    g = gpd.read_file(FARMS).to_crs(epsg=UTM)
    gid = "farm_id" if "farm_id" in g.columns else g.columns[0]
    cen = g.geometry.centroid
    geo = pd.DataFrame({"farm_id": g[gid].astype(int).to_numpy(),
                        "x": cen.x.to_numpy(), "y": cen.y.to_numpy(),
                        "area_ha": g.geometry.area.to_numpy() / 1e4})

    d = sub.merge(geo, on="farm_id").merge(
        feat[["farm_id", "jun_baseline_db", "inc_20250814"]], on="farm_id", how="left")
    assert len(d) == len(sub), f"centroid join lost rows: {len(d)} vs {len(sub)}"
    n = len(d)
    print(f"{n} farms, {d.area_ha.sum():.1f} ha, mean field {np.sqrt(d.area_ha.mean()*1e4):.0f} m")

    # within-crop residual of the health index -- the exact quantity the modifier ranks
    hi = d.health_index.to_numpy(float)
    r = np.zeros(n)
    for c in CROPS:
        m = (d.crop_type == c).to_numpy()
        if m.sum() < 3:
            continue
        r[m] = (hi[m] - hi[m].mean()) / (hi[m].std() or 1.0)
    d["hi_resid"] = r

    x, y = d.x.to_numpy(), d.y.to_numpy()
    D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])

    # row-normalised inverse-distance weights inside SMOOTH_R, self excluded
    W = np.where((D > 0) & (D <= SMOOTH_R), 1.0 / np.maximum(D, 1.0), 0.0)
    rs = W.sum(1, keepdims=True)
    isolated = int((rs == 0).sum())
    W = W / np.where(rs == 0, 1.0, rs)
    print(f"weights: {SMOOTH_R:.0f} m, mean {(W > 0).sum(1).mean():.1f} neighbours, "
          f"{isolated} isolated")

    crop = d.crop_type.to_numpy()
    rows, vg_rows = [], []

    # ---- variograms: subject and the two confounds, same bins, same centroids ----
    for name, v in (("hi_resid", r),
                    ("incidence", d.inc_20250814.to_numpy(float)),
                    ("jun_baseline", d.jun_baseline_db.to_numpy(float))):
        vv = np.where(np.isfinite(v), v, np.nanmean(v))
        vv = (vv - vv.mean()) / (vv.std() or 1.0)
        vg = variogram(vv, D)
        sill = float(np.nanmean(vg[vg[:, 0] > 900, 1]))
        hit = vg[vg[:, 1] >= 0.95 * sill, 0]      # practical range: 95% of sill
        rng = float(hit[0]) if len(hit) else np.nan
        nugget = float(vg[0, 1])
        # A curve still climbing at the last lag has not reached a sill inside the
        # village, so its "range" is a LOWER BOUND and its structured fraction is
        # understated. Incidence angle does exactly this -- which is the point: the
        # positive control's structure runs off the edge of the AOI. [caught in F6]
        rising = bool(vg[-1, 1] > 1.05 * np.nanmean(vg[-4:-1, 1]))
        print(f"  {name:13s} nugget {nugget:.2f}  sill {sill:.2f}  "
              f"range {'>' if rising else '~'}{rng:.0f} m  "
              f"structured frac {1 - nugget/sill:.2f}{'  [still rising]' if rising else ''}")
        for c_, gm, npair in vg:
            vg_rows.append({"series": name, "lag_m": c_, "gamma": gm, "npair": npair})
        rows.append({"stat": f"vg_{name}", "nugget": nugget, "sill": sill,
                     "range_m": rng, "range_is_lower_bound": rising,
                     "structured_frac": 1 - nugget / sill})

    # ---- C1/C2/C3: Moran's I, raw then with each confound removed ----
    inc = np.nan_to_num(d.inc_20250814.to_numpy(float), nan=float(np.nanmean(d.inc_20250814)))
    jun = np.nan_to_num(d.jun_baseline_db.to_numpy(float), nan=float(np.nanmean(d.jun_baseline_db)))
    tests = {"raw": r,
             "minus_incidence": resid(r, inc[:, None]),
             "minus_inc_and_soil": resid(r, np.column_stack([inc, jun]))}
    morans = {}
    for name, v in tests.items():
        v = v - v.mean()
        obs, mu, sd, p = perm_p(v, W, crop)
        z_ = (obs - mu) / (sd or 1.0)
        morans[name] = obs
        print(f"  Moran {name:19s} I={obs:+.4f}  null {mu:+.4f}+-{sd:.4f}  "
              f"z={z_:+.1f}  p={p:.4f}")
        rows.append({"stat": f"moran_{name}", "I": obs, "null_mean": mu,
                     "null_sd": sd, "z": z_, "p": p})

    # ---- the decomposition, on what survived both confounds ----
    r_clean = tests["minus_inc_and_soil"] - tests["minus_inc_and_soil"].mean()
    land = W @ r_clean                       # neighbourhood mean, self excluded
    farm = r_clean - land
    v_land = float(np.var(land)) / float(np.var(r_clean))
    v_farm = float(np.var(farm)) / float(np.var(r_clean))
    print(f"  variance split: landscape {v_land:.3f}  farm-specific {v_farm:.3f}  "
          f"(cross {1 - v_land - v_farm:+.3f})")
    rows.append({"stat": "variance_split", "landscape": v_land, "farm_specific": v_farm})

    d["landscape"] = np.round(land, 4)
    d["farm_specific"] = np.round(farm, 4)
    d["actionable_rank"] = d.groupby("crop_type").farm_specific.rank(pct=True).round(4)

    # how much does ranking on the actionable part reorder the advice list?
    worst_hi = set(d.nsmallest(97, "hi_resid").farm_id)
    worst_act = set(d.nsmallest(97, "farm_specific").farm_id)
    overlap = len(worst_hi & worst_act)
    print(f"  bottom-decile advice list: {overlap}/97 farms shared between "
          f"raw ranking and actionable ranking")
    rows.append({"stat": "bottom_decile_overlap", "shared": overlap, "of": 97})

    # ---- C4: the forecast must be untouched ----
    total = float((sub.yield_forecast_t_ha * d.area_ha).sum())
    print(f"  C4 village total from untouched forecast: {total:.1f} t")

    d[["village_id", "farm_id", "crop_type", "health_index", "hi_resid",
       "landscape", "farm_specific", "actionable_rank"]].to_csv(
        TABLES / "p3_spatial.csv", index=False)
    pd.DataFrame(rows).to_csv(TABLES / "p3_spatial_stats.csv", index=False)
    pd.DataFrame(vg_rows).to_csv(TABLES / "p3_spatial_variogram.csv", index=False)
    log("p3_spatial", n=n, moran_raw=morans["raw"],
        moran_clean=morans["minus_inc_and_soil"],
        landscape_frac=v_land, farm_frac=v_farm, village_total_t=total)


if __name__ == "__main__":
    main()
