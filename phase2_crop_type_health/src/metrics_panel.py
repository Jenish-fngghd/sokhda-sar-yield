"""Fixed metric panel for validating every change against the same yardstick.

Written for the review-fixes pass. The point is that no fix gets accepted on a
narrative -- each one is measured on the identical panel before and after, and a
fix that moves nothing, or moves the wrong thing, is reported as such.

Run:  python src/metrics_panel.py                 # panel for results/submission.csv
      python src/metrics_panel.py --tag baseline  # label the run in the ledger
      python src/metrics_panel.py --compare results/baseline/submission_BASELINE.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, RESULTS, RESULTS_AUX, farm_centroids, log


def _rho(a, b):
    """Spearman on the pairwise-complete rows; nan if too few."""
    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    return float(spearmanr(a[m], b[m]).statistic)


def morans_i(values, xy, k=8, seed=0, nperm=199):
    """Moran's I on a k-nearest-neighbour graph, with a permutation p-value.

    Row-standardised weights, so I is bounded in roughly [-1, 1] and the
    expectation under no autocorrelation is -1/(n-1) rather than 0.
    """
    from scipy.spatial import cKDTree
    v = np.asarray(values, "float64")
    m = np.isfinite(v)
    v, xy = v[m], np.asarray(xy)[m]
    n = len(v)
    if n < 20:
        return np.nan, np.nan
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    idx = idx[:, 1:]                                  # drop self
    z = v - v.mean()
    num = (z[:, None] * z[idx]).mean(axis=1).sum()    # row-standardised
    I = num / (z ** 2).sum() * 1.0
    rng = np.random.default_rng(seed)
    null = np.empty(nperm)
    for i in range(nperm):
        zp = rng.permutation(z)
        null[i] = (zp[:, None] * zp[idx]).mean(axis=1).sum() / (zp ** 2).sum()
    # (1 + hits) / (nperm + 1): a permutation p can never be 0 -- the observed value is
    # itself one of the possible arrangements. Reporting p=0.000 for Moran's I is an
    # impossible claim; the floor here is 1/200 = 0.005.
    p = float((1 + int((null >= I).sum())) / (nperm + 1))
    return float(I), p


def panel(sub_path=RESULTS / "submission.csv"):
    """Metric panel for one submission.

    A submission is only interpretable against the features it was BUILT from, so when
    scoring a snapshot in results/baseline/ we load that snapshot's features and debug
    table too. Mixing a baseline submission with current features would attribute a
    feature-stage change to whatever fix is being tested -- exactly the comparison this
    panel exists to make trustworthy.
    """
    sub = pd.read_csv(sub_path)
    snap = sub_path.parent if sub_path.parent.name == "baseline" else RESULTS
    suffix = "_BASELINE" if snap.name == "baseline" else ""

    def load(stem):
        cand = snap / f"{stem}{suffix}.csv"
        return cand if cand.exists() else RESULTS / f"{stem}.csv"

    feat = pd.read_csv(load("farm_features"))
    dbg_path = load("d4_debug")
    dbg = pd.read_csv(dbg_path) if dbg_path.exists() else None
    wit_path = RESULTS_AUX / "witness.csv"      # witnesses are external, never versioned
    wit = pd.read_csv(wit_path) if wit_path.exists() else None

    d = sub.merge(feat, on="farm_id", how="left")
    if wit is not None:
        d = d.merge(wit, on="farm_id", how="left")
    if dbg is not None:
        d = d.merge(dbg[["farm_id", "area_ha"]].rename(columns={"area_ha": "area_dbg"}),
                    on="farm_id", how="left")

    M = {}

    # --- schema / coverage -------------------------------------------------
    M["rows"] = len(sub)
    M["nan_total"] = int(sub.isna().sum().sum())
    M["farm_ids_ok"] = bool(sorted(sub.farm_id) == list(range(1, 967)))
    M["crops_ok"] = not (set(sub.crop_type) - set(CROPS))
    M["health_min"], M["health_max"] = float(sub.health_index.min()), float(sub.health_index.max())
    M["yield_max_t_ha"] = float(sub.yield_estimate_to_date.max())

    # --- crop mix vs the R1 prior -----------------------------------------
    area = d.groupby("crop_type")["area_ha"].sum()
    M["area_share"] = {c: round(100 * float(area.get(c, 0)) / float(area.sum()), 2) for c in CROPS}

    # --- health distribution: is it uniform BY CONSTRUCTION? --------------
    # a percentile rank forces mean 50 and sd ~28.9 in every crop; a magnitude-
    # preserving index does not. This is the tell.
    g = sub.groupby("crop_type")["health_index"]
    M["health_mean_by_crop"] = {c: round(float(v), 2) for c, v in g.mean().items()}
    M["health_sd_by_crop"] = {c: round(float(v), 2) for c, v in g.std().items()}
    M["health_village_mean"] = round(float(sub.health_index.mean()), 3)

    # --- is yield independent of health? ----------------------------------
    # within a crop both crop-level terms are constant, so if completion and
    # accumulation are per-crop constants this comes back exactly 1.000.
    M["rho_health_yield_within_crop"] = {}
    for c in CROPS:
        m = sub.crop_type == c
        if m.sum() > 10:
            M["rho_health_yield_within_crop"][c] = round(
                _rho(sub.health_index[m], sub.yield_estimate_to_date[m]), 4)

    # --- residual biases ---------------------------------------------------
    M["rho_health_area"] = round(_rho(sub.health_index, d["area_ha"]), 4)
    M["rho_health_npix_oct"] = round(_rho(sub.health_index, d["npix_20251013"]), 4)

    # --- witnesses (never inputs) -----------------------------------------
    if wit is not None:
        M["rho_health_ndvi"] = round(_rho(sub.health_index, d["s2_ndvi_20251013"]), 4)
        M["rho_health_vh"] = round(_rho(sub.health_index, d["s1_vh_db"]), 4)
        M["rho_health_ndvi_by_crop"] = {}
        for c in CROPS:
            m = sub.crop_type == c
            if m.sum() > 20:
                M["rho_health_ndvi_by_crop"][c] = round(
                    _rho(sub.health_index[m], d.loc[m, "s2_ndvi_20251013"]), 3)
        # crop separation on each witness -- the strongest existing result
        for col, name in [("s2_ndvi_20251013", "ndvi"), ("s1_vh_db", "vh")]:
            groups = [d.loc[sub.crop_type == c, col].dropna().values for c in CROPS]
            groups = [x for x in groups if len(x) > 5]
            if len(groups) > 2:
                H, p = kruskal(*groups)
                M[f"crop_sep_{name}_H"] = round(float(H), 1)
                M[f"crop_sep_{name}_p"] = f"{p:.2e}"
                M[f"crop_median_{name}"] = {
                    c: round(float(np.nanmedian(d.loc[sub.crop_type == c, col])), 3)
                    for c in CROPS}

    # --- the rubric's own uniformity criterion ----------------------------
    # cv_20250814 is IN the index (circular); cv_20251013 is the independent test.
    M["rho_health_cv_aug_CIRCULAR"] = round(_rho(sub.health_index, d["cv_20250814"]), 4)
    M["rho_health_cv_oct_INDEPENDENT"] = round(_rho(sub.health_index, d["cv_20251013"]), 4)

    # --- spatial structure -------------------------------------------------
    cent = farm_centroids()
    cent = cent[sub.farm_id.values - 1]
    I, p = morans_i(sub.health_index.values, cent)
    M["morans_I_health"] = round(I, 4)
    M["morans_p_health"] = round(p, 4)

    return M


def show(M, title="PANEL"):
    print(f"\n{'='*68}\n{title}\n{'='*68}")
    for k, v in M.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"    {kk:<12} {vv}")
        else:
            print(f"{k:<34} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="current")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()

    M = panel()
    show(M, f"METRIC PANEL -- {a.tag}")
    log("panel", tag=a.tag, health_village_mean=M["health_village_mean"],
        rho_health_area=M["rho_health_area"],
        rho_health_ndvi=M.get("rho_health_ndvi"),
        morans_I=M.get("morans_I_health"))

    if a.compare:
        B = panel(Path(a.compare))
        show(B, "BASELINE")
        print(f"\n{'='*68}\nDELTA (current - baseline)\n{'='*68}")
        for k in M:
            if isinstance(M[k], (int, float)) and isinstance(B.get(k), (int, float)):
                dv = M[k] - B[k]
                flag = "  <-- moved" if abs(dv) > 1e-9 else ""
                print(f"{k:<34} {B[k]:>10} -> {M[k]:>10}  ({dv:+.4f}){flag}")


if __name__ == "__main__":
    main()
