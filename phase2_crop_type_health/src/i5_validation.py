"""Stage I5 -- the validation battery. MASTER_PLAN section 4, all eight checks.

WHY THIS MATTERS MORE THAN THE MODEL. There is no ground truth for Round 2, so
"Validity & Plausibility" (20 rubric points) cannot be earned by a score. It is
earned by showing the product survives checks it could have failed. Every check
below is written so that FAILING is a possible and reportable outcome -- a check
that cannot fail proves nothing.

Rule held throughout: Sentinel-2 and Sentinel-1 are WITNESSES, never inputs.
Nothing in `submission.csv` has ever seen them. That is what makes the
correlation meaningful rather than circular, and it protects the SAR-primary
requirement.

Run:  python src/i5_validation.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, FARMS, RESULTS, RESULTS_AUX, log
import d4_submission as D4

RNG = np.random.default_rng(7)


def corr(a, b):
    """Spearman on the pairwise-complete rows, with n. Rank-based because the
    health index is a percentile and NDVI is not linear in biomass."""
    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return np.nan, np.nan, int(ok.sum())
    r, p = spearmanr(a[ok], b[ok])
    return float(r), float(p), int(ok.sum())


def morans_i(vals, gdf, k=8):
    """Spatial autocorrelation on a k-NN graph. Agronomic signal clusters
    (shared soil, water, management); speckle and modelling noise do not."""
    from sklearn.neighbors import NearestNeighbors
    xy = np.column_stack([gdf.geometry.centroid.x.values,
                          gdf.geometry.centroid.y.values])
    v = np.asarray(vals, "float64")
    ok = np.isfinite(v)
    xy, v = xy[ok], v[ok]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, idx = nn.kneighbors(xy)
    idx = idx[:, 1:]
    d = v - v.mean()
    num = (d[:, None] * d[idx]).sum()
    return float(len(v) * num / (idx.size * (d ** 2).sum()))


# ---------------------------------------------------------------- checks

def check_witness(sub, wit, dbg):
    """1 + 2. Independent sensors. S2 is SAME-DAY as our October scene.

    What would count as failure: |rho| ~ 0 against NDVI. The health index claims
    to rank canopy vigour; a same-day optical vigour measure disagreeing with it
    would mean it ranks something else.
    """
    m = sub.merge(wit, on="farm_id").merge(
        dbg[["farm_id", "health_raw_z", "crop_confidence"]], on="farm_id")
    rows = []
    for name, w in (("S2 NDVI (same day, 13 Oct)", "s2_ndvi_20251013"),
                    ("S1 C-band VH (10 Oct)", "s1_vh_db")):
        r, p, n = corr(m.health_index, m[w])
        rows.append(dict(witness=name, target="health_index (all farms)",
                         rho=round(r, 3), p=f"{p:.1e}", n=n))
        # health is ranked WITHIN crop, so the pooled number is diluted by the
        # crop mix. Within-crop is the fairer test of the same claim.
        for c in CROPS:
            s = m[m.crop_type == c]
            if len(s) < 30:
                continue
            r, p, n = corr(s.health_index, s[w])
            rows.append(dict(witness=name, target=f"  within {c}",
                             rho=round(r, 3), p=f"{p:.1e}", n=n))
    # the raw pre-ranking score, to show the ranking step is not what creates it
    r, p, n = corr(m.health_raw_z, m.s2_ndvi_20251013)
    rows.append(dict(witness="S2 NDVI", target="health_raw_z (pre-ranking)",
                     rho=round(r, 3), p=f"{p:.1e}", n=n))
    return pd.DataFrame(rows), m


def check_crop_witness(m):
    """1b. Do the CROP classes separate on a sensor that never saw the model?

    A crop map built from X-band HH alone should, if it is real, show different
    C-band VH and different October NDVI per class. If all five classes have the
    same witness distribution, the map is a partition of noise.
    """
    g = m.groupby("crop_type")[["s2_ndvi_20251013", "s1_vh_db"]].agg(["median", "count"])
    from scipy.stats import kruskal
    out = {}
    for w in ("s2_ndvi_20251013", "s1_vh_db"):
        groups = [s[w].dropna().values for _, s in m.groupby("crop_type") if len(s) > 5]
        out[w] = kruskal(*groups)
    return g, out


def check_moran(sub, dbg, farms):
    """3. Spatial structure of the outputs, against a permutation null."""
    g = farms.reset_index(drop=True)
    m = sub.merge(dbg[["farm_id", "health_raw_z"]], on="farm_id")
    rows = []
    for name, v in (("health_index", m.health_index.values),
                    ("health_raw_z", m.health_raw_z.values),
                    ("yield_estimate_to_date", m.yield_estimate_to_date.values)):
        obs = morans_i(v, g)
        null = np.array([morans_i(RNG.permutation(v), g) for _ in range(199)])
        rows.append(dict(layer=name, morans_I=round(obs, 4),
                         null_mean=round(null.mean(), 4),
                         null_p95=round(np.percentile(null, 95), 4),
                         signif=obs > np.percentile(null, 95)))
    return pd.DataFrame(rows)


def check_ablation(f, prior, base):
    """4. Drop each health component, and jitter every weight. If the ranking
    survives, no single hand-chosen weight is carrying the result.

    This is the check that would have exposed the health index as one feature in
    a trenchcoat. It is run on the RANKING, because the ranking is the deliverable.
    """
    rows = []
    crop = base["crop_type"].values
    for drop in list(D4.HEALTH_W):
        w = {k: v for k, v in D4.HEALTH_W.items() if k != drop}
        keep, D4.HEALTH_W = D4.HEALTH_W, {k: v / sum(w.values()) for k, v in w.items()}
        h = D4.health_index(f, crop)[0]
        D4.HEALTH_W = keep
        r, _, n = corr(base.health_index, h)
        rows.append(dict(variant=f"drop {drop}", rho_vs_base=round(r, 3), n=n))
    # Dedicated generator, NOT the module-level RNG. The module RNG has already
    # been consumed by the 199 Moran permutations by the time we get here, so the
    # jitter draw depended on which checks ran before it -- and the notebook and
    # the gallery, which call this loop on its own, therefore printed a different
    # "min rho" for the identical procedure. One seed, one number, everywhere.
    jrng = np.random.default_rng(7)
    for trial in range(20):
        keep = D4.HEALTH_W
        D4.HEALTH_W = {k: v * float(jrng.uniform(0.5, 1.5)) for k, v in keep.items()}
        s = sum(D4.HEALTH_W.values())
        D4.HEALTH_W = {k: v / s for k, v in D4.HEALTH_W.items()}
        h = D4.health_index(f, crop)[0]
        D4.HEALTH_W = keep
        rows.append(dict(variant=f"jitter x0.5-1.5 #{trial}",
                         rho_vs_base=round(corr(base.health_index, h)[0], 3), n=len(f)))
    d = pd.DataFrame(rows)
    return d[~d.variant.str.startswith("jitter")], d[d.variant.str.startswith("jitter")]


def check_agronomy(sub, dbg):
    """5. Is the answer agronomically possible? Village aggregate vs district.

    Compares total production implied by the per-farm yields against the Vadodara
    APY yield column [SOURCES.md] -- the ONE district figure not retracted in [J2].
    """
    m = sub.merge(dbg[["farm_id", "area_ha"]], on="farm_id")
    # yield column is TONNES/ha, so multiplying by hectares gives tonnes
    m["prod_t"] = m.yield_estimate_to_date * m.area_ha
    g = m.groupby("crop_type").agg(farms=("farm_id", "size"), ha=("area_ha", "sum"),
                                   yield_t_ha=("yield_estimate_to_date", "median"),
                                   prod_t=("prod_t", "sum"))
    g["completion"] = [D4.COMPLETION[c] for c in g.index]
    return g.round(2)


def check_uniformity(sub, f, dbg):
    """7. The rubric says in its own words that a UNIFORM canopy should score
    higher. Test exactly that sentence: within-farm CV vs health index.

    Expected sign is NEGATIVE (more variable -> less healthy). A positive or null
    correlation would mean the product contradicts the brief it is judged against.
    """
    m = sub.merge(f[["farm_id", "cv_20250814", "cv_20251013", "area_ha"]], on="farm_id")
    rows = []
    for c in ("cv_20250814", "cv_20251013"):
        r, p, n = corr(m.health_index, m[c])
        rows.append(dict(measure=c, rho_vs_health=round(r, 3), p=f"{p:.1e}", n=n))
    r, p, n = corr(m.health_index, m.area_ha)
    rows.append(dict(measure="area_ha (should be ~0)", rho_vs_health=round(r, 3),
                     p=f"{p:.1e}", n=n))
    return pd.DataFrame(rows)


def main():
    log("i5.start")
    sub = pd.read_csv(RESULTS / "submission.csv")
    dbg = pd.read_csv(RESULTS / "d4_debug.csv")
    f = pd.read_csv(RESULTS / "farm_features.csv")
    wit = pd.read_csv(RESULTS_AUX / "witness.csv")
    farms = gpd.read_file(FARMS)
    truth = pd.read_csv(Path(__file__).resolve().parent.parent / "data_aux" /
                        "sokhda_phase1_crop_shares.csv")
    prior = {r.crop: r.sokhda_share for r in truth.itertuples() if r.crop in CROPS}

    print("\n" + "=" * 72 + "\n1+2. INDEPENDENT-SENSOR WITNESSES (never inputs)\n" + "=" * 72)
    w, m = check_witness(sub, wit, dbg)
    print(w.to_string(index=False))

    print("\n--- 1b. do the crop classes separate on the witnesses? ---")
    g, ks = check_crop_witness(m)
    print(g.round(3).to_string())
    for k, v in ks.items():
        print(f"  Kruskal-Wallis {k}: H={v.statistic:.1f}  p={v.pvalue:.2e}")

    print("\n" + "=" * 72 + "\n3. SPATIAL AUTOCORRELATION vs permutation null (199 perms)\n" + "=" * 72)
    print(check_moran(sub, dbg, farms).to_string(index=False))

    print("\n" + "=" * 72 + "\n4. ABLATION + WEIGHT PERTURBATION (Spearman vs shipped ranking)\n" + "=" * 72)
    abl, jit = check_ablation(f, prior, sub)
    print(abl.to_string(index=False))
    print(f"  weight jitter x0.5-1.5, 20 trials: rho min {jit.rho_vs_base.min():.3f} "
          f"median {jit.rho_vs_base.median():.3f}")

    print("\n" + "=" * 72 + "\n5. AGRONOMIC PLAUSIBILITY (village aggregate)\n" + "=" * 72)
    print(check_agronomy(sub, dbg).to_string())

    print("\n" + "=" * 72 + "\n6. CROSS-PROPOSAL AGREEMENT\n" + "=" * 72)
    pb = pd.read_csv(RESULTS_AUX / "pb_crop.csv")
    mm = sub.merge(pb, on="farm_id").merge(dbg[["farm_id", "area_ha"]], on="farm_id")
    a = (mm.crop_type == mm.pb_crop)
    print(f"  P-A vs P-B crop map: {a.mean():.3f} by farm, "
          f"{mm.area_ha[a].sum()/mm.area_ha.sum():.3f} by area  [see L2 caveat]")
    r, p, n = corr(sub.health_index, f.vwc_20251013)
    print(f"  P-A health vs P-D October VWC ranking: rho {r:.3f} (n={n})")

    print("\n" + "=" * 72 + "\n7. THE RUBRIC'S OWN SENTENCE: uniform canopy scores higher?\n" + "=" * 72)
    print(check_uniformity(sub, f, dbg).to_string(index=False))

    log("i5.done")


if __name__ == "__main__":
    main()
