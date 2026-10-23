"""Stage I7 / proposal P-B -- discriminative clustering with an MRF spatial prior.

WHY THIS EXISTS. The D4 crop map is a hand-weighted evidence sum: I chose the
coefficients in `crop_evidence` from the literature and from Round 1's measured
signatures. That is defensible but it is my prior, not the data's structure. P-B
asks the opposite question -- if we let the FEATURES cluster themselves under a
spatial smoothness prior and only impose the area shares, do we land in the same
place? Agreement is evidence the map is real; disagreement localises where the
hand weights are carrying the result.

THE MODEL (Bi, Sun, Zhu, "A Graph-Based Semisupervised Deep Learning Model for
PolSAR Image Classification" / the discriminative-clustering formulation,
TGRS 2017) [E2]:

    E(Y, W | X) = Ec(Y, W | X) + Es(Y | X)

  Ec  softmax regression on the features with L2 on W -- the DISCRIMINATIVE term
  Es  MRF label smoothness, S_ij = |y_i - y_j| * exp(-||v_i - v_j||^2 / 2 sigma)

minimised by ALTERNATION: fix Y, fit W (convex, L-BFGS); fix W, update Y under
the unary -log p(y|x,W) plus the pairwise term. Labels are LATENT throughout --
nothing here is trained on ground truth, and per [J7] the recovered Round 1 crop
mix enters only as the area constraint, never as a per-farm label.

TWO DEPARTURES FROM THE PAPER, BOTH DELIBERATE:

1. ICM instead of belief propagation for the Y-step. On a planar 8-NN graph of
   966 nodes with a 5-label alphabet, ICM converges in a handful of sweeps and
   the energy is monotone non-increasing by construction, which is checkable.
   BP would buy a better optimum of an objective whose weights are themselves
   assumptions. Not worth the machinery.
   # ponytail: ICM is a local minimum; swap to BP if the energy plateaus high.

2. The class-imbalance correction. The paper reweights each class by 1/N_j.
   Here that matters more than usual: Cotton is 43% of village area and Maize
   6% [J1], so an unweighted fit would learn "predict Cotton" and stop. Applied
   as sklearn's `class_weight="balanced"`, which IS the 1/N_j rule.

Run:  python src/pb_mrf.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, FARMS, RESULTS, RESULTS_AUX, log
from d4_submission import fit_prior, z

K_NN = 8           # same graph as the Moran's I in [J8], so the two are comparable
BETA = 0.6         # MRF coupling. Below ~0.3 nothing smooths, above ~1.5 it floods.
SIGMA = 1.0        # feature-distance scale in the edge weight, in z units

# Features the clustering is allowed to see. Deliberately RAW and standardised
# rather than the D4 evidence combinations -- feeding it `crop_evidence` output
# would just relearn my own weights and the agreement number would be circular.
# Exclusions carry their own findings: ktex [H2] (single-look K-distribution is
# speckle, not texture), subcoh [I1] (sub-look coherence is dominated by geometry),
# vwc_20250606 [K3] (degenerate -- it IS the WCM calibration date).
FEATS = ["g0_db_20250606", "g0_db_20250619", "g0_db_20250814", "g0_db_20251013",
         "d_aug_jun06", "d_oct_aug", "d_oct_jun", "temporal_cv", "temporal_range_db",
         "season_integral", "cv_20250814", "glcm_resid_20250814", "glcm_ent_20250814",
         "vwc_20251013"]


def features(f):
    """Standardised design matrix. Robust z per column, NaN -> 0 (= the median)."""
    cols = [c for c in FEATS if c in f.columns]
    X = np.column_stack([z(f[c].values) for c in cols])
    return np.where(np.isfinite(X), X, 0.0), cols


def graph(f, k=K_NN):
    """k-NN graph on farm centroids, with Bi et al.'s feature-similarity weights.

    Edge weight exp(-||v_i - v_j||^2 / 2 sigma) means adjacency alone does NOT
    force agreement -- two touching fields that look different in the features
    are only weakly coupled. That is the whole point: the prior says "neighbours
    tend to grow the same thing", not "neighbours must".
    """
    g = gpd.read_file(FARMS).to_crs(3857)
    g = g.iloc[: len(f)] if len(g) >= len(f) else g
    xy = np.column_stack([g.geometry.centroid.x.values, g.geometry.centroid.y.values])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, idx = nn.kneighbors(xy)
    return idx[:, 1:]                      # drop self


def energy(lab, logp, nb, w):
    """Total E = Ec + Es. Monotone non-increase of this is the ICM self-check."""
    ec = -logp[np.arange(len(lab)), lab].sum()
    es = BETA * (w * (lab[:, None] != lab[nb])).sum()
    return ec + es, ec, es


def icm(logp, nb, w, lab, sweeps=30):
    """Fix W, minimise over Y. Each node takes the label minimising its own
    unary + pairwise cost given the current neighbours; repeat to convergence."""
    lab = lab.copy()
    prev = None
    for s in range(sweeps):
        for i in np.random.permutation(len(lab)):
            pair = BETA * np.array(
                [(w[i] * (lab[nb[i]] != c)).sum() for c in range(logp.shape[1])])
            lab[i] = np.argmin(-logp[i] + pair)
        e = energy(lab, logp, nb, w)[0]
        if prev is not None and abs(prev - e) < 1e-6:
            break
        prev = e
    return lab, s + 1


def pb(f, prior_shares, P_init, seed=0):
    """Alternating minimisation. Returns (labels, posterior, per-iteration trace)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    X, cols = features(f)
    nb = graph(f)
    # edge weights from feature distance, per Bi et al.'s S_ij
    d2 = ((X[:, None, :] - X[nb]) ** 2).sum(-1)
    w = np.exp(-d2 / (2.0 * SIGMA * X.shape[1]))

    area = f["area_ha"].values.astype("float64")
    area = np.where(np.isfinite(area) & (area > 0), area, np.nanmedian(area))
    prior = np.array([prior_shares[c] for c in CROPS], dtype="float64")
    prior = prior / prior.sum()

    lab = P_init.argmax(axis=1)            # warm start from the D4 posterior
    trace = []
    for it in range(8):
        # --- W-step: convex, closed by L-BFGS. 1/N_j via class_weight.
        if len(np.unique(lab)) < 2:
            break
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X, lab)
        Pw = np.zeros((len(f), len(CROPS)))
        Pw[:, clf.classes_] = clf.predict_proba(X)
        # --- area constraint [J3]/[J1], applied to the unary BEFORE the MRF so
        # smoothing cannot then be blamed for breaking it.
        Q, b = fit_prior(np.maximum(Pw, 1e-9), prior, area)
        logp = np.log(np.maximum(Q, 1e-12))
        # --- Y-step
        new, sweeps = icm(logp, nb, w, lab)
        e, ec, es = energy(new, logp, nb, w)
        chg = float((new != lab).mean())
        trace.append(dict(it=it, E=e, Ec=ec, Es=es, changed=chg, sweeps=sweeps))
        lab = new
        if chg < 0.005:
            break
    Qf, _ = fit_prior(np.maximum(Q, 1e-9), prior, area)
    return lab, Qf, trace


def selfcheck():
    """Two properties that must hold or the optimiser is not optimising.

    1. ICM never increases the energy.
    2. With a strong coupling and pure-noise unaries, the MRF must produce MORE
       neighbour agreement than the unaries alone -- otherwise the pairwise term
       is not doing anything and every agreement number below is meaningless.
    """
    global BETA
    rng = np.random.default_rng(0)
    n, k, C = 200, 4, 3
    nb = np.array([rng.choice([j for j in range(n) if j != i], k, replace=False)
                   for i in range(n)])
    logp = np.log(rng.dirichlet(np.ones(C), n))
    w = np.ones((n, k))
    lab0 = logp.argmax(1)
    keep = BETA
    BETA = 2.0
    lab1, _ = icm(logp, nb, w, lab0)
    e0 = energy(lab0, logp, nb, w)[0]
    e1 = energy(lab1, logp, nb, w)[0]
    assert e1 <= e0 + 1e-9, f"ICM increased energy: {e0} -> {e1}"
    a0 = (lab0[:, None] == lab0[nb]).mean()
    a1 = (lab1[:, None] == lab1[nb]).mean()
    assert a1 > a0, f"MRF did not smooth: {a0:.3f} -> {a1:.3f}"
    BETA = keep
    print(f"pb selfcheck OK (E {e0:.1f}->{e1:.1f}, neighbour agreement {a0:.2f}->{a1:.2f})")


def main():
    selfcheck()
    log("pb.start")
    f = pd.read_csv(RESULTS / "farm_features.csv")
    dbg = pd.read_csv(RESULTS / "d4_debug.csv")
    f = f.merge(dbg[["farm_id", "crop_type"] + [f"p_{c}" for c in CROPS]],
                on="farm_id", how="left")
    P_init = f[[f"p_{c}" for c in CROPS]].fillna(1.0 / len(CROPS)).values
    truth = pd.read_csv(Path(__file__).resolve().parent.parent / "data_aux" /
                        "sokhda_phase1_crop_shares.csv")
    prior = {r.crop: r.sokhda_share for r in truth.itertuples() if r.crop in CROPS}

    lab, Q, trace = pb(f, prior, P_init)
    pb_crop = np.array(CROPS)[lab]
    d4_crop = f["crop_type"].values

    print("\n=== alternating minimisation ===")
    print(pd.DataFrame(trace).round(3).to_string(index=False))

    area = f["area_ha"].values
    print("\n=== area share, P-B vs D4 vs the [J1] constraint ===")
    rows = []
    for c in CROPS:
        rows.append(dict(crop=c, prior=round(prior[c], 3),
                         D4=round(area[d4_crop == c].sum() / area.sum(), 3),
                         PB=round(area[pb_crop == c].sum() / area.sum(), 3)))
    print(pd.DataFrame(rows).to_string(index=False))

    agree = (pb_crop == d4_crop)
    print(f"\n=== agreement P-B vs D4 ===\nby farm  {agree.mean():.3f}"
          f"\nby area  {area[agree].sum() / area.sum():.3f}")
    print("\nper-crop recall of the D4 map by P-B:")
    print(pd.crosstab(pd.Series(d4_crop, name="D4"),
                      pd.Series(pb_crop, name="PB")).to_string())

    f["pb_crop"] = pb_crop
    for i, c in enumerate(CROPS):
        f[f"pb_p_{c}"] = Q[:, i]
    keep = ["farm_id", "pb_crop"] + [f"pb_p_{c}" for c in CROPS]
    f[keep].to_csv(RESULTS_AUX / "pb_crop.csv", index=False)
    log("pb.done", agree_farm=round(float(agree.mean()), 3),
        agree_area=round(float(area[agree].sum() / area.sum()), 3))


if __name__ == "__main__":
    main()
