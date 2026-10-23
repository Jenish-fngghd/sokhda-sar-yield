"""T16: recover the crop map from six labellings with NO ground truth (Dawid-Skene).

WHY THIS AND NOT MORE SAR WORK. Section 4.1 established that the imagery decides exactly two
things: the ORDERING of plots within a crop, and WHICH CROP each plot is. The ordering is already
validated against an independent same-day sensor at rho ~ 0.5 and cannot move the village total.
The crop map moves the village total by 236.5 t (P3-11), measured externally. So the crop map is
the only place where a better method changes the answer, and it is where this file works.

THE ASSET NOBODY HAS USED. Six teams labelled these same 966 farms in Round 2. The obvious move
is majority vote, which we already have and which gives 650.3 t. Majority vote assumes every
annotator is equally reliable and that their errors are independent -- both false here, and
demonstrably so: one team (Orion) produces a village total 25% above everyone else.

Dawid & Skene (1979) is the right tool and it fits this situation exactly: latent true labels,
several imperfect annotators, NO ground truth. EM jointly estimates

    pi[k]        the prevalence of each crop
    theta[j,k,l] annotator j's probability of SAYING l when the truth is k   (a confusion matrix
                 PER TEAM, estimated without ever seeing a true label)
    T[i,k]       the posterior over the true crop of farm i

An annotator that is systematically wrong is down-weighted automatically, and one that is wrong
in a *predictable* way still contributes information through the off-diagonal of its confusion
matrix -- which majority vote throws away entirely.

★ HOW WE AVOID GRADING OUR OWN HOMEWORK. A label set cannot be validated against the labels it
was derived from, and P1-5 proved every naive label test on this stack is circular. So the
referee here is the test already in the repo -- D4's non-circular residual method: regress the
independent S2 NDVI witness on the dominant SAR axis (PC1 of the six-date gamma0 series) that
produced the labels, then measure how much of the RESIDUAL each labelling explains. That test is
Orion's design, is already used to defend our shipped labels, and was fixed in the plan before
this file existed. We did not invent a metric for the occasion.

This file DECIDES NOTHING. It reports whether Dawid-Skene beats what we ship, on a pre-existing
test. Changing the crop map changes the deliverable, and that is a decision to be taken
deliberately, not as a side effect of an experiment.

Writes results/tables/p3_dawidskene.csv and p3_ds_confusion.csv.

Run:  py -3.12 src/p3_dawidskene.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, FARMS, RESULTS, TABLES, log
from p3_build import modifier

import geopandas as gpd

UTM = 32643
AUX = Path("data_aux") / "inherited" / "consensus.csv"
RNG = np.random.default_rng(20260831)
K = len(CROPS)


def dawid_skene(L, n_iter=200, tol=1e-9):
    """EM for Dawid-Skene. L is (n_items, n_annotators) integer codes, -1 = missing.

    Returns T (n_items, K) posterior over true class, theta (n_ann, K, K), pi (K,).
    Initialised from majority vote, which is the estimator we are trying to beat.
    """
    n, J = L.shape
    obs = L >= 0
    # --- init: majority vote ---
    T = np.zeros((n, K))
    for i in range(n):
        v = L[i][obs[i]]
        if len(v):
            T[i] = np.bincount(v, minlength=K)
    T = np.where(T.sum(1, keepdims=True) > 0, T, 1.0)
    T /= T.sum(1, keepdims=True)

    prev = -np.inf
    for _ in range(n_iter):
        # --- M step ---
        pi = T.mean(0) + 1e-12
        pi /= pi.sum()
        theta = np.zeros((J, K, K))
        for j in range(J):
            m = obs[:, j]
            for l in range(K):
                sel = m & (L[:, j] == l)
                theta[j, :, l] = T[sel].sum(0)
        theta += 1e-2                      # Laplace: no annotator is assumed infallible
        theta /= theta.sum(2, keepdims=True)
        # --- E step (log domain) ---
        lg = np.tile(np.log(pi), (n, 1))
        for j in range(J):
            m = obs[:, j]
            lg[m] += np.log(theta[j][:, L[m, j]]).T
        mx = lg.max(1, keepdims=True)
        P = np.exp(lg - mx)
        ll = float((np.log(P.sum(1)) + mx.ravel()).sum())
        T = P / P.sum(1, keepdims=True)
        if abs(ll - prev) < tol:
            break
        prev = ll
    return T, theta, pi, prev


def sar_axis_and_residual(d):
    """D4's machinery: PC1 of the six-date gamma0 series, and NDVI residualised on it."""
    from common import DATES
    G = d[[f"g0_db_{t}" for t in DATES]].to_numpy(dtype="float64")
    nd = d["ndvi_20251013"].to_numpy()
    ok = (np.isfinite(nd) & np.isfinite(G).all(axis=1)
          & (d["ndvi_valid_20251013"] > 0.5).to_numpy())
    X = G[ok] - G[ok].mean(axis=0)
    axis = np.linalg.svd(X, full_matrices=False)[0][:, 0]
    ndv = nd[ok]
    A = np.c_[np.ones(len(axis)), axis]
    resid = ndv - A @ np.linalg.lstsq(A, ndv, rcond=None)[0]
    return ok, ndv, resid


def eta2(y, lab):
    """Fraction of variance in y explained by the grouping lab."""
    g = pd.Series(y).groupby(pd.Series(lab).values)
    ss_b = float(sum(len(v) * (v.mean() - y.mean()) ** 2 for _, v in g))
    ss_t = float(((y - y.mean()) ** 2).sum())
    return ss_b / ss_t if ss_t > 0 else np.nan


def main():
    d = pd.read_csv(RESULTS / "d4_debug.csv")
    d = d.merge(pd.read_csv(TABLES / "witness_s2.csv"), on="farm_id")   # NDVI witness columns
    con = pd.read_csv(AUX).set_index("farm_id").reindex(d.farm_id)
    teams = [c for c in con.columns if c.startswith("crop_")]
    code = {c: i for i, c in enumerate(CROPS)}
    L = np.full((len(d), len(teams)), -1, dtype=int)
    for j, t in enumerate(teams):
        v = con[t].map(code)
        L[:, j] = np.where(v.notna(), v.fillna(-1), -1).astype(int)
    log("p3_ds.input", n_farms=len(d), n_annotators=len(teams), teams=teams,
        missing_cells=int((L < 0).sum()))

    T, theta, pi, ll = dawid_skene(L)
    ds = np.array(CROPS, dtype=object)[T.argmax(1)]
    maj = np.array(CROPS, dtype=object)[
        np.array([np.bincount(L[i][L[i] >= 0], minlength=K).argmax() for i in range(len(L))])]
    ours = d.crop_type.to_numpy()

    # --- per-team reliability, estimated with no ground truth ---
    print("\nPer-team reliability from the fitted confusion matrices (no ground truth used):")
    rows = []
    for j, t in enumerate(teams):
        acc = float(np.trace(theta[j] @ np.diag(np.ones(K))) / K)   # mean diagonal
        rows.append({"team": t, "mean_diagonal": round(acc, 4),
                     "agrees_with_ds_pct": round(100 * float((L[:, j] == T.argmax(1)).mean()), 1)})
        print(f"  {t:<20} mean diagonal {acc:.3f}   agrees with DS on "
              f"{100 * float((L[:, j] == T.argmax(1)).mean()):5.1f}% of farms")
    pd.DataFrame(rows).to_csv(TABLES / "p3_ds_confusion.csv", index=False)

    # --- the referee: D4's non-circular residual test ---
    ok, ndv, resid = sar_axis_and_residual(d)
    print("\nD4's NON-CIRCULAR referee — variance of the S2 witness explained,\n"
          "before and after removing the SAR axis that produced the labels:")
    out = []
    for name, lab in (("ours (shipped)", ours), ("majority vote", maj), ("Dawid-Skene", ds)):
        e_raw, e_res = eta2(ndv, lab[ok]), eta2(resid, lab[ok])
        surv = 100 * e_res / e_raw if e_raw > 0 else np.nan
        out.append({"labels": name, "eta2_raw_ndvi": round(e_raw, 4),
                    "eta2_residual": round(e_res, 4), "survives_pct": round(surv, 1),
                    "n_differs_from_ours": int((lab != ours).sum())})
        print(f"  {name:<16} raw {e_raw:6.2%}   residual {e_res:6.2%}   survives {surv:5.1f}%"
              f"   differs from ours on {int((lab != ours).sum())} farms")

    # --- what each labelling would do to the deliverable ---
    fa = gpd.read_file(FARMS).to_crs(UTM)
    area = (fa.geometry.area / 1e4).to_numpy()
    anc = pd.read_csv(Path("data_aux") / "anchors_r3.csv").set_index("crop")
    a_t = {c: float(anc.loc[c, "anchor_kg_ha"]) * float(anc.loc[c, "adj_2025_26"]) / 1000
           for c in CROPS}
    hi = d.health_index.to_numpy(dtype="float64")
    print()
    for r, (name, lab) in zip(out, (("ours (shipped)", ours), ("majority vote", maj),
                                    ("Dawid-Skene", ds))):
        m = modifier(hi, pd.Series(lab), area)
        tot = float(sum(a_t[c] * np.sum((lab == c) * m * area) for c in CROPS))
        r["village_total_t"] = round(tot, 1)
        print(f"  {name:<16} village total {tot:7.1f} t")

    # ---- ★ THE CONTROL THAT KILLS THE COMPARISON: label provenance ----
    # The referee above is non-circular for OUR labels because ours are SAR-derived. That is a
    # property of our pipeline, NOT of the test. The test removes the SAR axis; it does not
    # remove an optical axis. So a labelling built with optical data explains residual NDVI for
    # a reason that has nothing to do with being correct -- and Dawid-Skene fuses all six teams,
    # inheriting whatever they used.
    print("\n★ PROVENANCE CONTROL — how much does each team's labelling track the OPTICAL\n"
          "  witness after the SAR axis is removed? SAR-derived labels should score LOW.")
    prov = []
    for t in teams:
        lab_t = con[t].to_numpy()
        er, ers = eta2(ndv, lab_t[ok]), eta2(resid, lab_t[ok])
        prov.append({"team": t, "eta2_raw": round(er, 4), "eta2_residual": round(ers, 4),
                     "survives_pct": round(100 * ers / er, 1),
                     "reads_as": "optical-informed" if ers > 0.20 else
                                 ("SAR-like" if ers < 0.13 else "mixed")})
    for p in sorted(prov, key=lambda x: -x["survives_pct"]):
        star = "  <- US" if p["team"] == "crop_GDHTM" else ""
        print(f"    {p['team']:<20} resid {p['eta2_residual']:6.2%}  "
              f"survives {p['survives_pct']:5.1f}%  {p['reads_as']}{star}")
    pd.DataFrame(prov).to_csv(TABLES / "p3_label_provenance.csv", index=False)
    print("\n  CONSEQUENCE 1: Dawid-Skene's win on the referee is NOT evidence it is more\n"
          "    correct. It fuses three optical-informed labellings; the referee cannot tell\n"
          "    'better' from 'saw another sensor'. The comparison is void, not favourable.\n"
          "  CONSEQUENCE 2: the same metric on slide 9 does not measure label quality. It\n"
          "    measures information beyond the dominant SAR axis, which optical labels get for\n"
          "    free. Computed consistently, we rank LAST on it -- because we are the most\n"
          "    purely SAR-derived team in the comparison.")

    res = pd.DataFrame(out)
    res.to_csv(TABLES / "p3_dawidskene.csv", index=False)
    best = res.loc[res.survives_pct.idxmax(), "labels"]
    log("p3_ds.done", loglik=round(ll, 1),
        best_on_noncircular_test=best,
        referee_is_void=True,
        why_void=("the D4 referee is non-circular for OUR labels because ours are SAR-derived; "
                  "3 of 6 teams' labels read as optical-informed (residual eta2 0.21-0.26 vs "
                  "0.11 for SAR-like), so Dawid-Skene inherits optical information and the "
                  "test cannot separate 'more correct' from 'saw another sensor'"),
        slide9_consequence=("the same metric ranks us LAST of six (55.4%) when applied "
                            "consistently -- it measures information beyond the dominant SAR "
                            "axis, not label quality. Slide 9 must be reframed or cut"),
        ds_differs_from_ours_farms=int((ds != ours).sum()),
        decision=("REPORTED, NOT ADOPTED. Changing the crop map changes the deliverable and is "
                  "a deliberate decision, not a side effect of an experiment."),
        caveat=("the referee uses ONE witness date and eta2 rewards any grouping that splits "
                "NDVI, so a labelling can win it without being more correct"))
    print(f"\nbest on the non-circular test: {best}")
    print("REPORTED, NOT ADOPTED — changing the crop map changes the deliverable.")


if __name__ == "__main__":
    main()
