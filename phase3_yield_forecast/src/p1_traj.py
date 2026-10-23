"""Phase 1.2: the six-date trajectory per crop, the groundnut-lift test, identifiability.

★ PREDICTIONS, WRITTEN BEFORE THE PLOT WAS LOOKED AT. Round 2's single most valuable prose
correction came from plotting a trajectory and finding the written physical story wrong
while the feature was right, so the order matters: predict, then plot, then report which
held. Drawn from the §4 crop calendar.

  PRED-A  Cotton separates FURTHER from the three cereals across 13 Oct -> 12 Nov. The
          cereals are off-field by mid-October and go to bare soil; cotton is still
          standing and picking. The cotton-minus-cereal gap should therefore grow.
  PRED-B  Groundnut shows a step across 29 Oct -> 12 Nov larger than the cereals show in
          the same fortnight. Lifting replaces standing canopy with disturbed bare soil.
  PRED-C  Rice, maize and bajra are mutually indistinguishable after 13 Oct. All three are
          bare soil by then, and bare soil does not carry a crop identity.
  PRED-D  EVERY crop rises on 29 October, by a similar amount, because P1-3 measured a
          +4 dB scene-wide wetness term there. This one is a control on the other three:
          it is crop-independent by construction, so if it comes out crop-dependent the
          referencing is wrong before any biology is discussed.

THE GROUNDNUT-LIFT TEST (§4, consequence 4). 29 Oct and 12 Nov sit either side of the
lift for most plots. A field that is genuinely groundnut should show a lifting signature
in that fortnight; a field mislabelled as groundnut should not. Round 2 could not run this
because its season ended before its data did.

  statistic   mean d(12 Nov - 29 Oct) over groundnut, minus the same over the three
              cereals pooled. Referencing to the cereals removes the scene-wide wetness
              term that PRED-D says is there, which is the whole reason the cereals are
              the right control rather than a village mean.
  null        the crop labels permuted between groundnut and cereal farms, 10 000 times.
  CONTROL     the same statistic computed WITHIN the cereals (rice against maize+bajra).
              All three were off-field before 13 October, so a fortnight signature there
              is measuring dew, look side, harvest traffic or noise -- not lifting.
              **If the control fires, the test is discarded and said to be discarded.**
  CONTROL 2   coverage. 29 October's mirrored swath drops 134 farms. If the drop is
              crop-biased, every number in this file is biased with it, so it is tested
              before anything else is read.

The labels are Round 2's own five-class assignment, which is an estimate and not truth --
they are the hypothesis under test here, not the ground for it.

Writes results/tables/p1_trajectory.csv, p1_groundnut_lift.csv, p1_identifiability.csv
and results/figures/p1_trajectory.png.

Run:  py -3.12 src/p1_traj.py
"""
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUX, CROPS, DATES, FIGURES, TABLES, log

CEREALS = ["Rice", "Maize", "Bajra"]
N_PERM = 10000
RNG = np.random.default_rng(20260830)


def doy(d):
    return _date(int(d[:4]), int(d[4:6]), int(d[6:8])).timetuple().tm_yday


def load():
    f = pd.read_csv(TABLES / "p1_farm_features.csv")
    lab = pd.read_csv(AUX / "inherited" / "label_dist.csv")[["farm_id", "crop_type",
                                                             "p_assigned", "entropy"]]
    d = f.merge(lab, on="farm_id", how="left")
    assert len(d) == len(f) == 966, (len(d), len(f))
    return d


def coverage_control(d):
    """Is 29 October's swath loss crop-biased?  Chi-square on present/absent by crop."""
    from scipy.stats import chi2_contingency
    rows = []
    for date in DATES:
        present = d[f"npix_{date}"] > 0
        tab = pd.crosstab(d.crop_type, present)
        if tab.shape[1] < 2:
            rows.append({"date": date, "chi2": 0.0, "p": 1.0, "missing": 0})
            continue
        chi2, p, _, _ = chi2_contingency(tab)
        rows.append({"date": date, "chi2": round(float(chi2), 2), "p": float(p),
                     "missing": int((~present).sum()),
                     **{f"miss_{c}": int(((~present) & (d.crop_type == c)).sum())
                        for c in CROPS}})
    out = pd.DataFrame(rows)
    log("p1_traj.coverage_control",
        worst_p=float(out.p.min()),
        biased_dates=[r.date for r in out.itertuples() if r.p < 0.05])
    return out


def trajectory(d):
    """Area-weighted mean gamma0 dB per crop per date, with dispersion.

    Area-weighted because the deliverable aggregates as sum(yield x area): a village
    statistic that weights a 0.05 ha plot equally with a 2 ha one is not the statistic the
    submission is built from.
    """
    rows = []
    for c in CROPS:
        g = d[d.crop_type == c]
        for date in DATES:
            v = g[f"g0_db_{date}"].to_numpy()
            w = g["area_ha"].to_numpy()
            ok = np.isfinite(v) & np.isfinite(w)
            if ok.sum() < 3:
                continue
            m = np.average(v[ok], weights=w[ok])
            rows.append({"crop": c, "date": date, "doy": doy(date), "n": int(ok.sum()),
                         "area_ha": round(float(w[ok].sum()), 2),
                         "mean_db": round(float(m), 3),
                         "wsd_db": round(float(np.sqrt(np.average((v[ok] - m) ** 2,
                                                                  weights=w[ok]))), 3),
                         "p25": round(float(np.percentile(v[ok], 25)), 3),
                         "p75": round(float(np.percentile(v[ok], 75)), 3)})
    return pd.DataFrame(rows)


def perm_test(a, b, n=N_PERM):
    """Two-sample mean difference, permutation null. Returns (diff, p_two_sided)."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    null = np.empty(n)
    for i in range(n):
        p = RNG.permutation(pool)
        null[i] = p[:na].mean() - p[na:].mean()
    return float(obs), float((np.abs(null) >= abs(obs)).mean()), float(null.std(ddof=1))


def groundnut_lift(d):
    col = "d_nov_oct29"
    ok = d[col].notna()
    rows = []

    gn = d.loc[ok & (d.crop_type == "Groundnut"), col].to_numpy()
    ce = d.loc[ok & d.crop_type.isin(CEREALS), col].to_numpy()
    ct = d.loc[ok & (d.crop_type == "Cotton"), col].to_numpy()

    obs, p, sd = perm_test(gn, ce)
    rows.append({"test": "PRIMARY groundnut vs cereals", "n_a": len(gn), "n_b": len(ce),
                 "mean_a": round(gn.mean(), 3), "mean_b": round(ce.mean(), 3),
                 "diff_db": round(obs, 3), "null_sd": round(sd, 3), "p": p,
                 "effect_size": round(obs / sd, 2) if sd else np.nan})

    # CONTROL: rice against maize+bajra. All three off-field before 13 Oct, so any
    # fortnight signature here is dew, look side, harvest traffic or noise.
    r = d.loc[ok & (d.crop_type == "Rice"), col].to_numpy()
    mb = d.loc[ok & d.crop_type.isin(["Maize", "Bajra"]), col].to_numpy()
    obs_c, p_c, sd_c = perm_test(r, mb)
    rows.append({"test": "CONTROL rice vs maize+bajra", "n_a": len(r), "n_b": len(mb),
                 "mean_a": round(r.mean(), 3), "mean_b": round(mb.mean(), 3),
                 "diff_db": round(obs_c, 3), "null_sd": round(sd_c, 3), "p": p_c,
                 "effect_size": round(obs_c / sd_c, 2) if sd_c else np.nan})

    # cotton is still standing and still picking: it should NOT show a lift step
    obs_t, p_t, sd_t = perm_test(ct, ce)
    rows.append({"test": "REFERENCE cotton vs cereals", "n_a": len(ct), "n_b": len(ce),
                 "mean_a": round(ct.mean(), 3), "mean_b": round(ce.mean(), 3),
                 "diff_db": round(obs_t, 3), "null_sd": round(sd_t, 3), "p": p_t,
                 "effect_size": round(obs_t / sd_t, 2) if sd_t else np.nan})

    out = pd.DataFrame(rows)
    ctrl_fired = bool(out.loc[1, "p"] < 0.05)
    log("p1_traj.groundnut_lift", primary_diff_db=round(obs, 3), primary_p=p,
        control_diff_db=round(obs_c, 3), control_p=p_c,
        control_fired=ctrl_fired,
        verdict=("DISCARDED -- control fired" if ctrl_fired
                 else "SIGNATURE" if p < 0.05 else "NO SIGNATURE"))
    return out


def identifiability(d):
    """What does n=6 buy over n=4?  Degrees of freedom, and the empirical rank.

    The df count is arithmetic. The rank is not: it asks how many independent directions
    the observed six-point trajectories actually span, which is the number a curve fit can
    hope to recover regardless of how many parameters it declares.
    """
    D = d[[f"g0_db_{x}" for x in DATES]].to_numpy()
    D = D[np.isfinite(D).all(axis=1)]
    Z = D - D.mean(axis=0)
    ev = np.linalg.svd(Z, compute_uv=False) ** 2
    ev = ev / ev.sum()

    models = [("linear", 2), ("quadratic", 3), ("asymmetric Gaussian", 4),
              ("logistic (single)", 4), ("double logistic", 6), ("SAFY (min free set)", 4)]
    rows = [{"model": m, "params": k, "df_at_n4": 4 - k, "df_at_n6": 6 - k,
             "identifiable_at_n6": bool(6 - k > 0)} for m, k in models]
    out = pd.DataFrame(rows)
    log("p1_traj.identifiability", n_farms=len(D),
        var_explained_pc=[round(float(v), 4) for v in ev],
        cum_2pc=round(float(ev[:2].sum()), 4), cum_3pc=round(float(ev[:3].sum()), 4),
        newly_identifiable=[m for m, k in models if 4 - k <= 0 < 6 - k])
    out.attrs["ev"] = ev
    return out, ev


def main():
    d = load()

    cc = coverage_control(d)
    print("\nCOVERAGE CONTROL -- is the swath loss crop-biased?")
    print(cc[["date", "missing", "chi2", "p"] + [f"miss_{c}" for c in CROPS]].to_string(index=False))

    tr = trajectory(d)
    tr.to_csv(TABLES / "p1_trajectory.csv", index=False)
    piv = tr.pivot(index="date", columns="crop", values="mean_db")[CROPS]
    print("\nAREA-WEIGHTED MEAN gamma0 dB PER CROP")
    print(piv.round(2).to_string())
    print("\n  cotton minus cereal mean:")
    cer = piv[CEREALS].mean(axis=1)
    print("   " + "  ".join(f"{x[4:]}:{piv.Cotton[x] - cer[x]:+.2f}" for x in piv.index))
    print("\n  change from previous date, per crop:")
    print(piv.diff().round(2).to_string())

    gl = groundnut_lift(d)
    gl.to_csv(TABLES / "p1_groundnut_lift.csv", index=False)
    print("\nGROUNDNUT-LIFT TEST  (d = gamma0 12 Nov minus 29 Oct, dB)")
    print(gl.to_string(index=False))

    idn, ev = identifiability(d)
    idn.to_csv(TABLES / "p1_identifiability.csv", index=False)
    print("\nIDENTIFIABILITY AT n=6")
    print("  variance explained by trajectory PCs: "
          + "  ".join(f"{v:.3f}" for v in ev))
    print(idn.to_string(index=False))

    _figure(tr, piv, gl, ev)


def _figure(tr, piv, gl, ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    colors = {"Rice": "#2b6cb0", "Cotton": "#c53030", "Maize": "#d69e2e",
              "Bajra": "#805ad5", "Groundnut": "#2f855a"}

    ax = axes[0]
    for c in CROPS:
        g = tr[tr.crop == c]
        ax.errorbar(g.doy, g.mean_db, yerr=g.wsd_db / np.sqrt(g.n), fmt="o-",
                    color=colors[c], capsize=3, label=f"{c} ({g.area_ha.iloc[0]:.0f} ha)")
    for x, t in ((doy("20251029"), "29 Oct\nwet"), (doy("20251112"), "12 Nov")):
        ax.axvline(x, color="#aaa", ls=":", lw=1)
        ax.text(x, ax.get_ylim()[1], t, fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("day of year 2025"); ax.set_ylabel(r"area-weighted $\gamma^0$ (dB)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("Six-date trajectory per crop\n(bars = weighted SE of the mean)", fontsize=10)

    ax = axes[1]
    step = piv.loc["20251112"] - piv.loc["20251029"]
    ax.bar(range(len(CROPS)), [step[c] for c in CROPS],
           color=[colors[c] for c in CROPS], edgecolor="#333")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(CROPS))); ax.set_xticklabels(CROPS, rotation=20, fontsize=8)
    ax.set_ylabel("dB change, 29 Oct -> 12 Nov")
    ax.set_title("The lift fortnight\n(cereals are the control: they are already bare)",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.bar(np.arange(1, len(ev) + 1), np.cumsum(ev), color="#4a5568", edgecolor="#333")
    ax.axhline(0.95, color="#c53030", ls="--", lw=1, label="95%")
    ax.set_xlabel("trajectory principal components")
    ax.set_ylabel("cumulative variance explained")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("How many directions do 6 dates actually span?", fontsize=10)

    fig.tight_layout()
    p = FIGURES / "p1_trajectory.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    log("p1_traj.fig", name=p.name)


if __name__ == "__main__":
    main()
