"""T8: the defensibility battery. Plausibility & Defensibility is 25 points.

Every threshold was fixed in internal/PLAN_R3.md BEFORE this ran, and every check states
what a failure would MEAN. Deciding a threshold after seeing the number turns a sanity check
into a rationalisation.

  D1  forecast vs Round 2's shipped yield-to-date, per crop.  ★ Only we can run this.
  D2  village production against the Round 2 cross-team spread.
  D3  per-crop yields inside published statistical ranges, cotton in LINT.
  D4  non-circular label test -- the residual method, because P1-5 proved every naive
      label test on this stack is circular.
  D5  cross-crop ordering against agronomic expectation.
  D8  degenerate-output assertions: a collapsed forecast must FAIL, not pass a schema.

Writes results/tables/p3_checks.csv.

Run:  py -3.12 src/p3_checks.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUX, CROPS, DATES, R2, RESULTS, TABLES, log

# D1: expected forecast / R2-to-date ratio. Cotton from the plan's ADD-2 reasoning
# (R2 completion 0.45 read as a determined-yield fraction); cereals were harvested before
# R2's last date so their forecast should merely reproduce it.
D1_EXPECT = {"Cotton": (1.5, 4.0), "Groundnut": (1.0, 1.6),
             "Rice": (0.8, 1.3), "Maize": (0.8, 1.3), "Bajra": (0.5, 1.3)}
# D2: five teams' Round 2 village totals over the same 447.5 ha. A sixth summed to
# 21,566 t (~48 t/ha), a unit error, and is excluded from any cross-team statistic.
D2_BAND = (578.0, 1268.0)
# D3: plausible kharif yield ranges, t/ha. Cotton is LINT, which is why it looks small.
D3_RANGE = {"Cotton": (0.3, 1.2), "Rice": (1.0, 3.5), "Maize": (1.2, 4.0),
            "Bajra": (0.8, 3.0), "Groundnut": (1.0, 3.2)}

R = []


def rec(check, name, ok, detail):
    R.append({"check": check, "item": name, "pass": bool(ok), "detail": detail})
    print(f"  {'PASS' if ok else 'FAIL'}  {check} {name:<12} {detail}")


def main():
    sub = pd.read_csv(RESULTS / "submission.csv")
    dbg = pd.read_csv(RESULTS / "d4_debug.csv")
    r2 = pd.read_csv(R2 / "results" / "submission.csv")
    assert list(sub.farm_id) == list(r2.farm_id), "farm_id ordering differs between rounds"

    area = dbg.area_ha.to_numpy()
    crop = sub.crop_type
    y = sub.yield_forecast_t_ha.to_numpy()

    # ---------------- D1 : forecast vs Round 2 yield-to-date ----------------
    print("\nD1  forecast vs Round 2 yield-to-date, per crop  "
          "(the check no other team can run)")
    same = (crop.to_numpy() == r2.crop_type.to_numpy())
    print(f"      crop labels unchanged on {same.sum()}/966 farms "
          f"({100*same.mean():.1f}%) -- ratios below are on those farms only")
    for c in CROPS:
        m = (crop == c).to_numpy() & same
        if m.sum() < 10:
            continue
        a = float(np.sum(y[m] * area[m]) / area[m].sum())
        b = float(np.sum(r2.yield_estimate_to_date.to_numpy()[m] * area[m]) / area[m].sum())
        ratio = a / b
        lo, hi = D1_EXPECT[c]
        rec("D1", c, lo <= ratio <= hi,
            f"ratio {ratio:.2f} (expected {lo}-{hi}); R3 {a:.3f} vs R2 {b:.3f} t/ha")
    tot_r3 = float(np.sum(y * area))
    tot_r2 = float(np.sum(r2.yield_estimate_to_date.to_numpy() * area))
    rec("D1", "village", tot_r3 >= tot_r2,
        f"final forecast {tot_r3:.1f} t must be >= yield-to-date {tot_r2:.1f} t "
        f"(ratio {tot_r3/tot_r2:.2f})")

    # ---------------- D2 : cross-team plausibility band ----------------
    print("\nD2  village production against the Round 2 cross-team spread")
    rec("D2", "band", D2_BAND[0] <= tot_r3 <= D2_BAND[1],
        f"{tot_r3:.1f} t within {D2_BAND[0]}-{D2_BAND[1]} t "
        f"(Megalodon 578, our R2 595, Orion 1002, Coding Bits 1268)")

    # ---------------- D3 : published yield ranges ----------------
    print("\nD3  per-crop yield inside published ranges (cotton in LINT)")
    for c in CROPS:
        m = (crop == c).to_numpy()
        aw = float(np.sum(y[m] * area[m]) / area[m].sum())
        lo, hi = D3_RANGE[c]
        rec("D3", c, lo <= aw <= hi, f"{aw:.3f} t/ha in [{lo}, {hi}]")

    # ---------------- D4 : the non-circular label test ----------------
    # P1-5 proved that contrasting label-classes on THIS backscatter stack is circular,
    # because Round 2's labels were derived from these trajectories. Orion's method is the
    # fix: regress the independent witness on the SAR ranking axis that produced the
    # labels, then run the group test on the RESIDUAL. If the labels carry nothing beyond
    # that axis, the residual test collapses -- which is what happened to their own Tier-2
    # labels (19% of raw NDVI variance, 0.2% of the residual).
    print("\nD4  non-circular label test -- do the labels survive removing the SAR axis?")
    w = pd.read_csv(TABLES / "witness_s2.csv")
    d = dbg.merge(w, on="farm_id")
    nd = d["ndvi_20251013"].to_numpy()
    G = d[[f"g0_db_{x}" for x in DATES]].to_numpy()
    ok = np.isfinite(nd) & np.isfinite(G).all(axis=1) & (d["ndvi_valid_20251013"] > 0.5).to_numpy()
    X = G[ok] - G[ok].mean(axis=0)
    axis = np.linalg.svd(X, full_matrices=False)[0][:, 0]        # PC1 = dominant SAR axis
    ndv, cr = nd[ok], d.crop_type.to_numpy()[ok]

    A = np.c_[np.ones(len(axis)), axis]
    resid = ndv - A @ np.linalg.lstsq(A, ndv, rcond=None)[0]

    def eta(vals):
        gm = vals.mean()
        ssb = sum((cr == c).sum() * (vals[cr == c].mean() - gm) ** 2 for c in CROPS
                  if (cr == c).sum() > 2)
        return float(ssb / np.sum((vals - gm) ** 2))

    e_raw, e_res = eta(ndv), eta(resid)
    h_raw = kruskal(*[ndv[cr == c] for c in CROPS if (cr == c).sum() > 2])
    h_res = kruskal(*[resid[cr == c] for c in CROPS if (cr == c).sum() > 2])
    survived = e_res / e_raw if e_raw > 0 else 0.0
    rec("D4", "residual", e_res > 0.02 and h_res.pvalue < 0.05,
        f"labels explain {100*e_raw:.1f}% of raw NDVI (p={h_raw.pvalue:.1e}) and "
        f"{100*e_res:.1f}% of the residual (p={h_res.pvalue:.1e}) -- "
        f"{100*survived:.0f}% survives removing the SAR axis")
    log("p3_checks.d4", eta_raw=round(e_raw, 4), eta_residual=round(e_res, 4),
        fraction_surviving=round(survived, 4),
        p_raw=float(h_raw.pvalue), p_residual=float(h_res.pvalue),
        orion_tier2_comparison={"raw": 0.19, "residual": 0.002})

    # ---------------- D5 : cross-crop ordering ----------------
    print("\nD5  cross-crop ordering against agronomy")
    td = dbg.yield_to_date_t_ha.to_numpy()
    order = {c: float(np.sum(td[(crop == c).to_numpy()] * area[(crop == c).to_numpy()])
                      / area[(crop == c).to_numpy()].sum()) for c in CROPS}
    rec("D5", "cotton_last", order["Cotton"] == min(order.values()),
        "cotton must have the LOWEST yield-to-date -- it is the only crop still picking: "
        + ", ".join(f"{k} {v:.2f}" for k, v in sorted(order.items(), key=lambda kv: kv[1])))

    # ---------------- D8 : degenerate output ----------------
    print("\nD8  degenerate-output assertions")
    rec("D8", "rows", len(sub) == 966 and sub.farm_id.is_unique, f"{len(sub)} unique rows")
    rec("D8", "not_constant", sub.yield_forecast_t_ha.nunique() > 20,
        f"{sub.yield_forecast_t_ha.nunique()} distinct yields")
    rec("D8", "crops", sub.crop_type.nunique() >= 3,
        f"{sub.crop_type.nunique()} crop classes")
    rec("D8", "units", sub.yield_forecast_t_ha.max() < 25.0,
        f"max {sub.yield_forecast_t_ha.max():.3f} t/ha")
    rec("D8", "health", sub.health_index.between(0, 100).all()
        and sub.health_index.std() > 1.0,
        f"health sd {sub.health_index.std():.1f} pts, range "
        f"{sub.health_index.min():.0f}-{sub.health_index.max():.0f}")

    out = pd.DataFrame(R)
    out.to_csv(TABLES / "p3_checks.csv", index=False)
    n_fail = int((~out["pass"]).sum())
    log("p3_checks.done", n=len(out), passed=int(out["pass"].sum()), failed=n_fail,
        failing=[f"{r.check}:{r.item}" for r in out.itertuples() if not r._3])
    print(f"\n{int(out['pass'].sum())} passed, {n_fail} failed")


if __name__ == "__main__":
    main()
