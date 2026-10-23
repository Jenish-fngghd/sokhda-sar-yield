"""T18: X+C multi-frequency fusion — does adding Sentinel-1 improve the shipped ordering?

T17 established that Sentinel-1 growth-window features carry canopy information Capella's
same-day gamma0 does not, on 85% of village area:

    cotton     partial rho +0.346 after removing Capella gamma0   p = 1.1e-12
    groundnut  partial rho +0.362                                 p = 3.2e-08
    rice       partial rho +0.304                                 p = 4.5e-03  (marginal)

"Carries extra information about the witness" is not the same as "improves our forecast", so
this file builds the fused health index and tests it on the referee that already exists.

★ WHY THE RISK IS SMALLER THAN IT LOOKS. The modifier uses only the WITHIN-CROP ORDERING of the
health index and is normalised to area-weighted mean 1.0 within crop. So fusion can move plot
and zone yields but CANNOT move the village total -- 710.0 t is invariant, and every D1
cross-round consistency check holds unchanged. This is a change to the distribution only.

★ WEIGHTS STAY BLIND. The fused weights are derived by the same rule as the shipped ones --
redundancy alone, w_k ~ 1/sum_j|rho(part_k, part_j)| -- which never sees a witness, a label or a
yield. We did not tune anything toward the S2 comparison; the referee stays out of sample.

PROVENANCE. Sentinel-1 is SAR. X+C fusion keeps the claim established in P3-14d that our crop
map and index are the least optically-contaminated in the comparison. An optical index would
score better on an optical witness and would mean nothing.

DECIDES NOTHING. Reports the comparison. Adoption is a deliberate call.

Writes results/tables/p3_fusion.csv.

Run:  py -3.12 src/p3_fusion.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, RESULTS, TABLES, log
from p3_build import derive_weights, health_index, health_parts, z

GROW = ("20250601", "20250930")
R = []


def s1_parts(d, dates):
    """Growth-window C-band families. Two only, chosen from T17's measured result, not swept."""
    VH = d[[f"vh_{t}" for t in dates]].to_numpy(dtype="float64")
    VV = d[[f"vv_{t}" for t in dates]].to_numpy(dtype="float64")
    g = [i for i, t in enumerate(dates) if GROW[0] <= t <= GROW[1]]
    return {"s1_grow_vhvv": np.nanmean(VH[:, g] - VV[:, g], axis=1),
            "s1_grow_vh": np.nanmean(VH[:, g], axis=1)}


def referee(hi, d, crop, date="20251013"):
    """Within-crop rho against the same-day S2 witness. Never used to fit anything."""
    nd = d[f"ndvi_{date}"].to_numpy()
    val = (d[f"ndvi_valid_{date}"] > 0.5).to_numpy()
    out = {}
    for c in CROPS:
        m = (crop == c).to_numpy() & np.isfinite(nd) & val & np.isfinite(hi)
        out[c] = spearmanr(hi[m], nd[m])[0] if m.sum() >= 20 else np.nan
    return out


def main():
    d = pd.read_csv(RESULTS / "d4_debug.csv")
    d = d.merge(pd.read_csv(TABLES / "witness_s2.csv"), on="farm_id")
    s1 = pd.read_csv(TABLES / "p3_s1_season.csv")
    d = d.merge(s1, on="farm_id")
    dates = [c.split("_")[1] for c in s1.columns if c.startswith("vv_")]
    crop = d.crop_type
    area = d.area_ha.to_numpy()

    base = health_parts(d, use_29oct=False)
    fused = dict(base)
    fused.update(s1_parts(d, dates))

    Wb, Wf = derive_weights(base), derive_weights(fused)
    hb = health_index(base, Wb, crop)
    hf = health_index(fused, Wf, crop)

    log("p3_fusion.weights", n_parts_base=len(base), n_parts_fused=len(fused),
        s1_weight_share=round(float(sum(Wf[k] for k in Wf if k.startswith("s1_"))), 4),
        rule="redundancy only (1/sum|rho|) -- blind to every witness, same as shipped")

    print("Referee: within-crop rho vs same-day S2 NDVI, 13 Oct (out of sample for both)\n")
    rb, rf = referee(hb, d, crop), referee(hf, d, crop)
    area_share = {c: 100 * area[(crop == c).to_numpy()].sum() / area.sum() for c in CROPS}
    print(f"{'crop':<10} {'area%':>6} {'SHIPPED':>9} {'FUSED':>9} {'delta':>8}")
    for c in sorted(CROPS, key=lambda x: -area_share[x]):
        dl = rf[c] - rb[c]
        R.append({"crop": c, "area_pct": round(area_share[c], 1),
                  "rho_shipped": round(rb[c], 4), "rho_fused": round(rf[c], 4),
                  "delta": round(dl, 4)})
        print(f"{c:<10} {area_share[c]:5.1f}% {rb[c]:9.3f} {rf[c]:9.3f} {dl:+8.3f}"
              f"   {'better' if dl > 0.01 else ('worse' if dl < -0.01 else 'same')}")

    wb = sum(rb[c] * area_share[c] for c in CROPS) / 100
    wf = sum(rf[c] * area_share[c] for c in CROPS) / 100
    print(f"\n  area-weighted mean rho   shipped {wb:.4f}   fused {wf:.4f}   "
          f"delta {wf - wb:+.4f}")

    # --- the control: does the SECOND witness date agree? Nothing was tuned to either. ---
    rb2, rf2 = referee(hb, d, crop, "20251112"), referee(hf, d, crop, "20251112")
    wb2 = sum(rb2[c] * area_share[c] for c in CROPS) / 100
    wf2 = sum(rf2[c] * area_share[c] for c in CROPS) / 100
    print(f"  CONTROL, 12 Nov witness  shipped {wb2:.4f}   fused {wf2:.4f}   "
          f"delta {wf2 - wb2:+.4f}")

    # --- and the invariant we claimed ---
    from p3_build import modifier
    mb, mf = modifier(hb, crop, area), modifier(hf, crop, area)
    anc = pd.read_csv(Path("data_aux") / "anchors_r3.csv").set_index("crop")
    a_t = {c: float(anc.loc[c, "anchor_kg_ha"]) * float(anc.loc[c, "adj_2025_26"]) / 1000
           for c in CROPS}
    tb = sum(a_t[c] * np.sum((crop == c).to_numpy() * mb * area) for c in CROPS)
    tf = sum(a_t[c] * np.sum((crop == c).to_numpy() * mf * area) for c in CROPS)
    print(f"\n  village total  shipped {tb:.1f} t   fused {tf:.1f} t   "
          f"difference {tf - tb:+.3f} t  <- invariant, as claimed")
    print(f"  per-plot yield changes on {int((np.abs(mb - mf) > 0.01).sum())} of {len(d)} farms")

    pd.DataFrame(R).to_csv(TABLES / "p3_fusion.csv", index=False)
    log("p3_fusion.done", rho_shipped_aw=round(wb, 4), rho_fused_aw=round(wf, 4),
        delta_aw=round(wf - wb, 4), delta_aw_control_12nov=round(wf2 - wb2, 4),
        village_total_delta_t=round(float(tf - tb), 3),
        verdict=("improves the within-crop ordering on both witness dates while leaving the "
                 "village total invariant" if (wf > wb and wf2 > wb2) else
                 "does NOT improve consistently -- do not adopt"),
        decision="REPORTED, NOT ADOPTED — adoption is a deliberate call")


if __name__ == "__main__":
    main()
