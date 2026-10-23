"""T1 (gate): can any X-band feature serve as a GAI / canopy proxy?

WHAT THIS DECIDES. The SAFY-family recommendation (Phase 2, R-C C-1) ingests a Green Area
Index time series. R-B B-1 establishes from the literature that X-band correlates POORLY
with LAI and biomass and saturates at very low biomass -- LAI belongs to C-band, biomass to
L-band. Driving a light-use-efficiency model with a proxy the physics says is weak is exactly
the failure mode the research phase was told to avoid. So SAFY is gated on this test, and if
the test fails, SAFY is dead and we report the negative.

KILL CRITERION, as written in the plan: if no feature clears |rho| = 0.5 WITHIN CROP on at
least three dates, SAFY is dead.

★ WITHIN CROP, and that is not a detail. A pooled correlation across all 966 farms would be
driven by between-crop differences -- and P1-5 established that Round 2's crop labels were
themselves derived from these backscatter trajectories, so a pooled correlation would be
partly correlating the SAR with itself. Within-crop correlation asks the only honest question:
among plots the pipeline calls the same crop, does the X-band feature rank canopy the way an
independent optical sensor ranks it?

WITNESS USE, and why this is legitimate. Sentinel-2 is a frozen witness. Using it to TEST
whether a feature works is validation, not promotion: no S2 value enters a shipped number,
and if T1 passes it is the X-band feature that gets used, not the NDVI. S2 stays unread by
the model. [prompt 5.1]

★ AVAILABILITY LIMIT, discovered at run time and reported rather than worked around. Only
THREE of our six Capella dates can be tested at all, because the monsoon eliminated usable
optical between 11 May and 13 October (Phase 0 census: July minimum cloud 92.6%). Of those,
two are SAME-DAY matches -- 13 Oct and 12 Nov -- and 29 Oct's nearest clear S2 is 23 October,
six days BEFORE it and, worse, before the 23-28 Oct rain that P1-3c showed changed the scene
by +4 dB. That pair is reported and then discounted, not quietly averaged in.

So the pre-registered "at least three dates" is NOT SATISFIABLE ON CLEAN PAIRS. That is a
fact about the data, not a result, and the honest response is to report the criterion as
unmeetable and judge on the two same-day pairs instead of silently relaxing it.

Writes results/tables/p3_t1_gai.csv.

Run:  py -3.12 src/p3_t1_gai.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUX, CROPS, DATES, TABLES, log

RHO_KILL = 0.5
MIN_N = 20                # a within-crop correlation on fewer plots is not worth reporting
FEATURES = ["g0_db", "b0_db", "cv", "ktex", "glcm_resid", "nesz_margin_db"]


def s2_dates(w):
    return sorted(set(re.findall(r"ndvi_(\d{8})", " ".join(w.columns))))


def main():
    f = pd.read_csv(TABLES / "p1_farm_features.csv")
    w = pd.read_csv(TABLES / "witness_s2.csv")
    lab = pd.read_csv(AUX / "inherited" / "label_dist.csv")[["farm_id", "crop_type"]]
    d = f.merge(w, on="farm_id").merge(lab, on="farm_id")
    assert len(d) == 966, len(d)

    avail = s2_dates(w)
    # pair each Capella date with its nearest clear S2 scene, and record the gap honestly
    pairs = []
    for c in DATES:
        ci = pd.Timestamp(c)
        gaps = {s: (pd.Timestamp(s) - ci).days for s in avail}
        best = min(gaps, key=lambda s: abs(gaps[s]))
        pairs.append((c, best, gaps[best]))
    log("p3_t1.pairs", pairs=[(a, b, g) for a, b, g in pairs],
        same_day=[a for a, b, g in pairs if g == 0])

    rows = []
    for cap, s2, gap in pairs:
        if abs(gap) > 10:
            log("p3_t1.skip", capella=cap, nearest_s2=s2, gap_days=gap,
                reason="no optical within 10 days -- monsoon cloud")
            continue
        nd = d[f"ndvi_{s2}"].to_numpy()
        valid = d[f"ndvi_valid_{s2}"].to_numpy()
        for feat in FEATURES:
            col = f"{feat}_{cap}"
            if col not in d:
                continue
            x = d[col].to_numpy()
            for crop in CROPS:
                m = ((d.crop_type == crop).to_numpy() & np.isfinite(x) & np.isfinite(nd)
                     & (valid > 0.5))
                if m.sum() < MIN_N:
                    continue
                rho, p = spearmanr(x[m], nd[m])
                rows.append({"capella": cap, "s2": s2, "gap_days": gap,
                             "same_day": gap == 0, "feature": feat, "crop": crop,
                             "n": int(m.sum()), "rho": round(float(rho), 4),
                             "p": float(p), "clears": bool(abs(rho) >= RHO_KILL)})

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "p3_t1_gai.csv", index=False)

    same = out[out.same_day]
    print(f"\nWITHIN-CROP |rho| vs same-day Sentinel-2 NDVI   (kill threshold {RHO_KILL})")
    piv = same.pivot_table(index="feature", columns=["capella", "crop"], values="rho")
    print(piv.round(2).to_string())

    print(f"\n{'feature':<16}{'n cells':>9}{'|rho| med':>11}{'|rho| max':>11}"
          f"{'cells clearing':>16}")
    for feat, g in same.groupby("feature"):
        a = g.rho.abs()
        print(f"{feat:<16}{len(g):>9}{a.median():>11.3f}{a.max():>11.3f}"
              f"{int(g.clears.sum()):>10} / {len(g)}")

    best = same.loc[same.rho.abs().idxmax()] if len(same) else None
    n_dates_clearing = same[same.clears].capella.nunique()
    verdict = ("PASS" if n_dates_clearing >= 3 else
               "FAIL -- criterion unmeetable: only "
               f"{same.capella.nunique()} same-day pairs exist (monsoon cloud Jun-Sep)")
    log("p3_t1.verdict", rho_kill=RHO_KILL,
        same_day_pairs=sorted(same.capella.unique()),
        dates_with_any_clearing=int(n_dates_clearing),
        best_feature=(None if best is None else best.feature),
        best_rho=(None if best is None else float(best.rho)),
        best_crop=(None if best is None else best.crop),
        n_cells_clearing=int(same.clears.sum()), n_cells=int(len(same)),
        verdict=verdict, safy="ALIVE" if n_dates_clearing >= 3 else "DEAD")


if __name__ == "__main__":
    main()
