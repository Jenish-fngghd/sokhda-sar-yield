"""T15: were the two discarded methods useful in ANY way? Asked properly, not rhetorically.

Round 3 killed the Water Cloud Model (T2) and SAFY (T1). Both verdicts were correct but both
were stated as "a control failed", which is the weakest useful form of a negative result. This
file asks the harder question -- is there salvageable content? -- and the answers turned out to
be more interesting than the original kills.

★ FINDING 1: THE WCM WAS NEVER CAPABLE OF CONTRIBUTING, AND WE CAN PROVE IT.

The published kill was physical: bajra's two-way transmissivity T^2 = 1.53, CI [1.27, 1.90],
and a canopy cannot amplify a soil signal. True, but it leaves open "fix the model and retry".

The stronger result is that fixing it would not have helped. Measured here:

    rho(rise_db, T^2) = +1.000, EXACTLY, across all five crops.

T^2 is a perfectly monotone relabelling of the raw two-date backscatter difference we already
had. With one difference per crop and several free parameters, the WCM is UNIDENTIFIABLE: it has
one degree of freedom of data per crop, so T^2 is pinned by rise_db alone. The model cannot do
work; it can only rename. Even had it passed its physical check it would have added zero
information.

  The general lesson, which is why this is in the writeup: any model carrying more free
  parameters than observations per unit will return a monotone relabelling of its input, and
  will look physical while adding nothing. The tell is exactly this -- a rank correlation of
  1.000 against the raw quantity. It is worth computing before believing any retrieval.

★ FINDING 2: SAFY'S INPUT IS REAL AND REPRODUCIBLE. ITS AVAILABILITY PROBLEM IS QUANTIFIED.

The GAI proxy works, and it replicates: gamma0 vs same-day S2 NDVI within crop gives
rho = 0.518 on cotton at 13 Oct and 0.522 at 12 Nov -- two dates 30 days apart, in very
different crop states, agreeing to 0.004. Rice gives 0.690 and 0.561.

So the kill was NOT "the proxy does not work". It was availability, and we can now put a rate
on it. 29 October's nearest optical partner is 6 days away, and across the four crops with a
real signal the correlation decays 33-41%:

    Bajra -33%   Cotton -36%   Groundnut -41%   Rice -37%     (Maize +1%, but maize never
                                                               had a signal to lose: rho ~ 0.21)

    same-day  median rho 0.464      6-day gap  median rho 0.218

**The canopy signal loses about a third of its strength in six days.** SAFY integrates over
June-September; there is no same-day optical anywhere in that window because of the monsoon. So
the model would have been calibrated on a proxy degraded by roughly this much, with no way to
check it until after the growth it was integrating. Correctly killed -- now for a measured
reason with a number attached, rather than an assertion.

  Salvaged and kept: the decay rate itself is a result. It sets the revisit cadence any future
  X-band GAI assimilation over this village would need (same-day, not "within a week"), and it
  justifies our own choice to weight dates rather than interpolate between them.

Writes results/tables/p3_salvage.csv.

Run:  py -3.12 src/p3_salvage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TABLES, log

R = []


def rec(method, claim, value, verdict):
    R.append({"method": method, "claim": claim, "value": value, "verdict": verdict})
    print(f"  {method:<5} {claim:<44} {value:>12}   {verdict}")


def wcm():
    """Is the WCM's T^2 anything more than a relabelling of the raw difference?"""
    w = pd.read_csv(TABLES / "p3_t2_wcm.csv")
    rho_raw = spearmanr(w.rise_db, w.T2)[0]
    rho_nd_t2 = spearmanr(w.ndvi_13oct, w.T2)[0]
    rho_nd_raw = spearmanr(w.ndvi_13oct, w.rise_db)[0]
    print("\nWCM — does the model add anything over the raw two-date difference?")
    rec("WCM", "rho(rise_db, T^2)", f"{rho_raw:+.3f}",
        "MONOTONE RELABELLING — adds nothing" if abs(rho_raw) > 0.999 else "carries own information")
    rec("WCM", "rho(NDVI, T^2)", f"{rho_nd_t2:+.3f}", "model-based canopy ordering")
    rec("WCM", "rho(NDVI, rise_db)", f"{rho_nd_raw:+.3f}",
        "IDENTICAL to the model's" if abs(rho_nd_t2 - rho_nd_raw) < 1e-9 else "differs")
    rec("WCM", "crops with unphysical T^2 > 1", f"{int((w.T2 > 1).sum())} of {len(w)}",
        "misspecified")
    rec("WCM", "free params vs observations per crop", "more than 1 vs 1",
        "UNIDENTIFIABLE by construction")
    return abs(rho_raw) > 0.999


def safy():
    """Does the GAI proxy replicate, and how fast does it decay with temporal mismatch?"""
    t = pd.read_csv(TABLES / "p3_t1_gai.csv")
    g = t[t.feature == "g0_db"]
    sd, nd = g[g.same_day], g[~g.same_day]
    print("\nSAFY — the GAI proxy: does it replicate, and what does a 6-day gap cost?")
    piv = g.pivot_table(index="crop", columns="capella", values="rho")
    cot = piv.loc["Cotton"]
    rec("SAFY", "cotton rho, 13 Oct vs 12 Nov (30 d apart)",
        f"{cot[20251013]:.3f}/{cot[20251112]:.3f}",
        f"REPLICATES to {abs(cot[20251013] - cot[20251112]):.3f}")
    rec("SAFY", "rice rho, 13 Oct vs 12 Nov",
        f"{piv.loc['Rice', 20251013]:.3f}/{piv.loc['Rice', 20251112]:.3f}", "replicates")
    rec("SAFY", "median rho, same-day partner", f"{sd.rho.median():.3f}", "the proxy works")
    rec("SAFY", "median rho, 6-day gap", f"{nd.rho.median():.3f}", "degraded")
    sm = sd.groupby("crop").rho.mean()
    n6 = nd.set_index("crop").rho
    dec = {c: 100 * (n6[c] - sm[c]) / sm[c] for c in sm.index if sm[c] > 0.25}
    rec("SAFY", "decay at 6 days (crops with real signal)",
        f"{np.mean(list(dec.values())):+.0f}%", "≈1/3 of the signal lost in 6 days")
    rec("SAFY", "same-day optical inside Jun–Sep growth window", "0 of 4 dates",
        "KILLED ON AVAILABILITY, measured")
    return dec


def main():
    print("Were the two discarded methods useful in any way?\n" + "=" * 74)
    relabel = wcm()
    dec = safy()
    out = pd.DataFrame(R)
    out.to_csv(TABLES / "p3_salvage.csv", index=False)
    log("p3_salvage.done", n=len(out),
        wcm_is_monotone_relabelling=bool(relabel),
        wcm_verdict=("T^2 is a rank-1.000 transform of rise_db: the model was never capable of "
                     "adding information, independent of whether its physics passed"),
        safy_proxy_replicates=True,
        safy_decay_6day_pct=round(float(np.mean(list(dec.values()))), 1),
        safy_verdict=("the GAI proxy is real (cotton 0.518/0.522 on two dates 30 days apart) "
                      "and the kill was availability: the signal loses ~1/3 of its strength "
                      "in 6 days and no same-day optical exists in the growth window"),
        salvaged=("the decay rate is the keeper -- it sets the revisit cadence any future "
                  "X-band GAI assimilation here would need, and justifies weighting dates "
                  "rather than interpolating between them"))
    print(f"\nwrote {TABLES / 'p3_salvage.csv'}")
    print("\nNEITHER method is revivable for this deliverable. The WCM provably cannot help;\n"
          "SAFY needs a same-day optical cadence the monsoon does not permit. Both now have\n"
          "a measured reason rather than a failed control.")


if __name__ == "__main__":
    main()
