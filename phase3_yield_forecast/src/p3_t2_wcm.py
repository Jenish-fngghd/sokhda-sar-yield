"""T2 (gate): re-test the Water Cloud Model, whose rejection premise has changed.

WHY RE-OPEN IT. Round 2 rejected the WCM inversion as a health component for a measured
reason: the observed August/June ratio sat below the model's floor at any monsoon-plausible
soil moisture, and the discrepancy was the same size as our own calibration uncertainty. The
calibration is now fixed and confirmed three independent ways (P1-2, P1-8), so the second half
of that reason is gone.

★ AND PHASE 1 HANDED THE WCM ITS OWN CENTRAL PREDICTION, MEASURED. The WCM writes

      sigma0_total = A*V1*cos(th)*(1 - T2)  +  T2 * sigma0_soil,
      T2 = exp(-2*B*V2 / cos(th))                      two-way canopy transmissivity, 0 < T2 <= 1

Between two dates close enough that the canopy has not changed much, the vegetation term
cancels and what survives is

      delta_sigma0_observed  ~=  T2 * delta_sigma0_soil

i.e. **a soil-moisture change reaches the sensor ATTENUATED BY THE CANOPY ABOVE IT**. P1-6b
measured exactly that on 13 Oct -> 29 Oct, when 81 mm of rain (a documented state-wide event,
Phase 2 A-2) moved ERA5 soil moisture 0.133 -> 0.372: every crop rose, and the rise was
ORDERED BY CANOPY COVER -- cotton +0.49 dB (most canopy), rice +1.07, groundnut +1.20,
maize +2.05, bajra +2.60 (bare). A 2.1 dB spread ordered by canopy is the WCM's own prediction
observed directly, on our own six dates, with a bare-soil control already present.

THE TEST. Fit T2 per crop as the ratio of that crop's observed rise to the rise over an
INDEPENDENTLY DEFINED bare reference, then check the model's own constraints.

★ THE BARE REFERENCE IS DEFINED BY SENTINEL-2, NOT BY US. Using the cereals as the bare
reference would make the cereal control true by construction -- the mistake P1-5 caught. So
bare ground is the lowest-NDVI decile on the 13 October S2 scene (same-day with the Capella
acquisition, 0-day gap), which is an optical instrument that knows nothing about our labels or
our backscatter.

CONTROLS, both fixed before the run, both able to fail:

  C1  PHYSICALITY.  T2 <= 1 for every crop, within noise. T2 > 1 means the canopy AMPLIFIED
      the soil signal, which the model forbids. If crops come back above 1, the WCM reading
      of P1-6b is wrong and this is coincidence.
  C2  ORDERING.     The fitted T2 must order the crops the same way an independent canopy
      proxy does -- more canopy, less transmission. The proxy is 13 Oct S2 NDVI, again not
      ours. Spearman(T2, NDVI) must be NEGATIVE and significant.

  If C1 or C2 fails, the WCM component is discarded and said to be discarded.

Unlike the groundnut-lift test (P1-5), this is NOT circular: it tests a coefficient against a
prediction the model fixes in advance, and both the bare reference and the ordering proxy come
from a sensor that played no part in producing the labels or the features.

Writes results/tables/p3_t2_wcm.csv.

Run:  py -3.12 src/p3_t2_wcm.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUX, CROPS, TABLES, log

DRY, WET = "20251013", "20251029"        # the pair the rain separates
S2_REF = "20251013"                      # same-day with DRY, 0-day gap
SM = {"20251013": 0.133, "20251029": 0.372}     # ERA5-Land at the overpass hour
BARE_DECILE = 0.10
N_BOOT = 2000
RNG = np.random.default_rng(20260830)


def main():
    f = pd.read_csv(TABLES / "p1_farm_features.csv")
    w = pd.read_csv(TABLES / "witness_s2.csv")
    lab = pd.read_csv(AUX / "inherited" / "label_dist.csv")[["farm_id", "crop_type"]]
    d = f.merge(w, on="farm_id").merge(lab, on="farm_id")

    rise = (d[f"g0_db_{WET}"] - d[f"g0_db_{DRY}"]).to_numpy()
    ndvi = d[f"ndvi_{S2_REF}"].to_numpy()
    ok = np.isfinite(rise) & np.isfinite(ndvi) & (d[f"ndvi_valid_{S2_REF}"] > 0.5).to_numpy()

    # ---- bare reference, defined by Sentinel-2 alone ----
    thr = np.nanquantile(ndvi[ok], BARE_DECILE)
    bare = ok & (ndvi <= thr)
    rise_bare = float(np.median(rise[bare]))
    log("p3_t2.bare_reference", source="Sentinel-2 NDVI 13 Oct, lowest decile",
        ndvi_threshold=round(float(thr), 4), n_plots=int(bare.sum()),
        rise_db=round(rise_bare, 3),
        soil_moisture_step=[SM[DRY], SM[WET]])
    assert rise_bare > 0.5, f"bare ground rose only {rise_bare:.2f} dB -- no soil signal to attenuate"

    rows = []
    for c in CROPS:
        m = ok & (d.crop_type == c).to_numpy()
        if m.sum() < 20:
            continue
        r = float(np.median(rise[m]))
        t2 = r / rise_bare
        # bootstrap the ratio: both numerator and denominator are medians of noisy samples
        bs = np.empty(N_BOOT)
        idx_c, idx_b = np.where(m)[0], np.where(bare)[0]
        for i in range(N_BOOT):
            bs[i] = (np.median(rise[RNG.choice(idx_c, len(idx_c))])
                     / np.median(rise[RNG.choice(idx_b, len(idx_b))]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"crop": c, "n": int(m.sum()),
                     "rise_db": round(r, 3), "T2": round(t2, 3),
                     "T2_lo": round(float(lo), 3), "T2_hi": round(float(hi), 3),
                     "ndvi_13oct": round(float(np.median(ndvi[m])), 4),
                     "physical": bool(lo <= 1.0)})

    out = pd.DataFrame(rows).sort_values("T2")
    out.to_csv(TABLES / "p3_t2_wcm.csv", index=False)

    print(f"\nbare reference (S2 lowest NDVI decile, n={int(bare.sum())}): "
          f"rise {rise_bare:+.2f} dB over the rain event\n")
    print(f"{'crop':<11}{'n':>5}{'rise dB':>9}{'T2':>7}{'95% CI':>16}"
          f"{'NDVI 13Oct':>12}{'T2<=1':>8}")
    for _, r in out.iterrows():
        print(f"{r.crop:<11}{r.n:>5}{r.rise_db:>+9.2f}{r.T2:>7.2f}"
              f"{f'[{r.T2_lo:.2f}, {r.T2_hi:.2f}]':>16}{r.ndvi_13oct:>12.3f}"
              f"{'yes' if r.physical else 'NO':>8}")

    # ---- C1 physicality ----
    c1 = bool(out.physical.all())
    # ---- C2 ordering against an independent canopy proxy ----
    rho, p = spearmanr(out.T2, out.ndvi_13oct)
    c2 = bool(rho < 0 and p < 0.10)

    log("p3_t2.controls",
        C1_physicality="PASS" if c1 else "FAIL",
        C1_detail={r.crop: [r.T2_lo, r.T2_hi] for r in out.itertuples()},
        C2_ordering="PASS" if c2 else "FAIL",
        C2_rho=round(float(rho), 3), C2_p=round(float(p), 4),
        verdict=("WCM SUPPORTED" if (c1 and c2) else "WCM DISCARDED -- a control failed"))


if __name__ == "__main__":
    main()
