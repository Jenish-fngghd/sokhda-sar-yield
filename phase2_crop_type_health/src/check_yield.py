"""Validate the yield deliverable -- the one output with no external reference.

WHY THIS EXISTS. The health index has two independent witnesses. Yield had none, and
the obvious check is worthless: comparing our village aggregate against the district
APY figure just returns the season-completion constant, because the APY figure is an
INPUT. That comparison confirms arithmetic, not agronomy.

WHAT CAN ACTUALLY BE TESTED. Yield = district anchor x completion(farm) x
accumulation(farm). The anchor is an input and is untestable here. The two per-farm
terms are ours, and they make falsifiable claims about the world:

  completion(farm)   claims to measure how far through its cycle a field is on 13 Oct
  accumulation(farm) claims to measure how much canopy was carried across the season

Both are checkable against sensors that never enter the model.

THIS TEST HAS ALREADY EARNED ITS KEEP. On first run it showed the completion term was
INVERTED: d_oct_aug correlates POSITIVELY with same-day NDVI in all five crops, so a
field that brightened between August and October has MORE standing biomass, not less.
The term was reading standing crop as senescence and the sign was flipped. No internal
consistency check would have caught it -- only a sensor that disagrees can.

Run:  python src/check_yield.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUX, CROPS, RESULTS, RESULTS_AUX, log

MIN_N = 20


def rho(a, b):
    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_N:
        return np.nan, np.nan, int(ok.sum())
    r = spearmanr(a[ok], b[ok])
    return float(r.statistic), float(r.pvalue), int(ok.sum())


def main():
    sub = pd.read_csv(RESULTS / "submission.csv")
    f = pd.read_csv(RESULTS / "farm_features.csv")
    w = pd.read_csv(RESULTS_AUX / "witness.csv")
    dbg = pd.read_csv(RESULTS / "d4_debug.csv")
    # farm_features and d4_debug both carry area_ha; take one explicitly rather than
    # letting the merge suffix them and then reading whichever pandas happened to pick
    m = sub.merge(f.drop(columns=["area_ha"], errors="ignore"), on="farm_id")
    m = m.merge(w, on="farm_id").merge(dbg[["farm_id", "area_ha"]], on="farm_id")

    print("=" * 74)
    print("0. WHY THE OBVIOUS CHECK IS CIRCULAR")
    print("=" * 74)
    apy = pd.read_csv(AUX / "vadodara_apy.csv").set_index("crop")
    for c in CROPS:
        g = m[m.crop_type == c]
        a = float(apy.loc[c, "yield_kg_ha_2022_23"]) / 1000.0
        print(f"  {c:11s} ours {g.yield_estimate_to_date.median():6.3f} t/ha  "
              f"APY {a:6.3f}  ratio {g.yield_estimate_to_date.median()/a:5.2f}")
    print("  The ratio simply returns the season-completion constant: the APY figure is")
    print("  an INPUT, so this confirms arithmetic, not agronomy. Reported, not claimed.")

    print()
    print("=" * 74)
    print("1. DOES THE COMPLETION TERM MEASURE WHAT IT CLAIMS?  (falsifiable)")
    print("=" * 74)
    print("  Claim: higher d_oct_aug = MORE standing crop on 13 Oct = LESS complete.")
    print("  Test:  within crop, correlate d_oct_aug against same-day S2 NDVI.")
    print("  A POSITIVE correlation supports the claim; negative would refute it.\n")
    ok_all = True
    for c in CROPS:
        g = m[m.crop_type == c]
        r, p, n = rho(g.d_oct_aug, g.s2_ndvi_20251013)
        verdict = "supports" if (np.isfinite(r) and r > 0) else "REFUTES"
        ok_all &= bool(np.isfinite(r) and r > 0)
        print(f"    {c:11s} rho {r:+.3f}  p {p:.1e}  n {n:4d}   {verdict}")
    print(f"\n  All five crops agree: {ok_all}")
    print("  This is what caught the original sign error -- the term was reading")
    print("  standing crop as senescence.")

    print()
    print("=" * 74)
    print("2. DOES THE ACCUMULATION TERM TRACK BIOMASS?")
    print("=" * 74)
    # The two single-date witnesses are the WRONG SHAPE for this term: season_integral
    # runs 12 Jun - 13 Oct and they are instants. The matched witness is a season
    # integral built the same way from a sensor that is not ours -- see witness_season.py.
    # Cumulative NDVI, the textbook choice, is impossible here: Sokhda had ZERO
    # Sentinel-2 scenes under 20% cloud in June, July, August or September, so the whole
    # accumulation period has no optical record. C-band saw it; optical did not.
    sp = RESULTS_AUX / "witness_season.csv"
    matched = pd.read_csv(sp) if sp.exists() else None
    if matched is not None:
        m = m.merge(matched, on="farm_id", how="left")
    print("  single-date witnesses (wrong shape for an integral), then the matched one:\n")
    hdr = f"    {'crop':11s} {'vs NDVI':>9} {'vs C-VH':>9}"
    print(hdr + (f" {'vs INT C-VH':>12} {'p':>9}" if matched is not None else ""))
    rows = []
    for c in CROPS:
        g = m[m.crop_type == c]
        r1, p1, n = rho(g.season_integral, g.s2_ndvi_20251013)
        r2, _, _ = rho(g.season_integral, g.s1_vh_db)
        line = f"    {c:11s} {r1:+9.3f} {r2:+9.3f}"
        if matched is not None:
            r3, p3, n3 = rho(g.season_integral, g.s1_vh_season_integral)
            line += f" {r3:+12.3f} {p3:9.1e}"
            rows.append((c, r3, p3, n3))
        print(line + f"   n {n:4d}")
    print("\n  Against the single-date witnesses this reads mixed, and honestly so: for a")
    print("  crop harvested before October both are looking at soil, so they cannot")
    print("  testify about what accumulated in July. The season-integrated C-band column")
    print("  is the like-for-like test -- integral against integral, same trapezoid, a")
    print("  different satellite, band, polarisation and geometry.")
    if rows:
        pos = sum(1 for _, r, _, _ in rows if np.isfinite(r) and r > 0)
        sig = sum(1 for _, r, p, _ in rows if np.isfinite(r) and r > 0 and p < 0.05)
        print(f"\n  Matched witness: positive in {pos}/{len(rows)} crops, "
              f"significant at p<0.05 in {sig}/{len(rows)}.")
        log("check_yield.matched_witness", positive=pos, significant=sig,
            **{c: round(float(r), 3) for c, r, _, _ in rows})

    print()
    print("=" * 74)
    print("3. IS THE YIELD FIELD STILL A RELABELLING OF HEALTH?")
    print("=" * 74)
    for c in CROPS:
        g = m[m.crop_type == c]
        r, _, n = rho(g.health_index, g.yield_estimate_to_date)
        print(f"    {c:11s} rho(health, yield) {r:+.3f}   n {n:4d}")
    print("  These were all exactly 1.000 before the yield rebuild. Anything below 1")
    print("  means the yield column carries information health does not.")

    print()
    print("=" * 74)
    print("4. INTERNAL SANITY")
    print("=" * 74)
    prod = (m.area_ha * m.yield_estimate_to_date).sum()
    print(f"  village production {prod:.0f} t over {m.area_ha.sum():.0f} ha")
    print(f"  yield range        {m.yield_estimate_to_date.min():.3f} - "
          f"{m.yield_estimate_to_date.max():.3f} t/ha")
    r, p, n = rho(m.yield_estimate_to_date, m.area_ha)
    print(f"  rho(yield, plot area) {r:+.3f} (p {p:.2f}) -- ideally ~0; a per-hectare")
    print("    yield that rises with plot size is partly a size artefact, not agronomy.")
    print("    It enters through season_integral (rho +0.247 with area). Residualising")
    print("    that on log(npix) -- the fix that worked for GLCM entropy -- was tried and")
    print("    made it WORSE (area rho 0.247 -> 0.908), because the linear-in-log model")
    print("    does not describe this variable. Left uncorrected and reported instead.")

    log("check_yield", completion_all_positive=bool(ok_all),
        rho_health_yield_cotton=round(rho(m[m.crop_type == "Cotton"].health_index,
                                          m[m.crop_type == "Cotton"].yield_estimate_to_date)[0], 3),
        rho_yield_area=round(r, 3))


if __name__ == "__main__":
    main()
