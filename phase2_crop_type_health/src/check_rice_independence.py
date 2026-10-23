"""Leave-channel-out test for the rice channel -- replacing a circular check.

The claim previously made was: "61% of flood-detected farms come back labelled
Rice against a 9% base rate -- 6.8x enrichment". That is a tautology. `flood_hit`
carries weight 2.2 in e["Rice"], the largest weight anywhere in the evidence
function, so farms are labelled Rice BECAUSE they flood. The check cannot fail.

A test that CAN fail: strip the flood channel out of the evidence entirely,
re-assign crops on what remains, and ask whether the flood-detected farms still
come back Rice on the strength of the other evidence alone.

  - If they do, the remaining channels independently corroborate the rice call.
  - If they do not, rice rests on a single channel. That is worth knowing and
    worth reporting; it is not a reason to hide the test.

Second, genuinely external test: rice paddies are hydrologically distinctive, so
we check the rice labels against a wetness signal that never enters the model at
all (Sentinel-1 C-VH, a different sensor and frequency).

Run:  python src/check_rice_independence.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d4_submission as D4
from common import AUX, CROPS, RESULTS, RESULTS_AUX, log


def main():
    f = pd.read_csv(RESULTS / "farm_features.csv")
    sub = pd.read_csv(RESULTS / "submission.csv")
    truth = pd.read_csv(AUX / "sokhda_phase1_crop_shares.csv").set_index("crop")
    prior = {c: float(truth.loc[c, "sokhda_share"]) for c in CROPS}

    shipped = sub.set_index("farm_id").loc[f.farm_id, "crop_type"].values
    flood = (f["g0_db_20250619"] - f["g0_db_20250606"]).values
    hit = np.isfinite(flood) & (flood >= D4.FLOOD_LO)

    print("=" * 74)
    print("THE OLD CHECK (circular -- reported here only to show why it fails)")
    print("=" * 74)
    print(f"  flood-detected farms      {hit.sum()}")
    print(f"  of those labelled Rice    {100*(shipped[hit]=='Rice').mean():.1f}%")
    print(f"  base rate Rice            {100*(shipped=='Rice').mean():.1f}%")
    print("  -> flood_hit has weight 2.2 in e[Rice]; this cannot come out any other way.")

    # ---- leave-channel-out --------------------------------------------------
    print()
    print("=" * 74)
    print("LEAVE-CHANNEL-OUT: re-assign crops with the flood channel REMOVED")
    print("=" * 74)
    orig = D4.crop_evidence

    def no_flood(fr):
        e, fl = orig(fr)
        # rebuild rice without its flood terms, and remove the flood penalty the
        # other four classes carry, so no class keeps a flood-derived advantage
        zf = D4.z(fl)
        fh = np.clip((fl - D4.FLOOD_LO) / (D4.FLOOD_HI - D4.FLOOD_LO), 0, 1)
        fh = np.where(np.isfinite(fl), fh, 0.0)
        e = dict(e)
        e["Rice"] = e["Rice"] - 2.2 * fh - 0.4 * zf
        for c, w in [("Cotton", 0.6), ("Groundnut", 0.6), ("Maize", 0.4), ("Bajra", 0.4)]:
            e[c] = e[c] + w * fh
        return e, fl

    # Deliberately NOT routed through assign_crop. With the flood channel gone the
    # prior fit degenerates -- it drives every class bias to ~422 and still places
    # only 1 farm as rice, because the argmax-matching loop pushes all five classes
    # up together when none of them can reach its target. That is a real fragility
    # in fit_prior under a stripped evidence set, and it is reported below, but it
    # would also make an enrichment RATIO meaningless (a ratio against a base rate
    # of one farm measures nothing). So the question is asked directly of the
    # evidence instead: with flood removed, do the flooded farms still RANK higher
    # on rice evidence than everyone else?
    e_nf, _ = no_flood(f)
    D4.crop_evidence = orig
    rice_ev = e_nf["Rice"]
    others = np.column_stack([e_nf[c] for c in CROPS if c != "Rice"]).max(axis=1)
    margin = rice_ev - others          # >0 means rice wins on residual evidence alone

    ok = np.isfinite(margin)
    m_hit, m_oth = margin[hit & ok], margin[~hit & ok]
    U, p_m = mannwhitneyu(m_hit, m_oth, alternative="greater")
    print(f"  residual rice-evidence margin, flooded farms   median {np.median(m_hit):+.3f}")
    print(f"  residual rice-evidence margin, other farms     median {np.median(m_oth):+.3f}")
    print(f"  Mann-Whitney (flooded > others), p = {p_m:.2e}")
    print(f"  farms where rice wins on residual evidence alone: {(margin > 0).sum()}")

    corroborated = (p_m < 0.05) and (np.median(m_hit) > np.median(m_oth))
    verdict = ("CORROBORATED -- with the flood channel removed the remaining evidence "
               "still ranks the flooded farms higher for rice"
               if corroborated else
               "NOT CORROBORATED -- rice rests on the flood channel alone. Reported as "
               "a single-channel dependency, which is the honest reading.")
    print(f"\n  VERDICT: {verdict}")
    print("\n  NOTE (separate finding): with flood stripped, fit_prior fails to place "
          "rice at all\n  (class biases saturate ~422, 1 farm assigned). The prior fit "
          "needs at least one\n  discriminating channel per class to converge -- worth "
          "stating as a limitation.")

    # ---- external sensor test ----------------------------------------------
    print()
    print("=" * 74)
    print("EXTERNAL TEST: rice labels vs Sentinel-1 C-VH (never an input)")
    print("=" * 74)
    w = pd.read_csv(RESULTS_AUX / "witness.csv")
    # reindex onto f's row order rather than relying on an inner join to preserve
    # length: `shipped` and `hit` are in f's order, so a witness fetch that ever drops
    # a farm would silently pair the wrong rows (or raise on the boolean mask)
    vh = (w.set_index("farm_id")["s1_vh_db"]
            .reindex(f.farm_id.values).values)
    isrice = shipped == "Rice"
    a, b = vh[isrice & np.isfinite(vh)], vh[~isrice & np.isfinite(vh)]
    U, p = mannwhitneyu(a, b)
    print(f"  rice   median VH {np.median(a):+.2f} dB  (n={len(a)})")
    print(f"  others median VH {np.median(b):+.2f} dB  (n={len(b)})")
    print(f"  Mann-Whitney p = {p:.2e}")
    print("  Harvested/drained paddy by mid-October should read LOW on C-VH: "
          f"{'consistent' if np.median(a) < np.median(b) else 'NOT consistent'}.")

    log("rice.independence", margin_p=f"{p_m:.2e}",
        margin_med_flooded=round(float(np.median(m_hit)), 3),
        margin_med_other=round(float(np.median(m_oth)), 3),
        vh_rice_med=round(float(np.median(a)), 2),
        vh_other_med=round(float(np.median(b)), 2), vh_p=f"{p:.2e}")


if __name__ == "__main__":
    main()
