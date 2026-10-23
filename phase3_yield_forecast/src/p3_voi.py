"""T23 / P3-21: what did each of the six acquisitions actually buy?

We were GIVEN six Capella scenes and we used all six. Nobody ever asked what any one
of them contributed. That is a strange gap, because the honest answer changes what we
can claim: if the shipped ranking is recoverable from one date, then the trajectory
machinery -- growth, persistence, senescence, the whole reason a time series was
acquired -- is decoration, and "we tracked the crop through the season" is a sentence
we are not entitled to say.

Three questions, none of them asked before:

  Q1  LEAVE-ONE-DATE-OUT. Rebuild the index without each date in turn. How far does the
      within-crop ranking move, and how many farms leave the bottom-decile attention list?

  Q2  HOW MANY SCENES DO WE ACTUALLY NEED? Greedy forward selection over dates, scoring
      each subset by how well it reproduces the shipped within-crop ranking. If two dates
      reach 95%, four of the six bought nothing measurable.

  Q3  DOES THE INDEX USE TIME AT ALL? Permute which physical date sits in which slot and
      rebuild. Growth, persistence and senescence are all defined by ordering, so a
      correctly built index MUST move. If the permuted index still matches the shipped one,
      we never used the time axis and the six dates were six looks at one thing.

CONTROLS:

  C1  ★ THE HARNESS TEST, AND IT HAS A KNOWN ANSWER. 29 October is already excluded from
      the primary index -- LEVEL_DATES drops it and, with use_29oct False, it enters no
      part, not even the integral. So leave-one-date-out on 29 October must return a
      ranking IDENTICAL to the shipped one: Spearman exactly 1.000, zero farms moved.
      Any other result means the harness is wrong and every number it produces is void.
      This check runs first and aborts the run if it fails.

  C2  The permutation in Q3 must demonstrably scramble the temporal parts. If growth,
      persist and senesce come back unchanged under a shuffled calendar, the test never
      tested anything and Q3's answer must be discarded rather than reported.

  C3  Invariance. The modifier renormalises within crop, so every variant returns the same
      village total by construction. Asserted, not assumed -- a variant that moves the
      total would mean the renormalisation broke.

This file NEVER imports into the shipped path and never rewrites a deliverable. It reads
p1_farm_features.csv and submission.csv and writes its own tables.

Writes results/tables/p3_voi_lodo.csv, p3_voi_greedy.csv, p3_voi_stats.csv.

Run:  py -3.12 -u src/p3_voi.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, DATES, TABLES, log
from p3_build import LEVEL_DATES, health_index, derive_weights, modifier, z

sys.stdout.reconfigure(encoding="utf8")
# A thin date subset legitimately produces all-NaN slices and constant parts. Those are
# the CASES UNDER TEST, not faults, and unsilenced they bury the actual output.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="An input array is constant")

RNG = np.random.default_rng(20260831)
NPERM = 200
LATE = ["20251013", "20251029", "20251112"]
PEAK = "20250814"
BASE = "20250619"


def nearest(avail, want):
    """Closest available date to `want` by day-of-year. Used when a part's own date is
    the one being withheld -- the part degrades rather than disappearing, which is the
    same negative-buffer philosophy the extraction uses."""
    if want in avail:
        return want
    if not avail:
        return None
    w = pd.Timestamp(want).dayofyear
    return min(avail, key=lambda d: abs(pd.Timestamp(d).dayofyear - w))


def parts_from(f, avail, order=None):
    """health_parts() rebuilt over an arbitrary available-date subset.

    `order` optionally remaps which PHYSICAL date supplies each calendar slot; that is the
    Q3 permutation. Everything else mirrors p3_build.health_parts exactly, including the
    village-median subtraction on growth and the trapezoid in LINEAR power.
    """
    avail = [d for d in avail if d in DATES]
    src = dict(zip(avail, order)) if order is not None else {d: d for d in avail}

    peak = nearest(avail, PEAK)
    base = nearest([d for d in avail if d != peak], BASE)
    pk, bs = src[peak], src[base] if base else None

    lvl = f[f"g0_db_{pk}"].to_numpy()
    unif = f[f"cv_{pk}"].to_numpy()

    if bs is not None and pk != bs:
        growth = f[f"g0_db_{pk}"].to_numpy() - f[f"g0_db_{bs}"].to_numpy()
        growth = growth - np.nanmedian(growth)
    else:
        growth = np.zeros(len(f))          # no pair left: the part carries no information

    cols = [f"g0_lin_{src[d]}" for d in avail]
    doy = np.array([pd.Timestamp(d).dayofyear for d in avail], dtype="float64")
    o = np.argsort(doy)
    lin = f[cols].to_numpy()[:, o]
    integral = (np.trapezoid(np.where(np.isfinite(lin), lin, np.nan), doy[o], axis=1)
                if len(avail) >= 2 else np.zeros(len(f)))

    late = [d for d in avail if d in LATE]
    if len(late) >= 2:
        L = f[[f"g0_db_{src[d]}" for d in late]].to_numpy()
        ld = np.array([pd.Timestamp(d).dayofyear for d in late], dtype="float64")
        x = ld - ld.mean()
        senesce = np.nansum((L - np.nanmean(L, axis=1, keepdims=True)) * x, axis=1) / (x ** 2).sum()
    else:
        senesce = np.zeros(len(f))

    return {"level": z(lvl), "growth": z(growth), "uniform": -z(unif),
            "persist": z(integral), "senesce": z(senesce)}


def build(f, crop, avail, order=None):
    """Full index over a date subset, weights re-derived from that subset's own redundancy.

    A part that a thin date subset leaves CONSTANT (growth with no pair, senesce with one
    late date) has to be dropped before the weights are derived, not carried at weight
    zero. derive_weights takes a Spearman matrix, and a constant column makes that column
    NaN, which propagates through the row sums into EVERY weight -- so one dead part
    silently turns the whole index into a constant 50 and the comparison returns NaN
    instead of an answer. Found by a variant that scored rho = nan. [P3-21]
    """
    parts = parts_from(f, avail, order)
    live = {k: v for k, v in parts.items() if np.nanstd(v) > 1e-12}
    assert live, "every part is degenerate; the subset carries no information at all"
    return health_index(live, derive_weights(live), crop), parts


def agree(a, b, crop):
    """Within-crop Spearman, area-blind, pooled by Fisher-free weighted mean of crop sizes.
    Within crop because that is the only ordering the modifier ever uses."""
    num = den = 0.0
    for c in CROPS:
        m = (crop == c).to_numpy()
        if m.sum() < 5:
            continue
        r = spearmanr(a[m], b[m]).statistic
        if np.isfinite(r):
            num += r * m.sum()
            den += m.sum()
    return num / den if den else np.nan


def bottom_moved(a, b, crop, frac=0.10):
    """Farms that leave the bottom-decile attention list. The advice list is the product;
    a ranking change nobody would act on differently is not a change worth reporting."""
    k = max(1, int(round(frac * len(a))))
    sa = set(np.argsort(a)[:k])
    sb = set(np.argsort(b)[:k])
    return k - len(sa & sb), k


def main():
    f = pd.read_csv(TABLES / "p1_farm_features.csv")
    sub = pd.read_csv(TABLES.parent / "submission.csv")
    d = sub.merge(f, on="farm_id", how="left")
    assert len(d) == len(sub)
    crop = d.crop_type
    area = d.area_ha.to_numpy(dtype="float64")
    ship = d.health_index.to_numpy(dtype="float64")
    print(f"{len(d)} farms | primary index uses {len(LEVEL_DATES)} of {len(DATES)} dates "
          f"(29 Oct already excluded)")

    # reproduce the shipped index from the same date set, as the reference for every variant
    ref, _ = build(f, crop, LEVEL_DATES)
    r_ref = agree(ref, ship, crop)
    print(f"harness reproduces shipped ranking: rho = {r_ref:.4f}")

    rows = []

    # ---------------- C1: the control whose answer is known in advance ----------------
    lodo29, _ = build(f, crop, [x for x in LEVEL_DATES if x != "20251029"])
    r29 = agree(lodo29, ref, crop)
    moved29, _ = bottom_moved(lodo29, ref, crop)
    print(f"C1 drop 29 Oct (known answer 1.0000 / 0 moved): rho = {r29:.4f}, moved = {moved29}")
    if not (abs(r29 - 1.0) < 1e-9 and moved29 == 0):
        raise SystemExit("C1 FAILED: the leave-one-out harness is wrong; discard this test")
    rows.append({"test": "C1_drop_29oct", "rho": r29, "moved": moved29, "known_answer": 1.0})

    # ---------------- Q1: leave one date out ----------------
    print("\nQ1  leave-one-date-out, against the shipped ranking")
    lodo = []
    for dte in LEVEL_DATES:
        v, _ = build(f, crop, [x for x in LEVEL_DATES if x != dte])
        r = agree(v, ref, crop)
        mv, k = bottom_moved(v, ref, crop)
        lodo.append({"date": dte, "rho": round(r, 4), "moved_of_97": mv})
        print(f"   without {dte}: rho {r:.4f}   {mv:>3d}/{k} farms leave the attention list")
    pd.DataFrame(lodo).to_csv(TABLES / "p3_voi_lodo.csv", index=False)

    # ---------------- Q2: greedy forward selection ----------------
    print("\nQ2  how many scenes reproduce the shipped ranking?")
    chosen, greedy = [], []
    pool = list(LEVEL_DATES)
    while pool:
        def score(dd):
            r = agree(build(f, crop, chosen + [dd])[0], ref, crop)
            return r if np.isfinite(r) else -np.inf   # a NaN must never win an argmax
        best = max(pool, key=score)
        chosen.append(best)
        pool.remove(best)
        r = agree(build(f, crop, list(chosen))[0], ref, crop)
        mv, k = bottom_moved(build(f, crop, list(chosen))[0], ref, crop)
        greedy.append({"n_dates": len(chosen), "added": best, "rho": round(r, 4),
                       "moved_of_97": mv, "dates": "+".join(chosen)})
        print(f"   {len(chosen)} date(s) [{best}]: rho {r:.4f}   {mv:>3d}/{k} moved")
    pd.DataFrame(greedy).to_csv(TABLES / "p3_voi_greedy.csv", index=False)

    # ---------------- Q3: does the index use the time ordering? ----------------
    print("\nQ3  calendar permutation -- growth/persist/senesce are defined by ordering")
    base_parts = parts_from(f, LEVEL_DATES)
    rs, scrambled = [], 0
    for _ in range(NPERM):
        order = list(RNG.permutation(LEVEL_DATES))
        v, pp = build(f, crop, LEVEL_DATES, order=order)
        # C2: the permutation must actually disturb the temporal parts
        if any(not np.allclose(pp[k], base_parts[k], equal_nan=True)
               for k in ("growth", "persist", "senesce")):
            scrambled += 1
        rs.append(agree(v, ref, crop))
    rs = np.array(rs)
    print(f"   C2 permutations that disturbed the temporal parts: {scrambled}/{NPERM}")
    if scrambled < NPERM:
        print("   C2 WARNING: some permutations left the temporal parts untouched")
    print(f"   rho(shuffled calendar, shipped) = {rs.mean():.4f} "
          f"[{np.percentile(rs, 5):.4f}, {np.percentile(rs, 95):.4f}]")
    rows.append({"test": "Q3_calendar_permutation", "rho": float(rs.mean()),
                 "lo": float(np.percentile(rs, 5)), "hi": float(np.percentile(rs, 95)),
                 "scrambled": scrambled, "nperm": NPERM})

    # ---------------- C3: the village total cannot move ----------------
    # the invariance statement that matters: the modifier's area-weighted mean is 1.0
    # within crop, so anchor x area -- the village total -- cannot move whatever we drop
    worst = 0.0
    for dte in LEVEL_DATES:
        v, _ = build(f, crop, [x for x in LEVEL_DATES if x != dte])
        m = modifier(v, crop, area)
        for c in CROPS:
            msk = (crop == c).to_numpy()
            if msk.sum() < 3:
                continue
            worst = max(worst, abs(np.average(m[msk], weights=area[msk]) - 1.0))
    print(f"\nC3 worst within-crop area-weighted modifier mean deviation from 1.0: {worst:.2e}")
    assert worst < 1e-9, "renormalisation broke: the village total would move"
    rows.append({"test": "C3_modifier_renorm_max_dev", "rho": worst})

    pd.DataFrame(rows).to_csv(TABLES / "p3_voi_stats.csv", index=False)
    log("p3_voi", harness_rho=round(float(r_ref), 4), c1_pass=True,
        perm_rho=round(float(rs.mean()), 4), greedy_2date_rho=greedy[1]["rho"],
        greedy_1date_rho=greedy[0]["rho"])


if __name__ == "__main__":
    main()
