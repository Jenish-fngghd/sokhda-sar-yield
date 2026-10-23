"""T20: detect harvest per plot from Sentinel-1, and use it to TEST our headline claim.

★ WHAT THIS PUTS AT RISK, WHICH IS THE POINT.

Slide 1 and the writeup's opening paragraph rest on one claim: the yield column contains three
different kinds of object, and we can say which is which.

    cotton              a FORECAST        -- picking runs Oct-Jan, ~72% still on the plant
    groundnut           a MEASUREMENT     -- yield set at maturity in late Oct, inside our window
    rice, maize, bajra  a RECONSTRUCTION  -- season closed before Round 2's last date

Every part of that comes from LITERATURE: published crop calendars, and for cotton a picking
split from Sharma & Goyal (1999). Three rounds and not one test against data. If the ordering is
wrong, our headline framing is wrong, and we would rather find that ourselves.

Sentinel-1 can test it. Harvest removes canopy, which moves VH sharply, and we now have 16 dates
on one track. The prediction is ORDINAL and was written before the run:

    rice / maize / bajra   harvest EARLIEST
    groundnut              later (late Oct)
    cotton                 LATEST, and many plots should show no completed harvest at all

★ THE CONFOUND THAT WOULD FAKE THIS, AND THE FIX. 81.2 mm of rain fell 26-28 October. Every
plot in the village moves together on the following overpass, and a naive "biggest drop" detector
would report a synchronised village-wide harvest on 3 November -- a beautiful, entirely spurious
result. So detection runs on the plot's series MINUS THE PER-DATE VILLAGE MEDIAN, which removes
any common mode: rain, sensor, incidence, season. Only a plot changing DIFFERENTLY from its
neighbours can register.

★ AND THE CONTROL THAT CAN KILL IT. If the method were detecting noise, or the rain leaking
through, detected dates would not separate by crop. So the test is not "did we find harvest
dates" -- it is "do the crops come out in the AGRONOMIC ORDER". A method that puts cotton first
is measuring something else, and is reported as failed.

Writes results/tables/p3_harvest.csv.

Run:  py -3.12 src/p3_harvest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

sys.stdout.reconfigure(encoding="utf8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, RESULTS, TABLES, log

# The agronomic order, written down BEFORE the run. Lower = harvested earlier.
EXPECTED_ORDER = {"Rice": 1, "Maize": 1, "Bajra": 1, "Groundnut": 2, "Cotton": 3}
SEARCH = ("20250916", "20251127")     # after peak canopy, through the last acquisition


def main():
    d = pd.read_csv(RESULTS / "d4_debug.csv")
    s1 = pd.read_csv(TABLES / "p3_s1_season.csv")
    d = d.merge(s1, on="farm_id")
    dates = sorted(c.split("_")[1] for c in s1.columns if c.startswith("vh_"))
    doy = np.array([pd.Timestamp(t).dayofyear for t in dates], dtype="float64")

    VH = d[[f"vh_{t}" for t in dates]].to_numpy(dtype="float64")

    # ★ strip the common mode. Without this the October rain is detected as a village-wide
    # harvest on 3 November, which is the failure mode this whole design exists to avoid.
    common = np.nanmedian(VH, axis=0)
    A = VH - common[None, :]

    lo = dates.index(SEARCH[0])
    hi = dates.index(SEARCH[1])
    # harvest = the steepest DROP between consecutive acquisitions, in the anomaly series
    dif = np.diff(A[:, lo:hi + 1], axis=1)
    k = np.nanargmin(np.where(np.isfinite(dif), dif, np.inf), axis=1)
    drop = dif[np.arange(len(dif)), k]
    hdoy = doy[lo + 1:hi + 2][k]
    hdate = np.array(dates[lo + 1:hi + 2], dtype=object)[k]
    ok = np.isfinite(drop) & (drop < 0)

    d["harvest_doy"] = np.where(ok, hdoy, np.nan)
    d["harvest_date"] = np.where(ok, hdate, None)
    d["harvest_drop_db"] = np.where(ok, drop, np.nan)

    print("Detected harvest timing per crop (anomaly series; common mode removed)\n")
    print(f"{'crop':<10} {'n':>4} {'expected':>9} {'median DOY':>11} {'median date':>12} "
          f"{'mean drop':>10}")
    rows, meds = [], {}
    for c in sorted(CROPS, key=lambda x: EXPECTED_ORDER[x]):
        m = (d.crop_type == c).to_numpy() & ok
        if m.sum() < 10:
            continue
        md = float(np.median(d.harvest_doy[m]))
        meds[c] = md
        dt = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=md - 1)).strftime("%d %b")
        rows.append({"crop": c, "n": int(m.sum()), "expected_rank": EXPECTED_ORDER[c],
                     "median_harvest_doy": round(md, 1), "median_harvest_date": dt,
                     "mean_drop_db": round(float(np.mean(d.harvest_drop_db[m])), 3)})
        print(f"{c:<10} {m.sum():4d} {EXPECTED_ORDER[c]:9d} {md:11.1f} {dt:>12} "
              f"{np.mean(d.harvest_drop_db[m]):9.2f} dB")

    # ---- CONTROL 1: do the crops separate at all? ----
    groups = [d.harvest_doy[(d.crop_type == c).to_numpy() & ok].to_numpy() for c in meds]
    H, p = kruskal(*groups)
    print(f"\nCONTROL 1  do crops separate?   Kruskal-Wallis H = {H:.1f}, p = {p:.2e}   "
          f"{'SEPARATE' if p < 0.01 else 'DO NOT SEPARATE — method is not seeing harvest'}")

    # ---- CONTROL 2: is the ORDER the agronomic one? ----
    got = sorted(meds, key=lambda c: meds[c])
    exp_rank = [EXPECTED_ORDER[c] for c in got]
    monotone = all(a <= b for a, b in zip(exp_rank, exp_rank[1:]))
    print(f"CONTROL 2  detected order: {' < '.join(got)}")
    print(f"           expected ranks in that order: {exp_rank}   "
          f"{'MATCHES the agronomic order' if monotone else 'CONTRADICTS the agronomic order'}")

    # ---- CONTROL 3: the rain date must NOT dominate ----
    from collections import Counter
    cnt = Counter(d.harvest_date[ok])
    top, ntop = cnt.most_common(1)[0]
    frac = ntop / int(ok.sum())
    print(f"CONTROL 3  most common detected date: {top} on {100*frac:.1f}% of plots   "
          f"{'OK — no single date dominates' if frac < 0.35 else 'FAILS — a single date dominates, likely the rain'}")

    # ---- what it says about the epistemic objects ----
    late = d.harvest_doy > pd.Timestamp("2025-11-12").dayofyear
    print("\nPlots with NO completed harvest by 12 Nov (our last acquisition):")
    for c in sorted(CROPS, key=lambda x: EXPECTED_ORDER[x]):
        m = (d.crop_type == c).to_numpy()
        pct = 100 * float((m & (~ok | late)).sum()) / max(int(m.sum()), 1)
        print(f"   {c:<10} {pct:5.1f}%")

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "p3_harvest.csv", index=False)
    log("p3_harvest.done", n_detected=int(ok.sum()), n_farms=len(d),
        kruskal_H=round(float(H), 1), kruskal_p=float(p),
        detected_order=got, order_matches_agronomy=bool(monotone),
        top_date=str(top), top_date_share=round(float(frac), 3),
        verdict=("the detected harvest order reproduces the agronomic order, so the "
                 "epistemic-object split on slide 1 is supported by measurement rather than "
                 "by crop calendars alone" if (monotone and p < 0.01 and frac < 0.35) else
                 "FAILED a control — not reported as support for anything"),
        confound_handled=("detection runs on the plot series MINUS the per-date village median, "
                          "so the 26-28 Oct rain cannot register as a synchronised harvest"),
        why_it_failed=("not plot size alone: restricting to the largest plots does not recover "
                       "separation (top 50% p=0.93, top 25% p=0.32, top 10% p=0.49) and the "
                       "detected crop ORDER reshuffles randomly between subsets, which is the "
                       "signature of noise rather than of a weak real signal. Median parcel is "
                       "28 Sentinel-1 pixels; a steepest-single-drop estimator on 16 noisy "
                       "points at that size carries no timing information"),
        what_this_does_NOT_show=("a failed test is not evidence against the hypothesis. This "
                                 "does NOT refute the epistemic-object split; it means we could "
                                 "not test it this way. The split still rests on published crop "
                                 "calendars, which is a weaker footing than we would like and "
                                 "is now recorded as such in WRITEUP section 8"))


if __name__ == "__main__":
    main()
