"""Phase 1.1: the look-side question and the wetness question, on stable targets.

TWO QUESTIONS, one instrument.

  Q1  LOOK SIDE. 29 October is RIGHT-looking; the other five are left. A dihedral corner
      -- a wall, a bund, a canal edge -- that throws a bright double bounce back to a
      sensor on one side need not do it at all from the other. If the reversal is a
      constant offset we can subtract it and 29 Oct enters temporal differences freely.
      If it is per-target orientation, no scalar fixes it and 29 Oct can only be used
      where the targets are isotropic. **This decides whether 29 Oct is usable at all in
      a change signal**, and it is a question no other team is likely to have asked.

  Q2  WETNESS. 01:37 and 19:22 local against four morning/midday passes. The Phase 0
      meteorology already reframed this: dew-point depression on the two night dates
      (4.9 and 7.4 degC) is DRIER than on 19 June (0.9) and 14 August (2.0), so dew is not
      the discriminator -- but 81 mm of unseasonal rain over 26-28 Oct put ERA5-Land soil
      moisture at 0.372 for the 29 Oct pass against 0.133 a fortnight earlier. So the test
      is soil moisture, and the discriminator is targets that cannot respond to it.

THE DESIGN, and the trap it had to be rebuilt to avoid. Three classes:

  WATER    persistently darkest and temporally quiet. Isotropic: no orientation, no dew,
           no canopy. Any test-pair difference here is calibration, noise or wind, and it
           is the floor every other class is judged against.
  BUILT    persistently brightest and temporally quiet. Dihedral, strongly orientation-
           dependent, and it CANNOT respond to soil moisture or dew. It is the control
           for Q2 and the instrument for Q1.
  CROP     inside the farm polygons and in neither of the above. Responds to both.

★ THE TRAP. The first version of this file defined the classes on the four Round-2 dates
and then evaluated all six. That is a selection bias, not an experiment: picking the
brightest 0.5% of pixels on dates A-D selects partly on speckle, so those pixels are
GUARANTEED to come out darker on any date not used to pick them -- and it duly produced a
clean 3.4 dB "drop" on both new dates, in both look directions, which is exactly the shape
a real finding would have. It is the same statistical manufacture that made Round 2's
coherence experiment uninformative (scoring bright pixels against an all-pixel floor), and
it was rebuilt rather than reported.

The fix is out-of-sample evaluation on four two-date folds. Every date is scored only
under folds that did not help define the mask, so every number below is out-of-sample,
and the in-sample-minus-out-of-sample gap is measured explicitly as `selection_bias_db`
so the size of the artefact is on the record next to the numbers it would have corrupted.

Every statistic is computed on the pixels valid on ALL SIX dates. Without that, a
coverage difference between dates masquerades as a radiometric one -- and 29 Oct's swath
already covers 5 points less of the AOI than the others.

PREDICTIONS, written before the run (§7.2's rule applied to §7.1):
  P1  water differs by < 1 dB across every pair -- if not, something is wrong upstream.
  P2  built-up shows a LARGER per-target spread on the look-reversed pair than on either
      same-look control. This is the anisotropy hypothesis.
  P3  crop-minus-built rises on 29 Oct relative to 13 Oct, by more than it does on 12 Nov,
      tracking soil moisture (0.372 / 0.133 / 0.220) rather than time of day.

KILL CRITERION for Q1: if the built-up per-target spread on the look-reversed pair is not
at least 1.5x the larger of the two same-look controls, the anisotropy hypothesis is dead
and 29 Oct may be differenced like any other date.

Writes results/tables/p1_stable_targets.csv, p1_looksides.csv, p1_stable_bias.csv
and results/figures/p1_stable_targets.png.

Run:  py -3.12 src/p1_stable.py
"""
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATES, FARMS, FIGURES, TABLES, db, log, scene

import geopandas as gpd

UTM = 32643
WATER_PCT, BUILT_PCT = 1.0, 99.5
SMOOTH = 3                    # pixels; water is spatially coherent, dark speckle is not
MIN_TARGET_PX = 4             # a "target" smaller than this is a speckle spike
SPREAD_RATIO_KILL = 1.5

# Two-date folds. Each Round-2 date is out-of-sample in exactly two of them, so no date
# is scored on a mask its own speckle helped choose. The two new dates are out-of-sample
# in all four, which is the point.
FOLDS = {"F1": ("20250606", "20250619"), "F2": ("20250814", "20251013"),
         "F3": ("20250606", "20251013"), "F4": ("20250619", "20250814")}

# (label, date_a, date_b, fold that excludes BOTH, what the pair isolates)
PAIRS = [
    ("TEST look-reversed", "20251029", "20251112", "F1",
     "0.094 deg apart in incidence, 14 days, OPPOSITE look sides"),
    ("CTRL same-look A",   "20250619", "20250814", "F3",
     "0.068 deg apart, 56 days, both left -- the tightest same-look incidence match"),
    ("CTRL same-look B",   "20251013", "20251112", "F1",
     "1.78 deg apart, 30 days, both left -- the tightest same-look TIME match"),
]

# ERA5-Land at the exact overpass hour, from p0_weather. Context only: one value for all
# 966 farms, so under the resolution-ceiling rule it may never rank farms. [prompt 5.1]
SOIL_M = {"20250606": 0.151, "20250619": 0.434, "20250814": 0.196,
          "20251013": 0.133, "20251029": 0.372, "20251112": 0.220}
WIND = {"20250606": 7.1, "20250619": 6.7, "20250814": 8.1,
        "20251013": 9.6, "20251029": 14.1, "20251112": 7.6}


def stack(quantity="beta0", grid="base"):
    out = {}
    for d in DATES:
        with rasterio.open(CACHE / f"{quantity}_{grid}_{d}.tif") as s:
            out[d] = s.read(1).astype("float64")
            tf, shape = s.transform, (s.height, s.width)
    return out, tf, shape


def crop_mask(tf, shape):
    from rasterio.features import rasterize
    farms = gpd.read_file(FARMS).to_crs(UTM)
    return rasterize([(g, 1) for g in farms.geometry], out_shape=shape, transform=tf,
                     dtype="uint8").astype(bool)


def classes(S, valid, define_on, inside):
    """Water / built / crop masks from `define_on` only."""
    D = np.array([db(S[d]) for d in define_on])
    m = D.mean(axis=0)
    quiet = D.std(axis=0, ddof=1) <= np.nanmedian(D.std(axis=0, ddof=1)[valid])

    # smooth before thresholding: on single-look-derived data a raw low percentile
    # selects speckle troughs, not water. Water is coherent; a trough is not.
    sm = ndi.uniform_filter(np.nan_to_num(np.where(valid, m, 0.0)), SMOOTH) / np.maximum(
        ndi.uniform_filter(valid.astype("float64"), SMOOTH), 1e-9)
    lo, hi = np.nanpercentile(sm[valid], [WATER_PCT, BUILT_PCT])
    water = valid & quiet & (sm <= lo)
    built = valid & quiet & (sm >= hi)
    return {"water": water, "built": built,
            "crop": valid & inside & ~water & ~built}


def main():
    S, tf, shape = stack()
    valid = np.all([np.isfinite(S[d]) & (S[d] > 0) for d in DATES], axis=0)
    inside = crop_mask(tf, shape)
    X = {d: db(S[d]) for d in DATES}

    M = {f: classes(S, valid, dd, inside) for f, dd in FOLDS.items()}
    for f, mk in M.items():
        log("p1_stable.classes", fold=f, define_on=list(FOLDS[f]),
            **{k: int(v.sum()) for k, v in mk.items()})

    # ---- the artefact, measured: in-sample minus out-of-sample, per class ----
    bias = []
    for cls in ("water", "built"):
        for f, dd in FOLDS.items():
            ins = [float(np.nanmedian(X[d][M[f][cls]])) for d in dd]
            out = [float(np.nanmedian(X[d][M[f][cls]])) for d in DATES if d not in dd]
            bias.append({"class": cls, "fold": f, "define_on": "+".join(dd),
                         "in_sample_db": round(np.mean(ins), 3),
                         "out_sample_db": round(np.mean(out), 3),
                         "selection_bias_db": round(np.mean(ins) - np.mean(out), 3)})
    bias = pd.DataFrame(bias)
    bias.to_csv(TABLES / "p1_stable_bias.csv", index=False)
    print("\nSELECTION-BIAS CONTROL (in-sample minus out-of-sample, dB)")
    for cls, g in bias.groupby("class"):
        print(f"  {cls:<7} per fold {[f'{v:+.2f}' for v in g.selection_bias_db]}"
              f"   mean {g.selection_bias_db.mean():+.2f}")
        log("p1_stable.selection_bias", cls=cls,
            mean_db=round(float(g.selection_bias_db.mean()), 3),
            per_fold={r.fold: float(r.selection_bias_db) for r in g.itertuples()})

    # ---- season table, every value out-of-sample ----
    rows = []
    for d in DATES:
        sc = scene(d)
        folds = [f for f, dd in FOLDS.items() if d not in dd]
        r = {"date": d, "look": sc["pointing"], "local": sc["local"][11:16],
             "incidence_deg": round(float(sc["incidence"]), 3), "n_folds": len(folds),
             "soil_moisture": SOIL_M[d], "wind_ms": WIND[d]}
        for k in ("water", "built", "crop"):
            v = [float(np.nanmedian(X[d][M[f][k]])) for f in folds]
            r[f"{k}_db"] = round(float(np.mean(v)), 3)
            r[f"{k}_fold_sd"] = round(float(np.std(v, ddof=1)), 3)
        # built-up cannot respond to soil moisture or dew; referencing to it removes any
        # scene-wide gain drift and leaves the moisture term.
        r["crop_minus_built"] = round(r["crop_db"] - r["built_db"], 3)
        r["water_minus_built"] = round(r["water_db"] - r["built_db"], 3)
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(TABLES / "p1_stable_targets.csv", index=False)

    print(f"\n{'date':<10}{'look':>6}{'local':>7}{'inc':>7}{'soilm':>7}{'wind':>6}"
          f"{'water':>9}{'built':>9}{'crop':>9}{'crop-built':>12}{'foldsd':>8}")
    for _, r in tab.iterrows():
        print(f"{r.date:<10}{r.look:>6}{r.local:>7}{r.incidence_deg:>7.2f}"
              f"{r.soil_moisture:>7.3f}{r.wind_ms:>6.1f}"
              f"{r.water_db:>9.2f}{r.built_db:>9.2f}{r.crop_db:>9.2f}"
              f"{r.crop_minus_built:>+12.2f}{max(r.water_fold_sd, r.built_fold_sd):>8.2f}")

    # ---- per-target spread: constant offset, or per-target orientation? ----
    sp = per_target_spread(X, M)
    sp.to_csv(TABLES / "p1_looksides.csv", index=False)
    print(f"\n{'class':<7}{'pair':<21}{'fold':>5}{'n':>6}{'mean':>8}{'sd':>8}{'IQR':>8}"
          f"{'p05..p95':>16}")
    for _, r in sp.iterrows():
        print(f"{r['class']:<7}{r.pair:<21}{r.fold:>5}{r.n_targets:>6}"
              f"{r.mean_diff_db:>+8.2f}{r.sd_db:>8.2f}{r.iqr_db:>8.2f}"
              f"{f'{r.p05_db:+.1f}..{r.p95_db:+.1f}':>16}")

    b = sp[sp["class"] == "built"].set_index("pair")
    w = sp[sp["class"] == "water"].set_index("pair")
    test = float(b.loc["TEST look-reversed", "sd_db"])
    ctrl = float(max(b.loc["CTRL same-look A", "sd_db"], b.loc["CTRL same-look B", "sd_db"]))
    log("p1_stable.lookside", built_sd_test_db=round(test, 3),
        built_sd_worst_control_db=round(ctrl, 3), ratio=round(test / ctrl, 3),
        # water is isotropic: if it spreads as much as built on the same pair, none of
        # the spread can be attributed to target orientation.
        water_sd_test_db=round(float(w.loc["TEST look-reversed", "sd_db"]), 3),
        kill_threshold=SPREAD_RATIO_KILL,
        verdict="ANISOTROPY" if test / ctrl >= SPREAD_RATIO_KILL else "NO ANISOTROPY DETECTED")

    t = tab.set_index("date")
    log("p1_stable.wetness",
        crop_minus_built={d: float(t.loc[d, "crop_minus_built"]) for d in DATES},
        rho_crop_soilm=round(float(pd.Series([t.loc[d, "crop_db"] for d in DATES]).corr(
            pd.Series([SOIL_M[d] for d in DATES]), method="spearman")), 3),
        rho_water_wind=round(float(pd.Series([t.loc[d, "water_db"] for d in DATES]).corr(
            pd.Series([WIND[d] for d in DATES]), method="spearman")), 3),
        rho_built_soilm=round(float(pd.Series([t.loc[d, "built_db"] for d in DATES]).corr(
            pd.Series([SOIL_M[d] for d in DATES]), method="spearman")), 3))

    _figure(tab, sp)


def per_target_spread(X, M):
    """Per-target dB difference for each pair -- the constant-offset vs anisotropy test.

    A constant offset shifts every target by the same amount, so the spread of per-target
    differences stays at the noise level. An orientation effect moves each target by a
    different amount, and the SPREAD is what detects that -- the mean cannot.

    Each pair uses the fold whose define-set contains NEITHER of its two dates.
    """
    rows = []
    for cls in ("built", "water"):
        for name, a, b, fold, why in PAIRS:
            lab, _ = ndi.label(M[fold][cls])
            keep = np.where(np.bincount(lab.ravel()) >= MIN_TARGET_PX)[0]
            keep = keep[keep > 0]
            diff = np.array(ndi.mean(X[a], lab, keep)) - np.array(ndi.mean(X[b], lab, keep))
            diff = diff[np.isfinite(diff)]
            rows.append({"class": cls, "pair": name, "fold": fold, "a": a, "b": b,
                         "why": why, "n_targets": len(diff),
                         "mean_diff_db": round(float(diff.mean()), 3),
                         "median_diff_db": round(float(np.median(diff)), 3),
                         "sd_db": round(float(diff.std(ddof=1)), 3),
                         "iqr_db": round(float(np.subtract(*np.percentile(diff, [75, 25]))), 3),
                         "p05_db": round(float(np.percentile(diff, 5)), 3),
                         "p95_db": round(float(np.percentile(diff, 95)), 3)})
    return pd.DataFrame(rows)


def _figure(tab, sp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    x = np.arange(len(tab))

    ax = axes[0]
    for k, c in (("water", "#2b6cb0"), ("built", "#c05621"), ("crop", "#2f855a")):
        ax.errorbar(x, tab[f"{k}_db"], yerr=tab[f"{k}_fold_sd"], fmt="o-", color=c,
                    capsize=3, label=k)
    for i, r in tab.iterrows():
        if r.look == "right":
            ax.axvspan(i - 0.35, i + 0.35, color="#fed7aa", alpha=0.5, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([d[4:] for d in tab.date], rotation=45)
    ax.set_ylabel(r"out-of-sample median $\beta^0$ (dB)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Stable targets across the season\n"
                 "(shaded = right-looking; bars = fold spread)", fontsize=10)

    ax = axes[1]
    ax.scatter(tab.soil_moisture, tab.crop_db, c="#2f855a", s=60, label="crop")
    ax.scatter(tab.soil_moisture, tab.built_db, c="#c05621", s=60, label="built (control)")
    for _, r in tab.iterrows():
        ax.annotate(r.date[4:], (r.soil_moisture, r.crop_db), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("ERA5-Land soil moisture 0-7 cm at overpass")
    ax.set_ylabel(r"$\beta^0$ (dB)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Q2: does cropland track soil moisture\nwhere built-up cannot?", fontsize=10)

    ax = axes[2]
    b = sp[sp["class"] == "built"]
    w = sp[sp["class"] == "water"]
    xx = np.arange(len(b))
    ax.bar(xx - 0.18, b.sd_db, 0.36, color="#c05621", edgecolor="#333", label="built")
    ax.bar(xx + 0.18, w.sd_db, 0.36, color="#2b6cb0", edgecolor="#333",
           label="water (isotropic)")
    ax.set_xticks(xx); ax.set_xticklabels(b.pair, rotation=15, fontsize=8)
    ax.set_ylabel("sd of per-target dB difference")
    ax.set_title("Q1: does look reversal move each target differently?", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    p = FIGURES / "p1_stable_targets.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    log("p1_stable.fig", name=p.name)


if __name__ == "__main__":
    main()
