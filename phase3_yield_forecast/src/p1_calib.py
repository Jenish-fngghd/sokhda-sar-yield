"""Phase 1.1: adjudicate the calibration convention on all six scenes.

The question. Capella ships `scale_factor`; the product is annotated `beta_nought`.
Round 1 and Round 2 used beta0 = SF * |z|^2. The capella-reader reference defines
beta0_complex = SF * DN, so POWER carries SF**2. ESA's note states a formula whose `sc`
is ambiguous and cannot settle it. Round 2's post-mortem settled it on the vendor's own
`nesz_peak` -- an absolute dB level Capella put in the product -- over four scenes, at
0.35 dB mean absolute error under SF**2 against 26-28 dB under SF.

The physics that makes nesz_peak an adjudicator: a scene contains smooth dark surfaces
(open water, quiet tarmac, dry smooth bare soil) whose return approaches the system
noise floor. The darkest percentiles must therefore sit NEAR the declared floor. Tens of
dB above it would mean the sensor never reaches its own noise anywhere in a 27 km strip,
which is not a thing a working SAR does.

WHAT IS NEW HERE, and it is the whole point: two scenes Round 2 never saw. If 29 Oct and
12 Nov behave like the four, SF**2 is confirmed on six scenes with six different scale
factors and six different declared floors, and the R2 defect is closed. If they do NOT,
that is a finding about those products -- not about the formula -- and it must be found
now, because both new dates are exactly the ones the forecast leans on.

KILL CRITERION, written before the run. Under SF**2, each new scene's |darkest-percentile
minus nesz_peak| must fall inside [min, max] of the four known scenes widened by 1.0 dB.
Outside that, the two new products are not radiometrically like the four and every
inter-date difference that spans the boundary is suspect until explained.

A SECOND TEST, added after the first run and reported alongside it, because the first
one turned out to be answerable in a way that changes what Round 2 concluded. The
darkest content in a scene sits AT OR ABOVE the noise floor, never below it: a negative
residual means the product's own darkest pixels are quieter than the noise the vendor
declares, which is not a thing that can happen. So the residual SIGN is a harder test
than its magnitude, and a small MAE bought by sitting below the floor is worse than a
larger one above it. Run it on all three quantities to find which one nesz_peak is
actually referenced to -- that is a question about the vendor's annotation, and only the
data can answer it.

Reads results/cache/{beta0,gamma0}_base_*.tif (written by p1_prep under SF**2).
Writes results/tables/p1_calibration.csv and results/figures/p1_calib_nesz.png.

Run:  py -3.12 src/p1_calib.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATES, FIGURES, TABLES, log, scene

R2_DATES = {"20250606", "20250619", "20250814", "20251013"}   # the four R2 also had
PCTS = [0.01, 0.1, 0.5, 1.0]                                  # darkest percentiles, %
QUANTITIES = ["beta0", "sigma0", "gamma0"]
TOL_DB = 1.0                                                  # kill-criterion widening


def dark_percentiles(date, quantity):
    """Darkest percentiles of a geocoded power raster, in dB, under SF**2.

    Multilooked (average resampling in p1_prep IS the multilook). It must be: a
    single-look power distribution has an exponential tail toward zero, so its 0.1
    percentile sits ~30 dB below its mean and would say nothing about a noise floor.
    """
    with rasterio.open(CACHE / f"{quantity}_base_{date}.tif") as s:
        a = s.read(1).astype("float64")
    a = a[np.isfinite(a) & (a > 0)]
    return {p: float(np.percentile(10 * np.log10(a), p)) for p in PCTS}, a.size


def main():
    rows = []
    for d in DATES:
        sc = scene(d)
        off = 10 * np.log10(sc["scale_factor"])   # SF**2 dB = SF dB + 10log10(SF)
        r = {"date": d, "known_to_r2": d in R2_DATES, "look": sc["pointing"],
             "incidence_deg": round(float(sc["incidence"]), 4),
             "scale_factor": sc["scale_factor"], "nesz_peak_db": sc["nesz_peak"],
             "sf2_minus_sf_db": round(off, 3)}
        for q in QUANTITIES:
            v, r["n_pixels"] = dark_percentiles(d, q)
            for p in PCTS:
                r[f"{q}_p{p}_db"] = round(v[p], 3)                     # our SF**2 chain
                r[f"{q}_p{p}_vs_nesz"] = round(v[p] - sc["nesz_peak"], 3)
                # what Round 2 would have printed: SF, not SF**2
                r[f"{q}_p{p}_sf_vs_nesz"] = round(v[p] - off - sc["nesz_peak"], 3)
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "p1_calibration.csv", index=False)

    P = 0.1     # the percentile R2 reported on; everything below is at this one
    print(f"\n{'date':<10}{'new':>5}{'look':>7}{'NESZ':>8}{'SF b0':>9}{'vs':>8}"
          f"{'SF^2 b0':>10}{'vs':>8}{'SF^2 g0':>10}{'vs':>8}")
    for _, r in df.iterrows():
        print(f"{r.date:<10}{'' if r.known_to_r2 else 'NEW':>5}{r.look:>7}"
              f"{r.nesz_peak_db:>8.2f}"
              f"{r[f'beta0_p{P}_db'] - r.sf2_minus_sf_db:>9.1f}"
              f"{r[f'beta0_p{P}_sf_vs_nesz']:>+8.1f}"
              f"{r[f'beta0_p{P}_db']:>10.2f}{r[f'beta0_p{P}_vs_nesz']:>+8.2f}"
              f"{r[f'gamma0_p{P}_vs_nesz'] + r.nesz_peak_db:>10.2f}"
              f"{r[f'gamma0_p{P}_vs_nesz']:>+8.2f}")

    # ---- which quantity is nesz_peak referenced to?  The sign test decides. ----
    print("\n  quantity   mean dB    MAE    sd   scenes BELOW the declared floor")
    for q in QUANTITIES:
        v = df[f"{q}_p{P}_vs_nesz"].to_numpy()
        n_bad = int((v < 0).sum())
        print(f"  {q:<10}{v.mean():>+8.2f}{np.abs(v).mean():>7.3f}{v.std(ddof=1):>6.3f}"
              f"   {n_bad}/{len(v)}"
              + ("   <-- physically admissible" if n_bad == 0 else "   IMPOSSIBLE"))
        log("p1_calib.reference_quantity", quantity=q,
            mean_db=round(float(v.mean()), 3), mae_db=round(float(np.abs(v).mean()), 3),
            sd_db=round(float(v.std(ddof=1)), 3), n_below_floor=n_bad,
            admissible=(n_bad == 0))

    old = df[df.known_to_r2]
    new = df[~df.known_to_r2]
    for q in QUANTITIES:
        e_old = old[f"{q}_p{P}_vs_nesz"].abs()
        e_new = new[f"{q}_p{P}_vs_nesz"].abs()
        lo, hi = e_old.min() - TOL_DB, e_old.max() + TOL_DB
        verdict = "PASS" if ((e_new >= lo) & (e_new <= hi)).all() else "FAIL"
        log("p1_calib.nesz", quantity=q, pct=P,
            mae_sf2_known=round(float(e_old.mean()), 3),
            mae_sf2_new=round(float(e_new.mean()), 3),
            known_band_db=[round(float(lo), 2), round(float(hi), 2)],
            new_abs_err=[round(float(v), 3) for v in e_new],
            kill_criterion=verdict)

    mae_sf = float(df[f"beta0_p{P}_sf_vs_nesz"].abs().mean())
    mae_sf2 = float(df[f"beta0_p{P}_vs_nesz"].abs().mean())
    log("p1_calib.verdict", n_scenes=len(df), pct=P,
        mae_beta0_under_SF=round(mae_sf, 2), mae_beta0_under_SF2=round(mae_sf2, 3),
        ratio=round(mae_sf / max(mae_sf2, 1e-9), 1))

    _figure(df, P)


def _figure(df, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    x = np.arange(len(df))
    lbl = [f"{d[6:8]} {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(d[4:6])-1]}"
           for d in df.date]
    col = ["#3b7dd8" if k else "#e07b39" for k in df.known_to_r2]

    ax = axes[0]
    ax.plot(x, df[f"beta0_p{P}_db"] - df.sf2_minus_sf_db, "s--", color="#999",
            label=r"darkest 0.1% under SF (Round 2)")
    ax.plot(x, df[f"beta0_p{P}_db"], "o-", color="#2b6", label=r"darkest 0.1% under SF$^2$")
    ax.plot(x, df.nesz_peak_db, "k^-", label="declared nesz_peak")
    ax.set_xticks(x); ax.set_xticklabels(lbl)
    ax.set_ylabel(r"$\beta^0$ dB"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("The noise floor adjudicates: SF sits ~28 dB above it,\n"
                 r"SF$^2$ lands on it", fontsize=10)

    ax = axes[1]
    ax.bar(x, df[f"beta0_p{P}_vs_nesz"], color=col, edgecolor="#333")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(lbl)
    ax.set_ylabel(r"darkest 0.1% $-$ nesz_peak (dB), SF$^2$")
    ax.set_title("Residual per scene (orange = new in Round 3)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    p = FIGURES / "p1_calib_nesz.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    log("p1_calib.fig", name=p.name)


if __name__ == "__main__":
    main()
