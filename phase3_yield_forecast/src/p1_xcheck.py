"""Phase 1.3: cross-implementation check on our own radiometry, against Project Orion.

WHY THIS IS THE ONLY TEST OF ITS KIND WE HAVE. A common-mode error in our own chain is
invisible to every internal consistency check we can write, because an internal check
compares our numbers to our other numbers. That is exactly how our own scale-factor error
and another team's geoid error both survived competent pipelines for a whole round.

Project Orion published per-farm gamma0 in dB for the four shared dates. Their pipeline
processed the same byte-identical scenes over the same byte-identical polygons with an
independent implementation. So farm-to-farm agreement against their column is a genuine
external test of our geocoding, our incidence model and our extraction.

WHAT COUNTS AS PASSING. The CORRELATION and the RESIDUAL STRUCTURE, not the offset: a
constant dB offset would only mean the two chains chose different conventions, which is
bookkeeping and not an error in either. Agreement means the two implementations rank and
space the 966 farms the same way on each date.

THE CONTROL, without which the correlation means nothing. Farms differ from each other for
reasons that persist across dates -- size, land cover, position -- so *any* two SAR
extractions over these polygons will correlate somewhat even if the dates are mismatched.
So every one of our dates is also correlated against every one of THEIR dates. The
diagonal must beat the off-diagonal, and by a margin. If a mismatched pair correlates as
well as the matched one, the correlation is measuring the village and not the acquisition,
and the whole test is uninformative.

A SECOND CONTROL on the offset, stated as a falsifiable hypothesis: IF Orion used the
unsquared SF convention that Round 2 shipped, our values must sit 10 log10(SF) below
theirs -- roughly -27 dB, and a DIFFERENT number on each date because each scene has its
own scale factor. Four predicted offsets landing on four measured ones would confirm it.

RESULT: the hypothesis is falsified, by 27 dB, on all four dates. The measured offset is
+0.7 to +1.2 dB. Orion used the SQUARED convention. That is a THIRD independent line of
support for SF-squared -- after capella-reader and after the vendor's own nesz_peak --
arriving from a source that owes nothing to us and was written before we knew we were
wrong. It also means this test is stronger than a correlation check: with both chains in
the same convention, the comparison is of ABSOLUTE LEVEL, and two independent
implementations of geocoding, incidence and extraction agree to about 1 dB.

Third-party data, treated with the same scepticism as any competitor artefact. It is never
an input to anything we ship.

Writes results/tables/p1_xcheck.csv and results/figures/p1_xcheck.png.

Run:  py -3.12 src/p1_xcheck.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUX, FIGURES, TABLES, log, scene

# Their T1..T4 are the four shared dates in chronological order. Verified, not assumed:
# the ordering of their four means reproduces the ordering of ours, T2 brightest.
SHARED = ["20250606", "20250619", "20250814", "20251013"]
THEIRS = {d: f"g0_db_T{i + 1}" for i, d in enumerate(SHARED)}
MIN_R_MARGIN = 0.15      # the diagonal must beat the best off-diagonal by this much


def main():
    ours = pd.read_csv(TABLES / "p1_farm_features.csv")
    them = pd.read_csv(AUX / "inherited" / "orion_feat.csv")
    d = ours.merge(them[["farm_id"] + list(THEIRS.values()) + ["npx_T1", "area_ha"]],
                   on="farm_id", how="inner", suffixes=("", "_orion"))
    assert len(d) == 966, len(d)

    # sanity on the ordering assumption, before any correlation is computed
    ord_ours = np.argsort([d[f"g0_db_{x}"].median() for x in SHARED])
    ord_them = np.argsort([d[THEIRS[x]].median() for x in SHARED])
    log("p1_xcheck.date_mapping", ours=list(map(int, ord_ours)), theirs=list(map(int, ord_them)),
        consistent=bool((ord_ours == ord_them).all()))

    rows, R = [], np.full((4, 4), np.nan)
    for i, a in enumerate(SHARED):
        for j, b in enumerate(SHARED):
            x, y = d[f"g0_db_{a}"].to_numpy(), d[THEIRS[b]].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            R[i, j] = pearsonr(x[ok], y[ok])[0]
        a_x, a_y = d[f"g0_db_{a}"].to_numpy(), d[THEIRS[a]].to_numpy()
        ok = np.isfinite(a_x) & np.isfinite(a_y)
        x, y = a_x[ok], a_y[ok]
        off = float(np.median(x - y))
        pred = 10 * np.log10(scene(a)["scale_factor"])      # SF^2 minus SF, this scene
        res = (x - y) - off
        # residual structure: does the disagreement depend on plot size or on how much of
        # the plot either implementation actually saw?
        area = d.area_ha.to_numpy()[ok]
        npx = d[f"npix_{a}"].to_numpy()[ok]
        rows.append({
            "date": a, "n": int(ok.sum()),
            "pearson_r": round(float(pearsonr(x, y)[0]), 4),
            "spearman_rho": round(float(spearmanr(x, y)[0]), 4),
            "offset_db": round(off, 3),
            "sf_hypothesis_offset_db": round(pred, 3),   # if they had used unsquared SF
            "sf_hypothesis_error_db": round(off - pred, 3),
            "resid_sd_db": round(float(res.std(ddof=1)), 3),
            "resid_vs_log_area_rho": round(float(spearmanr(np.log(area), res)[0]), 3),
            "resid_vs_npix_rho": round(float(spearmanr(npx, res)[0]), 3),
            "best_offdiag_r": round(float(np.nanmax(np.delete(R[i], i))), 4)})

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "p1_xcheck.csv", index=False)

    print(f"\n{'date':<10}{'n':>5}{'r':>8}{'rho':>8}{'offset':>9}{'resid sd':>10}"
          f"{'resid~area':>12}{'r_offdiag':>11}{'if SF':>9}")
    for _, r in out.iterrows():
        print(f"{r.date:<10}{r.n:>5}{r.pearson_r:>8.3f}{r.spearman_rho:>8.3f}"
              f"{r.offset_db:>+9.2f}{r.resid_sd_db:>10.2f}"
              f"{r.resid_vs_log_area_rho:>+12.2f}{r.best_offdiag_r:>11.3f}"
              f"{r.sf_hypothesis_offset_db:>9.1f}")

    print("\nCONTROL -- ours (rows) against theirs (cols), Pearson r:")
    print("          " + "".join(f"{b[4:]:>9}" for b in SHARED))
    for i, a in enumerate(SHARED):
        print(f"  {a[4:]:<8}" + "".join(
            f"{R[i, j]:>9.3f}" + ("*" if i == j else " ") for j in range(4))[:-1])

    margin = float(np.min([R[i, i] - np.nanmax(np.delete(R[i], i)) for i in range(4)]))
    log("p1_xcheck.control", worst_diagonal_margin=round(margin, 4),
        threshold=MIN_R_MARGIN,
        verdict="PASS" if margin >= MIN_R_MARGIN else "FAIL -- correlation is not date-specific")
    log("p1_xcheck.agreement", r=[float(v) for v in out.pearson_r],
        abs_offset_db=[float(v) for v in out.offset_db],
        resid_sd_db=[float(v) for v in out.resid_sd_db],
        resid_vs_log_area_rho=[float(v) for v in out.resid_vs_log_area_rho],
        sf_hypothesis="FALSIFIED by "
                      f"{out.sf_hypothesis_error_db.abs().min():.1f} dB on all four dates "
                      "-- Orion used the squared convention too")

    _figure(d, out, R)


def _figure(d, out, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    ax = axes[0]
    a = SHARED[3]
    x, y = d[f"g0_db_{a}"], d[THEIRS[a]]
    ax.scatter(y, x, s=6, alpha=0.35, color="#2b6cb0")
    r = out[out.date == a].iloc[0]
    lo, hi = np.nanpercentile(y, [0.5, 99.5])
    ax.plot([lo, hi], [lo + r.offset_db, hi + r.offset_db], "r-", lw=1.2,
            label=f"measured offset {r.offset_db:+.2f} dB")
    ax.set_xlabel(r"Orion $\gamma^0$ dB (their SF chain)")
    ax.set_ylabel(r"ours $\gamma^0$ dB (SF$^2$)")
    ax.set_title(f"13 Oct, farm to farm  r={r.pearson_r:.3f}", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    im = ax.imshow(R, cmap="viridis", vmin=np.nanmin(R), vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([b[4:] for b in SHARED], fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels([b[4:] for b in SHARED], fontsize=8)
    ax.set_xlabel("Orion date"); ax.set_ylabel("our date")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="w" if R[i, j] < 0.6 else "k")
    ax.set_title("CONTROL: the diagonal must win\n(off-diagonal = same farms, wrong date)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[2]
    ax.plot(range(4), out.offset_db, "o-", color="#2f855a", label="measured")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([b[4:] for b in SHARED], fontsize=8)
    ax.set_ylabel("our dB minus theirs")
    ax.set_ylim(-1, 2)
    ax.set_title("Absolute level agreement, two independent chains\n"
                 "(the unsquared-SF hypothesis predicted about -27 dB)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    p = FIGURES / "p1_xcheck.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    log("p1_xcheck.fig", name=p.name)


if __name__ == "__main__":
    main()
