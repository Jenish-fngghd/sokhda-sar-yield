"""Stage I10 -- the media gallery and the required cover image. [D10]

Presentation Quality is 10 rubric points and the wording is "legible at a glance".
A judge scrolling a gallery gives each figure a couple of seconds, so every figure
here is built to make ONE point, with the point written on the figure itself. No
figure needs the writeup to be understood.

Deliberate choices:
  * fixed colour identity per crop, held across every figure, so the reader learns
    it once
  * the cover carries the two headline products side by side -- what grows where,
    and how it is doing -- because that is the whole brief in one image
  * the validation figures show the FAILURES too. A gallery of only good news
    reads as marketing; the panel is scoring honesty under "Validity".

Run:  python src/i10_media.py       (writes results/figures/gallery_*.png + cover.png)
"""
import sys
import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, DATES, FARMS, FIGURES, RESULTS, RESULTS_AUX, log

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

# one identity per crop, reused everywhere -- the reader learns it once
CCOL = {"Rice": "#3b82f6", "Cotton": "#f59e0b", "Maize": "#ef4444",
        "Bajra": "#a855f7", "Groundnut": "#22c55e"}
NICE = {"20250606": "6 Jun", "20250619": "19 Jun", "20250814": "14 Aug", "20251013": "13 Oct"}


def load():
    g = gpd.read_file(FARMS).to_crs(32643).reset_index(drop=True)
    sub = pd.read_csv(RESULTS / "submission.csv")
    dbg = pd.read_csv(RESULTS / "d4_debug.csv")
    f = pd.read_csv(RESULTS / "farm_features.csv")
    wit = pd.read_csv(RESULTS_AUX / "witness.csv")
    g = g.iloc[: len(sub)].copy()
    for c in ("crop_type", "health_index", "yield_estimate_to_date"):
        g[c] = sub[c].values
    g["conf"] = dbg["crop_confidence"].values
    g["source"] = dbg["source"].values
    return g, sub, dbg, f, wit


def cover(g):
    """The required cover image: the two deliverables, one glance.

    Layout note: geopandas' built-in `legend=True` steals height from its own axes
    for the colourbar, which drops that panel's title below the other one and
    reads as a mistake. The colourbar therefore gets its own axes, so both map
    panels keep an identical box and their titles sit on one baseline.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.9))
    for c in CROPS:
        g[g.crop_type == c].plot(ax=ax[0], color=CCOL[c], edgecolor="white",
                                 linewidth=0.15)
    ax[0].set_title("Crop type", loc="left", pad=8)

    g.plot(ax=ax[1], column="health_index", cmap="RdYlGn", vmin=0, vmax=100,
           edgecolor="white", linewidth=0.15)
    # Say "within crop" on the figure. The diverging ramp centres at 50, so without
    # this a reader sees half the village in red and infers half the village is failing.
    # The score is relative to each crop's own median -- 40 means below par FOR THAT CROP,
    # not failing. A caption is cheaper than a misread map.
    ax[1].set_title("Crop health, 13 October 2025", loc="left", pad=8)
    ax[1].text(0, 1.005, "scored within crop: 50 = that crop's median",
               transform=ax[1].transAxes, fontsize=8.5, color="#666", va="bottom")

    cax = make_axes_locatable(ax[1]).append_axes("right", size="3.5%", pad=0.12)
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 100), "RdYlGn"),
                      cax=cax)
    cb.set_label("health index (0-100)", fontsize=9)
    cax.tick_params(labelsize=8)

    for a in ax:
        a.set_axis_off()
        a.set_aspect("equal")
    # legend under the left map, not on top of the fields it is describing
    ax[0].legend(handles=[Line2D([], [], marker="s", ls="", markersize=9,
                                 markerfacecolor=CCOL[c], markeredgecolor="none",
                                 label=f"{c} ({(g.crop_type == c).sum()})")
                          for c in CROPS],
                 loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5,
                 frameon=False, fontsize=9, handletextpad=0.4, columnspacing=1.4)

    fig.suptitle("Sokhda village, Vadodara — 966 farms from four Capella X-band SAR scenes",
                 fontsize=15, fontweight="bold", y=0.97)
    fig.text(0.5, 0.005, "Kharif 2025 · 6 Jun / 19 Jun / 14 Aug / 13 Oct · X-band HH SLC · "
             "no optical data enters the product", ha="center", fontsize=9.5, color="#555")
    fig.subplots_adjust(top=0.88, bottom=0.10, wspace=0.02)
    fig.savefig(FIGURES / "cover.png")
    plt.close(fig)


def fig_trajectory(f, sub):
    """The physics the crop map rests on: each crop draws a different season."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(4)
    for c in CROPS:
        m = sub.crop_type.values == c
        med = [np.nanmedian(f.loc[m, f"g0_db_{d}"]) for d in DATES]
        ax[0].plot(x, med, "-o", color=CCOL[c], label=c, lw=2)
    ax[0].set_xticks(x); ax[0].set_xticklabels([NICE[d] for d in DATES])
    ax[0].set_ylabel("median gamma0 (dB, uncalibrated)")
    ax[0].set_title("Seasonal backscatter trajectory", loc="left")
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    # The rice signature is the 19 JUNE PEAK. At co-polarised X-band, paddy at the
    # start of the season is dominated by DOUBLE BOUNCE off the stem-water
    # interface, and that rise persists up to ~46 days after transplanting; HH is
    # the favourable polarisation for inundated vegetation. Gujarat transplants
    # mid-to-late June, so 19 Jun sits inside that window and rice is the brightest
    # class on the date. By 14 Aug the canopy has closed and volume scattering plus
    # two-way attenuation suppress the double-bounce path, so rice falls back into
    # the pack -- it is NOT the darkest class then (bajra and groundnut are lower),
    # which is why the earlier "specular in August" caption was wrong.
    ry = np.nanmedian(f.loc[sub.crop_type.values == "Rice", "g0_db_20250619"])
    ax[0].annotate("rice, 19 Jun: double bounce off\nstem + standing water "
                   "(HH-favoured),\npeaks <=46 days after transplanting",
                   xy=(1.0, ry), xytext=(1.35, ry - 0.35), fontsize=8.0,
                   color=CCOL["Rice"], ha="left", va="top",
                   arrowprops=dict(arrowstyle="->", color=CCOL["Rice"], lw=1.2))

    for c in CROPS:
        m = sub.crop_type.values == c
        ax[1].scatter(f.loc[m, "d_aug_jun19"], f.loc[m, "d_oct_aug"], s=7,
                      color=CCOL[c], alpha=0.55, label=c, edgecolors="none")
    ax[1].axhline(0, color="#999", lw=0.7); ax[1].axvline(0, color="#999", lw=0.7)
    # A handful of extreme farms compress the whole population into a strip. The 1-99
    # clip was not enough: d_aug_jun19 has p1 = -24.9 while 90% of farms sit in -4..1.3,
    # so the panel rendered as an empty axis with everything jammed at the right edge.
    # Clip to robust limits and SAY how many points fall outside rather than hiding it.
    xlo, xhi = np.nanpercentile(f.d_aug_jun19, [5, 99.5])
    ylo, yhi = np.nanpercentile(f.d_oct_aug, [1, 99])
    xlo, xhi, ylo, yhi = xlo - 0.4, xhi + 0.4, ylo - 0.4, yhi + 0.4
    off = int((~((f.d_aug_jun19.between(xlo, xhi)) & (f.d_oct_aug.between(ylo, yhi)))).sum())
    ax[1].set_xlim(xlo, xhi); ax[1].set_ylim(ylo, yhi)
    ax[1].text(0.99, 0.02, f"axes clipped to robust limits; {off} of {len(f)} farms outside",
               transform=ax[1].transAxes, ha="right", va="bottom", fontsize=8, color="#777")
    ax[1].set_xlabel("Aug - Jun 19  (geometry-matched pair, dB)")
    ax[1].set_ylabel("Oct - Aug  (canopy retained, dB)")
    ax[1].set_title("The two differences that carry the crop signal", loc="left")
    fig.text(0.5, -0.04, "Differences, not absolute levels: a constant calibration "
             "offset cancels in a difference. [see writeup, Limitations]",
             ha="center", fontsize=9, color="#555")
    fig.savefig(FIGURES / "gallery_06_temporal_trajectory.png")
    plt.close(fig)


def fig_confidence(g):
    """Honesty figure: the map, and where we do not believe it."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.9))
    vmax = float(g.conf.quantile(0.98))
    g.plot(ax=ax[0], column="conf", cmap="viridis", vmin=0, vmax=vmax,
           edgecolor="white", linewidth=0.15)
    ax[0].set_title("Per-farm crop confidence — low BY DESIGN", loc="left", pad=8)
    ax[0].set_axis_off()
    cax = make_axes_locatable(ax[0]).append_axes("right", size="3.5%", pad=0.12)
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, vmax), "viridis"),
                      cax=cax)
    cb.set_label("posterior confidence", fontsize=9)
    cax.tick_params(labelsize=8)

    # Enumerate the provenance values that are ACTUALLY in the data. A hardcoded list
    # silently dropped 52 farms from this panel and its legend when the imputation
    # changed from village-median to spatial-same-crop: the legend read 895 + 0 + 19 =
    # 914 under a title claiming no farm was dropped. Derive, never assume.
    palette = {"measured": "#22c55e", "imputed_spatial_same_crop": "#f59e0b",
               "imputed_village_median": "#f59e0b", "rfi_flagged": "#ef4444"}
    present = [k for k in g.source.value_counts().index]
    cols = {k: palette.get(k, "#64748b") for k in present}
    assert sum((g.source == k).sum() for k in cols) == len(g), "provenance does not cover every farm"
    for k, col in cols.items():
        s = g[g.source == k]
        if len(s):
            s.plot(ax=ax[1], color=col, edgecolor="white", linewidth=0.15)
    ax[1].set_title("Provenance of every row — no farm dropped", loc="left", pad=8)
    ax[1].set_axis_off()
    ax[1].legend(handles=[Line2D([], [], marker="s", ls="", markersize=9,
                                 markerfacecolor=c, markeredgecolor="none",
                                 label=f"{k} ({(g.source == k).sum()})")
                          for k, c in cols.items()],
                 loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
                 frameon=False, fontsize=9, handletextpad=0.4, columnspacing=1.4)
    for a in ax:
        a.set_aspect("equal")
    fig.text(0.5, -0.01, "Missing coverage is spatially CLUSTERED in the north-west, not "
             "random — so imputation borrows from adjacent covered farms of the same crop.",
             ha="center", fontsize=9, color="#333")
    fig.subplots_adjust(bottom=0.09, wspace=0.02)
    fig.savefig(FIGURES / "gallery_04_coverage_and_confidence.png")
    plt.close(fig)


def fig_witness(sub, wit):
    """The strongest result: two sensors that never saw the model agree with it."""
    m = sub.merge(wit, on="farm_id")
    order = m.groupby("crop_type").s2_ndvi_20251013.median().sort_values(
        ascending=False).index.tolist()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, (col, lab) in zip(ax, [("s2_ndvi_20251013", "Sentinel-2 NDVI, 13 Oct (SAME DAY)"),
                                  ("s1_vh_db", "Sentinel-1 C-band VH, 10 Oct (dB)")]):
        data = [m.loc[m.crop_type == c, col].dropna().values for c in order]
        bp = a.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], order):
            patch.set_facecolor(CCOL[c]); patch.set_alpha(0.8); patch.set_edgecolor("#333")
        for med in bp["medians"]:
            med.set_color("#111"); med.set_linewidth(1.6)
        a.set_xticklabels(order, rotation=15)
        a.set_title(lab, loc="left", fontsize=11)
    fig.suptitle("Crop classes assigned from X-band HH alone, scored on two sensors "
                 "they never saw", fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.07,
             "Ordering is the result, not the p-value: on 13 Oct cotton is the only crop still "
             "standing and tops both;\nmaize is harvested and bottoms both. Exactly the crop "
             "calendar.   Kruskal-Wallis p = 9.3e-35 (NDVI), 2.7e-20 (VH).",
             ha="center", fontsize=9, color="#333")
    fig.savefig(FIGURES / "gallery_07_independent_validation.png")
    plt.close(fig)


def fig_robust(f, sub, dbg):
    """Two checks that could have failed: hyperparameter ablation, spatial null."""
    import d4_submission as D4
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors

    crop = sub.crop_type.values
    base = sub.health_index.values

    def sp(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return spearmanr(a[ok], b[ok])[0]

    drops = {}
    for d in list(D4.HEALTH_W):
        w = {k: v for k, v in D4.HEALTH_W.items() if k != d}
        keep, D4.HEALTH_W = D4.HEALTH_W, {k: v / sum(w.values()) for k, v in w.items()}
        drops[d] = sp(base, D4.health_index(f, crop)[0])
        D4.HEALTH_W = keep
    rng = np.random.default_rng(7)
    jit = []
    for _ in range(20):
        keep = D4.HEALTH_W
        p = {k: v * float(rng.uniform(0.5, 1.5)) for k, v in keep.items()}
        D4.HEALTH_W = {k: v / sum(p.values()) for k, v in p.items()}
        jit.append(sp(base, D4.health_index(f, crop)[0]))
        D4.HEALTH_W = keep

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    k = sorted(drops, key=drops.get)
    ax[0].barh(k, [drops[i] for i in k], color="#0ea5e9")
    ax[0].axvline(np.min(jit), color="#ef4444", ls="--", lw=1.4,
                  label=f"worst of 20 x0.5-1.5 weight jitters: {np.min(jit):.3f}")
    # Floor must sit below the WORST drop, else that bar renders as absent rather than
    # as "this component matters most" -- uniform lands at 0.686 and vanished under 0.8.
    ax[0].set_xlim(min(0.8, float(min(drops.values())) - 0.03), 1.0)
    ax[0].set_xlabel("Spearman rho vs the shipped ranking")
    ax[0].set_title("Drop each health component — the ranking survives", loc="left")
    # lower left now sits on top of the "uniform" bar, which the widened floor revealed
    ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")

    gg = gpd.read_file(FARMS).to_crs(32643).reset_index(drop=True)
    xy = np.column_stack([gg.geometry.centroid.x.values, gg.geometry.centroid.y.values])

    def moran(v):
        v = np.asarray(v, "float64"); ok = np.isfinite(v)
        _, idx = NearestNeighbors(n_neighbors=9).fit(xy[ok]).kneighbors(xy[ok])
        idx = idx[:, 1:]; d = v[ok] - v[ok].mean()
        return len(d) * (d[:, None] * d[idx]).sum() / (idx.size * (d ** 2).sum())

    v = dbg.health_raw_z.values
    null = np.array([moran(rng.permutation(v)) for _ in range(199)])
    ax[1].hist(null, bins=35, color="#cbd5e1", edgecolor="none",
               label="199 random permutations")
    ax[1].axvline(moran(v), color="#ef4444", lw=2.2, label=f"observed  I = {moran(v):.3f}")
    ax[1].set_xlabel("Moran's I (8 nearest neighbours)")
    ax[1].set_ylabel("count")
    ax[1].set_title("Health clusters spatially, far beyond chance", loc="left")
    ax[1].legend(frameon=False, fontsize=9)
    fig.text(0.5, -0.05, "Left: no hand-chosen weight is load-bearing.   "
             "Right: neighbouring fields share soil, water and management — "
             "modelling noise would not cluster.",
             ha="center", fontsize=9, color="#333")
    fig.savefig(FIGURES / "gallery_09_robustness.png")
    plt.close(fig)


def fig_negatives(f):
    """The failures, on purpose. Two panels, two different kinds of honest negative."""
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    labels = ["SELF\nJun19 vs\nitself", "NULL\nmis-\nregistered", "REAL\nco-\nregistered",
              "STABLE\nbrightest\n1%", "STABLE-NULL\nfair floor\nfor STABLE",
              "FARMS\nper-farm\nmean"]
    vals = [1.0000, 0.1161, 0.1254, 0.1596, 0.1217, 0.1286]
    cols = ["#22c55e", "#94a3b8", "#ef4444", "#f59e0b", "#94a3b8", "#ef4444"]
    ax[0].bar(range(6), vals, color=cols, width=0.68)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax[0].axhline(0.1161, color="#334155", ls="--", lw=1.2)
    # left of the bars: on the right it overprinted the last bar's value label
    ax[0].text(0.62, 0.128, "bias floor", fontsize=8, color="#334155", ha="left")
    ax[0].set_xticks(range(6))
    ax[0].set_xticklabels(labels, fontsize=7.6)
    ax[0].tick_params(axis="x", length=0)
    ax[0].set_ylim(0, 1.12); ax[0].set_ylabel("coherence")
    ax[0].set_title("Repeat-pass coherence, 19 Jun x 14 Aug\nsits at the noise floor",
                    loc="left", fontsize=11)

    # Binned medians, not a scatter. 966 semi-transparent points is a blob in which
    # a rho of -0.34 and a rho of -0.05 look identical -- which would hide the
    # entire point of the panel. Deciles of the x variable make both visible.
    sub = pd.read_csv(RESULTS / "submission.csv")
    mm = sub.merge(f[["farm_id", "cv_20250814", "cv_20251013"]], on="farm_id")
    for col, c, mk, lab in [
            ("cv_20250814", "#94a3b8", "o", "August CV — CIRCULAR, it IS the input"),
            ("cv_20251013", "#ef4444", "s", "October CV — independent, FAILS")]:
        ok = np.isfinite(mm[col]) & np.isfinite(mm.health_index)
        x, y = mm[col][ok].values, mm.health_index[ok].values
        r = spearmanr(x, y)[0]
        q = np.quantile(x, np.linspace(0, 1, 11))
        cx = [np.median(x[(x >= q[i]) & (x <= q[i + 1])]) for i in range(10)]
        cy = [np.median(y[(x >= q[i]) & (x <= q[i + 1])]) for i in range(10)]
        ax[1].plot(cx, cy, mk + "-", color=c, lw=2.2, markersize=6.5,
                   label=f"{lab}   rho = {r:+.3f}")
    ax[1].set_xlabel("within-farm coefficient of variation  (decile bins)")
    ax[1].set_ylabel("median health index")
    ax[1].legend(frameon=False, fontsize=8.8, loc="lower left")
    ax[1].set_title('"Uniform canopy scores higher" holds only\n'
                    'where the test is circular', loc="left", fontsize=11)

    fig.text(0.5, -0.10,
             "Left: the controls prove the estimator works (SELF = 1.00), but the stable-target "
             "control does not clear its own floor — so this is reported\nas UNINFORMATIVE, not "
             "as measured crop decorrelation.    Right: the relationship the brief asks for is "
             "present on the date that feeds the\nindex and absent on an independent date. "
             "Reported as a failure rather than quoted from the circular version.",
             ha="center", fontsize=8.8, color="#333")
    fig.subplots_adjust(wspace=0.24)
    fig.savefig(FIGURES / "gallery_10_negatives.png")
    plt.close(fig)


def fig_yield(g, sub, dbg):
    """Farm -> village aggregation, stated as a rule and shown as a table.

    "Aggregation from farm-level to village-level is logical and clearly defined" is its
    own rubric line, so the rule is written on the figure: village production is the
    AREA-WEIGHTED sum of per-farm yield (t/ha x ha), never a mean of per-hectare rates,
    which would let a 0.05 ha plot count as much as a 5 ha one. The yield map used to
    occupy the left half; it is now its own required gallery item, so that space goes to
    the summary table instead of repeating it.
    """
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1], wspace=0.30)
    a0 = fig.add_subplot(gs[0])
    a0.set_axis_off()

    m0 = sub.merge(dbg[["farm_id", "area_ha"]], on="farm_id")
    m0["prod_t"] = m0.yield_estimate_to_date * m0.area_ha
    rows = [["crop", "farms", "area ha", "median health", "median t/ha", "production t"]]
    for c in CROPS:
        s = m0[m0.crop_type == c]
        rows.append([c, f"{len(s)}", f"{s.area_ha.sum():,.1f}",
                     f"{s.health_index.median():.1f}",
                     f"{s.yield_estimate_to_date.median():.2f}", f"{s.prod_t.sum():,.0f}"])
    rows.append(["VILLAGE 1", f"{len(m0)}", f"{m0.area_ha.sum():,.1f}",
                 f"{m0.health_index.median():.1f}",
                 f"{m0.yield_estimate_to_date.median():.2f}", f"{m0.prod_t.sum():,.0f}"])
    t = a0.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="right",
                 colWidths=[0.22, 0.12, 0.16, 0.22, 0.18, 0.20])
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.7)
    for (r, c_), cell in t.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if r == 0:
            cell.set_facecolor("#f1f5f9"); cell.set_text_props(fontweight="bold")
        elif r == len(rows) - 1:
            cell.set_facecolor("#e0f2fe"); cell.set_text_props(fontweight="bold")
        if c_ == 0 and r > 0:
            cell.set_text_props(ha="left")
    a0.set_title("Village-level summary — all 966 farms, none dropped", loc="left", pad=16)
    a0.text(0, 0.08, "Rule: village production = Σ(farm yield t/ha × farm area ha).\n"
            "Area-weighted, never a mean of per-hectare rates.",
            transform=a0.transAxes, fontsize=9, color="#334155")

    a1 = fig.add_subplot(gs[1])
    m = sub.merge(dbg[["farm_id", "area_ha"]], on="farm_id")
    # yield is already TONNES/ha (host schema unit), so t/ha x ha = tonnes directly
    m["prod_t"] = m.yield_estimate_to_date * m.area_ha
    agg = m.groupby("crop_type").agg(ha=("area_ha", "sum"), t=("prod_t", "sum"))
    agg = agg.reindex(CROPS)
    a1.bar(agg.index, agg.t, color=[CCOL[c] for c in agg.index])
    for i, (c, r) in enumerate(agg.iterrows()):
        a1.text(i, r.t + 4, f"{r.t:.0f} t\n{r.ha:.0f} ha", ha="center", fontsize=8.5)
    a1.set_ylabel("production to date (tonnes)")
    a1.set_title("Village aggregate — one rule, all 966 farms", loc="left", pad=8)
    a1.set_ylim(0, agg.t.max() * 1.25)
    fig.text(0.5, -0.02, "Cotton reads low per hectare because it is only ~45% through "
             "picking on 13 Oct — this is yield TO DATE, not final yield.",
             ha="center", fontsize=9, color="#333")
    fig.savefig(FIGURES / "gallery_05_village_aggregate.png")
    plt.close(fig)


def _required_map(g, column, cmap, vmin, vmax, title, sub_note, cbar_label,
                  hist_label, fname, fmt="{:.0f}", panel="box"):
    """One of the two maps the brief requires by name, as a standalone figure.

    The guidelines list "a farm-level Health Index map (colour coded)" and "a farm-level
    Yield Estimate to Date map (colour coded)" as separate Media Gallery items. Ours
    previously existed only as panels inside two-panel composites, which is a judge
    hunting a checklist item across figures. Each now stands alone, and carries its own
    distribution so the map and the spread are read together rather than in two places.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.45, 1], wspace=0.16)
    axm, axh = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    v = g[column].values
    g.plot(ax=axm, column=column, cmap=cmap, vmin=vmin, vmax=vmax,
           edgecolor="white", linewidth=0.15)
    axm.set_axis_off(); axm.set_aspect("equal")
    axm.set_title(title, loc="left", pad=10, fontsize=13)

    # Colourbar goes UNDER the map, not beside it: on the right it sat between the two
    # panels and its label clipped the crop names on the distribution axis.
    cax = make_axes_locatable(axm).append_axes("bottom", size="3.2%", pad=0.14)
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin, vmax), cmap),
                      cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=9)
    cax.tick_params(labelsize=8)

    # a scale bar, because a map without one is a picture
    x0, x1, y0, y1 = *axm.get_xlim(), *axm.get_ylim()
    bar = 500.0
    bx, by = x0 + 0.06 * (x1 - x0), y0 + 0.05 * (y1 - y0)
    axm.plot([bx, bx + bar], [by, by], color="#111", lw=2.4, solid_capstyle="butt")
    axm.text(bx + bar / 2, by + 0.012 * (y1 - y0), "500 m", ha="center",
             fontsize=8.5, color="#111")

    order = [c for c in CROPS if (g.crop_type == c).sum() >= 5]
    if panel == "box":
        # yield differs in LEVEL between crops, so a by-crop box is the informative view
        data = [g.loc[g.crop_type == c, column].dropna().values for c in order]
        bp = axh.boxplot(data, vert=False, patch_artist=True, widths=0.62,
                         medianprops=dict(color="#111", lw=1.5),
                         flierprops=dict(marker=".", ms=3, mfc="#94a3b8", mec="none"))
        for patch, c in zip(bp["boxes"], order):
            patch.set(facecolor=CCOL[c], alpha=0.8, edgecolor="#475569", lw=0.8)
        axh.set_yticks(range(1, len(order) + 1))
        axh.set_yticklabels([f"{c}  n={(g.crop_type == c).sum()}" for c in order],
                            fontsize=8.5)
        axh.set_title("Distribution by crop", loc="left", fontsize=11)
        for i, c in enumerate(order):
            med = float(np.nanmedian(g.loc[g.crop_type == c, column]))
            axh.text(med, i + 1.34, fmt.format(med), ha="center", fontsize=8,
                     color="#334155", fontweight="bold")
    else:
        # Health is scored WITHIN crop, so every crop's median is 50 by construction and a
        # by-crop box would look broken rather than informative. The honest view is the
        # pooled shape, with that fact stated on the axes instead of hidden.
        axh.hist(g[column].dropna().values, bins=28, color="#64748b",
                 edgecolor="white", linewidth=0.6)
        axh.axvline(50, color="#b91c1c", lw=1.6, ls="--")
        # In axes coords, top-left: anchored at the line it ran off the right edge.
        axh.text(0.03, 0.97, "each crop centred at 50\nby construction",
                 transform=axh.transAxes, fontsize=8.5, color="#b91c1c",
                 va="top", ha="left")
        axh.set_ylabel("farms", fontsize=9)
        axh.set_title("Distribution, all 966 farms", loc="left", fontsize=11)
    axh.set_xlabel(hist_label, fontsize=9)
    axh.grid(axis="x", color="#e2e8f0", lw=0.7)
    axh.set_axisbelow(True)

    fig.suptitle("Sokhda village (village_id 1), Vadodara — 966 farms, Capella X-band HH SAR",
                 fontsize=12.5, fontweight="bold", y=1.0, x=0.02, ha="left")
    # The explanatory note is a full sentence, so it belongs under the figure as a caption.
    # Above the axes it wrapped to three lines and collided with the suptitle.
    fig.text(0.02, -0.035, "\n".join(textwrap.wrap(sub_note, 130)), ha="left",
             va="top", fontsize=9, color="#555")
    p = FIGURES / fname
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_method(_g=None):
    """One diagram of the whole method, including what is deliberately NOT connected.

A judge reading a gallery cannot reconstruct a dataflow from prose. The single most
load-bearing claim in this project -- that Sentinel-1 and Sentinel-2 never touch the
product -- is a statement about ARROWS, so it belongs in a diagram: the witness box
has no arrow into any deliverable, only into validation.

Layout note: the declared-input arrows are ELBOWS routed through the clear band at
y=52, between the feature box and the deliverables. Drawn straight they cut diagonally
across the feature box and through the deliverable headings.
"""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13.6, 8.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_axis_off()

    def box(x, y, w, h, head, body, fc, ec, hs=10, bs=8.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    facecolor=fc, edgecolor=ec, linewidth=1.5))
        if head:
            ax.text(x + w / 2, y + h - 2.2, head, ha="center", va="center",
                    fontsize=hs, fontweight="bold", color="#0f172a")
        if body:
            ax.text(x + w / 2, y + (h - 4.4) / 2 if head else y + h / 2, body,
                    ha="center", va="center", fontsize=bs, color="#1e293b", linespacing=1.5)

    def arrow(x1, y1, x2, y2, color="#475569", lw=1.6, style="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color=color, linewidth=lw, linestyle=style,
                                     shrinkA=1, shrinkB=1))

    def elbow(x1, y1, x2, y2, ymid, color, lw=1.6):
        ax.plot([x1, x1], [y1, ymid], color=color, lw=lw, solid_capstyle="round", zorder=1)
        ax.plot([x1, x2], [ymid, ymid], color=color, lw=lw, solid_capstyle="round", zorder=1)
        arrow(x2, ymid, x2, y2, color=color, lw=lw)

    PRIMARY, AUX, OUT, WIT = "#e0f2fe", "#fef3c7", "#dcfce7", "#f1f5f9"
    EP, EA, EO, EW = "#0284c7", "#d97706", "#16a34a", "#94a3b8"

    # ---- primary inputs --------------------------------------------------------
    ax.text(2, 96.5, "PRIMARY SOURCE — Capella X-band HH SLC", fontsize=11,
            fontweight="bold", color="#0369a1")
    ax.text(2, 93.4, "the only source of any shipped number", fontsize=9, color="#0369a1")
    for i, d in enumerate(DATES):
        box(2 + i * 15.8, 84, 14.5, 7, None, NICE[d], PRIMARY, EP, bs=10.5)
        arrow(9.2 + i * 15.8, 84, 33, 78.5)

    ax.text(98, 96.5, "DECLARED INPUTS", fontsize=11, fontweight="bold",
            color="#b45309", ha="right")
    ax.text(98, 93.4, "used, and labelled as inputs", fontsize=9, color="#b45309", ha="right")
    box(68, 84, 30, 7, None, "Round 1 crop-area shares\nvillage mix, MSE 11.071", AUX, EA)
    box(68, 74, 30, 7, None, "Vadodara APY district yield\npublished statistics", AUX, EA)

    # ---- processing chain ------------------------------------------------------
    box(2, 68, 63, 10, "PREPROCESSING",
        "β⁰ = scale·|z|²  →  γ⁰ = β⁰·tanθ  with PER-PIXEL θ from orbit state vectors\n"
        "geocode to EPSG:32643 with 225 GCPs · averaging = multi-look · 5 m + 2 m grids",
        PRIMARY, EP)
    arrow(33, 68, 33, 64)

    box(2, 52, 63, 12, "PER-FARM FEATURES",
        "parcel eroded before sampling so boundary pixels never mix two fields\n"
        "per-date γ⁰ · inter-date differences · within-farm CV · season integral · texture\n"
        "coverage 966/966 — 895 measured, 52 imputed, 19 RFI-flagged, provenance per row",
        PRIMARY, EP)

    # ---- deliverables ----------------------------------------------------------
    # opaque background: this label sits in the band the feed arrows cross, and without
    # it the text and the arrowheads overprint each other
    ax.text(50, 48.6, "THREE DELIVERABLES  ·  submission.csv, one row per farm",
            fontsize=11, fontweight="bold", color="#15803d", ha="center", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=3.0))
    for x in (17, 50, 83):
        arrow(33, 52, x, 46.4)
    # declared inputs routed through the clear band, not diagonally across the figure
    elbow(83, 84, 17, 46.4, 50.6, EA)
    elbow(83, 74, 83, 46.4, 50.6, EA)

    box(2, 30, 30, 16, "crop_type",
        "soft per-farm evidence, then\nbiased until AREA-weighted\nshares match Round 1\n"
        "— argmax only at the end", OUT, EO)
    box(35, 30, 30, 16, "health_index",
        "four families, weights from\nredundancy  w ∝ 1/Σ|ρ|\n(blind to every witness)\n"
        "scored WITHIN crop", OUT, EO)
    box(68, 30, 30, 16, "yield_estimate_to_date",
        "anchor × completion(farm)\n× accumulation(farm)\nobserved to 13 Oct —\n"
        "NOT a harvest forecast", OUT, EO)

    # ---- witnesses: the point is the missing arrow ------------------------------
    box(2, 6, 44, 17, "WITNESSES — never inputs",
        "Sentinel-2 L2A · Sentinel-1 RTC · NASA POWER\n\n"
        "same-day NDVI, 13 Oct, 0.003% cloud · C-band VH\n"
        "10-scene season integral, 12 Jun – 10 Oct, one orbit", WIT, EW)
    box(54, 6, 44, 17, "VALIDATION — tests that can fail",
        "crop separation p = 9.3×10⁻³⁵ · Moran's I = 0.096\n\n"
        "and the failures, reported: κ = 0.112 ·\ncoherence at the noise floor ·\n"
        "a witness caught an inverted completion sign", WIT, EW)
    arrow(46, 14.5, 54, 14.5, color="#64748b")
    for x in (17, 50, 83):
        arrow(x, 30, 76, 23, color="#94a3b8", lw=1.1, style=(0, (4, 3)))

    ax.text(50, 1.6, "No arrow runs from a witness into a deliverable — that is the design.",
            fontsize=11, fontweight="bold", color="#b91c1c", ha="center")

    fig.suptitle("Method — what feeds what, and what deliberately does not",
                 fontsize=14.5, fontweight="bold", y=1.02, x=0.02, ha="left")
    # SUPERSEDED. The gallery ships a hand-designed version of this diagram, which lives at
    # gallery_00_method_overview.png. This function keeps the layout reproducible from code,
    # so it writes to its own filename rather than clobbering the shipped asset on a re-run.
    p = FIGURES / "gallery_00_method_overview_generated.png"
    fig.savefig(p, bbox_inches="tight")
    # Also as SVG -- XML, so the diagram can be restyled by hand in Inkscape/Illustrator/
    # Figma without re-running the pipeline. svg.fonttype="none" keeps the labels as real
    # <text> elements (the default converts them to outlines, which cannot be edited).
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return p


def thumbnail(g):
    """The Kaggle card image, at EXACTLY 560x280 px.

    A card is read at thumbnail size, so this is not a shrunken cover: it carries two
    maps and four words. Anything smaller than ~7pt is unreadable on the card, so the
    only text is the label strip. Exact pixel size comes from figsize x dpi (5.6 x 2.8
    at 100 dpi) with no bbox_inches="tight", which would crop to content and change it.
    """
    fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
    fig.patch.set_facecolor("#0f172a")
    axl = fig.add_axes([0.015, 0.06, 0.47, 0.72])
    axr = fig.add_axes([0.505, 0.06, 0.47, 0.72])
    for c in CROPS:
        g[g.crop_type == c].plot(ax=axl, color=CCOL[c], edgecolor="none")
    g.plot(ax=axr, column="health_index", cmap="RdYlGn", vmin=0, vmax=100, edgecolor="none")
    for a, lab in ((axl, "CROP TYPE"), (axr, "HEALTH INDEX")):
        a.set_axis_off(); a.set_aspect("equal")
        a.set_facecolor("#0f172a")
        a.text(0.5, -0.04, lab, transform=a.transAxes, ha="center", va="top",
               fontsize=8, color="#cbd5e1", fontweight="bold")
    fig.text(0.5, 0.93, "966 FARMS FROM X-BAND SAR ALONE", ha="center", va="center",
             fontsize=12.5, color="white", fontweight="bold")
    fig.text(0.5, 0.845, "Sokhda, Gujarat  ·  kharif 2025  ·  4 Capella scenes",
             ha="center", va="center", fontsize=7.5, color="#94a3b8")
    p = FIGURES / "thumbnail_560x280.png"
    # rcParams set savefig.dpi=160 and savefig.bbox="tight" for the gallery. Passing
    # bbox_inches=None to savefig still falls back to the rcParam, so the override has to
    # happen in a context, or the card comes out 836x466 / 522x290 instead of 560x280.
    with mpl.rc_context({"savefig.bbox": None, "savefig.dpi": 100}):
        fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)
    # Kaggle states the card size exactly, so verify rather than trust the arithmetic.
    from PIL import Image
    with Image.open(p) as im:
        if im.size != (560, 280):
            im.convert("RGB").resize((560, 280), Image.LANCZOS).save(p)
    with Image.open(p) as im:
        assert im.size == (560, 280), f"thumbnail is {im.size}, must be 560x280"
    return p


def fig_map_crop(g):
    """Supporting map the brief names: crop classification, carried from Round 1.

    Kept separate from the two required maps so each gallery item answers exactly one
    question. The area table is on the figure because the Round-1 crop mix is what
    constrains this map -- the shares are an input, not an outcome, and saying so on the
    figure is more honest than letting a reader read them as an independent result.
    """
    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.06)
    axm, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for c in CROPS:
        g[g.crop_type == c].plot(ax=axm, color=CCOL[c], edgecolor="white", linewidth=0.15)
    axm.set_axis_off(); axm.set_aspect("equal")
    axm.set_title("Crop classification — Round 1 mix applied to the new boundaries",
                  loc="left", pad=10, fontsize=13)
    axm.legend(handles=[Line2D([], [], marker="s", ls="", markersize=9,
                               markerfacecolor=CCOL[c], markeredgecolor="none", label=c)
                        for c in CROPS],
               loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=5, frameon=False,
               fontsize=9.5, handletextpad=0.4, columnspacing=1.4)

    axt.set_axis_off()
    tot = float(g.area.sum()) / 1e4
    rows = [["crop", "farms", "area ha", "area %"]]
    for c in CROPS:
        s = g[g.crop_type == c]
        rows.append([c, f"{len(s)}", f"{s.area.sum()/1e4:,.1f}", f"{100*s.area.sum()/1e4/tot:.1f}%"])
    rows.append(["all", f"{len(g)}", f"{tot:,.1f}", "100.0%"])
    # explicit widths: the default squeezed "Groundnut" to "Groundn"
    t = axt.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="right",
                  colWidths=[0.34, 0.20, 0.24, 0.22])
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.55)
    for (r, c_), cell in t.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if r == 0:
            cell.set_facecolor("#f1f5f9"); cell.set_text_props(fontweight="bold")
        elif r == len(rows) - 1:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor("#f8fafc")
        if c_ == 0 and r > 0:
            cell.set_text_props(ha="left")
    # sits lower than the map title, which it otherwise collides with
    axt.text(0, 0.80, "Area by crop", transform=axt.transAxes, fontsize=11,
             fontweight="bold")
    axt.text(0, -0.02, "Round-1 area shares are an INPUT constraint,\nnot an independent result.",
             transform=axt.transAxes, fontsize=8.5, color="#666", va="top")

    fig.suptitle("Sokhda village (village_id 1), Vadodara — 966 farms, Capella X-band HH SAR",
                 fontsize=12.5, fontweight="bold", y=0.98, x=0.02, ha="left")
    p = FIGURES / "gallery_03_crop_classification_map.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_map_health(g):
    return _required_map(
        g, "health_index", "RdYlGn", 0, 100,
        "Farm-level Crop Health Index — 13 October 2025",
        "scored WITHIN crop: 50 = that crop's own median, so colour compares like with like",
        "health index (0-100)", "health index (0-100)",
        "gallery_01_health_index_map.png", panel="hist")


def fig_map_yield(g):
    return _required_map(
        g, "yield_estimate_to_date", "YlGnBu", 0.0,
        float(np.nanpercentile(g.yield_estimate_to_date, 99)),
        "Farm-level Yield Estimate to Date — as observed to 13 October 2025",
        "accumulated to the final acquisition, NOT a final harvest forecast. Level is set by crop "
        "(cotton is only ~45% through picking, so it reads pale) — the panel at right shows the "
        "within-crop spread, which is what the SAR contributes.",
        "yield to date (t/ha)", "yield to date (t/ha)",
        "gallery_02_yield_to_date_map.png", fmt="{:.2f}")


def fig_season_witness(sub):
    """The season C-band saw and optical did not, split by crop.

    Two things at once: the per-crop phenology that justifies the matched witness, and
    the reason it had to be radar. Optical usable-scene counts come from the same
    measurement `why_xband.py` makes, so the two figures cannot drift apart.
    """
    sp = RESULTS_AUX / "witness_season.csv"
    if not sp.exists():
        return None
    w = pd.read_csv(sp)
    cols = [c for c in w.columns if c.startswith("s1_vh_db_2025")]
    dates = pd.to_datetime([c.split("_")[-1] for c in cols])
    m = sub.merge(w, on="farm_id")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3),
                           gridspec_kw={"width_ratios": [1.35, 1]})
    colours = {"Cotton": "#0ea5e9", "Rice": "#16a34a", "Maize": "#f59e0b",
               "Bajra": "#a855f7", "Groundnut": "#ef4444"}
    for c in CROPS:
        s = m[m.crop_type == c]
        if len(s) < 20:
            continue
        ax[0].plot(dates, [s[col].median() for col in cols], "-o", ms=3.5, lw=1.6,
                   color=colours.get(c, "#666"), label=f"{c} (n={len(s)})")
    ax[0].set_ylabel("Sentinel-1 C-band VH (dB), farm median")
    ax[0].set_title("The season C-band observed — 10 scenes, one orbit", loc="left")
    ax[0].legend(frameon=False, fontsize=8.5, ncol=2)
    ax[0].tick_params(axis="x", rotation=30)

    # The optical record over the same window, as counted by why_xband.py
    opt = RESULTS_AUX / "why_xband_optical.csv"
    if opt.exists():
        o = pd.read_csv(opt)
        x = np.arange(len(o))
        ax[1].bar(x - 0.2, o.scenes, 0.4, label="S2 revisits", color="#cbd5e1")
        ax[1].bar(x + 0.2, o.usable, 0.4, label="usable (<20% cloud)", color="#0ea5e9")
        ax[1].set_xticks(x); ax[1].set_xticklabels([mm[-2:] for mm in o.month])
        ax[1].set_xlabel("month of 2025")
        ax[1].set_ylabel("scenes")
        ax[1].set_ylim(0, max(o.scenes) * 1.45)
        ax[1].legend(frameon=False, fontsize=9, loc="upper left")
        ax[1].set_title("The same season, optically: nothing until October", loc="left")
        ax[1].text(0.5, 0.62, "0 usable scenes\nJun–Sep", transform=ax[1].transAxes,
                   ha="center", fontsize=11, color="#b91c1c", fontweight="bold")

    # The rotated date labels on the left panel occupy the strip just under the axes,
    # so the caption has to clear them rather than sit at the usual -0.04.
    fig.text(0.5, -0.16, "Why the yield accumulation term is witnessed in C-band and not "
             "in cumulative NDVI: the optical record for the accumulation period is empty.",
             ha="center", fontsize=9, color="#555")
    p = FIGURES / "gallery_08_season_witness.png"
    fig.savefig(p); plt.close(fig)
    return p


def main():
    log("i10.start")
    g, sub, dbg, f, wit = load()
    cover(g);                       log("i10.fig", name="cover")
    fig_method();                   log("i10.fig", name="00_method_overview")
    # the two maps the brief requires by name, before the supporting ones
    fig_map_health(g);              log("i10.fig", name="01_health_index_map")
    fig_map_yield(g);               log("i10.fig", name="02_yield_to_date_map")
    fig_map_crop(g);                log("i10.fig", name="03_crop_classification_map")
    thumbnail(g);                   log("i10.fig", name="thumbnail_560x280")
    fig_trajectory(f, sub);         log("i10.fig", name="1_trajectory")
    fig_confidence(g);              log("i10.fig", name="2_confidence")
    fig_witness(sub, wit);          log("i10.fig", name="3_witness")
    fig_robust(f, sub, dbg);        log("i10.fig", name="4_robustness")
    fig_negatives(f);               log("i10.fig", name="5_negatives")
    fig_yield(g, sub, dbg);         log("i10.fig", name="6_yield")
    if fig_season_witness(sub):     log("i10.fig", name="8_season_witness")
    out = sorted(p.name for p in FIGURES.glob("cover.png")) + \
        sorted(p.name for p in FIGURES.glob("gallery_*.png"))
    log("i10.done", figures=len(out))
    print("\n".join(str(FIGURES / o) for o in out))


if __name__ == "__main__":
    main()
