"""Paper figures 1 to 5 (XAI and validation). Reads the frozen R3 outputs; writes nothing into R3.

  Fig 1  date removal test, with the 29 Oct known answer control
  Fig 2  exact per farm attribution of the health index (one flagged, one healthy farm)
  Fig 3  spatial validity: location (variogram, variance split) and outline (boundary pixels)
  Fig 4  speckle stability of the attention list
  Fig 5  village uncertainty breakdown and conformal coverage arithmetic

Every figure is checked after drawing: no hyphen or dash characters in any visible text, and
no two visible text boxes overlapping. The run stops if either check fails.

Run:  py -3.12 make_xai_figs.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

PLAIN = FuncFormatter(lambda x, _: f"{x:g}")   # 1, 10, 100 instead of 10^n with a minus sign

OUT = Path(__file__).resolve().parent
R3 = OUT.parents[2] / "phase3_yield_forecast"
sys.path.insert(0, str(R3 / "src"))
from p3_build import health_parts, derive_weights, health_index  # shipped code, not a copy

T = R3 / "results" / "tables"
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.unicode_minus": True,
                     "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5})
GREEN, ORANGE, GREY, DARK, BLUE, RED = "#2f855a", "#c05621", "#a0aec0", "#4a5568", "#2b6cb0", "#9b2c2c"
LABEL_EXTERNAL_T = 236.5   # village label term across five other teams' labels (R3 WRITEUP 7)


def sgn(x, nd=2):
    """Signed number with a true minus sign, so no hyphen ever reaches a figure."""
    return f"{x:+.{nd}f}".replace("-", "−")


def check_and_save(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # tick labels outside the view interval exist as objects but are never drawn: skip them
    ticks, drawn = set(), set()
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                ticks.update((id(tk.label1), id(tk.label2)))
            for tk in axis._update_ticks():
                drawn.update((id(tk.label1), id(tk.label2)))
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
             and (id(t) not in ticks or id(t) in drawn)]
    bad = [t.get_text() for t in texts if any(c in t.get_text() for c in "-–—")]
    assert not bad, f"{name}: hyphen or dash in {bad}"
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in texts]
    clash = [(a, b) for (a, ba), (b, bb) in itertools.combinations(boxes, 2) if ba.overlaps(bb)]
    assert not clash, f"{name}: overlapping text {clash}"
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{name}: {len(texts)} texts, no dashes, no overlaps")


def fig1():
    g = pd.read_csv(T / "p3_voi_greedy.csv")
    l = pd.read_csv(T / "p3_voi_lodo.csv")
    s = pd.read_csv(T / "p3_voi_stats.csv").set_index("test")
    c1 = s.loc["C1_drop_29oct"]
    assert c1.rho == 1.0 and c1.moved == 0          # the control must hold before we draw it

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    lab = ["29 Oct\n(control)"] + [pd.Timestamp(str(d)).strftime("%d %b") for d in l.date]
    moved = [0] + list(l.moved_of_97)
    rho = [1.0] + list(l.rho)
    col = [BLUE] + [ORANGE if m > 50 else GREY for m in l.moved_of_97]
    ax[0].bar(lab, moved, color=col)
    for i, (m, r_) in enumerate(zip(moved, rho)):
        ax[0].text(i, m + 1.5, f"{m}\nρ = {r_:.3f}", ha="center", fontsize=9)
    ax[0].text(0, 17, "Known answer:\nρ = 1, 0 moved", ha="center", fontsize=8, color=BLUE)
    ax[0].set_ylim(0, 95)
    ax[0].set_ylabel("Farms leaving the 97 farm attention list")
    ax[0].set_title("(a) Leave one date out and rebuild the index")
    ax[0].grid(alpha=0.3, axis="y")

    ax[1].plot(g.n_dates, g.rho, "o-", color=GREEN, lw=2.2, ms=7, label="Real calendar (greedy order)")
    for _, r_ in g.iterrows():
        ax[1].annotate(pd.Timestamp(str(r_.added)).strftime("%d %b"), (r_.n_dates, r_.rho),
                       textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8.5)
    q = s.loc["Q3_calendar_permutation"]
    ax[1].axhspan(q.lo, q.hi, color="#fed7d7", alpha=0.6, zorder=0,
                  label="Shuffled calendar, 5th to 95th percentile")
    ax[1].axhline(q.rho, color=RED, ls="--", lw=1.5, label=f"Shuffled calendar, mean ρ = {q.rho:.2f}")
    ax[1].set_xticks(range(1, 6))
    ax[1].set_ylim(-0.12, 1.1)
    ax[1].set_xlabel("Number of acquisitions used")
    ax[1].set_ylabel("Spearman ρ with the full index")
    ax[1].set_title("(b) Adding dates vs destroying the time axis")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    check_and_save(fig, "fig1_date_removal.png")


def fig2():
    f = pd.read_csv(T / "p1_farm_features.csv")
    lab = pd.read_csv(R3 / "data_aux" / "inherited" / "label_dist.csv")[["farm_id", "crop_type"]]
    d = f.merge(lab, on="farm_id")
    parts = health_parts(d, use_29oct=False)
    W = derive_weights(parts)
    hi = health_index(parts, W, d.crop_type)

    # control: must reproduce the frozen submission's health index
    sub = pd.read_csv(R3 / "results" / "submission.csv").set_index("farm_id")
    assert np.allclose(np.round(hi, 2), sub.loc[d.farm_id, "health_index"].to_numpy(), atol=0.011)

    sp = pd.read_csv(T / "p3_speckle_farm.csv").set_index("farm_id").loc[d.farm_id]
    e = pd.read_csv(T / "p3_edge_farm.csv").set_index("farm_id").loc[d.farm_id]
    P = pd.DataFrame(parts)
    cot = (d.crop_type == "Cotton").to_numpy()
    ref = P[cot].median()                          # typical cotton farm, part by part
    contrib = (P - ref) * pd.Series(W)             # exact: sums to S - S_ref
    S = sum(W[k] * P[k] for k in W)
    assert np.allclose(contrib.sum(axis=1), S - (ref * pd.Series(W)).sum())

    # Farm A: listed cotton farm robust to both artefacts of WRITEUP 9.8 and 9.10 (listed in
    # every speckle replicate AND still listed on interior pixels alone), smallest boundary
    # share. Uniformity is the edge inflated part, so this guard matters.
    # Farm B: top decile cotton farm with the smallest speckle spread.
    listed = (cot & sp.on_attention_list.to_numpy() & (sp["retain_F1.0"].to_numpy() == 1.0)
              & e.stays_core_only.to_numpy().astype(bool))
    a = int(np.flatnonzero(listed)[np.argmin(e.ring_frac.to_numpy()[listed])])
    top = cot & (hi >= np.nanpercentile(hi[cot], 90))
    b = int(np.flatnonzero(top)[np.argmin(sp.index_sd_F1.to_numpy()[top])])

    names = {"level": "Level\n(14 Aug)", "growth": "Growth\n(Jun → Aug)", "uniform": "Uniformity\n(14 Aug)",
             "persist": "Persistence\n(season)", "senesce": "Senescence\n(Oct → Nov)"}
    order = list(names)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, sharex=True)
    for k, (i, tag) in enumerate(((a, "Farm A, on the attention list"), (b, "Farm B, top 10% of cotton"))):
        v = contrib.iloc[i][order].to_numpy()
        ax[k].barh([names[o] for o in order], v, color=np.where(v < 0, "#c53030", GREEN))
        for y, x in enumerate(v):
            ax[k].text(x + (0.05 if x >= 0 else -0.05), y, sgn(x), va="center",
                       ha="left" if x >= 0 else "right", fontsize=9)
        ax[k].axvline(0, color="black", lw=0.8)
        ax[k].set_title(f"({'ab'[k]}) {tag}\nHealth index {hi[i]:.1f} of 100, total {sgn(v.sum())}")
        ax[k].set_xlabel("Contribution relative to a typical cotton farm (weight × part)")
        ax[k].grid(alpha=0.3, axis="x")
    ax[0].set_xlim(-1.6, 1.6)
    ax[0].invert_yaxis()
    w = ", ".join(f"{k} {W[k]:.3f}" for k in order)
    fig.text(0.5, -0.04, f"Weights from feature redundancy only: {w}. "
             "Bars sum exactly to the farm's score minus the typical farm's score.",
             ha="center", fontsize=8.5, color=DARK)
    fig.tight_layout()
    check_and_save(fig, "fig2_farm_attribution.png")
    print("  Farm A index", round(hi[a], 2), "boundary share", round(e.ring_frac.iloc[a], 3),
          "| Farm B index", round(hi[b], 2))


def fig3():
    v = pd.read_csv(T / "p3_spatial_variogram.csv")
    s = pd.read_csv(T / "p3_spatial_stats.csv").set_index("stat")
    fm = pd.read_csv(T / "p3_edge_farm.csv")
    st = pd.read_csv(T / "p3_edge_stats.csv").iloc[0]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.4))
    ax = ax.ravel()

    style = {"hi_resid": (GREEN, "o-", "Health index ranking"),
             "incidence": (ORANGE, "s--", "Incidence angle (positive control)"),
             "jun_baseline": (GREY, "^:", "June bare soil (control)")}
    for name, (col, mk, lab) in style.items():
        q = v[v.series == name]
        ax[0].plot(q.lag_m, q.gamma, mk, color=col, lw=2, ms=5, label=lab)
    ax[0].axhline(1.0, color="#cbd5e0", lw=1, zorder=0)
    ax[0].set_xlabel("Distance between farm centroids (m)")
    ax[0].set_ylabel("Semivariance (standardised)")
    ax[0].set_ylim(0, v.gamma.max() * 1.08)
    ax[0].set_title("(a) Variogram: the ranking has no spatial structure")
    ax[0].legend(frameon=False, loc="lower right")
    ax[0].grid(alpha=0.3)

    land = float(s.loc["variance_split", "landscape"])
    farm = float(s.loc["variance_split", "farm_specific"])
    m = s.loc["moran_raw"]
    ax[1].barh(["Neighbourhood\n(location)", "Farm itself"], [land, farm], color=[GREY, GREEN])
    for i, val in enumerate([land, farm]):
        ax[1].text(val + 0.02, i, f"{val:.0%}", va="center", fontsize=11, weight="bold")
    ax[1].set_xlim(0, 1.15)
    ax[1].set_xlabel("Share of within crop ranking variance")
    ax[1].set_title(f"(b) Moran's I = {sgn(m.I, 3)} (z = {sgn(m.z, 1)}, p = {m.p:.3f})")
    ax[1].grid(alpha=0.3, axis="x")

    ax[2].scatter(fm.npix_full.clip(lower=1), fm.ring_frac, s=9, alpha=0.35, color=DARK,
                  edgecolors="none")
    for lo, hi_ in [(0, 20), (20, 50), (50, 150), (150, 10 ** 9)]:
        k = (fm.npix_full >= lo) & (fm.npix_full < hi_)
        ax[2].plot([max(lo, 1), min(hi_, fm.npix_full.max())], [fm.ring_frac[k].median()] * 2,
                   color=ORANGE, lw=3, label="Median per size class" if lo == 0 else None)
    ax[2].axhline(float(st.median_ring_frac), color=RED, ls="--", lw=1.5,
                  label=f"Village median, {st.median_ring_frac:.0%} boundary")
    ax[2].set_xscale("log")
    ax[2].xaxis.set_major_formatter(PLAIN)
    ax[2].set_xlabel("Pixels in the parcel (log scale)")
    ax[2].set_ylabel("Fraction of the parcel that is boundary")
    ax[2].set_ylim(0, 1.05)
    ax[2].set_title("(c) Small parcels are mostly boundary pixels")
    ax[2].legend(frameon=True, loc="lower left")
    ax[2].grid(alpha=0.3, which="both")

    labels = ["Rank correlation\nwith the full index", "Attention list\nretained"]
    core = [float(st.rho_core_vs_shipped), float(st.retention_core)]
    rand = [float(st.rho_random_vs_shipped), float(st.retention_random)]
    x = np.arange(2)
    ax[3].bar(x - 0.18, core, 0.34, color=BLUE, label="Boundary pixels removed")
    ax[3].bar(x + 0.18, rand, 0.34, color=GREY, label="Same number removed at random (control)")
    for xi, (c_, r_) in enumerate(zip(core, rand)):
        ax[3].text(xi - 0.18, c_ + 0.02, f"{c_:.3f}", ha="center", fontsize=9.5, color=BLUE)
        ax[3].text(xi + 0.18, r_ + 0.02, f"{r_:.3f}", ha="center", fontsize=9.5, color=DARK)
    ax[3].axhline(float(st.chance_floor), color=RED, ls="--", lw=1.2, label="Chance level (10%)")
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(labels)
    ax[3].set_ylim(0, 1.15)
    ax[3].set_title("(d) Removing boundary pixels acts like random removal")
    ax[3].legend(loc="upper right", frameon=True)
    ax[3].grid(alpha=0.3, axis="y")
    fig.tight_layout(h_pad=2.5)
    check_and_save(fig, "fig3_spatial_validity.png")


def fig4():
    fm = pd.read_csv(T / "p3_speckle_farm.csv")
    sw = pd.read_csv(T / "p3_speckle_sweep.csv")
    nom = sw[sw.looks_per_pixel == 1.0].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    ax[0].scatter(fm.npix_peak.clip(lower=1), fm.index_sd_F1, s=9, alpha=0.35, color=DARK,
                  edgecolors="none")
    for lo, hi_ in [(0, 20), (20, 50), (50, 150), (150, 10 ** 9)]:
        k = (fm.npix_peak >= lo) & (fm.npix_peak < hi_)
        ax[0].plot([max(lo, 1), min(hi_, fm.npix_peak.max())], [fm.index_sd_F1[k].mean()] * 2,
                   color=ORANGE, lw=3, label="Mean per size class" if lo == 0 else None)
    ax[0].set_xscale("log")
    ax[0].xaxis.set_major_formatter(PLAIN)
    ax[0].set_xlabel("Pixels in the parcel at peak canopy (log scale)")
    ax[0].set_ylabel("Index spread across simulated passes (points)")
    ax[0].set_title(f"(a) Smaller parcels are less stable (ρ = {sgn(nom.c2b_sd_vs_npix)})")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.3, which="both")

    li = fm[fm.on_attention_list].sort_values("retain_F1.0", ascending=False)
    r = li["retain_F1.0"].to_numpy()
    col = [GREEN if v_ >= 0.8 else ORANGE if v_ < 0.5 else GREY for v_ in r]
    ax[1].bar(range(len(r)), r, color=col, width=1.0)
    ax[1].axhline(float(nom.chance_floor), color=RED, ls="--", lw=1.5, label="Chance level (10%)")
    ax[1].legend(loc="upper right")
    ax[1].text(19, 0.5, f"{int((r >= 0.8).sum())} farms\nin ≥ 80%\nof passes", fontsize=10,
               color="white", ha="center", va="center", weight="bold")
    ax[1].text(81, 0.68, f"{int((r < 0.5).sum())} farms\nin under 50%\nof passes", fontsize=10,
               color=ORANGE, ha="center", va="center", weight="bold")
    ax[1].set_xlim(-1, 97)
    ax[1].set_xlabel("The 97 attention list farms, sorted by stability")
    ax[1].set_ylabel("Share of simulated passes listing the farm")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title(f"(b) {nom.retention:.0%} of the attention list is retained on average")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    check_and_save(fig, "fig4_speckle_stability.png")


def fig5():
    u = pd.read_csv(T / "p3_uncertainty.csv").set_index("term")
    cb = pd.read_csv(T / "p3_conformal_beta.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    labels = ["SAR\nmodifier", "Crop label\n(own posterior)", "Crop label\n(five other teams)",
              "Anchor", "Label and\nanchor"]
    vals = [float(u.loc["structural / ensemble (village)", "value"]),
            float(u.loc["level: crop label (village)", "value"]), LABEL_EXTERNAL_T,
            float(u.loc["level: anchor (village)", "value"]),
            float(u.loc["level: both (village)", "value"])]
    b = ax[0].bar(labels, vals, color=["#c53030", GREY, DARK, BLUE, "#1a202c"], edgecolor="#222")
    for bar, v_ in zip(b, vals):
        ax[0].text(bar.get_x() + bar.get_width() / 2, v_ + 6, f"{v_:.0f} t", ha="center", fontsize=10)
    ax[0].set_ylabel("90% interval width, village production (t)")
    ax[0].set_ylim(0, max(vals) * 1.18)
    ax[0].set_title("(a) Village total uncertainty by source")
    ax[0].grid(alpha=0.3, axis="y")

    ax[1].plot(cb.assumed_label_error_beta, 100 * cb.guaranteed_coverage, "o-", color=BLUE, lw=2,
               label="Guaranteed coverage")
    ax[1].axhline(90, color="#c53030", ls="--", lw=1, label="Nominal 90%")
    for rr in cb.itertuples():
        ax[1].annotate(f"{100 * rr.guaranteed_coverage:.0f}%",
                       (rr.assumed_label_error_beta, 100 * rr.guaranteed_coverage),
                       textcoords="offset points", xytext=(9, 4), fontsize=8.5)
    ax[1].set_xlabel("Assumed crop label error rate β")
    ax[1].set_ylabel("Guaranteed coverage (%)")
    ax[1].set_ylim(40, 100)
    ax[1].set_xlim(-0.02, 0.45)
    ax[1].legend(loc="lower left")
    ax[1].grid(alpha=0.3)
    ax[1].set_title("(b) Conformal coverage falls as label error rises")
    fig.tight_layout()
    check_and_save(fig, "fig5_uncertainty.png")


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5):
        fn()
