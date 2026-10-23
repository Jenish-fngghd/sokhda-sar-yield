"""Paper figures for the known-answer SHAP benchmark (figures 6 to 9).

Same rules as figures 1 to 5: Arial, panel letters, 300 dpi, and an automatic check that no
visible text contains a hyphen or dash and that no two text boxes overlap.

Run:  .venv/Scripts/python.exe make_figs.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.unicode_minus": True,
                     "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5})
# Tailwind palette: (400 shade for fills, 600 shade for edges and lines). Same hue = same
# meaning in every figure: emerald truth, blue LightGBM, violet random forest, amber MLP,
# neutral random or unrelated, light blue (200 fill) path dependent SHAP.
TW = {"emerald": ("#34d399", "#059669"), "blue": ("#60a5fa", "#2563eb"),
      "lightblue": ("#bfdbfe", "#2563eb"), "violet": ("#a78bfa", "#7c3aed"),
      "amber": ("#fbbf24", "#d97706"), "neutral": ("#a3a3a3", "#525252"),
      "red": ("#f87171", "#dc2626"), "orange": ("#fb923c", "#ea580c"),
      "sky": ("#38bdf8", "#0284c7"), "rose": ("#fb7185", "#e11d48"), "teal": ("#2dd4bf", "#0d9488")}
TRUTH, LGBM, RF, MLP, NEUTRAL, PATH = "emerald", "blue", "violet", "amber", "neutral", "lightblue"
MODEL_HUE = {"lgbm": LGBM, "rf": RF, "mlp": MLP}

# Paper palette is pastelgrey (chosen 2026-10-16). Other trial variants: PALETTE=mono|pastel|
# light300|brewer, saved to figures/variants/.
# mono: one hue (Tailwind blue 100 to 900), shade encodes the series. pastel: 200 fill, 500 edge.
VARIANT = __import__("os").environ.get("PALETTE", "")
PALETTE = VARIANT or "pastelgrey"
if PALETTE == "mono":
    TW.update({"emerald": ("#1e3a8a", "#172554"), "blue": ("#1d4ed8", "#1e3a8a"),
               "lightblue": ("#60a5fa", "#1e3a8a"), "violet": ("#93c5fd", "#1e3a8a"),
               "amber": ("#dbeafe", "#1e3a8a"), "neutral": ("#d4d4d4", "#525252"),
               "red": ("#3b82f6", "#1e3a8a"), "orange": ("#bfdbfe", "#1e3a8a"),
               "sky": ("#1d4ed8", "#1e3a8a"), "rose": ("#60a5fa", "#1e3a8a"), "teal": ("#bfdbfe", "#1e3a8a")})
elif PALETTE == "pastel":
    TW.update({"emerald": ("#a7f3d0", "#10b981"), "blue": ("#bfdbfe", "#3b82f6"),
               "lightblue": ("#dbeafe", "#3b82f6"), "violet": ("#ddd6fe", "#8b5cf6"),
               "amber": ("#fde68a", "#f59e0b"), "neutral": ("#e5e5e5", "#737373"),
               "red": ("#fecaca", "#ef4444"), "orange": ("#fed7aa", "#f97316"),
               "sky": ("#bae6fd", "#0ea5e9"), "rose": ("#fecdd3", "#f43f5e"), "teal": ("#99f6e4", "#14b8a6")})
elif PALETTE == "light300":    # Tailwind 300 fill, 600 edge: light but more saturated
    TW.update({"emerald": ("#6ee7b7", "#059669"), "blue": ("#93c5fd", "#2563eb"),
               "lightblue": ("#dbeafe", "#2563eb"), "violet": ("#c4b5fd", "#7c3aed"),
               "amber": ("#fcd34d", "#d97706"), "neutral": ("#d4d4d4", "#525252"),
               "red": ("#fca5a5", "#dc2626"), "orange": ("#fdba74", "#ea580c"),
               "sky": ("#7dd3fc", "#0284c7"), "rose": ("#fda4af", "#e11d48"), "teal": ("#5eead4", "#0d9488")})
elif PALETTE == "pastelgrey":  # Tailwind 200 fill, one thin grey edge for every series
    g = "#737373"
    TW.update({"emerald": ("#a7f3d0", g), "blue": ("#bfdbfe", g), "lightblue": ("#e0ecfd", g),
               "violet": ("#ddd6fe", g), "amber": ("#fde68a", g), "neutral": ("#e5e5e5", g),
               "red": ("#fecaca", g), "orange": ("#fed7aa", g), "sky": ("#bae6fd", g),
               "rose": ("#fecdd3", g), "teal": ("#99f6e4", g)})
elif PALETTE == "brewer":      # ColorBrewer Pastel1/Pastel2 fills, dark grey edge
    g = "#525252"
    TW.update({"emerald": ("#b3e2cd", g), "blue": ("#b3cde3", g), "lightblue": ("#e3eaf5", g),
               "violet": ("#decbe4", g), "amber": ("#fff2ae", g), "neutral": ("#cccccc", g),
               "red": ("#fbb4ae", g), "orange": ("#fdcdac", g), "sky": ("#cbd5e8", g),
               "rose": ("#f4cae4", g), "teal": ("#ccebc5", g)})
# lines need a hue even when every bar edge is grey: Tailwind 500 shades
LINE = {"emerald": "#10b981", "blue": "#3b82f6", "neutral": "#737373"}
if VARIANT:
    OUT = OUT / "variants"
    OUT.mkdir(exist_ok=True)


def fill(h):
    """Bar or marker style: 400 fill with 600 edge."""
    return dict(color=TW[h][0], edgecolor=TW[h][1], linewidth=0.8)


STRATA = [("causal", TRUTH, "Index inputs (true drivers)"),
          ("copy", "red", "Near copies of an input (|ρ| ≥ 0.98)"),
          ("proxy", "orange", "Correlated proxies (0.5 ≤ |ρ| < 0.98)"),
          ("unrelated", NEUTRAL, "Unrelated features (|ρ| < 0.5)")]
SD_NOTE = "error bars show 1 SD across 5 seeds"


def legend(ax, title, **kw):
    """Paper legend: bold left aligned title, no frame."""
    kw.setdefault("frameon", False)
    return ax.legend(title=title, title_fontproperties={"weight": "bold", "size": 8.5},
                     alignment="left", **kw)
MODEL = {"lgbm": "LightGBM", "rf": "Random forest", "mlp": "MLP"}
EXPL = {"int": "SHAP\n(interventional)", "path": "SHAP\n(path dependent)", "agn": "SHAP\n(permutation)",
        "perm": "Permutation\nimportance", "gain": "Gain\nimportance"}
DLAB = {"20250606": "06 Jun", "20250619": "19 Jun", "20250814": "14 Aug", "20251013": "13 Oct",
        "20251029": "29 Oct", "20251112": "12 Nov", "static": "Static"}


def check_and_save(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
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
    if VARIANT:
        name = name.replace(".png", f"_{PALETTE}.png")
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{name}: {len(texts)} texts, no dashes, no overlaps")


def fig9_roar():
    r = pd.read_csv(RES / "roar.csv")
    r["ord"] = r.order.str.replace(r"random\d", "random", regex=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    style = {"truth": (TRUTH, "o-", "Known answer: truly most important first"),
             "shap": (LGBM, "s--", "SHAP: highest ranked first"),
             "random": (NEUTRAL, "^:", "Random: mean, band shows min to max")}
    for i, (lvl, xl, title) in enumerate((
            ("feature", "Features removed", "(a) Fit after ranked feature removal"),
            ("date", "Acquisition dates removed (all features of a date)", "(b) Fit after ranked date removal"))):
        q = r[r.level == lvl]
        for o, (c, mk, lab) in style.items():
            g = q[q.ord == o].groupby("k").r2
            ax[i].plot(g.mean().index, g.mean(), mk, color=LINE[c], mfc=TW[c][0], mec=LINE[c],
                       lw=2, ms=6, label=lab)
            if o == "random":
                ax[i].fill_between(g.min().index, g.min(), g.max(), color="#e5e5e5", lw=0)
        ax[i].set_xlabel(xl)
        ax[i].set_ylabel("Cross validated R² after retraining")
        ax[i].set_ylim(-0.03, 1.05)
        ax[i].set_title(title)
        ax[i].grid(alpha=0.3)
        legend(ax[i], "Removal order (LightGBM, realistic inputs)", loc="lower left", frameon=True)
    fig.tight_layout()
    check_and_save(fig, "fig9_roar.png")


def _M():
    M = pd.read_csv(RES / "metrics.csv")
    return M[~M.permuted]


def fig6_headline():
    """Where the attribution goes (D2, T1, no noise), and agreement at three levels."""
    M = _M()
    q = M[(M.design == "D2") & (M.target == "T1") & (M.noise == 0.0)]
    bars = [("Truth", None, None), ("LightGBM", "lgbm", "int"), ("LightGBM", "lgbm", "path"),
            ("LightGBM", "lgbm", "gain"), ("LightGBM", "lgbm", "perm"), ("Random forest", "rf", "int"),
            ("Random forest", "rf", "path"), ("MLP", "mlp", "agn")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    labels, bottoms = [], np.zeros(len(bars))
    vals = {s: [] for s, _, _ in STRATA}
    for name, m, e in bars:
        if m is None:
            labels.append("Truth")
            for s, _, _ in STRATA:
                vals[s].append(1.0 if s == "causal" else 0.0)
            continue
        z = q[(q.model == m) & (q.explainer == e)]
        labels.append(f"{name}\n{EXPL[e]}")
        vals["causal"].append(1 - z.null_mass.mean())
        for s in ("copy", "proxy", "unrelated"):
            vals[s].append(z[f"mass_{s}"].mean())
    x = np.arange(len(bars))
    for s, c, lab in STRATA:
        ax[0].bar(x, vals[s], 0.7, bottom=bottoms, label=lab, **fill(c))
        bottoms += np.array(vals[s])
    for i, v in enumerate(vals["causal"]):
        ax[0].text(i, v / 2, f"{v:.0%}", ha="center", va="center", color="#064e3b", fontsize=9, weight="bold")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=7.5)
    ax[0].set_ylabel("Share of total attribution")
    ax[0].set_ylim(0, 1.42)
    ax[0].set_yticks(np.arange(0, 1.01, 0.2))
    legend(ax[0], "Attribution received by (|ρ| = strongest correlation with an index input)",
           loc="upper center", ncol=2)
    ax[0].set_title("(a) Attribution allocation by feature type")

    rows = [("lgbm", "int"), ("rf", "int"), ("mlp", "agn")]
    lv = [("prec_at_k", "Top 10\nprecision"), ("rho_feature", "Feature\nranking ρ"),
          ("rho_family", "Family\nranking ρ"), ("rho_date", "Date\nranking ρ")]
    w = 0.26
    for j, (m, e) in enumerate(rows):
        z = q[(q.model == m) & (q.explainer == e)]
        mu = [z[k].mean() for k, _ in lv]
        sd = [z[k].std() for k, _ in lv]
        ax[1].bar(np.arange(len(lv)) + (j - 1) * w, mu, w, yerr=sd, capsize=2, **fill(MODEL_HUE[m]),
                  label=f"{MODEL[m]}, R² {z.r2_true.median():.2f}")
    ax[1].set_xticks(np.arange(len(lv)))
    ax[1].set_xticklabels([l for _, l in lv])
    ax[1].set_ylim(0, 1.32)
    ax[1].set_yticks(np.arange(0, 1.01, 0.2))
    ax[1].set_ylabel("Agreement with the known answer")
    legend(ax[1], f"Model and median R² of fit;\n{SD_NOTE}", loc="upper left", ncol=3)
    ax[1].set_title("(b) Agreement with known attribution by grouping level")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    check_and_save(fig, "fig6_attribution_leak.png")


def fig7_mechanism():
    L = pd.read_csv(RES / "leak_by_feature.csv")
    q = L[(L.target == "T1") & (L.noise == 0.0) & (L.model == "lgbm")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    FAM = {"level": "Level (backscatter intensity)", "texture": "Texture (speckle and GLCM)",
           "geometry": "Geometry (incidence angle)", "quality": "Quality (coverage, pixel count, noise floor)"}
    fc = {"level": "sky", "texture": "rose", "geometry": "teal", "quality": NEUTRAL}
    z = q[(q.explainer == "int") & (q.stratum != "causal")]
    for fam, c in fc.items():
        k = z[z.family == fam]
        ax[0].scatter(k.maxrho, 100 * k.share, s=24, c=TW[c][0], edgecolors=TW[c][1], linewidths=0.7,
                      label=FAM[fam])
    ax[0].set_xlabel("Largest |ρ| with any feature the index uses")
    ax[0].set_ylabel("Share of SHAP attribution (%)")
    ax[0].set_title("(a) Attribution to unused features and their correlation")
    legend(ax[0], "Feature family (one point per unused feature)", loc="upper left")
    ax[0].grid(alpha=0.3)

    c = q[(q.explainer == "int") & (q.stratum == "causal")].sort_values("true_share", ascending=False)
    pth = q[(q.explainer == "path") & (q.stratum == "causal")].set_index("feature").loc[c.feature]
    dn = {"20250606": "06 Jun", "20250619": "19 Jun", "20250814": "14 Aug", "20251013": "13 Oct",
          "20251112": "12 Nov"}
    kind = {"g0_db": "Backscatter dB", "g0_lin": "Backscatter linear", "cv": "Texture CV"}
    lab = [f"{kind[f.rsplit('_', 1)[0]]}, {dn[f.rsplit('_', 1)[1]]}" for f in c.feature]
    y = np.arange(len(c))
    ax[1].barh(y - 0.27, 100 * c.true_share, 0.27, label="Known answer (truth)", **fill(TRUTH))
    ax[1].barh(y, 100 * c.share, 0.27, label="SHAP, interventional", **fill(LGBM))
    ax[1].barh(y + 0.27, 100 * pth.share.to_numpy(), 0.27, label="SHAP, path dependent", **fill(PATH))
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(lab)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Share of attribution (%)")
    ax[1].set_title("(b) SHAP and known attribution for index inputs")
    legend(ax[1], "Attribution source (LightGBM)", loc="lower right")
    ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    check_and_save(fig, "fig7_leak_mechanism.png")


def fig8_dates():
    L = pd.read_csv(RES / "leak_by_feature.csv")
    q = L[(L.target == "T1") & (L.noise == 0.0)]
    dates = ["20250606", "20250619", "20250814", "20251013", "20251029", "20251112", "static"]
    series = [("Known answer (truth)", None, None, TRUTH),
              ("LightGBM, SHAP interventional", "lgbm", "int", LGBM),
              ("LightGBM, SHAP path dependent", "lgbm", "path", PATH),
              ("Random forest, SHAP interventional", "rf", "int", RF),
              ("MLP, SHAP permutation", "mlp", "agn", MLP)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.16
    for j, (lab, m, e, c) in enumerate(series):
        if m is None:
            v = q[(q.model == "lgbm") & (q.explainer == "int")].groupby("date").true_share.sum()
        else:
            v = q[(q.model == m) & (q.explainer == e)].groupby("date").share.sum()
        ax.bar(np.arange(len(dates)) + (j - 2) * w, [100 * v.get(d, 0) for d in dates], w, label=lab, **fill(c))
    ax.set_xticks(np.arange(len(dates)))
    ax.set_xticklabels([DLAB[d] for d in dates])
    ax.set_ylabel("Share of attribution (%)")
    ax.set_title("Attribution share by acquisition date")
    legend(ax, "Attribution source (29 Oct is unused by the index)", loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    check_and_save(fig, "fig8_dates.png")


def fig10_local():
    M = _M()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    cfg = [("lgbm", "int", "LightGBM"), ("rf", "int", "Random forest"), ("mlp", "agn", "MLP")]
    x = np.arange(4)
    w = 0.26
    ticks = ["Exact inputs\nno noise", "Exact inputs\nnoisy target",
             "Realistic inputs\nno noise", "Realistic inputs\nnoisy target"]
    for i, (metric, ylab, title) in enumerate((
            ("cos_group_med", "Median per farm cosine with truth", "(a) Per farm agreement at date and family levels"),
            ("why_flagged_agree", "Flagged farms with the right main cause", "(b) Main cause identification for low index farms"))):
        for j, (m, e, lab) in enumerate(cfg):
            vals, errs = [], []
            for D in ("D1", "D2"):
                for s in (0.0, 0.5):
                    z = M[(M.target == "T1") & (M.design == D) & (M.noise == s) & (M.model == m) & (M.explainer == e)]
                    vals.append(z[metric].mean())
                    errs.append(z[metric].std())
            ax[i].bar(x + (j - 1) * w, vals, w, yerr=errs, capsize=2, label=lab, **fill(MODEL_HUE[m]))
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(ticks)
        ax[i].set_ylabel(ylab)
        ax[i].set_ylim(0, 1.25)
        ax[i].set_yticks(np.arange(0, 1.01, 0.2))
        ax[i].set_title(title)
        legend(ax[i], f"Model; {SD_NOTE}", loc="upper right", ncol=3)
        ax[i].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    check_and_save(fig, "fig10_local.png")


if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or [n for n in dir() if n.startswith("fig")]):
        globals()[name]()
