"""The pipeline / architecture diagram, drawn rather than screenshotted.

Layout: inputs on top, the six processing stages across the middle, the ancillary sources
that enter as gates below them, the verification harness under that, deliverables at the
foot. Colours are the repo's own: red for what the radar decides, green for what a
published statistic decides, amber for a gate, slate for structure.

Every connector is orthogonal - down, across, down - and each horizontal run sits on its
own reserved lane, so no line is diagonal and none is drawn through a label.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "results" / "figures" / "p3_pipeline.png"

RADAR  = "#c53030"    # the imagery decides this
PUB    = "#2f855a"    # a published statistic decides this
STRUCT = "#4a5568"    # structure / bookkeeping
GATE   = "#b7791f"    # enters as a gate or correction, never as a predictor
INK    = "#14181d"
MUTED  = "#5a646e"
LINE   = "#8d979f"
PALE   = "#b8c0c6"
PAPER  = "#ffffff"
FAINT  = "#f6f8f9"

STAGES = [
    ("1", "Radiometry", RADAR,
     "beta0 = SF$^2$|z|$^2$, terrain-\nflattened to gamma0, geocoded\nto a common 2 m / 5 m grid",
     "verified against the vendor's\nown declared noise floor"),
    ("2", "Parcel series", STRUCT,
     "966 polygons x 6 dates\nbuffer ladder -5 / -2 / 0 m\ncoverage and noise gates",
     "111 features per farm"),
    ("3", "Health index", RADAR,
     "level - growth - uniformity\npersistence - senescence\nfixed weights, no fitting",
     "nothing fitted to an outcome"),
    ("4", "Crop map", RADAR,
     "trajectory-shape labels\n5 classes",
     "the ONLY place the SAR\nreaches the village total"),
    ("5", "Anchor x modifier", PUB,
     "anchor: a published statistic\nmodifier: normalised to area-\nweighted mean 1.0 WITHIN crop",
     "so the total is invariant\nto the imagery, by construction"),
    ("6", "Product", STRUCT,
     "SUM(yield x area)\n966 plots - 46 zones\n5 crops",
     "every row typed: forecast,\nmeasurement, reconstruction"),
]

#          label                             sub-label                                              colour  x      w      -> stage
INPUTS = [("6 x Capella X-band HH SLC", "6 Jun - 19 Jun - 14 Aug - 13 Oct - 29 Oct - 12 Nov 2025", RADAR,  0.035, 0.300, 0),
          ("966-parcel cadastral boundaries", "survey polygons, median 69 pixels",                 STRUCT, 0.360, 0.270, 1),
          ("Published yield statistics", "MOAFW / Gujarat DES, per crop",                          PUB,    0.665, 0.300, 4)]

#         source                   what it does                                                            -> stage
GATES = [("ERA5-Land", "29 Oct carries 14.1 m/s wind and 81.2 mm of\nrain: excluded from the level features", 2),
         ("Sentinel-2, independent", "same-day NDVI witness: tests the ordering,\nnever feeds it", 2),
         ("MOAFW, 10 Nov 2025", "cotton at 80% of first picking: dates the\ncompletion factor", 4),
         ("Sentinel-1, nine years", "one track, 2016-2025: tests whether 2025\nwas an ordinary year", 4)]

OUTPUTS = [("results/submission.csv", "966 rows - the deliverable"),
           ("p3_village_summary.csv", "710.0 t over 447.5 ha"),
           ("p3_zone_summary.csv", "46 sub-village zones"),
           ("p3_farm_uncertainty.csv", "four terms, reported apart")]


def card(ax, x, y, w, h, edge, fill=PAPER, lw=1.1, top=None, z=3, ls="solid"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=lw, edgecolor=edge, facecolor=fill,
                                zorder=z, linestyle=ls))
    if top:  # the coloured cap says who decides this step
        ax.add_patch(FancyBboxPatch((x, y + h - 0.006), w, 0.006, boxstyle="square,pad=0",
                                    linewidth=0, facecolor=top, zorder=z + 1))


def seg(ax, x0, y0, x1, y1, color, lw, ls="solid", head=False, z=1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>" if head else "-", mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def elbow(ax, x0, y0, x1, y1, lane, color=LINE, lw=1.3, ls="solid"):
    """Orthogonal connector: straight down to `lane`, across it, then down into the target."""
    if abs(x1 - x0) < 1e-4:                      # already aligned: one straight line
        seg(ax, x0, y0, x1, y1, color, lw, ls, head=True)
        return
    seg(ax, x0, y0, x0, lane, color, lw, ls)
    seg(ax, x0, lane, x1, lane, color, lw, ls)
    seg(ax, x1, lane, x1, y1, color, lw, ls, head=True)


def main():
    fig = plt.figure(figsize=(16.0, 9.0), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0",
                                facecolor=PAPER, edgecolor="none", zorder=0))

    margin, gap = 0.035, 0.017
    n = len(STAGES)
    w = (1 - 2 * margin - gap * (n - 1)) / n
    xs = [margin + i * (w + gap) for i in range(n)]
    mid = [x + w / 2 for x in xs]

    yin, hin = 0.800, 0.070                      # input row
    LANES_IN = [0.780, 0.766, 0.752]             # one lane each, so no two horizontals meet
    ys, hs = 0.498, 0.200                        # stage row
    RAIL = 0.446                                 # the single lane every gate connector uses
    yg, hg = 0.286, 0.108                        # gate row
    yv, hv = 0.194, 0.058                        # verification harness
    yo, ho = 0.060, 0.086                        # deliverables

    # ---- title -------------------------------------------------------------
    fig.text(margin, 0.962, "SOKHDA KHARIF 2025 - PIPELINE", fontsize=21,
             fontweight="bold", color=INK, ha="left", va="top")
    fig.text(margin, 0.925,
             "Six Capella X-band acquisitions to a typed, per-plot yield forecast. "
             "The radar decides an ordering and a crop label; a published statistic decides the level.",
             fontsize=11.5, color=MUTED, ha="left", va="top")

    # ---- inputs ------------------------------------------------------------
    fig.text(margin, 0.888, "INPUTS", fontsize=9.5, fontweight="bold", color=MUTED,
             ha="left", va="center")
    for (label, sub, col, x, iw, target), lane in zip(INPUTS, LANES_IN):
        card(ax, x, yin, iw, hin, edge="#c9d0d6", top=col)
        ax.text(x + 0.012, yin + hin - 0.025, label, fontsize=11, fontweight="bold",
                color=INK, va="center")
        ax.text(x + 0.012, yin + hin - 0.048, sub, fontsize=9.2, color=MUTED, va="center")
        elbow(ax, x + iw / 2, yin - 0.004, mid[target], ys + hs + 0.004, lane,
              color=PALE, lw=1.2)

    # ---- the six stages ----------------------------------------------------
    for (num, name, col, body, note), x in zip(STAGES, xs):
        card(ax, x, ys, w, hs, edge="#9aa4ac", top=col, lw=1.0)
        ax.text(x + 0.010, ys + hs - 0.029, f"{num}  {name}", fontsize=12.2,
                fontweight="bold", color=INK, va="center")
        ax.text(x + 0.010, ys + hs - 0.051, body, fontsize=8.9, color=INK,
                va="top", linespacing=1.55)
        ax.text(x + 0.010, ys + 0.019, note, fontsize=8.9, color=col, va="bottom",
                style="italic", linespacing=1.5)
    for x in xs[:-1]:
        seg(ax, x + w + 0.002, ys + hs / 2, x + w + gap - 0.002, ys + hs / 2,
            LINE, 1.6, head=True)

    # ---- ancillary, entering as gates --------------------------------------
    # The label has its own line between the stage row and the connector lane, and it ends
    # well left of the first riser, so nothing is ever drawn through it.
    fig.text(margin, 0.472, "ANCILLARY - ENTERS AS A GATE, NEVER AS A PREDICTOR",
             fontsize=9.5, fontweight="bold", color=GATE, ha="left", va="center")

    gutter = 0.062                               # kept clear so stage 6 can reach the harness
    gw = (1 - 2 * margin - gutter - gap * 3) / 4
    for i, (title, body, target) in enumerate(GATES):
        gx = margin + i * (gw + gap)
        card(ax, gx, yg, gw, hg, edge=GATE, fill="#fffdf7", lw=0.9, ls=(0, (4, 2)))
        ax.text(gx + 0.012, yg + hg - 0.028, title, fontsize=10.4, fontweight="bold",
                color=INK, va="center")
        ax.text(gx + 0.012, yg + hg - 0.050, body, fontsize=8.9, color=INK,
                va="top", linespacing=1.55)
        elbow(ax, gx + gw / 2, yg + hg + 0.004, mid[target], ys - 0.004, RAIL,
              color=GATE, lw=1.0, ls=(0, (3, 2)))

    # ---- verification ------------------------------------------------------
    card(ax, margin, yv, 1 - 2 * margin, hv, edge=STRUCT, fill=FAINT, lw=1.2)
    ax.text(margin + 0.014, yv + hv / 2 + 0.011, "VERIFICATION HARNESS", fontsize=10.4,
            fontweight="bold", color=STRUCT, va="center")
    ax.text(margin + 0.014, yv + hv / 2 - 0.013,
            "19 defensibility checks, thresholds fixed before the run  ·  12 failure-mode "
            "regression assertions  ·  spatial hold-out, refit in one half and applied to the "
            "other  ·  9 figures  ·  submission checksum-pinned",
            fontsize=9.4, color=INK, va="center")

    # stage 6 drops straight down the gutter: one vertical, no elbow, nothing crossed
    xdrop = 1 - margin - 0.024
    seg(ax, xdrop, ys - 0.004, xdrop, yv + hv + 0.004, LINE, 1.6, head=True)

    # ---- deliverables ------------------------------------------------------
    ow = (1 - 2 * margin - gap * 3) / 4
    for i, (name, sub) in enumerate(OUTPUTS):
        ox = margin + i * (ow + gap)
        card(ax, ox, yo, ow, ho, edge="#c9d0d6", top=STRUCT)
        ax.text(ox + 0.012, yo + ho - 0.029, name, fontsize=10.6, fontweight="bold",
                color=INK, va="center", family="monospace")
        ax.text(ox + 0.012, yo + ho - 0.053, sub, fontsize=9.2, color=MUTED, va="center")
        seg(ax, ox + ow / 2, yv - 0.004, ox + ow / 2, yo + ho + 0.004, PALE, 1.1, head=True)

    # ---- legend ------------------------------------------------------------
    keys = [(RADAR, "the radar decides this"), (PUB, "a published statistic decides this"),
            (GATE, "gate or correction"), (STRUCT, "structure and bookkeeping")]
    x0 = margin
    for col, lab in keys:
        ax.add_patch(FancyBboxPatch((x0, 0.018), 0.016, 0.007, boxstyle="square,pad=0",
                                    facecolor=col, edgecolor="none"))
        ax.text(x0 + 0.022, 0.0215, lab, fontsize=9.2, color=MUTED, va="center")
        x0 += 0.030 + 0.0062 * len(lab)

    fig.savefig(OUT, facecolor=PAPER)
    sz = fig.get_size_inches() * fig.dpi
    print(f"{OUT}  {int(sz[0])}x{int(sz[1])}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    main()
