"""I1 verification -- the land-cover dB gate [O1] plus a geocoding regression test.

Two questions this answers, both of which must be settled before features are
extracted from these products:

  1. Does our own GCP geocoding still put the farm boundaries on the fields?
     [F3] passed on the VENDOR's geocoding; this repeats the test on ours.
  2. Do the calibrated dB values order sensibly by land cover?

On (2) the honest answer is nuanced and is recorded in RESEARCH_LOG [G1]: the
ORDERING is correct and the dynamic range is right, but there is a large, nearly
constant absolute offset. Since the index is relative by construction that offset
is harmless -- but it must be stated, not hidden.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import plotting_extent
from scipy.ndimage import label, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATES, FARMS, FIGURES, log

import geopandas as gpd

LABEL = {"20250606": "Jun 06", "20250619": "Jun 19",
         "20250814": "Aug 14", "20251013": "Oct 13"}


def load(grid="base"):
    G, TH, ext = {}, {}, None
    for d in DATES:
        with rasterio.open(CACHE / f"gamma0_{grid}_{d}.tif") as s:
            G[d] = s.read(1); ext = plotting_extent(s)
        with rasterio.open(CACHE / f"incidence_{grid}_{d}.tif") as s:
            TH[d] = s.read(1)
    return G, TH, ext


def masks(G):
    """Persistent water and built-up: dark/bright on EVERY date.

    Requiring persistence across all four dates is what separates a real surface
    from a field that happened to be dark once.
    """
    S = np.stack([G[d] for d in DATES])
    fin = np.isfinite(S).all(axis=0)
    sm = np.stack([uniform_filter(np.nan_to_num(G[d]), 5) for d in DATES])
    dthr = np.array([np.nanpercentile(np.where(fin, sm[i], np.nan), 2.0) for i in range(4)])
    bthr = np.array([np.nanpercentile(np.where(fin, sm[i], np.nan), 99.0) for i in range(4)])
    dark = fin & (sm <= dthr[:, None, None]).all(axis=0)
    lab, _ = label(dark)
    sz = np.bincount(lab.ravel()); sz[0] = 0
    water = np.isin(lab, np.where(sz >= 40)[0])       # >=1000 m2 blobs only
    built = fin & (sm >= bthr[:, None, None]).all(axis=0)
    crop = fin & ~water & ~built
    return fin, water, built, crop


def main():
    farms = gpd.read_file(FARMS).to_crs(32643)
    farms["geometry"] = farms.geometry.make_valid()
    G, TH, ext = load("base")
    fin, water, built, crop = masks(G)
    dB = {d: 10 * np.log10(np.where(G[d] > 0, G[d], np.nan)) for d in DATES}

    fig = plt.figure(figsize=(21, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.5, 1.5, 1.0])

    for j, d in enumerate(DATES):
        ax = fig.add_subplot(gs[0, j])
        v = dB[d][fin]
        ax.imshow(dB[d], cmap="gray", extent=ext,
                  vmin=np.nanpercentile(v, 2), vmax=np.nanpercentile(v, 98))
        farms.boundary.plot(ax=ax, color="#00e5ff", linewidth=0.25)
        ax.set_title(f"{LABEL[d]}  gamma0 dB\nincidence {np.nanmean(TH[d][fin]):.2f} deg", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    cls = np.full(fin.shape, np.nan)
    cls[crop] = 0; cls[water] = 1; cls[built] = 2
    ax.imshow(cls, cmap=matplotlib.colors.ListedColormap(["#c8e6a0", "#1f4e9c", "#d62728"]),
              extent=ext, vmin=-0.5, vmax=2.5)
    farms.boundary.plot(ax=ax, color="#444", linewidth=0.2)
    ax.set_title(f"Persistent land cover\ncrop {crop.sum()} / water {water.sum()} / built {built.sum()} px",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    for d in DATES:
        ax.hist(dB[d][fin], bins=180, histtype="step", label=LABEL[d])
    ax.set_xlabel("gamma0 (dB, uncorrected absolute offset)")
    ax.set_title("Distribution per date", fontsize=10); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    w = [np.nanmedian(dB[d][water]) for d in DATES]
    c = [np.nanmedian(dB[d][crop]) for d in DATES]
    b = [np.nanmedian(dB[d][built]) for d in DATES]
    x = np.arange(4)
    ax.plot(x, b, "o-", color="#d62728", label="built-up")
    ax.plot(x, c, "s-", color="#2ca02c", label="cropland")
    ax.plot(x, w, "^-", color="#1f4e9c", label="water")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[d] for d in DATES], fontsize=8)
    ax.set_ylabel("median gamma0 (dB)")
    ax.set_title("Ordering is correct: built > crop > water", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    th = [float(np.nanmean(TH[d][fin])) for d in DATES]
    ax.plot(th, b, "o", color="#d62728", label="built-up")
    ax.plot(th, c, "s", color="#2ca02c", label="cropland")
    for k, d in enumerate(DATES):
        ax.annotate(LABEL[d], (th[k], c[k]), fontsize=7,
                    textcoords="offset points", xytext=(4, -8))
    ax.set_xlabel("scene mean incidence (deg)"); ax.set_ylabel("median gamma0 (dB)")
    ax.set_title("Angular dependence\n(Jun19 & Aug14 share geometry)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    rows = [["date", "incidence", "water dB", "crop dB", "built dB", "crop-water", "built-crop"]]
    for k, d in enumerate(DATES):
        rows.append([LABEL[d], f"{th[k]:.2f}", f"{w[k]:.2f}", f"{c[k]:.2f}", f"{b[k]:.2f}",
                     f"{c[k]-w[k]:.2f}", f"{b[k]-c[k]:.2f}"])
    t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    ax.set_title("I1 GATE: contrasts are the trustworthy quantity; the absolute level carries "
                 "an undeclared offset [B9]", fontsize=11, pad=18)

    fig.suptitle("I1 VERIFICATION -- calibrated, geocoded gamma0 on the 5 m BASE grid", fontsize=13)
    fig.tight_layout()
    p = FIGURES / "i1_01_verification.png"
    fig.savefig(p, dpi=135, bbox_inches="tight"); plt.close(fig)

    log("i1.gate", water_dB=[round(x, 2) for x in w], crop_dB=[round(x, 2) for x in c],
        built_dB=[round(x, 2) for x in b],
        crop_minus_water=[round(c[k] - w[k], 2) for k in range(4)],
        built_minus_crop=[round(b[k] - c[k], 2) for k in range(4)])
    log("i1.fig", name=p.name)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
