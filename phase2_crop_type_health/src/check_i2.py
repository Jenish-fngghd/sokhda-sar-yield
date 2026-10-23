"""I2 verification -- feature QA before anything is built on top of these numbers.

Answers four questions:
  1. Is every farm present, and how did each one get its pixels? (coverage = 10 pts)
  2. Are the distributions physically sensible?
  3. Which features are redundant? (correlated pairs add nothing but noise)
  4. Do the features vary SPATIALLY in a structured way, or do they look like noise?
     Real agronomic signal is spatially autocorrelated; speckle is not. This is a
     preview of the Moran's I validation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATES, FARMS, FIGURES, RESULTS, log

import geopandas as gpd

KEY = ["g0_db_20251013", "ref_aug", "ref_oct", "d_aug_jun19", "temporal_cv",
       "cv_20251013", "ktex_20251013", "glcm_ent_20251013", "season_integral", "area_ha"]


def morans_i(vals, gdf, k=8):
    """Moran's I with a k-nearest-neighbour weight matrix.

    Positive I means neighbouring farms resemble each other -- the signature of a
    real agronomic pattern (shared soil, irrigation, sowing date). Noise gives I ~ 0.
    """
    from scipy.spatial import cKDTree
    ok = np.isfinite(vals)
    xy = np.c_[gdf.centroid.x, gdf.centroid.y][ok]
    v = vals[ok] - vals[ok].mean()
    if len(v) < 30:
        return np.nan
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    idx = idx[:, 1:]
    num = (v[:, None] * v[idx]).sum()
    # I = (n/W) * sum_ij w_ij z_i z_j / sum_i z_i^2, and with binary kNN weights
    # W = n*k, so the n cancels and the k does NOT. Getting this wrong returns
    # values outside [-1, 1], which is how the bug announced itself.
    return float(num / (k * (v ** 2).sum()))


def main():
    f = pd.read_csv(RESULTS / "farm_features.csv")
    g = gpd.read_file(FARMS).to_crs(32643)
    g["geometry"] = g.geometry.make_valid()
    g = g.reset_index(drop=True)
    for c in KEY + ["qc_flag", "buffer_level", "n_dates"]:
        g[c] = f[c].values

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 4)

    # 1 -- coverage / QC
    ax = fig.add_subplot(gs[0, 0])
    qc = f.qc_flag.value_counts()
    ax.bar(qc.index, qc.values, color=["#1a9641", "#fdae61", "#d7191c"], edgecolor="#333")
    for i, (k, v) in enumerate(qc.items()):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"QC flags -- {len(f)} rows, none dropped", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    bl = f.buffer_level.value_counts().sort_index()
    names = {0: "-5 m", 1: "-2 m", 2: "unbuffered"}
    ax.bar([names.get(i, str(i)) for i in bl.index], bl.values,
           color="#3b7dd8", edgecolor="#333")
    for i, v in enumerate(bl.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title("Negative-buffer ladder level used", fontsize=10)

    # 2 -- distributions
    ax = fig.add_subplot(gs[0, 2])
    for d in DATES:
        ax.hist(f[f"g0_db_{d}"].dropna(), bins=50, histtype="step", label=d[4:])
    ax.set_xlabel("farm mean gamma0 (dB)"); ax.legend(fontsize=7)
    ax.set_title("Farm-level backscatter per date", fontsize=10)

    ax = fig.add_subplot(gs[0, 3])
    ax.hist(f.ref_aug.dropna(), bins=50, histtype="step", label="Aug - June base")
    ax.hist(f.ref_oct.dropna(), bins=50, histtype="step", label="Oct - June base")
    ax.axvline(0, color="#999", ls="--")
    ax.set_xlabel("dB vs bare-soil baseline"); ax.legend(fontsize=7)
    ax.set_title("Bare-soil-referenced change [B12]", fontsize=10)

    # 3 -- correlation
    ax = fig.add_subplot(gs[1, :2])
    C = f[KEY].corr(method="spearman")
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(KEY))); ax.set_xticklabels(KEY, rotation=90, fontsize=7)
    ax.set_yticks(range(len(KEY))); ax.set_yticklabels(KEY, fontsize=7)
    for i in range(len(KEY)):
        for j in range(len(KEY)):
            if i != j and abs(C.values[i, j]) > 0.75:
                ax.text(j, i, "!", ha="center", va="center", fontsize=11, color="k")
    plt.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title("Spearman correlation ( ! = |rho|>0.75, redundant )", fontsize=10)

    # 4 -- spatial structure + Moran's I
    mor = {}
    for j, col in enumerate(["ref_oct", "temporal_cv", "ktex_20251013", "glcm_ent_20251013"]):
        ax = fig.add_subplot(gs[1 + j // 2, 2 + j % 2] if j < 2 else gs[2, j - 2])
        v = g[col].values
        I = morans_i(v, g); mor[col] = I
        g.plot(ax=ax, column=col, cmap="viridis", edgecolor="none", legend=True,
               vmin=np.nanpercentile(v, 5), vmax=np.nanpercentile(v, 95),
               legend_kwds={"shrink": 0.6}, missing_kwds={"color": "#dddddd"})
        ax.set_title(f"{col}\nMoran's I = {I:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[2, 0])
    ax.axis("off")
    txt = ["Moran's I (k=8 neighbours)", ""]
    for c in KEY:
        txt.append(f"  {c:<22s} {morans_i(g[c].values if c in g else f[c].values, g):+.3f}")
    ax.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=8)
    ax.set_title("Spatial autocorrelation of every key feature", fontsize=10, loc="left")

    fig.suptitle("I2 VERIFICATION -- farm feature QA", fontsize=13)
    fig.tight_layout()
    p = FIGURES / "i2_01_features.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)

    hi = [(KEY[i], KEY[j], round(C.values[i, j], 2))
          for i in range(len(KEY)) for j in range(i + 1, len(KEY))
          if abs(C.values[i, j]) > 0.75]
    log("i2.qa", rows=len(f), redundant_pairs=hi,
        morans={k: round(v, 3) for k, v in mor.items()})
    log("i2.fig", name=p.name)
    print("redundant pairs:", hi)
    print("Moran's I:", {k: round(v, 3) for k, v in mor.items()})


if __name__ == "__main__":
    main()
