"""Dataset figures D1 to D5 for DATASET_NOTES.md, in the paper's pastelgrey palette.

Reads the frozen Round 3 tables and shapefiles (read only) and writes PNGs next to this file,
so the figures travel with the paper folder.

Run:  py -3.12 figures/dataset/make_dataset_figs.py
"""
import os
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
R3 = PAPER.parent / "phase3_yield_forecast"
DATA = Path(os.environ.get("AISE_DATA") or R3 / "anrf-aise-hack-2-0-round-3-sar-crop-yield-forecasting")
TAB = R3 / "results" / "tables"

# ponytail: reuse the benchmark's palette, legend and dash/overlap checker instead of copying them
sys.path.insert(0, str(PAPER / "shap_benchmark"))
import make_figs as mf  # noqa: E402

mf.OUT = HERE
TW, fill, legend, LINE = mf.TW, mf.fill, mf.legend, mf.LINE
GREY = "#737373"

CROPS = ["Cotton", "Groundnut", "Rice", "Bajra", "Maize"]
CROP_HUE = {"Cotton": "sky", "Groundnut": "orange", "Rice": "teal", "Bajra": "rose", "Maize": "amber"}
CROP_LINE = {"Cotton": "#0ea5e9", "Groundnut": "#f97316", "Rice": "#14b8a6", "Bajra": "#f43f5e",
             "Maize": "#f59e0b"}   # Tailwind 500, for lines
DATES = ["20250606", "20250619", "20250814", "20251013", "20251029", "20251112"]
DLAB = {d: mf.DLAB[d] for d in DATES}
TS = {d: pd.Timestamp(d) for d in DATES}

farms = gpd.read_file(DATA / "Farm_boundaries_shp" / "Farm_boundaries_shp" / "Sokhda_Farms.shp")
farms["geometry"] = farms.geometry.make_valid()
farms = farms.to_crs(32643)
village = gpd.read_file(DATA / "Village_Shp" / "Village_Shp" / "Sokhda_Village.shp").to_crs(32643)
feat = pd.read_csv(TAB / "p1_farm_features.csv")
sub = pd.read_csv(R3 / "results" / "submission.csv")
farms = farms.merge(sub[["farm_id", "crop_type"]], left_on="FID", right_on="farm_id")
farms = farms.merge(feat[["farm_id", "n_dates", "area_ha"]], on="farm_id")
# degenerate parcels sit hundreds of km away; keep what lies on the village
farms = farms[farms.centroid.within(village.buffer(2000).union_all())]


def km_axes(ax, x0, y0):
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(lambda v, _: f"{(v - x0) / 1000:.1f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{(v - y0) / 1000:.1f}")
    ax.set_xlabel("Distance east (km)")
    ax.set_ylabel("Distance north (km)")


def d1_study_area():
    b = np.vstack([village.total_bounds, farms.total_bounds])
    x0, y0 = b[:, :2].min(0) - 50
    x1, y1 = b[:, 2:].max(0) + 50
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    for c in CROPS:
        farms[farms.crop_type == c].plot(ax=ax[0], color=TW[CROP_HUE[c]][0], edgecolor=GREY, linewidth=0.25)
    hand = [Patch(facecolor=TW[CROP_HUE[c]][0], edgecolor=GREY, linewidth=0.6,
                  label=f"{c} ({(farms.crop_type == c).sum()} plots)") for c in CROPS]
    ax[0].legend(handles=hand, title="Crop (shipped crop map)", frameon=False, loc="upper left",
                 title_fontproperties={"weight": "bold", "size": 8.5}, alignment="left",
                 bbox_to_anchor=(1.0, 1.0))
    ax[0].set_title("(a) Sokhda farm plot distribution by assigned crop")

    cmap = ListedColormap(["#fecaca", "#fde68a", "#fef3c7", "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa"])
    norm = BoundaryNorm(np.arange(-0.5, 7.5), cmap.N)
    farms.plot(ax=ax[1], column="n_dates", cmap=cmap, norm=norm, edgecolor=GREY, linewidth=0.25)
    hand = [Patch(facecolor=cmap(k), edgecolor=GREY, linewidth=0.6,
                  label=f"{k} of 6 dates ({(farms.n_dates == k).sum()} plots)")
            for k in range(7) if (farms.n_dates == k).sum()]
    ax[1].legend(handles=hand, title="Capella images that see the plot", frameon=False, loc="upper left",
                 title_fontproperties={"weight": "bold", "size": 8.5}, alignment="left",
                 bbox_to_anchor=(1.0, 1.0))
    ax[1].set_title("(b) Capella acquisition coverage across farm plots")
    for a in ax:
        village.boundary.plot(ax=a, color="#404040", linewidth=1.0)
        a.set_xlim(x0, x1)
        a.set_ylim(y0, y1)
        km_axes(a, x0, y0)
    fig.tight_layout()
    mf.check_and_save(fig, "d1_study_area.png")


def d2_season():
    w = pd.read_csv(TAB / "p0_weather_hourly.csv", parse_dates=["time_utc"]).set_index("time_utc")
    day = pd.DataFrame({"rain": w.precipitation.resample("D").sum(),
                        "soil": w.soil_moisture_0_to_7cm.resample("D").mean()})
    day = day["2025-05-20":"2025-11-20"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.4), gridspec_kw={"height_ratios": [2.2, 1.4]})
    ax[0].bar(day.index, day.rain, width=1.0, **fill("blue"), label="Daily rainfall (ERA5 Land)")
    ax[0].set_ylabel("Rainfall (mm per day)")
    t = ax[0].twinx()
    t.tick_params(labelbottom=False, bottom=False)
    t.plot(day.index, day.soil, color=LINE["emerald"], lw=1.8, label="Topsoil moisture, 0 to 7 cm")
    t.set_ylabel("Soil moisture (m³/m³)")
    t.set_ylim(0, 0.8)
    ax[0].set_ylim(0, day.rain.max() * 1.9)
    for d in DATES:
        for a in ax:
            a.axvline(TS[d], color=GREY, lw=1.0, ls=(0, (4, 3)))
        ax[0].text(TS[d], day.rain.max() * 1.82, DLAB[d], ha="center", va="top", fontsize=8.5,
                   bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = t.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, title="Weather at Sokhda (dotted lines: Capella images)",
                 frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88),
                 title_fontproperties={"weight": "bold", "size": 8.5}, alignment="left")
    ax[0].set_title("(a) Rainfall, topsoil moisture and SAR acquisition dates")

    # indicative kharif calendar for central Gujarat (literature; see DATASET_NOTES.md 2.1)
    cal = {"Cotton": [("2025-06-15", "2025-10-01", "growing"), ("2025-10-01", "2025-11-20", "picking (runs to Jan)")],
           "Groundnut": [("2025-06-15", "2025-10-25", "growing"), ("2025-10-25", "2025-11-20", "mature, lifting")],
           "Rice": [("2025-06-25", "2025-09-25", "growing"), ("2025-09-25", "2025-10-20", "harvest")],
           "Bajra": [("2025-06-15", "2025-09-10", "growing"), ("2025-09-10", "2025-10-10", "harvest")],
           "Maize": [("2025-06-15", "2025-09-10", "growing"), ("2025-09-10", "2025-10-10", "harvest")]}
    for i, c in enumerate(CROPS):
        for s, e, lab in cal[c]:
            s, e = pd.Timestamp(s), pd.Timestamp(e)
            grow = lab == "growing"
            ax[1].barh(i, (e - s).days, left=s, height=0.62, color=TW[CROP_HUE[c]][0], edgecolor=GREY,
                       linewidth=0.6, hatch=None if grow else "////")
            if not grow:
                ax[1].text(e + pd.Timedelta(days=2), i, lab, va="center", fontsize=8)
    ax[1].set_yticks(range(len(CROPS)))
    ax[1].set_yticklabels(CROPS)
    ax[1].invert_yaxis()
    for a in ax:
        a.set_xlim(pd.Timestamp("2025-05-20"), pd.Timestamp("2025-12-25"))
    ax[0].tick_params(labelbottom=False)
    t.tick_params(labelbottom=False, bottom=False)
    ax[1].set_title("(b) Crop calendar (plain: growing; hatched: harvest window)")
    for a in ax:
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    mf.check_and_save(fig, "d2_season_timeline.png")


def d3_trajectories():
    f = feat.merge(sub[["farm_id", "crop_type"]], on="farm_id")
    x = [TS[d] for d in DATES]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axvspan(pd.Timestamp("2025-10-26"), pd.Timestamp("2025-10-31"), color="#e5e5e5", lw=0)
    ax.text(pd.Timestamp("2025-10-28"), 0.97, "rain and\nstrong wind", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8)
    for c in CROPS:
        g = f[f.crop_type == c][[f"g0_db_{d}" for d in DATES]]
        med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.fill_between(x, lo, hi, color=TW[CROP_HUE[c]][0], alpha=0.55, lw=0)
        ax.plot(x, med, "o-", color=CROP_LINE[c], mfc=TW[CROP_HUE[c]][0], mec=CROP_LINE[c], lw=1.8, ms=6,
                label=f"{c} (n = {len(g)})")
    ax.set_xticks(x)
    ax.set_xticklabels([DLAB[d] for d in DATES])
    ax.set_ylabel("Plot mean backscatter γ⁰ (dB, relative)")
    ax.set_title("Seasonal relative SAR backscatter profiles by crop")
    ax.grid(alpha=0.3)
    legend(ax, "Crop: median line, band shows middle 50% of plots", loc="lower left", ncol=3)
    fig.tight_layout()
    mf.check_and_save(fig, "d3_crop_trajectories.png")


def d4_plots_and_coverage():
    f = feat.merge(sub[["farm_id", "crop_type"]], on="farm_id")
    f = f[f.area_ha > 0.005]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.logspace(np.log10(0.01), np.log10(f.area_ha.max() * 1.05), 30)
    ax[0].hist(f.area_ha, bins=bins, **fill("sky"))
    ax[0].set_xscale("log")
    ax[0].set_xticks([0.01, 0.03, 0.1, 0.3, 1, 3])
    ax[0].set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1", "3"])
    ax[0].xaxis.set_minor_formatter(lambda v, _: "")
    med = f.area_ha.median()
    ax[0].axvline(med, color="#404040", lw=1.2, ls=(0, (4, 3)))
    ax[0].text(med * 1.12, ax[0].get_ylim()[1] * 0.93, f"median {med:.2f} ha",
               fontsize=8.5, va="top")
    ax[0].set_xlabel("Plot area (ha, log scale)")
    ax[0].set_ylabel("Number of plots")
    ax[0].set_title("(a) Distribution of farm plot area")

    obs = [(f[f"g0_db_{d}"].notna()).sum() for d in DATES]
    xx = np.arange(len(DATES))
    ax[1].bar(xx, obs, 0.6, **fill("blue"))
    for i, o in enumerate(obs):
        ax[1].text(i, o / 2, str(o), ha="center", va="center", fontsize=8.5)
    ax[1].axhline(len(f), color=GREY, lw=1.0, ls=(0, (4, 3)))
    ax[1].text(4, len(f) + 22, f"all valid plots ({len(f)})", ha="center", fontsize=8.5)
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels([DLAB[d] + ("\n(right looking)" if d == "20251029" else "") for d in DATES])
    ax[1].set_ylim(0, len(f) * 1.12)
    ax[1].set_ylabel("Plots with a radar measurement")
    ax[1].set_title("(b) Plot coverage across Capella acquisition dates")
    fig.tight_layout()
    mf.check_and_save(fig, "d4_plot_size_coverage.png")


def d5_correlation():
    names = {"g0_db": "γ⁰ mean (dB)", "g0_lin": "γ⁰ mean (linear)", "g0_db_med": "γ⁰ median (dB)",
             "b0_db": "β⁰ mean (dB)", "nesz_margin_db": "Margin above noise floor", "cv": "Within plot CV",
             "ktex": "K texture", "glcm_ent": "GLCM entropy", "glcm_resid": "GLCM residual",
             "inc": "Incidence angle", "cov": "Coverage", "npix": "Pixel count",
             "nesz_frac0": "Pixels below noise floor", "nesz_frac3": "Pixels within 3 dB of floor",
             "nesz_frac6": "Pixels within 6 dB of floor"}
    cols = [f"{k}_20250814" for k in names if f"{k}_20250814" in feat]
    C = feat[cols].corr(method="spearman").abs().to_numpy()
    lab = [names[c.replace("_20250814", "")] for c in cols]
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and np.isfinite(C[i, j]):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if C[i, j] > 0.7 else "#404040")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(lab, rotation=90, fontsize=8)
    ax.set_yticklabels(lab, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Absolute Spearman correlation between plots")
    ax.set_title("Correlation among SAR features on 14 August")
    fig.tight_layout()
    mf.check_and_save(fig, "d5_feature_correlation.png")


if __name__ == "__main__":
    for fn in (d1_study_area, d2_season, d3_trajectories, d4_plots_and_coverage, d5_correlation):
        fn()
