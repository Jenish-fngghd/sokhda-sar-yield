"""Stage I0.5 -- exploratory data analysis, BEFORE any preprocessing.

Why this runs first: the single fatal risk in the plan is geocoding/boundary
misalignment. If the farm polygons do not sit on the fields, every feature we
extract afterwards is attached to the wrong plot and nothing downstream can be
trusted. That is checkable right now, for free, because Capella ships a
geocoded preview (EPSG:32643, full 0.735 m resolution) alongside each SLC.

The previews are uint8 and display-scaled, so they are useless radiometrically
-- we never take a measurement from them. They are used here only for geometry
and visual interpretation. All radiometry comes from the SLC in stage I1.

Outputs figures to results/figures/ and a scene metadata table to results/.
Run:  python src/eda.py
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window, from_bounds

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import (DATES, FARMS, FIGURES, INCIDENCE_DEG, RESULTS, RESULTS_AUX, VILLAGE,
                    log, slc_path)

import geopandas as gpd

UTM = 32643
DATE_LABEL = {"20250606": "Jun 06\npre-monsoon", "20250619": "Jun 19\npre-monsoon",
              "20250814": "Aug 14\npeak vegetative", "20251013": "Oct 13\npost-monsoon"}


def preview_path(date):
    return sorted(slc_path(date).parent.glob("*GEO*preview.tif"))[0]


def load_vectors():
    """Farms + village in UTM, geometries repaired explicitly.

    GeoPandas silently autocorrects winding order on read; we call make_valid
    ourselves so the repair is visible and countable rather than invisible. [F2]
    """
    farms = gpd.read_file(FARMS).to_crs(UTM)
    village = gpd.read_file(VILLAGE).to_crs(UTM)
    n_bad = int((~farms.is_valid).sum())
    farms["geometry"] = farms.geometry.make_valid()
    farms["area_m2"] = farms.area
    log("eda.vectors", n_farms=len(farms), n_repaired=n_bad,
        total_ha=round(farms.area_m2.sum() / 1e4, 1),
        median_ha=round(farms.area_m2.median() / 1e4, 3),
        n_under_10m2=int((farms.area_m2 < 10).sum()))
    return farms, village


def read_preview_window(date, bounds, max_px=2400):
    """Read a preview over `bounds` (UTM), decimated to at most max_px on the long side."""
    with rasterio.open(preview_path(date)) as src:
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        step = max(1, int(max(win.width, win.height) // max_px))
        arr = src.read(1, window=win,
                       out_shape=(int(win.height // step), int(win.width // step)))
        extent_tf = src.window_transform(win)
    h, w = arr.shape
    extent = (extent_tf.c, extent_tf.c + w * step * extent_tf.a,
              extent_tf.f + h * step * extent_tf.e, extent_tf.f)
    return arr, extent


def fig_scene_metadata():
    """Per-scene acquisition parameters straight from *_extended.json."""
    rows = []
    for d in DATES:
        js = sorted(slc_path(d).parent.glob("*_extended.json"))[0]
        meta = json.loads(js.read_text(encoding="utf8"))
        img = meta["collect"]["image"]
        radar = meta["collect"].get("radar", {})
        rows.append(dict(
            date=d,
            incidence_deg=round(float(meta["collect"].get("center_pixel", {})
                                      .get("incidence_angle", INCIDENCE_DEG[d])), 3),
            scale_factor=img.get("scale_factor"),
            look=meta["collect"].get("radar", {}).get("pointing", radar.get("pointing")),
            range_res_m=round(float(img.get("range_resolution", np.nan)), 3),
            azimuth_res_m=round(float(img.get("azimuth_resolution", np.nan)), 3),
            pixel_spacing_col=round(float(img.get("pixel_spacing_column", np.nan)), 4),
            pixel_spacing_row=round(float(img.get("pixel_spacing_row", np.nan)), 4),
        ))
    import pandas as pd
    df = pd.DataFrame(rows)
    out = RESULTS_AUX / "scene_metadata.csv"
    df.to_csv(out, index=False)
    log("eda.metadata", path=str(out.name),
        incidence_spread_deg=round(df.incidence_deg.max() - df.incidence_deg.min(), 3))
    return df


def fig_alignment(farms, village):
    """★ THE critical figure: do the farm polygons land on actual fields?

    Three panels: whole village, and two zooms on dense plot clusters. If the
    boundaries are offset from the field edges visible in the radar image, stop
    and fix geocoding before doing anything else.
    """
    b = village.total_bounds
    arr, extent = read_preview_window("20251013", b)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    vmin, vmax = np.percentile(arr[arr > 0], [2, 98]) if (arr > 0).any() else (0, 255)

    axes[0].imshow(arr, cmap="gray", extent=extent, vmin=vmin, vmax=vmax, origin="upper")
    farms.boundary.plot(ax=axes[0], color="#00e5ff", linewidth=0.35)
    village.boundary.plot(ax=axes[0], color="#ff3b30", linewidth=1.4)
    axes[0].set_title("Oct 13 preview + all 966 farms\n(village outline red)", fontsize=10)

    # two zooms centred on the densest clusters of plots
    cent = np.c_[farms.centroid.x, farms.centroid.y]
    for ax, frac in zip(axes[1:], [0.25, 0.75]):
        cx, cy = np.quantile(cent, frac, axis=0)
        half = 350.0
        zb = (cx - half, cy - half, cx + half, cy + half)
        za, zext = read_preview_window("20251013", zb, max_px=1400)
        ax.imshow(za, cmap="gray", extent=zext, origin="upper",
                  vmin=vmin, vmax=vmax)
        farms.boundary.plot(ax=ax, color="#00e5ff", linewidth=0.9)
        ax.set_xlim(zb[0], zb[2]); ax.set_ylim(zb[1], zb[3])
        ax.set_title(f"700 m zoom @ q{frac:g}\nfield edges should track the cyan lines", fontsize=10)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("ALIGNMENT GATE -- boundaries vs radar field edges "
                 "(preview is display-scaled uint8; geometry only)", fontsize=12)
    fig.tight_layout()
    p = FIGURES / "eda_01_alignment.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    log("eda.fig", name=p.name)


def fig_quicklooks(farms, village):
    """The four dates side by side over the village -- the season at a glance."""
    b = village.total_bounds
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, d in zip(axes, DATES):
        arr, extent = read_preview_window(d, b)
        pos = arr[arr > 0]
        vmin, vmax = np.percentile(pos, [2, 98]) if pos.size else (0, 255)
        ax.imshow(arr, cmap="gray", extent=extent, vmin=vmin, vmax=vmax, origin="upper")
        village.boundary.plot(ax=ax, color="#ff3b30", linewidth=1.0)
        ax.set_title(f"{DATE_LABEL[d]}\nincidence {INCIDENCE_DEG[d]:.2f}deg", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Sokhda across the 2025 kharif season -- Capella X-band HH previews", fontsize=13)
    fig.tight_layout()
    p = FIGURES / "eda_02_quicklooks.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    log("eda.fig", name=p.name)


def fig_geometry(farms, village):
    """Plot-size distribution and where the degenerate polygons are. [F2]"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    ha = farms.area_m2 / 1e4

    axes[0].hist(ha, bins=60, color="#3b7dd8", edgecolor="white", linewidth=0.4)
    axes[0].axvline(ha.median(), color="#ff3b30", ls="--",
                    label=f"median {ha.median():.3f} ha")
    axes[0].set_xlabel("plot area (ha)"); axes[0].set_ylabel("farms")
    axes[0].set_title(f"Plot sizes -- {(ha < 0.2).mean()*100:.0f}% under 0.2 ha", fontsize=10)
    axes[0].legend(fontsize=8)

    order = np.sort(ha.values)
    axes[1].plot(np.arange(len(order)) / len(order) * 100, np.cumsum(order) / order.sum() * 100,
                 color="#3b7dd8")
    axes[1].set_xlabel("% of farms (smallest first)"); axes[1].set_ylabel("% of total area")
    axes[1].set_title("Area concentration", fontsize=10); axes[1].grid(alpha=0.3)

    village.boundary.plot(ax=axes[2], color="#999", linewidth=0.8)
    farms.plot(ax=axes[2], color="#dfe7f3", edgecolor="#9bb0cc", linewidth=0.2)
    bad = farms[farms.area_m2 < 10]
    if len(bad):
        bad.centroid.plot(ax=axes[2], color="#ff3b30", markersize=45, label=f"{len(bad)} plots <10 m2")
        axes[2].legend(fontsize=8)
    axes[2].set_title("Degenerate plots -- must survive to the CSV", fontsize=10)
    axes[2].set_xticks([]); axes[2].set_yticks([])

    fig.tight_layout()
    p = FIGURES / "eda_03_geometry.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    log("eda.fig", name=p.name)


def fig_slc_amplitude(n_win=6, size=512):
    """Raw |z| distributions per date, sampled from the SLC itself.

    This is the last look at the data before calibration, and it tells us the
    dynamic range and whether exact zeros exist (they break dB). Sampled from
    evenly spaced windows down the strip rather than one block, so the
    distribution is not dominated by a single land cover.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    stats = {}
    for d in DATES:
        with rasterio.open(slc_path(d)) as s:
            vals = []
            for k in range(n_win):
                row = int((k + 0.5) / n_win * (s.height - size))
                col = max(0, s.width // 2 - size // 2)
                a = s.read(1, window=Window(col, row, size, size)).astype("complex64")
                vals.append(np.abs(a).ravel())
            m = np.concatenate(vals)
        stats[d] = dict(n=int(m.size), zeros=int((m == 0).sum()),
                        p1=float(np.percentile(m, 1)), med=float(np.median(m)),
                        p99=float(np.percentile(m, 99)), mx=float(m.max()))
        pos = m[m > 0]
        axes[0].hist(pos, bins=200, range=(0, np.percentile(pos, 99.5)),
                     histtype="step", label=DATE_LABEL[d].replace("\n", " "))
        axes[1].hist(20 * np.log10(pos), bins=200, histtype="step",
                     label=DATE_LABEL[d].replace("\n", " "))
    axes[0].set_xlabel("|z| (raw DN)"); axes[0].set_ylabel("count")
    axes[0].set_title("Raw SLC amplitude", fontsize=10); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("20 log10 |z|  (uncalibrated dB)")
    axes[1].set_title("Same, log scale -- NOT yet calibrated", fontsize=10)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    p = FIGURES / "eda_04_slc_amplitude.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    for d, s in stats.items():
        log("eda.slc_amplitude", date=d, **{k: round(v, 2) if isinstance(v, float) else v
                                            for k, v in s.items()})
    return stats


def fig_landcover_candidates(farms, village):
    """Locate dark (water) and bright (built-up) targets for the I1 sanity gate.

    Two things depend on finding these:
      - the land-cover dB gate: water near NESZ, built-up strongly positive,
        cropland -5..-12 dB. That gate decides whether our calibration is right.
      - water-point referencing [E4]: every date is referenced to an in-scene
        water surface to cancel Capella's undeclared absolute calibration [B9].
    Candidates are proposed here from the preview and CONFIRMED on calibrated
    SLC in I1 -- a uint8 preview cannot settle a radiometric question.
    """
    from scipy.ndimage import uniform_filter

    b = village.buffer(1500).total_bounds
    arr, extent = read_preview_window("20251013", b, max_px=1600)
    nodata = arr == 0                      # swath exterior -- must be excluded, not
    valid = ~nodata                        # treated as "dark". [F3]

    # Water is spatially COHERENT; dark speckle is not. Smoothing before
    # thresholding is what separates the two -- a raw low-percentile threshold on
    # single-look data just selects speckle troughs (and, if nodata is not masked,
    # the swath exterior).
    a = arr.astype("float32")
    a[nodata] = np.nan
    sm = uniform_filter(np.nan_to_num(a), 9) / np.maximum(uniform_filter(valid.astype("float32"), 9), 1e-6)
    sm[nodata] = np.nan
    lo, hi = np.nanpercentile(sm, [1.0, 99.7])
    dark, bright = (sm <= lo) & valid, (sm >= hi) & valid

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pos = arr[valid]
    axes[0].imshow(arr, cmap="gray", extent=extent, origin="upper",
                   vmin=np.percentile(pos, 2), vmax=np.percentile(pos, 98))
    village.boundary.plot(ax=axes[0], color="#ff3b30", linewidth=1.0)
    axes[0].set_title("Oct 13, village + 1.5 km buffer", fontsize=10)

    axes[1].imshow(np.where(nodata, np.nan, dark), cmap="Blues", extent=extent, origin="upper")
    village.boundary.plot(ax=axes[1], color="#ff3b30", linewidth=1.0)
    axes[1].set_title(f"Smoothed darkest 1% (DN<={lo:.1f}), nodata masked\n"
                      f"candidate WATER -- {dark.sum()/valid.sum()*100:.2f}% of valid", fontsize=10)

    axes[2].imshow(np.where(nodata, np.nan, bright), cmap="Reds", extent=extent, origin="upper")
    village.boundary.plot(ax=axes[2], color="#ff3b30", linewidth=1.0)
    axes[2].set_title(f"Smoothed brightest 0.3% (DN>={hi:.1f})\ncandidate BUILT-UP", fontsize=10)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Land-cover reference candidates for the I1 dB gate and water-point referencing "
                 "-- CANDIDATES ONLY, confirmed on calibrated SLC in I1", fontsize=12)
    fig.tight_layout()
    p = FIGURES / "eda_05_landcover_candidates.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    log("eda.fig", name=p.name, dark_frac_pct=1.0, bright_frac_pct=0.3)


def fig_coverage(farms, village):
    """★ Per-farm data coverage per date -- overturns [A4]'s "100% coverage" claim.

    [A4] checked that the village BBOX sits inside the scene bounds. It does. But
    the swath is a rotated rectangle and its NW edge cuts through the village, so
    bbox containment says nothing about whether a given farm has pixels. Measured
    here per farm per date on the geocoded preview.

    Caveat carried forward: this is the GEOCODED PREVIEW footprint. Our own
    GCP-based geocoding in I1 may recover a few of these farms, so the numbers are
    re-measured there rather than trusted from here.
    """
    b = farms.total_bounds
    cov = np.full((len(farms), len(DATES)), np.nan)
    for j, d in enumerate(DATES):
        with rasterio.open(preview_path(d)) as s:
            win = from_bounds(*b, transform=s.transform).round_offsets().round_lengths()
            win = win.intersection(Window(0, 0, s.width, s.height))
            arr = s.read(1, window=win)
            tf = s.window_transform(win)
        valid = arr > 0
        from rasterio.features import geometry_mask
        for i, geom in enumerate(farms.geometry):
            m = geometry_mask([geom], out_shape=arr.shape, transform=tf, invert=True)
            if m.sum():
                cov[i, j] = valid[m].mean()
    np.save(RESULTS / "cache" / "farm_preview_coverage.npy", cov)

    usable = cov >= 0.5
    n_ok = usable.sum(axis=1)
    farms = farms.copy()
    farms["n_dates"] = n_ok

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cmap = matplotlib.colors.ListedColormap(["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"])
    village.boundary.plot(ax=axes[0], color="#333", linewidth=1.0)
    farms.plot(ax=axes[0], column="n_dates", cmap=cmap, vmin=-0.5, vmax=4.5,
               edgecolor="#666", linewidth=0.15, legend=True,
               legend_kwds={"label": "usable dates (>=50% pixels)", "shrink": 0.7})
    axes[0].set_title(f"Coverage per farm -- {int((n_ok == 4).sum())}/966 have all 4 dates",
                      fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].bar(range(5), [int((n_ok == k).sum()) for k in range(5)],
                color=["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"], edgecolor="#444")
    for k in range(5):
        axes[1].text(k, int((n_ok == k).sum()), str(int((n_ok == k).sum())),
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_xlabel("number of usable dates"); axes[1].set_ylabel("farms")
    axes[1].set_title("Farms by usable-date count", fontsize=10)

    per_date = [int((~usable[:, j]).sum()) for j in range(len(DATES))]
    axes[2].bar([DATE_LABEL[d].split("\n")[0] for d in DATES], per_date,
                color="#3b7dd8", edgecolor="#444")
    for k, v in enumerate(per_date):
        axes[2].text(k, v, str(v), ha="center", va="bottom", fontsize=9)
    axes[2].set_ylabel("farms without usable data")
    axes[2].set_title("Unusable farms per date", fontsize=10)

    fig.suptitle("COVERAGE IS NOT 100% -- the swath edge clips the village NW corner "
                 "(corrects [A4])", fontsize=12)
    fig.tight_layout()
    p = FIGURES / "eda_06_coverage.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    log("eda.coverage", all4=int((n_ok == 4).sum()), none=int((n_ok == 0).sum()),
        partial=int(((n_ok > 0) & (n_ok < 4)).sum()),
        per_date_unusable=dict(zip(DATES, per_date)))
    return cov


def main():
    log("eda.start")
    farms, village = load_vectors()
    fig_scene_metadata()
    fig_alignment(farms, village)
    fig_quicklooks(farms, village)
    fig_geometry(farms, village)
    fig_slc_amplitude()
    fig_landcover_candidates(farms, village)
    fig_coverage(farms, village)
    log("eda.done", figures=str(FIGURES))
    print(f"\nFigures written to {FIGURES}")


if __name__ == "__main__":
    main()
