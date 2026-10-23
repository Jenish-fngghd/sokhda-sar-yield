"""T4-T7: the Round 3 deliverable -- health index, final yield forecast, aggregation.

THE STRUCTURE, and why it is this and not something more ambitious.

    yield_forecast[p]  =  anchor(crop[p]) x modifier[p]

The anchor sets the LEVEL, which no label-free method can obtain from imagery alone. The
modifier distributes it across plots, and the modifier is where the SAR does its work. The
measured bottleneck (Phase 2, R-C C-4) is that in Round 2 the crop label explained
eta^2 = 0.820 of yield variance, leaving the SAR only 0.180, while two other teams reached
0.474 and 0.499. **Moving that number is the single highest-value change available**, so the
modifier is normalised WITHIN crop and its contribution is reported as one auditable figure.

★ THIS IS A FINAL FORECAST, NOT A YIELD-TO-DATE. Round 2 shipped `yield_estimate_to_date` and
multiplied DOWN by a completion factor. Round 3's deliverable is the final yield, so the
completion factor's role inverts: it is NOT applied to the forecast. It is used only to derive
`yield_to_date` as a separate column, which is what makes check D1 possible -- the one
consistency test no other team can run, because only we have a Round 2 to compare against.

★ 29 OCTOBER IS EXCLUDED FROM THE LEVEL FEATURES. P1-3c measured a +4.03 dB wetness term on
that date, P1-6b showed it is canopy-dependent so no scalar removes it, T2 showed the Water
Cloud Model cannot separate it either (both its controls failed), and Phase 2 A-2 established
the cause: a documented state-wide rain event, 239 talukas, over 10 lakh ha damaged, 23-28
October. The date is kept as a MOISTURE observation and as an ensemble member, never in the
primary level features. Both variants are computed and their spread is reported (D6).

Writes results/submission.csv, results/tables/p3_village_summary.csv,
p3_zone_summary.csv, p3_ensemble.csv and results/d4_debug.csv.

Run:  py -3.12 src/p3_build.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUX, CROPS, DATES, RESULTS, TABLES, log

import geopandas as gpd
from common import FARMS

UTM = 32643
VILLAGE_ID = 1              # R2's decoded dummy submission, NOT the shapefile's 22
YIELD_SPREAD = 0.30         # within-village farm-level yield CV; 20-35% is the common range
ZONE_M = 500.0              # sub-village grid; Aggregation is 15 points
MIN_ZONE_FARMS = 5
# level dates: 29 Oct excluded for the reason in the docstring
LEVEL_DATES = [d for d in DATES if d != "20251029"]


def z(x):
    """Robust z-score. Median and MAD, so one waterlogged plot cannot set the scale."""
    x = np.asarray(x, dtype="float64")
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return np.zeros_like(x)
    med = np.median(x[ok])
    mad = np.median(np.abs(x[ok] - med)) * 1.4826
    out = np.zeros_like(x)
    out[ok] = (x[ok] - med) / (mad if mad > 1e-9 else 1.0)
    return np.clip(out, -4, 4)


def health_parts(f, use_29oct):
    """The family scores the health index combines. Six dates instead of four.

      level     peak-canopy backscatter (14 Aug)
      growth    14 Aug minus 19 Jun -- the ONLY geometry-matched pair in the stack
                (28.692 vs 28.768 deg, 0.076 apart); every other pair spans 2.8-6.6 deg.
                Village median subtracted, because 19 Jun sits on monsoon-onset wet soil
                and a village-wide wetting is a common mode that cancels in any contrast.
      uniform   within-farm CV at peak canopy, negated: a uniform stand is a healthy stand
      persist   season integral -- canopy held across the series, not in one lucky date
      senesce   NEW IN R3. Late-season slope, which four dates could not measure at all.
                Built from 13 Oct and 12 Nov only when 29 Oct is excluded.
    """
    dates = DATES if use_29oct else LEVEL_DATES
    growth = (f["g0_db_20250814"] - f["g0_db_20250619"]).to_numpy()
    growth = growth - np.nanmedian(growth)

    lin = f[[f"g0_lin_{d}" for d in dates]].to_numpy()
    doy = np.array([pd.Timestamp(d).dayofyear for d in dates], dtype="float64")
    integral = np.trapezoid(np.where(np.isfinite(lin), lin, np.nan), doy, axis=1)

    late = ["20251013", "20251112"] if not use_29oct else ["20251013", "20251029", "20251112"]
    L = f[[f"g0_db_{d}" for d in late]].to_numpy()
    ld = np.array([pd.Timestamp(d).dayofyear for d in late], dtype="float64")
    x = ld - ld.mean()
    senesce = np.nansum((L - np.nanmean(L, axis=1, keepdims=True)) * x, axis=1) / (x ** 2).sum()

    return {"level": z(f["g0_db_20250814"].to_numpy()),
            "growth": z(growth),
            "uniform": -z(f["cv_20250814"].to_numpy()),
            "persist": z(integral),
            "senesce": z(senesce)}


def derive_weights(parts):
    """Weights from redundancy alone: w_k ~ 1 / sum_j |rho(part_k, part_j)|.

    Blind to every witness by construction -- it reads only the feature matrix. Weights
    chosen by watching NDVI would turn the held-out witness into a fitting target and
    forfeit the independence that makes the validation worth anything.
    """
    X = pd.DataFrame(parts)
    C = X.corr(method="spearman").abs().to_numpy()
    inv = 1.0 / C.sum(axis=1)
    return dict(zip(X.columns, inv / inv.sum()))


def health_index(parts, W, crop):
    """Composite vigour, 0-100, scored WITHIN crop.

    Within crop because cotton and groundnut differ by several dB for reasons unrelated to
    health, so a pooled score would largely re-measure crop type.

    The scale is a bounded transform of the within-crop robust z, NOT a percentile rank. A
    percentile forces mean 50 and a fixed spread on every crop whatever the data says, which
    makes the required village-level summary vacuous by construction. The normal CDF keeps
    the ordering identical while letting the DISTRIBUTION move.
    """
    S = sum(W[k] * v for k, v in parts.items() if k in W)
    out = np.full(len(S), np.nan)
    for c in CROPS:
        m = (crop == c).to_numpy()
        if m.sum() < 3:
            continue
        out[m] = 100.0 * norm.cdf(z(S[m]) / 2.0)
    return np.clip(out, 0, 100)


def modifier(hi, crop, area):
    """Per-plot yield multiplier, area-weighted mean exactly 1.0 WITHIN each crop.

    Forcing the within-crop area-weighted mean to 1.0 is what makes the anchor mean the
    thing it claims to be: the crop's area-weighted yield comes out equal to the anchor,
    so the modifier adds spread without moving the level. It also makes the SAR's
    contribution auditable as a single number -- the variance the modifier introduces.
    """
    out = np.ones(len(hi))
    for c in CROPS:
        m = (crop == c).to_numpy() & np.isfinite(hi)
        if m.sum() < 3:
            continue
        v = 1.0 + YIELD_SPREAD * z(hi[m]) / 2.0
        v = np.clip(v, 0.35, 1.9)
        out[m] = v / np.average(v, weights=area[m])       # exact by construction
    return out


def zones(farms):
    """Fixed 500 m grid over the village. One row is not a spatial product."""
    x0, y0 = farms.total_bounds[0], farms.total_bounds[1]
    cx, cy = farms.centroid.x.to_numpy(), farms.centroid.y.to_numpy()
    return (np.floor((cx - x0) / ZONE_M).astype(int) * 1000
            + np.floor((cy - y0) / ZONE_M).astype(int))


def main():
    f = pd.read_csv(TABLES / "p1_farm_features.csv")
    lab = pd.read_csv(AUX / "inherited" / "label_dist.csv")
    unc = pd.read_csv(AUX / "inherited" / "uncert.csv")[["farm_id", "health_sd_pts"]]
    gee = pd.read_csv(AUX / "inherited" / "gee.csv")[["farm_id", "dw_mode"]]
    anc = pd.read_csv(AUX.parent / "data_aux" / "anchors_r3.csv").set_index("crop")

    d = f.merge(lab, on="farm_id").merge(unc, on="farm_id", how="left") \
         .merge(gee, on="farm_id", how="left")
    assert len(d) == 966, len(d)
    crop, area = d.crop_type, d.area_ha.to_numpy()

    # ---- health, both ensemble variants ----
    variants = {}
    for name, use29 in (("primary_no29oct", False), ("alt_with29oct", True)):
        parts = health_parts(d, use29)
        W = derive_weights(parts)
        hi = health_index(parts, W, crop)
        variants[name] = hi
        log("p3_build.health", variant=name, weights={k: round(v, 4) for k, v in W.items()},
            median=round(float(np.nanmedian(hi)), 2),
            iqr=round(float(np.nanpercentile(hi, 75) - np.nanpercentile(hi, 25)), 2))

    hi = variants["primary_no29oct"]
    d["health_index"] = np.round(hi, 2)
    d["health_alt"] = np.round(variants["alt_with29oct"], 2)
    d["health_ensemble_spread"] = np.abs(d.health_index - d.health_alt)

    # ---- yield: FINAL forecast, no completion multiplier ----
    mod = modifier(hi, crop, area)
    d["modifier"] = np.round(mod, 4)
    base = crop.map(anc.anchor_kg_ha * anc.adj_2025_26).to_numpy() / 1000.0   # t/ha
    d["yield_forecast_t_ha"] = np.round(base * mod, 4)
    # the separate column that makes D1 possible
    d["yield_to_date_t_ha"] = np.round(d.yield_forecast_t_ha
                                       * crop.map(anc.completion_at_12nov).to_numpy(), 4)

    # ---- freeze-dividend transparency columns ----
    d["label_confidence"] = lab.set_index("farm_id").loc[d.farm_id, "p_assigned"].to_numpy()
    d["label_entropy"] = lab.set_index("farm_id").loc[d.farm_id, "entropy"].to_numpy()
    d["health_sd_pts"] = d.health_sd_pts                       # calibrated SAMPLING noise only
    # dw_mode is the ZONAL MEAN of Dynamic World's mode band, so it is a float, not a
    # label. Round it to the class index: 1 trees, 2 grass, 3 flooded_veg, 4 crops,
    # 5 shrub/scrub, 6 built. Crop-like is {crops, grass}. Rounding reproduces the R2
    # census exactly (689/91/70/50/36/25), which is the check that the mapping is right.
    dw = d.dw_mode.round()
    noncrop_dw = ~dw.isin([4.0, 2.0])
    low = d.health_index <= np.nanpercentile(d.health_index, 10)
    d["flag_non_crop"] = np.where(noncrop_dw & low, "both", np.where(noncrop_dw, "dw_only", ""))
    d["flag_qc"] = d.qc_flag
    log("p3_build.flags",
        dw_non_crop=int(noncrop_dw.sum()),
        dw_non_crop_area_pct=round(100 * area[noncrop_dw.to_numpy()].sum() / area.sum(), 2),
        both_rules=int((d.flag_non_crop == "both").sum()),
        both_area_pct=round(100 * area[(d.flag_non_crop == "both").to_numpy()].sum()
                            / area.sum(), 2),
        note="FLAG, never a filter -- they still enter crop shares and the village aggregate")

    # ---- T3: eta^2, how much of yield variance is just the crop label? ----
    # ★ The 0.820 baseline in the brief does NOT reproduce from Round 2's own shipped
    # submission, which measures 0.9248 unweighted / 0.9305 area-weighted. So the
    # comparison is made against R2's actual shipped column, re-measured here with the
    # identical estimator, rather than against a number we cannot reproduce.
    def eta2(vals, w):
        vals = np.asarray(vals, dtype="float64")
        grand = np.average(vals, weights=w)
        ssb = sum(w[(crop == c).to_numpy()].sum()
                  * (np.average(vals[(crop == c).to_numpy()],
                                weights=w[(crop == c).to_numpy()]) - grand) ** 2
                  for c in CROPS)
        return float(ssb / np.sum(w * (vals - grand) ** 2))

    y = d.yield_forecast_t_ha.to_numpy()
    ones = np.ones(len(y))
    r2sub = pd.read_csv(Path(__file__).resolve().parents[2] / "phase2_crop_type_health"
                        / "results" / "submission.csv")
    e_r2 = eta2(r2sub.yield_estimate_to_date.to_numpy(), ones)
    e_fc = eta2(y, ones)
    log("p3_build.eta2", r2_shipped_measured=round(e_r2, 4), brief_claimed=0.820,
        brief_reproduces=False,
        r3_forecast=round(e_fc, 4),
        r3_forecast_area_weighted=round(eta2(y, area), 4),
        r3_to_date=round(eta2(d.yield_to_date_t_ha.to_numpy(), ones), 4),
        sar_share_r2=round(1 - e_r2, 4), sar_share_r3=round(1 - e_fc, 4),
        verdict="IMPROVED on R2 as measured" if e_fc < e_r2 else "NOT IMPROVED",
        caveat="eta2 is dominated by the anchor spread across crops (0.73-2.51 t/ha), "
               "which is real agronomy; the within-crop ranking is validated separately "
               "against same-day S2 NDVI in T1")

    # ---- submission ----
    sub = pd.DataFrame({"village_id": VILLAGE_ID, "farm_id": d.farm_id,
                        "crop_type": crop, "health_index": d.health_index,
                        "yield_forecast_t_ha": d.yield_forecast_t_ha})
    assert len(sub) == 966 and sub.farm_id.is_unique
    assert sub.village_id.eq(VILLAGE_ID).all()
    assert sub.yield_forecast_t_ha.max() < 25.0, "looks like kg/ha -- a 1000x unit error"
    assert sub.yield_forecast_t_ha.min() >= 0
    assert sub.health_index.between(0, 100).all()
    assert sub.yield_forecast_t_ha.nunique() > 20, "the forecast has collapsed"
    sub.to_csv(RESULTS / "submission.csv", index=False)
    d.to_csv(RESULTS / "d4_debug.csv", index=False)

    # ---- aggregation: production = sum(yield x area), never a mean of rates ----
    rows = []
    for c in CROPS:
        m = (crop == c).to_numpy()
        prod = float(np.sum(y[m] * area[m]))
        rows.append({"crop": c, "n_plots": int(m.sum()),
                     "area_ha": round(float(area[m].sum()), 3),
                     "area_share_pct": round(100 * area[m].sum() / area.sum(), 2),
                     "anchor_t_ha": round(float(anc.loc[c, "anchor_kg_ha"]
                                                * anc.loc[c, "adj_2025_26"] / 1000), 4),
                     "yield_forecast_t_ha_aw": round(prod / area[m].sum(), 4),
                     "production_t": round(prod, 3),
                     "yield_to_date_t_ha_aw": round(float(np.sum(
                         d.yield_to_date_t_ha.to_numpy()[m] * area[m]) / area[m].sum()), 4),
                     "health_aw": round(float(np.average(hi[m], weights=area[m])), 2),
                     "epistemic_object": {"Cotton": "FORECAST",
                                          "Groundnut": "NEAR-COMPLETE MEASUREMENT"}
                                         .get(c, "RETROSPECTIVE RECONSTRUCTION")})
    vill = pd.DataFrame(rows)
    tot = float(np.sum(y * area))
    vill.loc[len(vill)] = {"crop": "ALL", "n_plots": 966,
                           "area_ha": round(float(area.sum()), 3), "area_share_pct": 100.0,
                           "anchor_t_ha": np.nan,
                           "yield_forecast_t_ha_aw": round(tot / area.sum(), 4),
                           "production_t": round(tot, 3),
                           "yield_to_date_t_ha_aw": round(float(np.sum(
                               d.yield_to_date_t_ha.to_numpy() * area) / area.sum()), 4),
                           "health_aw": round(float(np.average(hi, weights=area)), 2),
                           "epistemic_object": "THREE DIFFERENT OBJECTS -- see per-crop rows"}
    vill.to_csv(TABLES / "p3_village_summary.csv", index=False)

    # ---- sub-village zones ----
    fa = gpd.read_file(FARMS).to_crs(UTM)
    zid = zones(fa)
    zrows = []
    for zz in np.unique(zid):
        m = zid == zz
        if m.sum() < MIN_ZONE_FARMS:
            continue
        zrows.append({"zone_id": int(zz), "n_farms": int(m.sum()),
                      "area_ha": round(float(area[m].sum()), 3),
                      "health_aw": round(float(np.average(hi[m], weights=area[m])), 2),
                      "yield_aw_t_ha": round(float(np.sum(y[m] * area[m]) / area[m].sum()), 4),
                      "production_t": round(float(np.sum(y[m] * area[m])), 3),
                      "health_sd": round(float(np.nanstd(hi[m])), 2),
                      "dominant_crop": crop[m].mode().iloc[0]})
    zon = pd.DataFrame(zrows)
    zon.to_csv(TABLES / "p3_zone_summary.csv", index=False)

    pd.DataFrame({"farm_id": d.farm_id, "primary": d.health_index,
                  "alt_with_29oct": d.health_alt,
                  "spread": d.health_ensemble_spread}).to_csv(
        TABLES / "p3_ensemble.csv", index=False)

    log("p3_build.done", production_t=round(tot, 1),
        village_yield_t_ha=round(tot / area.sum(), 3), area_ha=round(float(area.sum()), 1),
        zones=len(zon), zone_health_spread_pts=round(float(zon.health_aw.max()
                                                          - zon.health_aw.min()), 1),
        ensemble_spread_median_pts=round(float(d.health_ensemble_spread.median()), 2))

    print("\nVILLAGE SUMMARY -- production = sum(yield x area)")
    print(vill.to_string(index=False))
    print(f"\nzones: {len(zon)} of >= {MIN_ZONE_FARMS} farms, "
          f"health spread {zon.health_aw.min():.1f} to {zon.health_aw.max():.1f} "
          f"({zon.health_aw.max() - zon.health_aw.min():.1f} points behind the single row)")


if __name__ == "__main__":
    main()
