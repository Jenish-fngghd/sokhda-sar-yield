# Phase 3 — Final Yield Forecast, Village Aggregation and Uncertainty

Phase 3 covers the 966 farm plots in Sokhda village (447.5 ha), using six Capella X-band HH SLC scenes from 6 June to 12 November 2025. It produces:

- a final yield forecast per plot;
- yield aggregated by crop at village and sub-village level;
- a health index and uncertainty per plot.

Most of the paper's methodology and validation is implemented here.

| | |
|:--|:--|
| **Paper sections** | SAR preprocessing; crop health index; yield forecasting and aggregation; validation and uncertainty |
| **Competition** | [ANRF AISEHack 2.0, Round 3](https://www.kaggle.com/competitions/anrf-aise-hack-2-0-round-3-sar-crop-yield-forecasting) |
| **Deliverable** | `results/submission.csv` (966 rows), md5 `8304407ed23bc67168fb26fca21a03d1` |
| **Result** | 710.0 t over 447.5 ha (1.587 t/ha); village 90% interval 598–918 t |
| **Requirements** | Phase 3 competition download (about 3.2 GB) |

---

## Contents

1. [Task](#1-task)
2. [Data](#2-data)
3. [Method](#3-method)
4. [Results](#4-results)
5. [Validation and Uncertainty](#5-validation-and-uncertainty)
6. [Repository Contents](#6-repository-contents)
7. [Usage](#7-usage)
8. [Script Reference](#8-script-reference)
9. [Limitations](#9-limitations)

---

## 1. Task

For each plot, Phase 3 produces:

- a final yield forecast;
- plot-level statistics supporting it;
- village-level summaries by crop.

Crop type is carried forward from Phase 2.

Output columns: `village_id, farm_id, crop_type, health_index, yield_forecast_t_ha` (t/ha; cotton as lint).

## 2. Data

All scenes are Stripmap, HH, ascending, about 9.65 GHz, annotated `beta_nought`. Each has 225 ground control points and vendor metadata, including orbit state vectors.

| Date | Local time (IST) | Look | Centre incidence | Scale factor | NESZ peak (dB) |
|:--|:--|:--|--:|--:|--:|
| 2025-06-06 | 12:55 | left | 35.244° | 0.00212186 | −26.13 |
| 2025-06-19 | 07:44 | left | 28.768° | 0.00236205 | −27.76 |
| 2025-08-14 | 08:41 | left | 28.692° | 0.00198903 | −27.97 |
| 2025-10-13 | 07:56 | left | 31.528° | 0.00136443 | −27.35 |
| 2025-10-29 | 01:37 | right | 29.840° | 0.00155765 | −27.74 |
| 2025-11-12 | 19:22 | left | 29.746° | 0.00162432 | −27.72 |

The first four scenes are identical to those of Phase 2.

**Ancillary inputs:**

| File | Contents |
|:--|:--|
| `data_aux/anchors_r3.csv` | Yield anchor per crop: source, base year, 2025–26 adjustment, completion on 12 November |
| `data_aux/inherited/label_dist.csv` | Phase 2 crop map and class probabilities |
| `data_aux/inherited/uncert.csv` | Per-plot sampling uncertainty of the health index |
| `data_aux/inherited/gee.csv` | Dynamic World and WorldCereal values per plot |
| `data_aux/inherited/anchors.csv` | Vadodara district crop statistics, 1997–2012 |
| `data_aux/inherited/consensus.csv`, `orion_feat.csv` | Third-party control tables |

`data_aux/inherited/README.md` documents the source, terms and use of each inherited table.

## 3. Method

```
yield_forecast[plot] = anchor(crop) × modifier[plot]
production           = Σ (yield × area)
```

### 3.1 Preprocessing — `p1_prep.py`, `p1_calib.py`

- **Radiometry:** β⁰ = SF² · |z|², σ⁰ = β⁰ · sin θ, γ⁰ = β⁰ · tan θ. The scale factor applies to amplitude and is therefore squared for power. θ is a per-pixel incidence angle computed from the orbit state vectors.
- **Calibration check:** with SF², the darkest content of each scene lies above the vendor's noise floor on all six scenes, by +1.86 dB on average. Without the square it would lie +29.3 dB above, which is implausible. σ⁰ and γ⁰ fall below the floor on 6 and 4 scenes, so the noise-floor test is applied in β⁰.
- **Geocoding:** each scene's control points, with power averaged onto 5 m and 2 m grids.
- **Scene constants:** acquisition dates are scanned from the data, and every scene constant is read from the vendor's `_extended.json`.

### 3.2 Features — `p1_features.py`

- **966 plots × 111 features,** extracted with a negative-buffer ladder of −5, −2 and 0 m.
- **Gates:** a coverage gate and a noise-floor gate per plot and date.
- **Coverage:** 832 plots are observed on all dates, 105 partly and 29 never. All 29 are cotton plots at the north-west edge of the swath.

### 3.3 Treatment of 29 October

- **Conditions:**
  - wind at acquisition was 14.1 m/s, against 6.7–9.6 m/s on the other dates;
  - ERA5-Land records 81.2 mm of rain on 26–28 October;
  - topsoil moisture was 0.372 m³/m³, against 0.133 on 13 October.
- **Wetting by crop:** brightening was ordered by canopy cover, so a single scene-wide correction cannot remove it:

  | Cotton | Rice | Groundnut | Maize | Bajra |
  |--:|--:|--:|--:|--:|
  | +0.49 dB | +1.07 dB | +1.20 dB | +2.05 dB | +2.60 dB |

- **Use:** the date is excluded from the level features and kept as a moisture observation and ensemble member.
- **Look side:** the right-looking geometry has no measurable effect on stable targets (spread ratio 0.944; rejection threshold 1.5).

### 3.4 Health index — `p3_build.py` (`health_parts`, `derive_weights`, `health_index`)

Each part is a robust z-score, (x − median) / (1.4826 · MAD), clipped to ±4.

| Part | Definition | Weight |
|:--|:--|--:|
| `level` | γ⁰ (dB) on 14 August | 0.149 |
| `growth` | γ⁰ 14 Aug − 19 Jun, minus its village median | 0.217 |
| `uniform` | −(within-plot CV of γ⁰) on 14 August | 0.249 |
| `persist` | Trapezoid integral of linear γ⁰ over day of year, excluding 29 October | 0.174 |
| `senesce` | Slope of γ⁰ (dB) between 13 October and 12 November | 0.211 |

- **Weights:** w_k ∝ 1 / Σ_j |ρ_Spearman(k, j)|.
- **Score:** H = 100 · Φ(z_crop(S) / 2), with S = Σ w_k · part_k.
- **Ensemble:** a variant that includes 29 October is also built, and the per-plot spread between the two is reported.

### 3.5 Yield anchors — `data_aux/anchors_r3.csv`

| Crop | Base (kg/ha) | 2025–26 adjustment | Anchor (t/ha) | Completion on 12 Nov |
|:--|--:|:--|--:|--:|
| Cotton (lint) | 752.7 | × 0.97 (USDA revision for rain damage) | 0.730 | 0.28 |
| Groundnut (pod) | 2514 | 1.00 | 2.514 | 0.95 |
| Rice (paddy) | 1690 | 1.00 | 1.690 | 1.00 |
| Bajra (grain) | 1910 | 1.00 (corrected from the APY value of 2714) | 1.910 | 1.00 |
| Maize (grain) | 2312 | 1.00 | 2.312 | 1.00 |

- **Cotton completion:** 0.80 × 0.35 = 0.28. MOAFW reported 80% of the first picking on 10 November 2025, and the first picking is 35% of the crop under a 35 / 50 / 15% picking split.
- **Bajra:** the APY value does not reproduce from the same table's area and production. Independent estimates, including the Gujarat State First Advance Estimate of 1786.66 kg/ha, lie near 1.9 t/ha.
- **Completion factors** are used only for a separate yield-to-date column.

### 3.6 Modifier and aggregation — `p3_build.py` (`modifier`, `zones`)

```
v = clip( 1 + 0.30 · z_crop(H) / 2 , 0.35 , 1.9 )        0.30 = YIELD_SPREAD (stated prior)
modifier = v / (area-weighted mean of v within the crop)  → mean exactly 1.0 per crop
```

- **What each part sets:** the SAR sets the order of plots within a crop; the anchor sets the level.
- **Village total:** it equals Σ_c anchor_c × area_c by construction.
- **Zones:** a fixed 500 m grid defines 46 zones with at least 5 plots each.

## 4. Results

| Crop | Plots | Area (ha) | Yield (t/ha) | Production (t) | Plot range (t/ha) | Type of estimate |
|:--|--:|--:|--:|--:|:--|:--|
| Cotton (lint) | 455 | 193.4 | 0.730 | 141.2 | 0.460–0.972 | Forecast |
| Groundnut | 221 | 137.7 | 2.514 | 346.1 | 1.615–3.391 | Near-complete measurement |
| Rice | 86 | 47.4 | 1.690 | 80.2 | 1.087–2.132 | Retrospective reconstruction |
| Bajra | 149 | 42.3 | 1.910 | 80.8 | 1.241–2.593 | Retrospective reconstruction |
| Maize | 55 | 26.7 | 2.312 | 61.8 | 1.817–2.933 | Retrospective reconstruction |
| **All** | **966** | **447.5** | **1.587** | **710.0** | | |

- **Health index:** ranges from 2.3 to 97.7 (sd 18.9). The within-crop median is 50.
- **Zones:** zone health ranges from 36.6 to 76.3.
- **Variance explained:** crop type explains η² = 0.9148 of the plot yield variance.

## 5. Validation and Uncertainty

**Pre-registered checks** (`p3_checks.py`): 19 of 19 pass. Among them are the Phase 2 → Phase 3 consistency ratios, whose expected ranges were fixed before the forecast:

| Crop | Phase 2 to date | Phase 3 final | Ratio | Expected |
|:--|--:|--:|--:|:--|
| Rice | 1.631 | 1.690 | 1.04 | 0.8–1.3 |
| Maize | 2.321 | 2.312 | 1.00 | 0.8–1.3 |
| Cotton | 0.396 | 0.730 | 1.84 | 1.5–4.0 |
| Groundnut | 1.931 | 2.514 | 1.30 | 1.0–1.6 |
| Bajra | 2.675 | 1.910 | 0.71 | 0.5–1.3 |
| Village | 594.9 t | 710.0 t | 1.19 | ≥ 1.0 |

**Uncertainty** (`p3_uncertainty.py`), four terms reported separately:

| Term | Scope | Value |
|:--|:--|--:|
| Sampling (split-half) | per plot | 5.8 health points |
| Structural (8-member ensemble) | per plot / village | 0.028 t/ha / 0.0 t |
| Crop label, own posterior | village | 40.7 t |
| Crop label, five other teams' labels | village | 236.5 t |
| Anchor (district year-to-year variability) | village | 385.2 t |
| Crop label + anchor, 90% interval | village | 597.8–917.8 t |
| Argmax bias | village | +31.5 t (shipped total biased low) |

Moran's I of the yield residual is 0.083 (z = 7.4), so plots are not independent for sub-village intervals.

**Post-freeze analyses.** These were run after the forecast was frozen, against criteria set in advance. None changes a shipped number.

| Script | Question | Result |
|:--|:--|:--|
| `p3_t2_wcm.py` | Can the Water Cloud Model separate canopy from soil wetting? | Rejected: bajra two-way transmissivity T² = 1.53 (95% CI 1.27–1.90) exceeds the physical limit of 1 |
| `p3_salvage.py` | Do the rejected methods carry any usable information? | The Water Cloud Model's T² is a rank-1.000 transform of the observed rise; the canopy proxy loses about a third of its strength in 6 days |
| `p3_t1_gai.py` | Can a SAR canopy proxy drive a growth model? | Rejected on availability: only 2 of 6 dates have same-day optical data, none in June–September. γ⁰ vs NDVI within crop: cotton ρ 0.52, rice 0.69 / 0.56 |
| `p3_dawidskene.py` | Does six-team label fusion improve the crop map? | Better on its comparison metric, but voided by its provenance control |
| `p3_fusion.py` | Does Sentinel-1 C-band improve the index? | Better on 13 October, worse on 12 November; not adopted |
| `p3_climatology.py` | Was 2025 an ordinary season? | Jun–Sep: z = −0.16, ordinary. 1 Oct–15 Nov (set `WINDOW` on line 68): z = −1.62, suggestive of late-season damage |
| `p3_harvest.py` | Can harvest timing be detected from Sentinel-1? | No: crops do not separate (Kruskal–Wallis p = 0.86) |
| `p3_representative.py`, `p3_repr_control.py` | Is Sokhda typical of its surroundings? | Brighter than about 270 comparable cropland blocks in all nine years (mean 88th percentile); reported as a flag |
| `p3_spatial.py` | Does the ranking reflect the plot or its location? | Variogram 93% nugget (incidence-angle control 94% structured); 96% of variance plot-specific; Moran's I 0.073, or 0.062 after removing incidence and soil |
| `p3_edge.py` | Does the ranking reflect plot boundaries? | Rank ρ 0.853 for core pixels vs 0.857 for a random-drop null; ρ(index, plot size) = −0.017 |
| `p3_voi.py` | What does each acquisition contribute? | Without 14 August, ρ = 0.211 with the shipped index; with the calendar shuffled, ρ = 0.312 |
| `p3_speckle.py` | Is the attention list stable under re-acquisition? | 69% of the 97 listed plots retained, against 10% by chance |

## 6. Repository Contents

```
phase3_yield_forecast/
├── README.md
├── src/                      34 scripts (Section 8)
├── data_aux/
│   ├── anchors_r3.csv
│   └── inherited/            Phase 2 crop map, per-plot tables, third-party controls, README
└── results/
    ├── submission.csv        deliverable
    ├── d4_debug.csv          per-plot provenance, index parts, class probabilities, uncertainty
    ├── tables/               47 tables
    └── figures/              14 figures
```

## 7. Usage

Place `anrf-aise-hack-2-0-round-3-sar-crop-yield-forecasting/` inside this folder, or set `AISE_DATA`. Every script imports `common.py`, which requires the scene files. The consistency check reads `../phase2_crop_type_health/results/submission.csv`; override this with `AISE_R2`.

**Quick route,** from the shipped feature table:

```bash
cd phase3_yield_forecast
python src/p3_build.py            # -> results/submission.csv and summary tables
python src/p3_checks.py           # 19 / 19
md5sum results/submission.csv     # 8304407ed23bc67168fb26fca21a03d1
```

**Full route,** from the raw SLC scenes (about 1.5 minutes; writes about 350 MB of cache):

```bash
python src/common.py              # self-check: six scenes, duplicate-file guard, calibration convention
python src/p1_prep.py
python src/p1_calib.py
python src/p1_features.py
python src/p3_build.py
python src/p3_checks.py
python src/tests_regression.py    # 12 / 12; needs results/cache/
```

**Analyses.** Run any script in Section 8 directly. `p3_climatology.py` (about 8 minutes) and `p3_representative.py` (about 46 minutes) download nine years of Sentinel-1 data.

## 8. Script Reference

Script prefixes are internal stages of this phase: `p0` ancillary data, `p1` imagery to features, `p3` forecast and analyses.

**Main pipeline**

| Script | Purpose | Writes | Also needs |
|:--|:--|:--|:--|
| `common.py` | Paths, scene selection, scene constants, guards | — | — |
| `p1_prep.py` | SLC → β⁰, σ⁰, γ⁰ on 5 m and 2 m grids | `results/cache/` | — |
| `p1_calib.py` | Calibration convention vs the noise floor | `tables/p1_calibration.csv`, figure | cache |
| `p1_features.py` | Per-plot features, coverage, noise-floor margins | `tables/p1_farm_features.csv`, `p1_coverage.csv`, `p1_nesz_margin.csv` | cache |
| `p3_build.py` | Health index, forecast, aggregation, zones | `results/submission.csv`, `d4_debug.csv`, `tables/p3_village_summary.csv`, `p3_zone_summary.csv`, `p3_ensemble.csv` | features |
| `p3_checks.py` | 19 pre-registered checks | `tables/p3_checks.csv` | Phase 2 submission |
| `tests_regression.py` | 12 failure-mode assertions | log | cache |

**Supporting analyses and figures**

| Script | Purpose | Also needs |
|:--|:--|:--|
| `p1_stable.py` | Look-side and wetness tests on stable targets | cache |
| `p1_traj.py` | Per-crop trajectories, groundnut-lift test, identifiability | features |
| `p1_xcheck.py` | Radiometry vs Project Orion's independent features | features |
| `p3_uncertainty.py` | Uncertainty terms | outputs |
| `p3_validate.py` | Spatial hold-out | features |
| `p3_figures.py`, `p3_pipeline_fig.py` | Figures and pipeline diagram | outputs |
| `p0_census.py`, `p0_weather.py`, `p0_witness.py` | Sentinel scene census, ERA5-Land weather, Sentinel check data | internet |

**Post-freeze analyses:** see Section 5. `p3_climatology.py`, `p3_representative.py`, `p3_repr_control.py` and `p3_s1_season.py` need internet access; `p3_edge.py` needs the cache.

**Helpers**

- `stac.py`: anonymous access to the Microsoft Planetary Computer.
- `datagovin.py`: data.gov.in client. It reads a personal key from `~/.config/aisehack/datagovin.key`, and no pipeline script calls it.
- `p0_inherit.py`: documents how `data_aux/inherited/` was assembled. It is not intended to be re-run.

## 9. Limitations

- **No ground truth:** there is no plot-level truth, so plot-level accuracy is not claimed.
- **Crop labels:** they come from Phase 2 and agree weakly with independent maps (κ = 0.112 against Sentinel-1 + Sentinel-2; mean pairwise κ = 0.060 across six teams).
- **Health index:** it is a within-crop ranking. Its agreement with same-day Sentinel-2 NDVI is very weak (area-weighted within-crop ρ = −0.02 on 13 Oct and +0.09 on 12 Nov).
- **Yield level:** it comes from published statistics. The anchor is the largest source of uncertainty (385.2 t).
- **Type of estimate:** the classification of each crop's estimate (forecast, measurement or reconstruction) rests on published crop calendars.
