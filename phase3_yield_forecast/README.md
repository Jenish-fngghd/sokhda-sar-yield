# Phase 3: Final yield forecast, village aggregation and uncertainty

Phase 3 works on the 966 farm plots in Sokhda village (447.5 ha) and produces:

- a **final yield forecast** for each plot;
- the **aggregate by crop** at village and sub-village level;
- a **health index and uncertainty** for each plot.

It uses six Capella X-band HH SLC scenes from 6 June to 12 November 2025. Most of the paper's methodology and validation is implemented here.

| | |
|:--|:--|
| Paper sections | 3.2 preprocessing, 3.4.2 health index, 3.5 yield and aggregation, 3.6 validation and uncertainty |
| Competition | ANRF AISEHack 2.0, Round 3: <https://www.kaggle.com/competitions/anrf-aise-hack-2-0-round-3-sar-crop-yield-forecasting> |
| Deliverable | `results/submission.csv` (966 rows), md5 `8304407ed23bc67168fb26fca21a03d1` |
| Result | **710.0 t** over 447.5 ha (1.587 t/ha); village 90% interval 598–918 t |
| Reproduces | byte for byte, from the raw SLCs (about 1.5 minutes) or from the shipped feature table (seconds) |
| Needs | the Phase 3 competition download (about 3.2 GB) |

---

## Contents

1. [Task](#1-task)
2. [Data](#2-data)
3. [Method](#3-method)
4. [Results](#4-results)
5. [Validation and uncertainty](#5-validation-and-uncertainty)
6. [Folder contents](#6-folder-contents)
7. [Script reference](#7-script-reference)
8. [Reproducing the result](#8-reproducing-the-result)
9. [Verification](#9-verification)
10. [Troubleshooting](#10-troubleshooting)
11. [Limitations](#11-limitations)

---

## 1. Task

For each plot:

- a **final** yield forecast, as opposed to the yield to date of Phase 2;
- plot-level statistics that justify it;
- village-level summaries by crop.

Crop type is carried forward from Phase 2 as an input. No ground truth or method was prescribed, and external data was allowed if explained.

Output columns: `village_id, farm_id, crop_type, health_index, yield_forecast_t_ha` (t/ha; cotton as **lint**).

## 2. Data

All six scenes are Stripmap, HH, ascending, about 9.65 GHz, annotated `beta_nought`, ENL 1.0. Each comes with 225 control points and vendor metadata, including the orbit state vectors.

| Date | Local time (IST) | Look | Centre incidence | Scale factor | NESZ peak (dB) | Note |
|:--|:--|:--|--:|--:|--:|:--|
| 2025-06-06 | 12:55 | left | 35.244° | 0.00212186 | −26.13 | as in Phase 2 |
| 2025-06-19 | 07:44 | left | 28.768° | 0.00236205 | −27.76 | as in Phase 2 |
| 2025-08-14 | 08:41 | left | 28.692° | 0.00198903 | −27.97 | as in Phase 2 |
| 2025-10-13 | 07:56 | left | 31.528° | 0.00136443 | −27.35 | as in Phase 2 |
| 2025-10-29 | 01:37 | **right** | 29.840° | 0.00155765 | −27.74 | new; inside a heavy-rain event |
| 2025-11-12 | 19:22 | left | 29.746° | 0.00162432 | −27.72 | new |

**Ancillary inputs:**

| File | Contents |
|:--|:--|
| `data_aux/anchors_r3.csv` | yield anchors per crop with source, base year, 2025–26 adjustment and completion at 12 November |
| `data_aux/inherited/label_dist.csv` | the Phase 2 crop map and its five-class posterior |
| `data_aux/inherited/uncert.csv` | per-plot sampling uncertainty of the health index |
| `data_aux/inherited/gee.csv` | Dynamic World and WorldCereal values per plot |
| `data_aux/inherited/anchors.csv` | Vadodara district crop statistics, 1997–2012 |
| `data_aux/inherited/consensus.csv`, `orion_feat.csv` | third-party control tables |

`data_aux/inherited/README.md` describes every inherited table, its source, licence notes and which script reads it.

## 3. Method

```
yield_forecast[plot] = anchor(crop) × modifier[plot]
production           = Σ (yield × area)        never a mean of per-hectare rates
```

### 3.1 Preprocessing (`p1_prep.py`, `p1_calib.py`)

- **Radiometry:** β⁰ = SF² · |z|², σ⁰ = β⁰ · sin θ, γ⁰ = β⁰ · tan θ. The scale factor applies to amplitude, so it is squared for power. θ is a per-pixel incidence angle computed from the orbit state vectors, and it matches the vendor's centre angle.
- **Check of the convention:** with SF², the darkest content of each scene sits above the vendor's declared noise floor on all six scenes, by +1.86 dB on average. Without the square it would sit +29.3 dB above on average, which is physically implausible. σ⁰ and γ⁰ fall below the floor on 6 and 4 scenes, which is why the noise-floor test is made in β⁰.
- **Geocoding:** each scene's own control points, with power averaged onto 5 m and 2 m grids (the multi-look happens in the same step).
- **Scene lookup:** dates are scanned from the data and constants are read from the vendor `_extended.json`; nothing is typed in by hand.

### 3.2 Features (`p1_features.py`)

- **Size:** 966 plots × 111 features.
- **Negative-buffer ladder:** −5, then −2, then 0 m, so that boundary pixels do not mix fields.
- **Gates:** a coverage gate and a noise-floor gate per plot and date, the latter in β⁰.
- **Quality control:** 832 plots are observed on all dates, 105 partly, and 29 never. All 29 are cotton plots at the north-west edge of the swath.

### 3.3 Handling of 29 October

- **Conditions:** wind at overpass was 14.1 m/s, against 6.7–9.6 m/s on the other dates. ERA5-Land shows 81.2 mm of rain on 26–28 October, and topsoil moisture at the overpass was 0.372 m³/m³ against 0.133 on 13 October.
- **Effect:** every crop brightened, by an amount ordered by canopy: cotton +0.49 dB, rice +1.07, groundnut +1.20, maize +2.05, bajra +2.60. No single scene-wide correction can remove that.
- **Decision:** the date is excluded from the level features. It is kept as a moisture observation and as an ensemble member.
- **Look side:** the right-looking geometry has no measurable effect on stable targets (spread ratio 0.944, against a rejection threshold of 1.5).

### 3.4 Health index (`p3_build.py`: `health_parts`, `derive_weights`, `health_index`)

Each part is a robust z-score: (x − median) / (1.4826 · MAD), clipped to ±4.

| Part | Definition | Weight |
|:--|:--|--:|
| `level` | γ⁰ (dB) on 14 August, peak canopy | 0.149 |
| `growth` | γ⁰ 14 Aug − 19 Jun, minus its village median (the only pair with matched geometry) | 0.217 |
| `uniform` | −CV of γ⁰ within the plot on 14 August | 0.249 |
| `persist` | trapezoid integral of linear γ⁰ over day of year (29 Oct excluded) | 0.174 |
| `senesce` | slope of γ⁰ (dB) over 13 Oct and 12 Nov | 0.211 |

- **Weights:** w_k ∝ 1 / Σ_j |ρ_Spearman(k, j)|. They come from the feature correlations alone and are never fitted to an outcome.
- **Score:** H = 100 · Φ(z_crop(S) / 2), where S = Σ w_k · part_k and z_crop is the robust z-score within the plot's crop.
- **Ensemble:** a second variant that includes 29 October is built, and the spread between the two variants is reported per plot.

### 3.5 Anchors (`data_aux/anchors_r3.csv`)

| Crop | Base (kg/ha) | 2025–26 adjustment | Anchor (t/ha) | Completion on 12 Nov |
|:--|--:|:--|--:|--:|
| Cotton (lint) | 752.7 | × 0.97 (USDA revision for rain damage) | 0.730 | 0.28 |
| Groundnut (pod) | 2514 | 1.00 | 2.514 | 0.95 |
| Rice (paddy) | 1690 | 1.00 | 1.690 | 1.00 |
| Bajra (grain) | 1910 | 1.00 (corrected from the APY 2714) | 1.910 | 1.00 |
| Maize (grain) | 2312 | 1.00 | 2.312 | 1.00 |

- **Cotton completion:** MOAFW, 10 November 2025, reported 80% of the first picking. With the 35 / 50 / 15% picking split this gives 0.80 × 0.35 = 0.28.
- **Bajra:** the APY value of 2714 kg/ha does not reproduce from the same table's own area and production. Independent estimators, including the Gujarat State First Advance Estimate of 1786.66 kg/ha, agree near 1.9 t/ha; `anchors_r3.csv` records the reasoning.
- **Completion factors** are used only for a separate yield-to-date column, never in the forecast.

### 3.6 Modifier and aggregation (`p3_build.py`: `modifier`, `zones`)

```
v = clip( 1 + 0.30 · z_crop(H) / 2 , 0.35 , 1.9 )        0.30 = YIELD_SPREAD, a stated prior
modifier = v / (area-weighted mean of v within the crop)  → mean exactly 1.0 per crop
```

The SAR therefore sets the order of plots within a crop, and the anchor sets the level. The village total equals Σ_c anchor_c × area_c by construction. A fixed 500 m grid gives **46 zones** with at least 5 plots each.

## 4. Results

| Crop | Plots | Area (ha) | Yield (t/ha) | Production (t) | Yield range (t/ha) | Kind of quantity |
|:--|--:|--:|--:|--:|:--|:--|
| Cotton (lint) | 455 | 193.4 | 0.730 | 141.2 | 0.460–0.972 | forecast: about 72% still on the plant on 12 Nov |
| Groundnut | 221 | 137.7 | 2.514 | 346.1 | 1.615–3.391 | near-complete measurement: mature by late October |
| Rice | 86 | 47.4 | 1.690 | 80.2 | 1.087–2.132 | retrospective reconstruction |
| Bajra | 149 | 42.3 | 1.910 | 80.8 | 1.241–2.593 | retrospective reconstruction |
| Maize | 55 | 26.7 | 2.312 | 61.8 | 1.817–2.933 | retrospective reconstruction |
| **All** | **966** | **447.5** | **1.587** | **710.0** | | |

- **Health index:** ranges from 2.3 to 97.7 (sd 18.9); within-crop medians are 50 by construction.
- **Zones:** zone health spans 36.6 to 76.3.
- **Variance explained:** the crop label explains η² = 0.9148 of the yield variance.

## 5. Validation and uncertainty

**Pre-registered checks** (`p3_checks.py`, 19 / 19 pass). These include the Phase 2 → Phase 3 consistency ratios, written down before the forecast existed:

| Crop | Phase 2 to date | Phase 3 final | Ratio | Expected |
|:--|--:|--:|--:|:--|
| Rice | 1.631 | 1.690 | 1.04 | 0.8–1.3 |
| Maize | 2.321 | 2.312 | 1.00 | 0.8–1.3 |
| Cotton | 0.396 | 0.730 | 1.84 | 1.5–4.0 |
| Groundnut | 1.931 | 2.514 | 1.30 | 1.0–1.6 |
| Bajra | 2.675 | 1.910 | 0.71 | 0.5–1.3 |
| Village | 594.9 t | 710.0 t | 1.19 | ≥ 1.0 |

**Uncertainty, four terms reported separately** (`p3_uncertainty.py`):

| Term | Scope | Value |
|:--|:--|--:|
| Sampling (split-half) | per plot | 5.8 health points |
| Structural (8-member ensemble) | per plot / village | 0.028 t/ha / 0.0 t (by construction) |
| Crop label, own posterior | village | 40.7 t |
| Crop label, five other teams' labels | village | 236.5 t |
| Anchor (district year-to-year spread) | village | 385.2 t |
| Label + anchor, 90% interval | village | [597.8, 917.8] t |
| Argmax bias (direction) | village | +31.5 t, so the shipped total is biased low |

**Post-freeze experiments.** These were run after the forecast was frozen, against kill criteria set in advance, and none changes a shipped number:

| Script | Question | Verdict |
|:--|:--|:--|
| `p3_t2_wcm.py` | Can the Water Cloud Model separate canopy from soil wetting? | killed: its physicality control failed (bajra two-way transmissivity T² = 1.53, 95% CI 1.27–1.90, above the physical limit of 1); `p3_salvage.py` then shows T² is a rank-1.000 transform of the observed rise, so the model could not add information |
| `p3_t1_gai.py` | Can SAFY's canopy proxy work here? | killed on availability: only 2 of 6 dates have a same-day optical partner, none in June to September (γ⁰ vs NDVI within crop: cotton ρ 0.52 on both, rice 0.69 and 0.56) |
| `p3_dawidskene.py` | Does six-team label fusion help? | won, then voided by its provenance control |
| `p3_fusion.py` | Does Sentinel-1 C-band help? | better on 13 Oct, worse on the held-out 12 Nov; rejected |
| `p3_climatology.py` | Was 2025 ordinary? | growing season (Jun–Sep) z = −0.16, ordinary; late season (1 Oct–15 Nov, set `WINDOW` on line 68) z = −1.62, suggestive of damage |
| `p3_harvest.py` | Can harvest timing be tested? | could not be resolved (crops do not separate, p = 0.86) |
| `p3_representative.py` | Is Sokhda typical of its surroundings? | brighter than about 270 like-sized cropland blocks in all nine years (mean 88th percentile); reported as a flag |
| `p3_spatial.py` | Is the ranking the field or its location? | ranking variogram 93% nugget (incidence-angle control 94% structured); 96% of the residual variance is plot-specific and 8.5% neighbourhood; Moran's I 0.073, 0.062 after removing incidence and soil |
| `p3_edge.py` | Is it the field or its outline? | ρ 0.853 vs 0.857 for a random-drop null; ρ(index, plot size) = −0.017 |
| `p3_voi.py` | What did each acquisition contribute? | withholding 14 Aug: ρ = 0.211 with the shipped index; calendar permutation: ρ = 0.312 |
| `p3_speckle.py` | Would a re-flight name the same farms? | 69% of the 97-farm attention list retained (chance 10%) |

## 6. Folder contents

```
phase3_yield_forecast/
├── README.md
├── src/                      34 scripts (Section 7)
├── data_aux/
│   ├── anchors_r3.csv
│   └── inherited/            Phase 2 crop map and per-plot tables, third-party controls, README
└── results/
    ├── submission.csv        the deliverable
    ├── d4_debug.csv          133 columns of provenance, parts, posterior and uncertainty per plot
    ├── tables/               47 tables written by the scripts (features, checks, uncertainty,
    │                         zones, spatial, edge, speckle, value of information, climatology, ...)
    └── figures/              14 figures (calibration, stable targets, trajectories, deliverable,
                              uncertainty, consistency, witnesses, climatology, spatial, VOI,
                              speckle, edge, pipeline diagram)
```

## 7. Script reference

Script names keep the competition's internal stage prefixes: `p0` ancillary data, `p1` imagery to features, `p3` deliverable and analyses. These are stages within Phase 3, not paper phases.

"Data" means the Phase 3 download, and every script needs it, because `common.py` checks for the six scenes on import. "Cache" means rasters written by `p1_prep.py`. "Internet" means the script downloads from the Microsoft Planetary Computer or weather services.

**Main pipeline**

| Script | Role | Writes | Also needs |
|:--|:--|:--|:--|
| `common.py` | paths, scene scan, scene constants, guards; run alone for a self-check | — | — |
| `p1_prep.py` | SLC → β⁰, σ⁰, γ⁰ on 5 m and 2 m grids | `results/cache/` | — |
| `p1_calib.py` | calibration convention against the noise floor | `tables/p1_calibration.csv`, figure | cache |
| `p1_features.py` | per-plot features, coverage, noise-floor margins | `tables/p1_farm_features.csv`, `p1_coverage.csv`, `p1_nesz_margin.csv` | cache |
| `p3_build.py` | health index, forecast, aggregation, zones | `results/submission.csv`, `d4_debug.csv`, `tables/p3_village_summary.csv`, `p3_zone_summary.csv`, `p3_ensemble.csv` | features |
| `p3_checks.py` | 19 pre-registered checks | `tables/p3_checks.csv` | Phase 2 submission |
| `tests_regression.py` | 12 failure-mode assertions | log | cache |

**Analyses before the freeze, and figures**

| Script | Role | Also needs |
|:--|:--|:--|
| `p1_stable.py` | look-side and wetness tests on stable targets | cache |
| `p1_traj.py` | trajectory per crop, groundnut-lift test, identifiability | features |
| `p1_xcheck.py` | radiometry vs Project Orion's independent features | features |
| `p3_uncertainty.py` | the four uncertainty terms | outputs |
| `p3_validate.py` | spatial hold-out | features |
| `p3_figures.py`, `p3_pipeline_fig.py` | the deliverable figures and the pipeline diagram | outputs |
| `p0_census.py`, `p0_weather.py`, `p0_witness.py` | Sentinel scene census, ERA5 weather at each overpass, Sentinel witnesses through harvest | internet |

**Post-freeze experiments:** see Section 5. `p3_climatology.py`, `p3_representative.py`, `p3_repr_control.py` and `p3_s1_season.py` need the internet; `p3_edge.py` needs the cache. `p3_salvage.py` re-examines the two killed methods (the Water Cloud Model is a rank-1.000 relabelling of its input; the canopy proxy loses about a third of its strength in 6 days).

**Helper modules:**

- `stac.py`: anonymous Planetary Computer access.
- `datagovin.py`: a data.gov.in client that reads a personal key from `~/.config/aisehack/datagovin.key`, never from the repository; no script calls it.
- `p0_inherit.py`: records how `data_aux/inherited/` was assembled. It reads a Phase 2 working folder that is not in this repository; do not re-run it.

## 8. Reproducing the result

Place `anrf-aise-hack-2-0-round-3-sar-crop-yield-forecasting/` inside this folder, or set `AISE_DATA` to its path. Check D1 reads `../phase2_crop_type_health/results/submission.csv` (override with `AISE_R2`).

**Quick route,** from the shipped feature table:

```bash
cd phase3_yield_forecast
python src/p3_build.py            # -> results/submission.csv and tables
python src/p3_checks.py           # 19 / 19
md5sum results/submission.csv     # 8304407ed23bc67168fb26fca21a03d1
```

**Full route,** from the raw SLCs (about 1.5 minutes on a 16-thread desktop; writes about 350 MB of cache):

```bash
python src/common.py              # self-check: six scenes, duplicate trap, squared scale factor
python src/p1_prep.py
python src/p1_calib.py
python src/p1_features.py
python src/p3_build.py
python src/p3_checks.py
python src/tests_regression.py    # 12 / 12 (needs the cache)
```

**Analyses:** run any script in Section 7 directly. `p3_climatology.py` takes about 8 minutes and `p3_representative.py` about 46 minutes, because they download nine years of Sentinel-1.

## 9. Verification

Every script was run on 2026-10-24 in a fresh copy of this repository, and every output was compared with the shipped copy.

| Check | Result |
|:--|:--|
| `p1_prep.py` → `p1_calib.py` → `p1_features.py` from the raw SLCs | `p1_farm_features.csv`, `p1_coverage.csv`, `p1_calibration.csv` byte-identical |
| `p3_build.py` | `submission.csv` md5 `8304407ed23bc67168fb26fca21a03d1`; village, zone and ensemble tables byte-identical |
| `p3_checks.py`, `tests_regression.py` | 19 / 19 and 12 / 12 |
| every other script (31 run in total) | the other shipped tables and all 14 figures byte-identical, including weather, scene census and witness tables re-downloaded from the internet |
| `d4_debug.csv` | identical values; six entries of `health_ensemble_spread` differ in the 16th significant digit between runs (floating-point order) |
| `p3_repr_control.py` | reproduces +1.051 dB (parcels), +1.007 dB (like for like), definition effect 0.044 dB |
| `p3_representative.py` | 2025 offset +0.99 dB against about 270 neighbours. The competition run reported +1.05 dB against 271: the online archive has changed slightly. `tables/p3_representative.csv` holds the re-run values. |
| `p3_climatology.py` (re-downloads the Sentinel-1 archive) | `p3_climatology.csv` byte-identical; growing-season z = −0.16 |
| `p3_climatology.py` with `WINDOW = ("10-01", "11-15")` (line 68) | late-season z = −1.62 (farms +0.48, landscape +1.01), as quoted in the script's own log note and the paper; 2025 at −0.42 dB against 2019 at −0.41 dB |

## 10. Troubleshooting

| Symptom | Fix |
|:--|:--|
| `FileNotFoundError: no SLCs under ...` | set `AISE_DATA` to the Phase 3 download |
| `HTTP Error 504` from a download script | a remote service was busy; re-run the script |
| `tests_regression.py` fails on a missing `gamma0_*.tif` | run `p1_prep.py` first to build `results/cache/` |
| check D1 skipped or failing | the Phase 2 submission is not at `../phase2_crop_type_health/results/`; set `AISE_R2` |
| CRS errors on Windows | keep `common.py`'s `PROJ_LIB` line ahead of every `rasterio` import |

## 11. Limitations

- **No ground truth:** there is no plot-level truth, so no plot-level accuracy is claimed.
- **Crop labels:** they come from Phase 2. They agree weakly with an independent map (κ = 0.112; mean pairwise κ = 0.060 across six teams), and the Phase 2 features behind them used the unsquared calibration.
- **Health index:** the composite barely agrees with same-day Sentinel-2 NDVI (area-weighted within-crop ρ = −0.02 (13 Oct) and +0.09 (12 Nov); `p3_fusion.py`). It is a within-crop ranking.
- **Yield level:** comes from published statistics. The anchor is the largest uncertainty (385.2 t).
- **Kinds of quantity:** the split into forecast, measurement and reconstruction rests on published crop calendars.
- **Spatial correlation:** the yield residual is spatially correlated (Moran's I = 0.083), so sub-village intervals must not assume independent plots.
