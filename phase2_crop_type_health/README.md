# Phase 2 — Plot-level Crop Type, Health Index and Yield to Date

Phase 2 uses four Capella X-band HH SLC scenes to produce three values for each of the 966 farm plots in Sokhda village (447.5 ha):

- a crop type;
- a health index from 0 to 100;
- a yield estimate up to the last acquisition (13 October 2025).

Its crop map is the crop input of Phase 3.

| | |
|:--|:--|
| **Paper section** | Plot-level crop type mapping |
| **Competition** | [ANRF AISEHack 2.0, Round 2](https://www.kaggle.com/competitions/anrf-aise-hack-2-0-round-2-sar-crop-health-yield-estimation) |
| **Deliverable** | `results/submission.csv` (966 rows), md5 `89b0e4e2aef63ace4989fc0a44590ee5` |
| **Requirements** | Phase 2 competition download; only its shapefiles for the quick route |

---

## Contents

1. [Task](#1-task)
2. [Data and Inputs](#2-data-and-inputs)
3. [Method](#3-method)
4. [Results](#4-results)
5. [Validation](#5-validation)
6. [Repository Contents](#6-repository-contents)
7. [Usage](#7-usage)
8. [Script Reference](#8-script-reference)
9. [Limitations](#9-limitations)
10. [Scope of this Release](#10-scope-of-this-release)

---

## 1. Task

For each plot (`farm_id` 1–966, the `FID` of the competition shapefile):

| Column | Meaning |
|:--|:--|
| `village_id`, `farm_id` | Identifiers |
| `crop_type` | Rice, Cotton, Maize, Bajra or Groundnut |
| `health_index` | 0–100 |
| `yield_estimate_to_date` | t/ha reached by the last acquisition; not a final harvest forecast |

No ground truth was provided for crop type, health or yield.

## 2. Data and Inputs

**Capella scenes.** All are left-looking, ascending Stripmap HH SLC scenes, annotated `beta_nought`, each with 225 ground control points.

| Date | Local time (IST) | Centre incidence | Scale factor | NESZ peak (dB) |
|:--|:--|--:|--:|--:|
| 2025-06-06 | 12:55 | 35.244° | 0.00212186 | −26.13 |
| 2025-06-19 | 07:44 | 28.768° | 0.00236205 | −27.76 |
| 2025-08-14 | 08:41 | 28.692° | 0.00198903 | −27.97 |
| 2025-10-13 | 07:56 | 31.528° | 0.00136443 | −27.35 |

The download's `20250619` folder also contains a copy of the 6 June scene. `src/common.py` selects a scene only when the date appears in both the folder name and the file name.

**Declared inputs** (`data_aux/`; provenance in `data_aux/SOURCES.md`):

| File | Role |
|:--|:--|
| `sokhda_phase1_crop_shares.csv` | Phase 1 crop shares for Sokhda (Cotton 43.0%, Groundnut 31.0%, Rice 10.6%, Bajra 9.4%, Maize 6.0%); used only as the area constraint on the crop map |
| `vadodara_apy.csv` | Vadodara district yield 2022–23 (Directorate of Agriculture, Gujarat); sets the yield level |

**Independent check data,** never model inputs:

- Sentinel-2 L2A, acquired the same day as the 13 October Capella scene;
- Sentinel-1 RTC, a single date plus a 10-scene season integral from 12 June to 10 October;
- NASA POWER rainfall.

## 3. Method

Quantities known only in aggregate are held fixed, and the SAR supplies the variation within them.

### 3.1 Preprocessing — `prep_r2.py`

- **Radiometry:** β⁰ = scale_factor · |z|², then γ⁰ = β⁰ · tan θ, where θ is a per-pixel incidence angle computed from the orbit state vectors. θ agrees with the vendor's centre angle to within 0.006°.
- **Geocoding:** slant range to EPSG:32643 using the scene's control points. Averaging power onto the output grid performs the multi-look. Two grids are produced: 5 m for per-plot statistics and 2 m for texture.
- **Calibration note:** Phase 3 established that the scale factor must be squared. Phase 2 is retained as submitted, so that its outputs reproduce exactly. The difference shifts each scene by a constant in dB. That cancels in within-date ranks, but not in the fixed dB thresholds of the rice flooding term.

### 3.2 Per-plot features — `farm_stats.py`, `sublook.py`, `wcm.py`

- **Erosion:** plots are eroded before sampling, with a fallback from eroded polygon to full polygon to centroid.
- **Features** (61 columns in `results/farm_features.csv`):
  - **per date:** γ⁰ in dB and linear units, within-plot CV, K-distribution texture, GLCM entropy and a size-corrected GLCM residual, incidence angle, pixel count;
  - **across dates:** differences (Aug − 19 Jun, Oct − Aug, Oct − Jun), referenced levels, the June baseline, temporal CV and range, the season integral;
  - **reported but not used in the index:**
    - sub-look coherence per date (`sublook.py`);
    - Water Cloud Model canopy water content and an inversion status flag per date, plus a per-plot calibration constant (`wcm.py`).
- **Coverage:** all 966 plots have a row. 895 are measured directly, 52 are filled from same-crop neighbours, and 19 are flagged for radio-frequency interference. Provenance is recorded in `results/d4_debug.csv`.

### 3.3 Crop type — `d4_submission.py` (`crop_evidence`, `assign_crop`, `fit_prior`)

1. **Evidence per crop:** a weighted sum of robust z-scores of SAR features:

   | Crop | Main evidence | Interpretation |
   |:--|:--|:--|
   | Rice | Flooding term: rise from 6 to 19 June, mapped to [0, 1] between 3 and 15 dB (weight 2.2); low Aug − Jun change | Standing water at transplanting |
   | Cotton | October and August levels, growth, texture; no flooding | Tall canopy still standing in October |
   | Groundnut | High October level; low August level and texture | Low closed canopy on lighter soils |
   | Maize | August level, growth, senescence; low October level | Tall at the August peak, bare by October |
   | Bajra | Weak, mostly negative terms | No reliable signature; placed mainly by the constraint |

   The Aug − Jun X-band term is weighted by its effect size (ρ = −0.555 for rice), measured against the Phase 1 estimate on the 28 villages other than Sokhda.
2. **Probabilities:** a softmax at temperature 1 converts the evidence into class probabilities.
3. **Area constraint:** a per-class bias is updated, `b ← b + 0.35 · clip(log(π/s), −2, 2)`, until the area-weighted argmax shares match the Phase 1 shares to within 0.005. The order of plots within a class is unchanged.
4. **Labels:** argmax is taken last.

### 3.4 Health index — `d4_submission.py` (`health_index`)

| Family | Measurement | Weight |
|:--|:--|--:|
| `level` | γ⁰ on 14 August (peak canopy) | 0.189 |
| `growth` | 14 Aug − 19 Jun, the only pair with matched geometry (0.076° apart) | 0.283 |
| `uniform` | −(within-plot CV) | 0.301 |
| `persist` | Season integral | 0.228 |

- **Weights:** w_k ∝ 1 / Σ_j |ρ(k, j)|, set by the correlations between families and never fitted to a check dataset.
- **Score:** 100 · Φ((S − median) / (1.4826 · MAD)), computed within each crop.

### 3.5 Yield to date — `d4_submission.py` (`yield_to_date`)

```
yield_to_date [t/ha] = district yield × completion (per crop) × accumulation (per plot, from SAR)
```

- **Completion by 13 October:** Cotton 0.45, Groundnut 0.75, Rice, Maize and Bajra 0.95.
- **Totals:** village and crop totals are Σ yield × area.

## 4. Results

| Crop | Plots | Median health | Median yield to date (t/ha) |
|:--|--:|--:|--:|
| Cotton | 455 | 51.9 | 0.344 |
| Groundnut | 221 | 50.0 | 1.863 |
| Bajra | 149 | 50.0 | 2.572 |
| Rice | 86 | 50.0 | 1.635 |
| Maize | 55 | 50.0 | 2.181 |

- **Village total:** 594.9 t over 447.5 ha to 13 October.
- **Health scores:** centred on 50 within each crop. Cotton's median is 51.9 because 50 of its plots are filled from neighbours after scoring and 4 are flagged for interference.
- **Cotton yield:** it reads low because it is reported as lint and is about 45% picked by 13 October.

## 5. Validation

| Test | Result |
|:--|:--|
| Crop classes on Sentinel-2 NDVI (13 Oct) | Kruskal–Wallis p = 9.3 × 10⁻³⁵ |
| Crop classes on Sentinel-1 VH | p = 2.7 × 10⁻²⁰ |
| Crop order on 13 Oct | Cotton highest and maize lowest on both sensors |
| Spatial structure of the health index | Moran's I = 0.096 (199-permutation null) |
| Season-integral term vs Sentinel-1 season integral | Cotton ρ = +0.305 (p = 5 × 10⁻¹⁰); rice +0.290 (p = 0.007); maize +0.058 (p = 0.68) and groundnut −0.087 (p = 0.20) not significant; bajra −0.219 (p = 0.008) |
| Plot labels vs an independent Sentinel-1 + Sentinel-2 map | Cohen's κ = 0.112 |
| Repeat-pass coherence, 19 Jun × 14 Aug | At the noise floor; the stable-target control did not clear its bias floor |

## 6. Repository Contents

```
phase2_crop_type_health/
├── README.md
├── src/                          21 scripts (Section 8)
├── notebooks/I9_pipeline.ipynb   end-to-end notebook; its last cell checks the submission
├── data_aux/
│   ├── sokhda_phase1_crop_shares.csv
│   ├── vadodara_apy.csv
│   └── SOURCES.md
└── results/
    ├── submission.csv            deliverable
    ├── farm_features.csv         per-plot features (966 × 61)
    ├── d4_debug.csv              provenance, evidence and class probabilities per plot
    └── tables/                   12 tables: scene metadata, independent checks, context,
                                  coherence, independent crop map, comparison clustering
```

## 7. Usage

Place `anrf-aise-hack-2-0-round-2-sar-crop-health-yield-estimation/` inside this folder, or set `AISE_DATA`.

**Quick route.** This rebuilds the submission from the shipped feature table and needs only the shapefiles.

```bash
cd phase2_crop_type_health
python src/d4_submission.py
md5sum results/submission.csv      # 89b0e4e2aef63ace4989fc0a44590ee5
```

**Full route,** from the raw SLC scenes (about 2 minutes):

```bash
python src/common.py               # self-check
python src/prep_r2.py              # -> results/cache/
python src/farm_stats.py           # -> results/farm_features.csv
python src/sublook.py
python src/wcm.py
python src/d4_submission.py        # -> results/submission.csv
```

**Validation scripts.** Some of these download open data.

```bash
python src/witness.py; python src/witness_season.py; python src/external_context.py
python src/i5_validation.py; python src/check_yield.py; python src/check_rice_independence.py
python src/independent_crop.py; python src/metrics_panel.py; python src/pb_mrf.py; python src/why_xband.py
```

**Notebook.** `notebooks/I9_pipeline.ipynb` runs the same steps with commentary. It needs `results/cache/` (from `prep_r2.py`) and the validation tables. Its final cell confirms that the crop type, health index and yield match `results/submission.csv`.

## 8. Script Reference

The "Needs" column uses three terms:

- **Data:** the Phase 2 download.
- **Cache:** the rasters from `prep_r2.py`.
- **Internet:** the script downloads open data. Several validation scripts import helpers from `witness.py` and may also download data.

| Script | Purpose | Writes | Needs |
|:--|:--|:--|:--|
| `common.py` | Paths, scene selection, guards; self-check when run | — | Data |
| `eda.py` | Exploratory checks; scene metadata | `tables/scene_metadata.csv`, figures | Data |
| `prep_r2.py` | SLC → calibrated, geocoded γ⁰ | `results/cache/` | Data |
| `check_i1.py` | Land-cover dB gate; geocoding regression test | log | Data, cache |
| `farm_stats.py` | Per-plot features | `results/farm_features.csv` | Data, cache |
| `sublook.py` | Sub-look coherence (`subcoh_*`) | `farm_features.csv` | Data, cache |
| `wcm.py` | Water Cloud Model columns (`vwc_*`, `vwcflag_*`, `wcm_k_db`) | `farm_features.csv` | Features |
| `check_i2.py` | Feature quality and redundancy | log | Features |
| `d4_submission.py` | Crop type, health index, yield to date | `results/submission.csv`, `d4_debug.csv` | Features, shapefiles |
| `i8_repeat.py` | Repeat-pass coherence | `tables/i8_repeat_coh.csv` | Data, cache |
| `witness.py` | Sentinel-2 and Sentinel-1 check data | `tables/witness.csv` | Internet |
| `witness_season.py` | Sentinel-1 season integral | `tables/witness_season.csv` | Internet |
| `external_context.py` | Rainfall and multi-year cropland context | `tables/context_*.csv` | Internet |
| `i5_validation.py` | Separation tests, Moran's I, ablation | log | Check tables |
| `check_yield.py` | Yield terms vs check data | log | Check tables |
| `check_rice_independence.py` | Leave-channel-out test of the rice term | log | Check tables |
| `independent_crop.py` | Independent Sentinel-1 + Sentinel-2 crop map and κ | `tables/independent_crop.csv`, `independent_features.csv` | Internet |
| `metrics_panel.py` | Fixed panel of evaluation metrics | log | Check tables |
| `pb_mrf.py` | Alternative clustering with a spatial prior (comparison) | `tables/pb_crop.csv` | Features |
| `why_xband.py` | Contribution of 1 m X-band relative to 10 m optical | `tables/why_xband_*.csv` | Check tables |
| `i10_media.py` | Figures | `results/figures/` | Results |

## 9. Limitations

- **No ground truth** for crop type, health or yield.
- **Per-plot crop labels are weak** (κ = 0.112). Results should be read at village and crop level.
- **Bajra:** its season-integral term disagrees with the Sentinel-1 season integral.
- **Calibration:** the unsquared calibration constant affects the fixed flooding thresholds, and Phase 3 inherits these crop labels.

## 10. Scope of this Release

Not included:

- **Land-parcel registry scripts and related tables:** they link plots to land records (privacy).
- **Field data-collection templates:** they are derived from the registry.
- **Clients for services that need personal credentials.**
- **Competition packaging, document-rendering and submission-gate scripts.**
- **The host's `Sokhda_Dummy_Submission.xlsx`:** available from the competition page.
