# Phase 2: Plot-level crop type, health index and yield to date

Phase 2 produces three values for each of the 966 farm plots in Sokhda village (447.5 ha), using four Capella X-band HH SLC scenes:

- a crop type;
- a health index from 0 to 100;
- a yield estimate up to the last acquisition (13 October 2025).

Its crop map is the crop input of Phase 3.

| | |
|:--|:--|
| Paper section | 3.4.1 Crop type: area-constrained assignment |
| Competition | ANRF AISEHack 2.0, Round 2: <https://www.kaggle.com/competitions/anrf-aise-hack-2-0-round-2-sar-crop-health-yield-estimation> |
| Deliverable | `results/submission.csv` (966 rows), md5 `89b0e4e2aef63ace4989fc0a44590ee5` |
| Reproduces | byte for byte, from the raw SLCs (about 2 minutes) or from the shipped feature store (seconds) |
| Needs | the Phase 2 competition download; only its shapefiles for the quick route |

---

## Contents

1. [Task](#1-task)
2. [Data and inputs](#2-data-and-inputs)
3. [Method](#3-method)
4. [Results](#4-results)
5. [Validation](#5-validation)
6. [Folder contents](#6-folder-contents)
7. [Script reference](#7-script-reference)
8. [Reproducing the result](#8-reproducing-the-result)
9. [Verification](#9-verification)
10. [Limitations](#10-limitations)
11. [What is deliberately not included](#11-what-is-deliberately-not-included)

---

## 1. Task

For each plot (`farm_id` 1–966, the `FID` of the competition shapefile) the deliverable gives:

| Column | Meaning |
|:--|:--|
| `village_id`, `farm_id` | identifiers |
| `crop_type` | one of Rice, Cotton, Maize, Bajra, Groundnut |
| `health_index` | 0–100 |
| `yield_estimate_to_date` | t/ha: the yield potential reached by the last acquisition, explicitly **not** a final harvest forecast |

No ground truth was given for crop, health or yield.

## 2. Data and inputs

**Capella scenes.** All four are left-looking, ascending Stripmap HH SLC scenes, annotated `beta_nought`. Each comes with 225 ground control points.

| Date | Local time (IST) | Centre incidence | Scale factor | NESZ peak (dB) |
|:--|:--|--:|--:|--:|
| 2025-06-06 | 12:55 | 35.244° | 0.00212186 | −26.13 |
| 2025-06-19 | 07:44 | 28.768° | 0.00236205 | −27.76 |
| 2025-08-14 | 08:41 | 28.692° | 0.00198903 | −27.97 |
| 2025-10-13 | 07:56 | 31.528° | 0.00136443 | −27.35 |

The download's `20250619` folder also contains a byte-identical copy of the 6 June scene. `src/common.py` requires the date to appear in both the folder name and the file name, so that copy is never read as 19 June.

**Declared inputs** (in `data_aux/`, provenance in `data_aux/SOURCES.md`):

| File | Role |
|:--|:--|
| `sokhda_phase1_crop_shares.csv` | Phase 1 crop shares for Sokhda: Cotton 43.0%, Groundnut 31.0%, Rice 10.6%, Bajra 9.4%, Maize 6.0%. Used only as the area constraint on the crop map. |
| `vadodara_apy.csv` | Vadodara district yield 2022–23 (Directorate of Agriculture, Gujarat). Sets the yield level. |

**Witnesses, never inputs.** These are Sentinel-2 L2A (same day as the 13 October Capella scene, 0.003% cloud), Sentinel-1 RTC (single date, plus a 10-scene season integral from 12 June to 10 October) and NASA POWER rainfall. They are fetched by the scripts, and no value from them enters `submission.csv`.

## 3. Method

The principle is dasymetric mapping: a quantity known only in aggregate is held fixed, and the SAR supplies the variation inside it.

### 3.1 Preprocessing (`prep_r2.py`)

- **Radiometry:** β⁰ = scale_factor · |z|², then γ⁰ = β⁰ · tan θ, where θ is a per-pixel incidence angle computed from the orbit state vectors. It matches the vendor's centre angle to within 0.006°.
- **Geocoding:** from slant range to EPSG:32643, using the 225 control points. Averaging power onto the output grid performs the multi-look in the same step. Two grids are produced: 5 m for per-plot statistics and 2 m for texture.
- **Calibration note:** Phase 3 found that the scale factor must be **squared** and corrected it. Phase 2 is kept exactly as submitted, so that its outputs reproduce byte for byte. The error shifts each scene by a constant in dB. That cancels in ranks within a date, but not in the fixed dB thresholds of the rice flooding term below. The paper describes the corrected chain of Phase 3.

### 3.2 Per-plot features (`farm_stats.py`, `sublook.py`, `wcm.py`)

- **Erosion:** plots are eroded before sampling, so that no pixel mixes two fields. The fallback ladder is erode, then no buffer, then the centroid.
- **Features** (61 columns in `results/farm_features.csv`):
  - per date: γ⁰ in dB and in linear units, within-plot CV, K-distribution texture, GLCM entropy and a size-corrected GLCM residual, incidence angle, pixel count;
  - across dates: differences (Aug − 19 Jun, Oct − Aug, Oct − Jun), referenced levels, the June baseline, temporal CV and range, and the season integral;
  - `sublook.py` adds sub-look coherence per date. `wcm.py` adds Water Cloud Model canopy water content per date, an inversion status flag per date, and a per-plot calibration constant (`wcm_k_db`). These are reported but not used in the index.
- **Coverage:** 966 of 966 plots have a row. 895 are measured directly, 52 are filled from same-crop neighbours (missing coverage clusters in the north-west), and 19 are flagged for radio-frequency interference. The source of every row is recorded in `results/d4_debug.csv`.

### 3.3 Crop type (`d4_submission.py`: `crop_evidence`, `assign_crop`, `fit_prior`)

1. **Evidence per crop:** a weighted sum of robust z-scores of SAR features, each with a physical reason:

   | Crop | Main evidence | Mechanism |
   |:--|:--|:--|
   | Rice | flooding term: the 19 Jun − 6 Jun rise mapped to [0, 1] between 3 and 15 dB (weight 2.2); low Aug − Jun change | standing water at transplanting gives strong double bounce |
   | Cotton | October and August levels, growth, texture; no flooding | tall woody canopy, still standing in October |
   | Groundnut | October level high, August level and texture low | low closed canopy on lighter soils |
   | Maize | August level, growth, senescence; October low | tall at the August peak, bare by October |
   | Bajra | weak, mostly negative terms | no usable signature; placed mainly by the constraint |

   The Aug − Jun X-band term is weighted by its effect size (ρ = −0.555 for rice), as measured against the Phase 1 village estimate on the 28 villages other than Sokhda.
2. **Probabilities:** a softmax at temperature 1 turns the evidence into class probabilities.
3. **Constraint:** a per-class bias on the log-probabilities is updated, `b ← b + 0.35 · clip(log(π/s), −2, 2)`, until the **area-weighted** argmax shares match the Phase 1 shares to within 0.005. This moves the decision boundary between classes without changing the order of plots within a class.
4. **Labels:** argmax only at the end.

Why constrain at all: in Phase 1, free per-pixel classification scored about five times worse than predicting zeros.

### 3.4 Health index (`d4_submission.py`: `health_index`)

Four families are robust z-scored across plots:

| Family | Measurement | Weight |
|:--|:--|--:|
| `level` | γ⁰ on 14 August (peak canopy) | 0.189 |
| `growth` | 14 Aug − 19 Jun, the only pair with matched geometry (0.076° apart) | 0.283 |
| `uniform` | −(within-plot CV) | 0.301 |
| `persist` | season integral | 0.228 |

- **Weights:** w_k ∝ 1 / Σ_j |ρ(k, j)|. They are fixed by the correlations between families and never fitted to any witness.
- **Scoring within crop:** each plot is scored against its own crop's median, so 50 means typical for that crop. The score is 100 · Φ((S − median) / (1.4826 · MAD)), using the median and median absolute deviation of the plot's crop.

### 3.5 Yield to date (`d4_submission.py`: `yield_to_date`)

```
yield_to_date [t/ha] = district yield (level) × completion (per crop) × accumulation (per plot, from SAR)
```

- **Completion by 13 October:** Cotton 0.45, Groundnut 0.75, Rice, Maize and Bajra 0.95.
- **Direction of the completion term:** it was corrected after Sentinel-2 contradicted the original assumption that harvested fields brighten. This happened in all five crops.
- **Totals:** village and crop totals are Σ yield × area, never a mean of rates.

## 4. Results

| Crop | Plots | Median health | Median yield to date (t/ha) |
|:--|--:|--:|--:|
| Cotton | 455 | 51.9 | 0.344 |
| Groundnut | 221 | 50.0 | 1.863 |
| Bajra | 149 | 50.0 | 2.572 |
| Rice | 86 | 50.0 | 1.635 |
| Maize | 55 | 50.0 | 2.181 |

- **Village total:** 447.5 ha and 594.9 t to 13 October.
- **Health medians:** scores are centred on 50 within each crop by construction. Cotton reads 51.9 because 50 of its plots are filled from same-crop neighbours after scoring and 4 are flagged for interference.
- **Cotton:** it reads low because it is only about 45% picked by 13 October and is reported as lint.

## 5. Validation

All values are recomputed on the final crop map.

| Test | Result |
|:--|:--|
| Crop classes separate on Sentinel-2 NDVI (13 Oct) | Kruskal–Wallis p = 9.3 × 10⁻³⁵ |
| Crop classes separate on Sentinel-1 VH | p = 2.7 × 10⁻²⁰ |
| Crop order on 13 Oct | cotton highest (still standing), maize lowest (harvested), on both witnesses |
| Health index spatial structure | Moran's I = 0.096 against a 199-permutation null |
| Season-integral term vs the Sentinel-1 season witness | cotton ρ = +0.305 (p = 5 × 10⁻¹⁰), rice +0.290 (p = 0.007), not significant for maize (+0.058, p = 0.68) and groundnut (−0.087, p = 0.20), **bajra −0.219 (p = 0.008)** |
| Plot labels vs an independent Sentinel-1 + Sentinel-2 map | **Cohen's κ = 0.112** (reported failure) |
| Repeat-pass coherence, 19 Jun × 14 Aug | at the noise floor; the stable-target control did not clear its bias floor (reported failure) |

## 6. Folder contents

```
phase2_crop_type_health/
├── README.md
├── src/                          21 scripts (Section 7)
├── notebooks/I9_pipeline.ipynb   end-to-end notebook; its last cell asserts the submission
├── data_aux/
│   ├── sokhda_phase1_crop_shares.csv
│   ├── vadodara_apy.csv
│   └── SOURCES.md
└── results/
    ├── submission.csv            the deliverable
    ├── farm_features.csv         per-plot feature store, 966 × 61
    ├── d4_debug.csv              provenance, evidence and five-class posterior per plot
    └── tables/                   12 tables: scene metadata, witnesses, context, coherence,
                                  independent crop map, comparison clustering, X-band vs optical
```

## 7. Script reference

"Data" means the Phase 2 competition download. "Internet" means the script downloads open data from the Microsoft Planetary Computer or NASA POWER. Several validation scripts import helpers from `witness.py` and may also download data when they run.

| Script | Role | Writes | Needs |
|:--|:--|:--|:--|
| `common.py` | paths, scene lookup, guards; run alone for a self-check | — | data |
| `eda.py` | exploratory checks before any processing; scene metadata | `tables/scene_metadata.csv`, figures | data |
| `prep_r2.py` | SLC → calibrated, geocoded γ⁰ rasters | `results/cache/` | data |
| `check_i1.py` | land-cover dB gate and a geocoding regression test | log | data, cache |
| `farm_stats.py` | per-plot features | `results/farm_features.csv` | data, cache |
| `sublook.py` | sub-look coherence columns (`subcoh_*`) | `farm_features.csv` | data, cache |
| `wcm.py` | Water Cloud Model columns (`vwc_*`, `vwcflag_*`, `wcm_k_db`) | `farm_features.csv` | features |
| `check_i2.py` | feature quality and redundancy checks | log | features |
| `d4_submission.py` | crop type, health index, yield to date | `results/submission.csv`, `d4_debug.csv` | features, shapefiles |
| `i8_repeat.py` | repeat-pass coherence, 19 Jun × 14 Aug (a reported negative) | `tables/i8_repeat_coh.csv` | data, cache |
| `witness.py` | Sentinel-2 and Sentinel-1 witnesses | `tables/witness.csv` | internet |
| `witness_season.py` | 10-scene Sentinel-1 season integral | `tables/witness_season.csv` | internet |
| `external_context.py` | rainfall and multi-year cropland context | `tables/context_*.csv` | internet |
| `i5_validation.py` | validation battery: separation tests, Moran's I, ablation | log | witness tables |
| `check_yield.py` | yield terms vs the witnesses | log | witness tables |
| `check_rice_independence.py` | leave-channel-out test of the rice flooding term | log | witness tables |
| `independent_crop.py` | independent Sentinel-1 + Sentinel-2 crop map and κ | `tables/independent_crop.csv`, `independent_features.csv` | internet |
| `metrics_panel.py` | fixed metric panel used to judge every change | log | witness tables |
| `pb_mrf.py` | alternative clustering with a spatial prior (comparison only) | `tables/pb_crop.csv` | features |
| `why_xband.py` | what about 1 m X-band adds over 10 m optical | `tables/why_xband_*.csv` | witness tables |
| `i10_media.py` | figures | `results/figures/` | results |

## 8. Reproducing the result

Place the Phase 2 download `anrf-aise-hack-2-0-round-2-sar-crop-health-yield-estimation/` inside this folder, or set `AISE_DATA` to its path.

**Quick route.** This needs only the shapefiles from the download:

```bash
cd phase2_crop_type_health
python src/d4_submission.py
md5sum results/submission.csv      # 89b0e4e2aef63ace4989fc0a44590ee5
```

**Full route, from the raw SLCs** (about 2 minutes on a 16-thread desktop):

```bash
python src/common.py               # self-check: four scenes, duplicate-file trap
python src/prep_r2.py              # -> results/cache/
python src/farm_stats.py           # -> results/farm_features.csv
python src/sublook.py
python src/wcm.py
python src/d4_submission.py        # -> results/submission.csv
```

**Validation scripts** (some download open data):

```bash
python src/witness.py; python src/witness_season.py; python src/external_context.py
python src/i5_validation.py; python src/check_yield.py; python src/check_rice_independence.py
python src/independent_crop.py; python src/metrics_panel.py; python src/pb_mrf.py; python src/why_xband.py
```

**Notebook.** `notebooks/I9_pipeline.ipynb` runs the same steps with commentary. Its last cell asserts that the crop type, health index and yield match `results/submission.csv` exactly. It needs `results/cache/` (run `prep_r2.py` first) and the witness tables.

## 9. Verification

Run on 2026-10-24 in a fresh copy of this repository:

| Check | Result |
|:--|:--|
| `prep_r2.py` → `farm_stats.py` → `sublook.py` → `wcm.py` from the raw SLCs | `farm_features.csv` byte-identical (md5 `42b022a51d28c2675f1f35415ee660d9`) |
| `d4_submission.py` (full and quick routes) | `submission.csv` md5 `89b0e4e2aef63ace4989fc0a44590ee5` |
| every other script in Section 7 | runs without error; all 12 tables identical to the shipped copies, including the witness and context tables re-downloaded from the internet |
| `notebooks/I9_pipeline.ipynb`, all cells | runs top to bottom and reproduces the submission exactly |

**Two tables regenerated for this release.** `independent_crop.csv` and `pb_crop.csv` from the competition had been computed against an earlier version of the crop map that differed in 43 plots. They are replaced by versions computed against the final map. The competition report quoted three statistics from that earlier map:

| Statistic | Report | Final map |
|:--|:--|:--|
| κ, plot labels vs independent map | 0.103 | 0.112 |
| Kruskal–Wallis p, NDVI / VH | 1.8 × 10⁻³⁴ / 7.7 × 10⁻²⁰ | 9.3 × 10⁻³⁵ / 2.7 × 10⁻²⁰ |
| Moran's I, health index | 0.105 | 0.096 |

## 10. Limitations

- **No ground truth** for crop, health or yield.
- **Per-plot crop labels are weak** (κ = 0.112). The village mix is constrained; single plot labels are not, so the product should be read at village and crop level.
- **Bajra:** its season-integral term contradicts the Sentinel-1 season witness.
- **Calibration:** the unsquared calibration constant (Section 3.1) affects the fixed flooding thresholds. Phase 3 inherits these crop labels.

## 11. What is deliberately not included

| Left out | Reason |
|:--|:--|
| Land-parcel registry scripts and survey-number tables | they link plots to land records (privacy) |
| Templates for a manual field ground-truth campaign | they are built from the land-parcel registry (privacy) |
| Krishi-DSS client and external-layer validator | they need a personal session token |
| `d11_ship.py`, packaging and document-rendering scripts | they check the competition's writeup package, not the science |
| `diagnose_fixes.py` | an internal comparison that needs a pre-fix baseline submission not kept here |
| The host's `Sokhda_Dummy_Submission.xlsx` | competition file; download it from the competition page if needed |
