# Label-free Plot-level Crop Yield Forecasting from X-band SAR

**Sokhda village, Vadodara, Gujarat, India · Kharif 2025 · 966 farm plots · 447.5 ha**

This repository contains the code, inputs and outputs for a three-phase study that:

- forecasts final crop yield for every farm plot in a smallholder village from Capella X-band HH single-look complex (SLC) SAR imagery, without yield, crop-health or crop-type labels;
- tests, against a known answer, whether SHAP-style explanations of models trained on such SAR features identify the true drivers.

The work was developed by Team GDHTM within ANRF AISEHack 2.0 (Remote Sensing track). The three competition rounds are referred to as **Phases 1, 2 and 3**.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Results](#2-key-results)
3. [Approach](#3-approach)
4. [Repository Structure](#4-repository-structure)
5. [Getting Started](#5-getting-started)
6. [Reproducing the Results](#6-reproducing-the-results)
7. [Paper Section to Code](#7-paper-section-to-code)
8. [Configuration](#8-configuration)
9. [Limitations](#9-limitations)
10. [Citation](#10-citation)
11. [License](#11-license)
12. [Acknowledgements](#12-acknowledgements)

---

## 1. Overview

| | |
|:--|:--|
| **Study area** | Sokhda village, Vadodara district, Gujarat, India: 966 farm plots, 447.5 ha, median plot 0.274 ha |
| **Season** | Kharif (monsoon) 2025, June to November |
| **Crops** | Rice, cotton, maize, bajra (pearl millet), groundnut |
| **Primary data** | Six Capella X-band (about 9.65 GHz) Stripmap SLC scenes, HH polarisation: 6 Jun, 19 Jun, 14 Aug, 13 Oct, 29 Oct and 12 Nov 2025 |
| **Ground truth** | None, for crop type, crop health or yield, at any level |

| Phase | Competition | Task | Folder |
|:--|:--|:--|:--|
| 1 | [Round 1](https://www.kaggle.com/competitions/anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge) | Cultivated area of five crops in 29 villages | [`phase1_village_crop_area/`](phase1_village_crop_area/) |
| 2 | [Round 2](https://www.kaggle.com/competitions/anrf-aise-hack-2-0-round-2-sar-crop-health-yield-estimation) | Crop type, health index and yield to date for 966 plots (four scenes) | [`phase2_crop_type_health/`](phase2_crop_type_health/) |
| 3 | [Round 3](https://www.kaggle.com/competitions/anrf-aise-hack-2-0-round-3-sar-crop-yield-forecasting) | Final yield forecast for 966 plots and village aggregation by crop (six scenes) | [`phase3_yield_forecast/`](phase3_yield_forecast/) |
| — | — | Known-answer benchmark of feature attribution; paper figures | [`paper/`](paper/) |

Each folder has its own README describing its method, inputs, outputs and usage.

---

## 2. Key Results

**Village forecast (Phase 3).** Production is 710.0 t over 447.5 ha (1.587 t/ha). The 90% interval from crop-label and anchor uncertainty is 598–918 t.

| Crop | Plots | Area (ha) | Yield (t/ha) | Production (t) | Type of estimate |
|:--|--:|--:|--:|--:|:--|
| Cotton (lint) | 455 | 193.4 | 0.730 | 141.2 | Forecast: about 72% of the crop still on the plant on 12 Nov |
| Groundnut | 221 | 137.7 | 2.514 | 346.1 | Near-complete measurement: mature by late October |
| Rice | 86 | 47.4 | 1.690 | 80.2 | Retrospective reconstruction |
| Bajra | 149 | 42.3 | 1.910 | 80.8 | Retrospective reconstruction |
| Maize | 55 | 26.7 | 2.312 | 61.8 | Retrospective reconstruction |
| **All** | **966** | **447.5** | **1.587** | **710.0** | |

**Other results**

- **Village crop areas (Phase 1):** official leaderboard MSE of 11.071 over 29 villages × 5 crops.
- **Validation without labels (Phase 3):** 19 of 19 pre-registered checks pass. 96% of the variance in the plot ranking is plot-specific rather than spatial.
- **Explanation benchmark (paper):** LightGBM reproduces the known index with R² = 0.97, yet only 40% of its SHAP attribution falls on the true inputs. Feature-level remove-and-retrain ranks SHAP above the true attribution. Aggregating attributions by acquisition date restores rank agreement (ρ = 0.99).

---

## 3. Approach

The study follows a dasymetric principle: quantities known only in aggregate are held fixed, and the SAR supplies the variation within them.

```
 Phase 1 (29 villages)              Phase 2 (966 plots, 4 scenes)          Phase 3 (966 plots, 6 scenes)
 Dynamic World cropland             SAR crop evidence per plot             SAR preprocessing (β⁰ = SF²·|z|²)
 × crop share                       + village shares as constraint   ──►   crop map from Phase 2
 × satellite signatures      ──►    → plot crop map                        within-crop health index
 → village crop shares                                                     district anchor × modifier
                                                                           → plot yield, village total, uncertainty
                                                                                      │
                                                      paper/shap_benchmark  ◄─────────┘
                                                      (health index used as a known answer)
```

| Known in aggregate | Distributed to | Signal from SAR |
|:--|:--|:--|
| Village crop shares (Phase 1) | 966 plot crop labels (Phase 2) | Per-plot crop evidence: flooding, canopy, timing |
| District yield statistics (Phase 3) | 966 plot yields (Phase 3) | Within-crop ranking from the health index |

The SAR determines which crop each plot carries and how plots rank within a crop. The yield level comes from published statistics.

---

## 4. Repository Structure

```
sokhda-sar-yield/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes                 files stored byte for byte (output checksums depend on it)
│
├── phase1_village_crop_area/      Phase 1: village crop areas
│   ├── notebook/                  submitted notebook, input tables, reference submission
│   └── feature_tables/            scripts that built the village tables from imagery
│
├── phase2_crop_type_health/       Phase 2: plot crop type, health index, yield to date
│   ├── src/                       21 scripts
│   ├── notebooks/                 end-to-end notebook
│   ├── data_aux/                  village crop shares and district yield
│   └── results/                   submission, feature store, tables
│
├── phase3_yield_forecast/         Phase 3: final forecast, aggregation, uncertainty
│   ├── src/                       34 scripts
│   ├── data_aux/                  yield anchors and inherited per-plot tables
│   └── results/                   submission, 47 tables, 14 figures
│
└── paper/
    ├── shap_benchmark/            known-answer benchmark of feature attribution
    └── figures/                   paper figures 1–5 and dataset figures D1–D5
```

---

## 5. Getting Started

### 5.1 Requirements

- Python 3.12 (all scripts were run with 3.12.8).
- The packages in `requirements.txt`, mainly:
  - numpy, pandas, scipy;
  - rasterio, geopandas, pyproj;
  - scikit-learn, scikit-image;
  - shap, LightGBM;
  - matplotlib.
- Optional, only to rebuild the Phase 1 input tables: a Google Earth Engine account, with `earthengine-api` and `geemap`.

### 5.2 Installation

```bash
git clone https://github.com/Jenish-fngghd/sokhda-sar-yield.git
cd sokhda-sar-yield
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5.3 Data

**Competition data, not redistributed.** The Capella scenes and plot shapefiles must be downloaded from the competition pages:

| Phase | Download folder | Place inside | Or set |
|:--|:--|:--|:--|
| 1 | `anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge/` | only needed to rebuild the village tables | `R1_DATA` |
| 2 | `anrf-aise-hack-2-0-round-2-sar-crop-health-yield-estimation/` | `phase2_crop_type_health/` | `AISE_DATA` |
| 3 | `anrf-aise-hack-2-0-round-3-sar-crop-yield-forecasting/` (about 3.2 GB) | `phase3_yield_forecast/` | `AISE_DATA` |

**Included in this repository:**

- the Phase 1 village tables;
- the Phase 2 and Phase 3 per-plot feature stores;
- the published statistics used as yield anchors;
- every output table and figure;
- two third-party control tables from other teams' public Round 2 submissions, documented in `phase3_yield_forecast/data_aux/inherited/README.md`.

**Open data used by the code:**

| Data | Access | Used in |
|:--|:--|:--|
| Sentinel-1 RTC, Sentinel-2 L2A | Microsoft Planetary Computer, anonymous | Phase 2 and 3 independent checks; Phase 3 climatology |
| Impact Observatory annual land use / land cover | Microsoft Planetary Computer | Phase 2 cropland context |
| ERA5-Land | Open-Meteo archive API | Phase 3 weather at each acquisition |
| NASA POWER rainfall | NASA POWER API | Phase 2 rainfall context |
| Sentinel-1, Sentinel-2, Dynamic World, ESA WorldCover and WorldCereal, ALOS-2 | Google Earth Engine | Phase 1 village tables |
| District crop statistics | Directorate of Agriculture, Gujarat; data.gov.in | Phase 2 and 3 yield anchors |

**Privacy.** No owner names, land-record survey numbers or other personal data are included.

---

## 6. Reproducing the Results

Each phase can be run from the shipped tables in minutes, or from the raw SLC scenes. Commands are given from the repository root. On Windows, `python` may be `py -3.12`.

| Phase | Command | Needs | Expected output |
|:--|:--|:--|:--|
| 1 | `jupyter nbconvert --to notebook --execute phase1_village_crop_area/notebook/final_submission.ipynb --output executed.ipynb` | nothing beyond this repository | `submission.csv` identical to `submission_11.071.csv` |
| 2 | `cd phase2_crop_type_health && python src/d4_submission.py` | Phase 2 shapefiles | `results/submission.csv`, md5 `89b0e4e2aef63ace4989fc0a44590ee5` |
| 3 | `cd phase3_yield_forecast && python src/p3_build.py && python src/p3_checks.py` | Phase 3 download | `results/submission.csv`, md5 `8304407ed23bc67168fb26fca21a03d1`; 19 / 19 checks pass |
| Benchmark | `cd paper/shap_benchmark && python analyze.py && python make_figs.py` | Phase 3 download | `results/summary.csv`; Figures 6–10 |
| Figures | `python paper/figures/xai/make_xai_figs.py` and `python paper/figures/dataset/make_dataset_figs.py` | Phase 3 download | Figures 1–5 and D1–D5 |

**Full runs from raw imagery:**

| Step | Approximate time (16-thread desktop) |
|:--|:--|
| Phase 2 preprocessing to submission | 2 minutes |
| Phase 3 preprocessing to submission | 1.5 minutes |
| Phase 3 climatology / representativeness analyses (download Sentinel-1 archives) | 8 / 46 minutes |
| Benchmark model grid and remove-and-retrain | about 12 hours of total run time; executed as Kaggle CPU jobs |

The phase READMEs give the full sequences.

---

## 7. Paper Section to Code

| Paper section | Code |
|:--|:--|
| Study area and dataset figures | `paper/figures/dataset/make_dataset_figs.py` |
| SAR preprocessing and feature extraction | `phase3_yield_forecast/src/p1_prep.py`, `p1_calib.py`, `p1_features.py` |
| Village-level crop area estimation | `phase1_village_crop_area/notebook/final_submission.ipynb`, `feature_tables/` |
| Plot-level crop type mapping | `phase2_crop_type_health/src/d4_submission.py` (`crop_evidence`, `assign_crop`, `fit_prior`) |
| Crop health index | `phase3_yield_forecast/src/p3_build.py` (`health_parts`, `derive_weights`, `health_index`) |
| Yield forecasting and village aggregation | `phase3_yield_forecast/src/p3_build.py` (`modifier`, `zones`); `data_aux/anchors_r3.csv` |
| Validation without ground truth | `phase3_yield_forecast/src/p3_checks.py`, `p3_spatial.py`, `p3_edge.py`, `p3_speckle.py`, `p3_climatology.py`, `p3_representative.py`; `phase2_crop_type_health/src/i5_validation.py`, `check_yield.py`, `independent_crop.py` |
| Testing the explanations | `phase3_yield_forecast/src/p3_voi.py`; `paper/shap_benchmark/` |
| Uncertainty | `phase3_yield_forecast/src/p3_uncertainty.py` |
| Figures 1–5 / 6–10 | `paper/figures/xai/make_xai_figs.py` / `paper/shap_benchmark/make_figs.py` |

---

## 8. Configuration

| Variable | Used by | Purpose | Default |
|:--|:--|:--|:--|
| `AISE_DATA` | Phases 2 and 3, paper | Path to the phase's competition download | the download folder inside the phase folder |
| `AISE_R2` | Phase 3 | Path to the Phase 2 folder (consistency check) | `../phase2_crop_type_health` |
| `AISE_VILLAGE` | Phases 2 and 3 | Village name used in shapefile names | `Sokhda` |
| `AISE_FARMS`, `AISE_VILLAGE_SHP` | Phases 2 and 3 | Explicit shapefile paths | derived from `AISE_DATA` |
| `R1_DATA`, `R1_WORK` | Phase 1 table builders | Phase 1 download; working folder | download folder name; current folder |
| `EE_PROJECT` | Phase 1 table builders | Google Earth Engine Cloud project ID | none |
| `NJOBS` | Benchmark | Parallel jobs for model fitting | 6 |

**Code comments.** The scripts were written during the competition. Their comments contain bracketed tags (for example `[J4]`) and mention internal planning documents; neither is part of this repository.

---

## 9. Limitations

- **No ground truth:** no plot-level truth exists for crop type, health or yield, so plot-level accuracy is not claimed.
- **Scope:** one village, one season, one polarisation.
- **Crop labels:**
  - per-plot labels agree weakly with an independent Sentinel-1 + Sentinel-2 map (Cohen's κ = 0.112);
  - the village crop mix is constrained, but individual labels are not;
  - the crop map contributes 236.5 t of village uncertainty.
- **Health index:**
  - it is a within-crop ranking;
  - its agreement with same-day Sentinel-2 NDVI is very weak (area-weighted within-crop ρ = −0.02 on 13 Oct and +0.09 on 12 Nov).
- **Yield level:** this comes from district statistics, not from the SAR; the anchor contributes 385.2 t of village uncertainty.
- **Type of estimate:** the classification of each crop's estimate (forecast, measurement or reconstruction) rests on published crop calendars.
- **Phase 1:** the leaderboard score of 11.071 depends on calibration against the competition leaderboard, which cannot be repeated.
- **Phase 2 calibration:** Phase 2 used an unsquared calibration constant, corrected in Phase 3; Phase 3 uses the Phase 2 crop labels.

---

## 10. Citation

If you use this code or its outputs, please cite the accompanying paper. The full reference will be added on publication.

---

## 11. License

The code and its outputs are released under the [MIT License](LICENSE). The competition data, the third-party tables and the open datasets remain under their own terms (see `LICENSE`).

---

## 12. Acknowledgements

**Team GDHTM:** Yash Sorathiya, Jenish Sorathiya, Yajurshi Velani, Mahi Parmar, Aayush Pandya.

We thank:

- **ANRF and the AISEHack 2.0 organisers**, for the competition and the Capella data;
- **Project Orion (Team Apes)**, for their public per-plot features;
- **CodingBits, 8bit, Megalodon, Orion and DeepThinkers**, for their public Round 2 crop labels;
- **the providers of the open datasets listed above.**
