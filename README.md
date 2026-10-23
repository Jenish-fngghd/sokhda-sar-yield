# Label-free plot-level crop yield forecasting from X-band SAR

**Sokhda village, Vadodara district, Gujarat, India · kharif 2025 · 966 farm plots · 447.5 ha**

This repository holds the code, inputs and outputs of the study behind the paper. The study forecasts the final crop yield of every farm plot in one smallholder village from Capella X-band HH single-look complex (SLC) SAR images, without any yield, crop-health or crop-type labels. It also tests, against a known answer, whether SHAP-style explanations of models trained on such SAR features can be trusted.

The work was carried out by Team GDHTM for ANRF AISEHack 2.0 (Remote Sensing track). The competition had three rounds. **The paper and this repository call them Phases 1, 2 and 3.**

---

## Contents

1. [Study at a glance](#1-study-at-a-glance)
2. [How the three phases connect](#2-how-the-three-phases-connect)
3. [Repository structure](#3-repository-structure)
4. [Paper section to code](#4-paper-section-to-code)
5. [Data availability](#5-data-availability)
6. [Installation](#6-installation)
7. [Quick start: reproduce every headline number](#7-quick-start-reproduce-every-headline-number)
8. [Full reproduction from the raw SAR images](#8-full-reproduction-from-the-raw-sar-images)
9. [Verification record](#9-verification-record)
10. [Environment variables](#10-environment-variables)
11. [Troubleshooting](#11-troubleshooting)
12. [Known limitations](#12-known-limitations)
13. [Citation, licence and team](#13-citation-licence-and-team)

---

## 1. Study at a glance

| | |
|:--|:--|
| **Site** | Sokhda village, Vadodara district, Gujarat, India: 966 farm plots, 447.5 ha, median plot 0.274 ha |
| **Season** | kharif (monsoon) 2025, June to November |
| **Crops** | rice, cotton, maize, bajra (pearl millet), groundnut |
| **Primary data** | six Capella X-band (about 9.65 GHz) Stripmap SLC scenes, HH polarisation only: 6 Jun, 19 Jun, 14 Aug, 13 Oct, 29 Oct and 12 Nov 2025 |
| **Labels** | none, for crop type, health or yield, at any level |
| **Approach** | quantities known only in aggregate are held fixed; the SAR supplies the variation inside them (dasymetric mapping) |

| Phase | Competition round and data | Task | Output |
|:--|:--|:--|:--|
| 1 | [Round 1](https://www.kaggle.com/competitions/anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge) | crop area (ha) of five crops in 29 villages | village crop shares; official leaderboard MSE **11.071** |
| 2 | [Round 2](https://www.kaggle.com/competitions/anrf-aise-hack-2-0-round-2-sar-crop-health-yield-estimation) | crop type, health index and yield to date for 966 plots (four scenes) | the plot crop map used by Phase 3 |
| 3 | [Round 3](https://www.kaggle.com/competitions/anrf-aise-hack-2-0-round-3-sar-crop-yield-forecasting) | final yield forecast for 966 plots and village aggregation by crop (six scenes) | **710.0 t** (1.587 t/ha); 90% interval 598–918 t |
| — | paper | known-answer benchmark of feature attribution; paper figures | LightGBM R² 0.97, yet only 40% of SHAP credit on the true inputs |

**Headline results**

| Crop | Plots | Area (ha) | Final yield (t/ha) | Production (t) | Kind of quantity |
|:--|--:|--:|--:|--:|:--|
| Cotton (lint) | 455 | 193.4 | 0.730 | 141.2 | forecast: about 72% still on the plant on 12 Nov |
| Groundnut | 221 | 137.7 | 2.514 | 346.1 | near-complete measurement |
| Rice | 86 | 47.4 | 1.690 | 80.2 | retrospective reconstruction |
| Bajra | 149 | 42.3 | 1.910 | 80.8 | retrospective reconstruction |
| Maize | 55 | 26.7 | 2.312 | 61.8 | retrospective reconstruction |
| **All** | **966** | **447.5** | **1.587** | **710.0** | |

---

## 2. How the three phases connect

```
 Phase 1 (29 villages)                Phase 2 (966 plots, 4 scenes)          Phase 3 (966 plots, 6 scenes)
 ─────────────────────                ─────────────────────────────          ─────────────────────────────
 Dynamic World cropland               SAR crop evidence per plot             SAR preprocessing (β⁰ = SF²·|z|²)
 × crop share W_c                     + village shares as constraint   ──►   crop map (from Phase 2)
 × satellite signatures        ──►    → plot crop map                        health index (5 parts, within crop)
 → village crop shares                                                       anchor (district statistics) × modifier
   (Sokhda = village 22)                                                     → plot yield, village total, uncertainty
                                                                                        │
                                                       paper/shap_benchmark  ◄──────────┘
                                                       (the health index as a known answer)
```

The common idea is **dasymetric mapping**: a total that is known only for a larger unit is held fixed, and a correlated signal from the SAR distributes it over smaller units.

| Known in aggregate | Distributed over | Signal from the SAR |
|:--|:--|:--|
| village crop shares (Phase 1) | 966 plot crop labels (Phase 2) | per-plot crop evidence (flooding, canopy, timing) |
| district yield statistics (Phase 3 anchors) | 966 plot yields (Phase 3) | within-crop ranking from the health index |

The SAR therefore decides two things: which crop each plot carries, and how plots rank within a crop. It does not set the yield level; published statistics do.

---

## 3. Repository structure

```
sokhda-sar-yield/
├── README.md                     this file
├── LICENSE                       MIT licence for the code (scope notes inside)
├── requirements.txt              pinned Python packages used for verification
├── .gitignore                    competition data and rebuilt rasters are never committed
│
├── phase1_village_crop_area/     Phase 1: village crop areas (29 villages)
│   ├── README.md
│   ├── notebook/                 the submitted notebook, its input tables, the 11.071 file
│   └── feature_tables/           scripts that built the village tables from imagery
│
├── phase2_crop_type_health/      Phase 2: plot crop type, health, yield to date
│   ├── README.md
│   ├── src/                      21 pipeline, validation and figure scripts
│   ├── notebooks/                end-to-end notebook that asserts the submission
│   ├── data_aux/                 village shares (from Phase 1) and district yield
│   └── results/                  submission, feature store, provenance, tables
│
├── phase3_yield_forecast/        Phase 3: final forecast, aggregation, uncertainty
│   ├── README.md
│   ├── src/                      34 scripts: pipeline, checks, ten post-freeze experiments
│   ├── data_aux/                 anchors and inherited per-farm tables
│   └── results/                  submission, provenance, 47 tables, 14 figures
│
└── paper/                        made for the paper, not for the competition
    ├── README.md
    ├── shap_benchmark/           known-answer test of feature attribution
    └── figures/                  Figures 1–5 (xai/) and dataset figures D1–D5 (dataset/)
```

---

## 4. Paper section to code

| Paper section | Code |
|:--|:--|
| 2 Study area and dataset (figures) | `paper/figures/dataset/make_dataset_figs.py` |
| 3.2 SAR preprocessing and feature extraction | `phase3_yield_forecast/src/p1_prep.py`, `p1_calib.py`, `p1_features.py` |
| 3.3 Village-level crop area estimation | `phase1_village_crop_area/notebook/final_submission.ipynb`; `feature_tables/` |
| 3.4.1 Crop type: area-constrained assignment | `phase2_crop_type_health/src/d4_submission.py` (`crop_evidence`, `assign_crop`, `fit_prior`) |
| 3.4.2 Crop health index | `phase3_yield_forecast/src/p3_build.py` (`health_parts`, `derive_weights`, `health_index`) |
| 3.5 Yield forecasting and village aggregation | `phase3_yield_forecast/src/p3_build.py` (`modifier`, aggregation, zones); `data_aux/anchors_r3.csv` |
| 3.6.1 Validation without ground truth | `phase3_yield_forecast/src/p3_checks.py`, `p3_spatial.py`, `p3_edge.py`, `p3_speckle.py`, `p3_climatology.py`, `p3_representative.py`; `phase2_crop_type_health/src/i5_validation.py`, `check_yield.py`, `independent_crop.py` |
| 3.6.2 Testing the explanations | `phase3_yield_forecast/src/p3_voi.py`; `paper/shap_benchmark/` |
| 3.6.3 Uncertainty | `phase3_yield_forecast/src/p3_uncertainty.py` |
| Figures 1–5 | `paper/figures/xai/make_xai_figs.py` |
| Figures 6–10 | `paper/shap_benchmark/make_figs.py` |
| Dataset figures D1–D5 | `paper/figures/dataset/make_dataset_figs.py` |

---

## 5. Data availability

### Not included: download from Kaggle

The competition data (Capella SLC scenes with vendor metadata and ground control points, the plot and village shapefiles, and the host's templates) is distributed by the competitions and is **not redistributed here**. Download it from each phase's competition page (links in [Section 1](#1-study-at-a-glance)).

| Phase | Kaggle download folder | Place it in | Or set |
|:--|:--|:--|:--|
| 1 | `anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge/` | only for rebuilding the village tables | `R1_DATA` |
| 2 | `anrf-aise-hack-2-0-round-2-sar-crop-health-yield-estimation/` | `phase2_crop_type_health/` | `AISE_DATA` |
| 3 | `anrf-aise-hack-2-0-round-3-sar-crop-yield-forecasting/` | `phase3_yield_forecast/` | `AISE_DATA` |

The Phase 3 download (about 3.2 GB) contains all six scenes. The first four are byte-identical to the Phase 2 scenes.

### Included

- **Everything needed to reproduce the deliverables without processing imagery:**
  - the Phase 1 village tables;
  - the Phase 2 and Phase 3 per-farm feature stores;
  - the published statistics used as anchors.
- **Every output table and figure** produced by the code.
- **Two third-party control tables** from other teams' public Round 2 submissions (`phase3_yield_forecast/data_aux/inherited/consensus.csv`, `orion_feat.csv`). They are used only as external checks, with attribution in that folder's README.

### Open data used by the code

| Data | Access | Used in |
|:--|:--|:--|
| Sentinel-1 RTC, Sentinel-2 L2A | Microsoft Planetary Computer, anonymous | Phase 2 and Phase 3 witnesses; Phase 3 climatology and representativeness |
| Impact Observatory 10 m annual land use / land cover | Microsoft Planetary Computer | Phase 2 cropland context |
| ERA5-Land reanalysis | Open-Meteo archive API, no key | Phase 3 weather at each overpass |
| NASA POWER rainfall | NASA POWER API, no key | Phase 2 rainfall context |
| Sentinel-1 GRD, Sentinel-2, Dynamic World, ESA WorldCover and WorldCereal, ALOS-2 | Google Earth Engine (account needed) | Phase 1 village tables; the Phase 3 table `inherited/gee.csv` |
| District crop statistics | Directorate of Agriculture, Gujarat (via Parmar and Bhatt 2025); data.gov.in | Phase 2 and Phase 3 anchors and anchor spread |

### A note on code comments

The scripts are the competition code, and their comments were written during the competition. Two kinds of reference in them point outside this repository:

- bracketed evidence tags such as `[J4]` or `[E1]`, which point to the team's internal research log;
- internal planning documents, such as `MASTER_PLAN.md` or `docs/VALIDATION_SOURCES.md`.

Those documents are not part of this release. The READMEs describe what each script does and what it was checked against.

### Privacy

No owner names, land-record survey numbers or other personal data are included. Scripts that linked plots to the state land-parcel registry were left out.

---

## 6. Installation

All verification runs used Python 3.12.8 on Windows 11. The benchmark's model grid and ROAR were run on Kaggle with Python 3.13; see `paper/README.md`.

```bash
git clone <this repository>
cd sokhda-sar-yield
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the verification runs. Its main packages are:

- numpy 2.4.6, pandas 2.3.0, scipy 1.15.3;
- rasterio 1.5.0, geopandas 1.1.4, pyproj 3.7.2;
- scikit-learn 1.6.1, scikit-image 0.25.2;
- shap 0.53.0, LightGBM 4.5.0;
- matplotlib 3.10.3.

Rebuilding the Phase 1 village tables also needs `earthengine-api` and `geemap` and a Google Earth Engine account. They are listed in `requirements.txt` but commented out.

On Windows the scripts were run as `py -3.12 script.py`; on Linux and macOS use `python script.py`.

---

## 7. Quick start: reproduce every headline number

These steps rebuild the deliverables from the shipped tables. They take a few minutes in total.

```bash
# Phase 1 — no competition data needed
cd phase1_village_crop_area/notebook
jupyter nbconvert --to notebook --execute final_submission.ipynb --output executed.ipynb
#   -> submission.csv, identical to submission_11.071.csv

# Phase 2 — needs only the Phase 2 plot and village shapefiles
cd ../../phase2_crop_type_health
python src/d4_submission.py              # -> results/submission.csv
md5sum results/submission.csv            #    89b0e4e2aef63ace4989fc0a44590ee5

# Phase 3 — needs the Phase 3 download present (common.py requires the scene files on import)
cd ../phase3_yield_forecast
python src/p3_build.py                   # -> results/submission.csv, village and zone tables
python src/p3_checks.py                  #    19 / 19 checks pass
python src/p3_uncertainty.py             #    village 90% interval [597.8, 917.8] t
md5sum results/submission.csv            #    8304407ed23bc67168fb26fca21a03d1

# Paper benchmark — needs the Phase 3 download present
cd ../paper/shap_benchmark
python analyze.py                        # -> results/summary.csv, hypotheses.json, ...
python make_figs.py                      # -> figures/fig6 ... fig10
```

---

## 8. Full reproduction from the raw SAR images

Each phase README gives the full sequence. In short:

| Step | Command | Time (16-thread desktop) |
|:--|:--|:--|
| Phase 2 preprocessing and features | `prep_r2.py` → `farm_stats.py` → `sublook.py` → `wcm.py` → `d4_submission.py` | about 2 min |
| Phase 3 preprocessing and features | `common.py` → `p1_prep.py` → `p1_calib.py` → `p1_features.py` → `p3_build.py` | about 1.5 min |
| Phase 3 checks and tests | `p3_checks.py`, `tests_regression.py` | under 1 min |
| Phase 3 analyses that download data | `p3_climatology.py`, `p3_representative.py` | about 8 min and 46 min |
| Benchmark setup and tuning | `run_benchmark.py setup`, `run_benchmark.py tune` | minutes |
| Benchmark model grid and ROAR | `run_benchmark.py grid all`, `run_benchmark.py roar` | about 12 hours of run time summed over the 105 grid runs (run as Kaggle CPU shards) |

The Phase 1 village tables were built with Google Earth Engine and are not re-run here (see `phase1_village_crop_area/README.md`).

---

## 9. Verification record

All checks below were run on **2026-10-24** in fresh copies of this repository. Every output was compared byte for byte with the shipped copy.

| Part | What was re-run | Result |
|:--|:--|:--|
| Phase 1 | the notebook | `submission.csv` byte-identical to `submission_11.071.csv`; the Stage 2 field equals the recorded 76.253 field to within 0.005 ha |
| Phase 1 | village tables | identical to the competition pipeline's cached tables |
| Phase 2 | raw SLCs → features → submission | `farm_features.csv` byte-identical; `submission.csv` md5 `89b0e4e2aef63ace4989fc0a44590ee5` |
| Phase 2 | every analysis script and the notebook | all 12 tables identical, including witness and context tables re-downloaded from the internet; the notebook reproduces the submission |
| Phase 3 | raw SLCs → features → forecast | feature, coverage and calibration tables byte-identical; `submission.csv` md5 `8304407ed23bc67168fb26fca21a03d1` |
| Phase 3 | checks, tests, every analysis script | 19 / 19 checks; 12 / 12 regression tests; all tables and 14 figures identical, with the two exceptions in the Phase 3 README (floating-point digits in one column; a re-downloaded archive) |
| Benchmark | setup, analysis, figures | `X.csv`, `truth.npz` and all analysis tables byte-identical; figures 6–10 byte-identical |
| Paper figures | Figures 1–5 and D1–D5 | byte-identical |

**Corrections made during verification.** Three Phase 2 statistics quoted in the competition report had been computed on an earlier version of the crop map, which differed from the final map in 43 farms. This repository and the paper use the values recomputed on the final map:

| Statistic | Competition report | Final map |
|:--|:--|:--|
| Cohen's κ, plot labels vs an independent Sentinel-1 + Sentinel-2 map | 0.103 | **0.112** |
| Kruskal–Wallis p, crop classes on Sentinel-2 NDVI / Sentinel-1 VH | 1.8 × 10⁻³⁴ / 7.7 × 10⁻²⁰ | **9.3 × 10⁻³⁵ / 2.7 × 10⁻²⁰** |
| Moran's I of the health index | 0.105 | **0.096** |

---

## 10. Environment variables

| Variable | Used by | Meaning | Default |
|:--|:--|:--|:--|
| `AISE_DATA` | Phase 2, Phase 3, paper | path to that phase's Kaggle download | the download folder inside the phase folder |
| `AISE_R2` | Phase 3 | path to the Phase 2 folder (for check D1) | `../phase2_crop_type_health` |
| `AISE_VILLAGE` | Phases 2 and 3 | village name in the shapefile names | `Sokhda` |
| `AISE_FARMS`, `AISE_VILLAGE_SHP` | Phases 2 and 3 | explicit shapefile paths | from `AISE_DATA` |
| `R1_DATA` | Phase 1 table builders | path to the Phase 1 download | `anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge` |
| `R1_WORK` | Phase 1 table builders | working folder for intermediate rasters | current folder |
| `EE_PROJECT` | Phase 1 table builders | your Google Earth Engine Cloud project ID | none |
| `NJOBS` | benchmark | parallel jobs for model fitting | 6 |

---

## 11. Troubleshooting

| Symptom | Cause and fix |
|:--|:--|
| `FileNotFoundError: no SLCs under ...` | Phase 3 (and the benchmark) require the scene files on import. Set `AISE_DATA` to the Phase 3 download. |
| `Neither Sokhda_Farms.dbf nor Sokhda_Farms.shp could be opened` | Phase 2 needs its plot and village shapefiles. Set `AISE_DATA` to the Phase 2 download. |
| CRS or `pyproj` errors on Windows | A `PROJ_LIB` setting from another GDAL install breaks `pyproj`. `common.py` clears it before `rasterio` loads; keep that line first. |
| "19 June" looks identical to "6 June" | The 19 June download folder also holds a copy of the 6 June scene. `common.py` matches the date in both the folder and the file name; do not bypass it. |
| `HTTP Error 504: Gateway Timeout` | A Planetary Computer or weather service was busy. Re-run the script. |
| `RuntimeWarning: ... invalid winding order` | Harmless; the plot shapefile is corrected automatically on read. |
| Out of memory in the benchmark grid | Lower `NJOBS`. The original local run stopped at 14 jobs; the grid was then run on Kaggle. |

---

## 12. Known limitations

These are stated in the paper and repeated here so that the code is not over-read.

- **No ground truth:** there is no plot-level truth for crop type, health or yield, so no plot-level accuracy is claimed. Results are checked against independent sensors, pre-registered consistency checks and controls with known answers.
- **One village, one season, one polarisation.**
- **Crop labels:** per-plot labels agree weakly with an independent map (κ = 0.112). The village crop mix is constrained; individual labels are not. The crop map is the second-largest source of uncertainty (236.5 t).
- **Health index:** the composite barely agrees with same-day Sentinel-2 NDVI. The area-weighted within-crop ρ is −0.02 on 13 October and +0.09 on 12 November (`p3_fusion.py`). It is a relative within-crop ranking, not a measurement of crop health.
- **Yield level:** this comes from district statistics, not from the SAR. The anchor is the largest source of uncertainty (385.2 t).
- **Kinds of quantity:** the split into forecast, near-complete measurement and reconstruction rests on published crop calendars, because a harvest-timing test with Sentinel-1 could not be resolved at this plot size.
- **Phase 1 score:** the 11.071 score relies on leaderboard calibration that cannot be re-run. Only the cropland and signature model transfers.
- **Phase 2 calibration:** Phase 2 used an unsquared calibration constant, corrected in Phase 3. Phase 3 still uses the Phase 2 crop labels.

---

## 13. Citation, licence and team

**Citation.** If you use this code or its outputs, please cite the paper. The reference will be added here on publication.

**Licence.** The code and its outputs are released under the [MIT License](LICENSE). Competition data, third-party tables and open datasets keep their own terms (see the scope notes in `LICENSE`).

**Team GDHTM:** Yash Sorathiya, Jenish Sorathiya, Yajurshi Velani, Mahi Parmar, Aayush Pandya.

**Acknowledgements.** We thank:

- **ANRF and the AISEHack 2.0 organisers**, for the competition and the Capella data;
- **Project Orion (Team Apes)**, whose public per-farm features made the cross-implementation check possible;
- **CodingBits, 8bit, Megalodon, Orion and DeepThinkers**, whose public Round 2 crop labels made the external label-uncertainty check possible;
- **the providers of the open data** listed in Section 5.
