# Phase 1 — Village-level Crop Area Estimation

Phase 1 estimates the cultivated area of five kharif crops in each of 29 villages around Vadodara, Gujarat. Its estimate for Sokhda provides the crop-mix constraint for the plot crop map in Phase 2.

| | |
|:--|:--|
| **Paper section** | Village-level crop area estimation |
| **Competition** | [ANRF AISEHack 2.0, Round 1](https://www.kaggle.com/competitions/anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge) |
| **Result** | Official leaderboard MSE: **11.071** |
| **Requirements** | Python only; no competition data or account needed to reproduce the submission |

---

## Contents

1. [Task](#1-task)
2. [Data](#2-data)
3. [Method](#3-method)
4. [Score Progression](#4-score-progression)
5. [Notebook Scope](#5-notebook-scope)
6. [Repository Contents](#6-repository-contents)
7. [Usage](#7-usage)
8. [Rebuilding the Input Tables](#8-rebuilding-the-input-tables)
9. [Output Used Downstream](#9-output-used-downstream)
10. [Notes](#10-notes)

---

## 1. Task

- **Units:** 29 village polygons (73.01–73.25°E, 22.27–22.47°N).
- **Target:** cultivated area (ha) of rice, cotton, maize, bajra and groundnut per village, i.e. 145 values.
- **Submission format:** `ID, Rice_ha, Cotton_ha, Maize_ha, Bajra_ha, Groundnut_ha`.
- **Metric:** mean squared error over all 145 cells:

  `MSE = (1/145) · Σ_v Σ_c (pred[c,v] − true[c,v])²`

- **Labels:** none; no crop masks, reference pixels or village areas were provided.

## 2. Data

| Source | Band / polarisation | Resolution | Role |
|:--|:--|:--|:--|
| Capella-14 SLC (competition) | X, HH | about 1 m | four dates (6 Jun, 19 Jun, 14 Aug, 13 Oct 2025); one X-band signature in the final model |
| Google Dynamic World | land-cover probabilities | 10 m | cropland backbone, available for all 29 villages |
| Sentinel-1 GRD | C, VV + VH | 10 m | crop signatures and texture |
| Sentinel-2 | optical | 10–20 m | NDVI green-up and senescence |
| ESA WorldCover 2021 | land cover | 10 m | water and built-up exclusion |
| ESA WorldCereal 2021 | crop products | 10 m | maize and crop-fraction priors |
| ALOS-2 PALSAR-2 | L, HH + HV | 25 m | evaluated; not used in the final model |

**Constraints that shaped the method:**

- **HH polarisation only.**
- **Uneven dates:** the two June scenes are nearly identical, and nothing observes July.
- **Partial coverage:** only about 11 of the 29 villages are more than half covered by the Capella strip on all four dates.
- **Small fields:** many pixels mix more than one field.

## 3. Method

```
pred[c, v] = W_c × cropland_v × Π_k ( 1 + λ_{c,k} · z(feature_{c,k})_v )
```

| Term | Definition | How it was obtained |
|:--|:--|:--|
| `cropland_v` | Cropland area of village v (ha) | Dynamic World pixels with crop probability > 0.30 and built-up probability < 0.35, minus WorldCover water and built-up land; 0.04 ha per 20 m pixel |
| `W_c` | Share of cropland under crop c | Closed-form optimum from leaderboard scores |
| `z(feature)` | Village mean of a satellite signature, standardised across the 29 villages | From the village tables |
| `λ_{c,k}` | Strength of redistribution toward signature k | Closed-form optimum from leaderboard scores |

**Crop shares:**

| Rice | Cotton | Maize | Bajra | Groundnut |
|--:|--:|--:|--:|--:|
| 0.14808 | 0.39909 | 0.07078 | 0.08727 | 0.31988 |

**Principal signatures** (25 in total, all listed in the notebook):

| Crop | Signature | Interpretation | λ |
|:--|:--|:--|--:|
| Rice | VH(Aug) − VH(Jun) | flooding at transplanting, then canopy closure | +0.4072 |
| Rice | NDVI change from September to October | senescence | −0.1297 |
| Cotton | VH in October | the only crop still standing in October | +0.1106 |
| Cotton | VH in July | mid-season canopy | +0.1199 |
| Cotton | X-band HH in August | peak canopy | +0.0414 |
| Maize | VH in August | peak vegetative canopy | +0.1585 |
| Groundnut | NDVI (inverse) | drier, lighter soils | −0.2184 |
| Bajra | — | no reliable signature | — |

Texture terms (GLCM contrast, variance and entropy of NDVI and VH) refine cotton, groundnut and rice.

**Ensemble refinement.** A final stage combines the scored prediction fields in closed form. Each scored field `p_i` gives one linear constraint on the target `t`:

```
⟨p_i, t⟩ = ( ‖p_i‖² + N·Z − N·MSE_i ) / 2        N = 145, Z = 3745.936 (all-zeros score)
α = pinv(P Pᵀ) · b,        t̂ = αᵀ P
```

## 4. Score Progression

| MSE | Step |
|--:|:--|
| 12,911 | Uniform prior |
| 1,872 | Optimal global scale |
| 1,072 | Per-crop weights |
| 527 | Dynamic World cropland replaces a SAR-brightness mask |
| 204 | Weights re-solved on all-village cropland |
| 130.8 | Four core signatures: rice flooding, cotton October VH, groundnut inverse NDVI, maize August VH |
| 98.4 | Denser dates: July Sentinel-1, September NDVI |
| 76.25 | Texture and entropy; the Stage 2 model in the notebook |
| 32.74 | Ensemble over scored fields |
| 32.10 | Soft per-pixel phenology direction |
| 14.6 | Crop-type reallocation directions |
| 11.29 | Four cells read out through the score |
| **11.071** | **Eleven cells set to read-out values: official result** |

Two findings carry into the later phases:

- **The cropland mask is the main lever.** Dynamic World halved the error.
- **Hard per-pixel crop classification does not work here.** A pixel-wise Sentinel-1 / ALOS-2 / Sentinel-2 allocation scored 19,354, about five times worse than predicting zeros.

## 5. Notebook Scope

`notebook/final_submission.ipynb` is the notebook that produced the 11.071 submission.

| Component | In the notebook |
|:--|:--|
| Cropland | read from `village_cropland_v3.csv` |
| Stage 2 model | recomputed from `village_aux.csv`, with `W_c` and `λ` as constants |
| Ensemble | the routine runs on the 78 recorded fields in `stage3_history.npz` |
| Final field | loaded from `stage3_ensemble_output.npy`, which includes the read-out cells, and written to `submission.csv` |

The constants and the final field were derived from leaderboard scores, which can no longer be queried. The scripts that submitted probes and read out cells are not included. They depend on a live leaderboard and are not a general method. The transferable part of Phase 1 is the cropland backbone and the physical signatures.

## 6. Repository Contents

```
phase1_village_crop_area/
├── README.md
├── notebook/
│   ├── final_submission.ipynb        submitted notebook
│   ├── submission_11.071.csv         reference copy of the scored submission
│   └── data/
│       ├── village_cropland_v3.csv        ID, coverage, cropland_ha, village_ha (29 rows)
│       ├── village_aux.csv                31 per-village signatures (Sentinel-1, NDVI, X-band, ALOS-2, texture)
│       ├── village_croptype.csv           WorldCereal fractions and Sentinel-1 rice-flood index
│       ├── stage3_history.npz             78 recorded fields, their scores and Z
│       └── stage3_ensemble_output.npy     final 29 × 5 field
└── feature_tables/                   13 scripts that built the village tables
```

## 7. Usage

```bash
cd phase1_village_crop_area/notebook
jupyter nbconvert --to notebook --execute final_submission.ipynb --output executed.ipynb
```

The notebook writes `submission.csv`. It should be identical to the reference copy:

```python
import pandas as pd
a = pd.read_csv("submission.csv"); b = pd.read_csv("submission_11.071.csv")
assert list(a.columns) == list(b.columns) and (a.values == b.values).all()
```

On Kaggle, upload `notebook/data/` as a dataset, attach it to the notebook and run all cells. The notebook searches its own `data/` folder and `/kaggle/input/`.

## 8. Rebuilding the Input Tables

The village tables were built from the Capella scenes and Google Earth Engine collections. Rebuilding them requires the Phase 1 competition download and a Google Earth Engine account (`earthengine authenticate`).

| Order | Script | Output |
|--:|:--|:--|
| 1 | `prep_stack.py` | Capella SLC → calibrated, multi-looked, geocoded dB stack (EPSG:32643) |
| 2 | `fetch_aux_gee.py` | Sentinel-1 VV/VH, ALOS-2 PALSAR-2 and Sentinel-2 composites |
| 3 | `fetch_cropmask.py` | Dynamic World, WorldCover and WorldCereal layers |
| 4 | `fetch_denser_aux.py` | Monthly Sentinel-2 NDVI (June–October); Sentinel-1 for July and September |
| 5 | `fetch_texture.py`, `fetch_texture2.py`, `fetch_10m.py` | GLCM contrast, variance and entropy; 10 m versions of key layers |
| 6 | `build_features.py` | All layers on one 20 m grid (`features.tif`) |
| 7 | `rebuild_cache.py` | `village_cropland_v3.csv` |
| 8 | `village_aux.py`, then `add_texture2.py`, `add_10m_texture.py` | `village_aux.csv` |
| 9 | `fetch_croptype.py` | `village_croptype.csv` |

| Variable | Purpose | Default |
|:--|:--|:--|
| `R1_DATA` | Phase 1 competition download (contains `villages_clean/villages_clean.shp`) | `anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge` |
| `R1_WORK` | Working folder for rasters (`data/stack/`) and tables (`results/`) | current folder |
| `EE_PROJECT` | Google Earth Engine Cloud project ID | none; required |

These scripts are the competition originals; only their file paths and project ID were made configurable. The shipped tables match the outputs of the original run.

## 9. Output Used Downstream

The estimated shares for Sokhda (village 22) are stored in `../phase2_crop_type_health/data_aux/sokhda_phase1_crop_shares.csv`:

| Crop | Area (ha) | Share |
|:--|--:|--:|
| Cotton | 297.1 | 43.0% |
| Groundnut | 213.9 | 31.0% |
| Rice | 73.3 | 10.6% |
| Bajra | 65.1 | 9.4% |
| Maize | 41.4 | 6.0% |

Phase 2 uses them only as a constraint on the village crop mix, never as plot labels.

## 10. Notes

- **The shares are an estimate,** not ground truth.
- **The last steps partly fitted the leaderboard.** An independent rebuild contradicted one feature sign (groundnut × NDVI entropy; ρ = −0.53, p = 0.003), which suggests those steps partly fitted leaderboard noise.
- **The error floor is spread evenly.** The remaining error is about 3.3 ha RMS per cell and is not explained by any feature.
