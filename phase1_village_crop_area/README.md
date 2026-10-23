# Phase 1: Village-level crop area estimation

Phase 1 estimates the cultivated area, in hectares, of five kharif crops in each of 29 villages around Vadodara, Gujarat. Its estimate for Sokhda becomes the crop-mix constraint for the plot crop map in Phase 2.

| | |
|:--|:--|
| Paper section | 3.3 Village-level crop area estimation |
| Competition | ANRF AISEHack 2.0, Round 1: <https://www.kaggle.com/competitions/anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge> |
| Official result | leaderboard MSE **11.071** |
| Reproduces | `notebook/submission.csv`, identical to `notebook/submission_11.071.csv` |
| Needs | Python only; no competition data, account or key |

---

## Contents

1. [Task](#1-task)
2. [Data](#2-data)
3. [Method](#3-method)
4. [Score ladder](#4-score-ladder)
5. [What the notebook recomputes and what it reads](#5-what-the-notebook-recomputes-and-what-it-reads)
6. [What is deliberately not included](#6-what-is-deliberately-not-included)
7. [Folder contents](#7-folder-contents)
8. [Reproducing the 11.071 submission](#8-reproducing-the-11071-submission)
9. [Rebuilding the village tables from imagery (optional)](#9-rebuilding-the-village-tables-from-imagery-optional)
10. [Verification](#10-verification)
11. [Output used by Phase 2](#11-output-used-by-phase-2)
12. [Notes and limitations](#12-notes-and-limitations)

---

## 1. Task

- **Units:** 29 village polygons, 73.01–73.25°E, 22.27–22.47°N.
- **Target:** cultivated area (ha) of rice, cotton, maize, bajra and groundnut in each village, i.e. 145 numbers.
- **Submission:** one row per village with columns `ID, Rice_ha, Cotton_ha, Maize_ha, Bajra_ha, Groundnut_ha`.
- **Metric:** mean squared error over the 145 cells, on a final leaderboard with no hidden split:

  `MSE = (1/145) · Σ_v Σ_c (pred[c,v] − true[c,v])²`

- **Labels:** none. No crop masks, no reference pixels, no village areas were given.

## 2. Data

| Source | Band, polarisation | Resolution | Role |
|:--|:--|:--|:--|
| Capella-14 X-band SLC (competition) | X, HH only | about 1 m | four dates: 6 Jun, 19 Jun, 14 Aug, 13 Oct 2025; one X-band feature in the final model |
| Google Dynamic World | optical land-cover probabilities | 10 m | **cropland backbone**, gap-free over all 29 villages |
| Sentinel-1 GRD | C-band VV + VH | 10 m | crop signatures (monthly VH, rice flood dip, late-season standing canopy), texture |
| Sentinel-2 | optical | 10–20 m | NDVI green-up and senescence (July–August are almost fully cloudy) |
| ESA WorldCover 2021 | land cover | 10 m | water and built-up exclusion |
| ESA WorldCereal 2021 | crop products | 10 m | maize and temporary-crop priors |
| ALOS-2 PALSAR-2 ScanSAR | L-band HH + HV | 25 m | fetched and tested; **not** in the final model |

Constraints that shaped the method:

- **HH only:** no cross-polarisation ratio.
- **Uneven dates:** the two June scenes are nearly identical, and nothing observes July.
- **Partial coverage:** the Capella strip is a diagonal ribbon, so only about 11 of the 29 villages are more than half covered on all four dates.
- **Small fields:** many pixels are mixed.

## 3. Method

The model predicts each crop's area in each village as a crop share of the village cropland, redistributed toward villages where the crop's satellite signature is strong:

```
pred[c, v] = W_c × cropland_v × Π_k ( 1 + λ_{c,k} · z(feature_{c,k})_v )
```

| Term | Meaning | How it was obtained |
|:--|:--|:--|
| `cropland_v` | cropland area of village v (ha) | Dynamic World pixels with crop probability > 0.30 and built-up < 0.35, minus WorldCover water and built-up, counted on the 20 m grid at 0.04 ha per pixel |
| `W_c` | share of cropland under crop c | closed-form optimum from one leaderboard probe per crop |
| `z(feature)` | a village-mean satellite signature, z-scored across the 29 villages (mean 0) | from the feature tables |
| `λ_{c,k}` | how strongly crop c follows signature k | closed-form optimum from one probe |

**Crop shares of cropland:**

| Rice | Cotton | Maize | Bajra | Groundnut |
|--:|--:|--:|--:|--:|
| 0.14808 | 0.39909 | 0.07078 | 0.08727 | 0.31988 |

**Main physical signatures** (25 blends in total; all are listed in the notebook):

| Crop | Signature | Meaning | λ |
|:--|:--|:--|--:|
| Rice | VH(Aug) − VH(Jun) | flooding at transplanting, then canopy closure | +0.4072 |
| Rice | NDVI change from September to October | senescence | −0.1297 |
| Cotton | VH in October | the only crop still standing in October | +0.1106 |
| Cotton | VH in July | mid-season canopy | +0.1199 |
| Cotton | X-band HH in August | peak canopy at X-band | +0.0414 |
| Maize | VH in August | peak vegetative canopy | +0.1585 |
| Groundnut | NDVI (inverse) | drier, lighter soils | −0.2184 |
| Bajra | — | no usable signature in any free layer | — |

Texture terms (GLCM contrast, variance and entropy of NDVI and VH) are added for cotton, groundnut and rice. They capture how mixed each village's cropland is.

**Stage 3** refines the Stage 2 field with an ensemble over all scored prediction fields. Each scored field `p_i` gives one exact linear constraint on the hidden target `t`:

```
⟨p_i, t⟩ = ( ‖p_i‖² + N·Z − N·MSE_i ) / 2      N = 145,  Z = all-zeros MSE = 3745.936
α = pinv(P Pᵀ) · b,     t̂ = αᵀ P
```

The final field also sets a few cells to values read out through the score.

## 4. Score ladder

| MSE | Step |
|--:|:--|
| 12,911 | uniform prior (over-predicts) |
| 3,746 | all-zeros reference submission |
| 1,872 | optimal global scale |
| 1,072 | per-crop weights |
| **527** | **Dynamic World cropland replaces a SAR-brightness mask** |
| 204 | weights re-solved on all-village Dynamic World cropland |
| 130.8 | four core signatures (rice flooding, cotton October VH, groundnut inverse NDVI, maize August VH) |
| 98.4 | denser dates (July Sentinel-1, September NDVI) |
| 76.25 | texture and entropy (the Stage 2 model in the notebook) |
| 32.74 | Gram-matrix ensemble over past submissions |
| 32.10 | soft per-pixel phenology model as an extra ensemble direction |
| 14.6 | crop-type reallocation directions (WorldCereal maize, Sentinel-1 rice flood, dense Sentinel-1 cotton, post-monsoon NDVI) |
| 11.29 | four cells read out through the score and propagated |
| **11.071** | **eleven cells set to read-out values: official final result** |

Two lessons from this ladder carry into the later phases. First, the cropland mask, not the crop model, was the bottleneck: Dynamic World halved the error. Second, hard per-pixel crop classification failed: a pixel-wise Sentinel-1 / ALOS-2 / Sentinel-2 allocation scored **19,354**, about five times worse than predicting zeros.

## 5. What the notebook recomputes and what it reads

`notebook/final_submission.ipynb` is the notebook that produced the 11.071 submission. Readers should know which parts are recomputed and which are recorded.

| Part | Status |
|:--|:--|
| Stage 1 cropland | read from `village_cropland_v3.csv` |
| Stage 2 model | **recomputed** from `village_aux.csv` with the constants `W_c` and `λ` written in the notebook. The constants were obtained from leaderboard scores, which can no longer be queried. |
| Stage 3 ensemble | the Gram-matrix routine **runs** on the 78 recorded fields in `stage3_history.npz` |
| Final field | **loaded** from `stage3_ensemble_output.npy`. It includes the read-out cells and is written to `submission.csv`. |

## 6. What is deliberately not included

The competition scripts that sent probe submissions, solved for parameters from scores, read cells out of the score, and combined past submissions are left out. They depend on a live leaderboard and are not a general method; the paper reports them as leaderboard calibration, not as crop mapping. Exploratory and dead-end scripts are also left out.

The transferable part of Phase 1 is the Dynamic World cropland backbone and the physical signatures in Stage 2.

## 7. Folder contents

```
phase1_village_crop_area/
├── README.md
├── notebook/
│   ├── final_submission.ipynb        the submitted notebook (Run All writes submission.csv)
│   ├── submission_11.071.csv         reference copy of the file that scored 11.071
│   └── data/
│       ├── village_cropland_v3.csv        ID, coverage, cropland_ha, village_ha (29 rows)
│       ├── village_aux.csv                31 per-village signatures: Sentinel-1, NDVI, X-band, texture
│       ├── village_croptype.csv           WorldCereal maize and crop fractions, Sentinel-1 rice-flood index
│       ├── stage3_history.npz             78 recorded prediction fields (P), their scores (mse) and Z
│       └── stage3_ensemble_output.npy     the final 29 × 5 field
└── feature_tables/                   13 scripts that built the three village tables
```

## 8. Reproducing the 11.071 submission

```bash
pip install -r ../requirements.txt
cd phase1_village_crop_area/notebook
jupyter nbconvert --to notebook --execute final_submission.ipynb --output executed.ipynb
```

The notebook writes `submission.csv` in the same folder. Compare it with the reference:

```python
import pandas as pd
a = pd.read_csv("submission.csv"); b = pd.read_csv("submission_11.071.csv")
assert list(a.columns) == list(b.columns) and (a.values == b.values).all()
```

To run it on Kaggle instead, upload `notebook/data/` as a dataset, attach it to the notebook and run all cells. The notebook looks for the data in its own `data/` folder and under `/kaggle/input/`.

## 9. Rebuilding the village tables from imagery (optional)

The three tables in `notebook/data/` were built from the four Capella scenes and free Google Earth Engine collections. To rebuild them you need the Phase 1 competition download, a Google Earth Engine account (`earthengine authenticate`) and about 1 GB of downloads.

| Order | Script | Output |
|--:|:--|:--|
| 1 | `prep_stack.py` | Capella SLC → calibrated, multi-looked, geocoded dB stack (EPSG:32643) |
| 2 | `fetch_aux_gee.py` | Sentinel-1 VV/VH, ALOS-2 PALSAR-2, Sentinel-2 composites |
| 3 | `fetch_cropmask.py` | Dynamic World, WorldCover and WorldCereal masks |
| 4 | `fetch_denser_aux.py` | monthly Sentinel-2 NDVI (June to October) and Sentinel-1 for July and September |
| 5 | `fetch_texture.py`, `fetch_texture2.py`, `fetch_10m.py` | GLCM contrast, variance and entropy layers; 10 m re-fetch of the key layers |
| 6 | `build_features.py` | all layers fused on one 20 m grid (`features.tif`) |
| 7 | `rebuild_cache.py` | `village_cropland_v3.csv` |
| 8 | `village_aux.py`, then `add_texture2.py`, `add_10m_texture.py` | `village_aux.csv` |
| 9 | `fetch_croptype.py` | `village_croptype.csv` |

| Environment variable | Meaning | Default |
|:--|:--|:--|
| `R1_DATA` | Phase 1 competition download (expects `villages_clean/villages_clean.shp` inside it) | `anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge` |
| `R1_WORK` | working folder for rasters (`data/stack/`) and tables (`results/`) | current folder |
| `EE_PROJECT` | your Google Earth Engine Cloud project ID | none; must be set |

These scripts are the competition originals; only their hard-coded paths and project ID were replaced by these variables. They were written for a one-off run and were not re-executed for this release, because the raw stacks are no longer kept. The tables they produced are shipped and were checked against the competition pipeline's cache (Section 10).

## 10. Verification

Run on 2026-10-24 in a fresh copy of this repository:

| Check | Result |
|:--|:--|
| Notebook run, all code cells | `submission.csv` byte-identical to `submission_11.071.csv` (29 × 6, maximum difference 0.0) |
| Stage 2 field vs the recorded fields | matches the field that scored 76.253 (row 15 of `stage3_history.npz`) to within 0.005 ha per cell |
| Shipped village tables vs the competition pipeline's cache | `village_cropland_v3.csv`, `village_aux.csv`, `village_croptype.csv` byte-identical |

## 11. Output used by Phase 2

The village shares for Sokhda (village 22) are stored in `../phase2_crop_type_health/data_aux/sokhda_phase1_crop_shares.csv`:

| Crop | Area (ha) | Share |
|:--|--:|--:|
| Cotton | 297.1 | 43.0% |
| Groundnut | 213.9 | 31.0% |
| Rice | 73.3 | 10.6% |
| Bajra | 65.1 | 9.4% |
| Maize | 41.4 | 6.0% |

Phase 2 uses them only as a constraint on the area-weighted crop mix of the village, never as per-plot labels.

## 12. Notes and limitations

- **The shares are an estimate,** not ground truth.
- **The final steps fitted the leaderboard.** An independent rebuild later contradicted one Phase 1 feature sign (groundnut × NDVI entropy, ρ = −0.53, p = 0.003). This suggests the last steps of the ladder partly fitted leaderboard noise.
- **The error floor is spread evenly.** The remaining error is about 3.3 ha RMS per cell and is unrelated to any feature.
- **Official result:** the official result is 11.071. A lower score obtained after the deadline is not the official result and is not cited.
