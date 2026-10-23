# Auxiliary inputs for Phase 2: provenance

Two small tables enter the Phase 2 deliverable. Both are declared inputs. Neither is a label.

## `sokhda_phase1_crop_shares.csv`: village crop-area shares

**Source:** this repository's Phase 1 solution (`../../phase1_village_crop_area/`), for Sokhda, which is village 22 of the 29 Phase 1 villages.

| Column | Meaning |
|:--|:--|
| `sokhda_ha`, `sokhda_share` | estimated area (ha) and area share of each crop in Sokhda |
| `r1_29village_ha`, `r1_29village_share` | the same totals over all 29 Phase 1 villages, for context |

| Crop | Sokhda (ha) | Share |
|:--|--:|--:|
| Cotton | 297.1 | 43.0% |
| Groundnut | 213.9 | 31.0% |
| Rice | 73.3 | 10.6% |
| Bajra | 65.1 | 9.4% |
| Maize | 41.4 | 6.0% |

**What it is:** an **estimate** of the village crop mix from the Phase 1 model, calibrated against the Phase 1 leaderboard (official MSE 11.071 over all 29 villages). It is not ground truth. Phase 2 uses it only as a constraint on the area-weighted crop mix of the village, never as per-farm labels and never as a fitting target.

## `vadodara_apy.csv`: district area, production and yield

**Source:** Directorate of Agriculture, Government of Gujarat (2024), reproduced in Tables 1–3 of Parmar, K. and Bhatt, B. (2025), "Evaluating agricultural patterns and crop shifts in Vadodara-Chhotaudepur District of Gujarat, India", *International Journal of Agriculture and Food Science* 7(5): 55–61, DOI `10.33545/2664844X.2025.v7.i5a.380` (open access).

**Year:** 2022–23. Areas are converted from the paper's hundreds of hectares and production from hundreds of tonnes. `area_share_of_5crops` is renormalised over the five study crops only.

| Crop | Area (ha) | Yield (kg/ha) | Share of the 5 crops |
|:--|--:|--:|--:|
| Cotton (lint) | 185,479 | 776 | 65.3% |
| Rice | 49,818 | 1,690 | 17.5% |
| Maize | 40,794 | 2,312 | 14.4% |
| Bajra | 7,022 | 2,714 | 2.5% |
| Groundnut | 1,004 | 2,514 | 0.35% |

**How it is used:** only the **yield** column enters Phase 2, as the district level of the yield-to-date estimate. The crop mix in this table is **not** used as a prior, for three reasons:

- it is for 2022–23;
- it covers the combined Vadodara-Chhotaudepur district, which includes upland areas unlike Sokhda;
- groundnut sowing in Gujarat was about 125% of normal in kharif 2025.

Cotton yield is lint, not seed cotton. Phase 3 later corrected the bajra value (2,714 kg/ha does not reproduce from the table's own area and production) and adjusted the anchors for 2025–26. See `../../phase3_yield_forecast/data_aux/anchors_r3.csv`.

## Not used as inputs

- **Sentinel-1 and Sentinel-2** (Microsoft Planetary Computer): fetched by `src/witness.py` and `src/witness_season.py` for validation only. No value from them enters `results/submission.csv`.
- **A digital elevation model:** not needed. The area's best-fit terrain slope is about 0.02°, which is negligible against the 6.55° spread in incidence between acquisition dates.
