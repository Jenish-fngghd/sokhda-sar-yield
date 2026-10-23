# Inherited per-farm tables

These tables were prepared before Phase 3. Phase 3 reads them; it does not rebuild them. `src/p0_inherit.py` records how they were copied and joined. It reads from the Phase 2 working folder, which is not part of this repository, so it is kept for provenance and should not be re-run.

The per-farm tables use `farm_id` 1–966, matching the `FID` of the competition's plot shapefile.

## Our own tables

| File | Contents | Origin | Read by |
|:--|:--|:--|:--|
| `label_dist.csv` | Phase 2 crop label per farm and the five-class posterior | the Phase 2 crop map | `p3_build.py`, `p1_traj.py`, `p3_t1_gai.py`, `p3_t2_wcm.py`, and the paper's benchmark and figures |
| `uncert.csv` | calibrated per-farm sampling standard deviation of the health index (885 farms with enough pixels) | Phase 2 split-half analysis | `p3_build.py` |
| `gee.csv` | Dynamic World class probabilities and ESA WorldCereal layers per farm | Google Earth Engine (Dynamic World, CC BY 4.0; ESA WorldCereal) | `p3_build.py` (the `dw_mode` non-crop flag) |
| `anchors.csv` | 461 season-wise crop records for Vadodara district, 1997–2012 (area and production) | data.gov.in, Open Government Data Platform India, district-wise season-wise crop production statistics | source of the year-to-year yield CVs written into `p3_uncertainty.py` |

The CVs in `p3_uncertainty.py` recompute exactly from `anchors.csv`, as kharif production ÷ area:

| Crop | Seasons | CV |
|:--|--:|--:|
| Cotton (lint) | 15 | 0.397 |
| Rice | 15 | 0.332 |
| Maize | 15 | 0.243 |
| Bajra | 16 | 0.283 |
| Groundnut | 16 | 0.491 |

## Third-party tables

These come from other teams' **public** Round 2 submissions to the same competition. They are used only as external controls, never as inputs to the forecast.

| File | Contents | Source | Read by |
|:--|:--|:--|:--|
| `consensus.csv` | crop label per farm from six teams (ours, CodingBits, 8bit, Megalodon, Orion, DeepThinkers), the majority vote and the agreement count | each team's public Round 2 submission on Kaggle | `p3_uncertainty.py` (externally measured label uncertainty, 236.5 t), `p3_dawidskene.py` |
| `orion_feat.csv` | Project Orion's per-farm γ⁰ for the four Phase 2 dates | Project Orion (Team Apes), <https://www.kaggle.com/code/sumit1703/project-orion> | `p1_xcheck.py` (cross-implementation radiometry check), `tests_regression.py` |

Credit for these tables belongs to their authors. Without them, `p3_uncertainty.py` reports every term except the externally measured label term, and `p1_xcheck.py` and one regression test cannot run.
