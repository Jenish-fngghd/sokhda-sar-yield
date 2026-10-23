# Paper experiments and figures

This folder holds the work done for the paper rather than for the competition. It has two parts:

- a **known-answer benchmark** that tests whether SHAP and related explanations recover the true drivers of a model trained on real X-band SAR features;
- the scripts for every paper figure.

Everything here reads the Phase 3 outputs and writes nothing into the phase folders.

| | |
|:--|:--|
| Paper sections | 2 (dataset figures), 3.6.2 Testing the explanations, Results 4.7, Discussion |
| Main result | LightGBM reproduces the index with R² = 0.97, yet only **40%** of its SHAP attribution falls on the true inputs, and remove-and-retrain ranks SHAP **above** the true attribution |
| Needs | the Phase 3 competition download present (`AISE_DATA`), because the shipped index code checks for the scenes on import |

---

## Contents

1. [Folder contents](#1-folder-contents)
2. [The known-answer benchmark](#2-the-known-answer-benchmark)
3. [Key results](#3-key-results)
4. [Reproducing the benchmark](#4-reproducing-the-benchmark)
5. [Figures](#5-figures)
6. [Verification](#6-verification)

---

## 1. Folder contents

```
paper/
├── README.md
├── shap_benchmark/
│   ├── PLAN.md              design, hypotheses H1–H8, thresholds: frozen before any model was trained
│   ├── RESULTS.md           every result, including failed hypotheses, controls, deviations, caveats
│   ├── truth.py             the known answer: the shipped index and its exact Shapley values
│   ├── run_benchmark.py     stages: setup, tune, grid, roar, versions
│   ├── analyze.py           scores every explanation against the truth; tests H1–H8
│   ├── make_figs.py         Figures 6–10 and the shared figure style
│   ├── kaggle/              the three Kaggle CPU kernels (k_rf, k_other, k_roar) that ran the grid and ROAR
│   ├── results/
│   │   ├── X.csv, truth.npz, meta.json        analysis set, exact attributions, designs and controls
│   │   ├── tuning_all.csv, tuning_best.csv    hyperparameter search
│   │   ├── runs.csv                           105 model runs: fit (R²) and run time
│   │   ├── attr/                              every out-of-fold attribution, one file per run
│   │   ├── metrics.csv, summary.csv           scores per run and explainer; means and sd over seeds
│   │   ├── leak_by_feature.csv                attribution share of every feature
│   │   ├── hypotheses.json, roar.csv          hypothesis tests; remove-and-retrain curves
│   │   └── versions_kaggle.json, *.log        environment of the Kaggle runs; tuning and analysis logs
│   └── figures/             fig6 to fig10
└── figures/
    ├── xai/                 Figures 1–5 and make_xai_figs.py
    └── dataset/             dataset figures D1–D5 and make_dataset_figs.py
```

## 2. The known-answer benchmark

**Why it is needed.** Papers that explain SAR crop models usually check SHAP plots for agronomic plausibility, or with remove-and-retrain (ROAR). Neither tells whether the highlighted features are the true drivers, because the true drivers are unknown. Here they are known: the shipped health index is an explicit function of SAR features, so its exact Shapley values can be computed by full enumeration.

**Design** (from `PLAN.md`):

| Element | Choice |
|:--|:--|
| Target T1 | the composite score S = Σ w_k · part_k of the Phase 3 index; 10 causal features (γ⁰ in dB and linear units on several dates, and within-plot CV on 14 Aug) |
| Target T2 | the shipped index itself, 100 · Φ(z_crop(S) / 2), scored within crop; the same 10 features plus crop |
| Inputs D1, exact | only the causal features |
| Inputs D2, realistic | every single-date measurement on all six dates (15 types per date) plus plot area: 91 features, 81 of them unused by the index |
| Noise | none, or Gaussian noise of 0.5 × sd added to the training target only (caps R² near 0.80) |
| Plots | 911 with complete index inputs (55 excluded); 1,365 missing values in unused features median-imputed |
| Models | LightGBM, random forest, and an MLP (T1 only), tuned per configuration |
| Explanations | interventional and path-dependent TreeSHAP; exact (D1) or permutation (D2) SHAP for the MLP; gain and permutation importance |
| Protocol | 5 seeds × 5-fold out-of-fold explanations; 105 model runs |
| Remove-and-retrain | LightGBM, D2, T1, no noise, 3 seeds; features or whole dates removed in truth order, SHAP order and 5 random orders |
| Controls | C1 efficiency of the exact values; C2 agreement with shap's Exact explainer; C3 SHAP on the true function puts no mass on unused features; C4 a permuted target gives no fit; C5 the function reproduces the shipped index |

**Scoring:**

- the share of attribution on true inputs, near copies (|ρ| ≥ 0.98), proxies (0.5 ≤ |ρ| < 0.98) and unrelated features;
- precision of the top 10;
- rank agreement at feature, family and date level;
- per-plot cosine similarity with the truth;
- whether the main cause of a low index is found for the bottom 10% of plots.

## 3. Key results

T1, realistic inputs, no noise, LightGBM unless stated. Full tables are in `shap_benchmark/RESULTS.md`.

| Measure | Value |
|:--|:--|
| Model fit | R² = 0.97 |
| Attribution on true inputs | 40% (near copies 14%, proxies 41%, unrelated 5%) |
| Same, other models and explainers | random forest 34%, MLP 22%, gain importance 34%, permutation importance 39% |
| Top-ranked feature | an unused median backscatter of 14 Aug (ρ 0.95 with a true input) gets 23%; the true mean backscatter of that date (true share 23%) gets 6% |
| Rank agreement with the truth | feature ρ = 0.41; family ρ = 0.95; date ρ = 0.99 |
| ROAR, 12 features removed | truth order R² 0.935, SHAP order 0.743, random 0.969: ROAR rates SHAP as more faithful than the truth |
| ROAR, one date removed | truth = SHAP = 0.47 against random 0.89: ROAR works at date level |
| Per plot, grouped by date and family | median cosine 0.94–0.99; main cause of a low index found for 91–92% of flagged plots (trees) |
| Per plot, single features | median cosine 0.37–0.51 |
| Crop identity (T2) | true share 15%; SHAP gives 11% (LightGBM) and 2% (random forest) |

**Hypotheses:**

- **Supported:** H1, H5 (negligibly) and H7 (more strongly than predicted).
- **Partly supported:** H4.
- **Not supported:** H2, H3, H6 and H8.

Every verdict is reported in `RESULTS.md`.

**Date shares are definition-dependent.** Figure 8 sums each feature's mean absolute attribution by date: 14 Aug gets 60% from LightGBM against a true 48%. If attributions are first summed within each date for every plot, LightGBM gives 54% against a true 51%, and the random forest still gives 74%. `RESULTS.md` reports both.

## 4. Reproducing the benchmark

Set `AISE_DATA` to the Phase 3 download (or place it inside `../phase3_yield_forecast/`). Then:

```bash
cd paper/shap_benchmark
python run_benchmark.py setup      # analysis set, designs, exact truth, controls C1–C3, C5
python run_benchmark.py tune       # hyperparameter search -> results/tuning_*.csv
python run_benchmark.py grid all   # 105 runs -> results/runs.csv, results/attr/   (about 12 CPU-hours)
python run_benchmark.py roar       # remove-and-retrain  -> results/roar.csv
python analyze.py                  # scores, hypotheses  -> results/metrics.csv, summary.csv, ...
python make_figs.py                # Figures 6-10
```

`NJOBS` sets the number of parallel jobs (default 6). The first local attempt with 14 jobs ran out of memory, so the grid and ROAR were run on Kaggle CPU in one environment: Python 3.13, numpy 2.1.3, scikit-learn 1.6.1, shap 0.53.0, LightGBM 4.5.0 (`results/versions_kaggle.json`). The kernels are in `kaggle/`. To re-run them under your own account:

1. Create a Kaggle dataset from `run_benchmark.py` and `results/{meta.json, X.csv, truth.npz, tuning_best.csv}`.
2. In each `kaggle/k_*/kernel-metadata.json`, change `id` and `dataset_sources` to your username and dataset.
3. Push the kernels, download their `results/` outputs into `results/`, and run `analyze.py`.

All grid and ROAR outputs are shipped, so `analyze.py` and `make_figs.py` can be run directly.

## 5. Figures

| Figure | Shows | Script | Output |
|:--|:--|:--|:--|
| 1 | date-removal test of the index, with the 29 Oct known-answer control | `figures/xai/make_xai_figs.py` | `figures/xai/fig1_date_removal.png` |
| 2 | exact per-plot attribution of the index for one flagged and one healthy plot (Farm A, Farm B) | same | `fig2_farm_attribution.png` |
| 3 | spatial validity: location (variogram, variance split) and outline (boundary pixels) | same | `fig3_spatial_validity.png` |
| 4 | re-flight (speckle) stability of the attention list | same | `fig4_speckle_stability.png` |
| 5 | village uncertainty terms and conformal coverage arithmetic | same | `fig5_uncertainty.png` |
| 6 | where SHAP credit goes (true inputs, copies, proxies, unrelated) and agreement by level | `shap_benchmark/make_figs.py` | `shap_benchmark/figures/fig6_attribution_leak.png` |
| 7 | credit to unused features vs their correlation with a true input; true vs SHAP credit per input | same | `fig7_leak_mechanism.png` |
| 8 | credit per acquisition date: truth vs each model | same | `fig8_dates.png` |
| 9 | remove-and-retrain at feature level and at date level | same | `fig9_roar.png` |
| 10 | per-plot agreement and main-cause identification | same | `fig10_local.png` |
| D1 | the 966 plots by crop, and how many radar images cover each plot | `figures/dataset/make_dataset_figs.py` | `figures/dataset/d1_study_area.png` |
| D2 | rainfall, soil moisture and the six image dates; typical crop calendar | same | `d2_season_timeline.png` |
| D3 | each crop's radar brightness through the season | same | `d3_crop_trajectories.png` |
| D4 | plot sizes; plots measured on each date | same | `d4_plot_size_coverage.png` |
| D5 | correlation between the 15 measurements of one image (14 Aug) | same | `d5_feature_correlation.png` |

```bash
python paper/figures/xai/make_xai_figs.py
python paper/figures/dataset/make_dataset_figs.py
python paper/shap_benchmark/make_figs.py
```

**Figure style:**

- **Palette:** Tailwind 200 fills with a thin grey edge, and one fixed colour per meaning in every figure. Emerald is the known answer, blue LightGBM, violet random forest, amber MLP and grey the random baseline. Crops are sky (cotton), orange (groundnut), teal (rice), rose (bajra) and amber (maize).
- **Automatic checks:** every script checks each figure after drawing, for no hyphen or dash characters in visible text and no overlapping text, and stops if either fails.
- **Resolution:** figures are saved at 300 dpi.

## 6. Verification

Run on 2026-10-24 in a fresh copy of this repository:

| Step | Result |
|:--|:--|
| `run_benchmark.py setup` | `X.csv` and `truth.npz` byte-identical. `meta.json` differs only in control C3, which uses sampling-based permutation SHAP: null mass 1.066 × 10⁻⁸ against 1.061 × 10⁻⁸, both 1.1 × 10⁻⁸ as reported |
| `analyze.py` | `summary.csv`, `metrics.csv`, `hypotheses.json` and `leak_by_feature.csv` byte-identical |
| `make_figs.py` | Figures 6–10 byte-identical |
| `make_xai_figs.py`, `make_dataset_figs.py` | Figures 1–5 and D1–D5 byte-identical |
