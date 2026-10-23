# Paper Experiments and Figures

This folder contains the work carried out for the paper rather than for the competition:

- **a known-answer benchmark** that tests whether SHAP and related explanations recover the true drivers of a model trained on real X-band SAR features;
- **the scripts for every paper figure.**

All scripts read the Phase 3 outputs and write only inside this folder.

| | |
|:--|:--|
| **Paper sections** | Dataset figures; testing the explanations; results and discussion of the benchmark |
| **Main result** | LightGBM reproduces the index with R² = 0.97, but only 40% of its SHAP attribution falls on the true inputs, and feature-level remove-and-retrain ranks SHAP above the true attribution |
| **Requirements** | Phase 3 competition download (`AISE_DATA`), because the index code requires the scene files on import |

---

## Contents

1. [Benchmark Design](#1-benchmark-design)
2. [Key Results](#2-key-results)
3. [Repository Contents](#3-repository-contents)
4. [Usage](#4-usage)
5. [Figures](#5-figures)

---

## 1. Benchmark Design

**Motivation.** Explanations of SAR crop models are usually judged by agronomic plausibility or by remove-and-retrain (ROAR). Neither establishes whether the highlighted features are the true drivers. Here the truth is known: the shipped health index is an explicit function of SAR features, so its exact Shapley values can be computed by full enumeration. The design was fixed in `shap_benchmark/PLAN.md` before any model was trained.

| Element | Specification |
|:--|:--|
| Target T1 | Composite score S of the Phase 3 index; 10 causal features (γ⁰ in dB and linear units on several dates; within-plot CV on 14 August) |
| Target T2 | The index itself, 100 · Φ(z_crop(S) / 2), with crop as an eleventh input |
| Inputs D1 (exact) | The causal features only |
| Inputs D2 (realistic) | Every single-date measurement on all six dates (15 types per date) plus plot area: 91 features, 81 unused by the index |
| Noise | None, or Gaussian noise of 0.5 × sd added to the training target |
| Sample | 911 plots with complete index inputs; 1,365 missing values in unused features median-imputed |
| Models | LightGBM, random forest, MLP (T1 only), tuned per configuration |
| Explanations | Interventional and path-dependent TreeSHAP; exact (D1) or permutation (D2) SHAP for the MLP; gain and permutation importance |
| Protocol | 5 seeds × 5-fold out-of-fold explanations; 105 model runs |
| ROAR | LightGBM, D2, T1, no noise, 3 seeds; features or whole dates removed in true, SHAP and 5 random orders |
| Controls | C1 efficiency of the exact values; C2 agreement with shap's Exact explainer; C3 no attribution to unused features when explaining the true function; C4 no fit on a permuted target; C5 reproduction of the shipped index |

**Scoring:**

- attribution share on true inputs, near copies (|ρ| ≥ 0.98), proxies (0.5 ≤ |ρ| < 0.98) and unrelated features;
- top-10 precision;
- rank agreement at feature, family and date level;
- per-plot cosine similarity;
- identification of the main cause of a low index among the lowest 10% of plots.

## 2. Key Results

T1, realistic inputs, no noise, LightGBM unless stated. Full results are in `shap_benchmark/RESULTS.md`.

| Measure | Value |
|:--|:--|
| Model fit | R² = 0.97 |
| Attribution on true inputs | 40% (near copies 14%, proxies 41%, unrelated 5%) |
| Other models and explainers | Random forest 34%, MLP 22%, gain importance 34%, permutation importance 39% |
| Top-ranked feature | An unused median backscatter of 14 August (ρ 0.95 with a true input) receives 23%; the true mean backscatter of that date (true share 23%) receives 6% |
| Rank agreement | Feature ρ = 0.41; family ρ = 0.95; date ρ = 0.99 |
| ROAR, 12 features removed | True order R² 0.935; SHAP order 0.743; random 0.969 |
| ROAR, one date removed | True and SHAP order 0.47; random 0.89 |
| Per plot, grouped by date and family | Median cosine 0.94–0.99; main cause identified for 91–92% of flagged plots (tree models) |
| Per plot, single features | Median cosine 0.37–0.51 |
| Crop identity (T2) | True share 15%; SHAP 11% (LightGBM), 2% (random forest) |

**Hypotheses:**

- supported: H1, H5 (negligible effect) and H7;
- partly supported: H4;
- not supported: H2, H3, H6 and H8.

Date-level shares depend on how they are formed. Figure 8 sums each feature's mean absolute attribution by date: 14 August receives 60% from LightGBM against a true 48%. Summing attributions within each date for every plot gives 54% against a true 51%, and 74% for the random forest.

## 3. Repository Contents

```
paper/
├── README.md
├── shap_benchmark/
│   ├── PLAN.md              design, hypotheses H1–H8 and thresholds
│   ├── RESULTS.md           results, controls, deviations and caveats
│   ├── truth.py             the index and its exact Shapley values
│   ├── run_benchmark.py     stages: setup, tune, grid, roar, versions
│   ├── analyze.py           scores explanations and tests H1–H8
│   ├── make_figs.py         Figures 6–10
│   ├── kaggle/              Kaggle CPU kernels used for the grid and ROAR
│   ├── results/             analysis set, exact attributions, tuning, runs, out-of-fold
│   │                        attributions (attr/), metrics, summaries, ROAR curves, environment
│   └── figures/             Figures 6–10
└── figures/
    ├── xai/                 Figures 1–5 and make_xai_figs.py
    └── dataset/             Dataset figures D1–D5 and make_dataset_figs.py
```

## 4. Usage

Set `AISE_DATA` to the Phase 3 download, or place it inside `../phase3_yield_forecast/`.

```bash
cd paper/shap_benchmark
python run_benchmark.py setup      # analysis set, exact attributions, controls
python run_benchmark.py tune       # hyperparameter search
python run_benchmark.py grid all   # 105 model runs (about 12 hours in total)
python run_benchmark.py roar       # remove-and-retrain
python analyze.py                  # scores and hypothesis tests
python make_figs.py                # Figures 6-10
```

All outputs of `grid` and `roar` are included, so `analyze.py` and `make_figs.py` can be run directly.

**Running on Kaggle.** The grid and ROAR were run on Kaggle CPU with Python 3.13, numpy 2.1.3, scikit-learn 1.6.1, shap 0.53.0 and LightGBM 4.5.0 (`results/versions_kaggle.json`). To re-run them:

1. Upload `run_benchmark.py` and `results/{meta.json, X.csv, truth.npz, tuning_best.csv}` as a Kaggle dataset.
2. Set `id` and `dataset_sources` in each `kaggle/k_*/kernel-metadata.json` to your account and dataset.
3. Push the kernels and copy their `results/` outputs into `results/`.

`NJOBS` sets the number of parallel jobs (default 6).

## 5. Figures

| Figure | Content | Script | File |
|:--|:--|:--|:--|
| 1 | Date-removal test of the index, with the 29 October known-answer control | `figures/xai/make_xai_figs.py` | `figures/xai/fig1_date_removal.png` |
| 2 | Exact per-plot attribution for one flagged and one healthy plot (Farm A, Farm B) | same | `fig2_farm_attribution.png` |
| 3 | Spatial validity: location (variogram, variance split) and outline (boundary pixels) | same | `fig3_spatial_validity.png` |
| 4 | Stability of the attention list under simulated re-acquisition | same | `fig4_speckle_stability.png` |
| 5 | Village uncertainty terms and conformal coverage | same | `fig5_uncertainty.png` |
| 6 | Allocation of SHAP attribution and agreement by level | `shap_benchmark/make_figs.py` | `shap_benchmark/figures/fig6_attribution_leak.png` |
| 7 | Attribution to unused features vs correlation with a true input; true vs SHAP attribution per input | same | `fig7_leak_mechanism.png` |
| 8 | Attribution by acquisition date: truth vs each model | same | `fig8_dates.png` |
| 9 | Remove-and-retrain at feature and date level | same | `fig9_roar.png` |
| 10 | Per-plot agreement and main-cause identification | same | `fig10_local.png` |
| D1 | Plots by crop; radar coverage per plot | `figures/dataset/make_dataset_figs.py` | `figures/dataset/d1_study_area.png` |
| D2 | Rainfall, soil moisture and acquisition dates; crop calendar | same | `d2_season_timeline.png` |
| D3 | Radar backscatter of each crop through the season | same | `d3_crop_trajectories.png` |
| D4 | Plot sizes; plots measured per date | same | `d4_plot_size_coverage.png` |
| D5 | Correlation among the 15 measurements of one acquisition (14 August) | same | `d5_feature_correlation.png` |

```bash
python paper/figures/xai/make_xai_figs.py
python paper/figures/dataset/make_dataset_figs.py
python paper/shap_benchmark/make_figs.py
```

**Figure style:**

- **Palette:** Tailwind 200 fills with a thin grey edge, and a fixed colour per meaning across figures:
  - emerald: known answer;
  - blue: LightGBM;
  - violet: random forest;
  - amber: MLP;
  - grey: random baseline;
  - crops: sky (cotton), orange (groundnut), teal (rice), rose (bajra), amber (maize).
- **Automatic checks:** each script checks every figure for hyphen or dash characters and overlapping text, and stops if either is found.
- **Resolution:** figures are saved at 300 dpi.
