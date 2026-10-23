# Known-Answer SHAP Benchmark on Real X-band SAR: Plan

Written and frozen **before** any model was trained. Hypotheses, metrics, controls and pass thresholds below are fixed in advance; every outcome is reported, including the ones that go against us.

## 1. Question

When SHAP explains a model trained on real SAR features, does it recover the true drivers?

None of the 10 compared XAI+SAR papers can answer this, because none knows its true attributions. We can. Our health index is an explicit function of a few SAR measurements, so its exact Shapley values can be computed. A black-box model trained to reproduce the index can then be explained with SHAP and the result scored against that exact answer.

The papers this addresses:
- **Gao 2026:** SHAP may be unreliable under multicollinearity.
- **Abbasi 2026, Tribuana 2026, Li 2026, Lin 2026:** SHAP judged on plausibility only.
- **Tsuchiya 2025:** attention used as date importance, untested.
- **Lee 2026:** ROAR as the faithfulness test.
- **Kushwaha 2026:** SHAP dominated by crop identity.

**Scope boundary.** The target is our index, not yield. The benchmark says nothing about yield accuracy; it measures only whether explanations recover a known truth.

## 2. Data

- **Source:** frozen R3 table `p1_farm_features.csv` (966 farms, six Capella X-band dates: 06 Jun, 19 Jun, 14 Aug, 13 Oct, 29 Oct, 12 Nov). Nothing in R3 is modified.
- **Analysis set:** farms with all 10 causal features finite (§3), since the exact truth needs them. Null features with missing values are median-imputed once, with no target information used.
- **Crop:** the shipped R2/R3 crop label, used only for target T2.

## 3. Targets (the known answers)

The shipped code (`p3_build.health_parts`, `derive_weights`, `health_index`) is re-expressed as a closed-form function f with every data-dependent constant **frozen** from the shipped 966-farm run: the robust-z medians and MADs, the village growth median, the weights, and the per-crop medians and MADs. Control C5 proves f reproduces the shipped values.

| Target | Definition | Inputs f actually uses (causal set) |
|:--|:--|:--|
| **T1** | Composite score S = Σ w_k · part_k | 10 features: `g0_db` (14 Aug, 19 Jun, 13 Oct, 12 Nov), `g0_lin` (06 Jun, 19 Jun, 14 Aug, 13 Oct, 12 Nov), `cv` (14 Aug) |
| **T2** | Shipped health index, 100·Φ(z_crop(S)/2), scored within crop | the same 10 + crop (11 players) |

- **Noise levels:** σ ∈ {0, 0.5} × sd(f), Gaussian, added to the training target only. σ = 0.5 caps R² at 0.80, mimicking a noisy real label. The truth is always the Shapley value of the noiseless f.
- **Ground truth:** exact interventional Shapley values of f, computed by full subset enumeration (2^10 or 2^11 coalitions) against a fixed background of 100 farms (seed 0). This is the same value function interventional TreeSHAP estimates, so explainer and truth are defined identically. Any feature f does not use has a true attribution of exactly 0.

## 4. Input designs

| Design | Features | What it represents |
|:--|:--|:--|
| **D1, exact inputs** | the 10 causal features (+ crop for T2) | best case: the model sees only what matters |
| **D2, realistic inputs** | every single-date raw measurement for all six dates (16 measurement types per date) plus parcel area: about 90 features, about 80 of which are nulls | what the compared papers do: feed all bands, statistics and dates |

- **Excluded from D2:** engineered multi-date features (`d_*`, `ref_*`, `season_integral`, `temporal_*`, `senescence_db_per_day`, `jun_baseline_db`), because their "true" attribution depends on how the function is parameterised and is not identifiable. QC and ID columns are excluded too.
- **Null features in D2 (deliberate):**
  - near-exact copies of causal features (`b0_db`, `nesz_margin_db`, `g0_db_med`; `ktex` vs `cv`);
  - all of 29 Oct, which is correlated with the adjacent dates but unused by f;
  - incidence angle, noise-floor fractions, coverage, pixel count, area.

**Evaluation levels**, each a stricter test of identifiability:
- **Feature level:** strict. Not identifiable for exact copies (ρ ≈ 1): no method can split credit between duplicates from data alone. Reported, but stratified by correlation to the causal set.
- **Date level:** 6 dates + static. Identifiable. This is the level at which the papers make claims ("late-season dates matter most").
- **Family level:** backscatter level / texture / geometry / quality and size. Identifiable: duplicates fall in the same family.
- **Date × family level:** the finest identifiable grouping. Used for local (per-farm) metrics.

Group attributions are sums of member attributions, computed identically for truth and explainers.

## 5. Models

The model families the compared papers use. Hyperparameters are tuned once per configuration by 5-fold CV on seed 0, then fixed.

| Model | Used by | Tuning grid |
|:--|:--|:--|
| LightGBM | Tribuana, Li, Rahman, Kushwaha (boosting) | num_leaves {7, 15, 31} × min_child_samples {5, 20}; lr 0.03, early stopping on a 15% inner split, row/col subsampling 0.8 |
| Random forest | Lee, Gao | min_samples_leaf {1, 5} × max_features {0.33, 1.0}; 500 trees |
| MLP | Abbasi (ANN) | alpha {1e-4, 1e-2}; (128, 64), standardised inputs and target, early stopping. **T1 only.** |

**Protocol:** 5-fold CV × 5 seeds. Every farm is explained by a model that never saw it (out-of-fold).

## 6. Explainers

| Explainer | Applied to | Why |
|:--|:--|:--|
| TreeSHAP, **interventional** (background = the same 100 farms) | LightGBM, RF | same definition as the truth: a fair test |
| TreeSHAP, **path-dependent** (shap's default) | LightGBM, RF | what the papers actually run |
| Exact SHAP (D1) / Permutation SHAP (D2, max_evals = 4(2p+1)) | MLP | model-agnostic SHAP |
| Permutation importance (10 repeats, OOF) | all | the global alternative Lee compares against |
| LightGBM gain importance | LightGBM | the default "feature importance" plot |

## 7. Metrics

1. **Model fidelity:** OOF R² against the training target and against the noiseless f.
2. **Null mass:** share of total |attribution| assigned to features whose true attribution is 0. Stratified by each null feature's max |Spearman ρ| with the causal set: copies (≥ 0.98), proxies (0.5 to 0.98), unrelated (< 0.5).
3. **Precision@10:** of the explainer's top 10 features (10 = size of the causal set), the share that are causal.
4. **Global rank agreement:** Spearman ρ between explainer and truth mean |attribution|, at feature, date and family level.
5. **29 Oct mass:** share of |attribution| on a date f never uses.
6. **Local agreement:** per-farm cosine similarity between explainer and truth at date × family level; median and IQR.
7. **"Why flagged" agreement:** for bottom-decile farms (by noiseless f), how often the explainer's most negative date × family group matches the truth's.

**ROAR** (LightGBM, D2, T1, σ = 0, 3 seeds): remove features in order of (a) the truth ranking, (b) global interventional SHAP, (c) random (5 orders), then retrain. Report OOF R² vs number removed. Repeated at date-group level (remove whole dates).

## 8. Controls (the run stops if a hard control fails)

| | Control | Pass condition |
|:--|:--|:--|
| C1 | Truth efficiency: Σ_j φ_j = f(x) − mean_bg f | max error < 1e-9 |
| C2 | Truth vs an independent implementation: `shap.explainers.Exact` on f itself (D1) | max \|difference\| < 1e-8 |
| C3 | Positive control: SHAP on the *true* function inside the D2 feature space (Permutation explainer) | null mass < 1e-6; date-level ρ with truth = 1 |
| C4 | Negative control: LightGBM on a permuted target (D2, T1) | OOF R² ≤ 0.05 (reported) |
| C5 | f reproduces the shipped score and index on the analysis set | max error < 1e-9 (S), < 0.011 (rounded index) |

## 9. Pre-registered predictions

These are predictions, not claims. Each is marked supported or not in RESULTS.md.

- **H1:** LightGBM fits well: OOF R² vs noiseless f ≥ 0.90 at σ = 0, in both D1 and D2.
- **H2:** In D1, interventional TreeSHAP recovers the truth: feature-level global ρ ≥ 0.90.
- **H3:** In D2, feature-level attribution leaks: null mass ≥ 20%, mostly to copies.
- **H4:** At date and family level, agreement recovers: ρ ≥ 0.90, 29 Oct mass < 5%.
- **H5:** Path-dependent TreeSHAP leaks more to nulls than interventional.
- **H6:** Across models, fit (R²) does not predict attribution accuracy.
- **H7:** Feature-level ROAR cannot separate truth-guided from random removal in D2, because copies stand in for removed features. Date-level ROAR can.
- **H8:** Noise (σ = 0.5) lowers local agreement more than global agreement.

## 10. Outputs

- `run_benchmark.py`: truth, controls, the full grid (parallel), and ROAR. Writes `results/*.csv` and `results/attr/*.npz`.
- `make_figs.py`: paper figures, with the same no-dash and no-overlap checks as figures 1–5.
- `RESULTS.md`: every metric, every hypothesis verdict, every control.
- **Environment:** isolated venv (`.venv`, system packages plus shap 0.53.0 and lightgbm 4.5.0) so the R3 environment is untouched.
