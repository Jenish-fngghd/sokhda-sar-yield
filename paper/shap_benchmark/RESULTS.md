# Evaluating Feature Attribution for X-Band SAR Data: Results from a Known-Answer Benchmark

Design, hypotheses and thresholds: `PLAN.md` (frozen before any model was trained). Every prediction is reported below, including those that failed. Numbers are means over 5 seeds × 5-fold out-of-fold explanations unless stated.

## Main findings

1. **Accurate models, misattributed features.** With realistic inputs (91 features: 90 single-date SAR measurements plus parcel area, 81 of them unused by the index; 911 farms), LightGBM reproduces the index almost perfectly (R² 0.97). Yet only **40%** of its SHAP attribution lands on features the index actually uses:
   - 41% goes to **proxies** of those features (0.5 ≤ ρ < 0.98);
   - 14% goes to **near copies** (ρ ≥ 0.98);
   - 5% goes to **unrelated** features.

   The same holds for the random forest (34% correct) and the MLP (22%), for gain importance (34%) and for permutation importance (39%).
2. **SHAP names the wrong top feature.** The highest SHAP feature (23% of all attribution) is the *median* backscatter of 14 Aug, a proxy (ρ 0.95) the index never uses. The true top input, *mean* backscatter of 14 Aug (23% of true attribution), receives 6%.
   - Of the 10 features SHAP ranks highest, only **4** are inputs of the index.
   - Feature-level rank agreement with the truth is ρ = 0.41.
3. **SHAP is exact; the interpretation is not.** The positive control (C3) shows SHAP computed on the *true* function puts 1e-8 of its mass on unused features. The error is not in SHAP's arithmetic. An accurate model learns to rely on correlated proxies, and SHAP faithfully reports the model, not the process that generated the data. Reading physical drivers off feature-level SHAP plots, as the compared papers do, is therefore unsafe whenever inputs are correlated, which is always the case for multi-statistic, multi-date SAR stacks.
4. **Grouping restores rankings, not magnitudes.**
   - **Rankings recover:** when each farm's attributions are summed within an acquisition date, the date ranking matches the truth (ρ 0.99, LightGBM and random forest). The date the index never uses (29 Oct) receives 0.6 to 1.2% across both targets.
   - **Magnitudes depend on the model and on how the shares are formed.** Two definitions are reported, because they give different numbers:
     - *Per-farm grouped sums* (attributions summed within a date for each farm, then averaged in absolute value; the definition behind the ρ above): 14 Aug receives **54%** from LightGBM against a true **51%**, **74%** from the random forest, and the MLP gives the unused 29 Oct **6%**.
     - *Summed feature shares* (each feature's mean absolute attribution, summed by date; the definition plotted in Figure 8): LightGBM gives 14 Aug **60%** against a true **48%** and under-credits 19 Jun (10% vs 20%) and 13 Oct (12% vs 20%); the random forest gives 14 Aug **77%**; the MLP gives the unused 29 Oct **11%**.
   - Which "date matters most" survives under both definitions and every model. *How much* it matters does not: the random forest over-credits 14 Aug under both, while LightGBM is close only when attributions are grouped per farm before taking absolute values.
5. **ROAR cannot detect the problem, and inverts it.** Lee et al. use ROAR as their faithfulness test. Here it rates SHAP as *more* faithful than the known answer:
   - removing the 12 truly used features leaves R² at **0.935** (copies stand in);
   - removing SHAP's top 12 drops it to **0.743**.

   At date level, ROAR works: removing the truly most important date drops R² to 0.47 in both orders, against 0.89 for a random date.
6. **Per farm, grouped explanations are good.**
   - **Cosine with the truth:** at date × family level, the median per-farm cosine is 0.94 to 0.99 for tree models.
   - **"Why flagged":** for bottom-decile farms, the main cause is identified correctly in **91 to 92%** of cases.
   - **Feature level:** the same explanations score only 0.37 to 0.51.
7. **Crop identity is under-credited, not over-credited.** In the within-crop index (T2), crop carries 15% of the true attribution. SHAP gives it 11% (LightGBM) and 2% (random forest). The concern in Kushwaha et al. (SHAP dominated by crop) does not transfer; the opposite error occurs.

## Hypotheses (pre-registered in PLAN.md section 9)

| | Prediction | Result | Verdict |
|:--|:--|:--|:--|
| H1 | LightGBM R² ≥ 0.90 (no noise), D1 and D2 | 0.974 (D1), 0.959 (D2) | **Supported** |
| H2 | Exact inputs: feature-level ρ ≥ 0.90 | LightGBM 0.89 (T1), 0.91 (T2); random forest 0.50, 0.38 | **Not supported**. Even with only true inputs, near-duplicates *among* them (dB vs linear backscatter of the same date, ρ ≈ 1) scramble feature-level credit. |
| H3 | Realistic inputs: null mass ≥ 20%, mostly to copies | null mass 52 to 66%; copies are only 16 to 27% of it, proxies the majority | **Not supported as stated**. Leakage is about three times the predicted 20% floor, and goes mainly to proxies, not copies. |
| H4 | Grouped: ρ ≥ 0.90, 29 Oct < 5% | date ρ 0.99 (T1), 0.96 / 0.89 (T2 LightGBM / RF); family ρ 0.95 (T1), 0.97 / 0.72 (T2); 29 Oct 0.6 to 1.2% | **Partly supported**: holds for T1 and for LightGBM on T2; fails for the random forest on T2. |
| H5 | Path-dependent SHAP leaks more than interventional | 58.6% vs 58.0%; more in 75% of runs | **Supported, but negligible** (0.6 percentage points). |
| H6 | Model fit does not predict attribution accuracy | across 20 configurations, ρ(R², per-farm cosine) = 0.91, ρ(R², feature ρ) = 0.78 | **Not supported**. Better-fitting models do give better explanations; but even the best fit (R² 0.97) leaves 60% of attribution on unused features. |
| H7 | Feature-level ROAR cannot separate truth from random; date-level can | feature level: truth-guided ≈ random (0.935 vs 0.969 at 12 removed) and SHAP-guided lower than both; date level: 0.47 vs 0.89 | **Supported**, and stronger than predicted: feature-level ROAR inverts the ranking of truth and SHAP. |
| H8 | Noise lowers local agreement more than global | local −0.027, date ranking −0.063 | **Not supported**. Global date ranking degrades more. |

## Controls

| | Control | Result | Pass |
|:--|:--|:--|:--|
| C1 | Truth efficiency, Σφ = f(x) − E f | max error 1.3e-14 (T1), 1.5e-13 (T2) | Yes |
| C2 | Truth vs shap's Exact explainer | max abs 9.2e-9 (T1), 3.8e-7 (T2); relative 6.4e-9, 8.4e-9 | Yes on relative error; see deviation 1 |
| C3 | SHAP on the true function, realistic inputs | null mass 1.1e-8; date ρ 0.991 | Null mass yes; date ρ see deviation 2 |
| C4 | LightGBM on a permuted target | R² −0.011 | Yes |
| C5 | f reproduces the shipped values | score max error 2.2e-16; index max error 0.005 (rounding) | Yes |

## Deviations from the plan (all decided before seeing attribution results)

1. **C2 threshold.** The plan set an absolute tolerance of 1e-8. T2 is on a 0 to 100 scale, and the difference scales with the target: relative error is 6e-9 to 8e-9 for both targets. That is floating-point accumulation, not a definitional difference, so C2 is judged on relative error (< 1e-7).
2. **C3 date ρ.** The plan said ρ = 1. The measured 0.991 comes from two groups with true attribution exactly 0 (29 Oct, static) that SHAP fills with ~1e-9 noise, so their order is arbitrary. Null mass, the substantive part of the control, passes at 1e-8.
3. **LightGBM grid.** The planned grid's best value sat at its edge (num_leaves 7). num_leaves 3 and 4 were added before any explanation was computed; the optimum is now interior (4) in most configurations.
4. **Compute.** The first local run was stopped by the operating system for low memory after 18 of 105 runs. Everything was then re-run from scratch on Kaggle CPU in one environment: Python 3.13, numpy 2.1.3, scikit-learn 1.6.1, shap 0.53.0, LightGBM 4.5.0. Truth and tuning were computed locally (Python 3.12) before that and uploaded unchanged. Seven downloaded result files arrived as HTML error pages; they were detected by a file-signature check and re-downloaded before analysis.

## Caveats

- **One MLP fit failed.** Realistic inputs, no noise, seed 1: R² −1.57, from wild extrapolation on 2 farms (predictions −21 and −16 against a target spread of 0.55). Excluding the worst 0.5% of farms (5 farms) gives R² 0.88. Its attribution metrics match the other four seeds (null mass 0.79 vs 0.76 to 0.79), so it is kept in all results; MLP R² is reported as a median (0.83).
- **The target is our index, not yield.** The benchmark measures whether explanations recover a known function of real SAR data. It says nothing about yield accuracy.
- **One village, one season, one function.** The leakage magnitude depends on the correlation structure of these 91 features. The mechanism (accurate models relying on correlated proxies) is general; the percentages are specific to this dataset.
- **Imputation.** 1,365 missing values in unused features were median-imputed with no target information. The 10 inputs of the index are complete in all 911 analysed farms (55 farms with any missing index input were excluded).

## What this means for the paper

- **Claim we can make:** in a real X-band SAR feature set with a known answer, SHAP explanations of accurate models attribute most credit to features the target never uses, name a non-causal feature as the top driver, and pass ROAR while doing so. Explanations become reliable only when attributions are aggregated to physically meaningful groups (dates, measurement families). Even then, magnitudes are model-dependent.
- **Why only we can make it:** none of the 10 compared XAI+SAR papers knows its true attributions. Our transparent index provides one.
- **How it supports our design:** it justifies an interpretable-by-design index over post-hoc explanation, and our group-level (date) validation in Figure 1 (date removal).

### Paper-ready discussion paragraph

Feature attribution was evaluated against a known answer by treating the shipped crop health index as the target function, so that its exact Shapley values were available. With 91 realistic inputs, of which 81 are unused by the index, LightGBM reproduced the index almost perfectly (R² = 0.97), yet only 40% of its SHAP attribution fell on the true inputs; 41% went to correlated proxies, 14% to near copies and 5% to unrelated features. The highest-ranked feature was an unused median backscatter of 14 August (|ρ| = 0.95 with a true input), which received 23% of the attribution, while the true mean backscatter of the same date, carrying 23% of the known attribution, received 6%. The same pattern held for the random forest, the MLP, gain importance and permutation importance, and a positive control confirmed that SHAP itself is exact: the error arises because accurate models rely on correlated substitutes, and SHAP faithfully reports the model rather than the process that generated the data. Remove-and-retrain (ROAR), the faithfulness test used by Lee et al. (2026), did not detect this and instead ranked SHAP above the known answer (R² = 0.743 versus 0.935 after removing twelve features), because correlated copies stand in for removed inputs. Aggregation to physically meaningful groups restored much of the agreement: date-level rankings matched the truth (ρ = 0.99), date-level ROAR separated informed from random removal (R² = 0.47 versus 0.89), and per-farm grouped explanations identified the main cause of a low index in 91–92% of flagged farms. Date-level magnitudes, however, remained model-dependent. Feature-level SHAP plots on multi-date SAR stacks should therefore not be read as physical drivers, and explanations should be reported and tested at the level of acquisition dates and measurement families.

## Verification (2026-10-21)

Every number in this file was recomputed from `results/summary.csv`, `metrics.csv`, `leak_by_feature.csv`, `roar.csv`, `hypotheses.json`, `meta.json`, `versions_kaggle.json` and the stored out-of-fold predictions in `attr/`. All values match, with these corrections made at verification:

- **Date shares.** The earlier text mixed two definitions: the ranking ρ used per-farm grouped sums, while the magnitudes (60%, 77%, 11%) came from summed feature shares, the definition in Figure 8. Both are now stated, with the per-farm grouped values (54% vs 51%, 74%, 6%) added. The finding "rankings recover, magnitudes do not" holds under both definitions for the random forest and the MLP; for LightGBM it holds only for summed feature shares.
- **"Halves 19 Jun and 13 Oct"** became "under-credits", since 13 Oct is 12% vs 20%, not half.
- **Feature count** now states 90 SAR measurements plus parcel area, and the 911 analysed farms.
- **H3** now says "about three times the predicted 20% floor" (observed 52 to 66%).

Checked and unchanged: ROAR values (feature level 0.935 / 0.743 / 0.969 at 12 removed; date level 0.466 for both informed orders vs 0.890 mean of random first removals); top-feature shares (23.4% vs 6.4% for the true input, 6.1% path dependent); precision at 10 of 0.40; random-forest feature ρ 0.395; crop share in T2 (true 14.9%, LightGBM 11.0%, random forest 2.0%); failed MLP seed (R² −1.571, two predictions −21.0 and −16.3, target sd 0.554, 0.879 after removing the 5 worst farms, null mass 0.786 vs 0.762 to 0.793); environment (Python 3.13, numpy 2.1.3, scikit-learn 1.6.1, shap 0.53.0, LightGBM 4.5.0).

## Figures (`figures/`)

| File | Shows |
|:--|:--|
| `fig6_attribution_leak.png` | (a) Where attribution goes, by stratum, for every model and explainer. (b) Agreement at feature, family and date level. |
| `fig7_leak_mechanism.png` | (a) Leaked share vs correlation with a true input. (b) True vs SHAP credit for each true input. |
| `fig8_dates.png` | Attribution share per acquisition date: truth vs each model. |
| `fig9_roar.png` | Remove-and-retrain at feature level (a) and date level (b). |
| `fig10_local.png` | Per-farm agreement (a) and "why flagged" agreement (b) across inputs and noise. |

All figures pass the automated no-dash and no-overlap checks and were inspected by eye.

## Proposed paper captions

**Figure 6. Attribution allocation and agreement with the known answer.** (a) Attribution assigned to true inputs and unused features. (b) Agreement with the known attribution at feature, family, and date levels. Results use realistic inputs; error bars show one standard deviation across five seeds.

**Figure 7. Attribution assigned to unused features.** (a) LightGBM SHAP attribution versus each unused feature's strongest correlation with an index input. (b) Known and SHAP attribution for the inputs used by the index. Results use realistic inputs with no target noise.

**Figure 8. Attribution by acquisition date.** Known and model-based attribution shares are grouped by date. The 29 October acquisition is not used by the index; any attribution assigned to it reflects the model's explanation, not a direct input to the target.

**Figure 9. Remove-and-retrain evaluation.** Cross-validated LightGBM R² after removing features (a) or acquisition dates (b) in known, SHAP, or random order. Shading shows the range across random removal orders.

**Figure 10. Per-farm attribution agreement.** (a) Agreement with known attributions after grouping by date and feature family. (b) Share of the lowest-scoring farms for which the main attributed cause matches the known answer. Results compare exact and realistic inputs, with and without target noise.

## Figure-by-figure discussion

**Scope of Figures 6–10.** These figures evaluate explanations of the known health-index function, not crop yield or yield-forecast accuracy. The target has an exact attribution answer because it is an explicit function of a subset of the measured SAR features. The benchmark tests whether models and explainers recover that answer when they are given either only the index inputs or a broader, realistic set containing correlated and irrelevant measurements.

### Figure 6 — Attribution leakage and recovery by grouping

Panel (a) shows where each method assigns its total attribution under the realistic-input design. The known answer assigns all attribution to the index inputs. In contrast, LightGBM SHAP assigns 40% to those inputs, 41% to correlated proxies, 14% to near-copies, and 5% to unrelated features. Random-forest SHAP assigns 34% to the true inputs, while MLP SHAP assigns 22%. Thus, good predictive fit does not ensure that attribution stays with the variables used to construct the target. Panel (b) shows how agreement depends on the level of comparison: feature-level rankings are weak (Spearman ρ = 0.41 for LightGBM and 0.39 for random forest), whereas family- and date-level rankings are much closer to the known answer for the tree models (date-level ρ ≈ 0.99). Aggregating correlated measurements into meaningful groups therefore improves ranking agreement, although it does not establish that individual feature shares are correct.

### Figure 7 — Correlation provides a route for attribution leakage

Panel (a) relates the attribution assigned to each unused feature to its strongest absolute correlation with any index input. The points do not show a simple one-to-one relationship: most unused features receive little attribution, including some that are highly correlated with an index input. However, the largest attribution to an unused feature goes to a backscatter proxy correlated with a true input (|ρ| ≈ 0.95), illustrating how a correlated substitute can receive substantial credit. Correlation alone does not determine attribution, but the known target function confirms that these high-attribution variables are not direct inputs to that function. Panel (b) compares the known and LightGBM SHAP shares for the actual index inputs. The clearest mismatch is 14 August backscatter in dB: it receives about 23% of the known attribution but about 6% from either interventional or path-dependent SHAP. In contrast, 14 August texture CV receives attribution close to its known share. The error is therefore uneven across inputs: SHAP can recover some drivers while substantially under-crediting others and assigning the displaced credit to correlated, unused features.

### Figure 8 — Date rankings are stronger than date attribution magnitudes

Figure 8 forms each date's share by summing the mean absolute attribution of every feature from that date. On this definition the known answer is concentrated on 14 August (about 48%), with 19 June and 13 October each contributing about 20%, 12 November about 11%, and little contribution from 6 June; 29 October is unused by the index. All explainers identify 14 August as the largest date-level contributor, but their estimated shares differ substantially. LightGBM assigns roughly 60% to 14 August, while random forest assigns about 77%; both under-credit 19 June and 13 October. The MLP distributes more attribution across dates and assigns around 11% to the unused 29 October acquisition. These results indicate that the broad date ordering is more stable than the estimated magnitude of each date's contribution, and that an unused date can still receive credit when its measurements correlate with dates that the index uses. The magnitudes also depend on how shares are formed: when attributions are first summed within each date for every farm, LightGBM gives 14 August 54% against a true 51%, the random forest still gives 74%, and the MLP gives 29 October 6%.

### Figure 9 — ROAR gives different conclusions at feature and date levels

Figure 9 applies remove-and-retrain (ROAR) to LightGBM with realistic inputs and no target noise. In panel (a), removing features in SHAP-ranked order causes model fit to fall more quickly than removing features in the known-answer order. After the 12 truly used features are removed, the model retains R² = 0.935 because correlated alternatives can substitute for them. Removing SHAP's top 12 features lowers R² to 0.743, while random removal leaves R² near 0.97. Since the SHAP-ranked removal produces a larger performance drop, feature-level ROAR would make SHAP appear more faithful—even though Figure 6 shows that much SHAP credit went to unused features. Panel (b) gives a different result when whole acquisition dates are removed: truth- and SHAP-guided removal follow nearly the same curve, and removing the leading date reduces R² to about 0.47, compared with about 0.89 for a random date. ROAR is therefore informative at the date grouping in this experiment but can mislead at the individual-feature level when correlated substitutes are present.

### Figure 10 — Local explanations recover grouped patterns, with model and noise effects

Panel (a) measures per-farm cosine agreement between the explanation and the known answer after contributions are grouped by acquisition date and measurement family. With realistic inputs and no target noise, the tree models retain high median agreement (approximately 0.94–0.99); agreement decreases when noise is added to the target, especially for MLP under realistic inputs. Panel (b) asks a targeted question about the lowest-scoring 10% of farms: does the explanation identify the same most-negative date-by-family group as the known answer? Under realistic inputs without noise, LightGBM and random forest identify that main cause in about 91–92% of cases. The MLP is less stable under realistic inputs, particularly with target noise. These local results show that grouped explanations can preserve farm-level diagnostic information in this controlled benchmark, while the variability across models and noise conditions argues against treating any one local attribution as definitive.

## Reproduce

```
.venv/Scripts/python.exe run_benchmark.py setup     # truth + controls (needs frozen R3 tables)
.venv/Scripts/python.exe run_benchmark.py tune
.venv/Scripts/python.exe run_benchmark.py grid all  # or: grid rf / grid other (Kaggle shards in kaggle/)
.venv/Scripts/python.exe run_benchmark.py roar
.venv/Scripts/python.exe analyze.py
.venv/Scripts/python.exe make_figs.py
```

Outputs: `results/metrics.csv` (per run × explainer), `summary.csv`, `leak_by_feature.csv`, `hypotheses.json`, `roar.csv`, `runs.csv`, `attr/*.npz` (all out-of-fold attributions).
