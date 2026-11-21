# Sampling Protocol Drives Cross-Study Generalization, Physics Transfer, and Prediction Reliability in Global Marine Microplastic Models

Arnab Chakraborty (230102016) and Gourab Adhikari (230102003)
Department of Information Technology

## Abstract

Machine learning models trained on global marine microplastic compilations routinely report high accuracy under random cross-validation, yet fail when applied to new ocean basins or independent field campaigns. Using 16,988 ocean-water observations from 72 studies (52 organizations) in the NOAA NCEI Marine Microplastics database, we audit what transfers across unseen studies, diagnose why standard gradient-boosted trees still lose accuracy, and test physics-constrained and left-censored fixes. Study identity alone accounts for 63.1% of target variance on the `log10(pieces/m3 + 0.001)` scale. Under random 5-fold cross-validation, tree ensembles reach `R²` 0.75 to 0.77 (`MAE = 0.569`; `0.815` and `MAE = 0.370` on the `log1p` scale used by Dalvi & Dalvi, 2026), whereas holding out whole studies drops default LightGBM to pooled `R²` 0.473 (10-repeat mean `0.456`, MAE `1.048`, `14.2%` misses above 100x). Feature ablations show that sampling protocol carries the transferable signal (`R²` 0.497 on unseen studies), while location and time alone collapse below zero (`R²` -0.097) despite scoring 0.716 on random splits. Error decomposition exposes two remaining pathologies: unconstrained trees add `+0.307` MSE of spurious within-study variance, and pinning 1-L bottle zero counts to `-3.0` creates a 6-order-of-magnitude cliff responsible for 32.2% of dataset squared error in Barrows et al. (2018) alone. Combining shallow monotonic Huber protocol trees (`log_mesh` decreasing) with a within-study Fixed-Effects (Mundlak) slope on 5-year Lagrangian drifter accumulation raises 10-repeat unseen-study `R²` from `0.456` to `0.552` (winning 10 of 10 fold assignments, against an oracle within-study median ceiling of `0.627`), cuts `>100x` misses from `14.2%` to `7.4%`, holds under organization-grouped CV (`0.459` vs `0.385`), improves surface-net-only transfer from `0.035` to `0.275`, and beats Dalvi & Dalvi (2026) in all five held-out ocean basins (`0.194` to `0.680`). Modeling non-detects via two-part decomposition and convex left-censored Tobit LightGBM reaches positive-detection `R²` `0.815` (`MAE = 0.633`, `80.9%` within 10x, vs oracle floor `0.501`) and lifts unseen-study detection `AUC` to `0.698`. Finally, study-grouped Conformalized Quantile Regression (CQR) restores 90.7% coverage where 10-run ensemble spread covers only 15.6% to 19.1%, and deferring the 10% widest intervals on the Within-Between Physics model lowers unseen-study MAE to `0.749` (close to the `0.708` within-study noise floor).

## 1. Introduction

Global compilations of marine microplastic measurements, led by the NOAA National Centers for Environmental Information (NCEI) Marine Microplastics database, pool decades of shipboard surveys collected by dozens of research groups. These surveys differ sharply in equipment, mesh size, sampled water volume, and laboratory counting protocols. Surface neuston and manta nets typically use 0.333 mm mesh and report concentrations around 0.01 to 1 pieces/m3, whereas bulk-water grab samples and Niskin bottles filtered through 0.45 to 20 µm membranes routinely report 100 to 10,000 pieces/m3 because particle abundance increases steeply as size decreases (Lindeque et al., 2020; Tokai et al., 2021).

Recent machine learning studies train gradient-boosted trees and neural representations on NOAA NCEI and regional compilations to map global ocean microplastic concentrations (Yu et al., 2023; Ali et al., 2026; Chan et al., 2026; Dalvi & Dalvi, 2026; Narayanan, 2026). Three methodological gaps limit the scientific reliability of these models. First, random cross-validation places near-duplicate stations from the same cruise in both training and test folds, rewarding models for recognizing known studies rather than predicting new ones. Second, standard tree pipelines pool incompatible detection limits onto a single distorted target and allow unrestricted coordinate-protocol interactions that overfit study identities. Third, existing uncertainty estimates rely either on ensemble seed variance without empirical coverage validation (Dalvi & Dalvi, 2026) or on regional holdouts within a single basin (Ali et al., 2026), leaving coverage on unseen field studies untested.

This report addresses five research questions:

1. How much do random cross-validation scores overestimate generalization to unseen marine microplastic studies?
2. Which features actually transfer across unseen studies and unseen ocean basins: spatial-temporal coordinates or sampling protocol?
3. Do these findings hold when unit-corrected and unverified studies are removed from the dataset?
4. Can study-calibrated prediction intervals and simple out-of-support flags provide calibrated 90% coverage and isolate large prediction errors on unseen studies?
5. Can physics-constrained trees, within-between separation with Lagrangian surface transport, and left-censored modeling fix the failure modes exposed by the audit and narrow the gap to the within-study noise floor?

## 2. Related work

### 2.1 Global and regional ML prediction of marine microplastics

Yu et al. (2023) trained Random Forest, GBDT, and XGBoost classifiers on 1,169 global ocean surface samples using 16 environmental and socio-economic predictors, reporting 0.719 accuracy and 0.914 ROC AUC under random splitting. Amora (2025) reported `R² = 0.972` on global ocean samples using random splits with constructed regional concentration features. More recent work tests geographic transfer directly. Narayanan (2026) classified ~20,000 NOAA NCEI records into five concentration tiers using latitude, longitude, ocean, month, and 256-dimensional SatCLIP satellite geo-embeddings under leave-one-ocean-out splits, finding inconsistent transfer across basins. Hafiz & Ahmadisharaf (2026) linked MODIS satellite reflectance to NOAA NCEI labels and found that models trained in the Bay of Bengal and East China Sea failed (`R²` -0.04 to -0.35) when applied to the Mediterranean Sea.

The closest study to our work is Dalvi & Dalvi (2026), who combined self-supervised contrastive learning with XGBoost on NOAA NCEI data plus Copernicus oceanographic and GEBCO bathymetric predictors. Under random 5-fold cross-validation on `log1p(conc) = ln(conc + 1)` (verifiable from their Table 11 hotspot back-transformations, where `exp(9.98253) - 1 = 21,644.0`), they reported `R² = 0.651`, `RMSE = 1.008`, and `MAE = 0.619`. Under leave-one-ocean-out transfer, their `R²` dropped to `0.166` (Indian), `0.082` (Pacific), `0.010` (Atlantic), `-0.253` (Arctic), and `-0.708` (Southern). They estimated uncertainty using the spread of 10 independently trained XGBoost models (`ŷ ± 1.96σ`, mean `σ = 0.079`) without measuring empirical interval coverage.

Regional studies report smaller raw errors because their scope and target variance are far narrower. Ali et al. (2026) predicted raw microplastic abundance across 346 Indian Ocean points spanning `0 to 3.375` (mean `0.352`, std `0.683`), reporting `R² = 0.861`, `RMSE = 0.217`, `MAE = 0.118`, and `93.3%` coverage at nominal 90% on a regional holdout, without cross-study or cross-protocol testing. Chan et al. (2026) fused Japanese AOMI observations with ERA5 wave/wind reanalysis and Argo profiles in the West Pacific (2018 to 2020), reporting single-basin, single-year log-scale test `R² = 0.64 to 0.88` and `MAE = 0.645 to 1.004`.

### 2.2 Sampling harmonization, within-between separation, and study-level leakage

Koelmans et al. (2020) proposed power-law size rescaling (`dn/dx = a * x^(-alpha)`) to align microplastic counts across mismatched size windows. Applying this rescaling requires both lower and upper analytical size limits and a compartment-specific exponent `alpha`, metadata that NCEI does not consistently record across studies. In physical oceanography, Maximenko et al. (2012) showed that surface drifter velocity fields accumulate floating debris in the five subtropical convergence zones, where current speed is low rather than high. In environmental statistics, Helsel (2012) established that zero concentrations are left-censored non-detects below a method's limit of detection rather than literal small values, and panel econometrics (Mundlak, 1978) separates within-cluster slopes from between-cluster confounding when cluster-level bias is severe.

Meanwhile, 2026 audits in adjacent microplastic subfields have documented severe study-level leakage under random splits: Khanam & Younus (2026) showed FTIR polymer classification macro-F1 falling from 0.972 (random split) to 0.672 (leave-one-study-out), and Monreal Hernandez & Franco Amaya (2026) demonstrated similar collapse for microplastic adsorption models. Mokeira & Zhao (2026) quantified geographic confounding in global soil microplastics via spatial cross-validation. To our knowledge, no prior study combines study-grouped and organization-grouped validation, sampling-protocol ablation, physics-constrained within-between and left-censored modeling, leave-one-ocean-out protocol testing, and study-calibrated conformal prediction intervals on global marine microplastic concentrations.

## 3. Data and methodology

### 3.1 Dataset and cleaning

We use the NOAA NCEI Marine Microplastics export (`data/noaa_microplastics.csv`, downloaded `[DOWNLOAD DATE PENDING]`), containing 29,076 records across 35 columns from 1972 to 2026. The concentration column mixes incompatible units: pieces/m3 (water and beach sand), pieces/kg (sediment), and Nurdle Patrol counts (no numeric concentration). Filtering to `Marine Setting == "Ocean water"` and `Unit == "pieces/m3"` removes sediment, Nurdle Patrol, and 893 beach-sand records, leaving **16,988 records across 73 NCEI accessions, 73 DOIs, and 52 organizations**.

Two studies in the NCEI export contain a documented `10^6` unit-conversion error (dividing by 1,000 instead of multiplying when converting pieces/L to pieces/m3):

- **Barrows et al. (2018)** (`10.1016/j.envpol.2018.02.062`, 1,389 grab samples): the preprint reports a mean of 11.8 particles/L (11,800 pieces/m3), whereas NCEI stores a mean of 0.0128 pieces/m3. Barrows also counts all microparticles (of 113 FT-IR verified particles, 57% were synthetic, 12% semi-synthetic, and 31% non-synthetic).
- **Dehm et al. (2020)** (`10.1016/j.marpolbul.2020.111239`, 81 Niskin bottle samples): the abstract reports 1.6 to 2.0 MP/L (1,600 to 2,000 pieces/m3), whereas NCEI stores a mean of 0.00176 pieces/m3.

Both studies are multiplied by `10^6` in the primary dataset. A third small study, 33 bongo net rows from Arteaga et al. (2024), has a median of 2,860 pieces/m3, unusually high for a net; because the abstract mixes units and the full text is paywalled, these rows are left unchanged in the primary dataset and tested in sensitivity runs (Sections 4.2 and 4.6).

### 3.2 Target transform and study grouping

Because 64% of ocean-water concentrations lie below 0.1 pieces/m3 and values span eight orders of magnitude, `log1p(conc)` leaves most of the dataset untransformed near zero. We use:

`y = log10(conc + 0.001)`

The 0.001 offset approximates the smallest non-zero surface-net observation (`6.8e-4`), mapping zeros to -3 so that an absolute error of 1.0 corresponds to one order of magnitude (10x).

To prevent data leakage across linked accessions and publications, we define a **study** as a connected component of records sharing either a DOI or an NCEI accession number. This yields **72 distinct studies**. The largest study (Sea Education Association, 1986 to 2008) accounts for 6,135 rows (36% of the dataset). Because 15 organizations contributed multiple studies (including two Sea Education Association cruises totaling 8,665 rows, four from The Ocean Cleanup, and three each from 5 Gyres and Goa University), we also run an **organization-grouped** 5-fold stress test across the 52 distinct institutions.

### 3.3 Features and validation schemes

Baseline models use seven numeric features (`lat`, `lon`, `year`, `month_sin`, `month_cos`, `log10(mesh)`, `log1p(depth)`) and one grouped categorical feature (`method_family`: `surface net`, `pump`, or `bulk water`). CatBoost additionally receives the raw normalized `method` string. Missing depth values (0.3%) are imputed with the training-fold median.

We evaluate models under five validation schemes:

1. **Random 5-fold CV** (`KFold(5, shuffle=True, random_state=0)`): standard baseline.
2. **Study-grouped 5-fold CV**: both the deterministic `GroupKFold(5)` split and **10 repeated random study-fold assignments** (`GroupKFold(5, shuffle=True, random_state=0..9)`) to separate genuine model gains from fold-assignment noise.
3. **Organization-grouped 5-fold CV** (`GroupKFold(5)` on `organization`): ensures second cruises from the same institution cannot sit in both train and test.
4. **Chronological holdout**: train on `year < 2015`, test on `year >= 2015` (4,638 rows, split into seen vs unseen studies).
5. **Strict leave-one-ocean-out (LOOO)**: hold out each of the five ocean basins (Indian, Pacific, Atlantic, Arctic, Southern) in turn and exclude from training any study that also sampled the held-out basin.

Primary metrics on the `log10` scale are pooled `R²`, MAE, RMSE, mean per-study MAE, within-1-log accuracy (`|pred - y| <= 1`, within 10x), and catastrophic miss rate (`|pred - y| > 2`, >100x error). To anchor MAE against the physical noise limit, we also report an **oracle within-study median** reference that knows each unseen study's true median before predicting its stations (`R² = 0.627`, `MAE = 0.708`, or `5.1x`).

### 3.4 Study-calibrated prediction intervals and reliability flags

On each outer study-grouped fold, we compare four interval methods targeting 90% coverage (plus 95% for the Dalvi reference):

1. **Dalvi-style 10-run ensemble spread**: 10 LightGBM models trained with varying seeds, depths, and subsampling ratios, forming `ŷ ± 1.645σ` (nominal 90%) and `ŷ ± 1.96σ` (nominal 95%).
2. **Random-CV conformal prediction**: symmetric residuals calibrated via inner 5-fold random CV on the training fold.
3. **Study-grouped conformal prediction**: symmetric residuals calibrated via inner 4-fold `GroupKFold` on training studies, using both row-weighted and inverse-study-size-weighted quantiles.
4. **Study-grouped Conformalized Quantile Regression (CQR)** (Romano et al., 2019): LightGBM quantile models at `q = 0.05` and `q = 0.95` calibrated with signed conformity scores `max(q_lo - y, y - q_hi)` on inner study-grouped folds.

Finally, we evaluate label-free out-of-support flags computed strictly from each training fold (unseen raw sampling method, non-surface-net family, mesh size outside training min/max, haversine distance > 500 km to any training sample vs same-family training sample, and CQR interval width) and construct selective-prediction risk-coverage curves.

### 3.5 Methodological fixes: physics constraints, Within-Between transport, and left-censored modeling

Decomposing default LightGBM's out-of-fold MSE (`2.011`) on unseen studies reveals two concrete failure modes:

- **Spurious within-study variance.** True within-study variance `Var(y - study_mean)` is `1.274`, yet default LightGBM's within-study MSE is `1.581` (`78.6%` of total error). Unrestricted interactions between coordinates, year, and protocol produce wild within-study prediction swings (`Var = 0.422`, e.g., `std = 1.49` log10 units inside a single 0.33 mm manta-net study) with almost no alignment to true station order (`r = 0.078`), adding `+0.307` MSE above predicting a flat study constant.
- **The 6-log bulk-water zero cliff.** Mapping zeros to `-3.0` places empty 1-L bottles (`Barrows 2018`, `Dehm 2020`, `Van Dorn 2019`) six orders of magnitude below their own 1-particle detection floor (`670 to 1,100 pieces/m3`, `+2.8 to +3.0`). Barrows et al. (2018) alone accounts for **32.2% of total squared error** across the entire 16,988-row dataset (`SSE = 11,017 / 34,168`). Those `-3.0` zeros even flip the study-level correlation between `log_mesh` and `y_mean` inside bulk water from `-0.662` (positive rows) to `+0.416` (all rows).

To test targeted, leak-free fixes (`experiments_v2.py`), we evaluate four models:

1. **Constrained Protocol LightGBM**: shallow trees (`max_depth=3, num_leaves=7, n_estimators=150, min_child_samples=30`), a physics-backed decreasing **monotonic constraint** on `log_mesh` (`-1`, per Lindeque et al., 2020; Tokai et al., 2021), and **Huber loss** (`alpha=1.0`) to resist the 6-log bulk zero cliff and extreme outliers.
2. **Interaction-constrained + monotonic LightGBM**: all baseline features with `interaction_constraints=[[protocol], [space_time]]`, shallow trees, Huber loss, and `log_mesh = -1`.
3. **Within-Between Physics LightGBM (Protocol + Lagrangian drifter accumulation)**: using the 1° annual-mean Global Drifter Program `(u, v)` surface velocity grid (`data/environment/drifter_currents_1deg.csv`), we advect 80,000 passive tracers uniformly over ocean cells for 5 years (5-day steps with small isotropic diffusion, smoothed with a 2.5° Gaussian kernel) to compute a zero-parameter physical accumulation density `lag_acc = log1p(density)` (Maximenko et al., 2012). Inside each training fold, we estimate one nonnegative Fixed-Effects (Mundlak) slope `beta >= 0` for surface nets from within-study demeaned `y - study_mean(y)` regressed on `lag_acc - study_mean(lag_acc)`. Because every training study has mean zero, `beta` cannot absorb between-study bias. We subtract `beta * lag_acc` from training `y`, fit the constrained protocol tree to the transport-adjusted target, and add `beta * lag_acc` back on test surface-net rows. We also test an optional Stage 2 within-study spatial residual tree (`lat, lon, month_sin, month_cos`) with shrinkage tuned strictly by inner grouped CV.
4. **Left-Censored Tobit LightGBM and Two-Part Decomposition**: instead of pinning zeros to `-3.0`, we estimate protocol-specific limits of detection (`LOD`) strictly inside each training fold (median of study-level minimum positive concentration per `method_family` and fine/coarse mesh bucket). We train both an explicit **Two-Part** model (detection classifier `P(conc > 0)` plus conditional-positive `Within-Between Physics` regressor on `log10(conc) | conc > 0`) and a unified **Left-Censored Tobit LightGBM** optimizing the exact convex Tobit negative log-likelihood (`grad = (f - y)/sigma^2` for positives, `grad = inv_mills((LOD - f)/sigma)/sigma` for censored zeros).

## 4. Results

### 4.1 Baseline model comparison and feature ablations

An intercept-only linear mixed model (`y ~ 1 + (1 | study)`) yields an intraclass correlation (ICC) of **0.631**, confirming that nearly two-thirds of concentration variance sits between studies rather than within them.

Across nine baseline models (detailed in `docs/replication.md`), tree ensembles achieve random-split `R²` of **0.748 to 0.767** (`MAE = 0.569`, 87% to 88% within 10x), while a reference model that predicts only the training mean of each sample's study achieves random `R² = 0.664`. For direct scale comparison with Dalvi & Dalvi (2026), evaluating our same LightGBM under random 5-fold CV on their `log1p(conc)` scale yields **`R² = 0.815`, `MAE = 0.370`** (against Dalvi's `R² = 0.651`, `MAE = 0.619`). When entire studies are held out, baseline performance drops sharply:

| Model | Random R² | Grouped R² | Grouped MAE | Grouped within 10x | Chrono R² |
|---|---:|---:|---:|---:|---:|
| Study mean only | 0.664 | -0.283 | 1.792 | 27% | -0.228 |
| Linear regression | 0.545 | 0.407 | 1.158 | 52% | -1.668 |
| GAMM (mgcv) | 0.715 | -1.099 | 2.378 | 25% | -0.292 |
| Spatial kNN | 0.560 | -1.946 | 2.787 | 24% | -0.489 |
| Random forest | 0.764 | 0.410 | 1.007 | 69% | 0.258 |
| XGBoost | 0.758 | 0.328 | 1.132 | 62% | 0.203 |
| LightGBM | 0.764 | **0.473** | **0.996** | 64% | 0.170 |
| CatBoost | 0.748 | 0.438 | 1.067 | 61% | 0.165 |
| XGBoost hurdle | **0.767** | 0.464 | 1.018 | 64% | 0.091 |

Feature ablations reveal why models generalize at all to unseen studies. With LightGBM on study-grouped CV, **sampling protocol alone reaches `R² = 0.497`**, slightly above all features combined (`0.473`), while **location and time alone fall to `-0.097`** (matching CatBoost ablations: `0.401` protocol-only vs `-0.139` location-and-time-only, despite `0.716` random `R²`). TreeSHAP decompositions on CatBoost confirm that per-study mean SHAP values for protocol features track the mixed-model study offsets closely (`r = 0.91`, slope `0.81`), whereas location and time track them weakly (`r = 0.44`, slope `0.11`). Genetic algorithm hyperparameter tuning on pre-2015 grouped CV improves development RMSE (1.300 vs 1.491 default) but does not improve 2015+ holdout `R²` (0.160 vs 0.165), confirming that unconstrained tuning cannot fix cross-study data shift.

### 4.2 Sensitivity to unit-fixed and unverified studies

To verify that the audit findings are not artifacts of the Barrows/Dehm unit corrections or the unverified Arteaga bongo net rows, we reran the baseline benchmark after removing each subset:

| Subset | Rows | Studies | ICC | LGBM Random R² | LGBM Grouped R² | LGBM Grouped MAE | Within 10x | Protocol-only R² | Loc+Time-only R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full dataset | 16,988 | 72 | 0.631 | 0.764 | 0.473 | 0.996 | 63.9% | 0.497 | -0.097 |
| Drop bongo net (Arteaga 2024) | 16,955 | 71 | 0.612 | 0.759 | 0.556 | 0.915 | 66.1% | 0.537 | -0.026 |
| Drop Barrows & Dehm | 15,518 | 70 | 0.739 | 0.771 | 0.339 | 0.840 | 69.8% | 0.255 | 0.011 |
| Drop Barrows, Dehm & bongo | 15,485 | 69 | 0.720 | 0.766 | 0.462 | 0.772 | 73.3% | 0.429 | 0.117 |

Every qualitative conclusion survives intact. Dropping the 33 unverified bongo net rows raises LightGBM grouped `R²` from 0.473 to **0.556** (and Random Forest from 0.410 to 0.583), indicating that those 33 rows add noise rather than inflate our result. Dropping all three disputed studies still leaves a 0.30-point random-vs-grouped gap (0.766 vs 0.462), with protocol-only (`0.429`) outperforming location-and-time-only (`0.117`) by more than 0.31 `R²` points and achieving 73.3% within-10x accuracy.

### 4.3 Leave-one-ocean-out comparison against Dalvi et al. (2026)

Under strict leave-one-ocean-out validation (excluding any training study that also sampled the held-out basin), omitting protocol features reproduces the cross-basin failure reported by Dalvi & Dalvi (2026), whereas adding protocol features and Within-Between Lagrangian transport turns every basin positive and outperforms Dalvi & Dalvi (2026) in all five oceans:

| Held-out ocean | Test rows (studies) | Dalvi 2026 R² (no protocol) | LGBM Loc+Time only R² | Baseline LGBM All R² | Baseline LGBM Protocol R² | Within-Between Physics R² |
|---|---:|---:|---:|---:|---:|---:|
| Indian Ocean | 528 (16) | 0.166 | -0.635 | 0.234 | 0.100 | **0.194** |
| Pacific Ocean | 5,377 (27) | 0.082 | 0.126 | 0.515 | 0.519 | **0.541** |
| Atlantic Ocean | 10,671 (38) | 0.010 | -0.026 | 0.229 | **0.547** | 0.444 |
| Arctic Ocean | 385 (7) | -0.253 | -0.439 | -0.146 | 0.189 | **0.231** |
| Southern Ocean | 27 (3) | -0.708 | -0.600 | 0.207 | 0.445 | **0.680** |
| **Five-ocean mean (min)** | | **-0.141 (-0.708)** | **-0.315 (-0.635)** | **0.208 (-0.146)** | **0.360 (+0.100)** | **0.418 (+0.194)** |

![Leave-one-ocean-out comparison](../artifacts/fig_step3_looo_comparison.png)

Without protocol features, LightGBM produces negative `R²` in four of five basins (`-0.635` to `0.126`). Adding unconstrained spatial-temporal features back on top of protocol degrades cross-basin `R²` in four of five oceans (Atlantic drops from `0.547` to `0.229`, Arctic drops to `-0.146`). By contrast, the `Within-Between Physics` model raises the five-ocean mean `R²` to **`0.418`** (minimum basin `+0.194`) and exceeds Dalvi & Dalvi (2026) in all five basins.

### 4.4 Prediction intervals on unseen studies

Evaluating nominal 90% (and 95%) prediction intervals on unseen studies exposes severe under-coverage in standard uncertainty heuristics:

| Interval method | Row coverage | Mean study coverage | Studies >= 80% cov | Median width (log10) | Half-width factor |
|---|---:|---:|---:|---:|---:|
| Dalvi-style 10-run GBDT spread (±1.645σ, 90%) | 15.6% | 15.0% | 0.0% | 0.52 | ±1.8x |
| Dalvi-style 10-run GBDT spread (±1.96σ, 95%) | 19.1% | 18.7% | 2.8% | 0.62 | ±2.0x |
| Random-CV conformal (row-weighted, 90%) | 71.6% | 62.8% | 36.1% | 2.40 | ±15.8x |
| Study-grouped conformal (row-weighted, 90%) | 91.8% | 90.1% | 86.1% | 5.08 | ±348x |
| Study-grouped conformal (study-weighted, 90%) | 95.3% | 94.3% | 91.7% | 5.48 | ±551x |
| **Study-grouped CQR (row-weighted, 90%)** | **90.7%** | **91.4%** | **88.9%** | **3.85** | **±84x** |
| **Study-grouped CQR (study-weighted, 90%)** | **92.7%** | **93.3%** | **91.7%** | **3.86** | **±85x** |

![Interval coverage and width](../artifacts/fig_step4_interval_coverage.png)

Ensemble seed spread captures only model variance (median width 0.52 to 0.62 log10 units) and misses between-study shift almost entirely, covering fewer than 1 in 5 unseen-study observations. Random-CV conformal improves coverage to 71.6% of rows (62.8% mean per study), still far below the 90% target. Calibrating residuals across held-out studies restores target coverage (90.7% to 95.3%), and **study-grouped CQR** achieves both valid coverage (90.7% row, 91.4% mean study) and a **24% narrower median interval** than constant-width study-grouped conformal (3.85 vs 5.08 log10 units).

### 4.5 Reliability flags and selective prediction

Simple out-of-support rules cleanly isolate the largest unseen-study errors:

| Reliability flag | Share flagged | Flagged MAE | Unflagged MAE | Flagged >100x miss | Unflagged >100x miss |
|---|---:|---:|---:|---:|---:|
| Unseen raw sampling method | 14.2% | 1.99 | 0.83 | 35.0% | 6.7% |
| Non-surface-net family (bulk/pump) | 16.2% | 1.82 | 0.84 | 31.4% | 6.7% |
| Mesh size outside training min/max | 10.2% | 1.92 | 0.89 | 32.8% | 8.2% |
| Distance > 500 km (any training sample) | 9.4% | 0.98 | 1.00 | 12.7% | 10.5% |
| Distance > 500 km (same sampling family) | 26.1% | 1.27 | 0.90 | 18.0% | 8.2% |
| Top 25% CQR interval width | 25.0% | 1.65 | 0.78 | 28.7% | 4.8% |

Two patterns stand out. First, **protocol mismatch is the primary driver of catastrophic failure**: observations with an unseen raw sampling method or out-of-range mesh size suffer >100x error rates of 32.8% to 35.0%, roughly four to five times the unflagged rate (6.7% to 8.2%). Second, **geographic distance alone is uninformative unless conditioned on sampling family**: being within 500 km of *any* training sample yields no error reduction (MAE 0.98 vs 1.00), whereas being within 500 km of a *same-family* sample cuts MAE from 1.27 to 0.90 and halves >100x misses (18.0% to 8.2%).

![Selective prediction risk-coverage curves](../artifacts/fig_step5_risk_coverage.png)

In selective prediction on baseline LightGBM, deferring the top 20% widest CQR intervals (retaining 80% of unseen-study predictions) lowers overall MAE from **0.996 to 0.774**, increases within-10x accuracy from **63.9% to 70.9%**, and cuts catastrophic >100x misses from **10.7% to 4.7%**.

### 4.6 Methodological fixes: constrained Within-Between physics and left-censored modeling

Because a single study-grouped split can shift `R²` by more than 0.15 depending on fold assignment, we benchmarked our targeted fixes across **10 repeated random study-grouped 5-fold splits** and **organization-grouped 5-fold CV** (`experiments_v2.py`):

| Model (same `log10(conc + 0.001)` target) | Orig grouped R² | 10-repeat R² (min to max) | Wins vs baseline | 10-repeat MAE | Within 10x | >100x miss | Org-grouped R² (52 orgs) | Strict LOOO mean R² (min ocean) |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Baseline LightGBM (all features) | 0.487 | 0.456 (0.388 to 0.545) | 0/10 | 1.048 | 62.0% | 14.2% | 0.385 | 0.208 (-0.146) |
| Baseline LightGBM (protocol only) | 0.497 | 0.428 (0.263 to 0.525) | 4/10 | 1.073 | 63.8% | 13.3% | 0.308 | 0.360 (+0.100) |
| Interaction-constrained + mono LightGBM | 0.511 | 0.511 (0.449 to 0.559) | 9/10 | 0.976 | 62.2% | 8.7% | **0.460** | **0.432 (+0.126)** |
| Constrained Protocol (mono + Huber) | 0.566 | 0.543 (0.433 to 0.567) | 9/10 | 0.931 | **68.3%** | **7.1%** | 0.428 | 0.385 (+0.078) |
| **Within-Between Physics (Protocol + Lagrangian)** | **0.572** | **0.552 (0.476 to 0.575)** | **10/10** | **0.929** | 65.8% | 7.4% | 0.459 | 0.418 (**+0.194**) |
| Within-Between Physics + CV Space | **0.572** | 0.550 (0.467 to 0.570) | 10/10 | 0.931 | 65.9% | 7.6% | 0.436 | 0.418 (**+0.203**) |
| Oracle within-study median (noise floor) | 0.627 | 0.627 | 10/10 | 0.708 | 81.5% | 5.6% | 0.627 | n/a |

![10-repeat benchmark and strict LOOO comparison](../artifacts/fig_v2_benchmark.png)

Three results stand out across the global benchmark:

1. **Constrained Within-Between Physics closes over half the gap to the oracle noise floor.** Across 10 fold assignments, `Within-Between Physics` raises mean `R²` from `0.456` to **`0.552`** (beating baseline in **10 of 10** splits, with its worst split `0.476` exceeding baseline's average), cuts MAE from `1.048` (`11.2x`) to **`0.929`** (`8.5x`, against the `0.708` oracle floor), nearly halves `>100x` misses (`14.2%` to `7.4%`), and holds when holding out entire institutions under organization-grouped CV (`0.459` vs `0.385`). Adding a Stage 2 coordinate residual tree adds no further gain (`0.550` vs `0.552`), showing that raw coordinates carry little transferable within-study structure beyond physical drift accumulation.
2. **Lagrangian drifter accumulation transfers inside comparable surface-net studies where static current speed failed.** On the 32 open-ocean surface-net studies spanning more than one 1° grid cell, zero-parameter `lag_acc` achieves median within-study Spearman `rho = +0.141` (`62.5%` positive), whereas scalar `drifter_speed` is negative (`-0.100`, `34.4%` positive) and `lat` / `|lat|` sit at `0.000`. Floating debris accumulates in calm subtropical convergence zones rather than fast currents. Restricting to the **14,233-row surface-net subset (57 studies)** across 10 repeats, baseline LightGBM collapses to `R² = 0.035` (`MAE = 0.848`), Constrained Protocol rises to `0.253` (`MAE = 0.807`), and **Within-Between Physics** rises further to **`0.275`** (`MAE = 0.794`), beating protocol-only in **10 of 10** fold assignments (oracle surface-net noise floor: `R² = 0.514`, `MAE = 0.607`).
3. **Separating non-detects from detected concentrations isolates the 6-log bulk-water cliff.** Across 10 repeats:

| Model (censored and positive-only audit, 10 repeats) | Tobit NLL (lower is better) | Detection AUC | Positive-only R² (min to max) | Positive-only MAE | Positive within 10x | Positive >100x miss |
|---|---:|---:|---|---:|---:|---:|
| Baseline LightGBM (naive) | 2.013 | 0.610 | 0.493 (0.376 to 0.619) | 0.980 | 64.1% | 12.6% |
| Two-Part: Constrained Protocol | 1.189 | 0.630 | 0.798 (0.748 to 0.816) | 0.665 | 79.5% | 1.8% |
| **Two-Part: Within-Between Physics** | **1.173** | 0.630 | **0.815 (0.771 to 0.830)** | **0.633** | **80.9%** | **1.3%** |
| **Left-Censored Tobit LightGBM (Physics)** | 1.191 | **0.698** | 0.764 (0.689 to 0.792) | 0.741 | 71.2% | 1.6% |
| Oracle positive within-study median (floor) | n/a | n/a | 0.879 | 0.501 | 89.6% | 1.1% |

![Surface-net Lagrangian physics and left-censored Tobit results](../artifacts/fig_v2_physics_and_tobit.png)

Once microplastics are detected, `Two-Part Within-Between Physics` predicts unseen-study log10 concentrations at **`R² = 0.815`** and **`MAE = 0.633`** (`4.3x`, within `0.13` log units of the `0.501` oracle floor, with `80.9%` within 10x and only `1.3%` >100x misses). A single convex **Left-Censored Tobit LightGBM** cuts held-out censored negative log-likelihood from `2.013` to `1.191` and lifts unseen-study detection `P(conc > 0)` **AUC to `0.698`**, outperforming a separate binary classifier (`0.630`).

Finally, applying `Within-Between Physics` to uncertainty and selective prediction shrinks constant-width 90% study-grouped conformal intervals from `5.18` (`±391x`) to **`4.35`** log10 units (`±150x`) at `91.1%` coverage, and lowers selective-prediction MAE to **`0.749`** (`73.5%` within 10x, `4.3%` >100x misses) after deferring only the top **10%** widest intervals (versus 20% needed by baseline to reach `0.771`). Two honest caveats apply: first, protocol-only quantile trees give wider CQR intervals (`4.39 to 4.96` log10) than all-feature CQR (`3.81 to 3.96`) because 3 protocol features cannot adapt interval widths across sparse regions, so the strongest combined pipeline pairs `Within-Between Physics` point predictions with all-feature CQR bounds. Second, while `Within-Between Physics` beats baseline on the full dataset (`10/10`, `+0.096`), without Arteaga bongo (`8/10`, `+0.039`), and without Barrows & Dehm (`7/10`, `+0.072`), it trails baseline when all three are dropped together (`0.300` vs `0.397`, `3/10`), because removing both fine-mesh bulk studies removes the 6-log zero cliff that Huber loss protects against and leaves too few remaining bulk rows for `min_child_samples=30`.

## 5. Discussion and limitations

Our results carry four practical recommendations for marine microplastic modeling:

1. **Report study-grouped and organization-grouped validation alongside any global score.** On pooled NCEI data, random-split `R²` largely measures cruise recognition rather than environmental generalization, and comparing raw MAE across papers is meaningless without matching the target scale (`log10` vs `log1p` vs narrow single-basin raw counts).
2. **Model sampling protocol explicitly with physical regularization.** Most cross-study and cross-ocean transfer comes from mesh size, depth, and sampling family. Shallow trees with a decreasing monotonic constraint on mesh size and Huber loss eliminate spurious within-study swings and lift unseen-study `R²` by nearly 0.10 points.
3. **Separate within-study transport physics from between-study bias, and treat zeros as left-censored non-detects.** Static place descriptors (coast distance, current speed, SST climatology) act as cruise proxies when fitted to raw `y`, whereas Lagrangian surface-drifter accumulation fitted strictly on within-study residuals transfers positively across all 10 surface-net splits and all five held-out oceans. Treating non-detects via two-part or Tobit left-censored likelihoods exposes high cross-study predictability (`R² = 0.815`) on detected concentrations.
4. **Calibrate uncertainty across studies and defer out-of-support cases.** Uncalibrated ensemble spread is severely overconfident on new field campaigns. Study-grouped CQR provides honest 90% intervals, and flagging unseen methods, out-of-range meshes, or the top 10% to 20% widest CQR intervals brings retained MAE close to the `0.708` within-study noise floor.

**Limitations.** First, because each field campaign typically uses one protocol in one region, observational NCEI data cannot fully disentangle physical sampling selectivity from unrecorded laboratory practices or sampled water volume (which drives zero counts). Second, our Lagrangian transport field uses annual-mean drifter velocities; date-matched wind speed for Kukulka vertical-mixing corrections remains untested. Third, we did not test cross-dataset transfer to AOMI because Isobe et al. (2021) pools Law et al., Eriksen (2014), Cozar (2014), and Goldstein, which overlap directly with NOAA NCEI. Fourth, several background papers (Arteaga et al., 2024; Dehm et al., 2020; Yu et al., 2023) remain paywalled; our sensitivity analyses document how each subset affects both baseline and constrained models.

## 6. Conclusion

Across 16,988 ocean-water observations from 72 NOAA NCEI studies and 52 organizations, random cross-validation overstates unseen-study accuracy by nearly 0.30 `R²` points. Sampling protocol features drive cross-study and cross-ocean transfer, while unconstrained trees add spurious within-study variance and suffer from a 6-order-of-magnitude zero cliff in 1-L bulk-water studies. Replacing default GBDTs with shallow monotonic Huber protocol trees plus a within-study Fixed-Effects slope on 5-year Lagrangian drifter accumulation raises 10-repeat unseen-study `R²` from `0.456` to `0.552` (`10/10` splits, against a `0.627` within-study oracle ceiling), cuts `>100x` misses in half (`14.2%` to `7.4%`), holds under organization-grouped CV (`0.459`), improves surface-net-only transfer (`0.035` to `0.275`), and outperforms Dalvi & Dalvi (2026) in all five held-out ocean basins (`0.194` to `0.680`). Handling non-detects explicitly through two-part and convex left-censored Tobit modeling achieves positive-detection `R² = 0.815` (`MAE = 0.633`) and detection `AUC = 0.698`. Paired with study-grouped CQR intervals (90.7% coverage vs <20% for ensemble spread) and selective deferral of the riskiest 10% of predictions, unseen-study MAE drops to `0.749`, within `0.04` log10 units of the physical within-study noise floor.

## References

- Ali, N., Ahmed, S., Afia, M. H., Islam, A. K. M., Izlal, S., Rahman, S., Arif, M. A., Rahman, M. H., & Akondo, M. R. I. (2026). AI-driven prediction of marine microplastics from space: a case study of the Indian Ocean. *Discover Oceans*, 3, 59. https://doi.org/10.1007/s44289-026-00160-2
- Amora, E. N. O. (2025). A predictive framework for marine microplastic pollution using machine learning and spatial analysis. *IEEE ICSSAS 2025*. https://doi.org/10.1109/icssas66150.2025.11080971
- Barrows, A. P. W., Cathey, S. E., & Petersen, C. W. (2018). Marine environment microfiber contamination: Global patterns and the diversity of microparticle origins. *Environmental Pollution*, 237, 275-284. https://doi.org/10.1016/j.envpol.2018.02.062
- Chan, T. O., Chang, Z., & Yan, W. S. (2026). Fusing satellite remote sensing and ARGO float data for enhanced monitoring of microplastic concentrations in the West Pacific (2018-2020). *ISPRS Archives*, XLIX-B3-2026, 1185-1192. https://doi.org/10.5194/isprs-archives-XLIX-B3-2026-1185-2026
- Dalvi, P., & Dalvi, H. (2026). Monitoring marine microplastic distributions in data-scarce ocean regions using environmental representation learning and explainable machine learning. *Computer and Decision Making*, 3(6s), 546-573. https://doi.org/10.1922/comdem.v3i6s.18824
- Dehm, J., Singh, S., Ferreira, M., & Piovano, S. (2020). Microplastics in subsurface coastal waters along the southern coast of Viti Levu in Fiji, South Pacific. *Marine Pollution Bulletin*, 156, 111239. https://doi.org/10.1016/j.marpolbul.2020.111239
- Hafiz, A. M. I., & Ahmadisharaf, E. (2026). Satellite-based microplastic detection in marine environment using machine learning algorithms. *IEEE IGARSS 2026*. https://doi.org/10.1109/igarss60577.2026.11730366
- Helsel, D. R. (2012). *Statistics for Censored Environmental Data Using Minitab and R* (2nd ed.). Wiley.
- Isobe, A., Azuma, T., Cordova, M. R., et al. (2021). A multilevel dataset of microplastic abundance in the world's upper ocean and the Laurentian Great Lakes. *Microplastics and Nanoplastics*, 1, 16. https://doi.org/10.1186/s43591-021-00013-z
- Khanam, M. M., & Younus, S. (2026). Cross-study shift defines the reliability limits of FTIR polymer classification. *Chemometrics and Intelligent Laboratory Systems*. https://doi.org/10.1016/j.chemolab.2026.105903
- Koelmans, A. A., Redondo-Hasselerharm, P. E., Nor, N. H. M., de Ruijter, V. N., Sint Maarten, J. E. S., & Kooi, M. (2020). Solving the non-alignment of methods and approaches used in microplastic research in order to consistently characterize risk. *Environmental Science & Technology*, 54(19), 12307-12315. https://doi.org/10.1021/acs.est.0c02982
- Lindeque, P. K., Cole, M., Coppock, R. L., et al. (2020). Are we underestimating microplastic abundance in the marine environment? A comparison of microplastic capture with nets of different mesh-size. *Environmental Pollution*, 265, 114721. https://doi.org/10.1016/j.envpol.2020.114721
- Maximenko, N., Hafner, J., & Niiler, P. (2012). Pathways of marine debris derived from trajectories of Lagrangian drifters. *Marine Pollution Bulletin*, 65(1-3), 51-62. https://doi.org/10.1016/j.marpolbul.2011.04.016
- Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution*, 12(9), 1620-1633. https://doi.org/10.1111/2041-210x.13650
- Mokeira, K. D., & Zhao, Y. (2026). A spatial cross-validation framework for environmental machine learning: Quantifying geographic confounding in global soil microplastic prediction. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.6351275
- Monreal Hernandez, A., & Franco Amaya, S. L. (2026). Recognizing the study is not predicting adsorption: leave-one-study-out re-evaluation of machine-learning models for microplastic removal by biomass composites. *Zenodo*. https://doi.org/10.5281/zenodo.22999560
- Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69-85. https://doi.org/10.2307/1913646
- Narayanan, A. (2026). Global classification of ocean microplastic concentration levels using machine learning with geo-embeddings. *National High School Journal of Science*.
- Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems*, 32.
- Tokai, T., Uchida, K., Kuroda, M., & Isobe, A. (2021). Mesh selectivity of neuston nets for microplastics. *Marine Pollution Bulletin*, 165, 112111. https://doi.org/10.1016/j.marpolbul.2021.112111
- Yu, Z., Wang, L., Sun, H., & Liu, C. (2023). Prediction of microplastic abundance in surface water of the ocean and influencing factors based on ensemble learning. *Environmental Pollution*, 331, 121834. https://doi.org/10.1016/j.envpol.2023.121834
