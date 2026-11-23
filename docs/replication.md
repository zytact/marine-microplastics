# Replicating the results

This covers everything on `main` in `microplastics.ipynb`: data cleaning, features, validation, all nine
models, the GA search, SHAP and the map. The numbers in each section are the ones to match. If a checksum
(row count, study count, fold sizes) differs, stop and fix that first, because every score depends on it.

The best results:

| Test | Best model | R² | Within 10x |
|---|---|---|---|
| Random 5-fold | XGBoost hurdle / LightGBM / random forest | 0.76 to 0.77 | 87 to 88% |
| Unseen studies (grouped 5-fold) | LightGBM | 0.473 | 64% |
| Future (train before 2015, test 2015 on) | Random forest | 0.258 | 61% |

The unseen-studies score is the headline. Random-split scores are inflated because the model has already
seen other samples from the same study.

## 1. Environment

- Python 3.12 with `uv`. Versions from `uv.lock`: numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1,
  lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, statsmodels 0.15.0, shap 0.52.0.
- R with `mgcv`, called through `gamm.R`. Only the GAMM needs it.
- `SEED = 0` everywhere. Boosters and random forest use `n_jobs=4` / `thread_count=4`.

Different library versions can move scores in the second decimal. The fold assignment and data cleaning
matter far more.

## 2. Data

Source is `data/noaa_microplastics.csv`, the NCEI Marine Microplastics export: 29,076 records, 35 columns.

Steps, in order:

1. Keep rows where `Marine Setting == "Ocean water"` and `Unit == "pieces/m3"`. This drops sediment
   (pieces/kg), Nurdle Patrol counts and 893 beach rows that are in pieces/m3 but measure sand.
2. Multiply `Microplastics Measurement` by 10^6 for two studies that NCEI stores 10^6 too low. Their
   litre-to-m3 conversion divided by 1,000 instead of multiplying:
   - `https://doi.org/10.1016/j.envpol.2018.02.062` (Barrows et al. 2018, 1,389 grab samples). The
     preprint at `https://osf.io/download/h9z43/` reports 11.8 particles/L (n = 1,393), which is 11,800 per
     m3. The NCEI mean is 0.0128.
   - `https://doi.org/10.1016/j.marpolbul.2020.111239` (Dehm et al. 2020, Niskin bottles). The abstract on
     Europe PMC reports 1.6 and 2.0 MP/L, which is 1,600 to 2,000 per m3. The NCEI mean is 0.00176. The full
     text has no free copy.

   Skip this and the unseen-study R² collapses. With a `log1p` target and no fix, XGBoost scored -3.65.

   Both studies filter 1 L or less through a 0.45 µm membrane, and that pore size is what NCEI stores as
   `Mesh Size (mm)` (0.00045). Barrows counts all microparticles, not only plastic. Of the 113 it checked by
   FT-IR, 57% were synthetic, 12% semi-synthetic and 31% non-synthetic.
3. Parse `Sample Date` with format `%m/%d/%Y %I:%M:%S %p`.

Checksum: **16,988 records, 73 accessions, 73 DOIs, 72 studies.**

The 33 bongo net rows (Arteaga et al. 2024) have a median of 2,860 pieces/m3, which looks too high for a
net. We couldn't verify them, so they stay unchanged.

## 3. Target

`y = log10(conc + 0.001)`

Not `log1p`. 64% of concentrations are below 0.1, where `log1p(x) ≈ x`, so `log1p` leaves most of the data
squashed near zero. The 0.001 offset is about the smallest non-zero value (6.8e-4), so zeros map to -3 and
an error of 1 means one order of magnitude.

## 4. Features

| Feature | How it's built |
|---|---|
| `lat`, `lon` | Raw degrees |
| `year` | From the date |
| `month_sin`, `month_cos` | `sin(2π·month/12)`, `cos(2π·month/12)` |
| `log_mesh` | `log10(Mesh Size (mm))` |
| `log_depth` | `log1p(Water Sample Depth (m))`. 0.3% missing, imputed with the training-fold median |
| `method_family` | Categorical, see below |
| `method` | Raw method name, lowercased with `-` replaced by a space. CatBoost only |

`method_family` maps the lowercased method:

- **surface net**: neuston net, manta net, plankton net, avani net, bongo net
- **pump**: pump, intake seawater pump, surface water intake
- **bulk water**: everything else (grab sample, buckets, Niskin, Van Dorn, CTD rosette, glass jar)

Bulk water samples read about four orders of magnitude above surface nets, and this split carries most of
what transfers to new studies.

For the sklearn-style models (linear, random forest, XGBoost, LightGBM, hurdle) the preprocessing is one
`ColumnTransformer`:

- numeric columns: `SimpleImputer(strategy="median")` then `StandardScaler()`
- `method_family`: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

The numeric features of the test rows are also clipped to the training range before predicting. That only
changes linear regression and the GAMM, since trees are already flat outside the training range.

## 5. What a "study" is

The group for grouped CV. Rows are linked if they share a DOI or an NCEI accession, directly or through a
chain, and each linked set is one study. Grouping by either field alone fails. One paper was archived as
two accessions, and one accession holds a published Oceaneye survey plus later unpublished data.

```python
def linked_groups(a: pd.Series, b: pd.Series) -> pd.Series:
    group = a.astype(str)
    while not (merged := group.groupby(b).transform("min").groupby(a).transform("min")).equals(group):
        group = merged
    return group

study = linked_groups(doi, accession)
```

One study (Sea Education Association, 1986 to 2008) has 6,135 rows, 36% of the data. Study identity alone
explains 63% of the target variance (ICC from `smf.mixedlm("y ~ 1", df, groups=study)`).

## 6. Validation

| Scheme | Definition | Test fold sizes |
|---|---|---|
| Random | `KFold(5, shuffle=True, random_state=0)` | 3398, 3398, 3398, 3397, 3397 |
| Grouped | `GroupKFold(5)` on `study`, no shuffle | 6135, 2714, 2712, 2714, 2713 |
| Chronological | Train `year < 2015`, test `year >= 2015` | 4638 |

The fold sizes are the checksum. Our `GroupKFold` is deterministic, so with the same row order and the same
study labels you get the same folds. The grouped R² depends heavily on fold assignment. Across 10 random
assignments of studies to folds, the same model ranged from 0.38 to 0.55, so quote the range next to the
0.473.

In the chronological split, 25 studies are before 2015 and 54 after, and 7 appear in both. 2,155 of the
4,638 test rows come from those 7 studies.

## 7. Metrics

All on the log10 target, from out-of-fold predictions:

- **R²**, pooled. All out-of-fold predictions are scored together against the global variance. This is the
  headline. Per-fold R² uses each fold's own variance and is much lower (no model has a positive fold mean).
- **MAE** and RMSE in log10 units.
- **Study MAE**: the MAE of each study, then averaged, so small studies count as much as big ones.
- **Within 1 log**: share with `|pred - y| <= 1`, so within 10x of `conc + 0.001`. Closest thing to an
  accuracy.

## 8. Models

All take the features from section 4 and predict `y`.

| Model | Configuration |
|---|---|
| Study mean only | Training mean of the sample's own study, global mean for unseen studies. A reference, not a candidate |
| Linear regression | `LinearRegression()` |
| GAMM (mgcv) | `gamm.R`, see below |
| Spatial kNN | `KNeighborsRegressor(n_neighbors=15, weights="distance", metric="haversine", algorithm="ball_tree")` on radians of lat/lon only |
| Random forest | `RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0)` |
| XGBoost | `XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=0)` |
| LightGBM | `LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=0, verbose=-1)` |
| CatBoost | `CatBoostRegressor(iterations=500, learning_rate=0.05, random_seed=0)`, raw features plus `method`, with `method_family` and `method` passed as `cat_features`. No sklearn preprocessing |
| XGBoost hurdle | `XGBClassifier` for `conc > 0` on all rows, `XGBRegressor` on positive rows only, both with the XGBoost settings minus `max_depth`. Prediction = `p · reg + (1 - p) · log10(0.001)` |

The GAMM (`gamm.R`) is fitted with `mgcv::bam(method = "fREML", discrete = TRUE)`:

```r
y ~ s(lat, lon, bs = "sos", k = 100) + s(year, k = 8) + month_sin + month_cos +
    log_mesh + log_depth + fam_pump + fam_bulk + s(study, bs = "re")
```

`fam_pump` and `fam_bulk` are 0/1 flags (surface net is the baseline), dropped if constant in the training
fold. Missing numeric values are filled with the training median. Seen studies get their random effect.
Unseen studies are predicted with `exclude = "s(study)"`.

## 9. Results to match

| Model | Random R² | Grouped R² | Grouped MAE | Grouped within 10x | Chrono R² | Chrono within 10x |
|---|---|---|---|---|---|---|
| Study mean only | 0.664 | -0.283 | 1.792 | 27% | -0.228 | 39% |
| Linear regression | 0.545 | 0.407 | 1.158 | 52% | -1.668 | 44% |
| GAMM (mgcv) | 0.715 | -1.099 | 2.378 | 25% | -0.292 | 54% |
| Spatial kNN | 0.560 | -1.946 | 2.787 | 24% | -0.489 | 45% |
| Random forest | 0.764 | 0.410 | 1.007 | 69% | **0.258** | 61% |
| XGBoost | 0.758 | 0.328 | 1.132 | 62% | 0.203 | 59% |
| LightGBM | 0.764 | **0.473** | **0.996** | 64% | 0.170 | 59% |
| CatBoost | 0.748 | 0.438 | 1.067 | 61% | 0.165 | 63% |
| XGBoost hurdle | **0.767** | 0.464 | 1.018 | 64% | 0.091 | 61% |

Grouped R² per fold, LightGBM: -0.556, 0.004, 0.302, -0.438, 0.332 (mean -0.071).

Chronological R² split by whether the test study also sampled before 2015 (random forest): 0.322 on seen
studies, 0.124 on unseen ones. Random forest is the only model above zero on unseen ones.

## 10. Feature ablations (CatBoost, default settings)

| Features | Random R² | Grouped R² |
|---|---|---|
| All features | 0.748 | 0.438 |
| All, `method_family` without raw `method` | 0.750 | 0.437 |
| All + study ID as a feature | 0.748 | 0.455 |
| Sampling protocol only (`log_mesh`, `log_depth`, `method_family`, `method`) | 0.655 | 0.401 |
| Location and time only (`lat`, `lon`, `year`, month sin/cos) | 0.716 | -0.139 |

Most of what transfers to new studies is the sampling protocol. Location and time work on random folds only
because they identify the cruise.

## 11. Genetic algorithm search (CatBoost)

The search sees only the pre-2015 data. Fitness is the pooled RMSE from `GroupKFold(4)` on `study` within
those years. The best parameters are then scored on 2015 onwards.

Search space. Each gene is in [0, 1] and decoded as below. `bootstrap_type="Bernoulli"` is always set.

| Parameter | Range | Scale |
|---|---|---|
| `depth` | 3 to 8 | int (rounded) |
| `learning_rate` | 0.01 to 0.3 | log |
| `iterations` | 200 to 1000 | int (rounded) |
| `l2_leaf_reg` | 0.1 to 30 | log |
| `random_strength` | 0.1 to 10 | log |
| `subsample` | 0.5 to 1.0 | linear |
| `rsm` | 0.5 to 1.0 | linear |

Log decoding is `low · (high / low) ** gene`.

GA settings, with `np.random.default_rng(0)`:

- population 12, 10 generations, 102 evaluations in total
- the 2 best survive unchanged (elitism)
- each parent is the best of 3 random picks (tournament)
- crossover: `child = w·a + (1 - w)·b` with a random `w` per gene
- mutation: each gene gets `N(0, 0.15)` noise with probability 0.2, then is clipped to [0, 1]

The control is a random search with the same 102 evaluations, using `default_rng(1)`.

| | Dev CV RMSE | 2015+ R² | 2015+ within 10x |
|---|---|---|---|
| Default | 1.491 | 0.165 | 63% |
| Random search | 1.346 | 0.115 | 64% |
| Genetic algorithm | **1.300** | 0.160 | 65% |

GA best: depth 5, learning_rate 0.051, iterations 327, l2_leaf_reg 0.122, random_strength 7.5,
subsample 0.614, rsm 0.545. The GA wins on the development folds, but that doesn't carry over to the
holdout. Tuning doesn't fix a gap that comes from the data.

## 12. SHAP

CatBoost with the GA parameters, refit on all 16,988 rows, explained with `shap.TreeExplainer`. Expected
picture:

- The largest contributions come from `method`, `method_family` and `log_mesh`, then `lat` and `year`.
- Latitude peaks between about 20° and 45° N, the subtropical gyres.
- The year effect is flat until about 2005, then rises by about half an order of magnitude.
- The finest meshes push predictions up. If they push down, the unit fix is missing.
- The highest prediction is a Barrows grab sample, about 5,500 pieces/m3.

Per-study mean SHAP against the study offsets from the intercept-only mixed model: protocol features
r = 0.91 (slope 0.81), location and time r = 0.44 (slope 0.11).

## 13. Map

This is an exploratory scenario, not a validated map. The final CatBoost predicts every 2° cell (lat -79
to 89, lon -179 to 179) for one standard sample: year 2015, manta net, `method_family="surface net"`,
mesh 0.333 mm, depth 0. The 12 monthly predictions are averaged on the log scale. Cells more than 500 km
from any surface-net observation (BallTree haversine) are blanked.


## 14. Paper experiments (`core.py`, `experiments.py`, `experiments_v2.py`)

The paper's tables come from two scripts that share `core.py`, so both load the same data, build the same
features (no scaling for trees) and score the same baseline. `core.py` merges the three names NCEI uses for
Galway-Mayo Institute of Technology and the two it uses for Italy's National Research Council, which gives 49
organizations instead of 52. Every study-grouped and organization-grouped score is a mean over
10 random fold assignments (`GroupKFold(5, shuffle=True, random_state=0..9)`).

```sh
.venv/bin/python experiments_v2.py   # about 6 minutes on 12 cores
.venv/bin/python experiments.py      # about 12 minutes, needs catboost, xgboost, h5py and pyshp
.venv/bin/python external.py        # under a minute, external test on data/aomi/, data/pangaea/ and data/zhao2025/ (needs openpyxl)
.venv/bin/python paper/figures.py    # PDFs into paper/figures
cd paper && tectonic -X compile main.tex
```

Run the scripts one at a time. Each already uses all cores, and running them together oversubscribes the CPU and makes every run several times slower.

The `.venv` here has `catboost`, `xgboost-cpu`, `h5py`, `pyshp` and `openpyxl` installed with `uv pip install`, because `uv sync` stalls on
the CUDA wheel that `xgboost` pulls in.

Checksums: 16,988 rows, 72 studies, 49 organizations, years 1972 to 2024, study-constant oracle R² 0.666
(study mean) and MAE 0.708 (study median).

| Output | Paper | Value to match |
|---|---|---|
| `artifacts/audit_baselines.csv` | Table I | LightGBM random R² 0.763, grouped R² 0.456 |
| `artifacts/audit_sensitivity.csv` | Table II | Full data protocol-only 0.428, location and time only -0.022, as archived all features -0.088 |
| `artifacts/external_scores.csv`, `external_groups.csv`, `external_intervals.csv`, `external_aomi_overlap.csv`, `external_aomi_shared_studies.csv` | Table III, Section V-D | 3,044 samples in 39 groups. AOMI nets (2,429): family mean per study MAE 0.754, baseline 0.920, constrained 1.441 (1.080 with recorded net depth). Pump and bottle (571): protocol only 0.875, baseline 0.958, family means 1.322 and 1.455; with compilation pumps at 1 m and 10 m, baseline 0.960 and 0.999. Study CQR coverage 93.7% overall |
| `artifacts/audit_intervals.csv` | Table IV, Fig. 1 | Study CQR (study) coverage 90.9%, median width 3.85 |
| `artifacts/audit_flags.csv`, `audit_risk_coverage.csv` | Table V, Fig. 2 | Unseen method flagged MAE 2.34 |
| `artifacts/audit_environment.csv` | Section V-C | LightGBM 0.456 to 0.449 with covariates |
| `artifacts/audit_calibration.csv`, `audit_ranking.csv` | Section V-C | Global mean after 3 samples 0.556 |
| `artifacts/v2_diagnosis.csv` | Section IV-G | Baseline MSE 2.073, Barrows share of squared error 37.2% |
| `artifacts/v2_global_benchmark.csv`, `v2_global_repeats.csv` | Tables VI and VII, Fig. 3 | Constrained protocol 0.543, Within-Between Physics 0.552, baseline org-grouped 0.282 |
| `artifacts/v2_study_bootstrap.csv` | Section VI-B | Constrained vs. baseline R² gain 0.087, 95% interval -0.047 to 0.198 |
| `artifacts/v2_dalvi_scale.csv` | Fig. 4 | Constrained protocol LOOO mean 0.648 on ln(c+1) |
| `artifacts/v2_surface_nets.csv`, `v2_surface_net_ranking.csv` | Fig. 5a | 58 studies, Within-Between Physics 0.275 |
| `artifacts/v2_censored.csv` | Table VIII, Fig. 5b | Two-part Physics NLL 1.519 and positive R² 0.815, two-part baseline 0.708 |
| `artifacts/v2_sensitivity.csv` | Table IX | Without all three, baseline 0.397, Physics 0.300, baseline org-grouped 0.162 |