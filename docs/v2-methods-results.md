# V2 methods: constrained Within-Between physics and left-censored modeling

Reproducible script: `experiments_v2.py` (merged into `main` from `feat/constrained-within-between`)
Artifacts: `artifacts/v2_*.csv`, `artifacts/fig_v2_*.png`

## Why baseline LightGBM left error on the table

Auditing the Option A out-of-fold predictions (`artifacts/step4_oof_predictions.csv`) showed two structural failure modes in default GBDT training on NCEI ocean-water data:

1. **Spurious within-study variance.** True within-study variance is `1.274`, yet default LightGBM has within-study MSE `1.581` (78.6% of total error). Its within-study predictions swing widely (`var = 0.422`) with almost no rank alignment (`r = 0.078`), adding `+0.307` MSE above predicting a flat study constant. Unregularized protocol-only LightGBM (500 trees, 31 leaves) also memorizes study-specific `(mesh, depth)` combinations, swinging from `0.263` to `0.525` across 10 fold assignments.
2. **A 6-log zero cliff in bulk water.** Mapping `conc = 0` to `log10(0 + 0.001) = -3.0` works for 1,000 m³ surface nets where 1 particle sits near `0.001 pieces/m³`, but creates a 6-order-of-magnitude cliff for 1-L bottle studies where 1 particle is `670 to 1,100 pieces/m³` (`+2.8 to +3.0`). Barrows et al. (2018) alone generates `32.2%` of total squared error across all 16,988 rows, and those `-3.0` zeros flip the bulk-water correlation between `log_mesh` and study mean `y` from `-0.662` (positive rows) to `+0.416` (all rows).

## Methods evaluated

1. **Constrained Protocol LightGBM.** Shallow trees (`max_depth=3, num_leaves=7, n_estimators=150, min_child_samples=30`), a monotonic constraint (`log_mesh = -1`, per Lindeque et al. 2020 and Tokai et al. 2021), and Huber loss to limit distortion from the 6-log bulk-water zero cliff and outliers.
2. **Interaction-constrained + monotonic LightGBM.** Uses all features with `interaction_constraints=[[protocol], [space_time]]`, shallow trees, Huber loss, and `log_mesh = -1`.
3. **Within-Between Physics LightGBM (Protocol + Lagrangian drifter accumulation).** Computes a 5-year passive-tracer accumulation field (`lag_acc`) directly from the 1° Global Drifter Program `(u, v)` surface velocity grid (`data/environment/drifter_currents_1deg.csv`, Maximenko et al. 2012). Inside each training fold, estimates one nonnegative within-study Fixed-Effects (Mundlak) slope for surface nets on `y - study_mean` vs `lag_acc - study_mean(lag_acc)` (zero between-study leakage), subtracts that transport term from `y`, fits the constrained protocol model on the transport-adjusted target, and adds the transport term back on test rows.
4. **Left-Censored Tobit LightGBM and Two-Part Decomposition.** Replaces the artificial `-3.0` floor with protocol-estimated limits of detection learned inside each training fold, using an exact convex Tobit negative log-likelihood objective in LightGBM and comparing against a separate detection classifier plus conditional-positive regressor.

## Headline results

### 1. Global unseen-study benchmark (`y = log10(conc + 0.001)`)

| Model | Orig grouped R² | 10-repeat R² (min to max) | Wins vs baseline | 10-repeat MAE | Within 10x | >100x miss | Org-grouped R² (52 orgs) | Strict LOOO mean R² (min ocean) |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Baseline LightGBM (all features) | 0.487 | 0.456 (0.388 to 0.545) | 0/10 | 1.048 | 62.0% | 14.2% | 0.385 | 0.208 (-0.146) |
| Baseline LightGBM (protocol only) | 0.497 | 0.428 (0.263 to 0.525) | 4/10 | 1.073 | 63.8% | 13.3% | 0.308 | 0.360 (+0.100) |
| Interaction-constrained + mono LightGBM | 0.511 | 0.511 (0.449 to 0.559) | 9/10 | 0.976 | 62.2% | 8.7% | **0.460** | **0.432 (+0.126)** |
| Constrained Protocol (mono + Huber) | 0.566 | 0.543 (0.433 to 0.567) | 9/10 | 0.931 | **68.3%** | **7.1%** | 0.428 | 0.385 (+0.078) |
| Within-Between Physics (Protocol + Lagrangian) | **0.572** | **0.552 (0.476 to 0.575)** | **10/10** | **0.929** | 65.8% | 7.4% | 0.459 | 0.418 (**+0.194**) |
| Within-Between Physics + CV Space | **0.572** | 0.550 (0.467 to 0.570) | 10/10 | 0.931 | 65.9% | 7.6% | 0.436 | 0.418 (**+0.203**) |
| Oracle within-study median (noise floor) | 0.627 | 0.627 | 10/10 | 0.708 | 81.5% | 5.6% | 0.627 | n/a |

Constrained monotonic Huber protocol trees deliver `+0.087` of the `+0.096` global 10-repeat R² gain (`0.456` to `0.543`). Adding the within-study Lagrangian drifter slope contributes a modest `+0.009` incremental global lift (`0.543` to `0.552`) while improving organization-grouped CV (`0.428` to `0.459`), surface-net-only transfer (`10/10` splits), and strict LOOO (`5/5` oceans above Dalvi & Dalvi 2026, lifting the Indian Ocean from `0.078` to `0.194`).

Strict leave-one-ocean-out R² (`Within-Between Physics` vs Dalvi & Dalvi 2026):
- Indian Ocean: `0.194` vs `0.166`
- Pacific Ocean: `0.541` vs `0.082`
- Atlantic Ocean: `0.444` vs `0.010`
- Arctic Ocean: `0.231` vs `-0.253`
- Southern Ocean: `0.680` vs `-0.708`

`Within-Between Physics` outperforms Dalvi & Dalvi (2026) in all five ocean basins.

### 2. Surface-net-only subset (14,233 rows, 57 studies)

On 32 open-ocean surface-net studies spanning more than one 1° grid cell, raw zero-parameter `lag_acc` achieves median within-study Spearman `rho = +0.141` (`62.5%` positive), whereas `drifter_speed` is negative (`-0.100`, `34.4%` positive) and `lat` / `|lat|` sit at `0.000`. On 10-repeat surface-net grouped CV:

- Baseline LightGBM (all features): `R² = 0.035`, `MAE = 0.848`, `>100x = 6.6%`
- Constrained Protocol (mono + Huber): `R² = 0.253`, `MAE = 0.807`, `>100x = 4.8%`
- Within-Between Physics (Protocol + Lagrangian): **`R² = 0.275`**, **`MAE = 0.794`**, beating protocol-only in **10 of 10** fold assignments (oracle noise floor: `R² = 0.514`, `MAE = 0.607`).

### 3. Left-censored Tobit and two-part decomposition (10 repeats)

| Model | Tobit NLL (lower is better) | Detection AUC | Positive-only R² (min to max) | Positive-only MAE | Positive within 10x | Positive >100x miss |
|---|---:|---:|---|---:|---:|---:|
| Baseline LightGBM (naive) | 2.013 | 0.610 | 0.493 (0.376 to 0.619) | 0.980 | 64.1% | 12.6% |
| Two-Part: Constrained Protocol | 1.189 | 0.630 | 0.798 (0.748 to 0.816) | 0.665 | 79.5% | 1.8% |
| Two-Part: Within-Between Physics | **1.173** | 0.630 | **0.815 (0.771 to 0.830)** | **0.633** | **80.9%** | **1.3%** |
| Left-Censored Tobit LightGBM (Physics) | 1.191 | **0.698** | 0.764 (0.689 to 0.792) | 0.741 | 71.2% | 1.6% |
| Oracle positive within-study median | n/a | n/a | 0.879 | 0.501 | 89.6% | 1.1% |

### 4. Uncertainty intervals and selective prediction

- **Constant-width study-grouped conformal (90% target):** `Within-Between Physics` shrinks median interval width from `5.18` (`±391x`) to `4.35` log10 units (`±150x`) at `91.1%` row coverage.
- **Selective prediction:** Deferring only the top **10%** widest CQR intervals on `Within-Between Physics` drops unseen-study MAE from `0.906` to **`0.749`** (`73.5%` within 10x, `4.3%` >100x misses), approaching the `0.708` within-study oracle floor while retaining 90% of rows. Option A Baseline required deferring 20% of rows to reach `0.771`.
- **CQR width trade-off:** Protocol-only quantile regressors produced wider CQR intervals (`4.39 to 4.96`, `96.3%` study-weighted over-coverage) than all-feature CQR (`3.81 to 3.96`) because 3 protocol features cannot adapt interval widths across sparse spatial regions. Pairing `Within-Between Physics` point predictions with all-feature CQR is the stronger combination.

### 5. Sensitivity audit (10 repeats)

- Full dataset: `0.552` vs `0.456` (`+0.096`, `10/10` wins)
- Drop Arteaga bongo net: `0.507` vs `0.467` (`+0.039`, `8/10` wins)
- Drop Barrows & Dehm: `0.294` vs `0.223` (`+0.072`, `7/10` wins)
- Drop all three (15,485 rows, 69 studies): `0.300` vs `0.397` (`-0.097`, `3/10` wins). Removing both major fine-mesh bulk studies and the bongo outlier removes the 6-log bulk zero cliff that Huber loss guards against and leaves very few bulk-water rows for `min_child_samples=30`.
