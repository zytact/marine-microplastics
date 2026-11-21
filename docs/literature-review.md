# Literature review and novelty audit

Updated 2026-11-21.

## Verdict

The main claim holds. Across OpenAlex, PubMed, and the 2026 marine microplastics papers already collected in `~/Projects/marine-microplastic-review-paper/papers`, no published marine microplastic concentration model combines:

1. Study-grouped cross-validation (holding out entire publications and NCEI accessions rather than random rows).
2. Explicit ablation separating sampling-protocol features (sampling family, raw method, mesh size, water depth) from spatial-temporal features on unseen studies and held-out ocean basins.
3. Study-calibrated prediction intervals (conformal and CQR) and label-free reliability flags evaluated for empirical coverage on unseen studies.

The closest conceptual papers demonstrating study-level data leakage in microplastic ML were published in 2026 in different subfields (FTIR spectroscopy and laboratory adsorption), confirming that the problem is active and recognized, while ocean concentration modeling has not yet been audited this way.

## Closest marine microplastic ML papers

| Paper | Venue and DOI | Data and task | Validation | Uncertainty | Gap relative to our work |
|---|---|---|---|---|---|
| Dalvi & Dalvi (2026) | *Computer and Decision Making* 3(6s):546-573, `10.1922/comdem.v3i6s.18824` (OA) | NOAA NCEI, contrastive representation learning + XGBoost regression | Random CV (`R² = 0.651`) and leave-one-ocean-out (`R²`: Indian 0.166, Pacific 0.082, Atlantic 0.010, Arctic -0.253, Southern -0.708) | 10-run XGBoost ensemble spread (`ŷ ± 1.96σ`, mean `σ = 0.079`); no empirical coverage check | No sampling-protocol features, no study-grouped split, no empirical coverage validation of intervals |
| Ali et al. (2026) | *Discover Oceans* 3:59, `10.1007/s44289-026-00160-2` (OA) | 346 Indian Ocean points, 12 satellite/reanalysis features, GBR/XGBoost/LightGBM | Random CV (`R² = 0.843`) and regional geographic holdout | 90% prediction interval coverage on regional holdout | Single basin, 346 points, no study-grouped split, no sampling-protocol ablation |
| Narayanan (2026) | *NHSJS* (OA) | ~20,000 NOAA NCEI records, 5-tier concentration classification with SatCLIP geo-embeddings | Stratified CV and leave-one-ocean-out | None | Classification only, no sampling-protocol features, no study-grouped split |
| Yu et al. (2023) | *Environmental Pollution* 331:121834, `10.1016/j.envpol.2023.121834` (Paywalled) | 1,169 global surface-water samples, 16 environmental/socio-economic predictors, multi-class intervals | Random split (XGBoost accuracy 0.719, AUC 0.914) | None | Multi-class environmental features only; no study-grouped split or protocol ablation |
| Hafiz & Ahmadisharaf (2026) | *IEEE IGARSS 2026*, `10.1109/igarss60577.2026.11730366` (Paywalled full text, OA abstract) | MODIS reflectance + NOAA NCEI, RF/GB stacked ensemble | Regional transfer (Bay of Bengal/ECS to Mediterranean fails with negative `R²`) | None | Optical remote-sensing transfer, not sampling-protocol or study-grouped uncertainty |
| Enders et al. (2026) | *Microplastics and Nanoplastics* 6:62, `10.1186/s43591-026-00236-y` (OA) | Marine sediments (NIXVEGS) | Spatial CV | None | Sediments, not water column; no study-grouped protocol ablation |
| Amora (2025) | *IEEE ICSSAS 2025*, `10.1109/icssas66150.2025.11080971` (Paywalled) | Global ocean samples, constructed features including Regional Microplastic Concentration | Random split (`R² = 0.972`) | None | Random split with regional concentration feature; illustrates optimistic leakage |

## Parallel study-leakage and methodological papers (for Related Work)

- **Khanam & Younus (2026)**, *Cross-study shift defines the reliability limits of FTIR polymer classification*, *Chemometrics and Intelligent Laboratory Systems*, `10.1016/j.chemolab.2026.105903` (OA). Shows XGBoost macro-F1 drops from 0.972 (random split) to 0.672 (leave-one-study-out) across 4,783 spectra from 4 studies, and evaluates selective prediction thresholds. Strong methodological parallel to our study-grouped evaluation and reliability flags.
- **Monreal Hernandez & Franco Amaya (2026)**, *Recognizing the study is not predicting adsorption: leave-one-study-out re-evaluation of machine-learning models for microplastic removal by biomass composites*, Zenodo `10.5281/zenodo.22999560` (OA). Demonstrates that random splits on pooled microplastic studies measure study recognition rather than physical generalization.
- **Mokeira & Zhao (2026)**, *A Spatial Cross-Validation Framework for Environmental Machine Learning: Quantifying Geographic Confounding in Global Soil Microplastic Prediction*, SSRN `10.2139/ssrn.6351275` (OA). Quantifies geographic confounding in global soil microplastics.
- **Meyer & Pebesma (2021)**, *Predicting into unknown space? Estimating the area of applicability of spatial prediction models*, *Methods in Ecology and Evolution*, `10.1111/2041-210x.13650` (OA). Foundational reference for flagging out-of-support environmental predictions.
- **Lindeque et al. (2020)** (`10.1016/j.envpol.2020.114721`) and **Tokai et al. (2021)** (`10.1016/j.marpolbul.2021.112111`). Empirical physics showing why finer meshes and bulk-water methods measure orders of magnitude higher particle counts than 0.333 mm surface nets.
- **Isobe et al. (2021)** (`10.1186/s43591-021-00013-z`). Dataset paper behind AOMI; pools Law et al., Eriksen 2014, Cozar 2014, and Goldstein, which overlap directly with NOAA NCEI. Explains why NOAA-to-AOMI transfer is not an independent external test.

## Verification of the Koelmans (2020) harmonisation note

Verified citation: **Koelmans et al. (2020)**, *Solving the Non-alignment of Methods and Approaches Used in Microplastic Research in Order to Consistently Characterize Risk*, *Environmental Science & Technology*, 54(19), 12307-12315, `10.1021/acs.est.0c02982` (building on Kooi & Koelmans 2019).

The method rescales particle counts from one measured size window to a default window (1 to 5,000 µm) using a power-law particle size distribution (`dn/dx = a * x^(-alpha)`). Applying it requires the lower and upper analytical size limits and a compartment-specific power-law exponent `alpha`. The NOAA NCEI export provides mesh size for most rows, but does not record analytical detection limits, upper size cutoffs, or polymer/shape distributions consistently across studies. Physics-based Koelmans rescaling is therefore not directly computable from NCEI metadata alone, which is why protocol-aware ML modeling and reliability flagging are needed.

## Paywalled papers encountered and need rating

| Paper | DOI | Why it matters | Need rating |
|---|---|---|---|
| Arteaga et al. (2024) (bongo net study in NCEI) | Paywalled full text (abstract mixes units) | 33 bongo net rows in NCEI have a median of 2,860 pieces/m3, far above other nets. Full text would confirm whether a unit error occurred. | **Low** (if Step 2 sensitivity checks show results are unchanged without these 33 rows) |
| Dehm et al. (2020) | `10.1016/j.marpolbul.2020.111239` | Unit fix (`10^6`) was verified against the Europe PMC abstract (1.6-2.0 MP/L vs 0.00176 in NCEI). Full text would only verify per-station numbers. | **Low** (abstract confirms the 10^6 error, and Step 2 tests results without it) |
| Yu et al. (2023) | `10.1016/j.envpol.2023.121834` | Cited global ocean surface ML paper (1,169 samples, XGBoost multi-class). PubMed summary already gives dataset size, features, and metrics. | **Low-to-Medium** (nice to verify exact CV split details, but abstract/summary suffices for Related Work) |
| Koelmans et al. (2020) | `10.1021/acs.est.0c02982` | Power-law size rescaling framework. Core equations and data requirements are fully documented in open summaries and Kooi & Koelmans (2019). | **Low** |
| Hafiz & Ahmadisharaf (2026) | `10.1109/igarss60577.2026.11730366` | MODIS + NOAA NCEI cross-basin remote sensing transfer. Full abstract is open on OpenAlex and contains all key R² numbers. | **Low** |
| Amora (2025) | `10.1109/icssas66150.2025.11080971` | Random-split global ocean ML (`R² = 0.972`). Abstract is open and sufficient. | **Very Low** |
