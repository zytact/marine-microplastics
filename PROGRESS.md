# Progress

Updated 2026-11-23.

## Where things live

- Paper (IEEE journal draft): `paper/main.tex`, built with `tectonic`. Figures come from `paper/figures.py`, which reads `artifacts/`. It replaces the old markdown report and v2 write-up.
- Paper experiments: `experiments.py` (baselines, sensitivity, intervals, flags) and `experiments_v2.py` (diagnosis, constrained models, Dalvi-scale LOOO, surface nets, censored models, robustness). Both import `core.py`. Run commands and checksums: `docs/replication.md` section 14.
- Notebook `microplastics.py`: the original exploratory audit. The paper no longer quotes it (its GAMM needs R, which is not installed here). It still uses `StandardScaler` and 52 raw organization names. `environment.py` ports its environmental covariates for `experiments.py`.
- Literature audit: `docs/literature-review.md`.

## Settled

Every paper score is a mean over 10 study-fold assignments. Single-split numbers from before 2026-11-22 (0.473, 0.487, 5.08 and similar) are retired.

Paper framing: an audit. Random splits overstate transfer by about 0.31 R2, location and time transfer nothing (-0.022), and protocol carries what transfers. Gemini 4 Neon reviewed the paper five times on 2026-11-22, and the fifth pass found nothing substantive. On 2026-11-23 a GPT 6.2 Astra critique, adjudicated by Neon, led to the as-archived runs, the study bootstrap and the all-feature two-part baseline. All findings are addressed. Two code nits were left alone, and the third review judged both acceptable: the drift field's Gaussian smoothing does not wrap at the antimeridian (changes 68 of 16,988 values by more than 0.01), and the study-weighted conformal quantile has no finite-sample correction. Paired significance tests over the 10 correlated fold assignments were judged invalid and are not reported. A non-NCEI test set is stated as a limitation.

Findings the paper must keep stating, because they cut against the headline:
- Protocol's advantage over location and time mostly comes from Barrows and Dehm. Without them protocol-only scores 0.100 against 0.055.
- The constrained models lose to the baseline once Barrows, Dehm and the bongo study are all removed (0.300 against 0.397). A minimum leaf size of 10 does not fix it. Without the bulk-water studies, protocol carries little (protocol-only scores -0.085 by organization on that subset), and location and time help the baseline separate the remaining net studies.
- With the archived units (no correction) every study-grouped score is negative, and the constrained models do worse than the baseline (-0.184 against -0.088). The protocol result depends on the correction.
- Archived-unit R2 is not comparable with corrected R2: the correction doubles the variance of y (1.92 to 3.81), and baseline MAE barely moves (1.054 against 1.048). Say "negative study-grouped R2", never "no model transfers".
- Weighted per study, the constrained models beat the baseline only on the full data (1.021 against 1.041). They tie or lose on every reduced or archived version.
- External test (`external.py`, Table IX): 8 AOMI projects absent from NCEI, 1,084 surface-net samples. No trained model beats the study-weighted NCEI net mean (MAE 0.769); baseline LightGBM is the best model (0.872, slightly better per study). The protocol-based models are worse than either constant (1.42 to 1.49) because the constrained model steps at the SEA 0.335 mm mesh. Study CQR covers 94.3%. The test has no bulk-water samples. NCEI's live service on 2026-11-23 matched our export exactly (96 accessions), so it adds no new studies. The live AOMI portal needs browser downloads under its own terms.
- A bootstrap over the 72 studies separates none of the R2 gains from zero (constrained vs. baseline -0.047 to 0.198). Only the study-MAE drop from Huber plus monotonic over shallow trees is clear (-0.142 to -0.011). Paired tests over the 10 assignments remain invalid.
- Most of the two-part gain on detected samples comes from modeling zeros separately (all-feature two-part 0.708 against 0.493). The constraints add 0.708 to 0.815.
- The censored models are scored with one mixed likelihood, because on average 34.6% of detected samples sit below the estimated detection limit, where a Tobit likelihood is undefined. Two-part Physics wins on likelihood and level. Tobit wins only on detection AUC.
- Organization-grouped scores are means over 10 assignments. The single split overstated the baseline (0.405 against 0.282).
- The drifter term adds 0.009 globally and nothing on the ln(c+1) scale. Its value is inside surface-net studies.
- Compare with Dalvi et al. only on their ln(c+1) target (`v2_dalvi_scale.csv`), never on log10.

Gourab's separate notebooks (not in the repo) reach about 0.8 R2 with a random 80:20 split, a `log1p` target and beach rows. His study-grouped scores are -0.16 to -0.39. His data lacks the unit fix.

Unit errors in Barrows (full OSF preprint) and Dehm (Europe PMC abstract) are verified against primary sources. NCEI has not corrected the archive. Arteaga (33 bongo-net rows) remains unverified.

The 2023 Environmental Pollution paper is cited as Zhen et al. The publisher's Crossref deposit gives the first author as given name Yu, family name Zhen. OpenAlex reverses it, but the publisher record wins.

Arnab wants results explained in plain language with numbers.

Gourab sent a ChatGPT/Codex revision of commit cb24722 on 2026-11-23 (PDF plus reports, kept outside the repo). We kept this repo's paper as the base and adopted its verified fixes (Zhen naming, Chan values, Dehm urban/rural means, archived MAE nuance, per-study robustness). We rejected its "proposed rescaling" framing, its in-manuscript placeholders, and its last-digit changes, which come from a different package environment.

## Open items

- Arnab does not want the GitHub repo named in the paper. The availability statement says "available from the authors on request".

- Fill the NCEI download date in `paper/references.bib` (`[DOWNLOAD DATE PENDING]`, needs Gourab).
- Pick the target journal. Gemini suggested IEEE Access or IEEE Journal of Oceanic Engineering.
- Confirm the framing and paper text with Gourab.
- Gourab needs R with `mgcv` installed to run the notebook.
- Full `uv sync` stalls on the 291 MiB `nvidia-nccl-cu13` wheel pulled by `xgboost`. Use `xgboost-cpu` or the existing `.venv/bin/python` for CPU runs.
