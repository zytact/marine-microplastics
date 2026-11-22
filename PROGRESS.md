# Progress

Updated 2026-11-21.

## Where things live

- Notebook sections 5.1 to 5.3 (`microplastics.py`): environmental covariates, few-sample calibration, within-study ranking.
- Option A experiments: `experiments.py`. V2 constrained within-between, drifter accumulation and Tobit experiments: `experiments_v2.py`, `docs/v2-methods-results.md`.
- Literature audit: `docs/literature-review.md`. Replication checksums: `docs/replication.md`. Report draft: `docs/report-draft.md`.

## Settled

Environmental covariates stay out of the main models. Over 10 random study fold assignments their grouped R2 gain averaged -0.024 for CatBoost and -0.005 for LightGBM.

The grouped pooled R2 in notebook section 4 is one fold assignment. Across 10 assignments it ranges from 0.38 to 0.55, so quote that range beside the 0.44 and 0.47.

Gourab's separate notebooks (not in the repo) reach about 0.8 R2 with a random 80:20 split, a `log1p` target and beach rows. His study-grouped scores are -0.16 to -0.39. On our grouped folds, XGBoost scores 0.33 with our setup, 0.12 with `log1p` and -3.65 with `log1p` and no `UNIT_FIXES`. His data lacks the unit fix.

Publication framing: an audit and reliability study (study-grouped and organization-grouped validation, inflated random-split scores, sampling protocol as the transferable signal, calibrated CQR intervals, and fixes for spurious within-study variance and the 6-log bulk-water zero cliff). Most of the v2 global R2 gain comes from constrained monotonic Huber protocol trees (+0.087); Lagrangian drifter accumulation is a secondary gain (+0.009 global, 10/10 on surface nets, 5/5 LOOO basins) and should not be oversold as a standalone physics breakthrough.

Unit errors in Barrows (full OSF preprint) and Dehm (Europe PMC abstract) are verified against primary sources; NOAA NCEI has not updated the archive yet, and sensitivity checks drop both studies. Only Arteaga (33 bongo net rows) remains unverified.

Arnab wants results explained in plain language with numbers.

## Open items

- Get Gourab's NCEI download date for `data/noaa_microplastics.csv` and replace `[DOWNLOAD DATE PENDING]` in `docs/report-draft.md`.
- Confirm the Option A framing and report text with Gourab before final submission formatting.
- Gourab needs R with `mgcv` installed to run the notebook.
- Full `uv sync` stalls on the 291 MiB `nvidia-nccl-cu13` wheel pulled by `xgboost`. Use `xgboost-cpu` or the existing `.venv/bin/python` for CPU runs.
