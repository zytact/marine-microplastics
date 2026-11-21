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

Publication framing given to Arnab: a validation study (study-grouped testing, inflated random-split scores, sampling method as the transferable signal, the NCEI unit errors, the 3-sample calibration result). NCEI has not confirmed the unit errors.

Arnab wants results explained in plain language with numbers.

## Open items

- Get Gourab's NCEI download date for `data/noaa_microplastics.csv` and replace `[DOWNLOAD DATE PENDING]` in `docs/report-draft.md`.
- Confirm the Option A framing and report text with Gourab before final submission formatting.
- Gourab needs R with `mgcv` installed to run the notebook.
- Full `uv sync` stalls on the 291 MiB `nvidia-nccl-cu13` wheel pulled by `xgboost`. Use `xgboost-cpu` or the existing `.venv/bin/python` for CPU runs.
