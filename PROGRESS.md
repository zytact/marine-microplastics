# Progress

Updated 2026-11-21.

## Current status

Option A (steps 1 through 6) is complete in the repo, and v2 methodological fixes are committed on branch `feat/constrained-within-between`:
- Literature audit and paywalled-paper need ratings: `docs/literature-review.md`
- Reproducible Option A experiments (steps 2 to 5) and figures: `experiments.py`, `artifacts/step*.csv`, `artifacts/fig_step*.png`
- V2 constrained Within-Between physics, Lagrangian drifter accumulation, and left-censored Tobit experiments: `experiments_v2.py`, `docs/v2-methods-results.md`, `artifacts/v2_*.csv`, `artifacts/fig_v2_*.png`
- Replication checksums (sections 1 to 17): `docs/replication.md`
- Research-paper-style B.Tech project report draft (updated with v2 fixes): `docs/report-draft.md`

## Open items

- **NCEI download date.** Get Gourab's download date for `data/noaa_microplastics.csv` and replace `[DOWNLOAD DATE PENDING]` in `docs/report-draft.md`.
- **Partner alignment.** Confirm Option A framing and report text with Gourab before final submission formatting (LaTeX/Word template).
- **Environment note.** `.venv` has the locked offline packages (`lightgbm`, `scikit-learn`, `statsmodels`, `matplotlib`). Full `uv sync` stalls on the 291 MiB `nvidia-nccl-cu13` wheel pulled by `xgboost`; use `xgboost-cpu` or `.venv/bin/python` for CPU runs.
