# Progress

Updated 2026-11-22.

## Where things live

- Paper (IEEE journal draft): `paper/main.tex`, built with `tectonic`. Figures come from `paper/figures.py`, which reads `artifacts/`. It replaces the old markdown report and v2 write-up.
- Paper experiments: `experiments.py` (baselines, sensitivity, intervals, flags) and `experiments_v2.py` (diagnosis, constrained models, Dalvi-scale LOOO, surface nets, censored models, robustness). Both import `core.py`. Run commands and checksums: `docs/replication.md` section 14.
- Notebook `microplastics.py`: the original exploratory audit. The paper no longer quotes it (its GAMM needs R, which is not installed here). It still uses `StandardScaler` and 52 raw organization names. `environment.py` ports its environmental covariates for `experiments.py`.
- Literature audit: `docs/literature-review.md`.

## Settled

Every paper score is a mean over 10 study-fold assignments. Single-split numbers from before 2026-11-22 (0.473, 0.487, 5.08 and similar) are retired.

Paper framing: an audit. Random splits overstate transfer by about 0.31 R2, location and time transfer nothing (-0.022), and protocol carries what transfers. Gemini 4 Neon reviewed the paper five times on 2026-11-22, and the fifth pass found nothing substantive. All findings are addressed. Two code nits were left alone, and the third review judged both acceptable: the drift field's Gaussian smoothing does not wrap at the antimeridian (changes 68 of 16,988 values by more than 0.01), and the study-weighted conformal quantile has no finite-sample correction. Paired significance tests over the 10 correlated fold assignments were judged invalid and are not reported. A non-NCEI test set is stated as a limitation.

Findings the paper must keep stating, because they cut against the headline:
- Protocol's advantage over location and time mostly comes from Barrows and Dehm. Without them protocol-only scores 0.100 against 0.055.
- The constrained models lose to the baseline once Barrows, Dehm and the bongo study are all removed (0.300 against 0.397). A minimum leaf size of 10 does not fix it. Without the bulk-water studies, protocol carries little (protocol-only scores -0.085 by organization on that subset), and location and time help the baseline separate the remaining net studies.
- The censored models are scored with one mixed likelihood, because 32% to 39% of detected samples sit below the estimated detection limit, where a Tobit likelihood is undefined. Two-part Physics wins on likelihood and level. Tobit wins only on detection AUC.
- Organization-grouped scores are means over 10 assignments. The single split overstated the baseline (0.405 against 0.282).
- The drifter term adds 0.009 globally and nothing on the ln(c+1) scale. Its value is inside surface-net studies.
- Compare with Dalvi et al. only on their ln(c+1) target (`v2_dalvi_scale.csv`), never on log10.

Gourab's separate notebooks (not in the repo) reach about 0.8 R2 with a random 80:20 split, a `log1p` target and beach rows. His study-grouped scores are -0.16 to -0.39. His data lacks the unit fix.

Unit errors in Barrows (full OSF preprint) and Dehm (Europe PMC abstract) are verified against primary sources. NCEI has not corrected the archive. Arteaga (33 bongo-net rows) remains unverified.

The 2023 Environmental Pollution paper is cited as Yu et al. OpenAlex gives the first author as Zhen Tao Yu. Crossref splits the name the other way round.

Arnab wants results explained in plain language with numbers.

## Open items

- Fill the NCEI download date in `paper/references.bib` (`[DOWNLOAD DATE PENDING]`, needs Gourab).
- Pick the target journal. Gemini suggested IEEE Access or IEEE Journal of Oceanic Engineering.
- Confirm the framing and paper text with Gourab.
- Gourab needs R with `mgcv` installed to run the notebook.
- Full `uv sync` stalls on the 291 MiB `nvidia-nccl-cu13` wheel pulled by `xgboost`. Use `xgboost-cpu` or the existing `.venv/bin/python` for CPU runs.
