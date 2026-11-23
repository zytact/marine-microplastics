# Marine microplastic concentration prediction

Semester project by Arnab Chakraborty and Gourab Adhikari. `microplastics.ipynb` predicts microplastic
concentration in sea water from the NOAA NCEI Marine Microplastics database. It compares nine models
under random, study-grouped and chronological validation, tunes CatBoost with a genetic algorithm and
explains the result with SHAP. The data is the NCEI export in `data/noaa_microplastics.csv`.

Section 5.1 tests environmental covariates. `fetch_environment.py` downloads their sources into
`data/environment`. All were downloaded on 2026-11-10.

| File | Dataset | Source | Licence |
| --- | --- | --- | --- |
| `ne_10m_coastline.zip` (3.1 MB) | Natural Earth 1:10m coastline, v5.0.0 | https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip | Public domain |
| `ne_10m_populated_places_simple.zip` (0.7 MB) | Natural Earth 1:10m populated places, v5.1.2 | https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places_simple.zip | Public domain |
| `Meijer2021_midpoint_emissions.zip` (0.7 MB) | Meijer et al. (2021), riverine plastic emission per outlet, midpoint scenario, Sci. Adv. 7, eaaz5803 | https://doi.org/10.6084/m9.figshare.14515590.v1 | CC BY 4.0 |
| `sst.ltm.1991-2020.nc` (3.4 MB) | NOAA OI SST V2 monthly long-term mean 1991-2020, 1°, from NOAA PSL | https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/sst.ltm.1991-2020.nc | None stated. PSL asks for the acknowledgement "data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov" |
| `drifter_currents_1deg.csv` (0.9 MB) | Global Drifter Program annual mean near-surface velocity, v3.10, Laurindo et al. (2017), doi:10.1016/j.dsr.2017.04.009. Derived: averaged from 0.25° to 1° by `fetch_environment.py` | https://erddap.aoml.noaa.gov/gdp/erddap/griddap/drifter_annualmeans | CC BY 4.0 |

The external test in `external.py` uses `data/aomi/Level012R1.csv`, the Level 0 to 2 file of Isobe et al. (2021), Microplastics and Nanoplastics 1:16, doi:10.1186/s43591-021-00013-z, Additional file 3 (https://static-content.springer.com/esm/art%3A10.1186%2Fs43591-021-00013-z/MediaObjects/43591_2021_13_MOESM3_ESM.zip), CC BY 4.0, downloaded on 2026-11-23.

## Running

The GAMM runs in R through `gamm.R`, so R with the `mgcv` package must be on the path. On Fedora,
`sudo dnf install R-core` includes it.

```sh
uv sync
uv run jupyter lab microplastics.ipynb
```

No GPU is needed. A full run takes about an hour on 4 cores, most of it the GA and random searches. Their
results are cached in `artifacts/` and reused while the search configuration is unchanged, so later runs
take about 20 minutes, half of it the environmental covariate tests in section 5.1.

`microplastics.py` is the same notebook in jupytext percent format, paired with the `.ipynb`. Edit
either one and jupytext keeps them in sync. It is also easier to read in a diff.

## Paper

`paper/main.tex` is the IEEE journal draft. Its numbers come from `experiments.py` and `experiments_v2.py`,
which share data loading and models through `core.py` and write CSVs to `artifacts/`. `paper/figures.py`
draws the figures from those CSVs. `docs/replication.md` section 14 lists the commands and the values each
output should match.

```sh
.venv/bin/python experiments_v2.py
.venv/bin/python experiments.py
.venv/bin/python paper/figures.py
cd paper && tectonic -X compile main.tex
```
