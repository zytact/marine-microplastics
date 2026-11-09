# Progress

Updated 2026-11-10.

## State

Branch `feat/environmental-features` adds section 5.1 and is not merged into `main`. Arnab has not asked for a merge or a push.

## Settled

Environmental covariates stay out of the main models. Section 5.1 tested coast distance, river mouth distance, river plastic input, population, SST climatology and current speed. Over 10 random study fold assignments the grouped R2 gain averaged -0.024 for CatBoost and -0.005 for LightGBM.

The grouped pooled R2 in section 4 is one fold assignment. Across the 10 assignments it ranges from 0.38 to 0.55, so quote that range beside the 0.44 and 0.47.

## Pending decisions

Arnab wants results explained in plain language with numbers. He asked for an accuracy-style percentage but has not asked to change the regression task to classification.

Untested ideas for better predictions: fields that vary by year such as SST and wind at the sampling date, comparable-net subsets as the main model, measurement harmonisation, concentration categories.

## Data questions

- `data/noaa_microplastics.csv` came from Gourab's NCEI download. Its download date is unknown and is needed to cite it.
- The bongo net rows (Arteaga et al. 2024) have a median of 2,860 pieces/m3, far above other nets. The abstract mixes units and the full text is paywalled, so they are unverified and left unchanged.
- Gourab needs R with `mgcv` installed to run the notebook.

## Unresolved review notes

- The earlier chat recommendation to harmonise measurements using "Koelmans 2020" was unverified. Check the citation and required metadata before pursuing it.
