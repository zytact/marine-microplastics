from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from core import (
    ARTIFACTS,
    DATA,
    Predictor,
    load_data,
    score,
    study_constant_oracle,
    with_features,
)
from experiments import interval_split, interval_summary
from experiments_v2 import BASELINE, CONSTRAINED, MODELS, PHYSICS

AOMI_DATA = Path("data/aomi/Level012R1.csv")
AOMI_COLUMNS = ["year", "month", "day", "lon", "lat", "conc", "fragment_ratio", "level1", "wind", "wave_height",
                "level2p", "level2w1", "level2w2", "project"]


class AomiProject(NamedTuple):
    source: str
    method: str
    mesh_mm: float
    measured_per_m3: bool


AOMI_PROJECTS_NOT_IN_NCEI = {
    2: AomiProject("T/V Umitaka, Japan (unpublished)", "neuston net", 0.35, True),
    3: AomiProject("Ministry of the Environment, Japan (unpublished)", "neuston net", 0.35, True),
    5: AomiProject("Cozar et al. 2014", "neuston net", 0.2, False),
    18: AomiProject("Zhang et al. (Bohai Sea)", "manta net", 0.33, True),
    19: AomiProject("Zhao et al. (East China Sea)", "neuston net", 0.333, True),
    22: AomiProject("Indonesian Institute of Sciences (unpublished)", "neuston net", 0.35, True),
    23: AomiProject("Ifremer (unpublished)", "manta net", 0.3, True),
    24: AomiProject("Pacific Geographical Institute and Maritime State Univ. (unpublished)", "neuston net", 0.1, True),
}

AOMI_PROJECTS_IN_NCEI = {
    1: "Law et al. 2014", 4: "Collignon et al. 2014", 6: "Cózar et al. 2015", 7: "Cózar et al. 2017",
    8: "Doyle et al. 2011", 10: "Goldstein et al. 2013", 11: "de Lucia et al. 2014", 12: "Lusher et al. 2015",
    13: "Lusher et al. 2014", 14: "Pan et al. 2019", 15: "Pedrotti et al. 2016", 16: "Reisser et al. 2013",
    17: "Suaria et al. 2016", 20: "Law et al. 2010", 25: "Kanhai et al. 2017", 26: "Yakushev et al. 2021",
    27: "Kanhai et al. 2018",
}

EXTERNAL_MODELS = [BASELINE, "LightGBM, protocol only", "LightGBM, location and time only", CONSTRAINED, PHYSICS]


def read_aomi() -> pd.DataFrame:
    return pd.read_csv(AOMI_DATA, skiprows=[1], names=AOMI_COLUMNS, header=0)


def shared_study_medians() -> pd.DataFrame:
    aomi = read_aomi()
    raw = pd.read_csv(DATA, low_memory=False)
    ncei = raw[(raw["Marine Setting"] == "Ocean water") & (raw.Unit == "pieces/m3")]
    rows = []
    for project, reference in AOMI_PROJECTS_IN_NCEI.items():
        a = aomi.loc[aomi.project == project, "conc"]
        n = ncei.loc[ncei["Short Reference"] == reference, "Microplastics Measurement"]
        rows.append({"project": project, "ncei_reference": reference, "aomi_rows": len(a), "ncei_rows": len(n),
                     "aomi_median": a.median(), "ncei_median": n.median()})
    res = pd.DataFrame(rows)
    res["median_ratio"] = (res.aomi_median / res.ncei_median).where(res.ncei_median > 0)
    return res


def load_aomi() -> pd.DataFrame:
    raw = read_aomi()
    raw = raw[raw.project.isin(AOMI_PROJECTS_NOT_IN_NCEI)].reset_index(drop=True)
    project = raw.project.map(AOMI_PROJECTS_NOT_IN_NCEI)
    label = "AOMI project " + raw.project.astype(str)
    return with_features(pd.DataFrame({
        "date": pd.to_datetime(raw[["year", "month", "day"]]),
        "lat": raw.lat,
        "lon": raw.lon,
        "depth": 0.0,
        "method": project.map(lambda p: p.method),
        "mesh": project.map(lambda p: p.mesh_mm),
        "conc": raw.conc,
        "doi": label,
        "accession": label,
        "organization": project.map(lambda p: p.source),
        "measured_per_m3": project.map(lambda p: p.measured_per_m3),
    }))


def surface_net_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), train.loc[train.method_family == "surface net", "y"].mean())


def surface_net_study_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    nets = train[train.method_family == "surface net"]
    return np.full(len(test), nets.groupby("study").y.mean().mean())


def external_scores(test: pd.DataFrame, predictions: dict[str, np.ndarray], subset: str) -> list[dict[str, float | str]]:
    y = test.y.to_numpy()
    oracle = study_constant_oracle(test.y, test.study)
    rows = [{"model": name, **score(y, pred, test.study)} for name, pred in predictions.items()]
    rows.append({"model": "Study-constant oracle", **oracle})
    return [{"subset": subset, "y_variance": y.var(ddof=1), **row} for row in rows]


def main() -> None:
    ncei, aomi = load_data(), load_aomi()
    print(f"Train: {len(ncei):,} NCEI rows. Test: {len(aomi):,} AOMI rows from {aomi.study.nunique()} projects not in NCEI")
    predictors: dict[str, Predictor] = {"NCEI surface-net mean, per sample": surface_net_mean,
                                         "NCEI surface-net mean, per study": surface_net_study_mean} | {m: MODELS[m] for m in EXTERNAL_MODELS}
    predictions = {name: predict(ncei, aomi) for name, predict in predictors.items()}

    measured = aomi.measured_per_m3.to_numpy()
    scores = pd.DataFrame(
        external_scores(aomi, predictions, "All 8 projects")
        + external_scores(aomi[measured], {k: v[measured] for k, v in predictions.items()}, "7 projects measured per m3")
    )
    per_project = pd.DataFrame([
        {"project": study, "source": group.organization.iat[0], "rows": len(group), "mean_y": group.y.mean(),
         **{f"bias {name}": predictions[name][group.index].mean() - group.y.mean() for name in predictors}}
        for study, group in aomi.groupby("study")
    ])

    combined = pd.concat([ncei, aomi], ignore_index=True)
    test_idx = np.arange(len(ncei), len(combined))
    split = interval_split(combined, [(np.arange(len(ncei)), test_idx)]).iloc[test_idx].reset_index(drop=True)
    intervals = interval_summary(aomi, split)
    intervals["half_width_factor_x"] = 10 ** (intervals.median_width_log10 / 2)

    shared = shared_study_medians()
    shared.to_csv(ARTIFACTS / "external_aomi_shared_studies.csv", index=False)
    scores.to_csv(ARTIFACTS / "external_aomi_scores.csv", index=False)
    per_project.to_csv(ARTIFACTS / "external_aomi_projects.csv", index=False)
    intervals.to_csv(ARTIFACTS / "external_aomi_intervals.csv", index=False)
    for frame in (shared, scores, per_project, intervals):
        print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    import external

    external.main()
