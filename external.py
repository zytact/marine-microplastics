import io
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from core import (
    ARTIFACTS,
    DATA,
    METHOD_FAMILY,
    Predictor,
    load_data,
    score,
    study_constant_oracle,
    with_features,
)
from experiments import interval_split, interval_summary
from experiments_v2 import BASELINE, CONSTRAINED, MODELS, PHYSICS

AOMI_DOWNLOAD = Path("data/aomi/aomi_v4.0_survey_points_2026-11-23.zip")
AOMI_COLUMNS = {
    "Organization": "organization",
    "DOI": "doi",
    "Source of Data Collection": "source",
    "Creative commons license": "licence",
    "Sampling start Date": "date",
    "GPS_Lat Int": "lat",
    "GPS_Lon Int": "lon",
    "Equipment Type": "equipment",
    "Mesh openings (mm)": "mesh",
    "Lower end depth (m)": "net_depth",
    "Particle density_m3 (d<5mm)": "conc",
}
AOMI_ORGANIZATION_FIXES = {"Nihonkai environmental seavice Inc.": "Nihonkai environmental service Inc."}
AOMI_METHODS = {
    "neuston net": "neuston net",
    "manta net": "manta net",
    "plankton nets(22)": "plankton net",
    "plankton net": "plankton net",
    "nets except manta and neuston": "plankton net",
    "avani net": "avani net",
    "pump sampling": "pump",
    "stainless steel bucket": "stainless steel bucket",
}
PANGAEA_DIR = Path("data/pangaea")


class PangaeaDataset(NamedTuple):
    source: str
    method: str
    mesh_mm: float
    excluded: tuple[str, str] | None = None
    concentrations: tuple[str, ...] = ("Microplastic [#/m**3]",)


PANGAEA_DATASETS = {
    "941076": PangaeaDataset("Leistenschneider et al., Weddell Sea surface", "manta net", 0.3),
    "941107": PangaeaDataset("Leistenschneider et al., Weddell Sea subsurface", "intake seawater pump", 0.3,
                             ("Method comm", "ballast")),
    "937685": PangaeaDataset("Roscher et al., Weser estuary and German Bight", "pump", 0.01, ("Location", "Middle Weser")),
    "990052": PangaeaDataset("Reineccius et al., eastern Pacific shelf", "intake seawater pump", 0.01,
                             concentrations=("Microplastic [#/m**3] (Fourier", "Microplastic [#/m**3] (Counted")),
}
ZHAO_FILE = Path("data/zhao2025/Dataset of Submeraged Microplastics from a single depth (1-60 m).xlsx")
ZHAO_COLUMNS = ["reference", "lat", "lon", "conc", "mesh", "fraction_above_250", "non_fiber", "method", "corrected",
                "size_fraction", "applied"]
ZHAO_METHODS = {"Ship's underway pump": "intake seawater pump", "Niskin bottles": "niskin bottle", "MultiNet": "plankton net"}
ZHAO_REFERENCES_IN_NCEI = {"Lusher et al. 2014", "Lusher et al. 2015", "Kanhai et al. 2017", "Kanhai et al. 2018",
                           "Yakushev et al. 2021"}
SAMPLING_TO_PUBLICATION_YEARS = 2
ZHAO_PUMP_DEPTH_SENSITIVITY = (1.0, 10.0)
GREAT_LAKES = {"lat": (41.0, 49.5), "lon": (-93.0, -75.5)}
SAME_SAMPLE_KM = 15.0
SAME_SAMPLE_DAYS = 31
EXTERNAL_MODELS = [BASELINE, "LightGBM, protocol only", "LightGBM, location and time only", CONSTRAINED, PHYSICS]


def normalized_doi(doi: pd.Series) -> pd.Series:
    return doi.str.lower().str.strip().str.replace(r"^(https?://)?(dx\.)?doi\.org/", "", regex=True)


def read_aomi() -> pd.DataFrame:
    with zipfile.ZipFile(AOMI_DOWNLOAD) as archive:
        text = archive.read("survey_data.csv").decode("utf-8-sig")
    table = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    raw = pd.read_csv(io.StringIO(table), low_memory=False)
    raw.columns = [c.split(":", 1)[1].strip() for c in raw.columns]
    aomi = raw[list(AOMI_COLUMNS)].rename(columns=AOMI_COLUMNS)
    lat, lon = pd.to_numeric(aomi.lat, errors="coerce"), pd.to_numeric(aomi.lon, errors="coerce")
    swapped = (lat.abs() > 90) & (lon.abs() <= 90)
    organization = aomi.organization.replace(AOMI_ORGANIZATION_FIXES)
    return aomi.assign(
        date=pd.to_datetime(aomi.date, errors="coerce"),
        lat=lat.where(~swapped, lon),
        lon=lon.where(~swapped, lat),
        coordinates_swapped=swapped,
        mesh=pd.to_numeric(aomi.mesh, errors="coerce"),
        net_depth=pd.to_numeric(aomi.net_depth, errors="coerce"),
        conc=pd.to_numeric(aomi.conc, errors="coerce"),
        organization=organization,
        doi_key=normalized_doi(aomi.doi.astype("string")),
        group=organization.fillna("unknown organization") + " | " + aomi.doi.fillna("no DOI"),
    ).pipe(lambda frame: frame.assign(survey_year=frame.date.dt.year.astype("Int64")))


def read_ncei_water() -> pd.DataFrame:
    raw = pd.read_csv(DATA, low_memory=False)
    water = raw[(raw["Marine Setting"] == "Ocean water") & (raw.Unit == "pieces/m3")]
    return pd.DataFrame({
        "date": pd.to_datetime(water["Sample Date"], format="%m/%d/%Y %I:%M:%S %p").dt.normalize(),
        "lat": water["Latitude (degree)"],
        "lon": water["Longitude (degree)"],
        "conc": water["Microplastics Measurement"],
        "doi_key": normalized_doi(water.DOI.astype("string")),
    })


def valid_position(frame: pd.DataFrame) -> pd.Series:
    return frame.lat.between(-90, 90) & frame.lon.between(-180, 180)


def same_sample_in_ncei(aomi: pd.DataFrame, ncei: pd.DataFrame) -> np.ndarray:
    located = (valid_position(aomi) & aomi.date.notna()).to_numpy()
    tree = BallTree(np.radians(ncei[["lat", "lon"]].to_numpy()), metric="haversine")
    nearby = tree.query_radius(np.radians(aomi.loc[located, ["lat", "lon"]].to_numpy()), r=SAME_SAMPLE_KM / 6371.0)
    ncei_dates = ncei.date.to_numpy()
    window = np.timedelta64(SAME_SAMPLE_DAYS, "D")
    match = np.zeros(len(aomi), dtype=bool)
    match[located] = [
        bool(len(idx)) and bool((np.abs(ncei_dates[idx] - date) <= window).any())
        for idx, date in zip(nearby, aomi.loc[located, "date"].dt.normalize().to_numpy(), strict=True)
    ]
    return match


def ncei_overlap(aomi: pd.DataFrame, ncei: pd.DataFrame) -> pd.DataFrame:
    flags = aomi.assign(
        from_ncei=aomi.source.fillna("").str.startswith("NOAA"),
        doi_in_ncei=aomi.doi_key.isin(set(ncei.doi_key.dropna())).fillna(False),
        same_sample=same_sample_in_ncei(aomi, ncei),
    )
    units = flags.groupby(["group", "survey_year"], dropna=False).agg(
        rows=("group", "size"), from_ncei=("from_ncei", "mean"), doi_in_ncei=("doi_in_ncei", "mean"),
        same_sample=("same_sample", "mean"),
    )
    units["in_ncei"] = (units.from_ncei > 0.5) | (units.doi_in_ncei > 0.5) | (units.same_sample > 0.2)
    return units.reset_index()


def in_great_lakes(frame: pd.DataFrame) -> pd.Series:
    return frame.lat.between(*GREAT_LAKES["lat"]) & frame.lon.between(*GREAT_LAKES["lon"])


def load_aomi_external(aomi: pd.DataFrame, overlap: pd.DataFrame, recorded_depth: bool = False) -> pd.DataFrame:
    lake_groups = set(aomi.groupby("group").filter(lambda g: in_great_lakes(g).mean() > 0.5).group)
    new_units = overlap.loc[~overlap.in_ncei & ~overlap.group.isin(lake_groups), ["group", "survey_year"]]
    is_new = aomi.merge(new_units.assign(new=True), on=["group", "survey_year"], how="left").new.fillna(False).to_numpy()
    method = aomi.equipment.str.lower().str.strip().map(AOMI_METHODS)
    usable = is_new & method.notna() & aomi.conc.notna() & aomi.date.notna() & valid_position(aomi)
    external = aomi[usable].reset_index(drop=True)
    return with_features(external.assign(
        method=method[usable].to_numpy(),
        depth=external.net_depth.fillna(0.0) if recorded_depth else 0.0,
        accession=external.group,
        doi=external.group,
        archive="AOMI",
    ))


def read_pangaea(dataset_id: str) -> pd.DataFrame:
    text = (PANGAEA_DIR / f"PANGAEA.{dataset_id}.tab").read_text(encoding="utf-8")
    return pd.read_csv(io.StringIO(text.split("*/\n", 1)[1]), sep="\t")


def load_pangaea_external() -> pd.DataFrame:
    frames = []
    for dataset_id, dataset in PANGAEA_DATASETS.items():
        raw = read_pangaea(dataset_id)
        if dataset.excluded:
            column, text = dataset.excluded
            raw = raw[~raw[column].fillna("").str.contains(text)]
        columns = [c for c in raw.columns if c.startswith(dataset.concentrations)]
        assert len(columns) == len(dataset.concentrations), (dataset_id, columns)
        depth = raw["Depth water [m]"] if "Depth water [m]" in raw else pd.Series(0.0, index=raw.index)
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(raw["Date/Time"]),
            "lat": raw.Latitude,
            "lon": raw.Longitude,
            "depth": depth,
            "method": dataset.method,
            "mesh": dataset.mesh_mm,
            "conc": raw[columns].sum(axis=1, min_count=1),
            "doi": f"https://doi.org/10.1594/PANGAEA.{dataset_id}",
            "accession": f"PANGAEA.{dataset_id}",
            "organization": dataset.source,
            "group": dataset.source,
            "archive": "PANGAEA",
        }))
    return with_features(pd.concat(frames, ignore_index=True))


def shared_study_medians(aomi: pd.DataFrame, ncei: pd.DataFrame) -> pd.DataFrame:
    compiled_by_aomi = aomi[~aomi.source.fillna("").str.startswith("NOAA") & aomi.doi_key.notna() & aomi.conc.notna()]
    rows = []
    for doi_key, group in compiled_by_aomi.groupby("doi_key"):
        matched = ncei.loc[ncei.doi_key == doi_key, "conc"]
        if len(matched):
            rows.append({"doi": doi_key, "aomi_rows": len(group), "ncei_rows": len(matched),
                         "aomi_median": group.conc.median(), "ncei_median": matched.median()})
    shared = pd.DataFrame(rows)
    both_positive = (shared.aomi_median > 0) & (shared.ncei_median > 0)
    shared["median_ratio"] = (shared.aomi_median / shared.ncei_median).where(both_positive)
    return shared


def micrometres_to_mm(size: pd.Series) -> pd.Series:
    return pd.to_numeric(size.astype("string").str.extract(r"([\d.]+)\s*um")[0], errors="coerce") / 1000.0


def load_zhao_external(ncei: pd.DataFrame, pump_depth: float | None = None) -> pd.DataFrame:
    raw = pd.read_excel(ZHAO_FILE, header=1, names=ZHAO_COLUMNS)
    raw = raw.assign(conc=pd.to_numeric(raw.conc, errors="coerce"))
    raw = raw[~raw.reference.isin(ZHAO_REFERENCES_IN_NCEI) & raw.conc.notna() & raw.method.isin(ZHAO_METHODS)]
    raw = raw.reset_index(drop=True)
    method = raw.method.map(ZHAO_METHODS)
    family_depth = ncei.groupby("method_family").depth.median()
    family = method.str.lower().map(METHOD_FAMILY).fillna("bulk water")
    publication_year = raw.reference.str.extract(r"(\d{4})")[0].astype(int)
    external = with_features(pd.DataFrame({
        "date": pd.NaT,
        "lat": raw.lat,
        "lon": raw.lon,
        "depth": family.map(family_depth) if pump_depth is None else family.map(family_depth).where(family != "pump", pump_depth),
        "method": method,
        "mesh": micrometres_to_mm(raw.mesh).fillna(micrometres_to_mm(raw.size_fraction)),
        "conc": raw.conc,
        "doi": raw.reference,
        "accession": raw.reference,
        "organization": raw.reference,
        "group": raw.reference,
        "archive": "Zhao et al. compilation",
    }))
    return external.assign(year=publication_year - SAMPLING_TO_PUBLICATION_YEARS)


def family_mean_per_sample(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return test.method_family.map(train.groupby("method_family").y.mean()).to_numpy()


def family_mean_per_study(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    study_means = train.groupby(["method_family", "study"]).y.mean()
    return test.method_family.map(study_means.groupby("method_family").mean()).to_numpy()


def subset_scores(test: pd.DataFrame, predictions: dict[str, np.ndarray], subset: str) -> pd.DataFrame:
    y = test.y.to_numpy()
    rows = [{"model": name, **score(y, pred[test.index], test.study)} for name, pred in predictions.items()]
    rows.append({"model": "Study-constant oracle", **study_constant_oracle(test.y, test.study)})
    return pd.DataFrame(rows).assign(subset=subset, y_variance=y.var(ddof=1), samples=len(test),
                                     groups=test.study.nunique())


def main() -> None:
    aomi, ncei_water = read_aomi(), read_ncei_water()
    overlap = ncei_overlap(aomi, ncei_water)
    ncei = load_data()
    external = pd.concat([load_aomi_external(aomi, overlap), load_pangaea_external(), load_zhao_external(ncei)],
                         ignore_index=True)
    print(f"AOMI v4.0: {len(aomi):,} records in {overlap.group.nunique()} groups and {len(overlap)} group-years, "
          f"{int((~overlap.in_ncei).sum())} group-years absent from NCEI")
    print(f"External test: {len(external):,} samples from {external.study.nunique()} groups")

    predictors: dict[str, Predictor] = {
        "NCEI family mean, per sample": family_mean_per_sample,
        "NCEI family mean, per study": family_mean_per_study,
    } | {m: MODELS[m] for m in EXTERNAL_MODELS}
    predictions = {name: predict(ncei, external) for name, predict in predictors.items()}
    subsets = {"All external": external, "AOMI": external[external.archive == "AOMI"],
               "PANGAEA": external[external.archive == "PANGAEA"],
               "Zhao et al. compilation": external[external.archive == "Zhao et al. compilation"],
               "Pump and bottle": external[external.method_family != "surface net"]}
    recorded = load_aomi_external(aomi, overlap, recorded_depth=True)
    recorded_predictions = {name: predict(ncei, recorded) for name, predict in predictors.items()}
    depth_runs = []
    for depth in ZHAO_PUMP_DEPTH_SENSITIVITY:
        alternative = pd.concat([external[external.archive != "Zhao et al. compilation"],
                                 load_zhao_external(ncei, pump_depth=depth)], ignore_index=True)
        pumps_and_bottles = alternative[alternative.method_family != "surface net"]
        alternative_predictions = {name: predict(ncei, pumps_and_bottles) for name, predict in predictors.items()}
        depth_runs.append(subset_scores(pumps_and_bottles.reset_index(drop=True), alternative_predictions,
                                        f"Pump and bottle, compilation pumps at {depth:g} m"))
    scores = pd.concat(
        [subset_scores(frame, predictions, name) for name, frame in subsets.items()]
        + [subset_scores(recorded, recorded_predictions, "AOMI, recorded net depth")] + depth_runs,
        ignore_index=True,
    )
    per_group = pd.DataFrame([
        {"group": study, "archive": group.archive.iat[0], "rows": len(group), "method": group.method.iat[0],
         "family": group.method_family.iat[0], "median_mesh_mm": group.mesh.median(),
         "first_year": group.year.min(), "last_year": group.year.max(), "mean_y": group.y.mean(),
         **{f"bias {name}": predictions[name][group.index].mean() - group.y.mean() for name in predictors}}
        for study, group in external.groupby("study")
    ])

    combined = pd.concat([ncei, external], ignore_index=True)
    test_idx = np.arange(len(ncei), len(combined))
    split = interval_split(combined, [(np.arange(len(ncei)), test_idx)]).iloc[test_idx].reset_index(drop=True)
    intervals = pd.concat([
        interval_summary(frame.reset_index(drop=True), split.loc[frame.index].reset_index(drop=True)).assign(subset=name)
        for name, frame in subsets.items()
    ], ignore_index=True)
    intervals["half_width_factor_x"] = 10 ** (intervals.median_width_log10 / 2)
    shared = shared_study_medians(aomi, ncei_water)

    outputs = {"external_aomi_overlap": overlap, "external_aomi_shared_studies": shared, "external_scores": scores,
               "external_groups": per_group, "external_intervals": intervals}
    for name, frame in outputs.items():
        frame.to_csv(ARTIFACTS / f"{name}.csv", index=False)
    for frame in (shared, scores, per_group, intervals):
        print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    import external

    external.main()
