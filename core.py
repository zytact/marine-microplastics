from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.ndimage import gaussian_filter
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree
from sklearn.preprocessing import OneHotEncoder

DATA = Path("data/noaa_microplastics.csv")
DRIFTER_DATA = Path("data/environment/drifter_currents_1deg.csv")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

SEED = 0
OFFSET_C = 1e-3
EARTH_RADIUS_M = 6.371e6
REPEATS = 10

METHOD_FAMILY = {
    "neuston net": "surface net",
    "manta net": "surface net",
    "plankton net": "surface net",
    "avani net": "surface net",
    "bongo net": "surface net",
    "pump": "pump",
    "intake seawater pump": "pump",
    "surface water intake": "pump",
}

ORGANIZATION_ALIASES = {
    "Galway-Mayo Institute of Technology, Ireland": "Galway-Mayo Institute of Technology, Galway, Ireland",
    "Atlantic Technological University, Galway, Ireland": "Galway-Mayo Institute of Technology, Galway, Ireland",
    "Institute of Marine Sciences – National Research Council, Italy": "National Research Council, Italy",
}

BARROWS_DOI = "https://doi.org/10.1016/j.envpol.2018.02.062"
DEHM_DOI = "https://doi.org/10.1016/j.marpolbul.2020.111239"
UNIT_FIXES = {BARROWS_DOI: 1e6, DEHM_DOI: 1e6}

NUMERIC = ["lat", "lon", "year", "month_sin", "month_cos", "log_mesh", "log_depth"]
FEATURES = NUMERIC + ["method_family"]
PROTOCOL_FEATURES = ["log_mesh", "log_depth", "method_family"]
LOCATION_TIME_FEATURES = ["lat", "lon", "year", "month_sin", "month_cos"]
SPACE_TIME_FEATURES = ["lat", "lon", "month_sin", "month_cos"]
OCEANS = ["Indian Ocean", "Pacific Ocean", "Atlantic Ocean", "Arctic Ocean", "Southern Ocean"]

DALVI_OCEAN_R2 = {
    "Indian Ocean": 0.166,
    "Pacific Ocean": 0.082,
    "Atlantic Ocean": 0.010,
    "Arctic Ocean": -0.253,
    "Southern Ocean": -0.708,
}

Folds = list[tuple[np.ndarray, np.ndarray]]
Predictor = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def linked_groups(first: pd.Series, second: pd.Series) -> pd.Series:
    group = first.astype(str)
    while not (merged := group.groupby(second).transform("min").groupby(first).transform("min")).equals(group):
        group = merged
    return group


def lagrangian_accumulation_field() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drifters = pd.read_csv(DRIFTER_DATA)
    u_grid = np.zeros((180, 360))
    v_grid = np.zeros((180, 360))
    ocean_mask = np.zeros((180, 360), dtype=bool)

    lat_idx = np.round(drifters.lat.to_numpy() + 89.5).astype(int)
    lon_idx = np.round(drifters.lon.to_numpy() + 179.5).astype(int) % 360
    u_grid[lat_idx, lon_idx] = drifters.u.to_numpy()
    v_grid[lat_idx, lon_idx] = drifters.v.to_numpy()
    ocean_mask[lat_idx, lon_idx] = True

    rng = np.random.default_rng(SEED)
    ocean_y, ocean_x = np.nonzero(ocean_mask)
    particle_count = 80_000
    sampled = rng.integers(0, len(ocean_y), size=particle_count)
    py = ocean_y[sampled].astype(float) + rng.random(particle_count) - 0.5
    px = ocean_x[sampled].astype(float) + rng.random(particle_count) - 0.5

    dt_seconds = 5.0 * 86400.0
    cell_rad = np.radians(1.0)
    for _ in range(365):
        iy = np.clip(np.round(py).astype(int), 0, 179)
        ix = np.round(px).astype(int) % 360
        cos_lat = np.maximum(np.cos(np.radians(py - 89.5)), 0.15)
        dy = v_grid[iy, ix] * dt_seconds / (EARTH_RADIUS_M * cell_rad) + rng.normal(0.0, 0.15, particle_count)
        dx = u_grid[iy, ix] * dt_seconds / (EARTH_RADIUS_M * cos_lat * cell_rad) + rng.normal(0.0, 0.15, particle_count)
        next_y = np.clip(py + dy, 0.0, 179.0)
        next_x = (px + dx) % 360.0
        stays_in_ocean = ocean_mask[np.clip(np.round(next_y).astype(int), 0, 179), np.round(next_x).astype(int) % 360]
        py = np.where(stays_in_ocean, next_y, py)
        px = np.where(stays_in_ocean, next_x, px)

    density = np.zeros((180, 360))
    np.add.at(density, (np.clip(np.round(py).astype(int), 0, 179), np.round(px).astype(int) % 360), 1.0)
    return ocean_y, ocean_x, np.log1p(gaussian_filter(density, sigma=2.5)), np.hypot(u_grid, v_grid)


def load_data() -> pd.DataFrame:
    raw = pd.read_csv(DATA, low_memory=False)
    data = raw.rename(columns={
        "Sample Date": "date",
        "Latitude (degree)": "lat",
        "Longitude (degree)": "lon",
        "Ocean": "ocean",
        "Marine Setting": "medium",
        "Water Sample Depth (m)": "depth",
        "Sampling Method": "method",
        "Mesh Size (mm)": "mesh",
        "Microplastics Measurement": "conc",
        "Unit": "unit",
        "NCEI Accession Number": "accession",
        "DOI": "doi",
        "Organization": "organization",
    })[["date", "lat", "lon", "ocean", "medium", "depth", "method", "mesh", "conc", "unit", "accession", "doi", "organization"]]
    data["date"] = pd.to_datetime(data.date, format="%m/%d/%Y %I:%M:%S %p")
    water = data[(data.medium == "Ocean water") & (data.unit == "pieces/m3")].reset_index(drop=True)
    water["conc"] *= water.doi.map(UNIT_FIXES).fillna(1.0)
    method = water.method.str.lower().str.replace("-", " ")
    month = water.date.dt.month

    ocean_y, ocean_x, accumulation, speed = lagrangian_accumulation_field()
    ocean_cells = np.radians(np.column_stack([ocean_y - 89.5, ocean_x - 179.5]))
    nearest = BallTree(ocean_cells, metric="haversine").query(np.radians(water[["lat", "lon"]].to_numpy()))[1][:, 0]

    return water.assign(
        organization=water.organization.replace(ORGANIZATION_ALIASES),
        method=method,
        method_family=method.map(METHOD_FAMILY).fillna("bulk water"),
        year=water.date.dt.year,
        month_sin=np.sin(2 * np.pi * month / 12),
        month_cos=np.cos(2 * np.pi * month / 12),
        log_mesh=np.log10(water.mesh),
        log_depth=np.log1p(water.depth),
        study=linked_groups(water.doi, water.accession),
        y=np.log10(water.conc + OFFSET_C),
        lag_acc=accumulation[ocean_y[nearest], ocean_x[nearest]],
        drifter_speed=speed[ocean_y[nearest], ocean_x[nearest]],
        abs_lat=np.abs(water.lat),
    )


def design_matrices(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    num_cols = [c for c in cols if c != "method_family"]
    cat_cols = [c for c in cols if c == "method_family"]
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    ct = ColumnTransformer(transformers)
    x_tr = ct.fit_transform(train[cols])
    x_te = ct.transform(test[cols])
    names = list(num_cols)
    if cat_cols:
        names += ct.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    return x_tr, x_te, names


def score(y: np.ndarray, pred: np.ndarray, studies: pd.Series) -> dict[str, float]:
    err = pred - y
    abs_err = np.abs(err)
    return {
        "R2": float(1.0 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "study_MAE": float(pd.Series(abs_err).groupby(studies.to_numpy()).mean().mean()),
        "within_1_log": float(np.mean(abs_err <= 1.0)),
        "miss_gt_2_log": float(np.mean(abs_err > 2.0)),
    }


def study_constant_oracle(y: pd.Series, studies: pd.Series) -> dict[str, float]:
    by_mean = score(y.to_numpy(), y.groupby(studies).transform("mean").to_numpy(), studies)
    by_median = score(y.to_numpy(), y.groupby(studies).transform("median").to_numpy(), studies)
    return {"R2": by_mean["R2"], **{k: v for k, v in by_median.items() if k != "R2"}}


def repeated_folds(df: pd.DataFrame, groups: str = "study", repeats: int = REPEATS) -> list[Folds]:
    return [list(GroupKFold(5, shuffle=True, random_state=r).split(df, groups=df[groups])) for r in range(repeats)]


def out_of_fold(df: pd.DataFrame, folds: Folds, predict: Predictor) -> np.ndarray:
    pred = np.empty(len(df))
    for tr_idx, te_idx in folds:
        pred[te_idx] = predict(df.iloc[tr_idx], df.iloc[te_idx])
    return pred


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / np.sum(weights)
    return float(values[order][min(np.searchsorted(cum, q, side="left"), len(values) - 1)])


def lgbm(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], target: np.ndarray | None = None, **params) -> np.ndarray:
    x_tr, x_te, _ = design_matrices(train, test, cols)
    defaults = dict(n_estimators=500, learning_rate=0.05, random_state=SEED, n_jobs=4, verbose=-1)
    y = train.y.to_numpy() if target is None else target
    return LGBMRegressor(**(defaults | params)).fit(x_tr, y).predict(x_te)


def predict_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return lgbm(train, test, FEATURES)


def predict_protocol_only(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return lgbm(train, test, PROTOCOL_FEATURES)


def predict_location_time_only(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return lgbm(train, test, LOCATION_TIME_FEATURES)


SHALLOW = dict(n_estimators=150, max_depth=3, num_leaves=7, min_child_samples=30)


def shallow_protocol(train: pd.DataFrame, test: pd.DataFrame, target: np.ndarray | None = None, cols: list[str] = PROTOCOL_FEATURES,
                     huber: bool = True, monotone_mesh: bool = True, **params) -> np.ndarray:
    x_tr, x_te, names = design_matrices(train, test, cols)
    settings = dict(learning_rate=0.05, random_state=SEED, n_jobs=4, verbose=-1) | SHALLOW
    if huber:
        settings["objective"] = "huber"
    if monotone_mesh:
        settings["monotone_constraints"] = [-1 if n == "log_mesh" else 0 for n in names]
    y = train.y.to_numpy() if target is None else target
    return LGBMRegressor(**(settings | params)).fit(x_tr, y).predict(x_te)


def predict_constrained_protocol(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return shallow_protocol(train, test)


def predict_interaction_constrained(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_tr, x_te, names = design_matrices(train, test, FEATURES)
    protocol = [i for i, n in enumerate(names) if n in ("log_mesh", "log_depth") or n.startswith("method_family_")]
    space_time = [i for i in range(len(names)) if i not in protocol]
    model = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, num_leaves=7, min_child_samples=50, objective="huber",
        interaction_constraints=[protocol, space_time], monotone_constraints=[-1 if n == "log_mesh" else 0 for n in names],
        random_state=SEED, n_jobs=4, verbose=-1,
    )
    return model.fit(x_tr, train.y.to_numpy()).predict(x_te)


def within_study_slope(frame: pd.DataFrame, target: str = "y") -> float:
    nets = frame[frame.method_family == "surface net"]
    y_within = nets[target] - nets.groupby("study")[target].transform("mean")
    a_within = nets.lag_acc - nets.groupby("study").lag_acc.transform("mean")
    denom = float(np.dot(a_within, a_within))
    return max(0.0, float(np.dot(a_within, y_within) / denom)) if denom > 1e-6 else 0.0


def transport_offset(frame: pd.DataFrame, beta: float) -> np.ndarray:
    return np.where(frame.method_family == "surface net", beta * frame.lag_acc.to_numpy(), 0.0)


def predict_within_between_physics(train: pd.DataFrame, test: pd.DataFrame, cols: list[str] = PROTOCOL_FEATURES, **params) -> np.ndarray:
    beta = within_study_slope(train)
    adjusted = train.y.to_numpy() - transport_offset(train, beta)
    return shallow_protocol(train, test, adjusted, cols=cols, **params) + transport_offset(test, beta)


def predict_with_space_stage(train: pd.DataFrame, test: pd.DataFrame, use_physics: bool) -> np.ndarray:
    def first_stage(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        beta = within_study_slope(tr) if use_physics else 0.0
        residual_target = tr.y.to_numpy() - transport_offset(tr, beta)
        return shallow_protocol(tr, te, residual_target) + transport_offset(te, beta), residual_target

    def space_stage(tr: pd.DataFrame, te: pd.DataFrame, residual_target: np.ndarray) -> np.ndarray:
        within = residual_target - pd.Series(residual_target).groupby(tr.study.to_numpy()).transform("mean").to_numpy()
        return lgbm(tr, te, SPACE_TIME_FEATURES, within, n_estimators=120, learning_rate=0.03, max_depth=3, num_leaves=7, min_child_samples=60)

    base_oof = np.empty(len(train))
    space_oof = np.empty(len(train))
    for itr, iva in GroupKFold(4).split(train, groups=train.study):
        tr_in, va_in = train.iloc[itr], train.iloc[iva]
        base_oof[iva], residual = first_stage(tr_in, va_in)
        space_oof[iva] = space_stage(tr_in, va_in, residual)
    y = train.y.to_numpy()
    gamma = min([0.0, 0.25, 0.5, 0.75, 1.0], key=lambda g: float(np.mean((y - base_oof - g * space_oof) ** 2)))

    base, residual = first_stage(train, test)
    return base if gamma == 0.0 else base + gamma * space_stage(train, test, residual)
