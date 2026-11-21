from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.ndimage import gaussian_filter
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
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

BARROWS_DOI = "https://doi.org/10.1016/j.envpol.2018.02.062"
DEHM_DOI = "https://doi.org/10.1016/j.marpolbul.2020.111239"
UNIT_FIXES = {BARROWS_DOI: 1e6, DEHM_DOI: 1e6}

NUMERIC = ["lat", "lon", "year", "month_sin", "month_cos", "log_mesh", "log_depth"]
FEATURES = NUMERIC + ["method_family"]
PROTOCOL_FEATURES = ["log_mesh", "log_depth", "method_family"]
SPACE_TIME_FEATURES = ["lat", "lon", "month_sin", "month_cos"]
OCEANS = ["Indian Ocean", "Pacific Ocean", "Atlantic Ocean", "Arctic Ocean", "Southern Ocean"]

DALVI_OCEAN_R2 = {
    "Indian Ocean": 0.166,
    "Pacific Ocean": 0.082,
    "Atlantic Ocean": 0.010,
    "Arctic Ocean": -0.253,
    "Southern Ocean": -0.708,
}

BLUE, ORANGE, AQUA, RED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#c93b3b", "#7c4dff"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "font.size": 10,
})


def linked_groups(first: pd.Series, second: pd.Series) -> pd.Series:
    group = first.astype(str)
    while not (merged := group.groupby(second).transform("min").groupby(first).transform("min")).equals(group):
        group = merged
    return group


def build_lagrangian_accumulation_field() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drifters = pd.read_csv(DRIFTER_DATA)
    u_grid = np.zeros((180, 360), dtype=float)
    v_grid = np.zeros((180, 360), dtype=float)
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
    dlat_rad = np.radians(1.0)
    dlon_rad = np.radians(1.0)

    for _ in range(365):
        iy = np.clip(np.round(py).astype(int), 0, 179)
        ix = np.round(px).astype(int) % 360
        cos_lat = np.maximum(np.cos(np.radians(py - 89.5)), 0.15)
        dy = (v_grid[iy, ix] * dt_seconds) / (EARTH_RADIUS_M * dlat_rad) + rng.normal(0.0, 0.15, size=particle_count)
        dx = (u_grid[iy, ix] * dt_seconds) / (EARTH_RADIUS_M * cos_lat * dlon_rad) + rng.normal(0.0, 0.15, size=particle_count)
        next_y = np.clip(py + dy, 0.0, 179.0)
        next_x = (px + dx) % 360.0
        stays_in_ocean = ocean_mask[np.clip(np.round(next_y).astype(int), 0, 179), np.round(next_x).astype(int) % 360]
        py = np.where(stays_in_ocean, next_y, py)
        px = np.where(stays_in_ocean, next_x, px)

    density = np.zeros((180, 360), dtype=float)
    np.add.at(density, (np.clip(np.round(py).astype(int), 0, 179), np.round(px).astype(int) % 360), 1.0)
    log_accumulation = np.log1p(gaussian_filter(density, sigma=2.5))
    speed = np.hypot(u_grid, v_grid)
    return ocean_y, ocean_x, log_accumulation, speed


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
    })[
        ["date", "lat", "lon", "ocean", "medium", "depth", "method", "mesh", "conc", "unit", "accession", "doi", "organization"]
    ]
    data["date"] = pd.to_datetime(data.date, format="%m/%d/%Y %I:%M:%S %p")
    water = data[(data.medium == "Ocean water") & (data.unit == "pieces/m3")].reset_index(drop=True)
    water["conc"] *= water.doi.map(UNIT_FIXES).fillna(1.0)
    method = water.method.str.lower().str.replace("-", " ")
    family = method.map(METHOD_FAMILY).fillna("bulk water")
    month = water.date.dt.month

    ocean_y, ocean_x, log_accumulation, speed = build_lagrangian_accumulation_field()
    ocean_coords = np.radians(np.column_stack([ocean_y - 89.5, ocean_x - 179.5]))
    nearest_cell = BallTree(ocean_coords, metric="haversine").query(np.radians(water[["lat", "lon"]].to_numpy()))[1][:, 0]

    return water.assign(
        method=method,
        method_family=family,
        year=water.date.dt.year,
        month_sin=np.sin(2 * np.pi * month / 12),
        month_cos=np.cos(2 * np.pi * month / 12),
        log_mesh=np.log10(water.mesh),
        log_depth=np.log1p(water.depth),
        study=linked_groups(water.doi, water.accession),
        y=np.log10(water.conc + OFFSET_C),
        lag_acc=log_accumulation[ocean_y[nearest_cell], ocean_x[nearest_cell]],
        drifter_speed=speed[ocean_y[nearest_cell], ocean_x[nearest_cell]],
        abs_lat=np.abs(water.lat),
    )


def build_design_matrices(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
    out_names = list(num_cols)
    if cat_cols:
        out_names.extend(ct.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist())
    return x_tr, x_te, out_names


def score_predictions(y: np.ndarray, pred: np.ndarray, studies: pd.Series) -> dict[str, float]:
    err = pred - y
    abs_err = np.abs(err)
    study_mae = pd.Series(abs_err).groupby(studies.to_numpy()).mean().mean()
    return {
        "R2": float(1.0 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "study_MAE": float(study_mae),
        "within_1_log": float(np.mean(abs_err <= 1.0)),
        "miss_gt_2_log": float(np.mean(abs_err > 2.0)),
    }


def within_study_rank_stats(df: pd.DataFrame, pred: np.ndarray) -> tuple[float, float, int]:
    frame = pd.DataFrame({"y": df.y.to_numpy(), "pred": pred, "study": df.study.to_numpy()})
    rhos_varying = []
    rhos_all_eligible = []
    for _, grp in frame.groupby("study"):
        if len(grp) < 20 or grp.y.nunique() <= 1:
            continue
        if grp.pred.nunique() <= 1:
            rhos_all_eligible.append(0.0)
        else:
            rho = float(grp.pred.corr(grp.y, method="spearman"))
            rhos_varying.append(rho)
            rhos_all_eligible.append(rho)
    med_all = float(np.median(rhos_all_eligible)) if rhos_all_eligible else float("nan")
    med_varying = float(np.median(rhos_varying)) if rhos_varying else float("nan")
    return med_all, med_varying, len(rhos_varying)


def predict_baseline_all(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_tr, x_te, _ = build_design_matrices(train, test, FEATURES)
    model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=SEED, n_jobs=4, verbose=-1)
    return model.fit(x_tr, train.y.to_numpy()).predict(x_te)


def predict_baseline_protocol(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_tr, x_te, _ = build_design_matrices(train, test, PROTOCOL_FEATURES)
    model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=SEED, n_jobs=4, verbose=-1)
    return model.fit(x_tr, train.y.to_numpy()).predict(x_te)


def predict_interaction_constrained(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_tr, x_te, names = build_design_matrices(train, test, FEATURES)
    proto_idx = [i for i, n in enumerate(names) if n in ("log_mesh", "log_depth") or n.startswith("method_family_")]
    space_idx = [i for i in range(len(names)) if i not in proto_idx]
    monotone = [-1 if n == "log_mesh" else 0 for n in names]
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        min_child_samples=50,
        objective="huber",
        interaction_constraints=[proto_idx, space_idx],
        monotone_constraints=monotone,
        random_state=SEED,
        n_jobs=4,
        verbose=-1,
    )
    return model.fit(x_tr, train.y.to_numpy()).predict(x_te)


def fit_constrained_protocol_target(train: pd.DataFrame, test: pd.DataFrame, target: np.ndarray, cols: list[str] = PROTOCOL_FEATURES, **params) -> np.ndarray:
    x_tr, x_te, names = build_design_matrices(train, test, cols)
    monotone = [-1 if n == "log_mesh" else 0 for n in names]
    defaults = dict(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        min_child_samples=30,
        objective="huber",
        monotone_constraints=monotone,
        random_state=SEED,
        n_jobs=4,
        verbose=-1,
    )
    model = LGBMRegressor(**(defaults | params))
    return model.fit(x_tr, target).predict(x_te)


def predict_constrained_protocol(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return fit_constrained_protocol_target(train, test, train.y.to_numpy())


def fit_within_study_lagrangian_slopes(train: pd.DataFrame) -> dict[str, float]:
    y_within = train.y.to_numpy() - train.groupby("study")["y"].transform("mean").to_numpy()
    x_within = train.lag_acc.to_numpy() - train.groupby("study")["lag_acc"].transform("mean").to_numpy()
    slopes: dict[str, float] = {}
    for fam in train.method_family.unique():
        mask = (train.method_family == fam).to_numpy()
        denom = float(np.dot(x_within[mask], x_within[mask]))
        if denom < 1e-6 or fam != "surface net":
            slopes[fam] = 0.0
        else:
            slopes[fam] = max(0.0, float(np.dot(x_within[mask], y_within[mask]) / denom))
    return slopes


def predict_within_between_physics(train: pd.DataFrame, test: pd.DataFrame, cols: list[str] = PROTOCOL_FEATURES) -> np.ndarray:
    slopes = fit_within_study_lagrangian_slopes(train)
    tr_transport = train.method_family.map(slopes).fillna(0.0).to_numpy() * train.lag_acc.to_numpy()
    te_transport = test.method_family.map(slopes).fillna(0.0).to_numpy() * test.lag_acc.to_numpy()
    y_adjusted = train.y.to_numpy() - tr_transport
    proto_pred = fit_constrained_protocol_target(train, test, y_adjusted, cols=cols)
    return proto_pred + te_transport


def predict_within_between_space_tuned(train: pd.DataFrame, test: pd.DataFrame, use_physics: bool = False) -> np.ndarray:
    n_studies = train.study.nunique()
    n_inner = min(4, n_studies)
    if n_inner < 2:
        return predict_within_between_physics(train, test) if use_physics else predict_constrained_protocol(train, test)

    inner_folds = list(GroupKFold(n_inner).split(train, groups=train.study))
    base_oof = np.empty(len(train))
    space_oof = np.empty(len(train))
    for itr, iva in inner_folds:
        tr_in, va_in = train.iloc[itr], train.iloc[iva]
        if use_physics:
            slopes = fit_within_study_lagrangian_slopes(tr_in)
            tr_t = tr_in.method_family.map(slopes).fillna(0.0).to_numpy() * tr_in.lag_acc.to_numpy()
            va_t = va_in.method_family.map(slopes).fillna(0.0).to_numpy() * va_in.lag_acc.to_numpy()
            base_oof[iva] = fit_constrained_protocol_target(tr_in, va_in, tr_in.y.to_numpy() - tr_t) + va_t
            y_res = tr_in.y.to_numpy() - tr_t
        else:
            base_oof[iva] = fit_constrained_protocol_target(tr_in, va_in, tr_in.y.to_numpy())
            y_res = tr_in.y.to_numpy()
        y_within = y_res - pd.Series(y_res).groupby(tr_in.study.to_numpy()).transform("mean").to_numpy()
        x_tr_s, x_va_s, _ = build_design_matrices(tr_in, va_in, SPACE_TIME_FEATURES)
        m2 = LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=3, num_leaves=7, min_child_samples=60, random_state=SEED, n_jobs=4, verbose=-1)
        space_oof[iva] = m2.fit(x_tr_s, y_within).predict(x_va_s)

    y_tr = train.y.to_numpy()
    gamma_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_gamma = min(gamma_grid, key=lambda g: float(np.mean((y_tr - (base_oof + g * space_oof)) ** 2)))

    if use_physics:
        slopes = fit_within_study_lagrangian_slopes(train)
        tr_t = train.method_family.map(slopes).fillna(0.0).to_numpy() * train.lag_acc.to_numpy()
        te_t = test.method_family.map(slopes).fillna(0.0).to_numpy() * test.lag_acc.to_numpy()
        p_base = fit_constrained_protocol_target(train, test, y_tr - tr_t) + te_t
        y_res = y_tr - tr_t
    else:
        p_base = fit_constrained_protocol_target(train, test, y_tr)
        y_res = y_tr

    if best_gamma == 0.0:
        return p_base

    y_within = y_res - pd.Series(y_res).groupby(train.study.to_numpy()).transform("mean").to_numpy()
    x_tr_s, x_te_s, _ = build_design_matrices(train, test, SPACE_TIME_FEATURES)
    m2 = LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=3, num_leaves=7, min_child_samples=60, random_state=SEED, n_jobs=4, verbose=-1)
    return p_base + best_gamma * m2.fit(x_tr_s, y_within).predict(x_te_s)


def run_cv_predictions(df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], fn) -> np.ndarray:
    pred = np.empty(len(df))
    for tr_idx, te_idx in folds:
        pred[te_idx] = fn(df.iloc[tr_idx], df.iloc[te_idx])
    return pred


def block1_global_benchmark(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_folds = [list(GroupKFold(5, shuffle=True, random_state=r).split(df, groups=df.study)) for r in range(REPEATS)]
    orig_folds = list(GroupKFold(5).split(df, groups=df.study))
    org_folds = list(GroupKFold(5).split(df, groups=df.organization))
    tr_chrono, te_chrono = df[df.year < 2015], df[df.year >= 2015]
    unseen_chrono_mask = ~te_chrono.study.isin(tr_chrono.study).to_numpy()

    models = {
        "Baseline LightGBM (all features)": predict_baseline_all,
        "Baseline LightGBM (protocol only)": predict_baseline_protocol,
        "Interaction-constrained + mono LightGBM": predict_interaction_constrained,
        "Constrained Protocol LightGBM (mono+Huber)": predict_constrained_protocol,
        "Within-Between LightGBM (Protocol + CV Space)": lambda tr, te: predict_within_between_space_tuned(tr, te, use_physics=False),
        "Within-Between Physics (Protocol + Lagrangian)": lambda tr, te: predict_within_between_physics(tr, te),
        "Within-Between Physics + CV Space": lambda tr, te: predict_within_between_space_tuned(tr, te, use_physics=True),
    }

    repeat_rows = []
    summary_rows = []
    looo_rows = []

    base_r2_per_repeat: list[float] = []
    proto_r2_per_repeat: list[float] = []

    for name, fn in models.items():
        r2s, maes, smaes, w1s, m2s, rhos_all, rhos_var, k_var = [], [], [], [], [], [], [], []
        for rep, folds in enumerate(repeat_folds):
            pred = run_cv_predictions(df, folds, fn)
            sc = score_predictions(df.y.to_numpy(), pred, df.study)
            med_all, med_var, kv = within_study_rank_stats(df, pred)
            r2s.append(sc["R2"])
            maes.append(sc["MAE"])
            smaes.append(sc["study_MAE"])
            w1s.append(sc["within_1_log"])
            m2s.append(sc["miss_gt_2_log"])
            rhos_all.append(med_all)
            rhos_var.append(med_var)
            k_var.append(kv)
            repeat_rows.append({"model": name, "repeat": rep, **sc, "median_rho_all": med_all, "median_rho_varying": med_var})

        if name == "Baseline LightGBM (all features)":
            base_r2_per_repeat = list(r2s)
        if name == "Constrained Protocol LightGBM (mono+Huber)":
            proto_r2_per_repeat = list(r2s)

        orig_sc = score_predictions(df.y.to_numpy(), run_cv_predictions(df, orig_folds, fn), df.study)
        org_sc = score_predictions(df.y.to_numpy(), run_cv_predictions(df, org_folds, fn), df.study)

        pred_c = fn(tr_chrono, te_chrono)
        chrono_all = score_predictions(te_chrono.y.to_numpy(), pred_c, te_chrono.study)
        chrono_unseen = score_predictions(te_chrono.y.to_numpy()[unseen_chrono_mask], pred_c[unseen_chrono_mask], te_chrono.study.iloc[unseen_chrono_mask])

        ocean_r2s = []
        for ocean in OCEANS:
            te_o = df[df.ocean == ocean]
            tr_o = df[(df.ocean != ocean) & (~df.study.isin(te_o.study))]
            pred_o = fn(tr_o, te_o)
            sc_o = score_predictions(te_o.y.to_numpy(), pred_o, te_o.study)
            ocean_r2s.append(sc_o["R2"])
            looo_rows.append({"model": name, "ocean": ocean, "dalvi_R2": DALVI_OCEAN_R2[ocean], **sc_o})

        wins_vs_base = sum(a > b for a, b in zip(r2s, base_r2_per_repeat))
        wins_vs_proto = sum(a > b for a, b in zip(r2s, proto_r2_per_repeat)) if proto_r2_per_repeat else 0

        summary_rows.append({
            "model": name,
            "orig_grouped_R2": orig_sc["R2"],
            "repeat10_R2_mean": float(np.mean(r2s)),
            "repeat10_R2_min": float(np.min(r2s)),
            "repeat10_R2_max": float(np.max(r2s)),
            "wins_vs_baseline_of_10": wins_vs_base,
            "wins_vs_proto_of_10": wins_vs_proto,
            "repeat10_MAE": float(np.mean(maes)),
            "repeat10_study_MAE": float(np.mean(smaes)),
            "repeat10_within_1_log": float(np.mean(w1s)),
            "repeat10_miss_gt_2_log": float(np.mean(m2s)),
            "within_study_rho_all": float(np.nanmean(rhos_all)),
            "within_study_rho_varying": float(np.nanmean(rhos_var)),
            "varying_studies_count": float(np.mean(k_var)),
            "org_grouped_R2": org_sc["R2"],
            "org_grouped_MAE": org_sc["MAE"],
            "chrono_R2": chrono_all["R2"],
            "chrono_unseen_R2": chrono_unseen["R2"],
            "looo_mean_R2": float(np.mean(ocean_r2s)),
            "looo_min_R2": float(np.min(ocean_r2s)),
        })

    oracle_pred = df.groupby("study")["y"].transform("median").to_numpy()
    oracle_sc = score_predictions(df.y.to_numpy(), oracle_pred, df.study)
    summary_rows.append({
        "model": "Oracle within-study median (noise floor)",
        "orig_grouped_R2": oracle_sc["R2"],
        "repeat10_R2_mean": oracle_sc["R2"],
        "repeat10_R2_min": oracle_sc["R2"],
        "repeat10_R2_max": oracle_sc["R2"],
        "wins_vs_baseline_of_10": 10,
        "wins_vs_proto_of_10": 10,
        "repeat10_MAE": oracle_sc["MAE"],
        "repeat10_study_MAE": oracle_sc["study_MAE"],
        "repeat10_within_1_log": oracle_sc["within_1_log"],
        "repeat10_miss_gt_2_log": oracle_sc["miss_gt_2_log"],
        "within_study_rho_all": 0.0,
        "within_study_rho_varying": np.nan,
        "varying_studies_count": 0.0,
        "org_grouped_R2": oracle_sc["R2"],
        "org_grouped_MAE": oracle_sc["MAE"],
        "chrono_R2": np.nan,
        "chrono_unseen_R2": np.nan,
        "looo_mean_R2": np.nan,
        "looo_min_R2": np.nan,
    })

    summary_df = pd.DataFrame(summary_rows)
    repeats_df = pd.DataFrame(repeat_rows)
    looo_df = pd.DataFrame(looo_rows)
    summary_df.to_csv(ARTIFACTS / "v2_global_benchmark.csv", index=False)
    repeats_df.to_csv(ARTIFACTS / "v2_global_repeats.csv", index=False)
    looo_df.to_csv(ARTIFACTS / "v2_looo_comparison.csv", index=False)
    return summary_df, repeats_df, looo_df


def block2_surface_nets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nets = df[df.method_family == "surface net"].reset_index(drop=True)
    repeat_folds = [list(GroupKFold(5, shuffle=True, random_state=r).split(nets, groups=nets.study)) for r in range(REPEATS)]
    org_folds = list(GroupKFold(5).split(nets, groups=nets.organization))

    raw_rank_rows = []
    for feat in ["lag_acc", "drifter_speed", "abs_lat", "lat", "lon"]:
        rhos = [
            float(g[feat].corr(g.y, method="spearman"))
            for _, g in nets.groupby("study")
            if len(g) >= 20 and g.y.nunique() > 1 and g[feat].nunique() > 1
        ]
        raw_rank_rows.append({
            "feature": feat,
            "studies_evaluated": len(rhos),
            "median_spearman_rho": float(np.median(rhos)),
            "mean_spearman_rho": float(np.mean(rhos)),
            "share_positive": float(np.mean(np.array(rhos) > 0)),
        })
    raw_rank_df = pd.DataFrame(raw_rank_rows)
    raw_rank_df.to_csv(ARTIFACTS / "v2_surface_net_raw_ranking.csv", index=False)

    net_models = {
        "Baseline LightGBM (all features)": predict_baseline_all,
        "Constrained Protocol (mono+Huber)": lambda tr, te: fit_constrained_protocol_target(tr, te, tr.y.to_numpy(), cols=["log_mesh", "log_depth"]),
        "Within-Between Physics (Protocol + Lagrangian)": lambda tr, te: predict_within_between_physics(tr, te, cols=["log_mesh", "log_depth"]),
    }

    rows = []
    proto_r2s: list[float] = []
    base_r2s: list[float] = []
    for name, fn in net_models.items():
        r2s, maes, w1s, m2s, rhos_all, rhos_var, k_var = [], [], [], [], [], [], []
        for folds in repeat_folds:
            pred = run_cv_predictions(nets, folds, fn)
            sc = score_predictions(nets.y.to_numpy(), pred, nets.study)
            ma, mv, kv = within_study_rank_stats(nets, pred)
            r2s.append(sc["R2"])
            maes.append(sc["MAE"])
            w1s.append(sc["within_1_log"])
            m2s.append(sc["miss_gt_2_log"])
            rhos_all.append(ma)
            rhos_var.append(mv)
            k_var.append(kv)
        if name == "Baseline LightGBM (all features)":
            base_r2s = list(r2s)
        if name == "Constrained Protocol (mono+Huber)":
            proto_r2s = list(r2s)
        org_sc = score_predictions(nets.y.to_numpy(), run_cv_predictions(nets, org_folds, fn), nets.study)
        rows.append({
            "model": name,
            "repeat10_R2_mean": float(np.mean(r2s)),
            "repeat10_R2_min": float(np.min(r2s)),
            "repeat10_R2_max": float(np.max(r2s)),
            "wins_vs_baseline_of_10": sum(a > b for a, b in zip(r2s, base_r2s)),
            "wins_vs_proto_of_10": sum(a > b for a, b in zip(r2s, proto_r2s)) if proto_r2s else 0,
            "repeat10_MAE": float(np.mean(maes)),
            "repeat10_within_1_log": float(np.mean(w1s)),
            "repeat10_miss_gt_2_log": float(np.mean(m2s)),
            "within_study_rho_all": float(np.nanmean(rhos_all)),
            "within_study_rho_varying": float(np.nanmean(rhos_var)),
            "varying_studies_count": float(np.mean(k_var)),
            "org_grouped_R2": org_sc["R2"],
            "org_grouped_MAE": org_sc["MAE"],
        })

    oracle_pred = nets.groupby("study")["y"].transform("median").to_numpy()
    oracle_sc = score_predictions(nets.y.to_numpy(), oracle_pred, nets.study)
    rows.append({
        "model": "Oracle within-study median (noise floor)",
        "repeat10_R2_mean": oracle_sc["R2"],
        "repeat10_R2_min": oracle_sc["R2"],
        "repeat10_R2_max": oracle_sc["R2"],
        "wins_vs_baseline_of_10": 10,
        "wins_vs_proto_of_10": 10,
        "repeat10_MAE": oracle_sc["MAE"],
        "repeat10_within_1_log": oracle_sc["within_1_log"],
        "repeat10_miss_gt_2_log": oracle_sc["miss_gt_2_log"],
        "within_study_rho_all": 0.0,
        "within_study_rho_varying": np.nan,
        "varying_studies_count": 0.0,
        "org_grouped_R2": oracle_sc["R2"],
        "org_grouped_MAE": oracle_sc["MAE"],
    })
    net_df = pd.DataFrame(rows)
    net_df.to_csv(ARTIFACTS / "v2_surface_nets.csv", index=False)
    return net_df, raw_rank_df


def estimate_protocol_lod(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def bucket(frame: pd.DataFrame) -> pd.Series:
        return frame.method_family.astype(str) + "_" + np.where(frame.mesh < 0.05, "fine", "coarse")

    tr_b, te_b = bucket(train), bucket(test)
    pos = train[train.conc > 0]
    pos_b = tr_b[train.conc > 0]
    bucket_lod = pos.assign(b=pos_b).groupby(["b", "study"])["conc"].min().groupby("b").median()
    fam_lod = pos.groupby(["method_family", "study"])["conc"].min().groupby("method_family").median()
    fallback = float(pos.conc.quantile(0.01))
    lod_tr = np.log10(tr_b.map(bucket_lod).fillna(train.method_family.map(fam_lod)).fillna(fallback).to_numpy())
    lod_te = np.log10(te_b.map(bucket_lod).fillna(test.method_family.map(fam_lod)).fillna(fallback).to_numpy())
    return lod_tr, lod_te


def inverse_mills(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    safe = z > -30.0
    out[safe] = np.exp(norm.logpdf(z[safe]) - log_ndtr(z[safe]))
    out[~safe] = -z[~safe] - 1.0 / z[~safe]
    return out


def tobit_nll(mu: np.ndarray, y_pos: np.ndarray, is_pos: np.ndarray, lod: np.ndarray, sigma: float) -> np.ndarray:
    nll = np.empty_like(mu)
    nll[is_pos] = 0.5 * ((mu[is_pos] - y_pos[is_pos]) / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2.0 * np.pi)
    nll[~is_pos] = -log_ndtr((lod[~is_pos] - mu[~is_pos]) / sigma)
    return nll


def fit_predict_tobit(train: pd.DataFrame, test: pd.DataFrame, use_physics: bool = True, sigma: float = 0.9) -> dict[str, np.ndarray | float]:
    lod_tr, lod_te = estimate_protocol_lod(train, test)
    is_pos = (train.conc > 0).to_numpy()
    y_pos = np.where(is_pos, np.log10(np.maximum(train.conc.to_numpy(), 1e-12)), lod_tr)

    if use_physics:
        pos_tr = train[train.conc > 0].copy()
        pos_tr["y_pos"] = np.log10(pos_tr.conc.to_numpy())
        yw = pos_tr.y_pos.to_numpy() - pos_tr.groupby("study")["y_pos"].transform("mean").to_numpy()
        xw = pos_tr.lag_acc.to_numpy() - pos_tr.groupby("study")["lag_acc"].transform("mean").to_numpy()
        net_mask = (pos_tr.method_family == "surface net").to_numpy()
        denom = float(np.dot(xw[net_mask], xw[net_mask]))
        beta = max(0.0, float(np.dot(xw[net_mask], yw[net_mask]) / denom)) if denom > 1e-6 else 0.0
        tr_offset = np.where(train.method_family == "surface net", beta * train.lag_acc.to_numpy(), 0.0)
        te_offset = np.where(test.method_family == "surface net", beta * test.lag_acc.to_numpy(), 0.0)
    else:
        tr_offset = np.zeros(len(train))
        te_offset = np.zeros(len(test))

    y_pos_adj = y_pos - tr_offset
    lod_tr_adj = lod_tr - tr_offset

    x_tr, x_te, names = build_design_matrices(train, test, PROTOCOL_FEATURES)
    monotone = [-1 if n == "log_mesh" else 0 for n in names]

    def objective(preds: np.ndarray, _: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
        z = (lod_tr_adj - preds) / sigma
        lam = inverse_mills(z)
        grad = np.where(is_pos, (preds - y_pos_adj) / (sigma**2), lam / sigma)
        hess = np.where(is_pos, 1.0 / (sigma**2), np.maximum(lam * (lam + z) / (sigma**2), 1e-6))
        return grad, hess

    init_val = float(np.median(y_pos_adj[is_pos]))
    dtrain = lgb.Dataset(x_tr, label=y_pos_adj, init_score=np.full(len(train), init_val))
    params = dict(
        objective=objective,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        min_child_samples=30,
        monotone_constraints=monotone,
        verbose=-1,
        n_jobs=4,
        seed=SEED,
    )
    booster = lgb.train(params, dtrain, num_boost_round=150)
    mu_tr = init_val + booster.predict(x_tr) + tr_offset
    mu_te = init_val + booster.predict(x_te) + te_offset
    sig_hat = max(0.3, float(np.std(y_pos[is_pos] - mu_tr[is_pos])))
    z_te = (lod_te - mu_te) / sig_hat
    p_det = 1.0 - norm.cdf(z_te)
    e_pos = mu_te + sig_hat * inverse_mills(-z_te)
    return {"mu": mu_te, "e_pos": e_pos, "p_det": p_det, "lod_te": lod_te, "sigma": sig_hat}


def block3_censored_and_twopart(df: pd.DataFrame) -> pd.DataFrame:
    repeat_folds = [list(GroupKFold(5, shuffle=True, random_state=r).split(df, groups=df.study)) for r in range(REPEATS)]
    pos_mask = (df.conc > 0).to_numpy()
    y_pos_true = np.log10(np.maximum(df.conc.to_numpy(), 1e-12))
    is_pos_int = pos_mask.astype(int)

    methods = [
        "Baseline LightGBM (all features, naive)",
        "Two-Part: Constrained Protocol (mono+Huber)",
        "Two-Part: Within-Between Physics (Protocol + Lagrangian)",
        "Left-Censored Tobit LightGBM (Protocol + Lagrangian)",
    ]

    summary_rows = []
    for name in methods:
        nlls, aucs, briers, pos_r2s, pos_maes, pos_w1s, pos_m2s = [], [], [], [], [], [], []
        for folds in repeat_folds:
            mu_oof = np.empty(len(df))
            epos_oof = np.empty(len(df))
            pdet_oof = np.empty(len(df))
            nll_oof = np.empty(len(df))

            for tr_i, te_i in folds:
                tr, te = df.iloc[tr_i], df.iloc[te_i]
                _, lod_te = estimate_protocol_lod(tr, te)
                te_pos = (te.conc > 0).to_numpy()
                te_ypos = np.where(te_pos, np.log10(np.maximum(te.conc.to_numpy(), 1e-12)), lod_te)

                if name == "Baseline LightGBM (all features, naive)":
                    x_tr, x_te, _ = build_design_matrices(tr, te, FEATURES)
                    reg = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=SEED, n_jobs=4, verbose=-1).fit(x_tr, tr.y.to_numpy())
                    pred_te = reg.predict(x_te)
                    sig = max(0.3, float(np.std(tr.y.to_numpy() - reg.predict(x_tr))))
                    clf = LGBMClassifier(n_estimators=200, max_depth=3, num_leaves=7, learning_rate=0.05, random_state=SEED, n_jobs=4, verbose=-1)
                    pdet_oof[te_i] = clf.fit(x_tr, (tr.conc > 0).astype(int)).predict_proba(x_te)[:, 1]
                    mu_oof[te_i] = pred_te
                    epos_oof[te_i] = pred_te
                    nll_oof[te_i] = tobit_nll(pred_te, te_ypos, te_pos, lod_te, sig)

                elif name.startswith("Two-Part:"):
                    tr_p = tr[tr.conc > 0].copy()
                    tr_p["y"] = np.log10(tr_p.conc.to_numpy())
                    use_phys = "Physics" in name
                    if use_phys:
                        pred_pos = predict_within_between_physics(tr_p, te)
                        pred_pos_tr = predict_within_between_physics(tr_p, tr_p)
                    else:
                        pred_pos = predict_constrained_protocol(tr_p, te)
                        pred_pos_tr = predict_constrained_protocol(tr_p, tr_p)
                    sig = max(0.3, float(np.std(tr_p.y.to_numpy() - pred_pos_tr)))

                    x_tr_c, x_te_c, _ = build_design_matrices(tr, te, PROTOCOL_FEATURES)
                    clf = LGBMClassifier(n_estimators=150, max_depth=3, num_leaves=7, min_child_samples=30, learning_rate=0.05, random_state=SEED, n_jobs=4, verbose=-1)
                    pdet_oof[te_i] = clf.fit(x_tr_c, (tr.conc > 0).astype(int)).predict_proba(x_te_c)[:, 1]
                    mu_oof[te_i] = pred_pos
                    epos_oof[te_i] = pred_pos
                    nll_oof[te_i] = tobit_nll(pred_pos, te_ypos, te_pos, lod_te, sig)

                else:
                    out = fit_predict_tobit(tr, te, use_physics=True)
                    mu_oof[te_i] = out["mu"]
                    epos_oof[te_i] = out["e_pos"]
                    pdet_oof[te_i] = out["p_det"]
                    nll_oof[te_i] = tobit_nll(out["mu"], te_ypos, te_pos, lod_te, float(out["sigma"]))

            sc_pos = score_predictions(y_pos_true[pos_mask], epos_oof[pos_mask], df.study.iloc[pos_mask])
            nlls.append(float(np.mean(nll_oof)))
            aucs.append(float(roc_auc_score(is_pos_int, pdet_oof)))
            briers.append(float(brier_score_loss(is_pos_int, pdet_oof)))
            pos_r2s.append(sc_pos["R2"])
            pos_maes.append(sc_pos["MAE"])
            pos_w1s.append(sc_pos["within_1_log"])
            pos_m2s.append(sc_pos["miss_gt_2_log"])

        summary_rows.append({
            "model": name,
            "tobit_nll": float(np.mean(nlls)),
            "detect_AUC": float(np.mean(aucs)),
            "detect_Brier": float(np.mean(briers)),
            "pos_R2_mean": float(np.mean(pos_r2s)),
            "pos_R2_min": float(np.min(pos_r2s)),
            "pos_R2_max": float(np.max(pos_r2s)),
            "pos_MAE": float(np.mean(pos_maes)),
            "pos_within_1_log": float(np.mean(pos_w1s)),
            "pos_miss_gt_2_log": float(np.mean(pos_m2s)),
        })

    pos_df = df[pos_mask].copy()
    pos_df["y_pos"] = np.log10(pos_df.conc.to_numpy())
    oracle_pos = pos_df.groupby("study")["y_pos"].transform("median").to_numpy()
    oracle_pos_sc = score_predictions(pos_df.y_pos.to_numpy(), oracle_pos, pos_df.study)
    summary_rows.append({
        "model": "Oracle within-study median (positive noise floor)",
        "tobit_nll": np.nan,
        "detect_AUC": np.nan,
        "detect_Brier": np.nan,
        "pos_R2_mean": oracle_pos_sc["R2"],
        "pos_R2_min": oracle_pos_sc["R2"],
        "pos_R2_max": oracle_pos_sc["R2"],
        "pos_MAE": oracle_pos_sc["MAE"],
        "pos_within_1_log": oracle_pos_sc["within_1_log"],
        "pos_miss_gt_2_log": oracle_pos_sc["miss_gt_2_log"],
    })

    res = pd.DataFrame(summary_rows)
    res.to_csv(ARTIFACTS / "v2_censored_twopart.csv", index=False)
    return res


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w) / np.sum(w)
    idx = np.searchsorted(cum, q, side="left")
    return float(v[min(idx, len(v) - 1)])


def fit_quantile_baseline(train: pd.DataFrame, test: pd.DataFrame, q: float) -> np.ndarray:
    x_tr, x_te, _ = build_design_matrices(train, test, FEATURES)
    model = LGBMRegressor(n_estimators=500, learning_rate=0.05, objective="quantile", alpha=q, random_state=SEED, n_jobs=4, verbose=-1)
    return model.fit(x_tr, train.y.to_numpy()).predict(x_te)


def fit_quantile_constrained_physics(train: pd.DataFrame, test: pd.DataFrame, q: float) -> np.ndarray:
    slopes = fit_within_study_lagrangian_slopes(train)
    tr_t = train.method_family.map(slopes).fillna(0.0).to_numpy() * train.lag_acc.to_numpy()
    te_t = test.method_family.map(slopes).fillna(0.0).to_numpy() * test.lag_acc.to_numpy()
    x_tr, x_te, names = build_design_matrices(train, test, PROTOCOL_FEATURES)
    model = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        min_child_samples=30,
        objective="quantile",
        alpha=q,
        random_state=SEED,
        n_jobs=4,
        verbose=-1,
    )
    return model.fit(x_tr, train.y.to_numpy() - tr_t).predict(x_te) + te_t


def block4_intervals_and_selective(df: pd.DataFrame, alpha: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_folds = list(GroupKFold(5).split(df, groups=df.study))
    y = df.y.to_numpy()
    n = len(df)

    variants = {
        "Option A Baseline LightGBM": (predict_baseline_all, fit_quantile_baseline),
        "Within-Between Physics LightGBM": (predict_within_between_physics, fit_quantile_constrained_physics),
    }

    interval_rows = []
    rc_rows = []

    for label, (point_fn, quant_fn) in variants.items():
        point_oof = np.empty(n)
        cqr_row_lo, cqr_row_hi = np.empty(n), np.empty(n)
        cqr_st_lo, cqr_st_hi = np.empty(n), np.empty(n)
        sym_row_lo, sym_row_hi = np.empty(n), np.empty(n)

        for tr_i, te_i in outer_folds:
            tr = df.iloc[tr_i].reset_index(drop=True)
            te = df.iloc[te_i].reset_index(drop=True)

            p_te = point_fn(tr, te)
            point_oof[te_i] = p_te

            inner_folds = list(GroupKFold(4).split(tr, groups=tr.study))
            p_oof = np.empty(len(tr))
            qlo_oof = np.empty(len(tr))
            qhi_oof = np.empty(len(tr))
            for itr, iva in inner_folds:
                tr_in, va_in = tr.iloc[itr], tr.iloc[iva]
                p_oof[iva] = point_fn(tr_in, va_in)
                ql = quant_fn(tr_in, va_in, alpha / 2)
                qh = quant_fn(tr_in, va_in, 1.0 - alpha / 2)
                qlo_oof[iva], qhi_oof[iva] = np.minimum(ql, qh), np.maximum(ql, qh)

            y_tr = tr.y.to_numpy()
            abs_res = np.abs(y_tr - p_oof)
            q_sym = conformal_quantile(abs_res, alpha)
            sym_row_lo[te_i], sym_row_hi[te_i] = p_te - q_sym, p_te + q_sym

            cqr_scores = np.maximum(qlo_oof - y_tr, y_tr - qhi_oof)
            study_w = 1.0 / tr.groupby("study")["y"].transform("size").to_numpy()
            q_cqr_row = conformal_quantile(cqr_scores, alpha)
            q_cqr_st = weighted_quantile(cqr_scores, study_w, 1.0 - alpha)

            ql_te = quant_fn(tr, te, alpha / 2)
            qh_te = quant_fn(tr, te, 1.0 - alpha / 2)
            ql_te, qh_te = np.minimum(ql_te, qh_te), np.maximum(ql_te, qh_te)

            cqr_row_lo[te_i], cqr_row_hi[te_i] = ql_te - q_cqr_row, qh_te + q_cqr_row
            cqr_st_lo[te_i], cqr_st_hi[te_i] = ql_te - q_cqr_st, qh_te + q_cqr_st

        for sub_name, lo, hi in [
            ("Study-grouped conformal (row)", sym_row_lo, sym_row_hi),
            ("Study-grouped CQR (row)", cqr_row_lo, cqr_row_hi),
            ("Study-grouped CQR (study-weighted)", cqr_st_lo, cqr_st_hi),
        ]:
            covered = (y >= lo) & (y <= hi)
            width = hi - lo
            per_study_cov = pd.Series(covered.astype(float)).groupby(df.study.to_numpy()).mean()
            interval_rows.append({
                "model": label,
                "interval_method": sub_name,
                "row_coverage": float(covered.mean()),
                "mean_study_coverage": float(per_study_cov.mean()),
                "studies_ge_80pct": float((per_study_cov >= 0.80).mean()),
                "mean_width_log10": float(width.mean()),
                "median_width_log10": float(np.median(width)),
                "half_width_factor_x": float(10.0 ** (np.median(width) / 2.0)),
            })

        abs_err = np.abs(point_oof - y)
        width_rank = np.argsort(cqr_st_hi - cqr_st_lo)
        for keep_frac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            k = max(1, int(round(keep_frac * n)))
            kept = abs_err[width_rank[:k]]
            rc_rows.append({
                "model": label,
                "keep_fraction": keep_frac,
                "MAE": float(kept.mean()),
                "within_1_log": float((kept <= 1.0).mean()),
                "miss_gt_2_log": float((kept > 2.0).mean()),
            })

    int_df = pd.DataFrame(interval_rows)
    rc_df = pd.DataFrame(rc_rows)
    int_df.to_csv(ARTIFACTS / "v2_intervals.csv", index=False)
    rc_df.to_csv(ARTIFACTS / "v2_selective_prediction.csv", index=False)
    return int_df, rc_df


def block5_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "Full dataset": df,
        "Drop bongo net (Arteaga 2024)": df[df.method != "bongo net"],
        "Drop Barrows & Dehm": df[~df.doi.isin([BARROWS_DOI, DEHM_DOI])],
        "Drop Barrows, Dehm & bongo": df[(~df.doi.isin([BARROWS_DOI, DEHM_DOI])) & (df.method != "bongo net")],
    }
    rows = []
    for label, sub in subsets.items():
        sub = sub.reset_index(drop=True)
        rfolds = [list(GroupKFold(5, shuffle=True, random_state=r).split(sub, groups=sub.study)) for r in range(REPEATS)]
        base_r2, base_mae, base_m2 = [], [], []
        new_r2, new_mae, new_m2 = [], [], []
        for folds in rfolds:
            pb = run_cv_predictions(sub, folds, predict_baseline_all)
            pn = run_cv_predictions(sub, folds, predict_within_between_physics)
            sb = score_predictions(sub.y.to_numpy(), pb, sub.study)
            sn = score_predictions(sub.y.to_numpy(), pn, sub.study)
            base_r2.append(sb["R2"])
            base_mae.append(sb["MAE"])
            base_m2.append(sb["miss_gt_2_log"])
            new_r2.append(sn["R2"])
            new_mae.append(sn["MAE"])
            new_m2.append(sn["miss_gt_2_log"])
        rows.append({
            "subset": label,
            "rows": len(sub),
            "studies": sub.study.nunique(),
            "baseline_10rep_R2": float(np.mean(base_r2)),
            "physics_10rep_R2": float(np.mean(new_r2)),
            "R2_gain": float(np.mean(new_r2) - np.mean(base_r2)),
            "wins_of_10": sum(a > b for a, b in zip(new_r2, base_r2)),
            "baseline_MAE": float(np.mean(base_mae)),
            "physics_MAE": float(np.mean(new_mae)),
            "baseline_miss_gt_2": float(np.mean(base_m2)),
            "physics_miss_gt_2": float(np.mean(new_m2)),
        })
    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS / "v2_sensitivity.csv", index=False)
    return res


def make_figures(summary_df: pd.DataFrame, repeats_df: pd.DataFrame, looo_df: pd.DataFrame, net_df: pd.DataFrame, int_df: pd.DataFrame, rc_df: pd.DataFrame, censored_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    order = [
        "Baseline LightGBM (all features)",
        "Baseline LightGBM (protocol only)",
        "Interaction-constrained + mono LightGBM",
        "Constrained Protocol LightGBM (mono+Huber)",
        "Within-Between LightGBM (Protocol + CV Space)",
        "Within-Between Physics (Protocol + Lagrangian)",
        "Within-Between Physics + CV Space",
    ]
    short = [
        "Baseline LGBM (all)",
        "Baseline LGBM (proto)",
        "Interact-constr + mono",
        "Constrained proto (Huber)",
        "Within-Between + Space",
        "Within-Between + Physics",
        "Within-Between Phys+Space",
    ]
    y_pos = np.arange(len(order))
    for i, m in enumerate(order):
        vals = repeats_df[repeats_df.model == m].R2.to_numpy()
        color = BLUE if i >= 5 else (AQUA if i >= 2 else INK_MUTED)
        axes[0].plot([vals.min(), vals.max()], [i, i], color=color, alpha=0.45, linewidth=4, solid_capstyle="round")
        axes[0].scatter(vals, np.full_like(vals, i, dtype=float), color=color, s=22, alpha=0.75, zorder=3)
        axes[0].scatter([vals.mean()], [i], color=INK, marker="D", s=45, zorder=4)
    axes[0].set_yticks(y_pos, short)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Pooled R² across 10 study-grouped 5-fold splits", title="Unseen-study R² across 10 fold assignments")

    y_oc = np.arange(len(OCEANS))
    series = [
        ("Dalvi 2026 (no protocol)", [DALVI_OCEAN_R2[o] for o in OCEANS], INK_MUTED, "s"),
        ("Baseline LGBM (all)", [looo_df[(looo_df.model == "Baseline LightGBM (all features)") & (looo_df.ocean == o)].R2.iloc[0] for o in OCEANS], ORANGE, "o"),
        ("Within-Between Physics", [looo_df[(looo_df.model == "Within-Between Physics (Protocol + Lagrangian)") & (looo_df.ocean == o)].R2.iloc[0] for o in OCEANS], BLUE, "D"),
    ]
    for offset, (lbl, vals, col, mk) in zip((-0.2, 0.0, 0.2), series):
        axes[1].scatter(vals, y_oc + offset, color=col, marker=mk, s=55, label=lbl, zorder=3)
    axes[1].axvline(0.0, color=INK_MUTED, linewidth=1, linestyle="--")
    axes[1].set_yticks(y_oc, OCEANS)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Held-out ocean R² (strict study exclusion)", title="Strict leave-one-ocean-out transfer", xlim=(-0.82, 0.75))
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "fig_v2_benchmark.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    cqr_rows = int_df[int_df.interval_method != "Study-grouped conformal (row)"]
    labels = [f"{r.model.replace(' LightGBM', '')}\n({r.interval_method.replace('Study-grouped ', '')})" for _, r in cqr_rows.iterrows()]
    yp = np.arange(len(cqr_rows))
    cols = [ORANGE if "Baseline" in r.model else BLUE for _, r in cqr_rows.iterrows()]
    axes[0].barh(yp, cqr_rows.median_width_log10, color=cols)
    for idx, (_, r) in enumerate(cqr_rows.iterrows()):
        axes[0].text(r.median_width_log10 + 0.06, idx, f"{r.median_width_log10:.2f} ({r.row_coverage * 100:.1f}% cov)", va="center", fontsize=8.5)
    axes[0].set_yticks(yp, labels)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Median 90% CQR interval width (log10 units)", title="Calibrated 90% CQR width and empirical coverage", xlim=(0, 5.1))

    for lbl, col in [("Option A Baseline LightGBM", ORANGE), ("Within-Between Physics LightGBM", BLUE)]:
        part = rc_df[rc_df.model == lbl]
        axes[1].plot(part.keep_fraction * 100, part.MAE, marker="o", color=col, label=lbl.replace(" LightGBM", ""))
    axes[1].axhline(0.708, color=AQUA, linestyle="--", linewidth=1.2, label="Oracle within-study median floor (0.708)")
    axes[1].invert_xaxis()
    axes[1].set(xlabel="Predictions retained (%)", ylabel="MAE (log10 units)", title="Selective prediction MAE vs coverage")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "fig_v2_cqr_selective.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    net_plot = net_df[net_df.model != "Oracle within-study median (noise floor)"]
    yn = np.arange(len(net_plot))
    axes[0].barh(yn, net_plot.repeat10_R2_mean, color=[ORANGE, AQUA, BLUE])
    for idx, (_, r) in enumerate(net_plot.iterrows()):
        axes[0].text(r.repeat10_R2_mean + 0.01, idx, f"{r.repeat10_R2_mean:.3f} (MAE {r.repeat10_MAE:.3f})", va="center", fontsize=9)
    axes[0].set_yticks(yn, ["Baseline LGBM\n(all features)", "Constrained Proto\n(mono+Huber)", "Within-Between\n+ Lagrangian"])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="10-repeat study-grouped R²", title="Surface-net-only subset (14,233 rows, 57 studies)", xlim=(0, 0.42))

    cens_plot = censored_df[censored_df.model != "Oracle within-study median (positive noise floor)"]
    yc = np.arange(len(cens_plot))
    axes[1].barh(yc, cens_plot.pos_R2_mean, color=[ORANGE, AQUA, BLUE, PURPLE])
    for idx, (_, r) in enumerate(cens_plot.iterrows()):
        axes[1].text(r.pos_R2_mean + 0.01, idx, f"{r.pos_R2_mean:.3f} (NLL {r.tobit_nll:.2f})", va="center", fontsize=8.5)
    axes[1].set_yticks(yc, ["Baseline LGBM\n(naive)", "Two-Part:\nConstrained Proto", "Two-Part:\nPhysics + Proto", "Tobit Left-Censored\nLGBM + Physics"])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="10-repeat unseen-study R² on positive detections", title="Handling the 6-log bulk-water zero cliff", xlim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "fig_v2_physics_and_tobit.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df):,} rows, {df.study.nunique()} studies, {df.organization.nunique()} organizations")

    print("\n=== BLOCK 1: GLOBAL BENCHMARK (10 REPEATS + ORG-CV + CHRONO + STRICT LOOO) ===")
    summary_df, repeats_df, looo_df = block1_global_benchmark(df)
    print(summary_df.to_string(index=False))
    print("\n--- STRICT LOOO BREAKDOWN (Key Models) ---")
    key_models = [
        "Baseline LightGBM (all features)",
        "Baseline LightGBM (protocol only)",
        "Constrained Protocol LightGBM (mono+Huber)",
        "Within-Between Physics (Protocol + Lagrangian)",
        "Within-Between Physics + CV Space",
    ]
    print(looo_df[looo_df.model.isin(key_models)][["model", "ocean", "dalvi_R2", "R2", "MAE", "within_1_log"]].to_string(index=False))

    print("\n=== BLOCK 2: SURFACE-NET-ONLY SUBSET & RAW PHYSICS RANKING ===")
    net_df, raw_rank_df = block2_surface_nets(df)
    print(raw_rank_df.to_string(index=False))
    print()
    print(net_df.to_string(index=False))

    print("\n=== BLOCK 3: LEFT-CENSORED TOBIT & TWO-PART DECOMPOSITION ===")
    censored_df = block3_censored_and_twopart(df)
    print(censored_df.to_string(index=False))

    print("\n=== BLOCK 4: STUDY-GROUPED CQR INTERVALS & SELECTIVE PREDICTION ===")
    int_df, rc_df = block4_intervals_and_selective(df)
    print(int_df.to_string(index=False))
    print()
    print(rc_df.to_string(index=False))

    print("\n=== BLOCK 5: SENSITIVITY AUDIT (10 REPEATS) ===")
    sens_df = block5_sensitivity(df)
    print(sens_df.to_string(index=False))

    make_figures(summary_df, repeats_df, looo_df, net_df, int_df, rc_df, censored_df)
    print("\nSaved all v2 CSVs and PNG figures to artifacts/.")


if __name__ == "__main__":
    main()
