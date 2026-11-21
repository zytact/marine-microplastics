import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import BallTree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/noaa_microplastics.csv")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
SEED = 0
C = 1e-3

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

UNIT_FIXES = {
    "https://doi.org/10.1016/j.envpol.2018.02.062": 1e6,
    "https://doi.org/10.1016/j.marpolbul.2020.111239": 1e6,
}

BARROWS_DOI = "https://doi.org/10.1016/j.envpol.2018.02.062"
DEHM_DOI = "https://doi.org/10.1016/j.marpolbul.2020.111239"

NUMERIC = ["lat", "lon", "year", "month_sin", "month_cos", "log_mesh", "log_depth"]
CATEGORICAL = ["method_family"]
FEATURES = NUMERIC + CATEGORICAL
CATBOOST_FEATURES = FEATURES + ["method"]
LOCATION_TIME = ["lat", "lon", "year", "month_sin", "month_cos"]
PROTOCOL_LGBM = ["log_mesh", "log_depth", "method_family"]
PROTOCOL_CB = ["log_mesh", "log_depth", "method_family", "method"]

DALVI_OCEAN_R2 = {
    "Indian Ocean": 0.166,
    "Pacific Ocean": 0.082,
    "Atlantic Ocean": 0.010,
    "Arctic Ocean": -0.253,
    "Southern Ocean": -0.708,
}

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c93b3b"
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


def linked_groups(a: pd.Series, b: pd.Series) -> pd.Series:
    group = a.astype(str)
    while not (merged := group.groupby(b).transform("min").groupby(a).transform("min")).equals(group):
        group = merged
    return group


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
    water["conc"] *= water.doi.map(UNIT_FIXES).fillna(1)
    method = water.method.str.lower().str.replace("-", " ")
    family = method.map(METHOD_FAMILY).fillna("bulk water")
    month = water.date.dt.month
    return water.assign(
        method=method,
        method_family=family,
        year=water.date.dt.year,
        month_sin=np.sin(2 * np.pi * month / 12),
        month_cos=np.cos(2 * np.pi * month / 12),
        log_mesh=np.log10(water.mesh),
        log_depth=np.log1p(water.depth),
        study=linked_groups(water.doi, water.accession),
        y=np.log10(water.conc + C),
    )


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    num_cols = [c for c in features if c in NUMERIC]
    cat_cols = [c for c in features if c not in NUMERIC]
    transformers = []
    if num_cols:
        transformers.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    return ColumnTransformer(transformers)


def fit_predict_lgbm(train: pd.DataFrame, test: pd.DataFrame, features: list[str], **params) -> np.ndarray:
    defaults = dict(n_estimators=500, learning_rate=0.05, random_state=SEED, n_jobs=4, verbose=-1)
    pipe = make_pipeline(make_preprocessor(features), LGBMRegressor(**(defaults | params)))
    pipe.fit(train[features], train.y)
    return pipe.predict(test[features])


def fit_predict_hgb(train: pd.DataFrame, test: pd.DataFrame, features: list[str], **params) -> np.ndarray:
    defaults = dict(max_iter=500, learning_rate=0.05, max_depth=6, random_state=SEED)
    pipe = make_pipeline(make_preprocessor(features), HistGradientBoostingRegressor(**(defaults | params)))
    pipe.fit(train[features], train.y)
    return pipe.predict(test[features])


def fit_predict_rf(train: pd.DataFrame, test: pd.DataFrame, features: list[str], **params) -> np.ndarray:
    defaults = dict(n_estimators=300, min_samples_leaf=3, n_jobs=4, random_state=SEED)
    pipe = make_pipeline(make_preprocessor(features), RandomForestRegressor(**(defaults | params)))
    pipe.fit(train[features], train.y)
    return pipe.predict(test[features])


def score_predictions(y: np.ndarray, pred: np.ndarray, studies: pd.Series) -> dict[str, float]:
    err = pred - y
    abs_err = np.abs(err)
    study_mae = pd.Series(abs_err).groupby(studies.to_numpy()).mean().mean()
    return {
        "R2": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "study_MAE": float(study_mae),
        "within_1_log": float(np.mean(abs_err <= 1.0)),
        "miss_gt_2_log": float(np.mean(abs_err > 2.0)),
    }


def run_cv(df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], fn, features: list[str]) -> np.ndarray:
    pred = np.empty(len(df))
    for train_idx, test_idx in folds:
        pred[test_idx] = fn(df.iloc[train_idx], df.iloc[test_idx], features)
    return pred


def step2_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "Full dataset": df,
        "Drop bongo net (Arteaga 2024)": df[df.method != "bongo net"],
        "Drop Barrows & Dehm (unit-fixed)": df[~df.doi.isin([BARROWS_DOI, DEHM_DOI])],
        "Drop Barrows, Dehm & bongo": df[(~df.doi.isin([BARROWS_DOI, DEHM_DOI])) & (df.method != "bongo net")],
    }
    rows = []
    for label, sub in subsets.items():
        sub = sub.reset_index(drop=True)
        null_mixed = smf.mixedlm("y ~ 1", sub, groups=sub.study).fit()
        between = float(null_mixed.cov_re.iloc[0, 0])
        icc = between / (between + null_mixed.scale)

        rand_folds = list(KFold(5, shuffle=True, random_state=SEED).split(sub))
        grp_folds = list(GroupKFold(5).split(sub, groups=sub.study))

        lgbm_rand = score_predictions(sub.y.to_numpy(), run_cv(sub, rand_folds, fit_predict_lgbm, FEATURES), sub.study)
        lgbm_grp = score_predictions(sub.y.to_numpy(), run_cv(sub, grp_folds, fit_predict_lgbm, FEATURES), sub.study)
        lgbm_proto = score_predictions(sub.y.to_numpy(), run_cv(sub, grp_folds, fit_predict_lgbm, PROTOCOL_LGBM), sub.study)
        lgbm_loc = score_predictions(sub.y.to_numpy(), run_cv(sub, grp_folds, fit_predict_lgbm, LOCATION_TIME), sub.study)
        rf_grp = score_predictions(sub.y.to_numpy(), run_cv(sub, grp_folds, fit_predict_rf, FEATURES), sub.study)

        rows.append({
            "subset": label,
            "rows": len(sub),
            "studies": sub.study.nunique(),
            "ICC": icc,
            "LGBM_random_R2": lgbm_rand["R2"],
            "LGBM_grouped_R2": lgbm_grp["R2"],
            "LGBM_grouped_MAE": lgbm_grp["MAE"],
            "LGBM_grouped_within1": lgbm_grp["within_1_log"],
            "LGBM_protocol_only_R2": lgbm_proto["R2"],
            "LGBM_loc_time_only_R2": lgbm_loc["R2"],
            "RF_grouped_R2": rf_grp["R2"],
        })
    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS / "step2_sensitivity.csv", index=False)
    return res


def step3_leave_one_ocean_out(df: pd.DataFrame) -> pd.DataFrame:
    oceans = ["Indian Ocean", "Pacific Ocean", "Atlantic Ocean", "Arctic Ocean", "Southern Ocean"]
    valid = df[df.ocean.isin(oceans)].reset_index(drop=True)

    feature_sets = {
        "Location & time only": (LOCATION_TIME, LOCATION_TIME),
        "Protocol only": (PROTOCOL_LGBM, PROTOCOL_CB),
        "All features": (FEATURES, CATBOOST_FEATURES),
    }

    rows = []
    for ocean in oceans:
        test_mask = valid.ocean == ocean
        test = valid[test_mask]
        test_studies = set(test.study.unique())
        shared_studies = test_studies & set(valid.loc[~test_mask, "study"].unique())

        for strict in (False, True):
            if strict:
                train = valid[(~test_mask) & (~valid.study.isin(test_studies))]
            else:
                train = valid[~test_mask]

            for feat_label, (lgbm_feats, _) in feature_sets.items():
                for model_name, fn, feats in [
                    ("LightGBM", fit_predict_lgbm, lgbm_feats),
                    ("HistGB", fit_predict_hgb, lgbm_feats),
                    ("RandomForest", fit_predict_rf, lgbm_feats),
                ]:
                    pred = fn(train, test, feats)
                    m = score_predictions(test.y.to_numpy(), pred, test.study)
                    rows.append({
                        "ocean": ocean,
                        "test_rows": len(test),
                        "test_studies": test.study.nunique(),
                        "shared_studies_in_train": len(shared_studies),
                        "strict_study_exclusion": strict,
                        "features": feat_label,
                        "model": model_name,
                        "dalvi_R2": DALVI_OCEAN_R2[ocean],
                        **m,
                    })

    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS / "step3_leave_one_ocean_out.csv", index=False)
    return res


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w) / np.sum(w)
    idx = np.searchsorted(cum, q, side="left")
    return float(v[min(idx, len(v) - 1)])


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def step4_intervals(df: pd.DataFrame, alpha: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_folds = list(GroupKFold(5).split(df, groups=df.study))
    n = len(df)
    y = df.y.to_numpy()

    point_pred = np.empty(n)
    ens_mean = np.empty(n)
    ens_std = np.empty(n)

    naive_lo, naive_hi = np.empty(n), np.empty(n)
    grp_row_lo, grp_row_hi = np.empty(n), np.empty(n)
    grp_study_lo, grp_study_hi = np.empty(n), np.empty(n)
    cqr_row_lo, cqr_row_hi = np.empty(n), np.empty(n)
    cqr_study_lo, cqr_study_hi = np.empty(n), np.empty(n)

    ens_configs = [
        dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=s)
        for s in range(5)
    ] + [
        dict(n_estimators=500, learning_rate=0.04, max_depth=7, subsample=0.9, subsample_freq=1, colsample_bytree=0.9, random_state=s + 10)
        for s in range(5)
    ]

    for train_idx, test_idx in outer_folds:
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)

        p_test = fit_predict_lgbm(train, test, FEATURES)
        point_pred[test_idx] = p_test

        ens_preds = np.column_stack([fit_predict_lgbm(train, test, FEATURES, **cfg) for cfg in ens_configs])
        ens_mean[test_idx] = ens_preds.mean(axis=1)
        ens_std[test_idx] = ens_preds.std(axis=1, ddof=1)

        inner_rand = list(KFold(5, shuffle=True, random_state=SEED).split(train))
        rand_oof = np.empty(len(train))
        for itr, iva in inner_rand:
            rand_oof[iva] = fit_predict_lgbm(train.iloc[itr], train.iloc[iva], FEATURES)
        q_naive = conformal_quantile(np.abs(train.y.to_numpy() - rand_oof), alpha)
        naive_lo[test_idx] = p_test - q_naive
        naive_hi[test_idx] = p_test + q_naive

        inner_grp = list(GroupKFold(4).split(train, groups=train.study))
        grp_oof = np.empty(len(train))
        qlo_oof = np.empty(len(train))
        qhi_oof = np.empty(len(train))
        for itr, iva in inner_grp:
            tr_in, va_in = train.iloc[itr], train.iloc[iva]
            grp_oof[iva] = fit_predict_lgbm(tr_in, va_in, FEATURES)
            qlo_oof[iva] = fit_predict_lgbm(tr_in, va_in, FEATURES, objective="quantile", alpha=alpha / 2)
            qhi_oof[iva] = fit_predict_lgbm(tr_in, va_in, FEATURES, objective="quantile", alpha=1 - alpha / 2)

        abs_res = np.abs(train.y.to_numpy() - grp_oof)
        study_counts = train.groupby("study")["y"].transform("size").to_numpy()
        study_weights = 1.0 / study_counts

        q_grp_row = conformal_quantile(abs_res, alpha)
        q_grp_study = weighted_quantile(abs_res, study_weights, 1 - alpha)
        grp_row_lo[test_idx], grp_row_hi[test_idx] = p_test - q_grp_row, p_test + q_grp_row
        grp_study_lo[test_idx], grp_study_hi[test_idx] = p_test - q_grp_study, p_test + q_grp_study

        cqr_scores = np.maximum(qlo_oof - train.y.to_numpy(), train.y.to_numpy() - qhi_oof)
        q_cqr_row = conformal_quantile(cqr_scores, alpha)
        q_cqr_study = weighted_quantile(cqr_scores, study_weights, 1 - alpha)

        raw_lo_test = fit_predict_lgbm(train, test, FEATURES, objective="quantile", alpha=alpha / 2)
        raw_hi_test = fit_predict_lgbm(train, test, FEATURES, objective="quantile", alpha=1 - alpha / 2)
        raw_lo_test, raw_hi_test = np.minimum(raw_lo_test, raw_hi_test), np.maximum(raw_lo_test, raw_hi_test)

        cqr_row_lo[test_idx], cqr_row_hi[test_idx] = raw_lo_test - q_cqr_row, raw_hi_test + q_cqr_row
        cqr_study_lo[test_idx], cqr_study_hi[test_idx] = raw_lo_test - q_cqr_study, raw_hi_test + q_cqr_study

    methods = {
        "Dalvi-style 10-run GBDT spread (±1.645σ, nominal 90%)": (ens_mean - 1.645 * ens_std, ens_mean + 1.645 * ens_std),
        "Dalvi-style 10-run GBDT spread (±1.96σ, nominal 95%)": (ens_mean - 1.96 * ens_std, ens_mean + 1.96 * ens_std),
        "Random-CV conformal (row-weighted, 90%)": (naive_lo, naive_hi),
        "Study-grouped conformal (row-weighted, 90%)": (grp_row_lo, grp_row_hi),
        "Study-grouped conformal (study-weighted, 90%)": (grp_study_lo, grp_study_hi),
        "Study-grouped CQR (row-weighted, 90%)": (cqr_row_lo, cqr_row_hi),
        "Study-grouped CQR (study-weighted, 90%)": (cqr_study_lo, cqr_study_hi),
    }

    summary_rows = []
    for name, (lo, hi) in methods.items():
        covered = (y >= lo) & (y <= hi)
        width = hi - lo
        per_study_cov = pd.Series(covered.astype(float)).groupby(df.study.to_numpy()).mean()
        summary_rows.append({
            "method": name,
            "row_coverage": float(covered.mean()),
            "mean_study_coverage": float(per_study_cov.mean()),
            "median_study_coverage": float(per_study_cov.median()),
            "studies_ge_80pct": float((per_study_cov >= 0.80).mean()),
            "mean_width_log10": float(width.mean()),
            "median_width_log10": float(np.median(width)),
            "half_width_factor_x": float(10 ** (np.median(width) / 2)),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ARTIFACTS / "step4_intervals.csv", index=False)

    oof_frame = pd.DataFrame({
        "y": y,
        "study": df.study,
        "method": df.method,
        "method_family": df.method_family,
        "log_mesh": df.log_mesh,
        "lat": df.lat,
        "lon": df.lon,
        "pred": point_pred,
        "abs_err": np.abs(point_pred - y),
        "ens_std": ens_std,
        "cqr_width": cqr_study_hi - cqr_study_lo,
        "cqr_covered": ((y >= cqr_study_lo) & (y <= cqr_study_hi)).astype(int),
        "dalvi_covered": ((y >= ens_mean - 1.96 * ens_std) & (y <= ens_mean + 1.96 * ens_std)).astype(int),
    })
    oof_frame.to_csv(ARTIFACTS / "step4_oof_predictions.csv", index=False)
    return summary, oof_frame


def step5_reliability_flags(df: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_folds = list(GroupKFold(5).split(df, groups=df.study))
    n = len(df)

    unseen_method = np.zeros(n, dtype=bool)
    non_surface_net = (df.method_family != "surface net").to_numpy()
    mesh_out_of_range = np.zeros(n, dtype=bool)
    mesh_extreme = np.zeros(n, dtype=bool)
    dist_any_km = np.empty(n)
    dist_same_fam_km = np.empty(n)

    for train_idx, test_idx in outer_folds:
        tr = df.iloc[train_idx]
        te = df.iloc[test_idx]

        seen_methods = set(tr.method.unique())
        unseen_method[test_idx] = ~te.method.isin(seen_methods).to_numpy()

        tr_mesh = tr.log_mesh.dropna()
        lo_m, hi_m = tr_mesh.min(), tr_mesh.max()
        q05, q95 = tr_mesh.quantile(0.05), tr_mesh.quantile(0.95)
        te_mesh = te.log_mesh.to_numpy()
        mesh_out_of_range[test_idx] = np.isnan(te_mesh) | (te_mesh < lo_m) | (te_mesh > hi_m)
        mesh_extreme[test_idx] = np.isnan(te_mesh) | (te_mesh < q05) | (te_mesh > q95)

        tree_all = BallTree(np.radians(tr[["lat", "lon"]]), metric="haversine")
        dist_any_km[test_idx] = tree_all.query(np.radians(te[["lat", "lon"]]))[0][:, 0] * 6371.0

        for fam in te.method_family.unique():
            fam_mask = (te.method_family == fam).to_numpy()
            te_sub_idx = test_idx[fam_mask]
            tr_fam = tr[tr.method_family == fam]
            if len(tr_fam) == 0:
                dist_same_fam_km[te_sub_idx] = np.inf
            else:
                tree_fam = BallTree(np.radians(tr_fam[["lat", "lon"]]), metric="haversine")
                dist_same_fam_km[te_sub_idx] = tree_fam.query(np.radians(te.iloc[fam_mask][["lat", "lon"]]))[0][:, 0] * 6371.0

    far_any_500 = dist_any_km > 500
    far_same_fam_500 = dist_same_fam_km > 500
    high_cqr_width = oof.cqr_width.to_numpy() > np.quantile(oof.cqr_width.to_numpy(), 0.75)

    any_protocol_flag = unseen_method | mesh_out_of_range | non_surface_net
    any_support_flag = unseen_method | mesh_out_of_range | far_same_fam_500

    flags = {
        "Unseen raw sampling method": unseen_method,
        "Non-surface-net family (bulk/pump)": non_surface_net,
        "Mesh size outside training min/max": mesh_out_of_range,
        "Mesh size outside training 5-95%": mesh_extreme,
        "Distance > 500 km (any training sample)": far_any_500,
        "Distance > 500 km (same sampling family)": far_same_fam_500,
        "Top 25% CQR interval width": high_cqr_width,
        "Any protocol flag (unseen method / OOR mesh / non-net)": any_protocol_flag,
        "Any support flag (unseen method / OOR mesh / >500 km same family)": any_support_flag,
    }

    abs_err = oof.abs_err.to_numpy()
    rows = []
    for name, mask in flags.items():
        for group_label, m in [("Flagged", mask), ("Unflagged", ~mask)]:
            if m.sum() == 0:
                continue
            e = abs_err[m]
            rows.append({
                "flag": name,
                "group": group_label,
                "rows": int(m.sum()),
                "share": float(m.mean()),
                "MAE": float(e.mean()),
                "RMSE": float(np.sqrt(np.mean(e**2))),
                "within_1_log": float((e <= 1.0).mean()),
                "miss_gt_2_log": float((e > 2.0).mean()),
            })
    flag_df = pd.DataFrame(rows)
    flag_df.to_csv(ARTIFACTS / "step5_flags.csv", index=False)

    risk_scores = {
        "CQR interval width": oof.cqr_width.to_numpy(),
        "Dalvi ensemble std": oof.ens_std.to_numpy(),
        "Distance to same-family sample": np.minimum(dist_same_fam_km, 25000.0),
        "Combined flag count": (
            unseen_method.astype(int)
            + non_surface_net.astype(int)
            + mesh_out_of_range.astype(int)
            + far_same_fam_500.astype(int)
        ),
    }

    rc_rows = []
    for score_name, score_vals in risk_scores.items():
        order = np.argsort(score_vals)
        for keep_frac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            k = max(1, int(round(keep_frac * n)))
            kept = abs_err[order[:k]]
            rc_rows.append({
                "scorer": score_name,
                "keep_fraction": keep_frac,
                "MAE": float(kept.mean()),
                "within_1_log": float((kept <= 1.0).mean()),
                "miss_gt_2_log": float((kept > 2.0).mean()),
            })
    rc_df = pd.DataFrame(rc_rows)
    rc_df.to_csv(ARTIFACTS / "step5_risk_coverage.csv", index=False)
    return flag_df, rc_df


def make_figures(looo: pd.DataFrame, intervals: pd.DataFrame, rc: pd.DataFrame) -> None:
    oceans = ["Indian Ocean", "Pacific Ocean", "Atlantic Ocean", "Arctic Ocean", "Southern Ocean"]
    strict_lgbm = looo[(looo.strict_study_exclusion) & (looo.model == "LightGBM")]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y_pos = np.arange(len(oceans))
    series = [
        ("Dalvi 2026 (no protocol)", [DALVI_OCEAN_R2[o] for o in oceans], INK_MUTED, "s"),
        ("LightGBM: location & time only", [strict_lgbm[(strict_lgbm.ocean == o) & (strict_lgbm.features == "Location & time only")].R2.iloc[0] for o in oceans], ORANGE, "o"),
        ("LightGBM: protocol only", [strict_lgbm[(strict_lgbm.ocean == o) & (strict_lgbm.features == "Protocol only")].R2.iloc[0] for o in oceans], AQUA, "^"),
        ("LightGBM: all features", [strict_lgbm[(strict_lgbm.ocean == o) & (strict_lgbm.features == "All features")].R2.iloc[0] for o in oceans], BLUE, "D"),
    ]
    for offset, (label, vals, color, marker) in zip((-0.24, -0.08, 0.08, 0.24), series):
        ax.scatter(np.clip(vals, -1.0, 1.0), y_pos + offset, color=color, marker=marker, s=55, label=label, zorder=3)
    ax.axvline(0, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.set_yticks(y_pos, oceans)
    ax.invert_yaxis()
    ax.set(xlabel="Held-out ocean R² on log10 scale", title="Leave-one-ocean-out comparison against Dalvi et al. (2026)")
    ax.set_xlim(-0.85, 0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "fig_step3_looo_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    short_names = [
        "Dalvi ±1.645σ (90%)",
        "Dalvi ±1.96σ (95%)",
        "Random-CV conformal",
        "Study conformal (row)",
        "Study conformal (study)",
        "Study CQR (row)",
        "Study CQR (study)",
    ]
    ypos = np.arange(len(intervals))
    axes[0].barh(ypos - 0.18, intervals.row_coverage * 100, height=0.35, color=BLUE, label="All rows")
    axes[0].barh(ypos + 0.18, intervals.mean_study_coverage * 100, height=0.35, color=AQUA, label="Mean per study")
    axes[0].axvline(90, color=RED, linewidth=1.2, linestyle="--", label="90% target")
    axes[0].set_yticks(ypos, short_names)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Empirical coverage on unseen studies (%)", title="Interval coverage on unseen studies", xlim=(0, 105))
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    axes[1].barh(ypos, intervals.median_width_log10, color=ORANGE)
    axes[1].set_yticks(ypos, short_names)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Median interval width (log10 units)", title="Interval width")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "fig_step4_interval_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    colors = {
        "CQR interval width": BLUE,
        "Dalvi ensemble std": INK_MUTED,
        "Distance to same-family sample": AQUA,
        "Combined flag count": ORANGE,
    }
    for scorer, grp in rc.groupby("scorer"):
        axes[0].plot(grp.keep_fraction * 100, grp.MAE, marker="o", label=scorer, color=colors[scorer])
        axes[1].plot(grp.keep_fraction * 100, grp.miss_gt_2_log * 100, marker="o", label=scorer, color=colors[scorer])
    axes[0].invert_xaxis()
    axes[1].invert_xaxis()
    axes[0].set(xlabel="Predictions retained (%)", ylabel="MAE (log10 units)", title="Selective prediction: MAE vs coverage")
    axes[1].set(xlabel="Predictions retained (%)", ylabel="Large misses (>100x error, %)", title="Selective prediction: large misses vs coverage")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "fig_step5_risk_coverage.png", dpi=160)
    plt.close(fig)


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df):,} rows, {df.study.nunique()} studies, {df.accession.nunique()} accessions, {df.doi.nunique()} DOIs")

    print("\n--- STEP 2: DATA SENSITIVITY ---")
    s2 = step2_sensitivity(df)
    print(s2.to_string(index=False))

    print("\n--- STEP 3: LEAVE-ONE-OCEAN-OUT ---")
    s3 = step3_leave_one_ocean_out(df)
    pivot = s3[(s3.strict_study_exclusion) & (s3.model == "LightGBM")][
        ["ocean", "test_rows", "test_studies", "shared_studies_in_train", "features", "R2", "MAE", "within_1_log", "dalvi_R2"]
    ]
    print(pivot.to_string(index=False))

    print("\n--- STEP 4: PREDICTION INTERVALS ---")
    s4, oof = step4_intervals(df)
    print(s4.to_string(index=False))

    print("\n--- STEP 5: RELIABILITY FLAGS ---")
    s5_flags, s5_rc = step5_reliability_flags(df, oof)
    print(s5_flags.to_string(index=False))
    print("\n--- STEP 5: RISK-COVERAGE ---")
    print(s5_rc.to_string(index=False))

    make_figures(s3, s4, s5_rc)
    print("\nSaved all CSVs and PNG figures to artifacts/.")


if __name__ == "__main__":
    main()
