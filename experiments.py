import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import BallTree, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor

from core import (
    ARTIFACTS,
    FEATURES,
    NUMERIC,
    OFFSET_C,
    SEED,
    Folds,
    conformal_quantile,
    design_matrices,
    lgbm,
    load_data,
    out_of_fold,
    predict_baseline,
    predict_constrained_protocol,
    predict_location_time_only,
    predict_protocol_only,
    predict_within_between_physics,
    repeated_folds,
    score,
    sensitivity_subsets,
    study_constant_oracle,
    weighted_quantile,
)
from environment import ENVIRONMENT_FEATURES, environmental_covariates

BOOSTER = dict(n_estimators=500, learning_rate=0.05, random_state=SEED, n_jobs=4)
KEEP_FRACTIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def clipped(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    low, high = train[NUMERIC].min(), train[NUMERIC].max()
    return train, test.assign(**{c: test[c].clip(low[c], high[c]) for c in NUMERIC})


def predict_study_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return test.study.map(train.groupby("study").y.mean()).fillna(train.y.mean()).to_numpy()


def predict_linear(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_tr, x_te, _ = design_matrices(*clipped(train, test), FEATURES)
    return LinearRegression().fit(x_tr, train.y).predict(x_te)


def predict_knn(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    knn = KNeighborsRegressor(n_neighbors=15, weights="distance", metric="haversine", algorithm="ball_tree")
    return knn.fit(np.radians(train[["lat", "lon"]]), train.y).predict(np.radians(test[["lat", "lon"]]))


def predict_random_forest(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_tr, x_te, _ = design_matrices(train, test, FEATURES)
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=4, random_state=SEED).fit(x_tr, train.y).predict(x_te)


def predict_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_tr, x_te, _ = design_matrices(train, test, FEATURES)
    return XGBRegressor(**BOOSTER, max_depth=6).fit(x_tr, train.y).predict(x_te)


def catboost(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    model = CatBoostRegressor(iterations=500, learning_rate=0.05, random_seed=SEED, thread_count=4, verbose=0)
    cats = [c for c in ("method_family", "method") if c in cols]
    return model.fit(train[cols], train.y, cat_features=cats).predict(test[cols])


def predict_catboost(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return catboost(train, test, FEATURES + ["method"])


def predict_xgboost_hurdle(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    positive = train[train.conc > 0]
    x_tr, x_te, _ = design_matrices(train, test, FEATURES)
    x_pos, x_te_pos, _ = design_matrices(positive, test, FEATURES)
    p = XGBClassifier(**BOOSTER).fit(x_tr, train.conc > 0).predict_proba(x_te)[:, 1]
    return p * XGBRegressor(**BOOSTER).fit(x_pos, positive.y).predict(x_te_pos) + (1 - p) * np.log10(OFFSET_C)


BASELINES = {
    "Study mean only": predict_study_mean,
    "Linear regression": predict_linear,
    "Spatial kNN": predict_knn,
    "Random forest": predict_random_forest,
    "XGBoost": predict_xgboost,
    "LightGBM": predict_baseline,
    "CatBoost": predict_catboost,
    "XGBoost hurdle": predict_xgboost_hurdle,
}


def mean_fold_r2(df: pd.DataFrame, folds: Folds, pred: np.ndarray) -> float:
    return float(np.mean([score(df.y.to_numpy()[te], pred[te], df.study.iloc[te])["R2"] for _, te in folds]))


def step1_baselines(df: pd.DataFrame) -> pd.DataFrame:
    random_folds = list(KFold(5, shuffle=True, random_state=SEED).split(df))
    study_folds = repeated_folds(df)
    train_c, test_c = df[df.year < 2015], df[df.year >= 2015]
    unseen_c = ~test_c.study.isin(train_c.study).to_numpy()

    rows = []
    for name, predict in BASELINES.items():
        random_sc = score(df.y.to_numpy(), out_of_fold(df, random_folds, predict), df.study)
        split_scores, fold_r2s = [], []
        for folds in study_folds:
            pred = out_of_fold(df, folds, predict)
            split_scores.append(score(df.y.to_numpy(), pred, df.study))
            fold_r2s.append(mean_fold_r2(df, folds, pred))
        grouped = pd.DataFrame(split_scores)
        pred_c = predict(train_c, test_c)
        rows.append({
            "model": name,
            "random_R2": random_sc["R2"],
            "random_MAE": random_sc["MAE"],
            "grouped_R2_mean": grouped.R2.mean(),
            "grouped_R2_min": grouped.R2.min(),
            "grouped_R2_max": grouped.R2.max(),
            "grouped_fold_R2_mean": float(np.mean(fold_r2s)),
            "grouped_MAE": grouped.MAE.mean(),
            "grouped_study_MAE": grouped.study_MAE.mean(),
            "grouped_within_1_log": grouped.within_1_log.mean(),
            "grouped_miss_gt_2_log": grouped.miss_gt_2_log.mean(),
            "chrono_R2": score(test_c.y.to_numpy(), pred_c, test_c.study)["R2"],
            "chrono_unseen_R2": score(test_c.y.to_numpy()[unseen_c], pred_c[unseen_c], test_c.study.iloc[unseen_c])["R2"],
        })
    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS / "audit_baselines.csv", index=False)
    return res


def intraclass_correlation(df: pd.DataFrame) -> float:
    fit = smf.mixedlm("y ~ 1", df, groups=df.study).fit()
    between = float(fit.cov_re.iloc[0, 0])
    return between / (between + fit.scale)


def grouped_r2(df: pd.DataFrame, fold_sets: list[Folds], predict) -> float:
    return float(np.mean([score(df.y.to_numpy(), out_of_fold(df, folds, predict), df.study)["R2"] for folds in fold_sets]))


def step2_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in sensitivity_subsets(df).items():
        sub = sub.reset_index(drop=True)
        random_folds = list(KFold(5, shuffle=True, random_state=SEED).split(sub))
        study_folds = repeated_folds(sub)
        rows.append({
            "subset": label,
            "rows": len(sub),
            "studies": sub.study.nunique(),
            "ICC": intraclass_correlation(sub),
            "y_variance": sub.y.var(),
            "random_R2": score(sub.y.to_numpy(), out_of_fold(sub, random_folds, predict_baseline), sub.study)["R2"],
            "random_location_time_R2": score(sub.y.to_numpy(), out_of_fold(sub, random_folds, predict_location_time_only), sub.study)["R2"],
            "grouped_R2": grouped_r2(sub, study_folds, predict_baseline),
            "grouped_protocol_R2": grouped_r2(sub, study_folds, predict_protocol_only),
            "grouped_location_time_R2": grouped_r2(sub, study_folds, predict_location_time_only),
        })
    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS / "audit_sensitivity.csv", index=False)
    return res


ENSEMBLE = [
    dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=s)
    for s in range(5)
] + [
    dict(n_estimators=500, learning_rate=0.04, max_depth=7, subsample=0.9, subsample_freq=1, colsample_bytree=0.9, random_state=s + 10)
    for s in range(5)
]


def nearest_km(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if train.empty:
        return np.full(len(test), np.inf)
    tree = BallTree(np.radians(train[["lat", "lon"]]), metric="haversine")
    return tree.query(np.radians(test[["lat", "lon"]]))[0][:, 0] * 6371.0


def interval_split(df: pd.DataFrame, folds: Folds, alpha: float = 0.10) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=[], dtype=float)
    for train_idx, test_idx in folds:
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        y_tr = train.y.to_numpy()

        point = predict_baseline(train, test)
        physics = predict_within_between_physics(train, test)
        ensemble = np.column_stack([lgbm(train, test, FEATURES, **cfg) for cfg in ENSEMBLE])

        random_oof = out_of_fold(train, list(KFold(5, shuffle=True, random_state=SEED).split(train)), predict_baseline)
        q_random = conformal_quantile(np.abs(y_tr - random_oof), alpha)

        inner = list(GroupKFold(4).split(train, groups=train.study))
        point_oof = out_of_fold(train, inner, predict_baseline)
        physics_oof = out_of_fold(train, inner, predict_within_between_physics)
        lo_oof = out_of_fold(train, inner, lambda tr, te: lgbm(tr, te, FEATURES, objective="quantile", alpha=alpha / 2))
        hi_oof = out_of_fold(train, inner, lambda tr, te: lgbm(tr, te, FEATURES, objective="quantile", alpha=1 - alpha / 2))
        lo_oof, hi_oof = np.minimum(lo_oof, hi_oof), np.maximum(lo_oof, hi_oof)

        study_w = 1.0 / train.groupby("study").y.transform("size").to_numpy()
        res = np.abs(y_tr - point_oof)
        cqr = np.maximum(lo_oof - y_tr, y_tr - hi_oof)
        q_row, q_study = conformal_quantile(res, alpha), weighted_quantile(res, study_w, 1 - alpha)
        q_cqr_row, q_cqr_study = conformal_quantile(cqr, alpha), weighted_quantile(cqr, study_w, 1 - alpha)
        q_physics = conformal_quantile(np.abs(y_tr - physics_oof), alpha)

        lo = lgbm(train, test, FEATURES, objective="quantile", alpha=alpha / 2)
        hi = lgbm(train, test, FEATURES, objective="quantile", alpha=1 - alpha / 2)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

        mesh = train.log_mesh.dropna()
        same_family_km = np.empty(len(test))
        for family in test.method_family.unique():
            mask = (test.method_family == family).to_numpy()
            same_family_km[mask] = nearest_km(train[train.method_family == family], test[mask])

        ens_mean, ens_std = ensemble.mean(axis=1), ensemble.std(axis=1, ddof=1)
        block = {
            "base_pred": point,
            "physics_pred": physics,
            "ens_std": ens_std,
            "ens90_lo": ens_mean - 1.645 * ens_std, "ens90_hi": ens_mean + 1.645 * ens_std,
            "ens95_lo": ens_mean - 1.96 * ens_std, "ens95_hi": ens_mean + 1.96 * ens_std,
            "random_lo": point - q_random, "random_hi": point + q_random,
            "study_row_lo": point - q_row, "study_row_hi": point + q_row,
            "study_study_lo": point - q_study, "study_study_hi": point + q_study,
            "cqr_row_lo": lo - q_cqr_row, "cqr_row_hi": hi + q_cqr_row,
            "cqr_study_lo": lo - q_cqr_study, "cqr_study_hi": hi + q_cqr_study,
            "physics_row_lo": physics - q_physics, "physics_row_hi": physics + q_physics,
            "unseen_method": (~test.method.isin(set(train.method))).astype(float).to_numpy(),
            "mesh_below_range": (test.log_mesh < mesh.min()).astype(float).to_numpy(),
            "mesh_above_range": (test.log_mesh > mesh.max()).astype(float).to_numpy(),
            "far_any_km": nearest_km(train, test),
            "far_family_km": same_family_km,
        }
        for col, values in block.items():
            out.loc[df.index[test_idx], col] = values
    return out


INTERVAL_METHODS = {
    "Ensemble spread (±1.645σ, nominal 90%)": "ens90",
    "Ensemble spread (±1.96σ, nominal 95%)": "ens95",
    "Random-CV conformal": "random",
    "Study-grouped conformal (row)": "study_row",
    "Study-grouped conformal (study)": "study_study",
    "Study-grouped CQR (row)": "cqr_row",
    "Study-grouped CQR (study)": "cqr_study",
    "Study-grouped conformal (row), Within-Between Physics": "physics_row",
}


def interval_summary(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    y = df.y.to_numpy()
    rows = []
    for name, key in INTERVAL_METHODS.items():
        lo, hi = split[f"{key}_lo"].to_numpy(), split[f"{key}_hi"].to_numpy()
        covered = (y >= lo) & (y <= hi)
        per_study = pd.Series(covered.astype(float)).groupby(df.study.to_numpy()).mean()
        rows.append({
            "method": name,
            "row_coverage": covered.mean(),
            "mean_study_coverage": per_study.mean(),
            "studies_ge_80pct": (per_study >= 0.8).mean(),
            "median_width_log10": np.median(hi - lo),
        })
    return pd.DataFrame(rows)


def mesh_out_of_range(split: pd.DataFrame) -> np.ndarray:
    return (split.mesh_below_range.to_numpy() + split.mesh_above_range.to_numpy()) > 0


def flag_masks(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    cqr_width = (split.cqr_study_hi - split.cqr_study_lo).to_numpy()
    return pd.DataFrame({
        "abs_err": np.abs(split.base_pred.to_numpy() - df.y.to_numpy()),
        "Unseen sampling method": split.unseen_method.to_numpy() > 0,
        "Bulk water or pump": (df.method_family != "surface net").to_numpy(),
        "Mesh outside training range": mesh_out_of_range(split),
        "Mesh below training range": split.mesh_below_range.to_numpy() > 0,
        "Mesh above training range": split.mesh_above_range.to_numpy() > 0,
        ">500 km to any training sample": split.far_any_km.to_numpy() > 500,
        ">500 km to same-family training sample": split.far_family_km.to_numpy() > 500,
        "Widest 25% of CQR intervals": cqr_width > np.quantile(cqr_width, 0.75),
    })


def flag_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    abs_err = pooled.abs_err.to_numpy()
    rows = []
    for name in pooled.columns.drop("abs_err"):
        mask = pooled[name].to_numpy()
        rows.append({
            "flag": name,
            "share": mask.mean(),
            "flagged_MAE": abs_err[mask].mean(),
            "unflagged_MAE": abs_err[~mask].mean(),
            "flagged_miss_gt_2_log": (abs_err[mask] > 2).mean(),
            "unflagged_miss_gt_2_log": (abs_err[~mask] > 2).mean(),
        })
    return pd.DataFrame(rows)


def risk_coverage(df: pd.DataFrame, split: pd.DataFrame, seed: int) -> pd.DataFrame:
    tie_breaker = np.random.default_rng(seed).random(len(df))
    y = df.y.to_numpy()
    base_err = np.abs(split.base_pred.to_numpy() - y)
    physics_err = np.abs(split.physics_pred.to_numpy() - y)
    cqr_width = (split.cqr_study_hi - split.cqr_study_lo).to_numpy()
    flag_count = (
        split.unseen_method.to_numpy()
        + (df.method_family != "surface net").to_numpy()
        + mesh_out_of_range(split)
        + (split.far_family_km.to_numpy() > 500)
    )
    curves = {
        ("Baseline LightGBM", "CQR interval width"): (base_err, cqr_width),
        ("Baseline LightGBM", "Ensemble spread"): (base_err, split.ens_std.to_numpy()),
        ("Baseline LightGBM", "Distance to same-family sample"): (base_err, np.minimum(split.far_family_km.to_numpy(), 25000.0)),
        ("Baseline LightGBM", "Flag count"): (base_err, flag_count),
        ("Within-Between Physics", "CQR interval width"): (physics_err, cqr_width),
    }
    rows = []
    for (model, scorer), (err, risk) in curves.items():
        order = np.lexsort((tie_breaker, risk))
        for keep in KEEP_FRACTIONS:
            kept = err[order[: round(keep * len(err))]]
            rows.append({
                "model": model, "scorer": scorer, "keep_fraction": keep,
                "MAE": kept.mean(), "within_1_log": (kept <= 1).mean(), "miss_gt_2_log": (kept > 2).mean(),
            })
    return pd.DataFrame(rows)


def averaged(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    return pd.concat(frames).groupby(keys, sort=False).mean().reset_index()


def step3_intervals_and_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = Parallel(n_jobs=3)(delayed(interval_split)(df, folds) for folds in repeated_folds(df))
    intervals = averaged([interval_summary(df, s) for s in splits], ["method"])
    intervals["half_width_factor_x"] = 10 ** (intervals.median_width_log10 / 2)
    flags = flag_summary(pd.concat([flag_masks(df, s) for s in splits], ignore_index=True))
    curves = averaged([risk_coverage(df, s, seed) for seed, s in enumerate(splits)], ["model", "scorer", "keep_fraction"])
    intervals.to_csv(ARTIFACTS / "audit_intervals.csv", index=False)
    flags.to_csv(ARTIFACTS / "audit_flags.csv", index=False)
    curves.to_csv(ARTIFACTS / "audit_risk_coverage.csv", index=False)
    return intervals, flags, curves


def step4_environment(df: pd.DataFrame) -> pd.DataFrame:
    env = df.join(environmental_covariates(df))
    fold_sets = repeated_folds(env)
    feature_sets = {
        "Current features": FEATURES,
        "Current + environmental": FEATURES + ENVIRONMENT_FEATURES,
        "Environmental only": ENVIRONMENT_FEATURES,
    }
    models = {
        "LightGBM": lambda cols: lambda tr, te: lgbm(tr, te, cols),
        "CatBoost": lambda cols: lambda tr, te: catboost(tr, te, cols),
    }
    rows = []
    for model, make in models.items():
        per_split = {label: pd.Series([score(env.y.to_numpy(), out_of_fold(env, f, make(cols)), env.study)["R2"] for f in fold_sets])
                     for label, cols in feature_sets.items()}
        for label, r2 in per_split.items():
            gain = r2 - per_split["Current features"]
            rows.append({"model": model, "features": label, "R2_mean": r2.mean(), "R2_min": r2.min(), "R2_max": r2.max(),
                         "gain_vs_current": gain.mean(), "wins_vs_current": int((gain > 0).sum())})
    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS / "audit_environment.csv", index=False)
    return res


CALIBRATION_SAMPLES = [0, 1, 3, 5, 10, 20]
CALIBRATED_MODELS = {
    "Global mean": predict_study_mean,
    "LightGBM": predict_baseline,
    "CatBoost": predict_catboost,
    "Constrained protocol": predict_constrained_protocol,
    "Within-Between Physics": predict_within_between_physics,
}


def calibration_pool(df: pd.DataFrame, rng: np.random.Generator, size: int = 20) -> np.ndarray:
    position = pd.Series(rng.random(len(df))).groupby(df.study.to_numpy()).rank(method="first").to_numpy() - 1
    limit = np.minimum(size, df.groupby("study").y.transform("size").to_numpy() // 2)
    return np.where(position < limit, position, np.nan)


def calibrated(df: pd.DataFrame, pred: np.ndarray, pool: np.ndarray, k: int, shrink: pd.Series) -> np.ndarray:
    shown = pool < k
    error = pd.Series((df.y.to_numpy() - pred)[shown]).groupby(df.study.to_numpy()[shown])
    offset = df.study.map(error.sum() / (error.size() + shrink.reindex(error.size().index))).fillna(0.0).to_numpy()
    return pred + offset


def shrinkage_by_study(df: pd.DataFrame, folds: Folds) -> pd.Series:
    shrink = {}
    for train_idx, test_idx in folds:
        icc = intraclass_correlation(df.iloc[train_idx])
        shrink |= dict.fromkeys(df.study.iloc[test_idx].unique(), (1 - icc) / icc)
    return pd.Series(shrink)


def within_study_spearman(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"y": df.y.to_numpy(), "pred": pred, "study": df.study.to_numpy()})
    return np.array([
        g.pred.corr(g.y, method="spearman") if g.pred.nunique() > 1 else 0.0
        for _, g in frame.groupby("study") if len(g) >= 20 and g.y.nunique() > 1
    ])


def step5_calibration_and_ranking(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    fold_sets = repeated_folds(df)
    pools = [calibration_pool(df, rng) for _ in fold_sets]
    shrinks = [shrinkage_by_study(df, folds) for folds in fold_sets]
    preds = {name: [out_of_fold(df, f, predict) for f in fold_sets] for name, predict in CALIBRATED_MODELS.items()}

    calibration_rows = []
    for k in CALIBRATION_SAMPLES:
        per_model = {}
        for name, split_preds in preds.items():
            per_model[name] = np.array([
                score(df.y.to_numpy()[np.isnan(pool)], calibrated(df, pred, pool, k, shrink)[np.isnan(pool)], df.study[np.isnan(pool)])["R2"]
                for pred, pool, shrink in zip(split_preds, pools, shrinks, strict=True)
            ])
        for name, r2 in per_model.items():
            calibration_rows.append({"samples": k, "model": name, "R2_mean": r2.mean(),
                                     "wins_vs_global_mean": int((r2 > per_model["Global mean"]).sum())})

    study_mean = df.groupby("study").y.transform("mean").to_numpy()
    ranking_rows = []
    for name, split_preds in preds.items():
        if name == "Global mean":
            continue
        rhos = [within_study_spearman(df, p) for p in split_preds]
        centred = [p - pd.Series(p).groupby(df.study.to_numpy()).transform("mean").to_numpy() + study_mean for p in split_preds]
        ranking_rows.append({
            "model": name,
            "studies": len(rhos[0]),
            "median_spearman": np.mean([np.median(r) for r in rhos]),
            "share_positive": np.mean([(r > 0).mean() for r in rhos]),
            "R2_centred_on_true_study_mean": np.mean([score(df.y.to_numpy(), c, df.study)["R2"] for c in centred]),
        })
    calibration, ranking = pd.DataFrame(calibration_rows), pd.DataFrame(ranking_rows)
    calibration.to_csv(ARTIFACTS / "audit_calibration.csv", index=False)
    ranking.to_csv(ARTIFACTS / "audit_ranking.csv", index=False)
    return calibration, ranking


def main() -> None:
    df = load_data()
    print(f"{len(df):,} rows, {df.study.nunique()} studies, {df.organization.nunique()} organizations, years {df.year.min()} to {df.year.max()}")
    print(f"Study-constant oracle: {study_constant_oracle(df.y, df.study)}")
    steps = [
        ("BASELINES", step1_baselines),
        ("SENSITIVITY", step2_sensitivity),
        ("INTERVALS AND FLAGS", step3_intervals_and_flags),
        ("ENVIRONMENTAL COVARIATES", step4_environment),
        ("CALIBRATION AND RANKING", step5_calibration_and_ranking),
    ]
    for title, step in steps:
        print(f"\n--- {title} ---")
        result = step(df)
        for frame in result if isinstance(result, tuple) else [result]:
            print(frame.to_string(index=False))


if __name__ == "__main__":
    import experiments

    experiments.main()
