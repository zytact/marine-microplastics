from functools import partial
from typing import NamedTuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from core import (
    ARTIFACTS,
    BARROWS_DOI,
    DALVI_OCEAN_R2,
    FEATURES,
    OCEANS,
    PROTOCOL_FEATURES,
    SEED,
    Folds,
    Predictor,
    as_archived,
    design_matrices,
    load_data,
    out_of_fold,
    predict_baseline,
    predict_constrained_protocol,
    predict_interaction_constrained,
    predict_location_time_only,
    predict_protocol_only,
    predict_with_space_stage,
    predict_within_between_physics,
    repeated_folds,
    score,
    sensitivity_subsets,
    shallow_protocol,
    study_constant_oracle,
    transport_offset,
    within_study_slope,
)

BASELINE = "Baseline LightGBM, all features"
CONSTRAINED = "Constrained protocol (shallow, Huber, monotonic)"
PHYSICS = "Within-Between Physics"

MODELS: dict[str, Predictor] = {
    BASELINE: predict_baseline,
    "LightGBM, protocol only": predict_protocol_only,
    "LightGBM, location and time only": predict_location_time_only,
    "Shallow protocol, squared loss": partial(shallow_protocol, huber=False, monotone_mesh=False),
    "Shallow protocol, Huber": partial(shallow_protocol, monotone_mesh=False),
    "Shallow protocol, monotonic, squared loss": partial(shallow_protocol, huber=False),
    CONSTRAINED: predict_constrained_protocol,
    "Interaction-constrained, monotonic": predict_interaction_constrained,
    "Constrained protocol + coordinate stage": partial(predict_with_space_stage, use_physics=False),
    PHYSICS: predict_within_between_physics,
    "Within-Between Physics + coordinate stage": partial(predict_with_space_stage, use_physics=True),
}


def oof_predictions(df: pd.DataFrame, fold_sets: list[Folds], predict: Predictor) -> list[np.ndarray]:
    return Parallel(n_jobs=3)(delayed(out_of_fold)(df, folds, predict) for folds in fold_sets)


def split_scores(df: pd.DataFrame, fold_sets: list[Folds], predict: Predictor) -> pd.DataFrame:
    return pd.DataFrame([score(df.y.to_numpy(), p, df.study) for p in oof_predictions(df, fold_sets, predict)])


def wins(a: pd.Series, b: pd.Series) -> int:
    return int((a.to_numpy() > b.to_numpy()).sum())


def strict_looo(df: pd.DataFrame, predict: Predictor) -> dict[str, float]:
    out = {}
    for ocean in OCEANS:
        test = df[df.ocean == ocean]
        train = df[(df.ocean != ocean) & ~df.study.isin(test.study)]
        out[ocean] = score(test.y.to_numpy(), predict(train, test), test.study)["R2"]
    return out


def block0_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    fold_sets = repeated_folds(df)
    y = df.y.to_numpy()
    study = df.study.to_numpy()
    true_within = y - pd.Series(y).groupby(study).transform("mean").to_numpy()
    rows = []
    for folds in fold_sets:
        pred = out_of_fold(df, folds, predict_baseline)
        pred_within = pred - pd.Series(pred).groupby(study).transform("mean").to_numpy()
        sq = (pred - y) ** 2
        rows.append({
            "MSE": sq.mean(),
            "within_MSE": np.mean((pred_within - true_within) ** 2),
            "true_within_var": np.mean(true_within**2),
            "pred_within_var": np.mean(pred_within**2),
            "within_corr": np.corrcoef(pred_within, true_within)[0, 1],
            "barrows_SSE_share": sq[(df.doi == BARROWS_DOI).to_numpy()].sum() / sq.sum(),
            "betas": [within_study_slope(df.iloc[tr]) for tr, _ in folds],
        })
    frame = pd.DataFrame(rows)
    betas = np.concatenate(frame.pop("betas").to_numpy())
    summary = frame.mean().to_frame().T.assign(beta_min=betas.min(), beta_max=betas.max(), beta_median=np.median(betas))

    bulk = df[df.method_family == "bulk water"]
    positive = bulk[bulk.conc > 0]
    eligible = bulk.study.value_counts().loc[lambda n: n >= 5].index
    for label, frame_ in [("all", bulk), ("positive", positive)]:
        means = frame_[frame_.study.isin(eligible)].groupby("study")[["log_mesh", "y"]].mean()
        summary[f"bulk_mesh_corr_{label}"] = means.log_mesh.corr(means.y)
    summary["bulk_studies"] = len(eligible)
    summary["bulk_zero_share"] = float((bulk.conc == 0).mean())
    summary.to_csv(ARTIFACTS / "v2_diagnosis.csv", index=False)
    return summary


def block1_global_benchmark(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_sets = repeated_folds(df)
    org_fold_sets = repeated_folds(df, "organization")
    train_c, test_c = df[df.year < 2015], df[df.year >= 2015]
    unseen_c = ~test_c.study.isin(train_c.study).to_numpy()

    per_split = {name: split_scores(df, fold_sets, predict) for name, predict in MODELS.items()}
    per_org_split = {name: split_scores(df, org_fold_sets, predict) for name, predict in MODELS.items()}
    rows = []
    for name, predict in MODELS.items():
        splits = per_split[name]
        pred_c = predict(train_c, test_c)
        looo = strict_looo(df, predict)
        rows.append({
            "model": name,
            "R2_mean": splits.R2.mean(),
            "R2_min": splits.R2.min(),
            "R2_max": splits.R2.max(),
            "wins_vs_baseline": wins(splits.R2, per_split[BASELINE].R2),
            "wins_vs_constrained": wins(splits.R2, per_split[CONSTRAINED].R2),
            "MAE": splits.MAE.mean(),
            "study_MAE": splits.study_MAE.mean(),
            "within_1_log": splits.within_1_log.mean(),
            "miss_gt_2_log": splits.miss_gt_2_log.mean(),
            "org_R2": per_org_split[name].R2.mean(),
            "org_R2_min": per_org_split[name].R2.min(),
            "org_R2_max": per_org_split[name].R2.max(),
            "chrono_R2": score(test_c.y.to_numpy(), pred_c, test_c.study)["R2"],
            "chrono_unseen_R2": score(test_c.y.to_numpy()[unseen_c], pred_c[unseen_c], test_c.study.iloc[unseen_c])["R2"],
            **{f"looo_{o.split()[0].lower()}": r for o, r in looo.items()},
            "looo_mean": float(np.mean(list(looo.values()))),
            "looo_min": float(np.min(list(looo.values()))),
        })
    oracle = study_constant_oracle(df.y, df.study)
    rows.append({"model": "Study-constant oracle", "R2_mean": oracle["R2"], "MAE": oracle["MAE"], "study_MAE": oracle["study_MAE"],
                 "within_1_log": oracle["within_1_log"], "miss_gt_2_log": oracle["miss_gt_2_log"]})
    summary = pd.DataFrame(rows)
    repeats = pd.concat([s.assign(model=n, split=range(len(s))) for n, s in per_split.items()])
    summary.to_csv(ARTIFACTS / "v2_global_benchmark.csv", index=False)
    repeats.to_csv(ARTIFACTS / "v2_global_repeats.csv", index=False)
    return summary, repeats


def dalvi_row(name: str, frame: pd.DataFrame, predict: Predictor) -> dict[str, float | str]:
    ln = frame.assign(y=np.log1p(frame.conc))
    random_folds = list(KFold(5, shuffle=True, random_state=SEED).split(ln))
    random_sc = score(ln.y.to_numpy(), out_of_fold(ln, random_folds, predict), ln.study)
    return {"model": name, "random_R2": random_sc["R2"], "random_MAE": random_sc["MAE"], **strict_looo(ln, predict)}


def block2_dalvi_scale(df: pd.DataFrame) -> pd.DataFrame:
    models = {
        "LightGBM, location and time only": predict_location_time_only,
        BASELINE: predict_baseline,
        "LightGBM, protocol only": predict_protocol_only,
        CONSTRAINED: predict_constrained_protocol,
        PHYSICS: predict_within_between_physics,
    }
    rows = [{"model": "Dalvi et al. (2026), reported", "random_R2": 0.651, "random_MAE": 0.619, **DALVI_OCEAN_R2}]
    rows += [dalvi_row(name, df, predict) for name, predict in models.items()]
    rows += [dalvi_row(f"{name}, as archived", as_archived(df), models[name]) for name in (BASELINE, CONSTRAINED)]
    res = pd.DataFrame(rows)
    res["looo_mean"] = res[OCEANS].mean(axis=1)
    res.to_csv(ARTIFACTS / "v2_dalvi_scale.csv", index=False)
    return res


def block3_surface_nets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nets = df[df.method_family == "surface net"].reset_index(drop=True)
    fold_sets = repeated_folds(nets)
    org_fold_sets = repeated_folds(nets, "organization")

    features = ["lag_acc", "drifter_speed", "abs_lat", "lat", "lon"]
    eligible = [g for _, g in nets.groupby("study") if len(g) >= 20 and g.y.nunique() > 1 and (g[features].nunique() > 1).all()]
    rhos = {f: np.array([g[f].corr(g.y, method="spearman") for g in eligible]) for f in features}
    ranking = pd.DataFrame([
        {"feature": f, "studies": len(eligible), "median_spearman": np.median(r), "share_positive": np.mean(r > 0)}
        for f, r in rhos.items()
    ])
    ranking.to_csv(ARTIFACTS / "v2_surface_net_ranking.csv", index=False)

    net_cols = ["log_mesh", "log_depth"]
    models = {
        BASELINE: predict_baseline,
        CONSTRAINED: partial(shallow_protocol, cols=net_cols),
        PHYSICS: partial(predict_within_between_physics, cols=net_cols),
    }
    per_split = {name: split_scores(nets, fold_sets, predict) for name, predict in models.items()}
    rows = [{
        "model": name,
        "rows": len(nets),
        "studies": nets.study.nunique(),
        "R2_mean": s.R2.mean(),
        "R2_min": s.R2.min(),
        "R2_max": s.R2.max(),
        "wins_vs_baseline": wins(s.R2, per_split[BASELINE].R2),
        "wins_vs_constrained": wins(s.R2, per_split[CONSTRAINED].R2),
        "MAE": s.MAE.mean(),
        "org_R2": split_scores(nets, org_fold_sets, models[name]).R2.mean(),
    } for name, s in per_split.items()]
    oracle = study_constant_oracle(nets.y, nets.study)
    rows.append({"model": "Study-constant oracle", "R2_mean": oracle["R2"], "MAE": oracle["MAE"]})
    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS / "v2_surface_nets.csv", index=False)
    return res, ranking


def detection_limits(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def bucket(frame: pd.DataFrame) -> pd.Series:
        return frame.method_family + "_" + np.where(frame.mesh < 0.05, "fine", "coarse")

    positive = train[train.conc > 0]
    by_bucket = positive.assign(b=bucket(positive)).groupby(["b", "study"]).conc.min().groupby("b").median()
    by_family = positive.groupby(["method_family", "study"]).conc.min().groupby("method_family").median()
    fallback = float(positive.conc.quantile(0.01))

    def lookup(frame: pd.DataFrame) -> np.ndarray:
        return np.log10(bucket(frame).map(by_bucket).fillna(frame.method_family.map(by_family)).fillna(fallback).to_numpy())

    return lookup(train), lookup(test)


def inverse_mills(z: np.ndarray) -> np.ndarray:
    safe = z > -30.0
    out = np.empty_like(z)
    out[safe] = np.exp(norm.logpdf(z[safe]) - log_ndtr(z[safe]))
    out[~safe] = -z[~safe] - 1.0 / z[~safe]
    return out


def gaussian_nll(z: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    return 0.5 * ((z - mean) / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)


def detection_nll(p_detect: np.ndarray, positive: np.ndarray) -> np.ndarray:
    p = np.clip(p_detect, 1e-6, 1 - 1e-6)
    return np.where(positive, -np.log(p), -np.log(1 - p))


def mixed_scores(p_detect: np.ndarray, level: np.ndarray, z: np.ndarray, positive: np.ndarray, sigma: float) -> dict[str, np.ndarray]:
    detection = detection_nll(p_detect, positive)
    level_nll = np.where(positive, gaussian_nll(z, level, sigma), np.nan)
    return {"nll": detection + np.nan_to_num(level_nll), "detection_nll": detection, "level_nll": level_nll,
            "p": p_detect, "level": level}


def tobit_detection(f: np.ndarray, lod: np.ndarray, sigma: float) -> np.ndarray:
    return 1 - norm.cdf((lod - f) / sigma)


def tobit_level(f: np.ndarray, lod: np.ndarray, sigma: float) -> np.ndarray:
    return f + sigma * inverse_mills((f - lod) / sigma)


def positive_rmse(frame: pd.DataFrame, level: np.ndarray) -> float:
    positive = (frame.conc > 0).to_numpy()
    return float(np.sqrt(np.mean((np.log10(frame.conc.to_numpy()[positive]) - level[positive]) ** 2)))


def log_concentration(frame: pd.DataFrame, lod: np.ndarray) -> np.ndarray:
    return np.where(frame.conc > 0, np.log10(np.maximum(frame.conc.to_numpy(), 1e-12)), lod)


def fit_tobit(train: pd.DataFrame, test: pd.DataFrame, sigma: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    lod_tr, _ = detection_limits(train, test)
    positive = (train.conc > 0).to_numpy()
    z = log_concentration(train, lod_tr)
    beta = within_study_slope(train[positive].assign(z=z[positive]), target="z")
    z_adj = z - transport_offset(train, beta)
    lod_adj = lod_tr - transport_offset(train, beta)

    def objective(f: np.ndarray, _: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
        u = (lod_adj - f) / sigma
        lam = inverse_mills(u)
        grad = np.where(positive, (f - z_adj) / sigma**2, lam / sigma)
        hess = np.where(positive, 1 / sigma**2, np.maximum(lam * (lam + u) / sigma**2, 1e-6))
        return grad, hess

    x_tr, x_te, names = design_matrices(train, test, PROTOCOL_FEATURES)
    start = float(np.median(z_adj[positive]))
    params = dict(objective=objective, learning_rate=0.05, max_depth=3, num_leaves=7, min_child_samples=30,
                  monotone_constraints=[-1 if n == "log_mesh" else 0 for n in names], verbose=-1, n_jobs=4, seed=SEED)
    booster = lgb.train(params, lgb.Dataset(x_tr, label=z_adj, init_score=np.full(len(train), start)), num_boost_round=150)
    return start + booster.predict(x_tr) + transport_offset(train, beta), start + booster.predict(x_te) + transport_offset(test, beta)


class LevelModel(NamedTuple):
    predict: Predictor
    classifier_cols: list[str]
    keeps_zeros: bool


TOBIT = "Tobit LightGBM, Within-Between Physics"
LEVEL_MODELS = {
    "Baseline LightGBM, zeros at -3": LevelModel(predict_baseline, FEATURES, keeps_zeros=True),
    "Two-part, baseline LightGBM": LevelModel(predict_baseline, FEATURES, keeps_zeros=False),
    "Two-part, constrained protocol": LevelModel(predict_constrained_protocol, PROTOCOL_FEATURES, keeps_zeros=False),
    "Two-part, Within-Between Physics": LevelModel(predict_within_between_physics, PROTOCOL_FEATURES, keeps_zeros=False),
}
CENSORED_MODELS = [*LEVEL_MODELS, TOBIT]


SIGMA_GRID = np.round(np.arange(0.3, 2.51, 0.05), 2)


def censored_fold(name: str, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    _, lod_te = detection_limits(train, test)
    z_te = log_concentration(test, lod_te)
    pos_te = (test.conc > 0).to_numpy()
    inner = list(GroupKFold(4).split(train, groups=train.study))

    if name == TOBIT:
        f_oof, lod_oof = np.empty(len(train)), np.empty(len(train))
        for itr, iva in inner:
            f_oof[iva] = fit_tobit(train.iloc[itr], train.iloc[iva])[1]
            lod_oof[iva] = detection_limits(train.iloc[itr], train.iloc[iva])[1]
        pos_tr = (train.conc > 0).to_numpy()
        z_tr = log_concentration(train, lod_oof)

        def inner_score(sigma: float) -> float:
            level = tobit_level(f_oof, lod_oof, sigma)
            return mixed_scores(tobit_detection(f_oof, lod_oof, sigma), level, z_tr, pos_tr, positive_rmse(train, level))["nll"].mean()

        sigma = min(SIGMA_GRID, key=inner_score)
        level_sigma = positive_rmse(train, tobit_level(f_oof, lod_oof, sigma))
        f = fit_tobit(train, test)[1]
        return mixed_scores(tobit_detection(f, lod_te, sigma), tobit_level(f, lod_te, sigma), z_te, pos_te, level_sigma)

    model = LEVEL_MODELS[name]
    fit_rows = train if model.keeps_zeros else train[train.conc > 0].assign(y=lambda t: np.log10(t.conc))
    level_oof = out_of_fold(fit_rows, list(GroupKFold(4).split(fit_rows, groups=fit_rows.study)), model.predict)
    level_sigma = positive_rmse(fit_rows, level_oof)

    x_tr, x_te, _ = design_matrices(train, test, model.classifier_cols)
    classifier = LGBMClassifier(n_estimators=150, max_depth=3, num_leaves=7, min_child_samples=30, learning_rate=0.05,
                                random_state=SEED, n_jobs=4, verbose=-1)
    p = classifier.fit(x_tr, (train.conc > 0).astype(int)).predict_proba(x_te)[:, 1]
    return mixed_scores(p, model.predict(fit_rows, test), z_te, pos_te, level_sigma)


def censored_split(df: pd.DataFrame, folds: Folds, name: str) -> dict[str, float]:
    parts = {k: np.empty(len(df)) for k in ("nll", "detection_nll", "level_nll", "p", "level", "limit")}
    for tr, te in folds:
        for k, v in censored_fold(name, df.iloc[tr], df.iloc[te]).items():
            parts[k][te] = v
        parts["limit"][te] = detection_limits(df.iloc[tr], df.iloc[te])[1]
    positive = (df.conc > 0).to_numpy()
    z = np.log10(df.conc.to_numpy()[positive])
    sc = score(z, parts["level"][positive], df.study.iloc[positive])
    return {"nll": parts["nll"].mean(), "detection_nll": parts["detection_nll"].mean(),
            "level_nll": parts["level_nll"][positive].mean(), "auc": roc_auc_score(positive, parts["p"]),
            "pos_R2": sc["R2"], "pos_MAE": sc["MAE"], "pos_within_1_log": sc["within_1_log"], "pos_miss_gt_2_log": sc["miss_gt_2_log"],
            "pos_below_limit": (z < parts["limit"][positive]).mean()}


def block4_censored(df: pd.DataFrame) -> pd.DataFrame:
    fold_sets = repeated_folds(df)
    rows = []
    for name in CENSORED_MODELS:
        splits = pd.DataFrame(Parallel(n_jobs=3)(delayed(censored_split)(df, folds, name) for folds in fold_sets))
        rows.append({"model": name, **splits.mean().to_dict(), "pos_R2_min": splits.pos_R2.min(), "pos_R2_max": splits.pos_R2.max()})
    positive = df[df.conc > 0]
    oracle = study_constant_oracle(np.log10(positive.conc), positive.study)
    rows.append({"model": "Study-constant oracle", "pos_R2": oracle["R2"], "pos_MAE": oracle["MAE"],
                 "pos_within_1_log": oracle["within_1_log"], "pos_miss_gt_2_log": oracle["miss_gt_2_log"]})
    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS / "v2_censored.csv", index=False)
    return res


def block5_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    models = {
        BASELINE: predict_baseline,
        "LightGBM, protocol only": predict_protocol_only,
        "LightGBM, location and time only": predict_location_time_only,
        CONSTRAINED: predict_constrained_protocol,
        PHYSICS: predict_within_between_physics,
        f"{PHYSICS}, min 10 per leaf": partial(predict_within_between_physics, min_child_samples=10),
    }
    rows = []
    for label, sub in sensitivity_subsets(df).items():
        sub = sub.reset_index(drop=True)
        fold_sets = repeated_folds(sub)
        org_fold_sets = repeated_folds(sub, "organization")
        per_split = {name: split_scores(sub, fold_sets, predict) for name, predict in models.items()}
        per_org_split = {name: split_scores(sub, org_fold_sets, predict) for name, predict in models.items()}
        largest_org_share = sub.organization.value_counts(normalize=True).iloc[0]
        for name, s in per_split.items():
            rows.append({"subset": label, "rows": len(sub), "studies": sub.study.nunique(), "largest_org_share": largest_org_share,
                         "model": name, "R2_mean": s.R2.mean(), "wins_vs_baseline": wins(s.R2, per_split[BASELINE].R2),
                         "MAE": s.MAE.mean(), "study_MAE": s.study_MAE.mean(), "miss_gt_2_log": s.miss_gt_2_log.mean(),
                         "org_R2": per_org_split[name].R2.mean(),
                         "org_wins_vs_baseline": wins(per_org_split[name].R2, per_org_split[BASELINE].R2)})
    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS / "v2_sensitivity.csv", index=False)
    return res


SHALLOW = "Shallow protocol, squared loss"
BOOTSTRAP_PAIRS = [(CONSTRAINED, BASELINE), (PHYSICS, BASELINE), (PHYSICS, CONSTRAINED), (CONSTRAINED, SHALLOW)]
BOOTSTRAP_METRICS = ["R2", "MAE", "study_MAE"]


def block6_study_bootstrap(df: pd.DataFrame, draws: int = 1000) -> pd.DataFrame:
    fold_sets = repeated_folds(df)
    names = sorted({name for pair in BOOTSTRAP_PAIRS for name in pair})
    preds = {name: oof_predictions(df, fold_sets, MODELS[name]) for name in names}
    y = df.y.to_numpy()
    members = list(df.groupby("study").indices.values())

    def deltas(rows: np.ndarray, groups: pd.Series) -> dict[tuple[str, str, str], float]:
        scores = {name: [score(y[rows], p[rows], groups) for p in preds[name]] for name in names}
        return {(a, b, m): float(np.mean([sa[m] - sb[m] for sa, sb in zip(scores[a], scores[b], strict=True)]))
                for a, b in BOOTSTRAP_PAIRS for m in BOOTSTRAP_METRICS}

    observed = deltas(np.arange(len(df)), df.study)
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(draws):
        picked = rng.integers(len(members), size=len(members))
        rows = np.concatenate([members[i] for i in picked])
        groups = pd.Series(np.repeat(np.arange(len(picked)), [len(members[i]) for i in picked]))
        samples.append(deltas(rows, groups))
    boot = pd.DataFrame(samples)
    res = pd.DataFrame([
        {"model": a, "against": b, "metric": m, "delta": observed[a, b, m],
         "ci_low": boot[a, b, m].quantile(0.025), "ci_high": boot[a, b, m].quantile(0.975),
         "share_positive": (boot[a, b, m] > 0).mean()}
        for a, b, m in observed
    ])
    res.to_csv(ARTIFACTS / "v2_study_bootstrap.csv", index=False)
    return res


def main() -> None:
    df = load_data()
    print(f"{len(df):,} rows, {df.study.nunique()} studies, {df.organization.nunique()} organizations")
    blocks = [
        ("DIAGNOSIS", block0_diagnosis),
        ("GLOBAL BENCHMARK", block1_global_benchmark),
        ("DALVI SCALE", block2_dalvi_scale),
        ("SURFACE NETS", block3_surface_nets),
        ("CENSORED", block4_censored),
        ("SENSITIVITY", block5_sensitivity),
        ("STUDY BOOTSTRAP", block6_study_bootstrap),
    ]
    for title, block in blocks:
        print(f"\n=== {title} ===")
        result = block(df)
        for frame in result if isinstance(result, tuple) else [result]:
            print(frame.to_string(index=False))


if __name__ == "__main__":
    import experiments_v2

    experiments_v2.main()
