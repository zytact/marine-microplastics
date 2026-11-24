import time
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from core import (
    ARTIFACTS,
    REPEATS,
    SEED,
    Folds,
    Predictor,
    design_matrices,
    load_data,
    predict_baseline,
    predict_constrained_protocol,
    predict_interaction_constrained,
    predict_within_between_physics,
    repeated_folds,
)
from experiments_v2 import BASELINE, CONSTRAINED, LEVEL_MODELS, PHYSICS, censored_split, detection_nll, split_scores

TWO_PART_PHYSICS = "Two-part, Within-Between Physics"
DETECTION_COLS = LEVEL_MODELS[TWO_PART_PHYSICS].classifier_cols
ROUNDS = 150
CLASSIFIER = dict(n_estimators=ROUNDS, max_depth=3, num_leaves=7, min_child_samples=30, learning_rate=0.05,
                  random_state=SEED, n_jobs=4, verbose=-1)
AT_HALF = "0.5"
AT_F1 = "F1-optimal on validation"


class Scored(NamedTuple):
    positive: np.ndarray
    staged: np.ndarray

    @property
    def final(self) -> np.ndarray:
        return self.staged[-1]


class OuterFold(NamedTuple):
    train: Scored
    validation: Scored
    test: Scored
    test_index: np.ndarray
    train_rate: float


def detected(frame: pd.DataFrame) -> np.ndarray:
    return (frame.conc > 0).to_numpy()


def fit_staged(train: pd.DataFrame, frames: list[pd.DataFrame]) -> list[Scored]:
    classifier = LGBMClassifier(**CLASSIFIER).fit(design_matrices(train, train, DETECTION_COLS)[0], detected(train).astype(int))
    return [
        Scored(detected(frame), np.stack([classifier.predict_proba(x, num_iteration=k)[:, 1] for k in range(1, ROUNDS + 1)]))
        for frame in frames
        for x in [design_matrices(train, frame, DETECTION_COLS)[1]]
    ]


def inner_validation(train: pd.DataFrame) -> Scored:
    staged = np.empty((ROUNDS, len(train)))
    for itr, iva in GroupKFold(4).split(train, groups=train.study):
        staged[:, iva] = fit_staged(train.iloc[itr], [train.iloc[iva]])[0].staged
    return Scored(detected(train), staged)


def outer_fold(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> OuterFold:
    train, test = df.iloc[tr], df.iloc[te]
    on_train, on_test = fit_staged(train, [train, test])
    return OuterFold(on_train, inner_validation(train), on_test, te, float(detected(train).mean()))


def f1_optimal_threshold(scored: Scored) -> float:
    precision, recall, thresholds = precision_recall_curve(scored.positive, scored.final)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=precision + recall > 0)
    return float(thresholds[np.argmax(f1[:-1])])


def detection_metrics(positive: np.ndarray, p: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(positive, predicted, labels=[False, True]).ravel()
    return {
        "accuracy": accuracy_score(positive, predicted),
        "precision": precision_score(positive, predicted, zero_division=0),
        "recall": recall_score(positive, predicted, zero_division=0),
        "F1": f1_score(positive, predicted, zero_division=0),
        "AUC": roc_auc_score(positive, p),
        "log_loss": float(detection_nll(p, positive).mean()),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn, "n": len(positive),
    }


def at_threshold(scored: Scored, threshold: float) -> dict[str, float]:
    return detection_metrics(scored.positive, scored.final, scored.final >= threshold) | {"threshold": threshold}


def fold_mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return pd.DataFrame(rows).mean().to_dict()


def pooled(folds: list[OuterFold], n: int, values: list[np.ndarray]) -> np.ndarray:
    out = np.empty(n, dtype=values[0].dtype)
    for fold, v in zip(folds, values, strict=True):
        out[fold.test_index] = v
    return out


def assignment_metrics(folds: list[OuterFold], n: int) -> list[dict[str, float | str]]:
    thresholds = [f1_optimal_threshold(f.validation) for f in folds]
    positive = pooled(folds, n, [f.test.positive for f in folds])
    p = pooled(folds, n, [f.test.final for f in folds])
    at_f1 = pooled(folds, n, [f.test.final >= t for f, t in zip(folds, thresholds, strict=True)])
    rate = pooled(folds, n, [np.full(len(f.test_index), f.train_rate) for f in folds])
    rows = [
        {"split": "train", "threshold_rule": AT_HALF, **fold_mean([at_threshold(f.train, 0.5) for f in folds])},
        {"split": "validation", "threshold_rule": AT_HALF, **fold_mean([at_threshold(f.validation, 0.5) for f in folds])},
        {"split": "validation", "threshold_rule": AT_F1,
         **fold_mean([at_threshold(f.validation, t) for f, t in zip(folds, thresholds, strict=True)])},
        {"split": "test", "threshold_rule": AT_HALF, **detection_metrics(positive, p, p >= 0.5), "threshold": 0.5},
        {"split": "test", "threshold_rule": AT_F1, **detection_metrics(positive, p, at_f1), "threshold": float(np.mean(thresholds))},
        {"split": "test, per outer fold", "threshold_rule": AT_HALF, **fold_mean([at_threshold(f.test, 0.5) for f in folds])},
        {"split": "test, always detected (p = training detection rate)", "threshold_rule": "none",
         **detection_metrics(positive, rate, np.ones(n, dtype=bool)), "AUC": np.nan},
    ]
    return rows


def round_curve(scored: Scored) -> pd.DataFrame:
    return pd.DataFrame({
        "round": np.arange(1, ROUNDS + 1),
        "log_loss": [detection_nll(p, scored.positive).mean() for p in scored.staged],
        "AUC": [roc_auc_score(scored.positive, p) for p in scored.staged],
    })


def assignment_curve(folds: list[OuterFold], n: int) -> pd.DataFrame:
    test = Scored(pooled(folds, n, [f.test.positive for f in folds]),
                  np.stack([pooled(folds, n, [f.test.staged[k] for f in folds]) for k in range(ROUNDS)]))
    return pd.concat([
        pd.concat([round_curve(f.train) for f in folds]).groupby("round").mean().reset_index().assign(split="train"),
        pd.concat([round_curve(f.validation) for f in folds]).groupby("round").mean().reset_index().assign(split="validation"),
        round_curve(test).assign(split="test"),
        pd.concat([round_curve(f.test) for f in folds]).groupby("round").mean().reset_index().assign(split="test, per outer fold"),
    ])


def classifier_assignment(df: pd.DataFrame, folds: Folds, assignment: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = [outer_fold(df, tr, te) for tr, te in folds]
    metrics = pd.DataFrame(assignment_metrics(outer, len(df))).assign(assignment=assignment)
    return metrics, assignment_curve(outer, len(df)).assign(assignment=assignment)


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    values = frame.drop(columns="assignment")
    grouped = values.groupby(keys, sort=False)
    return grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_sd")).reset_index()


def block_a_classifier(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = Parallel(n_jobs=3)(delayed(classifier_assignment)(df, folds, i) for i, folds in enumerate(repeated_folds(df)))
    metrics = summarize(pd.concat([m for m, _ in results]), ["split", "threshold_rule"])
    curve = summarize(pd.concat([c for _, c in results]), ["split", "round"])
    metrics.to_csv(ARTIFACTS / "v3_classifier_metrics.csv", index=False)
    curve.to_csv(ARTIFACTS / "v3_classifier_curve.csv", index=False)
    return metrics, curve


def plot_classifier_curve(curve: pd.DataFrame) -> None:
    from paper.figures import AQUA, BLUE, FULL_WIDTH, NEUTRAL, panel_label, save

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4))
    for split, label, color in [("train", "Train", NEUTRAL), ("validation", "Validation (inner study folds)", AQUA),
                                ("test", "Test (unseen studies)", BLUE)]:
        part = curve[curve.split == split]
        for ax, metric in zip(axes, ("log_loss", "AUC"), strict=True):
            ax.plot(part["round"], part[f"{metric}_mean"], color=color, label=label)
    for ax, ylabel, letter in zip(axes, ("Log-loss", "AUC"), "ab", strict=True):
        ax.set_xlabel("Boosting rounds")
        ax.set_ylabel(ylabel)
        panel_label(ax, letter)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.tight_layout()
    save(fig, "fig_classifier_curve")


def random_folds(df: pd.DataFrame, repeats: int = REPEATS) -> list[Folds]:
    return [list(KFold(5, shuffle=True, random_state=r).split(df)) for r in range(repeats)]


REGRESSION_MODELS: dict[str, Predictor] = {
    BASELINE: predict_baseline,
    CONSTRAINED: predict_constrained_protocol,
    "Interaction-constrained, monotonic": predict_interaction_constrained,
    PHYSICS: predict_within_between_physics,
}
REGRESSION_METRICS = ["R2", "MAE", "study_MAE", "within_1_log", "miss_gt_2_log"]
DETECTED_MODELS = ["Baseline LightGBM, zeros at -3", "Two-part, baseline LightGBM", TWO_PART_PHYSICS]
DETECTED_METRICS = {"pos_R2": "R2", "pos_MAE": "MAE", "pos_within_1_log": "within_1_log",
                    "pos_miss_gt_2_log": "miss_gt_2_log", "auc": "detection_AUC", "detection_nll": "detection_log_loss"}


def comparison_rows(model: str, scope: str, splits: pd.DataFrame, unseen: pd.Series, names: dict[str, str]) -> list[dict[str, float | str]]:
    return [{
        "model": model, "scope": scope, "metric": names[column],
        "random_mean": splits[column].mean(), "random_sd": splits[column].std(), "random_seed0": splits[column].iloc[0],
        "unseen": unseen[column], "random_minus_unseen": splits[column].mean() - unseen[column],
    } for column in names]


def check_baseline(splits: pd.DataFrame) -> None:
    expected = pd.read_csv(ARTIFACTS / "audit_baselines.csv").set_index("model").loc["LightGBM"]
    reproduced = np.isclose(splits.R2.iloc[0], expected.random_R2, atol=1e-9) and np.isclose(splits.MAE.iloc[0], expected.random_MAE, atol=1e-9)
    if not reproduced:
        raise SystemExit(f"Baseline random split R2 {splits.R2.iloc[0]:.6f} does not reproduce {expected.random_R2:.6f}")


def block_b_random_split(df: pd.DataFrame) -> pd.DataFrame:
    fold_sets = random_folds(df)
    benchmark = pd.read_csv(ARTIFACTS / "v2_global_benchmark.csv").set_index("model")
    censored = pd.read_csv(ARTIFACTS / "v2_censored.csv").set_index("model")
    rows = []
    for name, predict in REGRESSION_MODELS.items():
        splits = split_scores(df, fold_sets, predict)
        if name == BASELINE:
            check_baseline(splits)
        rows += comparison_rows(name, "all samples", splits, benchmark.loc[name].rename(lambda c: c.removesuffix("_mean")),
                                {m: m for m in REGRESSION_METRICS})
    for name in DETECTED_MODELS:
        splits = pd.DataFrame(Parallel(n_jobs=3)(delayed(censored_split)(df, folds, name) for folds in fold_sets))
        rows += comparison_rows(name, "detected samples", splits, censored.loc[name], DETECTED_METRICS)
    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS / "v3_random_split.csv", index=False)
    return res


def main() -> None:
    df = load_data()
    print(f"{len(df):,} rows, {df.study.nunique()} studies, detected share {detected(df).mean():.4f}")
    started = time.perf_counter()
    metrics, curve = block_a_classifier(df)
    plot_classifier_curve(curve)
    print(f"\n=== DETECTION CLASSIFIER ({time.perf_counter() - started:.0f} s) ===")
    print(metrics.T.to_string())
    started = time.perf_counter()
    random_split = block_b_random_split(df)
    print(f"\n=== RANDOM SPLIT ({time.perf_counter() - started:.0f} s) ===")
    print(random_split.to_string(index=False))


if __name__ == "__main__":
    import experiments_v3

    experiments_v3.main()
