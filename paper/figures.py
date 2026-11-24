from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARTIFACTS = Path("artifacts")
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e6e5e1", "#9a9893"
FULL_WIDTH, COLUMN_WIDTH = 7.16, 3.5
OCEANS = ["Indian Ocean", "Pacific Ocean", "Atlantic Ocean", "Arctic Ocean", "Southern Ocean"]

BASELINE = "Baseline LightGBM, all features"
CONSTRAINED = "Constrained protocol (shallow, Huber, monotonic)"
PHYSICS = "Within-Between Physics"
MODEL_STYLE = {
    BASELINE: ("Baseline LightGBM", ORANGE, "o"),
    CONSTRAINED: ("Constrained protocol", AQUA, "^"),
    PHYSICS: ("Within-Between Physics", BLUE, "D"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def benchmark() -> None:
    summary = pd.read_csv(ARTIFACTS / "v2_global_benchmark.csv").set_index("model")
    repeats = pd.read_csv(ARTIFACTS / "v2_global_repeats.csv")
    order = [
        BASELINE,
        "LightGBM, protocol only",
        "Shallow protocol, squared loss",
        "Shallow protocol, Huber",
        "Shallow protocol, monotonic, squared loss",
        CONSTRAINED,
        "Interaction-constrained, monotonic",
        "Constrained protocol + coordinate stage",
        PHYSICS,
        "Within-Between Physics + coordinate stage",
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    for i, model in enumerate(order):
        values = repeats[repeats.model == model].R2.to_numpy()
        color = MODEL_STYLE.get(model, (None, NEUTRAL))[1]
        left.plot([values.min(), values.max()], [i, i], color=color, alpha=0.35, linewidth=3, solid_capstyle="round")
        left.scatter(values, np.full(len(values), i), color=color, s=9, zorder=3, linewidths=0)
        left.scatter([values.mean()], [i], color=INK, marker="|", s=60, zorder=4)
    left.axvline(summary.loc["Study-constant oracle", "R2_mean"], color=MUTED, linestyle="--", linewidth=0.8)
    labels = {
        CONSTRAINED: "Constrained protocol",
        "Interaction-constrained, monotonic": "Interaction-constrained",
        "Shallow protocol, squared loss": "Shallow (Table 6.1)",
        "Shallow protocol, Huber": "Shallow, Huber (Table 6.1)",
        "Shallow protocol, monotonic, squared loss": "Shallow, monotonic (Table 6.1)",
    }
    left.set_yticks(range(len(order)), [labels.get(m, m) for m in order])
    left.invert_yaxis()
    left.set_xlabel(r"Unseen-study $R^2$ (10 study-fold assignments)")
    panel_label(left, "a")

    y = np.arange(len(OCEANS))
    for offset, (model, (label, color, marker)) in zip((-0.22, 0.0, 0.22), MODEL_STYLE.items()):
        values = [summary.loc[model, f"looo_{o.split()[0].lower()}"] for o in OCEANS]
        right.scatter(values, y + offset, color=color, marker=marker, s=22, label=label, zorder=3)
    right.axvline(0, color=MUTED, linewidth=0.8)
    right.set_yticks(y, [o.replace(" Ocean", "") for o in OCEANS])
    right.invert_yaxis()
    right.set_xlabel(r"Held-out basin $R^2$, $\log_{10}$ target")
    right.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    panel_label(right, "b")
    fig.tight_layout()
    save(fig, "fig_benchmark")


def dalvi_scale() -> None:
    table = pd.read_csv(ARTIFACTS / "v2_dalvi_scale.csv").set_index("model")
    series = [
        ("Dalvi et al. (2026), reported", "Dalvi (2026), reported", NEUTRAL, "s"),
        ("LightGBM, location and time only", "LightGBM, location and time", ORANGE, "o"),
        (CONSTRAINED, "Constrained protocol", AQUA, "^"),
    ]
    floor = -1.1
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
    y = np.arange(len(OCEANS))
    for offset, (model, label, color, marker) in zip((-0.22, 0.0, 0.22), series):
        values = table.loc[model, OCEANS].to_numpy(dtype=float)
        ax.scatter(np.maximum(values, floor), y + offset, color=color, marker=marker, s=20, label=label, zorder=3)
        for i, v in enumerate(values):
            if v < floor:
                ax.annotate(f"{v:.2f}".replace("-", "\u2212"), (floor, i + offset), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6, color=MUTED)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(floor - 0.05, 0.85)
    ax.set_yticks(y, [o.replace(" Ocean", "") for o in OCEANS])
    ax.invert_yaxis()
    ax.set_xlabel(r"Held-out basin $R^2$, $\ln(c+1)$ target")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=1)
    fig.tight_layout()
    save(fig, "fig_dalvi_scale")


def surface_and_censored() -> None:
    nets = pd.read_csv(ARTIFACTS / "v2_surface_nets.csv").set_index("model")
    censored = pd.read_csv(ARTIFACTS / "v2_censored.csv").set_index("model")
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.3))

    models = list(MODEL_STYLE)
    y = np.arange(len(models))
    left.barh(y, nets.loc[models, "R2_mean"], color=[MODEL_STYLE[m][1] for m in models], height=0.6)
    for i, m in enumerate(models):
        low, high = nets.loc[m, ["R2_min", "R2_max"]]
        left.plot([low, high], [i, i], color=INK, linewidth=0.8)
    left.axvline(nets.loc["Study-constant oracle", "R2_mean"], color=MUTED, linestyle="--", linewidth=0.8)
    left.axvline(0, color=MUTED, linewidth=0.8)
    left.set_yticks(y, [MODEL_STYLE[m][0] for m in models])
    left.invert_yaxis()
    left.set_xlabel(r"Surface-net $R^2$ (10 study-fold assignments)")
    panel_label(left, "a")

    rows = [
        ("Baseline LightGBM, zeros at -3", "Baseline, zeros at $-3$", ORANGE),
        ("Two-part, baseline LightGBM", "Two-part, baseline", ORANGE),
        ("Two-part, constrained protocol", "Two-part, constrained", AQUA),
        ("Two-part, Within-Between Physics", "Two-part, Physics", BLUE),
        ("Tobit LightGBM, Within-Between Physics", "Tobit, Physics", VIOLET),
    ]
    y = np.arange(len(rows))
    right.barh(y, [censored.loc[m, "pos_R2"] for m, _, _ in rows], color=[c for _, _, c in rows], height=0.6)
    for i, (m, _, _) in enumerate(rows):
        right.text(0.91, i, f"NLL {censored.loc[m, 'nll']:.2f}", va="center", fontsize=7, color=MUTED)
    right.axvline(censored.loc["Study-constant oracle", "pos_R2"], color=MUTED, linestyle="--", linewidth=0.8)
    right.set_yticks(y, [label for _, label, _ in rows])
    right.invert_yaxis()
    right.set_xlim(0, 1.12)
    right.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    right.set_xlabel(r"$R^2$ on detected samples, unseen studies")
    panel_label(right, "b")
    fig.tight_layout()
    save(fig, "fig_surface_censored")


def intervals() -> None:
    table = pd.read_csv(ARTIFACTS / "audit_intervals.csv")
    table = table[~table.method.str.contains("Physics")]
    labels = [m.replace("Study-grouped", "Study").replace(", nominal 90%", "").replace(", nominal 95%", ", 95%") for m in table.method]
    y = np.arange(len(table))
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4), sharey=True)
    left.barh(y - 0.18, table.row_coverage * 100, height=0.34, color=BLUE, label="All samples")
    left.barh(y + 0.18, table.mean_study_coverage * 100, height=0.34, color=AQUA, label="Mean per study")
    left.axvline(90, color=INK, linestyle="--", linewidth=0.8)
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_xlim(0, 100)
    left.set_xlabel("Coverage on unseen studies (%)")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    panel_label(left, "a")
    right.barh(y, table.median_width_log10, color=NEUTRAL, height=0.6)
    right.set_xlabel(r"Median interval width ($\log_{10}$ units)")
    panel_label(right, "b")
    fig.tight_layout()
    save(fig, "fig_intervals")


def risk_coverage() -> None:
    curves = pd.read_csv(ARTIFACTS / "audit_risk_coverage.csv")
    styles = [
        ("Baseline LightGBM", "CQR interval width", "Baseline, CQR width", ORANGE, "-"),
        ("Baseline LightGBM", "Ensemble spread", "Baseline, ensemble spread", NEUTRAL, "--"),
        ("Baseline LightGBM", "Distance to same-family sample", "Baseline, same-family distance", NEUTRAL, ":"),
        ("Baseline LightGBM", "Flag count", "Baseline, flag count", VIOLET, "-."),
        ("Within-Between Physics", "CQR interval width", "Physics, CQR width", BLUE, "-"),
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4))
    for model, scorer, label, color, dash in styles:
        part = curves[(curves.model == model) & (curves.scorer == scorer)]
        left.plot(part.keep_fraction * 100, part.MAE, color=color, linestyle=dash, marker="o", markersize=3, label=label)
        right.plot(part.keep_fraction * 100, part.miss_gt_2_log * 100, color=color, linestyle=dash, marker="o", markersize=3)
    for ax, ylabel, letter in [(left, r"MAE ($\log_{10}$ units)", "a"), (right, r"Errors above $100{\times}$ (%)", "b")]:
        ax.invert_xaxis()
        ax.set_xlabel("Predictions retained (%)")
        ax.set_ylabel(ylabel)
        panel_label(ax, letter)
    fig.legend(*left.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    save(fig, "fig_risk_coverage")


if __name__ == "__main__":
    for make in (benchmark, dalvi_scale, surface_and_censored, intervals, risk_coverage):
        make()
