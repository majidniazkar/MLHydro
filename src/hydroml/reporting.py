"""Result tables and figures.

Writes one self-describing folder per run::

    results/<run_name>/
        config_used.yaml         exact configuration, including defaults
        data_summary.json        rows, features, split dates
        run_log.txt              console transcript
        metrics_long.csv         tidy: model, split, metric, value
        metrics_wide.csv         one row per model/split
        metrics.xlsx             sheets: train, val, test, best_params
        predictions.csv          date, split, observed, one column per model
        best_params.json
        tuning/<model>_trials.csv
        models/<model>.joblib | .keras
        figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

from .metrics import HIGHER_IS_BETTER     # noqa: E402

PLOT_RC = {
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def _apply_rc():
    plt.rcParams.update(PLOT_RC)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def metrics_tables(results: dict, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``results[model][split][metric]`` -> (long, wide) frames."""
    rows = []
    for model, splits in results.items():
        for split, vals in splits.items():
            for metric, value in vals.items():
                rows.append({"model": model, "split": split,
                             "metric": metric, "value": value})
    long = pd.DataFrame(rows)
    if long.empty:
        return long, long
    wide = long.pivot_table(index=["model", "split"], columns="metric",
                            values="value", sort=False).reset_index()
    wide.columns.name = None
    order = [m for m in cfg["reporting"]["metrics"] if m in wide.columns]
    return long, wide[["model", "split"] + order]


def rank_models(wide: pd.DataFrame, cfg) -> pd.DataFrame:
    """Leaderboard on ``reporting.rank_on`` split by ``reporting.rank_by``."""
    metric, split = cfg["reporting"]["rank_by"], cfg["reporting"]["rank_on"]
    sub = wide[wide["split"] == split]
    if sub.empty or metric not in sub.columns:
        return pd.DataFrame()
    return (sub.sort_values(metric, ascending=not HIGHER_IS_BETTER[metric])
               .reset_index(drop=True)
               .assign(rank=lambda d: np.arange(1, len(d) + 1)))


def write_tables(run_dir: Path, results: dict, predictions: pd.DataFrame,
                 best_params: dict, cfg) -> pd.DataFrame:
    long, wide = metrics_tables(results, cfg)
    fmts = set(cfg["reporting"]["formats"])

    if "csv" in fmts:
        long.to_csv(run_dir / "metrics_long.csv", index=False)
        wide.to_csv(run_dir / "metrics_wide.csv", index=False)
        if cfg["reporting"]["save_predictions"]:
            predictions.to_csv(run_dir / "predictions.csv", index=False)

    if "xlsx" in fmts:
        with pd.ExcelWriter(run_dir / "metrics.xlsx", engine="openpyxl") as xls:
            for split in ("train", "val", "test"):
                part = wide[wide["split"] == split].drop(columns="split")
                if not part.empty:
                    part.to_excel(xls, sheet_name=split, index=False)
            lb = rank_models(wide, cfg)
            if not lb.empty:
                lb.to_excel(xls, sheet_name="leaderboard", index=False)
            pd.DataFrame(
                [{"model": k, "params": json.dumps(v)} for k, v in best_params.items()]
            ).to_excel(xls, sheet_name="best_params", index=False)

    (run_dir / "best_params.json").write_text(
        json.dumps(best_params, indent=2, default=str), encoding="utf-8")
    return wide


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _thin(n: int, cap: int) -> slice:
    return slice(None) if n <= cap else slice(0, n, int(np.ceil(n / cap)))


def fig_hydrograph(predictions: pd.DataFrame, models: list[str], cfg,
                   split: str = "test"):
    _apply_rc()
    sub = predictions[predictions["split"] == split]
    if sub.empty:
        return None
    sel = _thin(len(sub), cfg["reporting"]["hydrograph_max_points"])
    sub = sub.iloc[sel]
    show = models[:4]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(sub["date"], sub["observed"], color="0.15", lw=1.4, label="Observed")
    for model, colour in zip(show, plt.cm.tab10.colors):
        if model in sub:
            ax.plot(sub["date"], sub[model], lw=1.0, alpha=0.9,
                    color=colour, label=model)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{cfg['data']['target_column']}")
    ax.set_title(f"Observed vs simulated - {split} period"
                 + (f" (every {sel.step}th point shown)" if sel.step else ""))
    ax.legend(ncol=min(5, len(show) + 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def fig_scatter(predictions: pd.DataFrame, models: list[str], cfg,
                split: str = "test"):
    _apply_rc()
    sub = predictions[predictions["split"] == split]
    models = [m for m in models if m in sub]
    if sub.empty or not models:
        return None
    ncol = min(4, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow),
                             squeeze=False)
    lim = float(np.nanmax([sub["observed"].max()] + [sub[m].max() for m in models]))
    for ax, model in zip(axes.ravel(), models):
        ax.scatter(sub["observed"], sub[model], s=6, alpha=0.35,
                   edgecolors="none", color="#1f77b4")
        ax.plot([0, lim], [0, lim], color="0.3", lw=1, ls="--")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Observed"); ax.set_ylabel("Simulated")
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    fig.suptitle(f"Observed vs simulated scatter - {split} period", y=1.0)
    fig.tight_layout()
    return fig


def fig_metric_bars(wide: pd.DataFrame, cfg):
    _apply_rc()
    metric, split = cfg["reporting"]["rank_by"], cfg["reporting"]["rank_on"]
    lb = rank_models(wide, cfg)
    if lb.empty:
        return None
    second = "rmse" if metric != "rmse" else "mae"
    has_second = second in lb.columns
    fig, axes = plt.subplots(1, 2 if has_second else 1,
                             figsize=(11 if has_second else 6, 0.32 * len(lb) + 2.2),
                             squeeze=False)
    ax = axes[0, 0]
    ax.barh(lb["model"], lb[metric], color="#2b7bba")
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric.upper()} ({split})")
    ax.set_title(f"Model skill by {metric.upper()}")
    if has_second:
        ax2 = axes[0, 1]
        ax2.barh(lb["model"], lb[second], color="#c96f2b")
        ax2.invert_yaxis()
        ax2.set_xlabel(f"{second.upper()} ({split})")
        ax2.set_title(f"Model error by {second.upper()}")
    fig.tight_layout()
    return fig


def fig_residuals(predictions: pd.DataFrame, models: list[str], cfg,
                  split: str = "test"):
    _apply_rc()
    sub = predictions[predictions["split"] == split]
    models = [m for m in models if m in sub]
    if sub.empty or not models:
        return None
    data = [(sub[m] - sub["observed"]).dropna().to_numpy() for m in models]
    fig, ax = plt.subplots(figsize=(max(6, 0.75 * len(models) + 2), 4.2))
    ax.boxplot(data, showfliers=False)
    # 'labels' was renamed 'tick_labels' in matplotlib 3.9; set them directly
    ax.set_xticks(np.arange(1, len(models) + 1))
    ax.set_xticklabels(models)
    ax.axhline(0, color="0.3", lw=1, ls="--")
    ax.set_ylabel(f"Residual (simulated - observed) [{cfg['data']['target_column']}]")
    ax.set_title(f"Residual distribution - {split} period")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def fig_lag_correlation(ctx):
    """Driver-to-target cross-correlation, plus the target's ACF/PACF."""
    _apply_rc()
    cc = ctx.get("cross_correlation")
    ac = ctx.get("autocorrelation")
    if cc is None and ac is None:
        return None
    ncol = 2 if (cc is not None and ac is not None) else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.6 * ncol, 4.2), squeeze=False)
    col = 0
    if cc is not None and not cc.empty:
        ax = axes[0, col]; col += 1
        for (name, part), colour in zip(cc.groupby("driver", sort=False),
                                        plt.cm.tab10.colors):
            ax.plot(part["lag"], part["corr"], marker="o", ms=3, lw=1.2,
                    color=colour, label=str(name))
        band = float(cc["conf_band"].iloc[0])
        ax.axhspan(-band, band, color="0.85", zorder=0)
        ax.axhline(0, color="0.4", lw=0.8)
        # mark the lag actually selected for each driver, when selection ran
        sel = ctx.get("lag_selection")
        if sel is not None and not sel.empty:
            pts = [(r.best_lag, r.best_corr) for r in sel.itertuples()
                   if r.selected_lags and pd.notna(r.best_lag)]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=90,
                           marker="*", color="0.15", zorder=5,
                           label="selected lag")
        ax.set_xlabel("Driver lag (timesteps)")
        ax.set_ylabel("Correlation with target")
        ax.set_title("Cross-correlation (training rows)")
        ax.legend(fontsize=8, ncol=max(1, len(cc['driver'].unique()) // 6))
    if ac is not None and not ac.empty:
        ax = axes[0, col]
        ax.bar(ac["lag"] - 0.2, ac["acf"], width=0.4, label="ACF", color="#2b7bba")
        ax.bar(ac["lag"] + 0.2, ac["pacf"], width=0.4, label="PACF", color="#c96f2b")
        band = float(ac["conf_band"].iloc[0])
        ax.axhspan(-band, band, color="0.85", zorder=0)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xlabel("Lag (timesteps)")
        ax.set_ylabel("Correlation")
        ax.set_title("Target memory")
        ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_seasonality(ctx):
    """Monthly distribution of the target, with the seasonality strength."""
    _apply_rc()
    tab = ctx.get("seasonal")
    if tab is None or tab.empty:
        return None
    monthly = tab[tab["group"].str.startswith("month_")].copy()
    if monthly.empty:
        return None
    monthly["m"] = monthly["group"].str.replace("month_", "").astype(int)
    monthly = monthly.sort_values("m")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.fill_between(monthly["m"], monthly["q10"], monthly["q90"], alpha=0.25,
                    color="#2b7bba", label="10-90th percentile")
    ax.plot(monthly["m"], monthly["median"], marker="o", color="#14456b",
            label="median")
    ax.plot(monthly["m"], monthly["mean"], marker="s", ls="--", color="#c96f2b",
            label="mean")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel(ctx["cfg"]["data"]["target_column"])
    strength = ctx.get("seasonality_strength")
    extra = "" if strength is None or not np.isfinite(strength) else \
        f" - month-of-year climatology explains {100 * strength:.1f}% of variance"
    ax.set_title(f"Seasonal cycle of the target{extra}", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_importance(ctx):
    """Permutation importance of each predictor."""
    _apply_rc()
    frames = ctx.get("importance") or {}
    frames = {k: v for k, v in frames.items() if v is not None and not v.empty}
    if not frames:
        return None
    ncol = min(2, len(frames))
    nrow = int(np.ceil(len(frames) / ncol))
    height = max(3.2, 0.26 * max(len(v) for v in frames.values()) + 1.6)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, height * nrow),
                             squeeze=False)
    for ax, (model, tab) in zip(axes.ravel(), frames.items()):
        tab = tab.sort_values("importance")
        ax.barh(tab["feature"], tab["importance"],
                xerr=tab["importance_std"], color="#2b7bba",
                error_kw={"ecolor": "0.5", "lw": 0.8})
        ax.axvline(0, color="0.4", lw=0.8)
        metric = tab["metric"].iloc[0]
        ax.set_xlabel(f"Drop in {metric.upper()} when shuffled")
        ax.set_title(f"{model} (baseline {metric.upper()}="
                     f"{tab['baseline'].iloc[0]:.3f})", fontsize=10)
    for ax in axes.ravel()[len(frames):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def fig_cv_metrics(ctx):
    """Fold-to-fold spread of the ranking metric under cross-validation."""
    _apply_rc()
    long = ctx.get("cv_long")
    if long is None or long.empty:
        return None
    metric = ctx["cfg"]["reporting"]["rank_by"]
    if metric not in long.columns:
        return None
    order = (long.groupby("model", sort=False)[metric].mean()
                 .sort_values(ascending=not HIGHER_IS_BETTER[metric]).index)
    data = [long.loc[long["model"] == m, metric].dropna().to_numpy() for m in order]
    keep = [(m, d) for m, d in zip(order, data) if d.size]
    if not keep:
        return None
    names = [m for m, _ in keep]
    data = [d for _, d in keep]
    fig, ax = plt.subplots(figsize=(max(6.5, 0.8 * len(names) + 2), 4.4))
    ax.boxplot(data, showfliers=False)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(d.size, i) + np.random.default_rng(0).normal(0, 0.04, d.size),
                   d, s=14, color="#c96f2b", zorder=3, alpha=0.8)
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names)
    ax.set_ylabel(metric.upper())
    n_folds = int(long["fold"].nunique())
    ax.set_title(f"Cross-validated {metric.upper()} across {n_folds} folds")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def _wrap_pred_figure(fn):
    def inner(ctx):
        return fn(ctx["predictions"], ctx["models"], ctx["cfg"], ctx["split"])
    return inner


FIGURES = {
    "hydrograph": _wrap_pred_figure(fig_hydrograph),
    "scatter": _wrap_pred_figure(fig_scatter),
    "residuals": _wrap_pred_figure(fig_residuals),
    "metric_bars": lambda ctx: fig_metric_bars(ctx["wide"], ctx["cfg"]),
    "lag_correlation": fig_lag_correlation,
    "seasonality": fig_seasonality,
    "importance": fig_importance,
    "cv_metrics": fig_cv_metrics,
}


def write_figures(run_dir: Path, wide: pd.DataFrame, predictions: pd.DataFrame,
                  cfg, extras: dict | None = None, log=print) -> list[Path]:
    """Render every figure named in ``reporting.figures`` that has data.

    ``extras`` carries the optional diagnostic and CV frames; a figure whose
    inputs are absent is skipped silently rather than failing the run.
    """
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lb = rank_models(wide, cfg)
    ordered = list(lb["model"]) if not lb.empty else [
        c for c in predictions.columns if c not in ("date", "split", "observed")]

    ctx = {"wide": wide, "predictions": predictions, "cfg": cfg,
           "models": ordered, "split": cfg["reporting"]["rank_on"]}
    ctx.update(extras or {})

    written = []
    for name in cfg["reporting"]["figures"]:
        if name not in FIGURES:
            raise KeyError(f"unknown figure {name!r}; available {sorted(FIGURES)}")
        try:
            fig = FIGURES[name](ctx)
        except Exception as exc:                          # noqa: BLE001
            log(f"    figure '{name}' failed: {type(exc).__name__}: {exc}")
            continue
        if fig is None:
            continue
        path = fig_dir / f"{name}.png"
        fig.savefig(path, dpi=cfg["reporting"]["dpi"])
        plt.close(fig)
        written.append(path)
        log(f"    figure -> {path.name}")
    return written
