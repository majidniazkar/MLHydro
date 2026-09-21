#!/usr/bin/env python
"""Run a trained hydroml model forward on future climate scenarios.

    python run_projection.py --run results/subbasins \
                             --future data/future_scenarios.xlsx

    python run_projection.py --run results/subbasins \
                             --future data/future_scenarios.xlsx \
                             --models lightgbm xgboost --refit

The scenario file supplies the same drivers the model was trained on, for one
or more scenarios. Three layouts are recognised automatically:

    <driver>_<scenario>   date | prec1_rcp45 | temp1_rcp45 | prec1_ssp370 | ...
    one sheet per scenario, each with the training column names
    long format with a 'scenario' column

Outputs go to ``projections/<run_name>/``:

    projections.xlsx        one sheet per scenario + summary + long format
    projections_long.csv
    summary.csv             mean / median / percentiles / change vs baseline
    figures/                time series, annual means, change bars
    projection_log.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import pandas as pd                                            # noqa: E402

from hydroml.projection import (load_run, project, read_scenarios,  # noqa: E402
                                summarise)
from hydroml.reporting import PLOT_RC                          # noqa: E402


class _Logger:
    def __init__(self, path: Path):
        self.path = path
        path.write_text("", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def fig_timeseries(long: pd.DataFrame, target: str, roll: int = 365):
    """Projected series per scenario, smoothed; one panel per model."""
    plt.rcParams.update(PLOT_RC)
    models = list(dict.fromkeys(long["model"]))
    fig, axes = plt.subplots(len(models), 1, sharex=True, squeeze=False,
                             figsize=(11, 3.2 * len(models)))
    for ax, model in zip(axes[:, 0], models):
        sub = long[long["model"] == model]
        for tag, part in sub.groupby("scenario", sort=False):
            s = pd.Series(part["projection"].to_numpy(),
                          index=pd.DatetimeIndex(part["date"])).sort_index()
            ax.plot(s.index, s.rolling(roll, min_periods=max(1, roll // 3)).mean(),
                    lw=1.6, label=str(tag))
        ax.set_ylabel(target)
        ax.set_title(f"{model} - {roll}-step rolling mean")
        ax.legend(fontsize=9, ncol=3)
    axes[-1, 0].set_xlabel("Date")
    fig.tight_layout()
    return fig


def fig_annual(long: pd.DataFrame, target: str, baseline: float | None = None):
    """Annual means per scenario, averaged over models."""
    plt.rcParams.update(PLOT_RC)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for tag, part in long.groupby("scenario", sort=False):
        s = (pd.Series(part["projection"].to_numpy(),
                       index=pd.DatetimeIndex(part["date"]))
             .groupby(pd.DatetimeIndex(part["date"]).year).mean())
        ax.plot(s.index, s.to_numpy(), marker="o", ms=3, lw=1.4, label=str(tag))
    if baseline is not None:
        ax.axhline(baseline, color="0.3", ls="--", lw=1.2,
                   label="observed baseline mean")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Annual mean {target}")
    ax.set_title("Projected annual mean, averaged over models")
    ax.legend(fontsize=9, ncol=3)
    fig.tight_layout()
    return fig


def fig_change(summary: pd.DataFrame, target: str):
    """Change in mean target against the observed baseline."""
    plt.rcParams.update(PLOT_RC)
    piv = summary.pivot_table(index="scenario", columns="model", values="change_%")
    fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * piv.size ** 0.5 + 4), 4.4))
    piv.plot(kind="bar", ax=ax, width=0.8, edgecolor="none")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_ylabel(f"Change in mean {target} vs baseline (%)")
    ax.set_xlabel("")
    ax.set_title("Projected change against the observed period")
    ax.legend(fontsize=9, title=None)
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="training run folder, e.g. results/subbasins")
    ap.add_argument("--future", required=True,
                    help="scenario file (.xlsx / .csv / .parquet)")
    ap.add_argument("--models", nargs="+",
                    help="models to project with (default: every model saved "
                         "by the run, or the best one if none were saved)")
    ap.add_argument("--historical",
                    help="override the historical input path recorded in the run")
    ap.add_argument("--out", default=None,
                    help="output folder (default: projections/<run_name>)")
    ap.add_argument("--refit", action="store_true",
                    help="refit on historical data with the run's chosen "
                         "hyper-parameters instead of loading the saved model")
    ap.add_argument("--roll", type=int, default=365,
                    help="smoothing window for the time-series figure")
    ap.add_argument("--dayfirst", action="store_true",
                    help="scenario dates are dd/mm/yyyy")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else Path("projections") / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    log = _Logger(out_dir / "projection_log.txt")

    log("=" * 78)
    log(f"hydroml projection  ->  {out_dir}")
    log("=" * 78)

    inputs = load_run(run_dir, data_path=args.historical, log=log)

    models = args.models
    if not models:
        saved = sorted({p.stem for p in (run_dir / "models").glob("*")
                        if p.suffix in (".joblib", ".pt", ".keras")})
        if not saved:
            lb = run_dir / "metrics_wide.csv"
            if lb.exists():
                wide = pd.read_csv(lb)
                rank_on = inputs.cfg["reporting"]["rank_on"]
                rank_by = inputs.cfg["reporting"]["rank_by"]
                sub = wide[wide["split"] == rank_on].sort_values(
                    rank_by, ascending=False)
                saved = [sub["model"].iloc[0]] if len(sub) else []
            if not saved:
                log("no saved models and no metrics table; use --models")
                return 2
            log(f"[model] no serialised models in {run_dir / 'models'}; "
                f"using the run's best model '{saved[0]}' with --refit")
            args.refit = True
        models = saved
    log(f"[model] projecting with: {', '.join(models)}")

    drivers = inputs.required_drivers
    log(f"[model] scenario drivers required: {drivers}")

    scenarios = read_scenarios(args.future, drivers,
                               inputs.cfg["data"]["date_column"],
                               dayfirst=args.dayfirst, log=log)

    long = project(inputs, scenarios, models, refit=args.refit, log=log)
    summary = summarise(long, inputs)

    long.to_csv(out_dir / "projections_long.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with pd.ExcelWriter(out_dir / "projections.xlsx", engine="openpyxl") as xls:
        for tag, part in long.groupby("scenario", sort=False):
            wide = part.pivot_table(index="date", columns="model",
                                    values="projection").reset_index()
            if len(models) > 1:
                wide["ensemble_mean"] = wide[models].mean(axis=1)
            wide.to_excel(xls, sheet_name=str(tag)[:31], index=False)
        summary.to_excel(xls, sheet_name="summary", index=False)
    log(f"\n[write] projections.xlsx, projections_long.csv, summary.csv")

    target = inputs.target
    baseline = float(inputs.dataset.y.mean())
    for name, fig in (("projection_timeseries", fig_timeseries(long, target, args.roll)),
                      ("projection_annual", fig_annual(long, target, baseline)),
                      ("projection_change", fig_change(summary, target))):
        path = out_dir / "figures" / f"{name}.png"
        fig.savefig(path, dpi=inputs.cfg["reporting"]["dpi"])
        plt.close(fig)
        log(f"    figure -> {path.name}")

    log("")
    cols = ["scenario", "model", "mean", "q05", "q95", "change_%",
            "annual_trend_per_decade"]
    log(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log(f"\nbaseline (observed mean {target}) = {baseline:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
