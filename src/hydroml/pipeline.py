"""Run orchestration: config -> data -> models -> results.

Typical use::

    from hydroml import run
    result = run("config.yaml")
    print(result.leaderboard)

or from the command line::

    python run_pipeline.py --config config.yaml
"""

from __future__ import annotations

import json
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .data import Dataset, build_dataset
from .fitting import fit_estimator, predict_raw
from .metrics import HIGHER_IS_BETTER, evaluate
from .models import REGISTRY, ModelSpec, resolve_models
from .reporting import rank_models, write_figures, write_tables
from .tuning import search

SPLITS = ("train", "val", "test")


class _Logger:
    """Print to the console and append to ``run_log.txt``."""

    def __init__(self, path: Path, verbose: bool = True):
        self.path = path
        self.verbose = verbose
        self.path.write_text("", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        if self.verbose:
            print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


@dataclass
class RunResult:
    config: Config
    dataset: Dataset
    metrics_wide: pd.DataFrame
    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    best_params: dict
    estimators: dict = field(default_factory=dict, repr=False)
    failures: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)
    run_dir: Path = None
    cv_folds: pd.DataFrame = field(default_factory=pd.DataFrame)
    cv_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict = field(default_factory=dict, repr=False)


def _save_estimator(est, spec: ModelSpec, out_dir: Path, log) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    inner = getattr(est, "model_", None)
    try:
        if inner is not None and hasattr(inner, "save"):          # keras
            inner.save(out_dir / f"{spec.key}.keras")
        elif inner is not None and hasattr(inner, "state_dict"):  # torch
            import torch
            torch.save({"state_dict": inner.state_dict(),
                        "hyperparameters": est.get_params()},
                       out_dir / f"{spec.key}.pt")
        else:
            import joblib
            joblib.dump(est, out_dir / f"{spec.key}.joblib")
    except Exception as exc:                                  # noqa: BLE001
        log(f"    warning: could not serialise {spec.key} ({type(exc).__name__}: {exc})")


def _run_diagnostics(cfg, ds: Dataset, run_dir: Path, log) -> dict:
    """Lag structure, seasonality and their tables/figures inputs.

    Computed on the training rows only, and written to ``diagnostics/`` so the
    feature set can be justified independently of any model.
    """
    from . import diagnostics as dg

    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_lag = int(cfg["diagnostics"]["max_lag"])
    tr = ds.splits["train"]
    y_train = ds.y.iloc[tr]
    extras: dict = {}

    ac = dg.autocorrelation_table(y_train, max_lag)
    ac.to_csv(out_dir / "target_autocorrelation.csv", index=False)
    extras["autocorrelation"] = ac
    sig = ac[(ac["lag"] >= 1) & ac["pacf_significant"]]["lag"].tolist()
    log(f"[diag] target PACF significant at lags {sig[:10]}"
        + (" ..." if len(sig) > 10 else ""))

    drivers = [c for c in ds.features
               if ds.X[c].nunique() > 2 and not c.startswith(("sin_", "cos_", "m_"))]
    if drivers:
        cc = dg.cross_correlation_table(
            ds.X.iloc[tr].assign(**{ds.target: y_train}), ds.target, drivers, max_lag)
        cc.to_csv(out_dir / "driver_cross_correlation.csv", index=False)
        extras["cross_correlation"] = cc
        peak = (cc.dropna(subset=["abs_corr"]).sort_values("abs_corr", ascending=False)
                  .groupby("driver", sort=False).head(1))
        log("[diag] strongest lag per predictor (already-lagged columns shift further): "
            + ", ".join(f"{r.driver}@+{int(r.lag)}={r.corr:.2f}"
                        for r in peak.head(8).itertuples()))

    seas = dg.seasonal_table(y_train, cfg["features"]["season_map"])
    seas.to_csv(out_dir / "seasonality.csv", index=False)
    extras["seasonal"] = seas
    strength = dg.seasonality_strength(y_train)
    extras["seasonality_strength"] = strength
    log(f"[diag] month-of-year climatology explains {100 * strength:.1f}% of the "
        "training-period target variance")
    return extras


def run(config_path: str | Path | None = "config.yaml", **overrides) -> RunResult:
    cfg = load_config(config_path, **overrides)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tuning").mkdir(exist_ok=True)

    log = _Logger(run_dir / "run_log.txt", cfg["run"]["verbose"])
    t_start = time.time()
    np.random.seed(cfg["run"]["seed"])

    log("=" * 78)
    log(f"hydroml run '{cfg['run']['name']}'  ->  {run_dir}")
    log("=" * 78)
    cfg.dump(run_dir / "config_used.yaml")

    # ------------------------------------------------------ deep backend ---
    # Imported first, on purpose: see hydroml.deep.preload_backend.
    if any(REGISTRY[k].kind == "keras"
           for k in cfg["models"]["enabled"] if k in REGISTRY):
        from .deep import available_backends, preload_backend
        try:
            log(f"[deep] backend '{preload_backend(cfg['deep']['backend'])}' "
                f"(installed: {', '.join(available_backends()) or 'none'})")
        except Exception as exc:                                  # noqa: BLE001
            log(f"[deep] no usable backend ({type(exc).__name__}: {exc}); "
                "the ann/lstm models will be skipped")

    # ---------------------------------------------------------------- data --
    ds = build_dataset(cfg, log=log)
    summary = ds.summary()
    (run_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[data] {summary['rows']} rows, {summary['n_features']} features, "
        f"{summary['period'][0]} to {summary['period'][1]}")
    for part in SPLITS:
        s = summary["splits"][part]
        log(f"       {part:<5} n={s['n']:<6} {s['start']} .. {s['end']}")
    log(f"[data] features: {', '.join(summary['features'])}")
    if len(ds.splits["val"]) == 0 and cfg["tuning"]["selection_set"] == "validation":
        log("[data] WARNING: validation split is empty; tuning will fail. "
            "Increase split.val_fraction.")

    # ------------------------------------------- prepared input data export --
    from .export import write_prepared_input
    write_prepared_input(run_dir, ds, cfg, log=log)

    report = (ds.lag_plan or {}).get("report")
    if report is not None and not report.empty:
        (run_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
        report.to_csv(run_dir / "diagnostics" / "lag_selection.csv", index=False)

    # -------------------------------------------------------- diagnostics --
    extras: dict = {}
    if report is not None and not report.empty:
        extras["lag_selection"] = report
    if cfg["diagnostics"]["enabled"]:
        extras.update(_run_diagnostics(cfg, ds, run_dir, log))

    if cfg["tuning"]["selection_set"] == "test" and cfg["tuning"]["enabled"]:
        msg = ("tuning.selection_set='test': hyper-parameters are being chosen on "
               "the test block, so reported test skill is optimistically biased. "
               "Use 'validation' or 'cv' for defensible numbers.")
        log(f"[tuning] WARNING: {msg}")
        warnings.warn(msg, stacklevel=2)

    # -------------------------------------------------------------- models --
    specs, skipped = resolve_models(cfg)
    for key, reason in skipped:
        log(f"[models] skipped '{key}' -- {reason}")
    log(f"[models] running {len(specs)}: {', '.join(s.key for s in specs)}")

    results: dict[str, dict] = {}
    best_params: dict[str, dict] = {}
    estimators: dict[str, object] = {}
    failures: dict[str, str] = {}
    pred_cols: dict[str, np.ndarray] = {}
    metric_names = cfg["reporting"]["metrics"]
    refit_on = cfg["tuning"]["refit_on"]

    for spec in specs:
        log(f"\n[{spec.key}] {spec.label}")
        t0 = time.time()
        try:
            params, trials = ({}, pd.DataFrame())
            if cfg["tuning"]["enabled"] and cfg["tuning"]["strategy"] != "none":
                params, trials = search(spec, cfg, ds, log=log)
                if cfg["tuning"]["save_trials"] and not trials.empty:
                    trials.to_csv(run_dir / "tuning" / f"{spec.key}_trials.csv",
                                  index=False)
            best_params[spec.key] = params

            fit_pos = (np.concatenate([ds.splits["train"], ds.splits["val"]])
                       if refit_on == "train_val" else ds.splits["train"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = fit_estimator(spec, params, cfg, ds, fit_pos, ds.splits["val"])

            full = np.full(len(ds.X), np.nan)
            results[spec.key] = {}
            for part in SPLITS:
                pos = ds.splits[part]
                if len(pos) == 0:
                    continue
                sim = predict_raw(est, cfg, ds, pos)
                full[pos] = sim
                results[spec.key][part] = evaluate(ds.y.to_numpy()[pos], sim, metric_names)
            pred_cols[spec.key] = full
            estimators[spec.key] = est

            for part, vals in results[spec.key].items():
                log("    " + f"{part:<5} " + "  ".join(
                    f"{m}={vals[m]:.4f}" for m in metric_names))
            if cfg["reporting"]["save_models"]:
                _save_estimator(est, spec, run_dir / "models", log)
            log(f"    done in {time.time() - t0:.1f}s")

        except Exception as exc:                                  # noqa: BLE001
            failures[spec.key] = f"{type(exc).__name__}: {exc}"
            log(f"    FAILED: {failures[spec.key]}")
            log("    " + traceback.format_exc(limit=3).replace("\n", "\n    "))

    if not results:
        raise RuntimeError(f"no model produced results; failures: {failures}")

    # ------------------------------------------------------------- outputs --
    split_label = pd.Series(index=ds.X.index, dtype=object)
    for part in SPLITS:
        split_label.iloc[ds.splits[part]] = part
    predictions = pd.DataFrame(
        {"date": ds.X.index, "split": split_label.to_numpy(),
         "observed": ds.y.to_numpy(), **pred_cols})

    log("")
    wide = write_tables(run_dir, results, predictions, best_params, cfg)
    leaderboard = rank_models(wide, cfg)

    # ------------------------------------------------- permutation importance --
    if (cfg["diagnostics"]["enabled"]
            and cfg["diagnostics"]["permutation_importance"]
            and cfg["diagnostics"]["importance_model"] != "none"):
        from .diagnostics import permutation_importance
        which = (list(estimators) if cfg["diagnostics"]["importance_model"] == "all"
                 else ([leaderboard["model"].iloc[0]] if not leaderboard.empty else []))
        part = cfg["diagnostics"]["importance_on"]
        if len(ds.splits[part]) == 0:
            fallback = next((p for p in ("test", "val", "train")
                             if len(ds.splits[p])), None)
            log(f"[diag] diagnostics.importance_on='{part}' is empty; "
                f"using '{fallback}' instead")
            part = fallback or part
        imp = {}
        for key in which:
            if key not in estimators or len(ds.splits[part]) == 0:
                continue
            try:
                imp[key] = permutation_importance(
                    estimators[key], cfg, ds, part=part,
                    metric=cfg["reporting"]["rank_by"],
                    n_repeats=cfg["diagnostics"]["importance_repeats"],
                    seed=cfg["run"]["seed"])
                imp[key].to_csv(run_dir / "diagnostics" /
                                f"importance_{key}.csv", index=False)
            except Exception as exc:                          # noqa: BLE001
                log(f"[diag] permutation importance for {key} failed: "
                    f"{type(exc).__name__}: {exc}")
        if imp:
            extras["importance"] = imp
            top = next(iter(imp.values())).head(5)
            log(f"[diag] top predictors ({part} split, "
                f"{cfg['reporting']['rank_by']} drop): "
                + ", ".join(f"{r.feature}={r.importance:.3f}"
                            for r in top.itertuples()))

    # ------------------------------------------------------ cross-validation --
    cv_long = cv_summary = pd.DataFrame()
    if cfg["evaluation"]["cv"]["enabled"]:
        log("")
        from .evaluation import cross_validate
        ran = [s for s in specs if s.key in results]
        cv_long, cv_summary = cross_validate(ran, cfg, ds, best_params, log=log)
        if not cv_long.empty:
            cv_long.to_csv(run_dir / "metrics_cv_folds.csv", index=False)
            cv_summary.to_csv(run_dir / "metrics_cv_summary.csv", index=False)
            extras["cv_long"] = cv_long
            if "xlsx" in cfg["reporting"]["formats"]:
                with pd.ExcelWriter(run_dir / "metrics_cv.xlsx",
                                    engine="openpyxl") as xls:
                    cv_summary.to_excel(xls, sheet_name="summary", index=False)
                    cv_long.to_excel(xls, sheet_name="folds", index=False)
            rb = cfg["reporting"]["rank_by"]
            cols = ["model", "folds"] + [c for c in (f"{rb}_mean", f"{rb}_std",
                                                     "rmse_mean", "rmse_std")
                                         if c in cv_summary.columns]
            log(f"[cv] summary ranked by mean {rb}")
            log(cv_summary.sort_values(
                f"{rb}_mean", ascending=not HIGHER_IS_BETTER[rb])[cols]
                .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    write_figures(run_dir, wide, predictions, cfg, extras=extras, log=log)

    log("")
    log(f"[leaderboard] {cfg['reporting']['rank_on']} split, "
        f"ranked by {cfg['reporting']['rank_by']}")
    if not leaderboard.empty:
        cols = ["rank", "model"] + [m for m in metric_names if m in leaderboard]
        log(leaderboard[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if failures:
        log(f"[failures] {json.dumps(failures, indent=2)}")
    log(f"\nTotal wall time {time.time() - t_start:.1f}s. Outputs in {run_dir}")

    return RunResult(
        config=cfg, dataset=ds, metrics_wide=wide, leaderboard=leaderboard,
        predictions=predictions, best_params=best_params, estimators=estimators,
        failures=failures, skipped=skipped, run_dir=run_dir,
        cv_folds=cv_long, cv_summary=cv_summary, diagnostics=extras,
    )
