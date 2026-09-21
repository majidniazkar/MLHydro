"""Run a trained model forward on future climate scenarios.

A training run leaves everything needed to reproduce its own preprocessing in
``results/<run>/``: the resolved configuration, the feature list, the chosen
lags, the selected hyper-parameters and the serialised estimators. This module
reloads that state, applies the **same** transformation chain to a scenario
table, and writes projected discharge per scenario.

What is reused, and why it matters
----------------------------------
* the **fitted scalers** -- refitting them on the scenario period would rescale
  a warmer, wetter future onto the training range and erase the climate signal
  the projection is meant to show;
* the **lag plan** -- ``prec1`` is shifted by the travel time chosen during
  training, not re-selected on scenario data (there is no target to select
  against);
* the **feature order** -- the matrix is reindexed to the training feature list
  and a missing column is an error, never a silent zero.

Autoregressive models
---------------------
If the trained feature set contains lags of the target (``streamflow_lag1``
...), those values do not exist in the future. The projection is then run
**recursively**: each step's prediction becomes the next step's lag input,
seeded from the last observed flows of the historical record. This is the
standard approach and it is what makes a multi-decade projection possible at
all, but errors compound and the result is a *scenario simulation*, not a
forecast. A model trained without ``target_lags`` is preferable for climate
projection; the pipeline says so in the log.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .data import add_calendar_features, build_dataset, load_table

__all__ = [
    "ProjectionInputs", "load_run", "read_scenarios", "detect_scenarios",
    "build_scenario_matrix", "project_scenario", "project", "summarise",
]

_LAG_RE = re.compile(r"^(?P<base>.+)_lag(?P<lag>\d+)$")


# --------------------------------------------------------------------------- #
# loading a finished run
# --------------------------------------------------------------------------- #
@dataclass
class ProjectionInputs:
    """Everything a projection needs from a completed training run."""
    run_dir: Path
    cfg: Config
    dataset: object                       # hydroml.data.Dataset
    best_params: dict = field(default_factory=dict)
    target_lags: list = field(default_factory=list)
    driver_lags: dict = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return list(self.dataset.features)

    @property
    def target(self) -> str:
        return self.dataset.target

    #: raw driver columns the scenario file must supply, i.e. the model's
    #: features minus everything the pipeline derives itself (target lags,
    #: calendar terms) and with lagged columns traced back to their source.
    required_drivers: list = field(default_factory=list)


def load_run(run_dir: str | Path, data_path: str | Path | None = None,
             log=print) -> ProjectionInputs:
    """Rebuild the training state from ``results/<run>/``.

    The historical dataset is rebuilt from ``config_used.yaml`` rather than
    stored, which keeps the scalers, the feature list and the split boundaries
    bit-identical to the training run as long as the input file has not
    changed. ``data_path`` overrides the recorded input location (useful when
    the run was produced on another machine).
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config_used.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found. Point --run at a results/<run_name> folder "
            "produced by run_pipeline.py."
        )
    overrides = {"data": {"path": str(data_path)}} if data_path else {}
    cfg = load_config(cfg_path, **overrides)
    log(f"[run] {run_dir}  target='{cfg['data']['target_column']}'")

    ds = build_dataset(cfg, log=lambda *_: None)
    log(f"[run] historical dataset rebuilt: {len(ds.X)} rows, "
        f"{ds.n_features} features, {ds.X.index.min().date()} .. "
        f"{ds.X.index.max().date()}")

    best = {}
    bp = run_dir / "best_params.json"
    if bp.exists():
        best = json.loads(bp.read_text(encoding="utf-8"))

    target_lags, driver_lags = [], {}
    for name in ds.features:
        m = _LAG_RE.match(name)
        if not m:
            continue
        base, lag = m.group("base"), int(m.group("lag"))
        if base == ds.target:
            target_lags.append(lag)
        else:
            driver_lags.setdefault(base, []).append(lag)
    raw_columns = set(load_table(cfg).columns)
    calendar_prefixes = ("sin_", "cos_", "m_")
    season_names = set(cfg["features"]["season_map"])
    required = set(driver_lags)
    for name in ds.features:
        if _LAG_RE.match(name) or name in season_names:
            continue
        if str(name).startswith(calendar_prefixes):
            continue
        if name in raw_columns:
            required.add(name)
    return ProjectionInputs(run_dir=run_dir, cfg=cfg, dataset=ds, best_params=best,
                            target_lags=sorted(target_lags), driver_lags=driver_lags,
                            required_drivers=sorted(required))


def load_estimator(inputs: ProjectionInputs, model: str, refit: bool = False,
                   log=print):
    """Return a fitted estimator: deserialised from the run, or refitted.

    scikit-learn and boosting models round-trip through joblib. PyTorch models
    are stored as a state dict plus their constructor arguments and are rebuilt
    here. Anything that cannot be loaded falls back to refitting on the
    historical training block with the hyper-parameters the run selected, which
    reproduces the same model when the input data is unchanged.
    """
    from .fitting import fit_estimator
    from .models import REGISTRY

    if model not in REGISTRY:
        raise KeyError(f"unknown model '{model}'; registry has {sorted(REGISTRY)}")
    spec = REGISTRY[model]
    params = inputs.best_params.get(model, {}) or {}
    mdir = inputs.run_dir / "models"

    if not refit:
        joblib_path = mdir / f"{model}.joblib"
        torch_path = mdir / f"{model}.pt"
        keras_path = mdir / f"{model}.keras"
        try:
            if joblib_path.exists():
                import joblib
                log(f"[model] {model}: loaded {joblib_path.name}")
                return joblib.load(joblib_path)
            if torch_path.exists():
                import torch
                from .fitting import build_estimator
                blob = torch.load(torch_path, weights_only=False)
                est = build_estimator(spec, {**params, **blob.get("hyperparameters", {})},
                                      inputs.cfg, inputs.dataset)
                est.model_ = est._build(inputs.dataset.n_features)
                est.model_.load_state_dict(blob["state_dict"])
                est.model_.eval()
                log(f"[model] {model}: loaded {torch_path.name}")
                return est
            if keras_path.exists():
                from .fitting import build_estimator
                import tensorflow as tf
                est = build_estimator(spec, params, inputs.cfg, inputs.dataset)
                est.model_ = tf.keras.models.load_model(keras_path)
                log(f"[model] {model}: loaded {keras_path.name}")
                return est
        except Exception as exc:                                  # noqa: BLE001
            log(f"[model] {model}: could not load saved file "
                f"({type(exc).__name__}: {exc}); refitting instead")

    ds = inputs.dataset
    tr = ds.splits["train"]
    if inputs.cfg["tuning"]["refit_on"] == "train_val":
        tr = np.concatenate([tr, ds.splits["val"]])
    log(f"[model] {model}: refitting on {len(tr)} historical rows "
        f"with params {params or '(defaults)'}")
    return fit_estimator(spec, params, inputs.cfg, ds, tr, ds.splits["val"])


# --------------------------------------------------------------------------- #
# reading the scenario file
# --------------------------------------------------------------------------- #
def _read_any(path: Path, sheet=0):
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, sheet_name=sheet)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported scenario file extension {suffix!r}")


def detect_scenarios(df: pd.DataFrame, drivers: list[str],
                     date_col: str) -> dict[str, dict[str, str]]:
    """Map ``scenario -> {driver: column}`` from the scenario table's columns.

    Three naming conventions are recognised, in this order:

    ``<driver>_<scenario>``   ``prec1_rcp45``, ``temp1_rcp45``  (the common one)
    ``<scenario>_<driver>``   ``rcp45_prec1``
    a ``scenario`` column     long format, one block of rows per scenario

    A scenario is only accepted when **every** driver the model needs is
    present for it, so a partially-populated scenario fails loudly instead of
    being modelled with missing inputs.
    """
    cols = [c for c in df.columns if c != date_col]
    found: dict[str, dict[str, str]] = {}

    for col in cols:
        name = str(col)
        for drv in drivers:
            for candidate in (f"{drv}_", f"{drv}"):
                if name.lower().startswith(candidate.lower()):
                    tag = name[len(candidate):].lstrip("_")
                    if tag:
                        found.setdefault(tag, {})[drv] = col
            if name.lower().endswith(f"_{drv}".lower()):
                tag = name[: -(len(drv) + 1)]
                if tag:
                    found.setdefault(tag, {})[drv] = col

    complete = {tag: m for tag, m in found.items()
                if set(m) == set(drivers)}
    return complete


def read_scenarios(path: str | Path, drivers: list[str], date_col: str,
                   dayfirst: bool = False, log=print) -> dict[str, pd.DataFrame]:
    """Return ``{scenario: frame}``, each frame indexed by date with `drivers` columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"scenario file not found: {path.resolve()}")

    if path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        sheets = pd.read_excel(path, sheet_name=None)
    else:
        sheets = {path.stem: _read_any(path)}

    def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
        sub = frame[[date_col] + drivers].copy()
        sub[date_col] = pd.to_datetime(sub[date_col], dayfirst=dayfirst)
        return sub.set_index(date_col).sort_index()

    def _has_all(frame) -> bool:
        return date_col in frame.columns and all(d in frame.columns for d in drivers)

    def _scenario_col(frame):
        return next((c for c in frame.columns if str(c).lower() == "scenario"), None)

    out: dict[str, pd.DataFrame] = {}

    # (1) several sheets, each a scenario with the training column names.
    #     Requires >1 matching sheet: a single sheet is ambiguous and is handled
    #     below, where a 'scenario' column would otherwise be ignored.
    matching = {name: f for name, f in sheets.items() if _has_all(f)}
    if len(matching) > 1 and all(_scenario_col(f) is None for f in matching.values()):
        out = {str(name): _tidy(f) for name, f in matching.items()}
        log(f"[scenarios] one sheet per scenario -> {sorted(out)}")
        return out

    frame = next(iter(matching.values()), next(iter(sheets.values())))
    if date_col not in frame.columns:
        raise KeyError(f"date column '{date_col}' not found in {path.name}. "
                       f"Columns: {list(frame.columns)}")

    # (2) long format: an explicit 'scenario' column
    scen_col = _scenario_col(frame)
    if scen_col is not None and _has_all(frame):
        for tag, part in frame.groupby(scen_col, sort=False):
            out[str(tag)] = _tidy(part)
        log(f"[scenarios] long format, '{scen_col}' column -> {sorted(out)}")
        return out

    # (3) wide format: the scenario tag lives in the column names
    mapping = detect_scenarios(frame, drivers, date_col)
    if mapping:
        dates = pd.to_datetime(frame[date_col], dayfirst=dayfirst)
        for tag, cols in mapping.items():
            sub = frame[[cols[d] for d in drivers]].copy()
            sub.columns = drivers
            sub.index = dates
            out[tag] = sub.sort_index()
        log(f"[scenarios] wide format, tag in column names -> {sorted(out)}")
        return out

    # (4) a single unnamed scenario: exactly the training columns, nothing else
    if _has_all(frame):
        tag = next((n for n, f in matching.items() if f is frame), "scenario")
        out[str(tag)] = _tidy(frame)
        log(f"[scenarios] single scenario '{tag}' (no scenario tag found)")
        return out

    raise ValueError(
        f"could not identify scenarios in {path.name}.\n"
        f"The trained model needs the driver(s) {drivers}. Expected columns named "
        f"'<driver>_<scenario>' (e.g. '{drivers[0]}_rcp45'), one sheet per "
        f"scenario, or a 'scenario' column.\n"
        f"Found columns: {list(frame.columns)}"
    )


# --------------------------------------------------------------------------- #
# feature construction for a future period
# --------------------------------------------------------------------------- #
def build_scenario_matrix(scenario: pd.DataFrame, inputs: ProjectionInputs,
                          warmup: pd.DataFrame | None = None,
                          log=print) -> pd.DataFrame:
    """Engineer the training feature columns for one scenario period.

    ``warmup`` is the tail of the historical driver record; it supplies the rows
    a lag reaches back into at the start of the projection. Without it the first
    ``max(lag)`` steps would be undefined and are dropped.

    Target lags are left as NaN here -- :func:`project_scenario` fills them
    recursively.
    """
    frame = scenario.copy()
    n_future = len(frame)
    if warmup is not None and len(warmup):
        shared = [c for c in frame.columns if c in warmup.columns]
        frame = pd.concat([warmup[shared], frame[shared]], axis=0).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]

    for base, lags in inputs.driver_lags.items():
        if base not in frame.columns:
            raise KeyError(
                f"the model needs lags of '{base}' but the scenario table has "
                f"no such driver (available: {list(scenario.columns)})"
            )
        for lag in lags:
            frame[f"{base}_lag{lag}"] = frame[base].shift(lag)

    frame, _ = add_calendar_features(frame, inputs.cfg)

    for lag in inputs.target_lags:
        frame[f"{inputs.target}_lag{lag}"] = np.nan

    missing = [f for f in inputs.features if f not in frame.columns]
    if missing:
        raise KeyError(
            f"cannot build the trained feature(s) {missing} from the scenario "
            f"table. Its columns are {list(scenario.columns)}; the model was "
            f"trained on {inputs.features}."
        )
    X = frame[inputs.features]
    X = X.iloc[-n_future:] if warmup is not None else X

    driver_cols = [f for f in inputs.features
                   if not f.startswith(f"{inputs.target}_lag")]
    incomplete = X[driver_cols].isna().any(axis=1)
    if incomplete.any():
        first_ok = (~incomplete).idxmax() if (~incomplete).any() else None
        log(f"    dropped {int(incomplete.sum())} leading row(s) with incomplete "
            f"lags; projection starts {first_ok.date() if first_ok is not None else 'never'}")
        X = X[~incomplete]
    return X


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #
def _affine(scaler, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-column ``(a, b)`` such that ``scaled = a * raw + b``.

    MinMaxScaler and StandardScaler are both per-column affine maps, so two
    probe rows recover their coefficients exactly. Plain arrays let the
    recursive loop rescale one value per step without calling scikit-learn.
    """
    if scaler is None:
        return np.ones(n_cols), np.zeros(n_cols)
    b = scaler.transform(np.zeros((1, n_cols)))[0]
    a = scaler.transform(np.ones((1, n_cols)))[0] - b
    return a, b


def project_scenario(est, X: pd.DataFrame, inputs: ProjectionInputs,
                     seed_history: np.ndarray | None = None) -> pd.Series:
    """Predict the target over ``X``; recursive when the model uses target lags."""
    ds = inputs.dataset
    clip = inputs.cfg["preprocess"]["clip_negative_predictions"]

    def _to_raw(scaled_pred):
        out = ds.inverse_y(np.asarray(scaled_pred).reshape(-1))
        return np.clip(out, 0.0, None) if clip else out

    if not inputs.target_lags:                      # direct, vectorised
        Xs = ds.x_scaler.transform(X.to_numpy(dtype=float)) if ds.x_scaler else \
            X.to_numpy(dtype=float)
        return pd.Series(_to_raw(est.predict(Xs)), index=X.index, name="projection")

    # recursive rollout ------------------------------------------------------
    max_lag = max(inputs.target_lags)
    if seed_history is None or len(seed_history) < max_lag:
        raise ValueError(
            f"recursive projection needs {max_lag} historical target values to "
            f"seed the lag buffer, got {0 if seed_history is None else len(seed_history)}"
        )
    history = deque(np.asarray(seed_history, dtype=float)[-max_lag:], maxlen=max_lag)
    lag_cols = {lag: X.columns.get_loc(f"{inputs.target}_lag{lag}")
                for lag in inputs.target_lags}

    raw = X.to_numpy(dtype=float)
    ax, bx = _affine(ds.x_scaler, raw.shape[1])
    ay, by = _affine(ds.y_scaler, 1)

    # Scale the whole matrix once. The target-lag columns are NaN here and are
    # overwritten each step with a[j]*prediction + b[j]; the affine identity is
    # verified against the fitted scaler below rather than assumed. Keeping the
    # scaler out of the loop is what makes a multi-decade daily rollout
    # practical: a per-step scikit-learn transform costs far more than the
    # arithmetic it performs.
    scaled = (ds.x_scaler.transform(np.nan_to_num(raw))
              if ds.x_scaler is not None else raw.copy())
    if ds.x_scaler is not None:
        probe = np.nan_to_num(raw[:1]).copy()
        probe[0, list(lag_cols.values())] = float(np.mean(history))
        if not np.allclose(ds.x_scaler.transform(probe)[0], ax * probe[0] + bx,
                           rtol=1e-9, atol=1e-12):
            raise RuntimeError(
                "the fitted feature scaler is not per-column affine, so the "
                "recursive path cannot rescale its own predictions safely. Use "
                "preprocess.scale_features: minmax | standard | none."
            )

    preds = np.empty(len(raw))
    buf = np.empty((1, raw.shape[1]))
    for t in range(len(raw)):
        buf[0] = scaled[t]
        for lag, j in lag_cols.items():
            buf[0, j] = ax[j] * history[-lag] + bx[j]   # history[-1] is t-1
        p = float(np.ravel(est.predict(buf))[0])
        value = (p - by[0]) / ay[0] if ds.y_scaler is not None else p
        preds[t] = max(value, 0.0) if clip else value
        history.append(preds[t])
    return pd.Series(preds, index=X.index, name="projection")


def project(inputs: ProjectionInputs, scenarios: dict[str, pd.DataFrame],
            models: list[str], refit: bool = False, log=print) -> pd.DataFrame:
    """Project every ``(scenario, model)`` pair; return a tidy long frame."""
    ds = inputs.dataset
    warmup_need = max([max(v) for v in inputs.driver_lags.values()] or [0])
    raw_hist = load_table(inputs.cfg)
    warmup = raw_hist.tail(max(warmup_need, 1)) if warmup_need else None
    seed = ds.y.to_numpy()[-max(inputs.target_lags or [1]):]

    if inputs.target_lags:
        log(f"[mode] recursive: the model uses {inputs.target}_lag"
            f"{inputs.target_lags}, which do not exist in the future. Each step "
            f"is fed its own previous prediction, seeded from the last "
            f"{len(seed)} observed value(s). Errors compound - see README.")
    else:
        log("[mode] direct: the model uses climate predictors only, so every "
            "step is independent of the projected ones.")

    estimators = {m: load_estimator(inputs, m, refit=refit, log=log) for m in models}

    rows = []
    for tag, frame in scenarios.items():
        log(f"[scenario] {tag}: {len(frame)} rows "
            f"{frame.index.min().date()} .. {frame.index.max().date()}")
        X = build_scenario_matrix(frame, inputs, warmup=warmup, log=log)
        for model, est in estimators.items():
            series = project_scenario(est, X, inputs, seed_history=seed)
            rows.append(pd.DataFrame({"scenario": tag, "model": model,
                                      "date": series.index,
                                      "projection": series.to_numpy()}))
            log(f"    {model:<22} mean={series.mean():.3f}  "
                f"min={series.min():.3f}  max={series.max():.3f}")
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def summarise(long: pd.DataFrame, inputs: ProjectionInputs,
              baseline: pd.Series | None = None) -> pd.DataFrame:
    """Per scenario/model statistics, and the change against the baseline.

    ``baseline`` defaults to the observed target over the historical record, so
    ``change_%`` answers "how does mean discharge under this scenario compare
    with the observed period the model was trained on?".
    """
    base = inputs.dataset.y if baseline is None else baseline
    base_mean = float(np.mean(base))
    rows = []
    for (tag, model), part in long.groupby(["scenario", "model"], sort=False):
        v = part["projection"].to_numpy()
        dates = pd.DatetimeIndex(part["date"])
        annual = pd.Series(v, index=dates).resample("YE").mean()
        rows.append({
            "scenario": tag, "model": model, "n": len(v),
            "start": str(dates.min().date()), "end": str(dates.max().date()),
            "mean": float(np.mean(v)), "median": float(np.median(v)),
            "q05": float(np.percentile(v, 5)), "q95": float(np.percentile(v, 95)),
            "min": float(np.min(v)), "max": float(np.max(v)),
            "baseline_mean": base_mean,
            "change_%": float(100.0 * (np.mean(v) - base_mean) / base_mean)
            if base_mean else np.nan,
            "annual_trend_per_decade": float(
                np.polyfit(np.arange(len(annual)), annual.to_numpy(), 1)[0] * 10)
            if len(annual) > 2 else np.nan,
        })
    return pd.DataFrame(rows)
