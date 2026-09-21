"""Configuration loading, defaulting and validation.

The user edits ``config.yaml`` only. Any key omitted there falls back to
``DEFAULTS`` below, so a minimal config is valid. Unknown top-level sections or
unknown keys inside a known section raise an error rather than being silently
ignored -- a silent typo in a config file is the single most common source of
"the template ran but did nothing".
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULTS: dict[str, Any] = {
    "run": {
        "name": "run",
        "output_dir": "results",
        "seed": 14,
        "n_jobs": -1,
        "verbose": True,
    },
    "data": {
        "path": "data/input_data.xlsx",
        "sheet": 0,
        "date_column": "date",
        "target_column": "flow",
        "feature_selection": "pattern",     # pattern | explicit | all
        "include_patterns": ["lag"],
        "feature_columns": [],
        "exclude_columns": [],
        "dayfirst": False,
        "resample": None,                   # e.g. "D", "MS"
        "resample_agg": "mean",
        "dropna": True,
        "target_min": None,                 # drop rows with target below this
    },
    "features": {
        "add_season_onehot": True,
        "season_map": {
            "Winter": [12, 1, 2],
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Autumn": [9, 10, 11],
        },
        "drop_first_season": False,
        "add_month_onehot": False,
        "doy_harmonics": 0,                 # k -> k sin + k cos terms
        "auto_lags": {
            "enabled": False,
            "select": "manual",             # manual | cross_correlation
            "columns": [],                  # [] -> every non-target column
            "lags": [1, 2, 3],              # list, or {driver: [lags]}
            "max_lag": 30,                  # search range when select=cross_correlation
            "top_k": 3,                     # lags kept per driver (1 = best only)
            "min_abs_corr": 0.1,
            # true: a driver replaced by a shifted copy (prec1 -> prec1_lag2) is
            # not also offered unlagged. A driver whose best lag is 0 keeps its
            # original column under its original name.
            "drop_unlagged": True,
        },
        "target_lags": [],                  # list of lags, or "auto"
        "target_lag_max": 10,
        "target_lag_top_k": 3,
    },
    "split": {
        "mode": "fraction",                 # index | fraction | date
        "train_end": None,                  # index mode
        "val_end": None,
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "train_end_date": None,             # date mode
        "val_end_date": None,
    },
    "preprocess": {
        "scale_features": "minmax",         # minmax | standard | none
        "scale_target": "minmax",
        "clip_negative_predictions": True,
    },
    "tuning": {
        "enabled": True,
        "strategy": "random",               # grid | random | none
        "n_iter": 30,
        "selection_metric": "rmse",
        "selection_set": "validation",      # validation | cv | test
        "cv_splits": 3,
        "save_trials": True,
        "max_grid_size": 20000,
        "refit_on": "train",                # train | train_val
        "refine_rounds": 1,                 # coordinate sweeps after the search
    },
    "evaluation": {
        "cv": {
            "enabled": False,
            "scheme": "expanding",          # expanding | blocked
            "n_splits": 5,
            "test_size": None,              # rows per fold test (expanding only)
            "gap": 0,                       # rows dropped around the test block
            # NB: named "over", not "on" -- YAML 1.1 parses a bare
            # `on:` key as the boolean True.
            "over": "all",                  # all | train_val
            "use_best_params": True,        # False -> nested search per fold
        },
    },
    "diagnostics": {
        "enabled": True,
        "max_lag": 30,
        "permutation_importance": True,
        "importance_model": "best",         # best | all | none
        "importance_on": "val",
        "importance_repeats": 5,
    },
    "models": {
        "enabled": [
            "linear_regression", "bayesian_ridge", "huber", "sgd",
            "knn", "svr", "decision_tree", "random_forest",
            "adaboost", "gradient_boosting", "hist_gradient_boosting",
            "xgboost", "lightgbm", "catboost", "ann", "lstm",
        ],
        "params": {},                       # fixed overrides, per model name
        "search_spaces": {},                # grid overrides, per model name
        "skip_unavailable": True,
    },
    "deep": {
        "backend": "auto",                  # auto | torch | tensorflow
        "ann": {
            "hidden_units": [18, 36, 36],
            "activation": "relu",
            "dropout": 0.0,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "mae",
            "epochs": 1000,
            "batch_size": 32,
            "patience": 50,
        },
        "lstm": {
            "sequence_mode": "sliding_window",   # or features_as_timesteps
            "window": 30,
            "units": [36, 36],
            "activation": "tanh",
            "dropout": 0.0,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "mae",
            "epochs": 300,
            "batch_size": 32,
            "patience": 30,
        },
    },
    "export": {
        "enabled": True,
        "filename": "prepared_input_data.xlsx",
        "formats": ["xlsx"],                # xlsx | csv
        "include_full_sheet": True,         # all rows with a 'split' label
    },
    "reporting": {
        "metrics": ["rmse", "mae", "nse", "kge", "pbias", "r2", "mape"],
        "figures": ["hydrograph", "scatter", "metric_bars", "residuals",
                    "lag_correlation", "seasonality", "importance", "cv_metrics"],
        "hydrograph_max_points": 2000,
        "rank_by": "nse",
        "rank_on": "test",
        "save_models": True,
        "save_predictions": True,
        "formats": ["csv", "xlsx"],
        "dpi": 200,
    },
}

_CHOICES = {
    ("data", "feature_selection"): {"pattern", "explicit", "all"},
    ("split", "mode"): {"index", "fraction", "date"},
    ("preprocess", "scale_features"): {"minmax", "standard", "none"},
    ("preprocess", "scale_target"): {"minmax", "standard", "none"},
    ("tuning", "strategy"): {"grid", "random", "none"},
    ("tuning", "selection_set"): {"validation", "cv", "test"},
    ("tuning", "refit_on"): {"train", "train_val"},
    ("reporting", "rank_on"): {"train", "val", "test"},
    ("deep", "backend"): {"auto", "torch", "tensorflow"},
    ("diagnostics", "importance_model"): {"best", "all", "none"},
    ("diagnostics", "importance_on"): {"train", "val", "test"},
}

#: Nested mappings whose keys are also validated against the defaults.
_NESTED = [
    ("features", "auto_lags"),
    ("evaluation", "cv"),
    ("deep", "ann"),
    ("deep", "lstm"),
]

_NESTED_CHOICES = {
    ("features", "auto_lags", "select"): {"manual", "cross_correlation"},
    ("evaluation", "cv", "scheme"): {"expanding", "blocked"},
    ("evaluation", "cv", "over"): {"all", "train_val"},
    ("deep", "lstm", "sequence_mode"): {"sliding_window", "features_as_timesteps"},
}


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


class Config(dict):
    """A validated configuration. Access sections as attributes or keys."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    @property
    def run_dir(self) -> Path:
        return Path(self["run"]["output_dir"]) / self["run"]["name"]

    def dump(self, path: Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(dict(self), sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )


def _validate(cfg: dict) -> None:
    # free-form sections whose keys are user-defined model names
    freeform = {("models", "params"), ("models", "search_spaces"),
                ("features", "season_map")}

    for section, defaults in DEFAULTS.items():
        for key in cfg[section]:
            if key not in defaults:
                raise KeyError(
                    f"unknown key '{key}' in config section '{section}'. "
                    f"Valid keys: {sorted(defaults)}"
                )
            if (section, key) in freeform:
                continue
            if (section, key) in _CHOICES and cfg[section][key] not in _CHOICES[(section, key)]:
                raise ValueError(
                    f"{section}.{key} must be one of "
                    f"{sorted(_CHOICES[(section, key)])}, got {cfg[section][key]!r}"
                )

    sp = cfg["split"]
    if sp["mode"] == "index" and sp["train_end"] is None:
        raise ValueError("split.mode='index' requires split.train_end")
    if sp["mode"] == "date" and sp["train_end_date"] is None:
        raise ValueError("split.mode='date' requires split.train_end_date")
    if sp["mode"] == "fraction":
        if not 0 < sp["train_fraction"] < 1:
            raise ValueError("split.train_fraction must be in (0, 1)")
        if not 0 <= sp["val_fraction"] < 1:
            raise ValueError("split.val_fraction must be in [0, 1)")
        if sp["train_fraction"] + sp["val_fraction"] >= 1:
            raise ValueError("train_fraction + val_fraction must leave room for a test set")
        # A two-way train/test split leaves no validation block, so
        # hyper-parameters cannot be selected on one.
        if (sp["val_fraction"] == 0 and cfg["tuning"]["enabled"]
                and cfg["tuning"]["selection_set"] == "validation"):
            raise ValueError(
                "split.val_fraction is 0, so there is no validation block to "
                "select hyper-parameters on. Either set tuning.selection_set: cv "
                "(cross-validation inside the training block - recommended for a "
                "strict train/test split) or give split.val_fraction a value "
                "such as 0.15."
            )

    for section, sub in _NESTED:
        defaults = DEFAULTS[section][sub]
        for key, value in (cfg[section][sub] or {}).items():
            if key not in defaults:
                raise KeyError(
                    f"unknown key '{key}' in config section '{section}.{sub}'. "
                    f"Valid keys: {sorted(defaults)}"
                )
            allowed = _NESTED_CHOICES.get((section, sub, key))
            if allowed and value not in allowed:
                raise ValueError(f"{section}.{sub}.{key} must be one of "
                                 f"{sorted(allowed)}, got {value!r}")

    tl = cfg["features"]["target_lags"]
    if not (tl == "auto" or isinstance(tl, (list, tuple))):
        raise ValueError("features.target_lags must be a list of lags or the "
                         f"string 'auto', got {tl!r}")

    ds = cfg["data"]
    if ds["feature_selection"] == "explicit" and not ds["feature_columns"]:
        raise ValueError("data.feature_selection='explicit' requires data.feature_columns")
    if ds["feature_selection"] == "pattern" and not ds["include_patterns"]:
        raise ValueError("data.feature_selection='pattern' requires data.include_patterns")

    from .metrics import METRICS
    tn = cfg["tuning"]
    if tn["selection_metric"] not in METRICS:
        raise ValueError(f"tuning.selection_metric must be one of {sorted(METRICS)}")
    ex = cfg["export"]
    bad_fmt = [f for f in ex["formats"] if f not in ("xlsx", "csv")]
    if bad_fmt:
        raise ValueError(f"export.formats must be 'xlsx' and/or 'csv', got {bad_fmt}")
    if ex["enabled"] and not ex["formats"]:
        raise ValueError("export.enabled is true but export.formats is empty")

    bad = [m for m in cfg["reporting"]["metrics"] if m not in METRICS]
    if bad:
        raise ValueError(f"unknown reporting.metrics {bad}; available {sorted(METRICS)}")


def load_config(path: str | Path | None = None, **overrides) -> Config:
    """Read ``path`` (YAML), merge onto :data:`DEFAULTS`, validate, return.

    ``overrides`` are merged last, which makes the pipeline scriptable::

        cfg = load_config("config.yaml", run={"name": "po_basin"})
    """
    user: dict = {}
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
        user = yaml.safe_load(text) or {}
        if not isinstance(user, dict):
            raise TypeError(f"{path} must contain a YAML mapping at the top level")
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise KeyError(
                f"unknown config section(s) {sorted(unknown)}; "
                f"valid sections: {sorted(DEFAULTS)}"
            )
    merged = _deep_merge(DEFAULTS, user)
    merged = _deep_merge(merged, overrides)
    _validate(merged)
    return Config(merged)
