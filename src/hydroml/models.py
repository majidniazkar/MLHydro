"""Model registry.

Each entry pairs a *factory* (``**params -> estimator`` with ``fit``/``predict``)
with a *default search space*. Adding a model to the template means adding one
``register(...)`` call here; using it means adding its key to
``models.enabled`` in ``config.yaml``.

Optional dependencies (xgboost, lightgbm, catboost, tensorflow) are probed
lazily: if a package is absent the model is reported as unavailable and skipped
instead of crashing the run.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

REGISTRY: dict[str, "ModelSpec"] = {}


@dataclass
class ModelSpec:
    key: str
    label: str
    factory: Callable[..., Any]
    space: dict[str, list] = field(default_factory=dict)
    requires: tuple[str, ...] = ()        # ALL of these must be importable
    requires_any: tuple[str, ...] = ()    # at least ONE of these
    kind: str = "sklearn"                 # "sklearn" or "keras"
    notes: str = ""

    def available(self) -> bool:
        return not self.missing()

    def missing(self) -> list[str]:
        gaps = [m for m in self.requires if importlib.util.find_spec(m) is None]
        if self.requires_any and all(
                importlib.util.find_spec(m) is None for m in self.requires_any):
            gaps.append(" or ".join(self.requires_any))
        return gaps


def register(spec: ModelSpec) -> ModelSpec:
    REGISTRY[spec.key] = spec
    return spec


def _int_range(lo, hi, step=1):
    return [int(v) for v in np.arange(lo, hi, step)]


def _float_range(lo, hi, step):
    return [round(float(v), 6) for v in np.arange(lo, hi, step)]


# --------------------------------------------------------------------------- #
# linear / kernel / neighbour models
# --------------------------------------------------------------------------- #
def _linear(**p):
    from sklearn.linear_model import LinearRegression
    return LinearRegression(**p)


register(ModelSpec("linear_regression", "Linear Regression", _linear, {}))


def _bayesian_ridge(**p):
    from sklearn.linear_model import BayesianRidge
    return BayesianRidge(**p)


register(ModelSpec(
    "bayesian_ridge", "Bayesian Ridge", _bayesian_ridge,
    {"alpha_1": [1e-6, 1e-4, 1e-2], "lambda_1": [1e-6, 1e-4, 1e-2]},
))


def _huber(**p):
    from sklearn.linear_model import HuberRegressor
    return HuberRegressor(**p)


register(ModelSpec(
    "huber", "Huber Regressor", _huber,
    {
        "epsilon": _float_range(1.05, 5.05, 0.25),
        "max_iter": _int_range(100, 1100, 100),
        "alpha": [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    },
))


def _sgd(seed=None, **p):
    from sklearn.linear_model import SGDRegressor
    return SGDRegressor(random_state=seed, **p)


register(ModelSpec(
    "sgd", "SGD Regressor", _sgd,
    {
        "max_iter": _int_range(100, 2100, 100),
        "loss": ["squared_error", "huber", "epsilon_insensitive",
                 "squared_epsilon_insensitive"],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
    },
))


def _knn(**p):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(**p)


register(ModelSpec(
    "knn", "K-Nearest Neighbours", _knn,
    {
        "n_neighbors": _int_range(2, 21),
        "p": [1, 2],
        "weights": ["distance", "uniform"],
    },
))


def _svr(**p):
    from sklearn.svm import SVR
    return SVR(**p)


register(ModelSpec(
    "svr", "Support Vector Regression", _svr,
    {
        "kernel": ["rbf", "poly"],
        "C": [0.1, 0.2, 0.4, 0.7, 1.0, 5.0, 10.0],
        "degree": [2, 3],
        "gamma": ["scale"],
        "epsilon": [0.01, 0.05, 0.1],
    },
    notes="O(n^2) training; consider subsampling for very long series.",
))


# --------------------------------------------------------------------------- #
# tree ensembles
# --------------------------------------------------------------------------- #
def _decision_tree(seed=None, **p):
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(random_state=seed, **p)


register(ModelSpec(
    "decision_tree", "Decision Tree", _decision_tree,
    {
        "criterion": ["squared_error", "friedman_mse"],
        "max_depth": _int_range(2, 21),
        "min_samples_split": _int_range(2, 10),
    },
))


def _random_forest(seed=None, n_jobs=-1, **p):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **p)


register(ModelSpec(
    "random_forest", "Random Forest", _random_forest,
    {
        "n_estimators": [100, 200, 400, 700, 1000],
        "max_depth": [None, 5, 9, 13, 17],
        "min_samples_split": [2, 4, 8],
        "max_features": [1.0, "sqrt"],
    },
    notes="criterion='absolute_error' is supported but ~100x slower; "
          "set it via models.params if you need it.",
))


def _adaboost(seed=None, **p):
    from sklearn.ensemble import AdaBoostRegressor
    return AdaBoostRegressor(random_state=seed, **p)


register(ModelSpec(
    "adaboost", "AdaBoost", _adaboost,
    {
        "loss": ["linear", "square", "exponential"],
        "n_estimators": [50, 100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
    },
))


def _gbr(seed=None, **p):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(random_state=seed, **p)


register(ModelSpec(
    "gradient_boosting", "Gradient Boosting", _gbr,
    {
        "loss": ["squared_error", "huber"],
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 8, 12],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.4],
        "min_samples_split": [2, 4, 8],
    },
))


def _hgbr(seed=None, **p):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=seed, **p)


register(ModelSpec(
    "hist_gradient_boosting", "Histogram Gradient Boosting", _hgbr,
    {
        "max_iter": [100, 200, 400, 800],
        "loss": ["squared_error", "absolute_error", "poisson"],
        "learning_rate": [0.02, 0.05, 0.1, 0.2, 0.4],
        "max_depth": [None, 4, 8, 13],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [5, 10, 20, 40],
    },
    notes="loss='poisson' requires a non-negative target.",
))


def _xgb(seed=None, n_jobs=-1, **p):
    from xgboost import XGBRegressor
    return XGBRegressor(random_state=seed, n_jobs=n_jobs, verbosity=0, **p)


register(ModelSpec(
    "xgboost", "XGBoost", _xgb,
    {
        "n_estimators": [100, 200, 400, 800],
        "max_depth": [3, 6, 9, 12],
        "learning_rate": [0.02, 0.05, 0.1, 0.3],
        "reg_alpha": [0.0, 0.3, 0.7, 1.5],
        "reg_lambda": [0.0, 0.3, 0.7, 1.5],
        "subsample": [0.8, 1.0],
    },
    requires=("xgboost",),
))


def _lgbm(seed=None, n_jobs=-1, **p):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1, **p)


register(ModelSpec(
    "lightgbm", "LightGBM", _lgbm,
    {
        "boosting_type": ["gbdt", "dart"],
        "n_estimators": [100, 200, 400, 800],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "max_depth": [-1, 3, 6, 10],
        "num_leaves": [8, 15, 31, 63],
        "min_child_samples": [5, 20, 50],
    },
    requires=("lightgbm",),
))


def _catboost(seed=None, **p):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(random_seed=seed, verbose=0, allow_writing_files=False, **p)


register(ModelSpec(
    "catboost", "CatBoost", _catboost,
    {
        "iterations": [200, 500, 1000, 1900],
        "learning_rate": [0.02, 0.05, 0.1, 0.3],
        "depth": [4, 5, 6, 8],
        "loss_function": ["RMSE", "MAE"],
    },
    requires=("catboost",),
))


# --------------------------------------------------------------------------- #
# neural networks (optional TensorFlow)
# --------------------------------------------------------------------------- #
def _mlp(seed=None, **p):
    from sklearn.neural_network import MLPRegressor
    p.setdefault("max_iter", 600)
    p.setdefault("early_stopping", True)
    return MLPRegressor(random_state=seed, **p)


register(ModelSpec(
    "mlp", "MLP (scikit-learn)", _mlp,
    {
        "hidden_layer_sizes": [(18, 36, 36), (32, 32), (64, 32, 16), (100,)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 5e-4],
    },
    notes="Needs no deep-learning framework; use it when torch/tensorflow "
          "are unavailable, or as a sanity check on the ann model.",
))


def _ann(seed=None, deep_cfg=None, n_features=None, **p):
    from .deep import make_dense
    kw = {**(deep_cfg or {}), **p}
    backend = kw.pop("backend", "auto")
    return make_dense(backend=backend, seed=seed, n_features=n_features, **kw)


register(ModelSpec(
    "ann", "ANN (Dense, torch/tensorflow)", _ann,
    {
        "hidden_units": [[18, 36, 36], [32, 32], [64, 32, 16]],
        "learning_rate": [1e-3, 5e-4],
        "batch_size": [32, 64],
        "dropout": [0.0, 0.1],
    },
    requires_any=("torch", "tensorflow"), kind="keras",
))


def _lstm(seed=None, deep_cfg=None, n_features=None, **p):
    from .deep import make_lstm
    kw = {**(deep_cfg or {}), **p}
    backend = kw.pop("backend", "auto")
    return make_lstm(backend=backend, seed=seed, n_features=n_features, **kw)


register(ModelSpec(
    "lstm", "LSTM (torch/tensorflow)", _lstm,
    {
        "units": [[36, 36], [64, 32], [50]],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "batch_size": [32, 64],
        "window": [15, 30, 60],
    },
    requires_any=("torch", "tensorflow"), kind="keras",
    notes="sequence_mode='sliding_window' (default) feeds the model real "
          "consecutive timesteps; 'features_as_timesteps' reproduces the "
          "original notebook, where the lag columns were used as the time axis.",
))


# --------------------------------------------------------------------------- #
def resolve_models(cfg) -> tuple[list[ModelSpec], list[tuple[str, str]]]:
    """Return ``(specs_to_run, skipped)`` for the enabled model keys."""
    enabled = cfg["models"]["enabled"]
    unknown = [k for k in enabled if k not in REGISTRY]
    if unknown:
        raise KeyError(
            f"models.enabled contains unknown key(s) {unknown}. "
            f"Available: {sorted(REGISTRY)}"
        )
    specs, skipped = [], []
    for key in enabled:
        spec = REGISTRY[key]
        if spec.available():
            specs.append(spec)
        else:
            reason = f"missing package(s): {', '.join(spec.missing())}"
            if not cfg["models"]["skip_unavailable"]:
                raise ImportError(f"model '{key}' unavailable -- {reason}")
            skipped.append((key, reason))
    return specs, skipped


def search_space(spec: ModelSpec, cfg) -> dict[str, list]:
    """Default space for ``spec``, overridden by ``models.search_spaces``."""
    space = {k: list(v) for k, v in spec.space.items()}
    space.update(cfg["models"]["search_spaces"].get(spec.key, {}) or {})
    # a fixed parameter is not a searched one
    for fixed in (cfg["models"]["params"].get(spec.key, {}) or {}):
        space.pop(fixed, None)
    return space
