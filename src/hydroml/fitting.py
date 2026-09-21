"""Estimator construction, fitting and back-transformation of predictions.

Every model in the template is trained on the **scaled** feature matrix and the
**scaled** target, and every prediction is pushed back through the target
scaler before any metric is computed. Keeping that in one place is what stops
the classic mistake of fitting on raw units and then inverse-transforming the
output (or the reverse), which silently corrupts the reported skill.
"""

from __future__ import annotations

import contextlib
import inspect

import numpy as np

from .data import Dataset
from .models import ModelSpec


def build_estimator(spec: ModelSpec, params: dict, cfg, ds: Dataset):
    """Instantiate ``spec`` with fixed config params + search params."""
    fixed = dict(cfg["models"]["params"].get(spec.key, {}) or {})
    merged = {**fixed, **(params or {})}

    sig = inspect.signature(spec.factory)
    accepts = set(sig.parameters)
    has_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    extra = {}
    if "seed" in accepts:
        extra["seed"] = cfg["run"]["seed"]
    if "n_jobs" in accepts:
        extra["n_jobs"] = cfg["run"]["n_jobs"]
    if "n_features" in accepts:
        extra["n_features"] = ds.n_features
    if "deep_cfg" in accepts:
        extra["deep_cfg"] = dict(cfg["deep"].get(spec.key, {}) or {})

    if not has_varkw:
        merged = {k: v for k, v in merged.items() if k in accepts}
    return spec.factory(**extra, **merged)


def fit_estimator(spec: ModelSpec, params: dict, cfg, ds: Dataset,
                  train_pos: np.ndarray, val_pos: np.ndarray | None = None):
    """Fit on ``train_pos``; Keras models also get ``val_pos`` for early stopping."""
    Xtr, ytr = ds.Xs[train_pos], ds.ys[train_pos].ravel()

    def _fit(estimator):
        if spec.kind == "keras" and val_pos is not None and len(val_pos) > 0:
            estimator.fit(Xtr, ytr, X_val=ds.Xs[val_pos], y_val=ds.ys[val_pos].ravel())
        else:
            estimator.fit(Xtr, ytr)
        return estimator

    est = build_estimator(spec, params, cfg, ds)
    try:
        return _fit(est)
    except (PermissionError, OSError) as exc:
        # Locked-down containers can forbid the worker pools joblib uses for
        # n_jobs != 1 (WinError 5 on named pipes, EPERM on fork). Retry
        # sequentially rather than dropping the model from the comparison.
        with _sequential():
            import copy
            alt = copy.deepcopy(dict(cfg))
            alt["run"]["n_jobs"] = 1
            try:
                return _fit(build_estimator(spec, params, alt, ds))
            except Exception:
                raise exc


@contextlib.contextmanager
def _sequential():
    """Force single-threaded execution for one fit.

    Two mechanisms are needed: joblib's backend covers estimators that build
    their own worker pool (random forests), while ``threadpool_limits`` covers
    the OpenMP-driven ones whose thread count comes from the runtime
    (histogram gradient boosting). ``threadpoolctl`` ships with scikit-learn.
    """
    import joblib
    with contextlib.ExitStack() as stack:
        try:
            import threadpoolctl
            stack.enter_context(threadpoolctl.threadpool_limits(limits=1))
        except ImportError:                   # pragma: no cover
            pass
        try:
            stack.enter_context(joblib.parallel_config(backend="sequential", n_jobs=1))
        except AttributeError:                # joblib < 1.3
            stack.enter_context(joblib.parallel_backend("sequential", n_jobs=1))
        yield


def predict_raw(est, cfg, ds: Dataset, pos: np.ndarray) -> np.ndarray:
    """Predict at positions ``pos`` and return values in the target's own units."""
    if len(pos) == 0:
        return np.empty(0)
    pred = ds.inverse_y(est.predict(ds.Xs[pos]))
    if cfg["preprocess"]["clip_negative_predictions"]:
        pred = np.clip(pred, 0.0, None)
    return pred
