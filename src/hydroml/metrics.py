"""Goodness-of-fit metrics for hydrological simulation.

All functions take 1-D array-likes ``obs`` (observed) and ``sim`` (simulated)
of equal length and return a float.

Note on R-squared
-----------------
The coefficient of determination ``1 - SSE/SST`` is *mathematically identical*
to the Nash-Sutcliffe Efficiency for a single series, so it is exposed only
once, as ``nse``. The metric named ``r2`` here is the **squared Pearson
correlation coefficient** (the definition used in the original notebook). The
two differ whenever the simulation is biased or mis-scaled: ``r2`` ignores bias
and variance errors, ``nse`` does not. Report both.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rmse", "mae", "nse", "kge", "pbias", "bias", "r2", "mape",
    "evaluate", "score", "METRICS", "HIGHER_IS_BETTER",
]


def _prep(obs, sim):
    obs = np.asarray(obs, dtype=float).ravel()
    sim = np.asarray(sim, dtype=float).ravel()
    if obs.shape != sim.shape:
        raise ValueError(f"shape mismatch: obs {obs.shape} vs sim {sim.shape}")
    ok = np.isfinite(obs) & np.isfinite(sim)
    return obs[ok], sim[ok]


def rmse(obs, sim) -> float:
    """Root mean squared error (same units as the target)."""
    o, s = _prep(obs, sim)
    return float(np.sqrt(np.mean((o - s) ** 2)))


def mae(obs, sim) -> float:
    """Mean absolute error (same units as the target)."""
    o, s = _prep(obs, sim)
    return float(np.mean(np.abs(o - s)))


def nse(obs, sim) -> float:
    """Nash-Sutcliffe Efficiency; 1 = perfect, 0 = no better than the mean."""
    o, s = _prep(obs, sim)
    denom = np.sum((o - np.mean(o)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((o - s) ** 2) / denom)


def kge(obs, sim) -> float:
    """Kling-Gupta Efficiency (Gupta et al., 2009); 1 = perfect."""
    o, s = _prep(obs, sim)
    if o.size < 2:
        return float("nan")
    so, ss = np.std(o), np.std(s)
    mo, ms = np.mean(o), np.mean(s)
    if so == 0 or ss == 0 or mo == 0:
        return float("nan")
    r = float(np.corrcoef(o, s)[0, 1])
    alpha = ss / so          # variability ratio
    beta = ms / mo           # bias ratio
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pbias(obs, sim) -> float:
    """Percent bias; 0 = unbiased, >0 = over-prediction."""
    o, s = _prep(obs, sim)
    tot = np.sum(o)
    if tot == 0:
        return float("nan")
    return float(100.0 * np.sum(s - o) / tot)


def bias(obs, sim) -> float:
    """Mean signed error (same units as the target)."""
    o, s = _prep(obs, sim)
    return float(np.mean(s - o))


def r2(obs, sim) -> float:
    """Squared Pearson correlation coefficient (bias-blind)."""
    o, s = _prep(obs, sim)
    if o.size < 2 or np.std(o) == 0 or np.std(s) == 0:
        return float("nan")
    return float(np.corrcoef(o, s)[0, 1] ** 2)


def mape(obs, sim) -> float:
    """Mean absolute percentage error, on non-zero observations only."""
    o, s = _prep(obs, sim)
    nz = o != 0
    if not nz.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs((o[nz] - s[nz]) / o[nz])))


METRICS = {
    "rmse": rmse, "mae": mae, "nse": nse, "kge": kge,
    "pbias": pbias, "bias": bias, "r2": r2, "mape": mape,
}

#: Direction of improvement, used by the hyper-parameter search.
HIGHER_IS_BETTER = {
    "rmse": False, "mae": False, "nse": True, "kge": True,
    "pbias": False, "bias": False, "r2": True, "mape": False,
}


def evaluate(obs, sim, names=None) -> dict:
    """Return ``{metric_name: value}`` for the requested metrics."""
    names = list(METRICS) if names is None else list(names)
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise KeyError(f"unknown metric(s) {unknown}; available: {sorted(METRICS)}")
    return {n: METRICS[n](obs, sim) for n in names}


def score(obs, sim, metric: str) -> float:
    """Signed score where **lower is always better** (for argmin selection)."""
    val = METRICS[metric](obs, sim)
    if metric in ("pbias", "bias"):
        val = abs(val)
    elif HIGHER_IS_BETTER[metric]:
        val = -val
    return float("inf") if not np.isfinite(val) else float(val)
