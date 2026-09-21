"""Input diagnostics: memory, lag structure, seasonality, feature importance.

These answer the question the model comparison cannot: *are the right
predictors in the table at all?* A tuned ensemble on a feature set that lacks
the basin's memory will plateau at a mediocre NSE no matter how long the search
runs, and no hyper-parameter will recover it.

Everything here is computed on the **training rows only**, so a lag chosen by
:func:`suggest_lags` has not seen the validation or test period.

No dependency beyond numpy/pandas: the partial autocorrelation is obtained from
the autocorrelation sequence by the Levinson-Durbin recursion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "acf", "pacf", "autocorrelation_table", "cross_correlation_table",
    "select_best_lags", "suggest_lags", "suggest_target_lags", "seasonal_table",
    "seasonality_strength", "permutation_importance",
]


# --------------------------------------------------------------------------- #
# memory of the target series
# --------------------------------------------------------------------------- #
def acf(y, max_lag: int = 30) -> np.ndarray:
    """Sample autocorrelation for lags ``0..max_lag`` (lag 0 = 1)."""
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    n = y.size
    if n < 3:
        return np.full(max_lag + 1, np.nan)
    yc = y - y.mean()
    denom = np.dot(yc, yc)
    if denom == 0:
        return np.full(max_lag + 1, np.nan)
    max_lag = int(min(max_lag, n - 2))
    return np.array([np.dot(yc[: n - k], yc[k:]) / denom for k in range(max_lag + 1)])


def pacf(y, max_lag: int = 30) -> np.ndarray:
    """Partial autocorrelation via Levinson-Durbin on the ACF sequence."""
    r = acf(y, max_lag)
    if np.isnan(r).all():
        return r
    m = r.size - 1
    out = np.empty(m + 1)
    out[0] = 1.0
    phi = np.zeros(m + 1)
    v = 1.0
    for k in range(1, m + 1):
        num = r[k] - sum(phi[j] * r[k - j] for j in range(1, k))
        refl = num / v if v > 0 else 0.0
        out[k] = refl
        new = phi.copy()
        new[k] = refl
        for j in range(1, k):
            new[j] = phi[j] - refl * phi[k - j]
        phi = new
        v *= max(1e-12, 1.0 - refl ** 2)
    return out


def autocorrelation_table(y, max_lag: int = 30) -> pd.DataFrame:
    """``lag, acf, pacf, significant`` for the target series."""
    n = int(np.isfinite(np.asarray(y, dtype=float)).sum())
    a, p = acf(y, max_lag), pacf(y, max_lag)
    band = 1.96 / np.sqrt(max(1, n))          # ~95% white-noise band
    lags = np.arange(a.size)
    return pd.DataFrame({
        "lag": lags, "acf": a, "pacf": p,
        "conf_band": band,
        "acf_significant": np.abs(a) > band,
        "pacf_significant": np.abs(p) > band,
    })


# --------------------------------------------------------------------------- #
# driver-to-target lag structure
# --------------------------------------------------------------------------- #
def cross_correlation_table(df: pd.DataFrame, target: str, drivers: list[str],
                            max_lag: int = 30) -> pd.DataFrame:
    """Correlation of each driver, shifted by ``lag`` days, against the target.

    A peak at lag ``k`` means the driver leads the target by ``k`` steps, i.e.
    ``driver_lag{k}`` is the informative predictor column to build.
    """
    y = df[target].astype(float)
    n = len(df)
    band = 1.96 / np.sqrt(max(1, n))
    rows = []
    for col in drivers:
        s = df[col].astype(float)
        for lag in range(0, int(max_lag) + 1):
            shifted = s.shift(lag)
            ok = shifted.notna() & y.notna()
            if ok.sum() < 5 or shifted[ok].std() == 0 or y[ok].std() == 0:
                r = np.nan
            else:
                r = float(np.corrcoef(shifted[ok], y[ok])[0, 1])
            rows.append({"driver": col, "lag": lag, "corr": r,
                         "abs_corr": abs(r) if np.isfinite(r) else np.nan,
                         "conf_band": band,
                         "significant": bool(np.isfinite(r) and abs(r) > band)})
    return pd.DataFrame(rows)


def select_best_lags(df: pd.DataFrame, target: str, drivers: list[str],
                     max_lag: int = 30, top_k: int = 1,
                     min_abs_corr: float = 0.1) -> tuple[dict[str, list[int]], pd.DataFrame]:
    """Per-driver travel-time selection: is a lagged copy better than the raw series?

    For each driver, the correlation with the target is computed at every lag
    from 0 to ``max_lag`` and the ``top_k`` strongest are kept. The comparison
    that matters hydrologically is the chosen lag against **lag 0**: a sub-basin
    whose runoff needs two days to reach the outlet correlates better as
    ``prec1`` shifted by 2 than as ``prec1`` itself, and the selected lag is an
    estimate of that travel time.

    Returns ``(plan, report)`` where ``plan`` maps driver -> chosen lags (usable
    directly as a lag plan) and ``report`` documents the decision per driver:

    ==================  =======================================================
    ``best_lag``        lag with the strongest \\|correlation\\|
    ``best_corr``       its signed correlation
    ``corr_at_lag0``    correlation of the unshifted series
    ``gain_vs_lag0``    \\|best_corr\\| - \\|corr_at_lag0\\|; >0 means lagging helps
    ``selected_lags``   what the plan will build
    ``decision``        human-readable summary
    ==================  =======================================================

    Computed on whatever rows are passed in -- :func:`hydroml.data.build_dataset`
    passes the pre-test slice, so the choice never sees the test period.
    """
    table = cross_correlation_table(df, target, drivers, max_lag)
    plan: dict[str, list[int]] = {}
    rows = []
    for col, part in table.groupby("driver", sort=False):
        part = part.dropna(subset=["abs_corr"])
        if part.empty:
            rows.append({"driver": str(col), "best_lag": None, "best_corr": np.nan,
                         "corr_at_lag0": np.nan, "gain_vs_lag0": np.nan,
                         "selected_lags": "", "n_rows": len(df),
                         "decision": "dropped (correlation undefined - constant column?)"})
            continue

        best = part.loc[part["abs_corr"].idxmax()]
        at0 = part[part["lag"] == 0]
        corr0 = float(at0["corr"].iloc[0]) if len(at0) else np.nan
        gain = float(best["abs_corr"] - abs(corr0)) if np.isfinite(corr0) else np.nan
        keep = bool(best["abs_corr"] >= min_abs_corr)
        chosen = sorted(int(v) for v in part.nlargest(int(top_k), "abs_corr")["lag"]) if keep else []

        if not keep:
            decision = (f"dropped (best |corr|={best['abs_corr']:.3f} < "
                        f"min_abs_corr={min_abs_corr})")
        elif chosen == [0]:
            decision = "used as-is (lag 0 is best)"
        else:
            lags = ", ".join(str(k) for k in chosen)
            decision = f"lagged by {lags}" + (f" (gain {gain:+.3f} over lag 0)"
                                              if np.isfinite(gain) else "")
        if keep:
            plan[str(col)] = chosen
        rows.append({"driver": str(col), "best_lag": int(best["lag"]),
                     "best_corr": float(best["corr"]), "corr_at_lag0": corr0,
                     "gain_vs_lag0": gain,
                     "selected_lags": ", ".join(str(k) for k in chosen),
                     "n_rows": len(df), "decision": decision})
    report = pd.DataFrame(rows)
    return plan, report


def suggest_lags(df: pd.DataFrame, target: str, drivers: list[str],
                 max_lag: int = 30, top_k: int = 3,
                 min_abs_corr: float = 0.1) -> dict[str, list[int]]:
    """Pick, per driver, the ``top_k`` lags with the strongest |correlation|.

    Lags below ``min_abs_corr`` are dropped, so a driver with no relationship
    to the target contributes no columns instead of ``top_k`` noise columns.
    """
    table = cross_correlation_table(df, target, drivers, max_lag)
    out: dict[str, list[int]] = {}
    for col, part in table.groupby("driver", sort=False):
        part = part.dropna(subset=["abs_corr"])
        part = part[part["abs_corr"] >= min_abs_corr]
        chosen = sorted(int(v) for v in part.nlargest(int(top_k), "abs_corr")["lag"])
        if chosen:
            out[str(col)] = chosen
    return out


def suggest_target_lags(y, max_lag: int = 10, top_k: int = 3,
                        min_abs_pacf: float = 0.05) -> list[int]:
    """Autoregressive lags of the target, ranked by |PACF| (lag >= 1).

    The PACF is the right statistic here: the ACF of a discharge series decays
    slowly and would nominate every lag, while the PACF isolates the lags that
    add information beyond the shorter ones.
    """
    tab = autocorrelation_table(y, max_lag)
    tab = tab[(tab["lag"] >= 1) & (tab["pacf"].abs() >= min_abs_pacf)]
    if tab.empty:
        return []
    ranked = tab.assign(strength=tab["pacf"].abs()).nlargest(int(top_k), "strength")
    return sorted(int(v) for v in ranked["lag"])


# --------------------------------------------------------------------------- #
# seasonality
# --------------------------------------------------------------------------- #
def seasonal_table(y: pd.Series, season_map: dict | None = None) -> pd.DataFrame:
    """Monthly (and optionally seasonal) statistics of the target."""
    s = pd.Series(np.asarray(y, dtype=float), index=y.index)
    g = s.groupby(s.index.month)
    monthly = pd.DataFrame({
        "group": [f"month_{m}" for m in g.size().index],
        "n": g.size().to_numpy(),
        "mean": g.mean().to_numpy(),
        "std": g.std().to_numpy(),
        "q10": g.quantile(0.10).to_numpy(),
        "median": g.median().to_numpy(),
        "q90": g.quantile(0.90).to_numpy(),
    })
    if not season_map:
        return monthly
    lookup = {m: name for name, months in season_map.items() for m in months}
    sg = s.groupby(s.index.month.map(lookup))
    seasonal = pd.DataFrame({
        "group": sg.size().index.astype(str),
        "n": sg.size().to_numpy(),
        "mean": sg.mean().to_numpy(),
        "std": sg.std().to_numpy(),
        "q10": sg.quantile(0.10).to_numpy(),
        "median": sg.median().to_numpy(),
        "q90": sg.quantile(0.90).to_numpy(),
    })
    return pd.concat([seasonal, monthly], ignore_index=True)


def seasonality_strength(y: pd.Series) -> float:
    """Fraction of target variance explained by the month-of-year mean.

    Near 0 means the calendar carries no information beyond what the drivers
    already provide; a large value justifies the season/month/harmonic
    predictors.
    """
    s = pd.Series(np.asarray(y, dtype=float), index=y.index).dropna()
    if s.empty or s.var() == 0:
        return float("nan")
    clim = s.groupby(s.index.month).transform("mean")
    return float(1.0 - ((s - clim) ** 2).sum() / ((s - s.mean()) ** 2).sum())


# --------------------------------------------------------------------------- #
# model-based feature importance
# --------------------------------------------------------------------------- #
def permutation_importance(est, cfg, ds, part: str = "val", metric: str = "nse",
                           n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Drop in ``metric`` when each feature column is shuffled.

    Model-agnostic (works for the Keras/Torch wrappers too) and computed in the
    target's own units, so the numbers are directly interpretable: an
    importance of 0.12 in NSE means shuffling that column costs 0.12 NSE.
    """
    from .fitting import predict_raw
    from .metrics import METRICS

    pos = ds.splits[part]
    if len(pos) == 0:
        return pd.DataFrame()
    obs = ds.y.to_numpy()[pos]
    base = METRICS[metric](obs, predict_raw(est, cfg, ds, pos))

    rng = np.random.default_rng(seed)
    saved = ds.Xs
    rows = []
    for j, name in enumerate(ds.features):
        drops = []
        for _ in range(int(n_repeats)):
            X = saved.copy()
            block = X[pos, j]
            X[pos, j] = rng.permutation(block)
            ds.Xs = X
            try:
                val = METRICS[metric](obs, predict_raw(est, cfg, ds, pos))
            finally:
                ds.Xs = saved
            drops.append(base - val)
        rows.append({"feature": name, "importance": float(np.mean(drops)),
                     "importance_std": float(np.std(drops))})
    out = pd.DataFrame(rows).sort_values("importance", ascending=False)
    out.insert(0, "metric", metric)
    out.insert(1, "baseline", base)
    return out.reset_index(drop=True)
