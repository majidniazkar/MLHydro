"""Cross-validated evaluation for time series.

The holdout protocol (train / validation / test) gives **one** number per model
from **one** period. That number is sensitive to whether the test years
happened to be wet or dry, which is exactly the criticism a reviewer raises
first. Cross-validation over several folds gives a mean and a spread, so two
models whose holdout NSE differs by 0.01 can be recognised as indistinguishable.

Two schemes, both respecting time order:

``expanding`` (default, :class:`~sklearn.model_selection.TimeSeriesSplit`)
    Fold *k* trains on everything before its test window. Mimics operational
    forecasting, where you only ever have the past.

``blocked``
    Contiguous equal blocks; each block is the test set once and the *other*
    blocks are the training set. Uses more data per fold and covers the whole
    record, but a model can train on a period later than the one it predicts.
    ``gap`` drops rows either side of the test block so autocorrelation does
    not leak across the boundary.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import Dataset
from .fitting import fit_estimator, predict_raw
from .metrics import evaluate
from .models import ModelSpec


def make_folds(cfg, ds: Dataset) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``[(train_positions, test_positions), ...]`` in time order."""
    c = cfg["evaluation"]["cv"]
    if c["over"] == "train_val":
        pool = np.concatenate([ds.splits["train"], ds.splits["val"]])
    else:
        pool = np.arange(len(ds.X))
    pool = np.sort(pool)
    n_splits = int(c["n_splits"])
    gap = int(c["gap"])

    if len(pool) < n_splits + 2:
        raise ValueError(
            f"only {len(pool)} rows available for {n_splits} CV folds; "
            "reduce evaluation.cv.n_splits"
        )

    folds = []
    if c["scheme"] == "expanding":
        from sklearn.model_selection import TimeSeriesSplit
        kw = {"n_splits": n_splits, "gap": gap}
        if c["test_size"]:
            kw["test_size"] = int(c["test_size"])
        for tr, te in TimeSeriesSplit(**kw).split(pool):
            folds.append((pool[tr], pool[te]))
    elif c["scheme"] == "blocked":
        blocks = np.array_split(pool, n_splits)
        for i, te in enumerate(blocks):
            tr = np.concatenate([b for j, b in enumerate(blocks) if j != i])
            if gap > 0:
                lo, hi = te[0] - gap, te[-1] + gap
                tr = tr[(tr < lo) | (tr > hi)]
            if len(tr) and len(te):
                folds.append((np.sort(tr), np.sort(te)))
    else:
        raise ValueError("evaluation.cv.scheme must be 'expanding' or 'blocked'")

    return [(tr, te) for tr, te in folds if len(tr) >= 10 and len(te) >= 2]


def cross_validate(specs: list[ModelSpec], cfg, ds: Dataset,
                   best_params: dict, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold-wise and summarised CV metrics for every model.

    ``best_params`` are the hyper-parameters chosen by the holdout search. With
    ``use_best_params: true`` they are reused in every fold, which is cheap but
    mildly optimistic because those parameters were selected with knowledge of
    the holdout validation block. With ``use_best_params: false`` the search is
    repeated inside each fold (nested CV): statistically clean, and
    ``n_splits`` times the cost.
    """
    c = cfg["evaluation"]["cv"]
    folds = make_folds(cfg, ds)
    metric_names = cfg["reporting"]["metrics"]
    log(f"[cv] {c['scheme']} scheme, {len(folds)} folds over '{c['over']}' rows"
        + ("" if c["use_best_params"] else ", re-tuning inside each fold"))
    for i, (tr, te) in enumerate(folds, 1):
        log(f"     fold {i}: train n={len(tr)} "
            f"({ds.X.index[tr[0]].date()}..{ds.X.index[tr[-1]].date()})  "
            f"test n={len(te)} "
            f"({ds.X.index[te[0]].date()}..{ds.X.index[te[-1]].date()})")

    rows = []
    for spec in specs:
        for i, (tr, te) in enumerate(folds, 1):
            params = best_params.get(spec.key, {})
            try:
                if not c["use_best_params"]:
                    params = _tune_within(spec, cfg, ds, tr)
                # hold back the tail of the fold's training rows for the
                # early-stopping monitor of the neural models
                cut = max(1, int(0.85 * len(tr)))
                inner_tr, inner_val = tr[:cut], tr[cut:]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    est = fit_estimator(spec, params, cfg, ds,
                                        inner_tr if spec.kind == "keras" else tr,
                                        inner_val)
                vals = evaluate(ds.y.to_numpy()[te],
                                predict_raw(est, cfg, ds, te), metric_names)
                rows.append({"model": spec.key, "fold": i, **vals})
            except Exception as exc:                          # noqa: BLE001
                log(f"     {spec.key} fold {i} failed: {type(exc).__name__}: {exc}")
                rows.append({"model": spec.key, "fold": i,
                             **{m: np.nan for m in metric_names}})
        got = [r for r in rows if r["model"] == spec.key
               and np.isfinite(r[metric_names[0]])]
        log(f"     {spec.key}: {len(got)}/{len(folds)} folds completed")

    long = pd.DataFrame(rows)
    if long.empty:
        return long, long
    summary = (long.groupby("model", sort=False)[metric_names]
                   .agg(["mean", "std"]))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary["folds"] = (long.groupby("model", sort=False)[metric_names[0]]
                            .apply(lambda s: int(s.notna().sum())).to_numpy())
    return long, summary


def _tune_within(spec: ModelSpec, cfg, ds: Dataset, train_pos: np.ndarray) -> dict:
    """Nested search: split the fold's training rows into fit / select parts."""
    from .tuning import build_candidates
    from .metrics import score
    from .models import search_space

    space = search_space(spec, cfg)
    candidates = build_candidates(space, cfg)
    if len(candidates) == 1:
        return candidates[0]
    cut = max(1, int(0.8 * len(train_pos)))
    inner_tr, inner_va = train_pos[:cut], train_pos[cut:]
    if len(inner_va) < 2:
        return {}
    metric = cfg["tuning"]["selection_metric"]
    obs = ds.y.to_numpy()[inner_va]
    best, best_val = {}, np.inf
    for params in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = fit_estimator(spec, params, cfg, ds, inner_tr, inner_va)
            val = score(obs, predict_raw(est, cfg, ds, inner_va), metric)
        except Exception:                                     # noqa: BLE001
            val = np.inf
        if val < best_val:
            best, best_val = params, val
    return best
