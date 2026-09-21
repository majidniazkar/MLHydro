"""Hyper-parameter search.

Three selection protocols, set by ``tuning.selection_set``:

``validation`` (default)
    Fit on the training block, score on the chronologically later validation
    block. The test block is untouched until the final evaluation.
``cv``
    :class:`sklearn.model_selection.TimeSeriesSplit` over the training block
    (expanding window). More robust, ``cv_splits`` times more expensive.
``test``
    Score directly on the test block. **This is what the original notebook
    did.** It is retained only so previously published numbers can be
    reproduced; it makes the reported test skill optimistically biased,
    because the test set has then been used for model selection. The pipeline
    prints a warning whenever it is active.
"""

from __future__ import annotations

import itertools
import time
import warnings

import numpy as np
import pandas as pd

from .data import Dataset
from .fitting import fit_estimator, predict_raw
from .metrics import score
from .models import ModelSpec, search_space


def build_candidates(space: dict[str, list], cfg) -> list[dict]:
    """Materialise the list of parameter dicts to try."""
    t = cfg["tuning"]
    if not space or not t["enabled"] or t["strategy"] == "none":
        return [{}]

    keys = list(space)
    sizes = [len(space[k]) for k in keys]
    total = int(np.prod(sizes)) if sizes else 1

    if t["strategy"] == "grid":
        if total > t["max_grid_size"]:
            raise ValueError(
                f"full grid has {total} combinations, above tuning.max_grid_size="
                f"{t['max_grid_size']}. Use strategy: random, shrink the space, or "
                "raise max_grid_size."
            )
        return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]

    # random search over the discrete space, without duplicates
    rng = np.random.default_rng(cfg["run"]["seed"])
    n_iter = min(int(t["n_iter"]), total)
    seen, out = set(), []
    guard = 0
    while len(out) < n_iter and guard < 50 * n_iter + 100:
        guard += 1
        pick = tuple(int(rng.integers(0, s)) for s in sizes)
        if pick in seen:
            continue
        seen.add(pick)
        out.append({k: space[k][i] for k, i in zip(keys, pick)})
    return out


def _time_series_folds(train_pos: np.ndarray, n_splits: int):
    from sklearn.model_selection import TimeSeriesSplit
    tss = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tss.split(train_pos):
        yield train_pos[tr], train_pos[va]


def _evaluate_candidate(spec: ModelSpec, params: dict, cfg, ds: Dataset) -> float:
    """Signed score (lower is better) for one candidate."""
    metric = cfg["tuning"]["selection_metric"]
    mode = cfg["tuning"]["selection_set"]
    tr, va, te = ds.splits["train"], ds.splits["val"], ds.splits["test"]

    if mode == "cv":
        scores = []
        for f_tr, f_va in _time_series_folds(tr, cfg["tuning"]["cv_splits"]):
            est = fit_estimator(spec, params, cfg, ds, f_tr, f_va)
            scores.append(score(ds.y.to_numpy()[f_va],
                                predict_raw(est, cfg, ds, f_va), metric))
        return float(np.mean(scores))

    target_pos = te if mode == "test" else va
    if len(target_pos) == 0:
        raise ValueError(
            "tuning.selection_set='validation' but the validation split is empty. "
            "Increase split.val_fraction or set split.val_end / val_end_date."
        )
    est = fit_estimator(spec, params, cfg, ds, tr, va)
    obs = ds.y.to_numpy()[target_pos]
    return score(obs, predict_raw(est, cfg, ds, target_pos), metric)


def search(spec: ModelSpec, cfg, ds: Dataset, log=print) -> tuple[dict, pd.DataFrame]:
    """Return ``(best_params, trials)`` for one model."""
    space = search_space(spec, cfg)
    candidates = build_candidates(space, cfg)
    metric = cfg["tuning"]["selection_metric"]

    if len(candidates) == 1 and not candidates[0]:
        return {}, pd.DataFrame()

    total = int(np.prod([len(v) for v in space.values()])) if space else 1
    log(f"    tuning {spec.key}: {len(candidates)} of {total} combination(s) "
        f"on '{cfg['tuning']['selection_set']}' by {metric}")
    if (cfg["tuning"]["strategy"] == "random" and total > 50
            and len(candidates) < 0.05 * total
            and cfg["tuning"]["refine_rounds"] < 1):
        log(f"      note: sampling {100 * len(candidates) / total:.1f}% of the "
            "grid with no refinement; raise tuning.n_iter or set "
            "tuning.refine_rounds: 1")

    rows: list[dict] = []
    state = {"best_val": np.inf, "best_params": {}}
    t0 = time.time()

    def try_params(params: dict, stage: str) -> float:
        key = tuple(sorted((k, repr(v)) for k, v in params.items()))
        if key in state.setdefault("seen", set()):
            return np.inf
        state["seen"].add(key)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = _evaluate_candidate(spec, params, cfg, ds)
            failed = ""
        except Exception as exc:                      # noqa: BLE001
            val, failed = np.inf, f"{type(exc).__name__}: {exc}"[:200]
        rows.append({**params, f"{metric}_selection": val,
                     "stage": stage, "error": failed})
        if val < state["best_val"]:
            state["best_val"], state["best_params"] = val, dict(params)
        return val

    for i, params in enumerate(candidates, 1):
        try_params(params, "search")
        if cfg["run"]["verbose"] and (i % max(1, len(candidates) // 5) == 0):
            log(f"      {i}/{len(candidates)} best {metric}="
                f"{_display(state['best_val'], metric):.4f}")

    # ---- local refinement -------------------------------------------------
    # A thin random sample rarely lands on a good value for every parameter at
    # once. Holding the best configuration fixed and sweeping one parameter at
    # a time costs sum(len(values)) fits per round instead of their product,
    # and typically recovers most of the gap to an exhaustive grid.
    for rnd in range(int(cfg["tuning"]["refine_rounds"])):
        if not space or not np.isfinite(state["best_val"]):
            break
        before = state["best_val"]
        for name, values in space.items():
            for value in values:
                trial = dict(state["best_params"])
                trial[name] = value
                try_params(trial, f"refine{rnd + 1}")
        gain = before - state["best_val"]
        log(f"      refine round {rnd + 1}: best {metric}="
            f"{_display(state['best_val'], metric):.4f} (improved by {gain:.4g})")
        if gain <= 1e-12:
            break

    best_val, best_params = state["best_val"], state["best_params"]
    trials = pd.DataFrame(rows).sort_values(f"{metric}_selection").reset_index(drop=True)
    if not np.isfinite(best_val):
        errs = trials["error"].replace("", np.nan).dropna().unique()[:3]
        raise RuntimeError(f"every candidate for '{spec.key}' failed; first errors: {list(errs)}")
    log(f"    best {metric}={_display(best_val, metric):.4f} "
        f"({time.time() - t0:.1f}s) -> {best_params}")
    return best_params, trials


def _display(signed: float, metric: str) -> float:
    """Undo the sign flip applied by :func:`hydroml.metrics.score`."""
    from .metrics import HIGHER_IS_BETTER
    return -signed if HIGHER_IS_BETTER[metric] else signed
