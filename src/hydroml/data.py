"""Data loading, feature engineering, chronological splitting and scaling.

Everything here is driven by the ``data``, ``features``, ``split`` and
``preprocess`` sections of the config -- no column name, file path or split
position is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

_SCALERS = {"minmax": MinMaxScaler, "standard": StandardScaler, "none": None}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_table(cfg) -> pd.DataFrame:
    """Read the input file and return a frame indexed by a DatetimeIndex."""
    d = cfg["data"]
    path = Path(d["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"input file not found: {path.resolve()}\n"
            "Set data.path in config.yaml to your climate/flow table."
        )
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls", ".xlsm"):
        df = pd.read_excel(path, sheet_name=d["sheet"])
    elif suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported input extension {suffix!r}")

    if isinstance(df, dict):  # sheet_name=None
        raise ValueError("data.sheet resolved to multiple sheets; name a single sheet")

    date_col, target_col = d["date_column"], d["target_column"]
    missing = [c for c in (date_col, target_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"column(s) {missing} not found in {path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=d["dayfirst"])
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"

    if d["resample"]:
        df = df.resample(d["resample"]).agg(d["resample_agg"])
    if d["target_min"] is not None:
        df = df[df[target_col] >= d["target_min"]]
    return df


# --------------------------------------------------------------------------- #
# feature engineering
# --------------------------------------------------------------------------- #
def _season_series(index: pd.DatetimeIndex, season_map: dict) -> pd.Series:
    lookup = {m: name for name, months in season_map.items() for m in months}
    missing = sorted(set(range(1, 13)) - set(lookup))
    if missing:
        raise ValueError(f"features.season_map does not cover month(s) {missing}")
    return pd.Series(index.month.map(lookup), index=index, name="season")


def resolve_lag_plan(df: pd.DataFrame, cfg, log=None) -> dict:
    """Decide which lags to build, from the config or from the data.

    With ``features.auto_lags.select: cross_correlation`` the lags are chosen
    by the strongest |correlation| between each shifted driver and the target,
    and the target's own lags by |PACF|. ``df`` must contain **only rows before
    the test period** -- :func:`build_dataset` passes the pre-test slice.
    """
    from .diagnostics import select_best_lags, suggest_target_lags

    f = cfg["features"]
    target = cfg["data"]["target_column"]
    auto = f["auto_lags"]
    plan: dict = {"drivers": {}, "target": [], "report": None,
                  "drop_unlagged": bool(auto["drop_unlagged"])}

    if auto["enabled"]:
        cols = list(auto["columns"]) or [c for c in df.columns if c != target]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"features.auto_lags.columns references missing column(s) {missing}")
        if auto["select"] == "cross_correlation":
            plan["drivers"], plan["report"] = select_best_lags(
                df, target, cols, max_lag=auto["max_lag"], top_k=auto["top_k"],
                min_abs_corr=auto["min_abs_corr"])
            if log:
                log(f"[lags] travel-time selection over lags 0..{auto['max_lag']} "
                    f"on {len(df)} pre-test rows:")
                for r in plan["report"].itertuples():
                    log(f"       {r.driver:<16} {r.decision}")
            if not plan["drivers"]:
                raise ValueError(
                    "cross-correlation lag selection found no driver above "
                    f"min_abs_corr={auto['min_abs_corr']}. Lower the threshold "
                    "or select lags manually."
                )
        else:
            lags = auto["lags"]
            plan["drivers"] = ({str(k): list(v) for k, v in lags.items()}
                               if isinstance(lags, dict)
                               else {c: list(lags) for c in cols})

    tl = f["target_lags"]
    if tl == "auto":
        plan["target"] = suggest_target_lags(
            df[target], max_lag=f["target_lag_max"], top_k=f["target_lag_top_k"])
        if log:
            log(f"[lags] target autoregressive lags by |PACF| -> "
                f"{plan['target'] or 'none above threshold'}")
    else:
        plan["target"] = list(tl)
    return plan


def add_calendar_features(frame: pd.DataFrame, cfg) -> tuple[pd.DataFrame, list[str]]:
    """Append the configured calendar predictors; return ``(frame, names)``.

    Shared by training and projection so that a future period gets **exactly**
    the same columns, in the same order, from the same season map -- a mismatch
    here would silently feed the model a permuted feature vector.
    """
    f = cfg["features"]
    out = frame
    names: list[str] = []
    if f["add_season_onehot"]:
        seasons = _season_series(out.index, f["season_map"])
        dummies = pd.get_dummies(seasons, prefix="", prefix_sep="", dtype=float)
        # stable column order, independent of which seasons happen to occur
        dummies = dummies.reindex(columns=list(f["season_map"]), fill_value=0.0)
        if f["drop_first_season"]:
            dummies = dummies.iloc[:, 1:]
        out = pd.concat([out, dummies], axis=1)
        names += list(dummies.columns)
    if f["add_month_onehot"]:
        md = pd.get_dummies(out.index.month, prefix="m", dtype=float)
        md.index = out.index
        md = md.reindex(columns=[f"m_{m}" for m in range(1, 13)], fill_value=0.0)
        out = pd.concat([out, md], axis=1)
        names += list(md.columns)
    k = int(f["doy_harmonics"])
    if k > 0:
        doy = out.index.dayofyear.to_numpy()
        period = 365.25
        for i in range(1, k + 1):
            out[f"sin_{i}"] = np.sin(2 * np.pi * i * doy / period)
            out[f"cos_{i}"] = np.cos(2 * np.pi * i * doy / period)
            names += [f"sin_{i}", f"cos_{i}"]
    return out, names


def engineer_features(df: pd.DataFrame, cfg,
                      lag_plan: dict | None = None
                      ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return ``(X, y, feature_names)`` after calendar and lag engineering."""
    d, f = cfg["data"], cfg["features"]
    target = d["target_column"]
    out = df.copy()
    if lag_plan is None:
        lag_plan = resolve_lag_plan(df, cfg)

    # --- lags of driver columns ---------------------------------------------
    # A chosen lag of 0 means "use the series as it is", so no `_lag0` copy is
    # created: the original column is kept under its own name. A driver that
    # *is* shifted has its unlagged source removed from the feature set when
    # features.auto_lags.drop_unlagged is true, so `prec1` is replaced by
    # `prec1_lag2` rather than both entering the model.
    generated: list[str] = []
    superseded: set[str] = set()
    for col, lags in lag_plan.get("drivers", {}).items():
        if col not in out.columns:
            raise KeyError(f"lag plan references missing column {col!r}")
        for lag in lags:
            if int(lag) == 0:
                generated.append(col)
                continue
            name = f"{col}_lag{lag}"
            if name not in out.columns:
                out[name] = out[col].shift(int(lag))
            generated.append(name)
        if lag_plan.get("drop_unlagged", True) and 0 not in [int(k) for k in lags]:
            superseded.add(col)

    # --- autoregressive lags of the target ----------------------------------
    for lag in lag_plan.get("target", []):
        name = f"{target}_lag{lag}"
        if lag <= 0:
            raise ValueError(f"target lag must be >= 1 (got {lag}): lag 0 would "
                             "feed the target to itself")
        if name not in out.columns:
            out[name] = out[target].shift(lag)
        generated.append(name)

    # --- base feature selection (before calendar dummies) -------------------
    # `superseded` holds driver columns that a chosen lag has replaced; they
    # stay in the frame (the lag columns were built from them) but are not
    # offered as predictors.
    reserved = {target}
    excluded = set(d["exclude_columns"]) | superseded
    candidates = [c for c in out.columns if c not in reserved | excluded]

    if d["feature_selection"] == "explicit":
        missing = [c for c in d["feature_columns"] if c not in out.columns]
        if missing:
            raise KeyError(f"data.feature_columns references missing column(s) {missing}")
        base = list(d["feature_columns"])
    elif d["feature_selection"] == "pattern":
        pats = [p.lower() for p in d["include_patterns"]]
        base = [c for c in candidates if any(p in str(c).lower() for p in pats)]
        base += [c for c in generated if c in candidates and c not in base]
        if not base:
            raise ValueError(
                f"no column matched data.include_patterns={d['include_patterns']}. "
                f"Candidates were {candidates}. Use feature_selection: all or "
                "list the columns explicitly."
            )
    else:  # all
        base = candidates

    # --- calendar features ---------------------------------------------------
    out, calendar = add_calendar_features(out, cfg)

    features = [c for c in base + calendar if c not in excluded]
    frame = out[features + [target]]
    if d["dropna"]:
        before = len(frame)
        frame = frame.dropna()
        dropped = before - len(frame)
    else:
        dropped = 0
    if frame.empty:
        raise ValueError("no rows left after feature engineering / dropna")

    X = frame[features].astype(float)
    y = frame[target].astype(float)
    X.attrs["rows_dropped_na"] = dropped
    X.attrs["lag_plan"] = lag_plan
    return X, y, features


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def make_splits(X: pd.DataFrame, y: pd.Series, cfg) -> dict[str, np.ndarray]:
    """Chronological train / validation / test positional index arrays.

    The validation block always sits between train and test in time, so no
    future information reaches model selection.
    """
    s = cfg["split"]
    n = len(X)
    pos = np.arange(n)

    if s["mode"] == "fraction":
        i_tr = int(round(s["train_fraction"] * n))
        i_va = i_tr + int(round(s["val_fraction"] * n))
    elif s["mode"] == "index":
        i_tr_total = int(s["train_end"])
        if not 0 < i_tr_total < n:
            raise ValueError(
                f"split.train_end={i_tr_total} is outside the usable row range "
                f"(0, {n}). Note that dropna during feature engineering removed "
                f"{X.attrs.get('rows_dropped_na', 0)} rows."
            )
        if s["val_end"] is not None:
            i_tr, i_va = i_tr_total, int(s["val_end"])
        else:
            # carve the tail of the training block off as validation
            i_va = i_tr_total
            i_tr = i_tr_total - int(round(s["val_fraction"] * i_tr_total))
    else:  # date
        idx = X.index
        i_tr_total = int((idx <= pd.Timestamp(s["train_end_date"])).sum())
        if i_tr_total in (0, n):
            raise ValueError(
                f"split.train_end_date={s['train_end_date']} puts every row on one "
                f"side of the split (data spans {idx.min().date()} to {idx.max().date()})"
            )
        if s["val_end_date"] is not None:
            i_tr = i_tr_total
            i_va = int((idx <= pd.Timestamp(s["val_end_date"])).sum())
        else:
            i_va = i_tr_total
            i_tr = i_tr_total - int(round(s["val_fraction"] * i_tr_total))

    if not 0 < i_tr <= i_va <= n:
        raise ValueError(f"degenerate split: train_end={i_tr}, val_end={i_va}, n={n}")
    if i_va == n:
        raise ValueError("the split leaves no test rows")

    return {"train": pos[:i_tr], "val": pos[i_tr:i_va], "test": pos[i_va:]}


# --------------------------------------------------------------------------- #
# scaling container
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """Everything a model needs, in both raw and scaled space.

    Scalers are fitted on the **training rows only**; validation and test rows
    are transformed with those parameters.
    """

    X: pd.DataFrame
    y: pd.Series
    splits: dict[str, np.ndarray]
    features: list[str]
    target: str
    x_scaler: object | None = None
    y_scaler: object | None = None
    Xs: np.ndarray = field(default=None, repr=False)
    ys: np.ndarray = field(default=None, repr=False)
    lag_plan: dict = field(default_factory=dict)

    # -- raw accessors ------------------------------------------------------
    def raw_X(self, part: str) -> pd.DataFrame:
        return self.X.iloc[self.splits[part]]

    def raw_y(self, part: str) -> pd.Series:
        return self.y.iloc[self.splits[part]]

    def index(self, part: str) -> pd.DatetimeIndex:
        return self.X.index[self.splits[part]]

    # -- scaled accessors ---------------------------------------------------
    def x(self, part: str) -> np.ndarray:
        return self.Xs[self.splits[part]]

    def y_scaled(self, part: str) -> np.ndarray:
        """1-D scaled target (sklearn regressors expect 1-D)."""
        return self.ys[self.splits[part]].ravel()

    def inverse_y(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        if self.y_scaler is None:
            return values.ravel()
        return self.y_scaler.inverse_transform(values).ravel()

    @property
    def n_features(self) -> int:
        return self.Xs.shape[1]

    def summary(self) -> dict:
        return {
            "rows": int(len(self.X)),
            "n_features": self.n_features,
            "features": list(self.features),
            "target": self.target,
            # the selection report is a DataFrame written separately to
            # diagnostics/lag_selection.csv; keep this summary JSON-safe
            "lag_plan": {k: v for k, v in (self.lag_plan or {}).items()
                         if k != "report"},
            "period": [str(self.X.index.min().date()), str(self.X.index.max().date())],
            "splits": {
                p: {
                    "n": int(len(i)),
                    "start": str(self.X.index[i[0]].date()) if len(i) else None,
                    "end": str(self.X.index[i[-1]].date()) if len(i) else None,
                }
                for p, i in self.splits.items()
            },
        }


def _pre_test_slice(raw: pd.DataFrame, cfg) -> pd.DataFrame:
    """Rows strictly before the test period, for leak-free lag selection.

    The boundary is computed on the *raw* frame. Feature engineering later
    drops the leading rows consumed by the shifts, which moves the real test
    boundary later, so this slice is always a subset of the eventual
    train+validation rows -- conservative by construction.
    """
    try:
        dummy = pd.DataFrame(index=raw.index)
        splits = make_splits(dummy, None, cfg)
        cut = int(splits["test"][0])
    except Exception:                                   # noqa: BLE001
        cut = int(0.8 * len(raw))
    return raw.iloc[:max(10, cut)]


def build_dataset(cfg, log=None) -> Dataset:
    """Config -> fully prepared :class:`Dataset` (load, engineer, split, scale)."""
    raw = load_table(cfg)
    needs_data = (cfg["features"]["auto_lags"]["enabled"]
                  and cfg["features"]["auto_lags"]["select"] == "cross_correlation"
                  ) or cfg["features"]["target_lags"] == "auto"
    lag_plan = resolve_lag_plan(_pre_test_slice(raw, cfg) if needs_data else raw,
                               cfg, log=log)
    X, y, features = engineer_features(raw, cfg, lag_plan)
    splits = make_splits(X, y, cfg)

    p = cfg["preprocess"]
    xs_cls = _SCALERS[p["scale_features"]]
    ys_cls = _SCALERS[p["scale_target"]]
    tr = splits["train"]

    if xs_cls is None:
        x_scaler, Xs = None, X.to_numpy(dtype=float)
    else:
        x_scaler = xs_cls()
        x_scaler.fit(X.iloc[tr].to_numpy(dtype=float))
        Xs = x_scaler.transform(X.to_numpy(dtype=float))

    if ys_cls is None:
        y_scaler, ys = None, y.to_numpy(dtype=float).reshape(-1, 1)
    else:
        y_scaler = ys_cls()
        y_scaler.fit(y.iloc[tr].to_numpy(dtype=float).reshape(-1, 1))
        ys = y_scaler.transform(y.to_numpy(dtype=float).reshape(-1, 1))

    return Dataset(
        X=X, y=y, splits=splits, features=features, target=cfg["data"]["target_column"],
        x_scaler=x_scaler, y_scaler=y_scaler, Xs=Xs, ys=ys, lag_plan=lag_plan,
    )
