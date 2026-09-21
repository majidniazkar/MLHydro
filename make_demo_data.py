#!/usr/bin/env python
"""Generate a synthetic daily climate/discharge table in the template's layout.

    python make_demo_data.py --out data/demo_data.xlsx --years 12

The output is **not** real data. It exists so the template can be smoke-tested
end to end, and so the expected input layout is unambiguous:

    date | precip_lag0..3 | tmean_lag0..2 | flow_lag1..3 | flow

A conceptual linear-reservoir rainfall-runoff model with snow storage and
seasonal evapotranspiration generates the discharge, which gives the ML models
a genuinely learnable but non-trivial signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def simulate(years: int = 12, seed: int = 7, start: str = "2004-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(365.25 * years), freq="D")
    doy = idx.dayofyear.to_numpy()

    # seasonal climate
    t_mean = 12.0 - 10.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    tmean = t_mean + rng.normal(0, 3.0, len(idx))
    wet = 0.30 + 0.16 * np.cos(2 * np.pi * (doy - 300) / 365.25)
    rains = rng.random(len(idx)) < wet
    precip = np.where(rains, rng.gamma(shape=0.9, scale=9.0, size=len(idx)), 0.0)

    # conceptual rainfall-runoff with snow store and two linear reservoirs
    snow = fast = slow = 0.0
    q = np.empty(len(idx))
    for i in range(len(idx)):
        snowfall = precip[i] if tmean[i] < 1.0 else 0.0
        rain = precip[i] - snowfall
        snow += snowfall
        melt = min(snow, max(0.0, 2.6 * (tmean[i] - 1.0)))
        snow -= melt
        inflow = rain + melt
        pet = max(0.0, 0.14 * tmean[i])
        eff = max(0.0, inflow - 0.45 * pet)
        fast = 0.63 * fast + 0.55 * eff
        slow = 0.975 * slow + 0.20 * eff
        q[i] = 0.55 * fast + 0.42 * slow + 1.4

    q *= np.exp(rng.normal(0, 0.055, len(idx)))          # multiplicative noise
    df = pd.DataFrame({"date": idx, "precip": precip.round(2),
                       "tmean": tmean.round(2), "flow": q.round(3)})

    # lagged predictors, matching the original notebook's naming convention
    for lag in range(0, 4):
        df[f"precip_lag{lag}"] = df["precip"].shift(lag)
    for lag in range(0, 3):
        df[f"tmean_lag{lag}"] = df["tmean"].shift(lag)
    for lag in range(1, 4):
        df[f"flow_lag{lag}"] = df["flow"].shift(lag)

    # Raw drivers are kept alongside the lag columns so that both feature
    # strategies can be demonstrated: `include_patterns: ["lag"]` picks up the
    # pre-built lags, while `features.auto_lags` builds its own from `precip`
    # and `tmean`.
    cols = (["date", "precip", "tmean"]
            + [c for c in df.columns if "lag" in c]
            + ["flow"])
    return df[cols].dropna().reset_index(drop=True)


def simulate_subbasins(years: int = 12, seed: int = 11, start: str = "2004-01-01",
                       travel_lags: tuple[int, ...] = (1, 3, 6),
                       areas: tuple[float, ...] = (0.5, 0.3, 0.2)) -> pd.DataFrame:
    """Sub-basin layout: ``date | prec1 temp1 prec2 temp2 ... | streamflow``.

    Each sub-basin generates runoff from its own precipitation and temperature,
    which then takes ``travel_lags[i]`` timesteps to reach the basin outlet.
    Only the outlet discharge is observed -- the usual situation, where no gauge
    exists at the sub-basin interfaces.

    The imposed travel times are ground truth for the lag-selection step: a
    correct implementation should recover lags close to ``travel_lags`` from the
    outlet series alone. **Synthetic data, not a real basin.**
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(365.25 * years), freq="D")
    doy = idx.dayofyear.to_numpy()
    n = len(idx)
    n_sub = len(travel_lags)

    cols: dict[str, np.ndarray] = {}
    routed = np.zeros(n)
    for i in range(n_sub):
        # sub-basins differ in elevation: colder and wetter further upstream
        offset = -2.5 * i
        t_seasonal = 12.0 + offset - 10.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
        temp = t_seasonal + rng.normal(0, 3.0, n)
        wet = 0.30 + 0.05 * i + 0.16 * np.cos(2 * np.pi * (doy - 300) / 365.25)
        prec = np.where(rng.random(n) < wet,
                        rng.gamma(shape=0.9, scale=9.0 + 2.0 * i, size=n), 0.0)

        snow = fast = slow = 0.0
        runoff = np.empty(n)
        for t in range(n):
            snowfall = prec[t] if temp[t] < 1.0 else 0.0
            rain = prec[t] - snowfall
            snow += snowfall
            melt = min(snow, max(0.0, 2.6 * (temp[t] - 1.0)))
            snow -= melt
            pet = max(0.0, 0.14 * temp[t])
            eff = max(0.0, rain + melt - 0.45 * pet)
            fast = 0.63 * fast + 0.55 * eff
            slow = 0.975 * slow + 0.20 * eff
            runoff[t] = 0.55 * fast + 0.42 * slow

        # channel routing: this sub-basin's contribution arrives `lag` days later
        contribution = np.concatenate([np.zeros(travel_lags[i]),
                                       runoff[: n - travel_lags[i]]])
        routed += areas[i] * contribution
        cols[f"prec{i + 1}"] = prec.round(2)
        cols[f"temp{i + 1}"] = temp.round(2)

    q = (routed + 1.4) * np.exp(rng.normal(0, 0.05, n))
    df = pd.DataFrame({"date": idx, **cols, "streamflow": q.round(3)})
    # drop the spin-up period, where the reservoirs are still filling
    return df.iloc[180:].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/demo_data.xlsx")
    ap.add_argument("--years", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layout", choices=["lagged", "subbasins"], default="lagged",
                    help="lagged: pre-built *_lag* columns (original notebook's "
                         "layout). subbasins: raw prec<i>/temp<i> per sub-basin "
                         "plus outlet streamflow, with known travel times.")
    ap.add_argument("--travel-lags", type=int, nargs="+", default=[1, 3, 6],
                    help="routing delay per sub-basin (subbasins layout)")
    args = ap.parse_args()

    if args.layout == "subbasins":
        df = simulate_subbasins(years=args.years, seed=args.seed,
                                travel_lags=tuple(args.travel_lags),
                                areas=tuple([1 / len(args.travel_lags)] * len(args.travel_lags)))
        print(f"imposed travel times: "
              + ", ".join(f"sub-basin {i + 1} -> {k} day(s)"
                          for i, k in enumerate(args.travel_lags)))
    else:
        df = simulate(years=args.years, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_excel(out, index=False)
    print(f"wrote {out}  rows={len(df)}  columns={list(df.columns)}")


if __name__ == "__main__":
    main()
