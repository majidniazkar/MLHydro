#!/usr/bin/env python
"""Generate a synthetic future-climate scenario file in the expected layout.

    python make_future_scenarios.py --out data/future_scenarios.xlsx \
        --scenarios rcp26 rcp45 rcp85 --subbasins 3 --start 2026 --end 2060

Output columns (wide layout, the one `run_projection.py` expects by default):

    date | prec1_rcp26 | temp1_rcp26 | prec2_rcp26 | ... | prec3_rcp85 | temp3_rcp85

i.e. ``<driver>_<scenario>`` where the driver names match the ones the model
was trained on. Use ``--layout sheets`` to write one sheet per scenario with
the training column names instead, or ``--layout long`` for a single table with
a ``scenario`` column; `run_projection.py` reads all three.

**Synthetic data, not a climate projection.** Each scenario applies a warming
trend and a precipitation-intensity change to the same stochastic weather
generator used by `make_demo_data.py`, so the file exercises the projection
code and documents the format. Replace it with your own downscaled GCM output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: warming (degC by the end of the period) and precipitation change (fraction)
#: loosely mimicking the ordering of the named pathways - illustrative only.
PRESETS = {
    "rcp26":  {"warming": 1.0, "prec_change": +0.02},
    "rcp45":  {"warming": 2.0, "prec_change": -0.03},
    "rcp60":  {"warming": 2.8, "prec_change": -0.06},
    "rcp85":  {"warming": 4.2, "prec_change": -0.10},
    "ssp126": {"warming": 1.2, "prec_change": +0.03},
    "ssp245": {"warming": 2.4, "prec_change": -0.04},
    "ssp370": {"warming": 3.6, "prec_change": -0.08},
    "ssp585": {"warming": 4.8, "prec_change": -0.12},
}


def generate(scenario: str, n_sub: int, start: str, end: str, seed: int) -> pd.DataFrame:
    preset = PRESETS.get(scenario, {"warming": 2.0, "prec_change": 0.0})
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D")
    doy = idx.dayofyear.to_numpy()
    n = len(idx)
    ramp = np.linspace(0.0, 1.0, n)                 # linear trend over the period

    out = {"date": idx}
    for i in range(n_sub):
        offset = -2.5 * i                            # colder further upstream
        t_seasonal = 12.0 + offset - 10.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
        temp = t_seasonal + preset["warming"] * ramp + rng.normal(0, 3.0, n)
        wet = 0.30 + 0.05 * i + 0.16 * np.cos(2 * np.pi * (doy - 300) / 365.25)
        wet = np.clip(wet * (1 + preset["prec_change"] * ramp), 0.01, 0.99)
        prec = np.where(rng.random(n) < wet,
                        rng.gamma(shape=0.9, scale=9.0 + 2.0 * i, size=n), 0.0)
        out[f"prec{i + 1}"] = prec.round(2)
        out[f"temp{i + 1}"] = temp.round(2)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/future_scenarios.xlsx")
    ap.add_argument("--scenarios", nargs="+", default=["rcp26", "rcp45", "rcp85"])
    ap.add_argument("--subbasins", type=int, default=3)
    ap.add_argument("--start", default="2026")
    ap.add_argument("--end", default="2060")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--layout", choices=["wide", "sheets", "long"], default="wide")
    args = ap.parse_args()

    start, end = f"{args.start}-01-01", f"{args.end}-12-31"
    frames = {s: generate(s, args.subbasins, start, end, args.seed + i)
              for i, s in enumerate(args.scenarios)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.layout == "sheets":
        with pd.ExcelWriter(out, engine="openpyxl") as xls:
            for tag, df in frames.items():
                df.to_excel(xls, sheet_name=tag[:31], index=False)
        cols = list(next(iter(frames.values())).columns)
    elif args.layout == "long":
        stacked = pd.concat([df.assign(scenario=tag) for tag, df in frames.items()],
                            ignore_index=True)
        cols = list(stacked.columns)
        (stacked.to_excel(out, index=False) if out.suffix != ".csv"
         else stacked.to_csv(out, index=False))
    else:
        wide = None
        for tag, df in frames.items():
            renamed = df.rename(columns={c: f"{c}_{tag}" for c in df.columns
                                         if c != "date"})
            wide = renamed if wide is None else wide.merge(renamed, on="date")
        cols = list(wide.columns)
        (wide.to_excel(out, index=False) if out.suffix != ".csv"
         else wide.to_csv(out, index=False))

    print(f"wrote {out}  layout={args.layout}  scenarios={args.scenarios}")
    print(f"columns: {cols[:8]}{' ...' if len(cols) > 8 else ''}")


if __name__ == "__main__":
    main()
