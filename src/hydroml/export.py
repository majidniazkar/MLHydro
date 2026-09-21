"""Export the prepared model input table.

The table the models actually see is not the file on disk: lags have been
applied, calendar predictors added, rows with missing values dropped, and the
record cut chronologically into train / validation / test. This module writes
that table out so it can be inspected, shared, or fed to another tool.

Values are in **original units** -- the scaling applied inside the pipeline is
fitted on the training block and inverted before any metric is computed, so it
is an implementation detail, not part of the dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["prepared_table", "write_prepared_input"]

SPLIT_ORDER = ("train", "val", "test")
SHEET_NAMES = {"train": "train", "val": "validation", "test": "test"}


def prepared_table(ds, cfg) -> pd.DataFrame:
    """The engineered feature matrix + target + split label, one row per date."""
    label = pd.Series(index=ds.X.index, dtype=object)
    for part in SPLIT_ORDER:
        pos = ds.splits.get(part)
        if pos is not None and len(pos):
            label.iloc[pos] = part
    out = ds.X.copy()
    out.insert(0, "split", label.to_numpy())
    out[ds.target] = ds.y.to_numpy()          # target last, as in the input file
    out.index.name = cfg["data"]["date_column"]
    return out.reset_index()


def write_prepared_input(run_dir: Path, ds, cfg, log=print) -> list[Path]:
    """Write the prepared table, split into one sheet/file per period.

    Excel output carries one sheet per non-empty split plus ``full`` (all rows
    with the ``split`` label) and, when lag selection ran, a ``lag_selection``
    sheet recording which lag was chosen for each driver and why.
    """
    e = cfg["export"]
    if not e["enabled"]:
        return []

    table = prepared_table(ds, cfg)
    stem = Path(e["filename"]).stem
    written: list[Path] = []
    parts = {p: table[table["split"] == p].drop(columns="split")
             for p in SPLIT_ORDER if (table["split"] == p).any()}
    report = (ds.lag_plan or {}).get("report")

    if "xlsx" in e["formats"]:
        path = run_dir / f"{stem}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as xls:
            for part, frame in parts.items():
                frame.to_excel(xls, sheet_name=SHEET_NAMES[part], index=False)
            if e["include_full_sheet"]:
                table.to_excel(xls, sheet_name="full", index=False)
            if report is not None and not report.empty:
                report.to_excel(xls, sheet_name="lag_selection", index=False)
            _summary_frame(parts, ds, cfg).to_excel(xls, sheet_name="summary", index=False)
        written.append(path)

    if "csv" in e["formats"]:
        for part, frame in parts.items():
            path = run_dir / f"{stem}_{SHEET_NAMES[part]}.csv"
            frame.to_csv(path, index=False)
            written.append(path)
        if e["include_full_sheet"]:
            path = run_dir / f"{stem}_full.csv"
            table.to_csv(path, index=False)
            written.append(path)

    shares = ", ".join(f"{SHEET_NAMES[p]} {len(f)} ({100 * len(f) / len(table):.0f}%)"
                       for p, f in parts.items())
    for path in written:
        log(f"    prepared input -> {path.name}")
    log(f"    rows: {shares}")
    return written


def _summary_frame(parts: dict, ds, cfg) -> pd.DataFrame:
    rows = []
    total = sum(len(f) for f in parts.values())
    for part, frame in parts.items():
        date_col = cfg["data"]["date_column"]
        rows.append({
            "period": SHEET_NAMES[part],
            "rows": len(frame),
            "share_%": round(100 * len(frame) / total, 2) if total else 0.0,
            "start": str(frame[date_col].iloc[0].date()) if len(frame) else "",
            "end": str(frame[date_col].iloc[-1].date()) if len(frame) else "",
        })
    rows.append({"period": "TOTAL", "rows": total, "share_%": 100.0,
                 "start": "", "end": ""})
    meta = pd.DataFrame(rows)
    meta.attrs["n_features"] = len(ds.features)
    return meta
