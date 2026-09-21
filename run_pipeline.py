#!/usr/bin/env python
"""Single entry point for the hydroml template.

    python run_pipeline.py                          # uses ./config.yaml
    python run_pipeline.py --config basins/po.yaml
    python run_pipeline.py --models xgboost lightgbm --name quick_check
    python run_pipeline.py --no-tuning              # fit defaults only
    python run_pipeline.py --list-models

Nothing in ``src/hydroml`` needs to change to run a new case study: point
``data.path`` at the new table, set the column names and the split, and run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hydroml import REGISTRY, run  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml", help="path to the YAML config")
    ap.add_argument("--data", help="override data.path")
    ap.add_argument("--name", help="override run.name (the output sub-folder)")
    ap.add_argument("--models", nargs="+", help="override models.enabled")
    ap.add_argument("--no-tuning", action="store_true",
                    help="skip hyper-parameter search; fit library/config defaults")
    ap.add_argument("--selection-set", choices=["validation", "cv", "test"],
                    help="override tuning.selection_set")
    ap.add_argument("--seed", type=int, help="override run.seed")
    ap.add_argument("--list-models", action="store_true",
                    help="print the model registry and exit")
    args = ap.parse_args(argv)

    if args.list_models:
        width = max(len(k) for k in REGISTRY)
        for key, spec in REGISTRY.items():
            state = "available" if spec.available() else f"needs {', '.join(spec.missing())}"
            print(f"{key:<{width}}  {spec.label:<32} [{state}]")
            if spec.notes:
                print(f"{'':<{width}}  note: {spec.notes}")
        return 0

    overrides: dict = {}
    if args.data:
        overrides.setdefault("data", {})["path"] = args.data
    if args.name:
        overrides.setdefault("run", {})["name"] = args.name
    if args.seed is not None:
        overrides.setdefault("run", {})["seed"] = args.seed
    if args.models:
        overrides.setdefault("models", {})["enabled"] = args.models
    if args.no_tuning:
        overrides.setdefault("tuning", {})["enabled"] = False
    if args.selection_set:
        overrides.setdefault("tuning", {})["selection_set"] = args.selection_set

    if not Path(args.config).exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2

    result = run(args.config, **overrides)
    return 0 if not result.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
