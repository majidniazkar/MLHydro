"""hydroml -- a configuration-driven template for ML hydrological modelling.

The whole workflow is declared in ``config.yaml``; the code below never needs
editing to move from one river basin to another.

    from hydroml import run
    result = run("config.yaml")
"""

from .config import DEFAULTS, Config, load_config
from .data import (Dataset, build_dataset, engineer_features, load_table,
                   make_splits, resolve_lag_plan)
from .diagnostics import (autocorrelation_table, cross_correlation_table,
                          permutation_importance, seasonal_table,
                          seasonality_strength, select_best_lags, suggest_lags,
                          suggest_target_lags)
from .evaluation import cross_validate, make_folds
from .export import prepared_table, write_prepared_input
from .metrics import METRICS, evaluate
from .models import REGISTRY, ModelSpec, register, resolve_models
from .pipeline import RunResult, run
from .projection import (ProjectionInputs, load_run, project, read_scenarios,
                         summarise)

__version__ = "1.3.0"

__all__ = [
    "run", "RunResult", "load_config", "Config", "DEFAULTS",
    "build_dataset", "Dataset", "load_table", "engineer_features", "make_splits",
    "resolve_lag_plan",
    "autocorrelation_table", "cross_correlation_table", "seasonal_table",
    "seasonality_strength", "select_best_lags", "suggest_lags",
    "suggest_target_lags", "prepared_table", "write_prepared_input",
    "permutation_importance",
    "cross_validate", "make_folds",
    "ProjectionInputs", "load_run", "project", "read_scenarios", "summarise",
    "evaluate", "METRICS",
    "REGISTRY", "ModelSpec", "register", "resolve_models",
    "__version__",
]
