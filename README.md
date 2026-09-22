# hydroml — a configuration-driven template for ML hydrological modelling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22895684.svg)](https://doi.org/10.5281/zenodo.22895684)

A template for streamflow simulation and climate-scenario projection where
**the code never changes between case studies**: to model a new river basin
you edit `config.yaml` and run one command.

```
pip install -r requirements.txt

# check the install on synthetic data (3 upstream sub-basins, imposed 1/3/6-day delays)
python make_demo_data.py --layout subbasins --out data/subbasin_data.xlsx --years 12
python run_pipeline.py --config config_subbasins.yaml

# your basin: edit config.yaml, then
python run_pipeline.py --config config.yaml
```

---

## Installation

```
pip install -r requirements.txt          # core + xgboost/lightgbm/catboost
pip install tensorflow                   # only if you enable ann / lstm
```

`run.n_jobs: -1` uses all cores. Some locked-down containers forbid the worker
pools joblib needs; the pipeline detects that and retries sequentially, but
setting `n_jobs: 1` avoids the wasted first attempt.


## 1. Layout

```
<repository root>
├── config.yaml              <-- the only file you edit for a new basin
├── config_demo.yaml             smoke test: pre-lagged input layout
├── config_subbasins.yaml        sub-basin input layout + travel-time selection
├── run_pipeline.py              CLI entry point (training + evaluation)
├── run_projection.py            CLI entry point (future scenarios)
├── make_future_scenarios.py     synthetic scenario file, documents the layout
├── make_demo_data.py            synthetic input, documents the expected layout
├── requirements.txt
├── data/                        put your input table here
├── notebooks/
│   └── 01_run_template.ipynb    thin notebook driver (same pipeline)
├── results/<run_name>/          everything a run produces
└── src/hydroml/
    ├── config.py                defaults + validation of config.yaml
    ├── data.py                  load, lag/calendar features, split, scaling
    ├── models.py                MODEL REGISTRY (add a model here)
    ├── deep.py                  ANN / LSTM wrappers, backend selection
    ├── deep_torch.py            PyTorch implementations of both
    ├── fitting.py               build / fit / back-transform predictions
    ├── tuning.py                hyper-parameter search + local refinement
    ├── diagnostics.py           ACF/PACF, lag selection, seasonality, importance
    ├── evaluation.py            time-series cross-validation
    ├── export.py                prepared input table (train/test sheets)
    ├── projection.py            future-scenario projection (direct + recursive)
    ├── metrics.py               RMSE, MAE, NSE, KGE, PBIAS, R2, MAPE
    ├── reporting.py             tables + figures
    └── pipeline.py              orchestration
```

The flow is **data → model → results**, with configuration as a fourth,
separate layer:

```
config.yaml ──► data.build_dataset ──► tuning.search ──► fitting.fit_estimator
                                                              │
                                   reporting.write_tables ◄────┘
                                   reporting.write_figures
```

## 2. Input data

Two layouts are supported, and both are demonstrated by `make_demo_data.py`.

**(a) Sub-basin layout** — raw per-sub-basin climate plus one gauge at the
outlet. This is the common field situation: no discharge is measured at the
sub-basin interfaces.

| date | prec1 | temp1 | prec2 | temp2 | prec3 | temp3 | streamflow |
|---|---|---|---|---|---|---|---|
| 2004-06-29 | 0.00 | 18.4 | 5.21 | 15.9 | 0.00 | 13.2 | 6.12 |

`prec<i>` / `temp<i>` belong to upstream sub-basin *i*; `streamflow` is the
target at the outlet. Runoff from a distant sub-basin arrives at the outlet
some timesteps later, so the pipeline **tests each driver at every lag from 0
to `max_lag`, keeps the best, and replaces the column with the shifted copy**
(section 4). Config: `config_subbasins.yaml`.

**(b) Pre-lagged layout** — the table already contains the shifted columns:

| date | precip_lag0 | precip_lag1 | tmean_lag0 | flow_lag1 | flow |
|---|---|---|---|---|---|
| 2004-01-04 | 0.0 | 12.4 | 2.1 | 5.31 | 5.02 |

Here `data.feature_selection: pattern` with `include_patterns: ["lag"]` picks
the predictors up as they are. Config: `config_demo.yaml`.

## 3. Column roles

**Nothing is inferred from column position, count, or content.** Exactly two
columns are named by you — `data.date_column` and `data.target_column` — and
both are checked before anything else happens; a missing one raises with the
file's actual column list. The date column becomes the DatetimeIndex, so it can
never be picked up as a predictor, and the target is always removed from the
candidate pool. The rest are predictors, chosen one of three ways:

| `data.feature_selection` | behaviour |
|---|---|
| `pattern`  | every column whose name contains any of `include_patterns` (default `["lag"]`) |
| `explicit` | exactly the columns listed in `feature_columns` |
| `all`      | everything except the target and `exclude_columns` |

`exclude_columns` is applied last as a veto — that is how a raw driver stays in
the file (for lag building) while being kept out of the feature matrix.
Whatever the rule resolves to is printed and stored in `data_summary.json`:

```
[data] features: prec1_lag1, temp1, prec2_lag3, temp2, prec3_lag6, temp3,
       streamflow_lag1, streamflow_lag2, streamflow_lag4, sin_1, cos_1, sin_2, cos_2
```

Read that line first on a new basin. It is the only unambiguous place a
column-naming mistake shows up — particularly in `pattern` mode, where a
predictor whose name lacks the pattern is dropped silently. If nothing matches
at all, the run stops and lists the candidate columns.

Calendar predictors are optional and configurable: season one-hot (with an
editable `season_map`, so southern-hemisphere basins are a config change),
month one-hot, and day-of-year sine/cosine harmonics.

Units and physical meaning are never inferred. Two consequences:
`preprocess.clip_negative_predictions: true` assumes a non-negative target
(right for discharge, wrong for an anomaly or a level change), and `mape` is
computed on non-zero observations only.

## 4. Travel-time (lag) selection

For the sub-basin layout the question *"is a lagged copy of this driver better
than the raw series?"* is answered per driver, from the data:

```yaml
features:
  auto_lags:
    enabled: true
    select: cross_correlation
    columns: []            # [] = every non-target column
    max_lag: 15
    top_k: 1               # keep the single best lag per driver
    min_abs_corr: 0.05
    drop_unlagged: true    # prec1 -> prec1_lag2, not both
```

Each driver is correlated with the target at every lag from 0 to `max_lag` and
the strongest `top_k` are kept. Three outcomes:

| outcome | what happens | logged as |
|---|---|---|
| best lag ≥ 1 | a column `prec1_lag2` is built by shifting; the unlagged source is dropped when `drop_unlagged` | `lagged by 2 (gain +0.213 over lag 0)` |
| best lag = 0 | the column is used **as it is**, under its original name — no `_lag0` copy is made | `used as-is (lag 0 is best)` |
| best correlation below `min_abs_corr` | the driver contributes no column | `dropped (best \|corr\|=0.03 < min_abs_corr=0.05)` |

The decision, the correlation at the chosen lag, the correlation at lag 0 and
the gain between them go to `diagnostics/lag_selection.csv` and to a
`lag_selection` sheet in the exported input table, so the feature set is
auditable rather than implicit. Chosen lags are starred on
`lag_correlation.png`.

Selection uses rows **before the test period only**. The boundary is computed
on the raw table, before `dropna` moves the real split later, so the slice is
always a subset of train+validation. `target_lags: auto` picks the target's own
autoregressive lags by |PACF| the same way.

A ground-truth check ships with the template:
`make_demo_data.py --layout subbasins` routes three sub-basins to the outlet
with imposed delays of 1, 3 and 6 timesteps, observable only at the outlet.

## 5. Exported model input table

The table the models see is not the file on disk — lags applied, calendar
predictors added, incomplete rows dropped — so every run writes it out:

```yaml
export:
  enabled: true
  filename: prepared_input_data.xlsx
  formats: [xlsx, csv]
  include_full_sheet: true
```

`results/<run>/prepared_input_data.xlsx` holds one sheet per period (`train`,
`validation` when one exists, `test`), a `full` sheet with every row and its
`split` label, a `summary` sheet of row counts/shares/date ranges, and the
`lag_selection` sheet. Values are in original units — the internal scaling is
fitted on the training block and inverted before any metric, so it is not part
of the dataset. With `formats: [csv]` the same content is written as
`prepared_input_data_train.csv` and so on.

For a strict two-way split, set `split.train_fraction: 0.75` with
`val_fraction: 0.0`; you then get exactly `train` and `test` sheets, and
`tuning.selection_set` must be `cv` (the config check says so explicitly,
because there is no validation block left to select on).

## 6. Splitting

Always chronological, always three blocks in time order:

```
|<---------- train ---------->|<-- validation -->|<---- test ---->|
         fit parameters          select models       report skill
```

```yaml
split:
  mode: fraction        # train_fraction / val_fraction
  mode: index           # train_end (row position), optional val_end
  mode: date            # train_end_date, optional val_end_date
```

In `index`/`date` mode with no explicit validation boundary, the last
`val_fraction` of the training block becomes the validation block.

## 7. Models

Enable by name; missing optional packages are skipped with a note rather than
crashing the run.

```
linear_regression  bayesian_ridge  huber  sgd  knn  svr
decision_tree  random_forest  adaboost  gradient_boosting
hist_gradient_boosting  xgboost  lightgbm  catboost
mlp                       scikit-learn dense network, no extra install
ann  lstm                 needs torch or tensorflow (see section 13)
```

`python run_pipeline.py --list-models` prints the registry and what is
installed.

Pin a parameter (it then leaves the search space), or replace a search space:

```yaml
models:
  params:
    random_forest: {criterion: absolute_error}
  search_spaces:
    xgboost:
      n_estimators: [200, 400, 800]
      max_depth: [4, 6, 8]
```

**Adding a new model** is the one code edit the template expects, and it is a
single `register(...)` call in `src/hydroml/models.py`:

```python
def _theil_sen(seed=None, **p):
    from sklearn.linear_model import TheilSenRegressor
    return TheilSenRegressor(random_state=seed, **p)

register(ModelSpec("theil_sen", "Theil-Sen", _theil_sen,
                   {"max_subpopulation": [1000, 5000]}))
```

## 8. Tuning

```yaml
tuning:
  strategy: random        # random (n_iter draws) | grid | none
  n_iter: 30
  refine_rounds: 1        # coordinate sweeps after the random draws
  selection_metric: rmse
  selection_set: validation   # validation | cv | test
```

* `validation` — fit on train, score on the later validation block.
* `cv` — `TimeSeriesSplit` (expanding window) inside the training block;
  `cv_splits` times more expensive, less sensitive to one unusual period.
* `test` — **selects on the test block.** Kept only so numbers produced under
  that older protocol can be regenerated. It selects hyper-parameters
  on the same rows used to report skill, so the reported test scores are
  optimistically biased. The pipeline prints a warning when it is active.

`refit_on: train_val` refits the selected configuration on train + validation
before the final test evaluation (a common choice once selection is done).

**Search coverage.** A random draw of `n_iter: 30` from a 2 880-point grid
samples 1% of it, and the odds that all six parameters land near their best
value together are poor. `refine_rounds: 1` follows the draws with a
coordinate sweep: hold the best configuration fixed, vary one parameter across
all its values, keep any improvement, repeat. That costs the *sum* of the
per-parameter option counts instead of their *product* — typically 15–40 extra
fits rather than thousands — and it is what recovers most of the gap to an
exhaustive grid. Set `refine_rounds: 0` to disable, or `strategy: grid` for the
full product (guarded by `max_grid_size`). Every candidate tried, including the
refinement stage it came from, is written to `tuning/<model>_trials.csv`.

Measured on the shipped `results/example_run` (validation RMSE, best of the
random draws vs best after one refinement round):

| model | search best | after refinement | gain |
|---|---|---|---|
| xgboost | 1.1024 | 0.9521 | 0.150 |
| decision_tree | 1.3448 | 1.2122 | 0.133 |
| lightgbm | 0.9486 | 0.9253 | 0.023 |
| hist_gradient_boosting | 1.0006 | 0.9865 | 0.014 |
| knn | 1.6746 | 1.6676 | 0.007 |
| huber | 1.2145 | 1.2103 | 0.004 |
| ann | 1.2138 | 1.2129 | 0.001 |
| lstm, mlp, random_forest | — | no improvement | 0 |

The sweep helps most where the space is large and the model is sensitive to a
single parameter (tree depth, learning rate), and does nothing when the random
draw already found the best neighbourhood. It can never *degrade* the outcome:
the best configuration is tracked across all stages, so a refinement candidate
that scores worse is recorded in the trials file and discarded.

## 9. Outputs

```
results/<run_name>/
├── config_used.yaml       exact configuration incl. all defaults — reproducibility
├── data_summary.json      rows, feature list, split dates
├── run_log.txt            full console transcript
├── metrics_long.csv       tidy: model, split, metric, value
├── metrics_wide.csv
├── metrics.xlsx           sheets: train, val, test, leaderboard, best_params
├── predictions.csv        date, split, observed, one column per model
├── best_params.json
├── tuning/<model>_trials.csv
├── models/<model>.joblib | .keras | .pt   (removed from the shipped
│                                          example runs to keep the zip small)
├── diagnostics/
│   ├── target_autocorrelation.csv     ACF + PACF with white-noise band
│   ├── driver_cross_correlation.csv   each driver vs target at lags 0..max_lag
│   ├── seasonality.csv                monthly/seasonal target statistics
│   └── importance_<model>.csv         permutation importance
├── metrics_cv_folds.csv     ) when evaluation.cv.enabled
├── metrics_cv_summary.csv   )
├── metrics_cv.xlsx          )
└── figures/
    hydrograph.png  scatter.png  metric_bars.png  residuals.png
    lag_correlation.png  seasonality.png  importance.png  cv_metrics.png
```

Metrics: `rmse`, `mae`, `nse`, `kge` (Gupta et al., 2009), `pbias`, `r2`,
`mape`, `bias`. Note that the coefficient of determination `1 - SSE/SST` is
identical to NSE, so it is reported once as `nse`; `r2` here is the **squared
Pearson correlation**, which is blind to bias and variance error — report it
alongside NSE/KGE, not instead of them.

## 10. Reproducibility

`run.seed` seeds NumPy, every estimator that accepts a random state, and Keras.
Each run writes `config_used.yaml`, so a result folder is self-describing:
same config + same input file + same package versions reproduces the numbers.

## 11. Diagnosing poor performance

Before reaching for another model, check the three things that dominate skill
in a lagged-regression setup. All of them are now reported automatically in
`results/<run>/diagnostics/`.

**1. Is the basin's memory in the feature set?**
`target_autocorrelation.csv` and the left panel of `lag_correlation.png` show
how far back the target and each driver carry information. If the PACF is
significant out to lag 5 but your table only supplies `flow_lag1`, no model can
recover the rest. Fix it by listing the lags explicitly, or let the data choose:

```yaml
features:
  auto_lags:
    enabled: true
    select: cross_correlation   # peak |corr| per driver, training rows only
    columns: [precipitation, temperature]
    max_lag: 30
    top_k: 3
    min_abs_corr: 0.1
  target_lags: auto             # by |PACF|
```

Selection uses **only rows before the test period**, so it does not leak.

**2. Are the predictors actually used?**
`importance_<model>.csv` and `importance.png` give permutation importance in
the ranking metric's own units: 0.12 means shuffling that column costs 0.12
NSE. Columns sitting at zero are dead weight — and a feature set where the only
non-zero bar is `flow_lag1` means the model is a persistence forecast wearing a
machine-learning hat, which looks excellent in NSE and has no predictive value.

**3. Does the calendar carry signal?**
`seasonality.csv` and the printed *month-of-year climatology explains X% of
variance* quantify what the season dummies, month dummies and day-of-year
harmonics can contribute. If X is small, `features.doy_harmonics` will not
rescue a weak run; if it is large, the cyclical encodings
(`doy_harmonics: 2`, which is smooth and costs 4 columns) usually beat the
four-level season dummy.

Two further causes are worth ruling out, both of which make honest numbers look
*worse* than an earlier optimistic setup rather than indicating a real problem:

* `tuning.selection_set: validation` (the default) chooses hyper-parameters
  without touching the test block. Scores obtained with `selection_set: test`
  were selected *on* the test block and are therefore biased upward.
  The comparison is not like-for-like; the validation-selected number is the
  defensible one.
* `tuning.refit_on: train` fits the final model on the training block only,
  leaving the validation years unused. Set `refit_on: train_val` to refit the
  chosen configuration on train + validation before testing — the usual choice
  once hyper-parameters are fixed, and it recovers the sample size the original
  single-split run had.

## 12. Cross-validation

```yaml
evaluation:
  cv:
    enabled: true
    scheme: expanding      # TimeSeriesSplit; or 'blocked'
    n_splits: 5
    gap: 0                 # rows dropped either side of the test block
    over: all              # all | train_val
    use_best_params: true  # false -> nested search inside each fold
```

Outputs `metrics_cv_folds.csv` (one row per model per fold),
`metrics_cv_summary.csv` (mean and standard deviation), `metrics_cv.xlsx` and
`cv_metrics.png`. Use it to answer "is model A really better than model B?" —
if the fold-to-fold spread exceeds the gap between their means, they are
indistinguishable on this record, whatever the single-split table says.

`use_best_params: true` reuses the hyper-parameters chosen on the holdout
validation block in every fold: cheap, mildly optimistic. `false` repeats the
search inside each fold (nested CV) — statistically clean and `n_splits` times
the cost.

Note the key is `over`, not `on`: YAML 1.1 parses a bare `on:` as the boolean
`true`, which would silently produce a nonsense key.

## 13. Neural networks

Three options, in increasing order of setup cost:

| key | framework | notes |
|---|---|---|
| `mlp` | scikit-learn | No extra install. Good first check on whether a dense network helps at all. |
| `ann` | torch **or** tensorflow | Dense network with early stopping on the validation block. |
| `lstm` | torch **or** tensorflow | Recurrent; `sequence_mode` decides what "time" means. |

```
pip install torch          # or: pip install tensorflow
```

`deep.backend: auto` uses whichever is installed. The LSTM's `sequence_mode`
matters more than any hyper-parameter:

* `sliding_window` (**default**) builds real sequences of `window` consecutive
  timesteps, shape `(n, window, n_features)`. Padding repeats the first row
  backwards, so sample *t* only ever sees rows ≤ *t*.
* `features_as_timesteps` reshapes to `(n, n_features, 1)`, i.e. the columns of
  the table are treated as the time axis. This reproduces the original
  notebook. It is worth knowing that this axis is not time at all — it
  interleaves precipitation lags, temperature lags and season dummies in
  column order — so the recurrent structure has nothing meaningful to
  integrate over.

Neural runs are the slow part of a comparison: budget roughly a second per
epoch per fit on CPU, multiplied by the number of tuning candidates and CV
folds. Start with `tuning.enabled: false` to get a baseline, then search.

## 14. Future projections

A trained run can be pushed forward onto climate-scenario input:

```
python run_projection.py --run results/subbasins --future data/future_scenarios.xlsx
```

`run_projection.py` rebuilds the run's preprocessing from `config_used.yaml`,
loads the serialised estimators from `results/<run>/models/`, and applies the
**same** chain to the scenario table. Three things are deliberately reused
rather than recomputed:

* the **fitted scalers** — refitting them on the future period would rescale a
  warmer, wetter climate back onto the training range and erase the signal the
  projection exists to show;
* the **lag plan** — `prec1` is shifted by the travel time chosen during
  training, not re-selected (there is no future target to select against);
* the **feature order** — the matrix is reindexed to the training feature list,
  and a feature that cannot be built is an error, never a silent zero.

### Scenario file layout

`i` is the scenario tag (`rcp26`, `rcp45`, `rcp85`, `ssp126`, `ssp370`, ...).
Three layouts are recognised automatically. All three were run on the same
two-scenario data (`ssp126`, `ssp585`, 2026-2030 daily) and the wide layout's
projections were diffed against both the sheet-per-scenario and the long
layout: 3 652 values each, maximum absolute difference exactly 0.

**(a) wide — `<driver>_<scenario>`** (the default, and what
`make_future_scenarios.py` writes):

| date | prec1_rcp45 | temp1_rcp45 | prec2_rcp45 | ... | prec1_ssp370 | ... |
|---|---|---|---|---|---|---|
| 2026-01-01 | 0.00 | 3.7 | 4.21 | | 1.02 | |

**(b) one sheet per scenario**, each sheet using the training column names
(`date | prec1 | temp1 | ...`), the sheet name being the scenario tag.

**(c) long** — one table with a `scenario` column.

The driver names must match the ones the model was trained on. The required
list is printed at the start of the run:

```
[model] scenario drivers required: ['prec1', 'prec2', 'prec3', 'temp1', 'temp2', 'temp3']
```

Note this is the list of **raw** drivers, not the lagged feature names — the
shifting is done for you. A scenario missing any of them is rejected with the
file's actual column list, rather than modelled with a gap.

Generate an example file to copy the format from:

```
python make_future_scenarios.py --out data/future_scenarios.xlsx \
    --scenarios rcp26 rcp45 rcp85 ssp370 --subbasins 3 --start 2026 --end 2060
```

### Two projection modes, and why it matters which one you are in

The log states which is active:

```
[mode] recursive: the model uses streamflow_lag[1, 2, 4], which do not exist
       in the future. Each step is fed its own previous prediction ...
[mode] direct: the model uses climate predictors only, so every step is
       independent of the projected ones.
```

A model trained with `features.target_lags` has no observed discharge to read
in the future, so the projection is run **recursively**: each step's prediction
becomes the next step's lag input, seeded from the last observed flows. This is
the standard approach and it is the only way such a model can run at all, but
errors compound and the result is a scenario *simulation*, not a forecast.

**For climate projection, prefer a model trained without `target_lags`.** The
reason is visible in the verification runs below: the autoregressive model has
much higher historical skill, but its projected scenario differences are
smaller and inconsistent between models, because most of its prediction comes
from its own previous output rather than from the climate input you changed.
The climate-only model has lower NSE and a cleaner scenario response.

Runtime follows from the mode: direct projection is vectorised (~45 s for
4 scenarios x 2 models x 35 years daily), recursive is one model call per
timestep (~8 min for 4 scenarios x 3 models over the same period). Project
with fewer models, or a shorter horizon, if that matters.

### Outputs

```
projections/<run_name>/
├── projections.xlsx        one sheet per scenario (date, one column per model,
│                           ensemble_mean) + a summary sheet
├── projections_long.csv    tidy: scenario, model, date, projection
├── summary.csv             mean, median, q05, q95, change_% vs the observed
│                           baseline, annual trend per decade
├── projection_log.txt
└── figures/  projection_timeseries.png  projection_annual.png
           projection_change.png
```

`--models a b c` restricts the model set, `--refit` refits on the historical
data with the run's selected hyper-parameters instead of loading the saved
files (use it when the `models/` folder is absent, as in the example runs
shipped here), and `--historical` points at the training table if it has moved.

## 15. Using it as a library

```python
import sys; sys.path.insert(0, "src")
from hydroml import run, load_config

result = run("config.yaml", run={"name": "po_basin"},
             models={"enabled": ["xgboost", "lightgbm"]})

print(result.leaderboard)
result.predictions.head()
result.estimators["xgboost"]
```

---

## 16. Licence, citation and reuse

Released under the MIT Licence — see `LICENSE`. You may use, modify and
redistribute it, including commercially, provided the copyright notice is kept.

If it contributes to a publication, please cite it:

Niazkar M. (2026). hydroml: a configuration-driven template for machine-learning hydrological modelling (Version 1.3.0) [software]. Zenodo. https://doi.org/10.5281/zenodo.22895684

`CITATION.cff` carries the same metadata in machine-readable form, which is
what GitHub's "Cite this repository" button and reference managers read.

Each GitHub release is archived by Zenodo under a version DOI; the DOI above is
the concept DOI and always resolves to the newest version.

### Acknowledgement of tool use

This template was developed with assistance from Claude (Anthropic). All research questions, data, modelling
decisions and validation are the author's; the assistant is a tool and is not a
contributor or author, per ICMJE and COPE guidance on AI in scholarly work.
State this in your methods section if the target journal requires an AI-use
disclosure — most now do.
