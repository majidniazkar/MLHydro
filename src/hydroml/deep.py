"""Keras wrappers with a scikit-learn-like interface.

Both wrappers expose ``fit(X, y, X_val=None, y_val=None)`` and ``predict(X)``
and take their architecture from the ``deep`` section of the config, so the
input dimension is always derived from the data -- never hard-coded.

Early stopping uses the **validation** split when one is supplied by the
pipeline. It never sees the test split.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DenseRegressor", "LSTMRegressor", "make_windows",
           "available_backends", "resolve_backend", "make_dense", "make_lstm"]


# --------------------------------------------------------------------------- #
# backend selection
# --------------------------------------------------------------------------- #
def available_backends() -> list[str]:
    """Deep-learning frameworks importable in this environment."""
    import importlib.util
    return [name for name, mod in (("tensorflow", "tensorflow"), ("torch", "torch"))
            if importlib.util.find_spec(mod) is not None]


def resolve_backend(requested: str = "auto") -> str:
    """Pick the framework to build with; ``auto`` prefers whatever is installed."""
    have = available_backends()
    if not have:
        raise ImportError(
            "the ann / lstm models need a deep-learning framework. Install "
            "either one:\n    pip install torch\n    pip install tensorflow"
        )
    if requested in ("auto", None):
        return have[0]
    if requested not in ("tensorflow", "torch"):
        raise ValueError("deep.backend must be 'auto', 'tensorflow' or 'torch'")
    if requested not in have:
        raise ImportError(f"deep.backend='{requested}' but it is not installed "
                          f"(available: {have or 'none'})")
    return requested


def preload_backend(requested: str = "auto") -> str:
    """Import the deep-learning framework **before** the rest of the pipeline.

    On Windows with a conda-forge scientific stack, importing PyTorch late in a
    run has been observed to fail with

        OSError: [WinError 127] The specified procedure could not be found.
        Error loading "...torch\\lib\\shm.dll"

    while the identical import as the first statement of the process succeeds.
    Bisection established *when* it fails, not *why*: importing numpy, scipy,
    scikit-learn, matplotlib or this package alone did not reproduce it; the
    failure appeared only once the pipeline had built its dataset. A conflict
    between the OpenMP runtimes loaded by the two stacks is the likely
    mechanism, but that was not confirmed -- only the workaround was.

    Importing the framework first also surfaces a broken installation
    immediately instead of after the data has been prepared.
    """
    name = resolve_backend(requested)
    __import__(name)
    return name


def make_dense(backend: str = "auto", **kw):
    """Backend-appropriate feed-forward regressor."""
    if resolve_backend(backend) == "torch":
        from .deep_torch import TorchDenseRegressor
        return TorchDenseRegressor(**kw)
    return DenseRegressor(**kw)


def make_lstm(backend: str = "auto", **kw):
    """Backend-appropriate LSTM regressor."""
    if resolve_backend(backend) == "torch":
        from .deep_torch import TorchLSTMRegressor
        return TorchLSTMRegressor(**kw)
    return LSTMRegressor(**kw)


def _tf():
    import tensorflow as tf
    return tf


def _set_seed(seed):
    if seed is None:
        return
    tf = _tf()
    try:
        tf.keras.utils.set_random_seed(int(seed))
    except AttributeError:  # very old keras
        np.random.seed(int(seed))
        tf.random.set_seed(int(seed))


def _optimizer(name: str, learning_rate: float):
    tf = _tf()
    name = (name or "adam").lower()
    table = {
        "adam": tf.keras.optimizers.Adam,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "sgd": tf.keras.optimizers.SGD,
        "nadam": tf.keras.optimizers.Nadam,
    }
    if name not in table:
        raise ValueError(f"unsupported optimizer {name!r}; choose from {sorted(table)}")
    return table[name](learning_rate=learning_rate)


def make_windows(X: np.ndarray, window: int) -> np.ndarray:
    """``(n, f) -> (n, window, f)`` sliding windows, left-padded with row 0.

    Padding repeats the first available row backwards, so sample ``t`` only
    ever sees rows ``<= t``: no future information enters the window.
    """
    X = np.asarray(X, dtype=float)
    n, f = X.shape
    window = max(1, int(window))
    pad = np.repeat(X[:1], window - 1, axis=0) if window > 1 else np.empty((0, f))
    padded = np.vstack([pad, X])
    out = np.empty((n, window, f), dtype=float)
    for t in range(n):
        out[t] = padded[t:t + window]
    return out


class _KerasBase:
    """Shared fit/predict plumbing."""

    def __init__(self, *, seed=None, n_features=None, epochs=200, batch_size=32,
                 patience=50, loss="mae", verbose=0, **_ignored):
        self.seed = seed
        self.n_features = n_features
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.loss = loss
        self.verbose = verbose
        self.model_ = None
        self.history_ = None

    # -- to be provided by subclasses --------------------------------------
    def _build(self, n_features: int):
        raise NotImplementedError

    def _shape(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        tf = _tf()
        _set_seed(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.n_features = X.shape[1]
        self.model_ = self._build(self.n_features)

        callbacks, validation_data = [], None
        if X_val is not None and len(X_val) > 0:
            validation_data = (self._shape(np.asarray(X_val, dtype=float)),
                               np.asarray(y_val, dtype=float).ravel())
            monitor = "val_loss"
        else:
            monitor = "loss"
        if self.patience > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor=monitor, mode="min", patience=self.patience,
                restore_best_weights=True))

        self.history_ = self.model_.fit(
            self._shape(X), y,
            validation_data=validation_data,
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=callbacks, verbose=self.verbose,
        )
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        return self.model_.predict(self._shape(X), verbose=0).ravel()

    def get_params(self, deep=True):  # sklearn-ish, used only for logging
        return {k: v for k, v in vars(self).items() if not k.endswith("_")}


class DenseRegressor(_KerasBase):
    """Feed-forward ANN."""

    def __init__(self, *, hidden_units=(18, 36, 36), activation="relu", dropout=0.0,
                 optimizer="adam", learning_rate=1e-3, **kw):
        super().__init__(**kw)
        self.hidden_units = list(hidden_units)
        self.activation = activation
        self.dropout = float(dropout)
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)

    def _shape(self, X):
        return X

    def _build(self, n_features):
        tf = _tf()
        layers = [tf.keras.layers.Input(shape=(n_features,))]
        for units in self.hidden_units:
            layers.append(tf.keras.layers.Dense(int(units), activation=self.activation))
            if self.dropout > 0:
                layers.append(tf.keras.layers.Dropout(self.dropout))
        layers.append(tf.keras.layers.Dense(1, activation="linear"))
        model = tf.keras.Sequential(layers)
        model.compile(optimizer=_optimizer(self.optimizer, self.learning_rate),
                      loss=self.loss, metrics=["mae"])
        return model


class LSTMRegressor(_KerasBase):
    """Stacked LSTM.

    ``sequence_mode``
        ``features_as_timesteps`` -- reshape ``(n, f)`` to ``(n, f, 1)``, i.e.
        the lag columns already present in the table are treated as the time
        axis. This reproduces the original notebook's formulation.
        ``sliding_window`` -- build ``(n, window, f)`` sequences from
        consecutive rows, the conventional formulation for a recurrent model.
    """

    def __init__(self, *, units=(36, 36), activation="tanh", dropout=0.0,
                 optimizer="adam", learning_rate=1e-4,
                 sequence_mode="sliding_window", window=30, **kw):
        super().__init__(**kw)
        self.units = list(units)
        self.activation = activation
        self.dropout = float(dropout)
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        if sequence_mode not in ("features_as_timesteps", "sliding_window"):
            raise ValueError("deep.lstm.sequence_mode must be "
                             "'features_as_timesteps' or 'sliding_window'")
        self.sequence_mode = sequence_mode
        self.window = int(window)

    def _shape(self, X):
        if self.sequence_mode == "features_as_timesteps":
            return X.reshape((X.shape[0], X.shape[1], 1))
        return make_windows(X, self.window)

    def _build(self, n_features):
        tf = _tf()
        if self.sequence_mode == "features_as_timesteps":
            timesteps, channels = n_features, 1
        else:
            timesteps, channels = self.window, n_features
        layers = [tf.keras.layers.Input(shape=(timesteps, channels))]
        for i, units in enumerate(self.units):
            last = i == len(self.units) - 1
            layers.append(tf.keras.layers.LSTM(
                int(units), activation=self.activation, return_sequences=not last))
            if self.dropout > 0:
                layers.append(tf.keras.layers.Dropout(self.dropout))
        layers.append(tf.keras.layers.Dense(1))
        model = tf.keras.Sequential(layers)
        model.compile(optimizer=_optimizer(self.optimizer, self.learning_rate),
                      loss=self.loss, metrics=["mae"])
        return model
