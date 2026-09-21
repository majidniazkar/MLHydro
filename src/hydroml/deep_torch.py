"""PyTorch implementations of the ANN and LSTM regressors.

API-identical to the Keras versions in :mod:`hydroml.deep` --
``fit(X, y, X_val=None, y_val=None)`` / ``predict(X)`` -- so the pipeline is
backend-agnostic. Having both means the neural models run wherever either
framework is installed, instead of being skipped because the *other* one is
missing.

Early stopping restores the best weights seen on the validation split, matching
Keras ``restore_best_weights=True``. It never sees the test split.
"""

from __future__ import annotations

import copy

import numpy as np

from .deep import make_windows

__all__ = ["TorchDenseRegressor", "TorchLSTMRegressor"]


def _torch():
    import torch
    return torch


def _loss_fn(name: str):
    torch = _torch()
    table = {"mae": torch.nn.L1Loss, "mse": torch.nn.MSELoss,
             "huber": torch.nn.HuberLoss,
             "mean_absolute_error": torch.nn.L1Loss,
             "mean_squared_error": torch.nn.MSELoss}
    key = (name or "mae").lower()
    if key not in table:
        raise ValueError(f"unsupported loss {name!r}; choose from {sorted(table)}")
    return table[key]()


def _optimizer(name: str, params, learning_rate: float):
    torch = _torch()
    table = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop,
             "sgd": torch.optim.SGD, "nadam": torch.optim.NAdam}
    key = (name or "adam").lower()
    if key not in table:
        raise ValueError(f"unsupported optimizer {name!r}; choose from {sorted(table)}")
    return table[key](params, lr=learning_rate)


class _TorchBase:
    def __init__(self, *, seed=None, n_features=None, epochs=200, batch_size=32,
                 patience=50, loss="mae", optimizer="adam", learning_rate=1e-3,
                 verbose=0, **_ignored):
        self.seed = seed
        self.n_features = n_features
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.verbose = verbose
        self.model_ = None
        self.history_ = {"loss": [], "val_loss": []}

    # -- provided by subclasses --------------------------------------------
    def _build(self, n_features: int):
        raise NotImplementedError

    def _shape(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        torch = _torch()
        if self.seed is not None:
            torch.manual_seed(int(self.seed))
            np.random.seed(int(self.seed))

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.n_features = X.shape[1]
        self.model_ = self._build(self.n_features)

        xb = torch.tensor(self._shape(X), dtype=torch.float32)
        yb = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            xv = torch.tensor(self._shape(np.asarray(X_val, dtype=float)),
                              dtype=torch.float32)
            yv = torch.tensor(np.asarray(y_val, dtype=float).ravel(),
                              dtype=torch.float32).unsqueeze(1)

        crit = _loss_fn(self.loss)
        opt = _optimizer(self.optimizer, self.model_.parameters(), self.learning_rate)
        n = xb.shape[0]
        best, best_state, stale = np.inf, None, 0

        for epoch in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(n)
            running = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                out = self.model_(xb[idx])
                loss = crit(out, yb[idx])
                loss.backward()
                opt.step()
                running += loss.item() * len(idx)
            train_loss = running / max(1, n)
            self.history_["loss"].append(train_loss)

            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    monitored = crit(self.model_(xv), yv).item()
                self.history_["val_loss"].append(monitored)
            else:
                monitored = train_loss

            if monitored < best - 1e-9:
                best, stale = monitored, 0
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                stale += 1
                if self.patience > 0 and stale >= self.patience:
                    break
            if self.verbose and epoch % 25 == 0:
                print(f"    epoch {epoch} loss={train_loss:.5f} monitor={monitored:.5f}")

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        torch = _torch()
        if self.model_ is None:
            raise RuntimeError("call fit() before predict()")
        self.model_.eval()
        xb = torch.tensor(self._shape(np.asarray(X, dtype=float)), dtype=torch.float32)
        with torch.no_grad():
            return self.model_(xb).numpy().ravel()

    def get_params(self, deep=True):
        return {k: v for k, v in vars(self).items() if not k.endswith("_")}


class TorchDenseRegressor(_TorchBase):
    """Feed-forward ANN (PyTorch)."""

    def __init__(self, *, hidden_units=(18, 36, 36), activation="relu",
                 dropout=0.0, **kw):
        super().__init__(**kw)
        self.hidden_units = list(hidden_units)
        self.activation = activation
        self.dropout = float(dropout)

    def _shape(self, X):
        return X

    def _build(self, n_features):
        torch = _torch()
        acts = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh,
                "sigmoid": torch.nn.Sigmoid, "elu": torch.nn.ELU,
                "leaky_relu": torch.nn.LeakyReLU, "gelu": torch.nn.GELU}
        if self.activation not in acts:
            raise ValueError(f"unsupported activation {self.activation!r}")
        layers, prev = [], n_features
        for units in self.hidden_units:
            layers += [torch.nn.Linear(prev, int(units)), acts[self.activation]()]
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(self.dropout))
            prev = int(units)
        layers.append(torch.nn.Linear(prev, 1))
        return torch.nn.Sequential(*layers)


class _LSTMNet:
    """Namespace holder so the nn.Module is only defined once torch exists."""


def _lstm_module(units, channels, dropout, activation):
    torch = _torch()

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()
            prev = channels
            for u in units:
                self.blocks.append(torch.nn.LSTM(prev, int(u), batch_first=True))
                prev = int(u)
            self.drop = torch.nn.Dropout(dropout) if dropout > 0 else None
            self.head = torch.nn.Linear(prev, 1)
            # tanh is the LSTM cell's own output activation; an extra
            # non-linearity is applied only when something else is asked for
            self.extra = torch.nn.Tanh() if activation == "tanh" else None

        def forward(self, x):
            for i, blk in enumerate(self.blocks):
                x, _ = blk(x)
                if self.drop is not None and i < len(self.blocks) - 1:
                    x = self.drop(x)
            last = x[:, -1, :]
            if self.drop is not None:
                last = self.drop(last)
            return self.head(last)

    return Net()


class TorchLSTMRegressor(_TorchBase):
    """Stacked LSTM (PyTorch). ``sequence_mode`` as in :mod:`hydroml.deep`."""

    def __init__(self, *, units=(36, 36), activation="tanh", dropout=0.0,
                 sequence_mode="sliding_window", window=30, **kw):
        super().__init__(**kw)
        self.units = list(units)
        self.activation = activation
        self.dropout = float(dropout)
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
        channels = 1 if self.sequence_mode == "features_as_timesteps" else n_features
        return _lstm_module(self.units, channels, self.dropout, self.activation)
