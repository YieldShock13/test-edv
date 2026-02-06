from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class LSTMForecaster:
    def __init__(self, lookback: int = 60, hidden_size: int = 32, num_layers: int = 1):
        if torch is None:
            warnings.warn("PyTorch not installed; LSTMForecaster is unavailable.")
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None

    def _build(self, input_size: int = 1):
        self.model = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.head = nn.Linear(self.hidden_size, 1)

    def _make_sequences(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        s = series.dropna().astype(float)
        X_list, y_list, idx_list = [], [], []
        for i in range(self.lookback, len(s) - 1):
            X_list.append(s.values[i - self.lookback:i])
            y_list.append(s.values[i + 1])
            idx_list.append(s.index[i + 1])
        X = np.array(X_list).reshape(-1, self.lookback, 1)
        y = np.array(y_list).reshape(-1, 1)
        return X, y, pd.DatetimeIndex(idx_list)

    def fit_predict(self, series: pd.Series, epochs: int = 5, lr: float = 1e-3) -> pd.Series:
        if torch is None:
            raise RuntimeError("Install torch to use LSTMForecaster")
        X, y, idx = self._make_sequences(series)
        self._build(input_size=1)
        model = self.model
        head = self.head
        assert model is not None and head is not None
        opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)
        loss_fn = torch.nn.MSELoss()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        for _ in range(epochs):
            opt.zero_grad()
            out, _ = model(X_t)
            pred = head(out[:, -1, :])
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
        with torch.no_grad():
            out, _ = model(X_t)
            pred = head(out[:, -1, :]).squeeze(-1).numpy()
        return pd.Series(pred, index=idx, name="lstm_pred")