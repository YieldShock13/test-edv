from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LinearRegression


class MacroNowcaster:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_cols: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        df = features.copy().join(target.rename("target"), how="inner").dropna()
        X = df[features.columns]
        y = df["target"]
        self.feature_cols = list(features.columns)
        self.model.fit(X, y)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        X = features[self.feature_cols].copy().fillna(method="ffill").fillna(0.0)
        yhat = self.model.predict(X)
        return pd.Series(yhat, index=features.index, name="nowcast")


def build_fx_nowcast(df_aligned: pd.DataFrame) -> pd.Series:
    # Toy target: next 5d FX return
    target = df_aligned["FX"].pct_change(5).shift(-5)
    features = df_aligned.drop(columns=["FX"])  # e.g., rate spreads
    model = MacroNowcaster()
    model.fit(features, target)
    return model.predict(features)