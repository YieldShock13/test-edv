from __future__ import annotations
import pandas as pd


def wti_features(oil_px: pd.Series, crude_stocks: pd.Series | None = None, sat_fill: pd.Series | None = None) -> pd.DataFrame:
    df = oil_px.rename("WTI").to_frame()
    df["WTI_RET_5D"] = df["WTI"].pct_change(5)
    df["WTI_VOL_20D"] = df["WTI"].pct_change().rolling(20).std() * (252 ** 0.5)
    if crude_stocks is not None:
        s = crude_stocks.rename("CRUDE_STOCKS").reindex(df.index).ffill()
        df["CRUDE_STOCKS_Z"] = (s - s.rolling(52).mean()) / s.rolling(52).std()
    if sat_fill is not None:
        sf = sat_fill.rename("SAT_FILL").reindex(df.index).ffill()
        df["SAT_FILL_Z"] = (sf - sf.rolling(52).mean()) / sf.rolling(52).std()
    return df