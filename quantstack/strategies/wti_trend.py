from __future__ import annotations
import pandas as pd
from ..data.yahoo import fetch_prices
from ..data.fred import fetch_series
from ..features.wti_features import wti_features
from ..backtest.engine import run_backtest, summarize


TICKER = "CL=F"  # WTI futures continuous
FRED_SERIES = ["DCOILWTICO", "WCESTUS1"]


def run() -> tuple[pd.DataFrame, dict]:
    cl = fetch_prices([TICKER])[TICKER].rename("CL").ffill()
    fred = fetch_series(FRED_SERIES)
    feats = wti_features(fred["DCOILWTICO"], fred["WCESTUS1"])
    signal = feats["WTI_RET_5D"].rolling(20).mean().clip(-1, 1)
    stats = run_backtest(cl.reindex(signal.index).ffill(), signal)
    return stats, summarize(stats)