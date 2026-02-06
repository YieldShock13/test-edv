from __future__ import annotations
import pandas as pd
from ..data.yahoo import fetch_prices
from ..backtest.engine import run_backtest, summarize


TICKER = "ZN=F"  # CBOT 10Y Note futures continuous


def run() -> tuple[pd.DataFrame, dict]:
    px = fetch_prices([TICKER])[TICKER].rename("PX")
    signal = px.pct_change(20).rolling(5).mean() * -1
    stats = run_backtest(px, signal)
    return stats, summarize(stats)