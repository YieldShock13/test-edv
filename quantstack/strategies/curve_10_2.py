from __future__ import annotations
import pandas as pd
from ..data.yahoo import fetch_prices
from ..data.fred import fetch_series
from ..features.rates_curve import compute_ten_two_spread
from ..backtest.engine import run_backtest, summarize


TICKERS = ["ZN=F", "ZT=F"]  # 10Y and 2Y futures
FRED_SERIES = ["DGS10", "DGS2"]


def run() -> tuple[pd.DataFrame, dict]:
    fut = fetch_prices(TICKERS)
    spread_px = (fut["ZN=F"] - fut["ZT=F"]).rename("SPREAD").dropna()
    fred = fetch_series(FRED_SERIES)
    ten_two = compute_ten_two_spread(fred)
    signal = ten_two["TEN_TWO_SPREAD"].pct_change(20).rolling(5).mean().clip(-1, 1)
    stats = run_backtest(spread_px.reindex(signal.index).ffill(), signal)
    return stats, summarize(stats)