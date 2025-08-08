from __future__ import annotations
import pandas as pd
from ..data.fred import fetch_series
from ..data.yahoo import fetch_prices
from ..features.fx_rates_diff import align_fx_and_rates
from ..backtest.engine import run_backtest, summarize


FRED_SERIES = ["DGS2", "IRLTLT01GBM156N"]  # US 2Y (daily), UK long-term (monthly) as proxy


def run() -> tuple[pd.DataFrame, dict]:
    fred = fetch_series(FRED_SERIES)
    fred["UK2Y_PROXY"] = fred["IRLTLT01GBM156N"].ffill()  # monthly proxy
    rates_spread = (fred["DGS2"] - fred["UK2Y_PROXY"]).rename("RATE_DIFF").to_frame()
    fx = fetch_prices(["GBPUSD=X"])  # GBPUSD
    aligned = align_fx_and_rates(fx[["GBPUSD=X"]], rates_spread)
    # Signal: when US-UK rate diff widens (USD stronger), short GBPUSD
    signal = (-aligned["RATE_DIFF"].rolling(5).mean()).clip(-1, 1)
    stats = run_backtest(aligned["GBPUSD=X"], signal)
    return stats, summarize(stats)