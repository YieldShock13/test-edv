from __future__ import annotations
import numpy as np
import pandas as pd


def run_backtest(price: pd.Series, signal: pd.Series, trading_cost_bps: float = 0.5, slippage_bps: float = 0.5) -> pd.DataFrame:
    price = price.dropna().astype(float)
    signal = signal.reindex(price.index).ffill().fillna(0.0).clip(-1, 1)
    ret = price.pct_change().fillna(0.0)

    # Turnover cost when signal changes
    turnover = signal.diff().abs().fillna(0.0)
    cost = turnover * (trading_cost_bps + slippage_bps) / 10000.0

    strat_ret = signal.shift(1).fillna(0.0) * ret - cost

    equity = (1 + strat_ret).cumprod()
    stats = pd.DataFrame({
        "price": price,
        "signal": signal,
        "ret": ret,
        "strat_ret": strat_ret,
        "equity": equity,
    })
    return stats


def summarize(stats: pd.DataFrame) -> dict:
    sr = stats["strat_ret"].mean() / (stats["strat_ret"].std() + 1e-9) * np.sqrt(252)
    cagr = stats["equity"].iloc[-1] ** (252 / len(stats)) - 1 if len(stats) > 0 else 0.0
    mdd = ((stats["equity"].cummax() - stats["equity"]) / stats["equity"].cummax()).max()
    return {"sharpe": float(sr), "cagr": float(cagr), "max_drawdown": float(mdd)}