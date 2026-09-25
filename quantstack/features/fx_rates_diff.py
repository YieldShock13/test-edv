from __future__ import annotations
import pandas as pd


def align_fx_and_rates(fx_df: pd.DataFrame, rates_spread_df: pd.DataFrame) -> pd.DataFrame:
    df = fx_df.rename(columns={fx_df.columns[0]: "FX"}).join(rates_spread_df, how="outer").sort_index().ffill()
    return df