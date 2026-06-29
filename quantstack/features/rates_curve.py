from __future__ import annotations
import pandas as pd


def compute_ten_two_spread(fred_df: pd.DataFrame) -> pd.DataFrame:
    df = fred_df.copy()
    for col in ["DGS10", "DGS2"]:
        if col not in df.columns:
            raise ValueError(f"Missing {col} in FRED dataframe")
    df["TEN_TWO_SPREAD"] = df["DGS10"] - df["DGS2"]
    return df[["TEN_TWO_SPREAD"]]


def compute_us_uk_10y_spread(us_series: pd.Series, uk_series: pd.Series) -> pd.DataFrame:
    spread = us_series.rename("US10Y").to_frame().join(uk_series.rename("UK10Y"), how="outer").sort_index()
    spread["US_UK_10Y_SPREAD"] = spread["US10Y"] - spread["UK10Y"]
    return spread[["US_UK_10Y_SPREAD"]]