from __future__ import annotations
import os
import pandas as pd

try:
    import nasdaqdatalink as nq  # type: ignore
except Exception:  # pragma: no cover
    nq = None  # type: ignore

from ..config import settings
from ..utils.io import save_df


def fetch_quandl_series(codes: list[str]) -> pd.DataFrame:
    if nq is None:
        raise RuntimeError("nasdaqdatalink is not installed. Remove Quandl usage or install it manually.")
    data_frames: list[pd.DataFrame] = []
    for code in codes:
        df = nq.get(code, api_key=settings.nasdaq_api_key)
        if not isinstance(df, pd.DataFrame):
            df = df.to_frame()
        # Standardize columns to 'Settle'/'Last'
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        price_col = None
        for cand in ["settle", "last", "close", "value"]:
            if cand in cols:
                price_col = cand
                break
        if price_col is None and len(df.columns) > 0:
            price_col = df.columns[0]
        s = df[price_col].rename(code)
        data_frames.append(s)
    out = pd.concat(data_frames, axis=1).sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def cache_quandl_series(codes: list[str], cache_name: str) -> str:
    df = fetch_quandl_series(codes)
    path = os.path.join(settings.data_cache_dir, f"quandl_{cache_name}.csv")
    save_df(df, path)
    return path