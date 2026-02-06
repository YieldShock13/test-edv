from __future__ import annotations
import os
import warnings
from typing import List, Dict
import pandas as pd
import requests

try:
    from fredapi import Fred  # type: ignore
except Exception:  # pragma: no cover - optional
    Fred = None  # type: ignore

from ..config import settings
from ..utils.io import save_df


FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_ids: List[str], start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    data: Dict[str, pd.Series] = {}

    fred_client = None
    if Fred is not None and settings.fred_api_key:
        try:
            fred_client = Fred(api_key=settings.fred_api_key)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"fredapi init failed, falling back to REST: {e}")

    for sid in series_ids:
        if fred_client is not None:
            s = pd.Series(fred_client.get_series(sid))
            s.name = sid
            data[sid] = s
        else:
            params = {
                "series_id": sid,
                "api_key": settings.fred_api_key or "",
                "file_type": "json",
            }
            if start_date:
                params["observation_start"] = start_date
            if end_date:
                params["observation_end"] = end_date
            r = requests.get(FRED_API_URL, params=params, timeout=30)
            r.raise_for_status()
            j = r.json()
            obs = j.get("observations", [])
            s = pd.Series({pd.to_datetime(o["date"]): float(o["value"]) if o["value"] not in (".", "") else float("nan") for o in obs})
            s.name = sid
            data[sid] = s

    df = pd.DataFrame(data).sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    return df


def cache_series(series_ids: List[str], cache_name: str) -> str:
    df = fetch_series(series_ids)
    path = os.path.join(settings.data_cache_dir, f"fred_{cache_name}.csv")
    save_df(df, path)
    return path