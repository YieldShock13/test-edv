from __future__ import annotations
import os
import pandas as pd
from ..config import settings


DEFAULT_PATH = os.path.join(settings.data_cache_dir, "satellite_wti_fill_levels.csv")


def load_satellite_fill_levels(path: str = DEFAULT_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        # Return empty placeholder with expected schema
        return pd.DataFrame(columns=["fill_level"], index=pd.to_datetime([]))
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df