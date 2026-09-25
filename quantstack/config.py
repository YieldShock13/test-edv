import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    data_cache_dir: str = os.environ.get("DATA_CACHE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_cache")))
    fred_api_key: str | None = os.environ.get("FRED_API_KEY")
    nasdaq_api_key: str | None = os.environ.get("NASDAQ_API_KEY") or os.environ.get("NASDQ_API_KEY")
    eia_api_key: str | None = os.environ.get("EIA_API_KEY")

settings = Settings()

os.makedirs(settings.data_cache_dir, exist_ok=True)