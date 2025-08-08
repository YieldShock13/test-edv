Quant Trading Stack (Futures + Macro Nowcasting)

Quickstart:
- Create `.env` with your API keys: `FRED_API_KEY`, `NASDAQ_API_KEY` (optional)
- Install deps: `pip install -r requirements.txt`
- Example: fetch data and run a strategy
  - `python -m quantstack.cli fetch fred --series DGS10 DGS2 DCOILWTICO WCESTUS1`
  - `python -m quantstack.cli features build`
  - `python -m quantstack.cli run-strategy curve10_2`

Notes:
- Data cache is stored under `data_cache/`
- LSTM is optional (PyTorch not required by default)
- Affordable sources used: FRED, Yahoo Finance, Nasdaq Data Link (Quandl)