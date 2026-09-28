
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

SAST = ZoneInfo("Africa/Johannesburg")

PORTFOLIO_START = pd.Timestamp("2026-09-28")
NAV_FILE = Path("portfolio_history.csv")


# ============================================================
# PORTFOLIO HELD FOR 28 SEP 2026
# ============================================================

portfolio = {
    # MAIN
    "Anglo American": ("AGL.JO", 2491, "Main"),
    "AB InBev": ("ANH.JO", 500, "Main"),
    "BHP Group": ("BHG.JO", 600, "Main"),
    "Bidcorp": ("BID.JO", 3000, "Main"),
    "British American Tobacco": ("BTI.JO", 1840, "Main"),
    "Bidvest": ("BVT.JO", 5000, "Main"),
    "Richemont": ("CFR.JO", 565, "Main"),
    "eMedia Holdings N": ("EMN.JO", 2517, "Main"),
    "Remgro": ("REM.JO", 6000, "Main"),
    "Reinet": ("RNI.JO", 2800, "Main"),
    "Shoprite": ("SHP.JO", 5500, "Main"),
    "Tiger Brands": ("TBS.JO", 3000, "Main"),
    "Valterra Platinum": ("VAL.JO", 250, "Main"),

    # SUBSET
    "FNB Berkshire ETN": ("BHETNC.JO", 2500, "Subset"),
    "10X S&P 500": ("CSP500.JO", 855, "Subset"),
    "Exxaro": ("EXX.JO", 120, "Subset"),
    "NewGold": ("GLD.JO", 77, "Subset"),
    "10X Active Income": ("INCOME.JO", 11300, "Subset"),
    "Satrix Capped All Share": ("STXCAP.JO", 2100, "Subset"),
    "Satrix MSCI EM ESG": ("STXEME.JO", 840, "Subset"),
    "Satrix STOXX Europe": ("STXEUR.JO", 2060, "Subset"),
    "Satrix Global Bond": ("STXGBD.JO", 1730, "Subset"),
    "Satrix SA Bond": ("STXGOV.JO", 13000, "Subset"),
    "Satrix S&P 500 Feeder": ("STX500.JO", 210, "Subset"),
}

holdings = pd.DataFrame(
    [
        {
            "Holding": name,
            "Ticker": values[0],
            "Qty": values[1],
            "Bucket": values[2]
        }
        for name, values in portfolio.items()
    ]
)

tickers = holdings["Ticker"].tolist()


# ============================================================
# DOWNLOAD DAILY DATA
# ============================================================

print("Downloading daily Yahoo data...")

raw = yf.download(
    tickers,
    period="10d",
    interval="1d",
    auto_adjust=False,
    progress=False,
    threads=True
)

if not isinstance(raw.columns, pd.MultiIndex):
    raise RuntimeError("Unexpected Yahoo column structure.")

if "Adj Close" not in raw.columns.get_level_values(0):
    raise RuntimeError("Adjusted Close unavailable.")

adj = raw["Adj Close"].copy()
close = raw["Close"].copy()


# ============================================================
# FIND TWO MOST RECENT COMPLETE TRADING DATES
# ============================================================

dates = adj.dropna(how="all").index.sort_values()

if len(dates) < 2:
    raise RuntimeError("Not enough daily observations.")

current_date = pd.Timestamp(dates[-1]).normalize()
previous_date = pd.Timestamp(dates[-2]).normalize()

print("Previous trading date:", previous_date.date())
print("Current trading date: ", current_date.date())


if current_date < PORTFOLIO_START:
    raise RuntimeError(
        "Current observation predates portfolio inception."
    )


# ============================================================
# CALCULATE SECURITY TOTAL RETURNS
# ============================================================

rows = []

for _, h in holdings.iterrows():

    ticker = h["Ticker"]

    if ticker not in adj.columns:
        print(f"WARNING: {ticker} missing.")
        continue

    prev_adj = adj.loc[previous_date, ticker]
    curr_adj = adj.loc[current_date, ticker]

    prev_close = close.loc[previous_date, ticker]
    curr_close = close.loc[current_date, ticker]

    if pd.isna(prev_adj) or pd.isna(curr_adj):
        print(f"WARNING: {ticker} missing adjusted close.")
        continue

    # Yahoo JSE quote units are immaterial for returns.
    total_return = float(curr_adj / prev_adj - 1)

    # Beginning-of-day economic value.
    #
    # We use previous ordinary close × quantity to establish
    # actual portfolio capital at risk at the beginning of day.
    #
    # Unit conversion /100 because Yahoo JSE prices are quoted
    # in cents for these instruments.
    opening_value = float(h["Qty"] * prev_close / 100)

    rows.append({
        "Holding": h["Holding"],
        "Ticker": ticker,
        "Bucket": h["Bucket"],
        "Qty": h["Qty"],
        "Opening_Value": opening_value,
        "Total_Return": total_return
    })


detail = pd.DataFrame(rows)

if detail.empty:
    raise RuntimeError("No valid holdings.")


# ============================================================
# COVERAGE CHECK
# ============================================================

expected = set(tickers)
received = set(detail["Ticker"])

missing = sorted(expected - received)

if missing:
    raise RuntimeError(
        "STOP — missing holdings: " + ", ".join(missing)
    )


# ============================================================
# PORTFOLIO RETURN
#
# Beginning-of-day market-value weighting.
# ============================================================

def calc_portfolio_return(df):

    capital = df["Opening_Value"].sum()

    if capital <= 0:
        return np.nan

    pnl_equivalent = (
        df["Opening_Value"]
        *
        df["Total_Return"]
    ).sum()

    return float(pnl_equivalent / capital)


combined_return = calc_portfolio_return(detail)

main_return = calc_portfolio_return(
    detail[detail["Bucket"] == "Main"]
)

subset_return = calc_portfolio_return(
    detail[detail["Bucket"] == "Subset"]
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 90)
print("OFFICIAL EOD PORTFOLIO RETURN")
print("=" * 90)

print(f"Date:      {current_date.date()}")
print(f"Combined:  {combined_return:+.4%}")
print(f"Main:      {main_return:+.4%}")
print(f"Subset:    {subset_return:+.4%}")

print()
print("These are Adjusted-Close total returns.")
print()


# ============================================================
# SAVE / UPDATE HISTORY
# ============================================================

new_row = pd.DataFrame([{
    "Date": current_date,
    "Combined": combined_return,
    "Main": main_return,
    "Subset": subset_return
}])


if NAV_FILE.exists():

    history = pd.read_csv(
        NAV_FILE,
        parse_dates=["Date"]
    )

    history = pd.concat(
        [history, new_row],
        ignore_index=True
    )

else:

    history = new_row.copy()


history = (
    history[
        history["Date"] >= PORTFOLIO_START
    ]
    .sort_values("Date")
    .drop_duplicates(
        subset=["Date"],
        keep="last"
    )
    .reset_index(drop=True)
)


history.to_csv(
    NAV_FILE,
    index=False
)


# ============================================================
# CUMULATIVE RETURN
# ============================================================

cum = history.copy()

for col in ["Combined", "Main", "Subset"]:

    cum[col] = (
        (1 + cum[col])
        .cumprod()
        - 1
    )


print("=" * 90)
print("STORED PORTFOLIO HISTORY")
print("=" * 90)

display_history = history.copy()

for col in ["Combined", "Main", "Subset"]:
    display_history[col] = display_history[col].map(
        lambda x: f"{x:+.4%}"
    )

print(display_history.to_string(index=False))


print()
print("=" * 90)
print("CUMULATIVE RETURN SINCE 28 SEP 2026")
print("=" * 90)

latest = cum.iloc[-1]

print(f"Combined: {latest['Combined']:+.4%}")
print(f"Main:     {latest['Main']:+.4%}")
print(f"Subset:   {latest['Subset']:+.4%}")

print()
print("portfolio_history.csv updated successfully.")
