
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

SAST = ZoneInfo("Africa/Johannesburg")

PORTFOLIO_START = pd.Timestamp("2026-09-28")
NAV_FILE = Path("portfolio_history.csv")
BASIS_FILE = Path("portfolio_basis.csv")


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
# CURRENT EOD DATE
# ============================================================

dates = adj.dropna(how="all").index.sort_values()

if len(dates) < 1:
    raise RuntimeError("No daily observations available.")

current_date = pd.Timestamp(dates[-1]).normalize()

print("Current trading date:", current_date.date())

if current_date < PORTFOLIO_START:
    raise RuntimeError(
        "Current observation predates portfolio inception."
    )


# ============================================================
# CURRENT EOD ADJUSTED-CLOSE BASIS
# ============================================================

basis_rows = []

for _, h in holdings.iterrows():

    ticker = h["Ticker"]

    if ticker not in adj.columns:
        raise RuntimeError(
            f"STOP — {ticker} absent from Yahoo response."
        )

    curr_adj = adj.loc[current_date, ticker]

    if pd.isna(curr_adj):
        raise RuntimeError(
            f"STOP — no EOD adjusted close for {ticker}"
        )

    # Yahoo JSE instruments quoted in cents.
    basis_price = float(curr_adj / 100)

    basis_rows.append({
        "Date": current_date,
        "Holding": h["Holding"],
        "Ticker": ticker,
        "Bucket": h["Bucket"],
        "Qty": float(h["Qty"]),
        "Adjusted_Basis": basis_price
    })

current_basis = pd.DataFrame(basis_rows)

if len(current_basis) != len(holdings):
    raise RuntimeError(
        "STOP — current EOD basis does not contain every holding."
    )


# ============================================================
# INCEPTION DAY
#
# 28 Sep 2026 establishes the official portfolio basis.
# There is deliberately NO portfolio return on inception day.
# ============================================================

if not BASIS_FILE.exists():

    current_basis.to_csv(BASIS_FILE, index=False)

    print()
    print("=" * 90)
    print("PORTFOLIO INCEPTION BASIS ESTABLISHED")
    print("=" * 90)
    print(f"Basis date: {current_date.date()}")
    print(f"Holdings:   {len(current_basis)}")
    print()
    print("No portfolio return recorded on inception day.")
    print("Tomorrow's return will be measured from this adjusted-close basis.")

    raise SystemExit(0)


# ============================================================
# LOAD PRIOR OFFICIAL EOD BASIS
# ============================================================

prior_basis = pd.read_csv(
    BASIS_FILE,
    parse_dates=["Date"]
)

required_cols = {
    "Date", "Holding", "Ticker",
    "Bucket", "Qty", "Adjusted_Basis"
}

missing_cols = required_cols - set(prior_basis.columns)

if missing_cols:
    raise RuntimeError(
        "STOP — portfolio_basis.csv missing columns: "
        + ", ".join(sorted(missing_cols))
    )

prior_basis["Date"] = pd.to_datetime(
    prior_basis["Date"]
).dt.normalize()

prior_date = prior_basis["Date"].max()

prior_basis = prior_basis[
    prior_basis["Date"] == prior_date
].copy()

print("Prior official basis date:", prior_date.date())


# Do not create a duplicate daily return if rerun on same EOD date.
if current_date <= prior_date:

    print()
    print(
        f"No new trading day to record. "
        f"Current EOD={current_date.date()}, "
        f"stored basis={prior_date.date()}."
    )

    raise SystemExit(0)


# ============================================================
# COVERAGE CHECK
# ============================================================

expected = set(holdings["Ticker"])

prior_received = set(prior_basis["Ticker"])
current_received = set(current_basis["Ticker"])

missing_prior = sorted(expected - prior_received)
missing_current = sorted(expected - current_received)

if missing_prior:
    raise RuntimeError(
        "STOP — holdings missing from prior basis: "
        + ", ".join(missing_prior)
    )

if missing_current:
    raise RuntimeError(
        "STOP — holdings missing from current basis: "
        + ", ".join(missing_current)
    )


# ============================================================
# DAILY SECURITY RETURNS
#
# Official return:
# current EOD adjusted close / prior official adjusted basis - 1
# ============================================================

detail = prior_basis.merge(
    current_basis[
        ["Ticker", "Adjusted_Basis"]
    ].rename(
        columns={"Adjusted_Basis": "Current_Adjusted_Basis"}
    ),
    on="Ticker",
    how="left",
    validate="one_to_one"
)

detail["Total_Return"] = (
    detail["Current_Adjusted_Basis"]
    /
    detail["Adjusted_Basis"]
    - 1
)

# Beginning-of-day economic value based on prior adjusted basis.
detail["Opening_Value"] = (
    detail["Qty"]
    *
    detail["Adjusted_Basis"]
)


# ============================================================
# PORTFOLIO RETURN
# ============================================================

def calc_portfolio_return(df):

    capital = df["Opening_Value"].sum()

    if capital <= 0:
        raise RuntimeError(
            "STOP — non-positive opening portfolio value."
        )

    pnl = (
        df["Opening_Value"]
        *
        df["Total_Return"]
    ).sum()

    return float(pnl / capital)


combined_return = calc_portfolio_return(detail)

main_return = calc_portfolio_return(
    detail[detail["Bucket"] == "Main"]
)

subset_return = calc_portfolio_return(
    detail[detail["Bucket"] == "Subset"]
)


print()
print("=" * 90)
print("OFFICIAL EOD PORTFOLIO RETURN")
print("=" * 90)

print(f"Prior basis: {prior_date.date()}")
print(f"Date:        {current_date.date()}")
print(f"Combined:    {combined_return:+.4%}")
print(f"Main:        {main_return:+.4%}")
print(f"Subset:      {subset_return:+.4%}")

print()
print("Returns use the stored prior-EOD adjusted-close basis.")
print()


# ============================================================
# SAVE CURRENT ADJUSTED CLOSE AS NEXT-DAY BASIS
# ============================================================

current_basis.to_csv(
    BASIS_FILE,
    index=False
)

print("Next-day adjusted-close basis saved.")
print(
    f"Basis date: {current_date.date()} | "
    f"Holdings: {len(current_basis)}"
)
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
