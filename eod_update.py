
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from zoneinfo import ZoneInfo

SAST = ZoneInfo("Africa/Johannesburg")

PORTFOLIO_START = pd.Timestamp("2026-09-28")
NAV_FILE = Path("portfolio_history.csv")
BASIS_FILE = Path("portfolio_basis.csv")

# ============================================================
# BHETNC SPECIAL HANDLING
#
# Yahoo historical data begins only on 28 Sep 2026.
# Actual verified 25 Sep 2026 close from broker reconciliation:
# 2,391 cents = R23.91.
#
# From 28 Sep onward we use the actual BHETNC.JO instrument.
# ============================================================

BHETNC_PREVIOUS_DATE = pd.Timestamp("2026-09-25")
BHETNC_PREVIOUS_CLOSE = 2391.0



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


holdings = pd.DataFrame([
    {
        "Holding": name,
        "Ticker": values[0],
        "Qty": values[1],
        "Bucket": values[2],
    }
    for name, values in portfolio.items()
])

tickers = holdings["Ticker"].tolist()


# ============================================================
# PRICE-UNIT REPAIR
#
# Same regime logic validated in historical audit.
# ============================================================

def repair_100x_regimes(series):

    s = series.copy().astype(float)

    if len(s) < 2:
        return s

    repaired = s.copy()
    scale = 1.0

    for i in range(1, len(s)):

        raw_prev = s.iloc[i - 1]
        raw_curr = s.iloc[i]

        if (
            not np.isfinite(raw_prev)
            or not np.isfinite(raw_curr)
            or raw_prev <= 0
            or raw_curr <= 0
        ):
            repaired.iloc[i] = raw_curr * scale
            continue

        raw_ratio = raw_curr / raw_prev

        if 0.005 <= raw_ratio <= 0.02:
            scale *= 100.0

        elif 50 <= raw_ratio <= 200:
            scale /= 100.0

        repaired.iloc[i] = raw_curr * scale

    return repaired


# ============================================================
# DOWNLOAD EACH HOLDING
#
# We deliberately use:
#
#   auto_adjust=False
#   actions=True
#   repair=False
#
# because OUR validated engine controls the reconstruction.
# ============================================================

print("Downloading raw Yahoo prices and corporate actions...")

security_data = {}

for ticker in tickers:

    hist = yf.Ticker(ticker).history(
        period="10d",
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=False
    ).sort_index()

    if hist.empty:
        raise RuntimeError(
            f"STOP — no Yahoo history for {ticker}"
        )

    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)

    if "Close" not in hist.columns:
        raise RuntimeError(
            f"STOP — Close unavailable for {ticker}"
        )

    close_raw = pd.to_numeric(
        hist["Close"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # PRICE NORMALISATION
    # --------------------------------------------------------

    # STXCAP's historical structural problem predates the
    # progression portfolio. From portfolio inception onward
    # it is already in the current price convention.
    #
    # Therefore NO old-history x100 adjustment is required
    # for the live EOD engine.

    close = repair_100x_regimes(
        close_raw
    )

    # --------------------------------------------------------
    # CASH DISTRIBUTIONS
    # --------------------------------------------------------

    if "Dividends" in hist.columns:

        dividends = pd.to_numeric(
            hist["Dividends"],
            errors="coerce"
        ).fillna(0.0)

    else:

        dividends = pd.Series(
            0.0,
            index=hist.index
        )

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    if "Stock Splits" in hist.columns:

        splits = pd.to_numeric(
            hist["Stock Splits"],
            errors="coerce"
        ).fillna(0.0)

    else:

        splits = pd.Series(
            0.0,
            index=hist.index
        )

    security_data[ticker] = pd.DataFrame({
        "Close": close,
        "Dividend": dividends,
        "Split": splits,
    })


# ============================================================
# FIND PORTFOLIO'S TWO MOST RECENT TRADING DATES
#
# Do NOT require every ticker to share every date.
# Individual coverage is checked below.
# ============================================================

all_dates = set()

for ticker, df in security_data.items():

    valid = df.index[
        df["Close"].notna()
    ]

    all_dates.update(valid)


if len(all_dates) < 2:
    raise RuntimeError(
        "STOP — fewer than two trading dates available."
    )


dates = sorted(all_dates)

current_date = pd.Timestamp(
    dates[-1]
).normalize()

previous_date = pd.Timestamp(
    dates[-2]
).normalize()


print("Previous trading date:", previous_date.date())
print("Current trading date: ", current_date.date())


if current_date < PORTFOLIO_START:
    raise RuntimeError(
        "Current observation predates portfolio inception."
    )


# ============================================================
# SECURITY RETURNS
#
# NON-REINVESTED:
#
# P&L =
#     Qty × (P1 - P0 + cash distribution)
#
# Return =
#     P&L / opening market value
#
# Because Yahoo JSE prices and dividends are both in the same
# quote convention, the /100 conversion cancels from the
# security return.
# ============================================================

rows = []


for _, h in holdings.iterrows():

    ticker = h["Ticker"]
    qty = float(h["Qty"])

    df = security_data[ticker]

    # ========================================================
    # BHETNC SPECIAL CASE
    # ========================================================

    if ticker == "BHETNC.JO":

        if current_date not in df.index:
            raise RuntimeError(
                "STOP — current BHETNC.JO market price unavailable."
            )

        # Seed only the verified pre-inception previous close.
        if previous_date == BHETNC_PREVIOUS_DATE:
            prev_close = BHETNC_PREVIOUS_CLOSE

        elif previous_date in df.index:
            prev_close = float(
                df.loc[previous_date, "Close"]
            )

        else:
            raise RuntimeError(
                f"STOP — BHETNC previous close unavailable "
                f"for {previous_date.date()}."
            )

        curr_close = float(
            df.loc[current_date, "Close"]
        )

        dividend = float(
            df.loc[current_date, "Dividend"]
        )

    # ========================================================
    # ALL OTHER HOLDINGS
    # ========================================================

    else:

        if (
            previous_date not in df.index
            or current_date not in df.index
        ):
            raise RuntimeError(
                f"STOP — required dates unavailable for {ticker}"
            )

        prev_close = float(
            df.loc[previous_date, "Close"]
        )

        curr_close = float(
            df.loc[current_date, "Close"]
        )

        dividend = float(
            df.loc[current_date, "Dividend"]
        )


    # ========================================================
    # HARD DATA CONTROLS
    # ========================================================

    if (
        not np.isfinite(prev_close)
        or not np.isfinite(curr_close)
        or prev_close <= 0
        or curr_close <= 0
    ):
        raise RuntimeError(
            f"STOP — invalid price for {ticker}"
        )


    price_ratio = (
        curr_close / prev_close
    )

    if (
        price_ratio > 5.0
        or price_ratio < 0.20
    ):
        raise RuntimeError(
            f"STOP — unresolved price discontinuity "
            f"for {ticker}: "
            f"{prev_close:.4f} -> {curr_close:.4f}"
        )


    if dividend < 0:
        raise RuntimeError(
            f"STOP — negative distribution for {ticker}"
        )


    if dividend > 0:

        distribution_ratio = (
            dividend / curr_close
        )

        if distribution_ratio > 0.25:
            raise RuntimeError(
                f"STOP — suspicious distribution for {ticker}: "
                f"{dividend:.4f} vs price {curr_close:.4f}"
            )


    # ========================================================
    # SPLIT CONTROL
    # ========================================================

    split = float(
        df.loc[current_date, "Split"]
    )

    if split != 0:

        raise RuntimeError(
            f"STOP — stock split detected for {ticker} "
            f"on {current_date.date()}: {split}. "
            f"Manual quantity verification required."
        )


    # ========================================================
    # NON-REINVESTED RETURN
    # ========================================================

    price_return = (
        curr_close / prev_close
    ) - 1.0

    total_return = (
        curr_close
        - prev_close
        + dividend
    ) / prev_close


    # ========================================================
    # OPENING ECONOMIC VALUE
    #
    # Yahoo JSE securities are quoted in cents.
    # ========================================================

    opening_value = (
        qty * prev_close / 100.0
    )

    price_pnl = (
        qty
        * (curr_close - prev_close)
        / 100.0
    )

    dividend_cash = (
        qty
        * dividend
        / 100.0
    )

    total_pnl = (
        price_pnl
        + dividend_cash
    )


    rows.append({
        "Holding": h["Holding"],
        "Ticker": ticker,
        "Bucket": h["Bucket"],
        "Qty": qty,

        "Previous_Close": prev_close,
        "Current_Close": curr_close,

        "Dividend_per_Unit": dividend,

        "Opening_Value": opening_value,

        "Price_Return": price_return,
        "Total_Return": total_return,

        "Price_PnL": price_pnl,
        "Dividend_Cash": dividend_cash,
        "Total_PnL": total_pnl,
    })


detail = pd.DataFrame(rows)


# ============================================================
# COVERAGE
# ============================================================

expected = set(tickers)
received = set(detail["Ticker"])

missing = sorted(
    expected - received
)

if missing:
    raise RuntimeError(
        "STOP — missing holdings: "
        + ", ".join(missing)
    )


# ============================================================
# PORTFOLIO RETURN
#
# Explicit P&L / opening capital.
#
# This is economically identical to beginning-value weighting
# the individual non-reinvested returns, but easier to audit.
# ============================================================

def calc_portfolio_return(df):

    capital = float(
        df["Opening_Value"].sum()
    )

    pnl = float(
        df["Total_PnL"].sum()
    )

    if capital <= 0:
        return np.nan

    return pnl / capital


combined_return = calc_portfolio_return(
    detail
)

main_return = calc_portfolio_return(
    detail[
        detail["Bucket"] == "Main"
    ]
)

subset_return = calc_portfolio_return(
    detail[
        detail["Bucket"] == "Subset"
    ]
)


# ============================================================
# AUDIT TABLE
# ============================================================

audit = detail[[
    "Holding",
    "Ticker",
    "Opening_Value",
    "Previous_Close",
    "Current_Close",
    "Dividend_per_Unit",
    "Price_Return",
    "Total_Return",
    "Price_PnL",
    "Dividend_Cash",
    "Total_PnL",
]].copy()


print()
print("=" * 140)
print("SECURITY RETURN AUDIT — NON-REINVESTED")
print("=" * 140)

display_audit = audit.copy()

for col in [
    "Price_Return",
    "Total_Return",
]:
    display_audit[col] = (
        display_audit[col]
        .map(lambda x: f"{x:+.4%}")
    )

for col in [
    "Opening_Value",
    "Previous_Close",
    "Current_Close",
    "Dividend_per_Unit",
    "Price_PnL",
    "Dividend_Cash",
    "Total_PnL",
]:
    display_audit[col] = (
        display_audit[col]
        .map(lambda x: f"{x:,.2f}")
    )

print(
    display_audit.to_string(
        index=False
    )
)


# ============================================================
# OFFICIAL EOD OUTPUT
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
print(
    "Method: raw price change + cash distributions; "
    "distributions are NOT reinvested."
)
print()


# ============================================================
# SAVE / UPDATE HISTORY
# ============================================================

new_row = pd.DataFrame([{
    "Date": current_date,
    "Combined": combined_return,
    "Main": main_return,
    "Subset": subset_return,
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
# RETURN COMPOSITION HISTORY
#
# Separate audit/display dataset only.
#
# Uses the SAME security-level P&L and SAME opening-capital
# denominator already used by the official portfolio return.
#
# Price contribution + income contribution must therefore
# equal the existing official daily portfolio return.
#
# This does NOT modify portfolio_history.csv or the official
# portfolio-return calculation.
# ============================================================

COMPOSITION_FILE = Path(
    "portfolio_return_composition.csv"
)


def calc_return_composition(df):

    capital = float(
        df["Opening_Value"].sum()
    )

    price_pnl = float(
        df["Price_PnL"].sum()
    )

    income_pnl = float(
        df["Dividend_Cash"].sum()
    )

    if capital <= 0:
        return np.nan, np.nan

    return (
        price_pnl / capital,
        income_pnl / capital
    )


combined_price, combined_income = (
    calc_return_composition(
        detail
    )
)

main_price, main_income = (
    calc_return_composition(
        detail[
            detail["Bucket"] == "Main"
        ]
    )
)

subset_price, subset_income = (
    calc_return_composition(
        detail[
            detail["Bucket"] == "Subset"
        ]
    )
)


# ------------------------------------------------------------
# HARD RECONCILIATION CONTROL
#
# Do not write anything if the decomposition does not reproduce
# the already-calculated official daily portfolio return.
# ------------------------------------------------------------

composition_checks = [
    (
        "Combined",
        combined_price + combined_income,
        combined_return
    ),
    (
        "Main",
        main_price + main_income,
        main_return
    ),
    (
        "Subset",
        subset_price + subset_income,
        subset_return
    ),
]

for (
    portfolio_name,
    decomposed_return,
    official_return
) in composition_checks:

    if not np.isclose(
        decomposed_return,
        official_return,
        rtol=0.0,
        atol=1e-12
    ):
        raise RuntimeError(
            "STOP — return-composition reconciliation failed "
            f"for {portfolio_name}. "
            f"Decomposed={decomposed_return:.12f}, "
            f"Official={official_return:.12f}"
        )


composition_row = pd.DataFrame([{
    "Date": current_date,

    "Combined_Price":
        combined_price,

    "Combined_Income":
        combined_income,

    "Main_Price":
        main_price,

    "Main_Income":
        main_income,

    "Subset_Price":
        subset_price,

    "Subset_Income":
        subset_income,
}])


if COMPOSITION_FILE.exists():

    composition_history = pd.read_csv(
        COMPOSITION_FILE,
        parse_dates=["Date"]
    )

    composition_history = pd.concat(
        [
            composition_history,
            composition_row
        ],
        ignore_index=True
    )

else:

    composition_history = (
        composition_row.copy()
    )


composition_history = (
    composition_history[
        composition_history["Date"]
        >= PORTFOLIO_START
    ]
    .sort_values("Date")
    .drop_duplicates(
        subset=["Date"],
        keep="last"
    )
    .reset_index(drop=True)
)


composition_history.to_csv(
    COMPOSITION_FILE,
    index=False
)


print()
print("=" * 90)
print("RETURN COMPOSITION")
print("=" * 90)

print(
    f"Combined: Price {combined_price:+.4%} | "
    f"Income {combined_income:+.4%} | "
    f"Total {(combined_price + combined_income):+.4%}"
)

print(
    f"Main:     Price {main_price:+.4%} | "
    f"Income {main_income:+.4%} | "
    f"Total {(main_price + main_income):+.4%}"
)

print(
    f"Subset:   Price {subset_price:+.4%} | "
    f"Income {subset_income:+.4%} | "
    f"Total {(subset_price + subset_income):+.4%}"
)

print()
print(
    "PASS — composition reconciles exactly to "
    "official daily portfolio returns."
)

print(
    "portfolio_return_composition.csv updated successfully."
)


# ============================================================
# CUMULATIVE PROGRESSION RETURN
#
# Daily portfolio returns compound through time.
#
# This does NOT mean individual cash dividends are assumed
# reinvested into the securities.
# ============================================================

cum = history.copy()

for col in [
    "Combined",
    "Main",
    "Subset",
]:

    cum[col] = (
        (1 + cum[col])
        .cumprod()
        - 1
    )


print("=" * 90)
print("STORED PORTFOLIO HISTORY")
print("=" * 90)

display_history = history.copy()

for col in [
    "Combined",
    "Main",
    "Subset",
]:

    display_history[col] = (
        display_history[col]
        .map(lambda x: f"{x:+.4%}")
    )


print(
    display_history.to_string(
        index=False
    )
)


print()
print("=" * 90)
print("CUMULATIVE RETURN SINCE 28 SEP 2026")
print("=" * 90)

latest = cum.iloc[-1]

print(
    f"Combined: {latest['Combined']:+.4%}"
)

print(
    f"Main:     {latest['Main']:+.4%}"
)

print(
    f"Subset:   {latest['Subset']:+.4%}"
)

print()
print(
    "portfolio_history.csv updated successfully."
)


# ============================================================
# CORRECTED NON-REINVESTED EOD BASIS
#
# Today's ordinary closing price becomes tomorrow's intraday
# price basis.
#
# Dividends are NOT embedded in this basis.
# Any cash distribution was already recognised explicitly in
# today's official EOD portfolio return.
# ============================================================

basis_rows = []

for _, row in detail.iterrows():

    basis_rows.append({

        "Date":
            current_date,

        "Holding":
            row["Holding"],

        "Ticker":
            row["Ticker"],

        "Bucket":
            row["Bucket"],

        "Qty":
            float(row["Qty"]),

        # Convert Yahoo JSE cents -> rand.
        "Price_Basis":
            float(row["Current_Close"]) / 100.0
    })


current_basis = pd.DataFrame(
    basis_rows
)


# ------------------------------------------------------------
# COVERAGE CONTROL
# ------------------------------------------------------------

expected_basis = set(
    holdings["Ticker"]
)

received_basis = set(
    current_basis["Ticker"]
)

missing_basis = sorted(
    expected_basis - received_basis
)

if missing_basis:

    raise RuntimeError(
        "STOP — cannot write next-day basis. Missing: "
        + ", ".join(missing_basis)
    )


if len(current_basis) != len(holdings):

    raise RuntimeError(
        "STOP — next-day basis row count does not "
        "match portfolio holdings."
    )


# ------------------------------------------------------------
# PRICE CONTROL
# ------------------------------------------------------------

if (
    current_basis["Price_Basis"].isna().any()
    or
    (current_basis["Price_Basis"] <= 0).any()
):

    raise RuntimeError(
        "STOP — invalid price in next-day basis."
    )


# ------------------------------------------------------------
# WRITE
# ------------------------------------------------------------

current_basis.to_csv(
    BASIS_FILE,
    index=False
)


print()
print("=" * 90)
print("NEXT-DAY INTRADAY BASIS")
print("=" * 90)

print(
    current_basis[
        [
            "Ticker",
            "Qty",
            "Price_Basis"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:,.2f}"
    )
)

print()
print(
    f"portfolio_basis.csv saved for "
    f"{current_date.date()}."
)

print(
    "Basis = ordinary EOD Close. "
    "Cash distributions are accounted for separately "
    "in official EOD return."
)
