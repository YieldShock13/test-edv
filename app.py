
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Portfolio Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

SAST = ZoneInfo("Africa/Johannesburg")
NOW = datetime.now(SAST)

PORTFOLIO_START = pd.Timestamp("2026-09-28")
NAV_FILE = Path("portfolio_history.csv")
BASIS_FILE = Path("portfolio_basis.csv")


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px;
    padding: 15px 18px;
}

.small-note {
    opacity: 0.70;
    font-size: 0.84rem;
}

.section-note {
    opacity: 0.70;
    margin-top: -10px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PORTFOLIO
# ============================================================

# Quantities / portfolio composition established at inception.
#
# IMPORTANT:
# Historical individual-security returns below are NOT used
# to create historical portfolio performance.

portfolio = {
    # ---------------- MAIN ----------------

    "Anglo American": {
        "ticker": "AGL.JO",
        "qty": 2491,
        "bucket": "Main"
    },

    "AB InBev": {
        "ticker": "ANH.JO",
        "qty": 500,
        "bucket": "Main"
    },

    "BHP Group": {
        "ticker": "BHG.JO",
        "qty": 600,
        "bucket": "Main"
    },

    "Bidcorp": {
        "ticker": "BID.JO",
        "qty": 3000,
        "bucket": "Main"
    },

    "British American Tobacco": {
        "ticker": "BTI.JO",
        "qty": 1840,
        "bucket": "Main"
    },

    "Bidvest": {
        "ticker": "BVT.JO",
        "qty": 5000,
        "bucket": "Main"
    },

    "Richemont": {
        "ticker": "CFR.JO",
        "qty": 565,
        "bucket": "Main"
    },

    "eMedia Holdings N": {
        "ticker": "EMN.JO",
        "qty": 2517,
        "bucket": "Main"
    },

    "Remgro": {
        "ticker": "REM.JO",
        "qty": 6000,
        "bucket": "Main"
    },

    "Reinet": {
        "ticker": "RNI.JO",
        "qty": 2800,
        "bucket": "Main"
    },

    "Shoprite": {
        "ticker": "SHP.JO",
        "qty": 5500,
        "bucket": "Main"
    },

    "Tiger Brands": {
        "ticker": "TBS.JO",
        "qty": 3000,
        "bucket": "Main"
    },

    "Valterra Platinum": {
        "ticker": "VAL.JO",
        "qty": 250,
        "bucket": "Main"
    },

    # ---------------- SUBSET ----------------

    "FNB Berkshire ETN": {
        "ticker": "BHETNC.JO",
        "qty": 2500,
        "bucket": "Subset"
    },

    "10X S&P 500": {
        "ticker": "CSP500.JO",
        "qty": 855,
        "bucket": "Subset"
    },

    "Exxaro": {
        "ticker": "EXX.JO",
        "qty": 120,
        "bucket": "Subset"
    },

    "NewGold": {
        "ticker": "GLD.JO",
        "qty": 77,
        "bucket": "Subset"
    },

    "10X Active Income": {
        "ticker": "INCOME.JO",
        "qty": 11300,
        "bucket": "Subset"
    },

    "Satrix Capped All Share": {
        "ticker": "STXCAP.JO",
        "qty": 2100,
        "bucket": "Subset"
    },

    "Satrix MSCI EM ESG": {
        "ticker": "STXEME.JO",
        "qty": 840,
        "bucket": "Subset"
    },

    "Satrix STOXX Europe": {
        "ticker": "STXEUR.JO",
        "qty": 2060,
        "bucket": "Subset"
    },

    "Satrix Global Bond": {
        "ticker": "STXGBD.JO",
        "qty": 1730,
        "bucket": "Subset"
    },

    "Satrix SA Bond": {
        "ticker": "STXGOV.JO",
        "qty": 13000,
        "bucket": "Subset"
    },

    "Satrix S&P 500 Feeder": {
        "ticker": "STX500.JO",
        "qty": 210,
        "bucket": "Subset"
    }
}


master = pd.DataFrame(portfolio).T
master.index.name = "Holding"

tickers = master["ticker"].tolist()


# ============================================================
# YAHOO DOWNLOAD
#
# 15-minute cache = dashboard refresh cadence.
# ============================================================

@st.cache_data(ttl=900)
def download_market_data(tickers):

    intraday = yf.download(
        tickers,
        period="5d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        threads=True
    )

    history = yf.download(
        tickers,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True
    )

    return intraday, history


intraday_raw, history_raw = download_market_data(tickers)


# ============================================================
# EXTRACT DATA
# ============================================================

def get_field(df, field):

    if isinstance(df.columns, pd.MultiIndex):

        if field not in df.columns.get_level_values(0):
            return pd.DataFrame()

        return df[field].copy()

    return pd.DataFrame()


intraday_close = get_field(
    intraday_raw,
    "Close"
)

history_close = get_field(
    history_raw,
    "Close"
)

adj_close = get_field(
    history_raw,
    "Adj Close"
)


# ============================================================
# JSE UNIT NORMALISATION
#
# Yahoo occasionally mixes cents and rand for JSE history.
#
# Repair ONLY obvious ~100x discontinuities.
# ============================================================

def normalise_jse_units(series):

    s = series.copy().sort_index()

    valid = s.dropna()

    if len(valid) < 2:
        return s

    cleaned = valid.copy()

    previous = float(cleaned.iloc[0])

    for i in range(1, len(cleaned)):

        current = float(cleaned.iloc[i])

        if previous <= 0 or current <= 0:
            previous = current
            continue

        ratio = current / previous

        if ratio > 50:

            candidate = current / 100

            if 0.50 <= candidate / previous <= 1.50:
                cleaned.iloc[i] = candidate

        elif ratio < 0.02:

            candidate = current * 100

            if 0.50 <= candidate / previous <= 1.50:
                cleaned.iloc[i] = candidate

        previous = float(cleaned.iloc[i])

    s.loc[cleaned.index] = cleaned

    return s


for ticker in adj_close.columns:

    adj_close[ticker] = normalise_jse_units(
        adj_close[ticker]
    )


# ============================================================
# LIVE PERFORMANCE
#
# RULE:
# If an official EOD basis exists from a PRIOR trading date,
# today's ordinary intraday market price is compared against
# that persisted ordinary EOD Close basis.
#
# On portfolio inception day only, before the first EOD basis
# exists, fall back to the prior ordinary market close.
# ============================================================

basis = pd.DataFrame()

if BASIS_FILE.exists():

    basis = pd.read_csv(
        BASIS_FILE,
        parse_dates=["Date"]
    )

    required_basis_cols = {
        "Date",
        "Holding",
        "Ticker",
        "Bucket",
        "Qty",
        "Price_Basis"
    }

    missing_cols = required_basis_cols - set(basis.columns)

    if missing_cols:
        raise RuntimeError(
            "Invalid portfolio_basis.csv. Missing: "
            + ", ".join(sorted(missing_cols))
        )

    basis["Date"] = pd.to_datetime(
        basis["Date"]
    ).dt.normalize()


live_rows = []

for holding, row in master.iterrows():

    ticker = row["ticker"]
    qty = float(row["qty"])
    bucket = row["bucket"]

    if ticker not in intraday_close.columns:
        continue

    s = intraday_close[ticker].dropna()

    if s.empty:
        continue

    idx = s.index

    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    local_idx = idx.tz_convert(SAST)
    local_dates = local_idx.date

    unique_dates = sorted(set(local_dates))

    if not unique_dates:
        continue

    today = unique_dates[-1]

    today_positions = np.where(
        local_dates == today
    )[0]

    if len(today_positions) == 0:
        continue

    today_s = s.iloc[today_positions]

    latest_raw = float(today_s.iloc[-1])

    # Yahoo JSE intraday quote in cents -> rand
    latest_price = latest_raw / 100

    basis_price = np.nan
    basis_date = None
    basis_source = None

    # --------------------------------------------------------
    # PRIMARY METHOD:
    # persisted previous ordinary EOD Close
    # --------------------------------------------------------

    if not basis.empty:

        b = basis[
            basis["Ticker"] == ticker
        ].copy()

        if not b.empty:

            # Only use a basis strictly earlier than today.
            b = b[
                b["Date"].dt.date < today
            ]

            if not b.empty:

                b = b.sort_values("Date").iloc[-1]

                basis_price = float(
                    b["Price_Basis"]
                )

                basis_date = pd.Timestamp(
                    b["Date"]
                )

                basis_source = (
                    "Prior EOD ordinary Close"
                )

    # --------------------------------------------------------
    # INCEPTION-DAY FALLBACK ONLY:
    # prior ordinary market close.
    # Once portfolio_basis.csv exists, this should no longer
    # be needed on subsequent trading days.
    # --------------------------------------------------------

    if pd.isna(basis_price):

        prior_dates = [
            d for d in unique_dates
            if d < today
        ]

        if not prior_dates:
            continue

        prior_day = prior_dates[-1]

        prior_positions = np.where(
            local_dates == prior_day
        )[0]

        prior_s = s.iloc[
            prior_positions
        ]

        if prior_s.empty:
            continue

        basis_price = (
            float(prior_s.iloc[-1]) / 100
        )

        basis_date = pd.Timestamp(
            prior_day
        )

        basis_source = (
            "Prior ordinary close "
            "(inception fallback)"
        )

    if basis_price <= 0:
        continue

    ret = (
        latest_price / basis_price - 1
    )

    opening_value = (
        qty * basis_price
    )

    current_value = (
        qty * latest_price
    )

    pnl = (
        current_value - opening_value
    )

    timestamp = local_idx[
        today_positions[-1]
    ]

    live_rows.append({
        "Holding": holding,
        "Ticker": ticker,
        "Bucket": bucket,
        "Qty": qty,
        "Prior_Close": basis_price,
        "Basis_Date": basis_date,
        "Basis_Source": basis_source,
        "Latest": latest_price,
        "Today_Return": ret,
        "Opening_Value": opening_value,
        "Current_Value": current_value,
        "PnL": pnl,
        "Last_Update": timestamp
    })


live = pd.DataFrame(live_rows)

# ============================================================
# PORTFOLIO SUMMARY
# ============================================================

def portfolio_stats(bucket=None):

    if bucket is None:
        x = live.copy()
        name = "Combined"

    else:
        x = live[live["Bucket"] == bucket].copy()
        name = bucket

    opening = x["Opening_Value"].sum()
    current = x["Current_Value"].sum()
    pnl = x["PnL"].sum()

    ret = (
        pnl / opening
        if opening
        else np.nan
    )

    return {
        "Portfolio": name,
        "Opening": opening,
        "Current": current,
        "PnL": pnl,
        "Return": ret
    }


combined = portfolio_stats()
main = portfolio_stats("Main")
subset = portfolio_stats("Subset")


# ============================================================
# HEADER
# ============================================================

st.image("assets/orca_ribbon.webp", use_container_width=True)

latest_timestamp = (
    live["Last_Update"].max()
    if not live.empty
    else None
)

if latest_timestamp is not None:

    st.markdown(
        f"""
        <div class="small-note">
        Latest market observation:
        {latest_timestamp.strftime("%d %b %Y %H:%M SAST")}
        · Market data cached for 15 minutes
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LIVE CARDS
# ============================================================

st.subheader("Live / Intraday Estimate")

st.markdown(
    """
    <div class="section-note">
    Ordinary intraday prices versus the previous official EOD
    price basis. Official daily portfolio performance uses
    raw price change plus cash distributions; distributions
    are not reinvested.
    </div>
    """,
    unsafe_allow_html=True
)


cols = st.columns(3)

stats = [
    combined,
    main,
    subset
]


for col, stat in zip(cols, stats):

    with col:

        st.metric(
            label=stat["Portfolio"],
            value=f"R{stat['Current']:,.0f}",
            delta=f"{stat['Return']:+.2%}"
        )

        st.caption(
            f"Today P&L: R{stat['PnL']:+,.0f}"
        )


# ============================================================
# PORTFOLIO PERFORMANCE
#
# ONLY ACTUAL OBSERVATIONS FROM 28 SEP 2026 FORWARD.
# ============================================================

st.divider()

st.subheader(
    "Actual Portfolio Performance"
)

st.markdown(
    """
    <div class="section-note">
    Performance record begins 28 September 2026.
    No historical portfolio performance is backfilled.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD STORED DAILY RETURNS
# ============================================================

if NAV_FILE.exists():

    nav = pd.read_csv(
        NAV_FILE,
        parse_dates=["Date"]
    )

else:

    nav = pd.DataFrame(
        columns=[
            "Date",
            "Combined",
            "Main",
            "Subset"
        ]
    )


if not nav.empty:

    nav = (
        nav[
            nav["Date"] >= PORTFOLIO_START
        ]
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last"
        )
    )


# ============================================================
# CUMULATIVE RETURN CHART
# ============================================================

if nav.empty:

    st.info(
        "The official portfolio track record begins at today's "
        "EOD close. The cumulative-return chart will populate "
        "as official daily returns are recorded."
    )

else:

    cumulative = nav.copy()

    for col in [
        "Combined",
        "Main",
        "Subset"
    ]:

        cumulative[col] = (
            (1 + cumulative[col])
            .cumprod()
            - 1
        )


    selected = st.radio(
        "Portfolio",
        ["Combined", "Main", "Subset"],
        horizontal=True
    )


    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cumulative["Date"],
            y=cumulative[selected] * 100,
            mode="lines+markers" if len(cumulative) == 1 else "lines",
            name=selected,
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Cumulative return: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        height=430,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        yaxis_title="Cumulative Return (%)",
        xaxis_title=None,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



    # ========================================================
    # RETURN COMPOSITION — FROM INCEPTION
    #
    # Read-only display using the separate composition history.
    # Daily price and income contributions are linked through
    # the official portfolio NAV path so that:
    #
    # Capital Gains + Income = official cumulative return.
    #
    # Nothing here modifies portfolio returns or stored data.
    # ========================================================

    composition_file = Path(
        "portfolio_return_composition.csv"
    )

    if composition_file.exists():

        composition = pd.read_csv(
            composition_file,
            parse_dates=["Date"]
        )

        composition = (
            composition[
                composition["Date"]
                >= PORTFOLIO_START
            ]
            .sort_values("Date")
            .drop_duplicates(
                subset=["Date"],
                keep="last"
            )
            .reset_index(drop=True)
        )

        price_col = (
            f"{selected}_Price"
        )

        income_col = (
            f"{selected}_Income"
        )

        if (
            price_col in composition.columns
            and income_col in composition.columns
        ):

            linked = (
                nav[
                    ["Date", selected]
                ]
                .merge(
                    composition[
                        [
                            "Date",
                            price_col,
                            income_col
                        ]
                    ],
                    on="Date",
                    how="inner"
                )
                .sort_values("Date")
                .reset_index(drop=True)
            )

            if not linked.empty:

                daily_total = pd.to_numeric(
                    linked[selected],
                    errors="coerce"
                )

                daily_price = pd.to_numeric(
                    linked[price_col],
                    errors="coerce"
                )

                daily_income = pd.to_numeric(
                    linked[income_col],
                    errors="coerce"
                )

                valid = (
                    daily_total.notna()
                    & daily_price.notna()
                    & daily_income.notna()
                )

                linked = linked.loc[
                    valid
                ].reset_index(drop=True)

                daily_total = daily_total.loc[
                    valid
                ].reset_index(drop=True)

                daily_price = daily_price.loc[
                    valid
                ].reset_index(drop=True)

                daily_income = daily_income.loc[
                    valid
                ].reset_index(drop=True)

                if not linked.empty:

                    nav_entering_day = (
                        (1.0 + daily_total)
                        .cumprod()
                        .shift(1)
                        .fillna(1.0)
                    )

                    capital_gain_contribution = float(
                        (
                            nav_entering_day
                            * daily_price
                        ).sum()
                    )

                    income_contribution = float(
                        (
                            nav_entering_day
                            * daily_income
                        ).sum()
                    )

                    linked_total = (
                        capital_gain_contribution
                        + income_contribution
                    )

                    official_cumulative = float(
                        (1.0 + daily_total).prod()
                        - 1.0
                    )

                    if not np.isclose(
                        linked_total,
                        official_cumulative,
                        rtol=0.0,
                        atol=1e-12
                    ):
                        raise RuntimeError(
                            "Return-composition display "
                            "does not reconcile to official "
                            "cumulative portfolio return."
                        )

                    st.markdown(
                        "**Return Composition**"
                    )

                    composition_cols = (
                        st.columns(3)
                    )

                    composition_cols[0].metric(
                        "Capital Gains",
                        f"{capital_gain_contribution:+.2%}"
                    )

                    composition_cols[1].metric(
                        "Income",
                        f"{income_contribution:+.2%}"
                    )

                    composition_cols[2].metric(
                        "Total Return",
                        f"{official_cumulative:+.2%}"
                    )


    # ========================================================
    # PORTFOLIO ANALYTICS — FROM INCEPTION
    #
    # Calculated ONLY from official daily portfolio returns
    # stored in portfolio_history.csv.
    #
    # Benchmark analytics use STXCAP.JO on matching dates.
    # Nothing here modifies the official portfolio track record.
    # ========================================================

    RF_ANNUAL = 0.0725
    TRADING_DAYS = 252
    WEEKS_PER_YEAR = 52

    r = (
        pd.to_numeric(
            nav[selected],
            errors="coerce"
        )
        .dropna()
    )

    # --------------------------------------------
    # Safe display helpers
    # --------------------------------------------

    def pct_or_dash(x):
        return (
            "–"
            if x is None or not np.isfinite(x)
            else f"{x:+.2%}"
        )

    def num_or_dash(x):
        return (
            "–"
            if x is None or not np.isfinite(x)
            else f"{x:.2f}"
        )

    def days_or_dash(x):
        return (
            "–"
            if x is None or not np.isfinite(x)
            else f"{int(x)}"
        )

    # --------------------------------------------
    # Cumulative return
    # --------------------------------------------

    cumulative_return = (
        (1.0 + r).prod() - 1.0
        if len(r) >= 1
        else np.nan
    )

    # --------------------------------------------
    # Daily annualised volatility
    # --------------------------------------------

    annualised_vol = (
        r.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
        if len(r) >= 2
        else np.nan
    )

    # --------------------------------------------
    # Weekly annualised volatility
    #
    # Compound official daily observations into
    # calendar weeks, then annualise weekly sigma.
    # --------------------------------------------

    weekly_frame = (
        nav[["Date", selected]]
        .dropna()
        .set_index("Date")
        .sort_index()
    )

    weekly_returns = (
        (1.0 + weekly_frame[selected])
        .resample("W-FRI")
        .prod()
        - 1.0
    )

    # Do not count an empty resampled week.
    weekly_counts = (
        weekly_frame[selected]
        .resample("W-FRI")
        .count()
    )

    weekly_returns = weekly_returns[
        weekly_counts > 0
    ]

    weekly_vol = (
        weekly_returns.std(ddof=1)
        * np.sqrt(WEEKS_PER_YEAR)
        if len(weekly_returns) >= 2
        else np.nan
    )

    # --------------------------------------------
    # Drawdown and recovery
    # --------------------------------------------

    if len(r) >= 1:

        nav_index = (
            (1.0 + r)
            .cumprod()
        )

        running_peak = (
            nav_index.cummax()
        )

        drawdown = (
            nav_index / running_peak
            - 1.0
        )

        max_drawdown = (
            float(drawdown.min())
            if len(r) >= 2
            else np.nan
        )

        trough_label = (
            drawdown.idxmin()
        )

        trough_pos = (
            list(nav_index.index)
            .index(trough_label)
        )

        peak_value = float(
            running_peak.loc[
                trough_label
            ]
        )

        post_trough = (
            nav_index.iloc[
                trough_pos + 1:
            ]
        )

        recovered = (
            post_trough[
                post_trough >= peak_value
            ]
        )

        if (
            max_drawdown < 0
            and not recovered.empty
        ):

            recovery_label = (
                recovered.index[0]
            )

            recovery_pos = (
                list(nav_index.index)
                .index(recovery_label)
            )

            days_to_recover = (
                recovery_pos
                - trough_pos
            )

        else:
            days_to_recover = np.nan

    else:

        max_drawdown = np.nan
        days_to_recover = np.nan

    # --------------------------------------------
    # Maximum upside
    # Largest positive official daily return.
    # --------------------------------------------

    positive_days = r[r > 0]

    max_upside = (
        float(positive_days.max())
        if not positive_days.empty
        else np.nan
    )

    # --------------------------------------------
    # Win / Loss Ratio
    #
    # Number of positive-return days divided by
    # number of negative-return days.
    # Zero-return days are excluded.
    # --------------------------------------------

    wins = int(
        (r > 0).sum()
    )

    losses = int(
        (r < 0).sum()
    )

    win_loss_ratio = (
        wins / losses
        if losses > 0
        else np.nan
    )

    # --------------------------------------------
    # Historical 95% Weekly Value-at-Risk
    #
    # Negative of the empirical 5th percentile of
    # actual compounded weekly portfolio returns,
    # displayed as a positive loss magnitude.
    #
    # Minimum 20 weekly observations before reporting.
    # --------------------------------------------

    if len(weekly_returns) >= 20:

        var_95_raw = float(
            np.quantile(
                weekly_returns,
                0.05
            )
        )

        var_95 = max(
            0.0,
            -var_95_raw
        )

    else:

        var_95 = np.nan

    # --------------------------------------------
    # Annualised portfolio return + Sharpe
    #
    # Geometric annualisation of actual official
    # daily observations from inception.
    # --------------------------------------------

    if len(r) >= 2:

        growth = float(
            (1.0 + r).prod()
        )

        annualised_return = (
            growth ** (
                TRADING_DAYS
                / len(r)
            )
            - 1.0
        )

    else:
        annualised_return = np.nan

    sharpe = (
        (
            annualised_return
            - RF_ANNUAL
        )
        / annualised_vol
        if (
            np.isfinite(annualised_return)
            and np.isfinite(annualised_vol)
            and annualised_vol > 0
        )
        else np.nan
    )

    # --------------------------------------------
    # STXCAP benchmark for Beta / Alpha
    #
    # Downloaded independently for analytics only.
    # Does NOT alter portfolio_history.csv.
    # --------------------------------------------

    beta = np.nan
    alpha = np.nan

    if len(nav) >= 2:

        benchmark_start = (
            nav["Date"].min()
            - pd.Timedelta(days=7)
        )

        benchmark_end = (
            nav["Date"].max()
            + pd.Timedelta(days=2)
        )

        try:

            benchmark_hist = (
                yf.Ticker("STXCAP.JO")
                .history(
                    start=benchmark_start,
                    end=benchmark_end,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    repair=False
                )
            )

            if not benchmark_hist.empty:

                if benchmark_hist.index.tz is not None:
                    benchmark_hist.index = (
                        benchmark_hist.index
                        .tz_localize(None)
                    )

                benchmark_close = (
                    pd.to_numeric(
                        benchmark_hist["Close"],
                        errors="coerce"
                    )
                    .dropna()
                    .sort_index()
                )

                # Apply same validated STXCAP
                # structural price normalization.
                benchmark_close = (
                    historical_price_repair(
                        "STXCAP.JO",
                        benchmark_close
                    )
                )

                benchmark_return = (
                    benchmark_close
                    .pct_change()
                    .dropna()
                    .rename("Benchmark")
                )

                portfolio_for_beta = (
                    nav[
                        ["Date", selected]
                    ]
                    .copy()
                    .rename(
                        columns={
                            selected:
                            "Portfolio"
                        }
                    )
                    .set_index("Date")
                )

                aligned = (
                    portfolio_for_beta
                    .join(
                        benchmark_return,
                        how="inner"
                    )
                    .dropna()
                )

                # Need enough aligned observations
                # and non-zero benchmark variance.
                if (
                    len(aligned) >= 2
                    and
                    aligned["Benchmark"]
                    .var(ddof=1) > 0
                ):

                    beta = float(
                        aligned[
                            ["Portfolio",
                             "Benchmark"]
                        ]
                        .cov()
                        .loc[
                            "Portfolio",
                            "Benchmark"
                        ]
                        /
                        aligned[
                            "Benchmark"
                        ]
                        .var(ddof=1)
                    )

                    rf_daily = (
                        (1.0 + RF_ANNUAL)
                        ** (1.0 / TRADING_DAYS)
                        - 1.0
                    )

                    portfolio_excess = (
                        aligned["Portfolio"]
                        - rf_daily
                    )

                    benchmark_excess = (
                        aligned["Benchmark"]
                        - rf_daily
                    )

                    alpha_daily = float(
                        portfolio_excess.mean()
                        -
                        beta
                        * benchmark_excess.mean()
                    )

                    # Geometric annualisation of
                    # estimated daily CAPM alpha.
                    if alpha_daily > -1:

                        alpha = (
                            (1.0 + alpha_daily)
                            ** TRADING_DAYS
                            - 1.0
                        )

        except Exception:
            # Analytics must never break the
            # official portfolio dashboard.
            beta = np.nan
            alpha = np.nan

    # --------------------------------------------
    # DISPLAY
    # --------------------------------------------

    st.markdown(
        "#### Performance & Risk Since Inception"
    )

    metric_row_1 = st.columns(4)

    metric_row_1[0].metric(
        "Cumulative Return",
        pct_or_dash(cumulative_return)
    )

    metric_row_1[1].metric(
        "Annualised Volatility",
        pct_or_dash(annualised_vol)
    )

    metric_row_1[2].metric(
        "Weekly Volatility",
        pct_or_dash(weekly_vol)
    )

    metric_row_1[3].metric(
        "Maximum Drawdown",
        pct_or_dash(max_drawdown)
    )

    metric_row_2 = st.columns(4)

    metric_row_2[0].metric(
        "Maximum Upside",
        pct_or_dash(max_upside)
    )

    metric_row_2[1].metric(
        "Days to Recover",
        days_or_dash(days_to_recover)
    )

    metric_row_2[2].metric(
        "Beta vs STXCAP",
        num_or_dash(beta)
    )

    metric_row_2[3].metric(
        "Alpha",
        pct_or_dash(alpha)
    )

    metric_row_3 = st.columns(3)

    metric_row_3[0].metric(
        "Sharpe Ratio",
        num_or_dash(sharpe)
    )

    metric_row_3[1].metric(
        "Win/Loss Ratio",
        num_or_dash(win_loss_ratio)
    )

    metric_row_3[2].metric(
        "95% Weekly VaR",
        pct_or_dash(var_95)
    )

    st.caption(
        "Statistics use official portfolio observations from "
        "28 September 2026 onward. Volatility is annualised "
        "from daily or weekly returns as labelled. Beta and "
        "CAPM alpha use STXCAP as benchmark and a 7.25% "
        "annual risk-free rate."
    )


    # ========================================================
    # DAILY RETURN HISTORY
    # ========================================================

    with st.expander(
        "Daily Portfolio Return History"
    ):

        daily_display = nav.copy()

        for col in [
            "Combined",
            "Main",
            "Subset"
        ]:

            daily_display[col] = (
                daily_display[col]
                .map(
                    lambda x:
                    f"{x:+.2%}"
                )
            )

        st.dataframe(
            daily_display.sort_values(
                "Date",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TODAY'S ASSET RETURNS — VISUAL
# ============================================================

st.divider()

st.subheader(
    "Today's Individual Asset Performance"
)


if not live.empty:

    chart_data = (
        live[
            [
                "Holding",
                "Bucket",
                "Today_Return"
            ]
        ]
        .copy()
        .sort_values(
            "Today_Return"
        )
    )


    fig_assets = px.bar(
        chart_data,
        x="Today_Return",
        y="Holding",
        orientation="h",
        color="Bucket",
        labels={
            "Today_Return":
                "Today's Return",
            "Holding":
                ""
        },
        hover_data={
            "Today_Return":
                ":.2%",
            "Bucket":
                True
        }
    )


    fig_assets.update_layout(
        height=700,
        xaxis_tickformat=".1%",
        legend_title_text="Portfolio",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )


    st.plotly_chart(
        fig_assets,
        use_container_width=True
    )


# ============================================================
# LIVE HOLDING TABLE
# ============================================================

with st.expander(
    "Live Holding Detail"
):

    live_display = (
        live[
            [
                "Holding",
                "Bucket",
                "Today_Return",
                "Current_Value",
                "PnL",
                "Last_Update"
            ]
        ]
        .copy()
    )


    live_display[
        "Today"
    ] = (
        live_display[
            "Today_Return"
        ]
        .map(
            lambda x:
            f"{x:+.2%}"
        )
    )


    live_display[
        "Value"
    ] = (
        live_display[
            "Current_Value"
        ]
        .map(
            lambda x:
            f"R{x:,.0f}"
        )
    )


    live_display[
        "P&L"
    ] = (
        live_display[
            "PnL"
        ]
        .map(
            lambda x:
            f"R{x:+,.0f}"
        )
    )


    live_display[
        "Updated"
    ] = (
        live_display[
            "Last_Update"
        ]
        .map(
            lambda x:
            x.strftime(
                "%H:%M"
            )
        )
    )


    st.dataframe(
        live_display[
            [
                "Holding",
                "Bucket",
                "Today",
                "Value",
                "P&L",
                "Updated"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INDIVIDUAL SECURITY HISTORICAL RETURNS
#
# IMPORTANT:
# COMPLETELY SEPARATE FROM PORTFOLIO PERFORMANCE.
# ============================================================

st.divider()

st.subheader(
    "Individual Asset Historical Returns"
)

st.markdown(
    """
    <div class="section-note">
    Reference information for individual securities only.
    These figures are not historical portfolio returns and
    are not used to construct the portfolio track record.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HISTORICAL PERIOD HELPERS — CONTROLLED NON-REINVESTED
#
# Return methodology:
#
#     (P1 - P0 + cumulative cash distributions) / P0
#
# Cash distributions are NOT reinvested.
# Raw Close is used; Yahoo Adjusted Close is NOT used.
# ============================================================

@st.cache_data(ttl=86400)
def download_historical_reference(ticker):

    hist = yf.Ticker(ticker).history(
        start="2020-09-01",
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=False
    ).sort_index()

    if hist.empty:
        return pd.DataFrame()

    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)

    return hist



def repair_100x_regimes(series, ticker):

    s = series.copy().astype(float)

    if len(s) < 2:
        return s

    scale = 1.0
    repaired = s.copy()

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


def historical_price_repair(ticker, series):

    s = (
        pd.to_numeric(series, errors="coerce")
        .astype(float)
        .copy()
    )

    # --------------------------------------------------------
    # STXCAP STRUCTURAL HISTORICAL REGIME
    # Validated in historical return audit:
    # pre-18 May 2026 observations require x100 normalization.
    # --------------------------------------------------------

    if ticker == "STXCAP.JO":

        mask = (
            s.index <
            pd.Timestamp("2026-05-18")
        )

        s.loc[mask] = (
            s.loc[mask] * 100.0
        )

    # --------------------------------------------------------
    # Other isolated/regime x100 errors.
    # Uses the same transition logic as the validated audit.
    # --------------------------------------------------------

    s = repair_100x_regimes(s, ticker)

    return s


def clean_historical_distributions(
    ticker,
    dividends
):

    d = (
        pd.to_numeric(
            dividends,
            errors="coerce"
        )
        .fillna(0.0)
    )

    d = d[d > 0].copy()

    if d.empty:
        return d

    # --------------------------------------------------------
    # Validated duplicate distribution records.
    #
    # Yahoo contains pairs with identical cash amounts where
    # one date is an artificial month-end record and the other
    # is the actual distribution record.
    #
    # Remove ONLY the explicitly validated duplicates.
    # --------------------------------------------------------

    duplicate_dates = {

        "STXCAP.JO": [
            "2021-12-31",
            "2022-02-28",
            "2022-04-29",
            "2022-06-30",
            "2022-08-31",
            "2022-10-31",
            "2022-12-30",
            "2023-02-28",
            "2023-04-28",
            "2023-08-31",
            "2023-10-31",
            "2023-12-29",
            "2024-02-29",
            "2024-04-30",
            "2024-06-28",
            "2024-08-30",
            "2024-10-31",
        ],

        "STXGBD.JO": [
            "2021-09-30",
            "2022-03-31",
            "2022-09-30",
            "2023-03-31",
            "2024-03-28",
            "2024-09-30",
        ],

        "STXGOV.JO": [
            "2022-06-30",
            "2022-09-30",
            "2022-12-30",
            "2023-03-31",
            "2023-06-30",
            "2023-09-29",
            "2023-12-29",
            "2024-03-28",
            "2024-06-28",
            "2024-09-30",
        ],
    }

    for dt in duplicate_dates.get(
        ticker,
        []
    ):

        timestamp = pd.Timestamp(dt)

        if timestamp in d.index:
            d = d.drop(timestamp)

    return d


def non_reinvested_endpoint_return(
    ticker,
    hist,
    target_start,
    target_end,
    tolerance=10
):

    if hist.empty or "Close" not in hist.columns:
        return np.nan

    close = historical_price_repair(
        ticker,
        hist["Close"]
    )

    close = (
        close
        .dropna()
        .sort_index()
    )

    if close.empty:
        return np.nan

    # --------------------------------------------------------
    # Require actual history around requested start.
    # This prevents short-history securities from being given
    # artificial 3Y/5Y results.
    # --------------------------------------------------------

    starts = close[
        (close.index >= target_start) &
        (
            close.index <=
            target_start +
            pd.Timedelta(days=tolerance)
        )
    ]

    ends = close[
        close.index <= target_end
    ]

    if starts.empty or ends.empty:
        return np.nan

    start_date = starts.index[0]
    end_date = ends.index[-1]

    if end_date <= start_date:
        return np.nan

    p0 = float(
        close.loc[start_date]
    )

    p1 = float(
        close.loc[end_date]
    )

    if (
        not np.isfinite(p0)
        or not np.isfinite(p1)
        or p0 <= 0
        or p1 <= 0
    ):
        return np.nan

    if "Dividends" in hist.columns:

        dividends = (
            clean_historical_distributions(
                ticker,
                hist["Dividends"]
            )
        )

        cash = float(
            dividends[
                (dividends.index > start_date) &
                (dividends.index <= end_date)
            ].sum()
        )

    else:
        cash = 0.0

    return (
        p1 - p0 + cash
    ) / p0


# ============================================================
# HISTORICAL END DATE
# ============================================================

hist_end = pd.Timestamp(
    adj_close.index.max()
).normalize()


periods = {

    "M/M":
        hist_end.replace(day=1),

    "YTD":
        pd.Timestamp(
            hist_end.year,
            1,
            1
        ),

    "1Y":
        hist_end -
        pd.DateOffset(years=1),

    "3Y":
        hist_end -
        pd.DateOffset(years=3),

    "5Y":
        hist_end -
        pd.DateOffset(years=5),
}


# ============================================================
# BHETNC HISTORICAL REFERENCE
#
# Yahoo BHETNC.JO history is unavailable before portfolio
# inception, so retain the established Berkshire × USD/ZAR
# proxy less the 1.00% annual ETN fee.
# ============================================================

@st.cache_data(ttl=86400)
def download_bhetnc_inputs():

    return yf.download(
        [
            "BRK-B",
            "USDZAR=X"
        ],
        start="2020-09-01",
        interval="1d",
        auto_adjust=False,
        progress=False
    )


bh_inputs = download_bhetnc_inputs()

if isinstance(
    bh_inputs.columns,
    pd.MultiIndex
):

    bh_adj = (
        bh_inputs["Adj Close"]
        .copy()
    )

else:

    bh_adj = pd.DataFrame()


def get_proxy_endpoint(
    s,
    target,
    tolerance=10
):

    s = s.dropna().sort_index()

    x = s[
        (s.index >= target) &
        (
            s.index <=
            target +
            pd.Timedelta(days=tolerance)
        )
    ]

    if x.empty:
        return None

    return (
        x.index[0],
        float(x.iloc[0])
    )


def bhetnc_return(start, end):

    if bh_adj.empty:
        return np.nan

    bs = get_proxy_endpoint(
        bh_adj["BRK-B"],
        start
    )

    fs = get_proxy_endpoint(
        bh_adj["USDZAR=X"],
        start
    )

    be_candidates = (
        bh_adj["BRK-B"]
        .dropna()
        .loc[:end]
    )

    fe_candidates = (
        bh_adj["USDZAR=X"]
        .dropna()
        .loc[:end]
    )

    if (
        bs is None
        or fs is None
        or be_candidates.empty
        or fe_candidates.empty
    ):
        return np.nan

    be = (
        be_candidates.index[-1],
        float(be_candidates.iloc[-1])
    )

    fe = (
        fe_candidates.index[-1],
        float(fe_candidates.iloc[-1])
    )

    years = (
        (
            max(be[0], fe[0])
            -
            max(bs[0], fs[0])
        ).days
        /
        365.25
    )

    if years <= 0:
        return np.nan

    factor = (
        (be[1] / bs[1])
        *
        (fe[1] / fs[1])
        *
        ((1 - 0.01) ** years)
    )

    return factor - 1


# ============================================================
# BUILD INDIVIDUAL-ASSET REFERENCE TABLE
# ============================================================

reference_rows = []

for holding, row in master.iterrows():

    ticker = row["ticker"]

    result = {
        "Asset": holding,
        "Portfolio": row["bucket"]
    }

    if ticker != "BHETNC.JO":

        hist = download_historical_reference(
            ticker
        )

    else:

        hist = pd.DataFrame()

    for period, period_start in periods.items():

        if ticker == "BHETNC.JO":

            result[period] = (
                bhetnc_return(
                    period_start,
                    hist_end
                )
            )

        else:

            result[period] = (
                non_reinvested_endpoint_return(
                    ticker,
                    hist,
                    period_start,
                    hist_end
                )
            )

    reference_rows.append(result)


reference = pd.DataFrame(
    reference_rows
)


# ============================================================
# REFERENCE TABLE DISPLAY
# ============================================================

reference_display = reference.copy()

for col in [
    "M/M",
    "YTD",
    "1Y",
    "3Y",
    "5Y"
]:

    reference_display[col] = (
        reference_display[col]
        .map(
            lambda x:
            "–"
            if pd.isna(x)
            else f"{x:+.2%}"
        )
    )


st.dataframe(
    reference_display,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Historical individual-security returns use raw price "
    "change plus cumulative cash distributions. Cash "
    "distributions are not reinvested. A dash indicates "
    "insufficient history for the requested period. "
    "BHETNC historical reference returns are reconstructed "
    "from Berkshire Hathaway Class B × USD/ZAR less the "
    "1.00% annual fee."
)



# ============================================================
# UPCOMING DIVIDENDS + PORTFOLIO NEWS
#
# Yahoo calendar/news information only.
# This section is informational and does not feed into any
# portfolio valuation, return, risk, or historical calculations.
# ============================================================

@st.cache_data(ttl=3600)
def get_yahoo_portfolio_updates(tickers):

    today = pd.Timestamp.now().normalize()

    dividend_rows = []
    news_rows = []

    for ticker in tickers:

        try:
            stock = yf.Ticker(ticker)

            # ------------------------------------------------
            # UPCOMING EX-DIVIDEND DATE
            # ------------------------------------------------

            try:
                calendar = stock.calendar

                if isinstance(calendar, dict):

                    ex_date = calendar.get(
                        "Ex-Dividend Date"
                    )

                    if ex_date is not None:

                        ex_date = pd.Timestamp(
                            ex_date
                        ).normalize()

                        if ex_date >= today:

                            dividend_rows.append(
                                {
                                    "Ticker": ticker,
                                    "Ex-Dividend Date": ex_date
                                }
                            )

            except Exception:
                pass

            # ------------------------------------------------
            # NEWS
            # ------------------------------------------------

            try:
                news = stock.get_news(
                    count=3,
                    tab="news"
                ) or []

                for item in news[:3]:

                    content = item.get(
                        "content",
                        {}
                    )

                    title = content.get(
                        "title"
                    )

                    pub_date = content.get(
                        "pubDate"
                    )

                    provider = (
                        content
                        .get("provider", {})
                        .get("displayName")
                    )

                    url = (
                        content
                        .get("clickThroughUrl", {})
                        .get("url")
                    )

                    if not url:
                        url = (
                            content
                            .get("canonicalUrl", {})
                            .get("url")
                        )

                    if title:

                        news_rows.append(
                            {
                                "Ticker": ticker,
                                "Published": pub_date,
                                "Source": provider,
                                "Headline": title,
                                "URL": url
                            }
                        )

            except Exception:
                pass

        except Exception:
            pass

    return (
        pd.DataFrame(dividend_rows),
        pd.DataFrame(news_rows)
    )


portfolio_tickers = (
    master["ticker"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

upcoming_dividends, portfolio_news = (
    get_yahoo_portfolio_updates(
        portfolio_tickers
    )
)


# ============================================================
# UPCOMING DIVIDEND CALENDAR
# ============================================================

st.subheader(
    "Upcoming Dividend Calendar"
)

if not upcoming_dividends.empty:

    upcoming_dividends = (
        upcoming_dividends
        .drop_duplicates()
        .sort_values("Ex-Dividend Date")
        .reset_index(drop=True)
    )

    upcoming_dividends[
        "Ex-Dividend Date"
    ] = (
        upcoming_dividends[
            "Ex-Dividend Date"
        ]
        .dt.strftime("%d %b %Y")
    )

    st.dataframe(
        upcoming_dividends,
        use_container_width=True,
        hide_index=True
    )

else:

    st.caption(
        "No upcoming ex-dividend dates are currently "
        "available from Yahoo Finance."
    )

st.caption(
    "Only future ex-dividend dates explicitly supplied by "
    "Yahoo Finance are shown. No dividend dates or amounts "
    "are estimated."
)


# ============================================================
# PORTFOLIO NEWS
# ============================================================

st.subheader(
    "Portfolio News"
)

if not portfolio_news.empty:

    portfolio_news[
        "Published"
    ] = pd.to_datetime(
        portfolio_news["Published"],
        errors="coerce",
        utc=True
    )

    portfolio_news = (
        portfolio_news
        .drop_duplicates(
            subset=["Headline"]
        )
        .sort_values(
            "Published",
            ascending=False
        )
        .reset_index(drop=True)
    )

    portfolio_news[
        "Published"
    ] = (
        portfolio_news[
            "Published"
        ]
        .dt.strftime(
            "%d %b %Y %H:%M UTC"
        )
    )

    st.dataframe(
        portfolio_news,
        use_container_width=True,
        hide_index=True,
        column_config={
            "URL":
                st.column_config.LinkColumn(
                    "Article"
                )
        }
    )

else:

    st.caption(
        "No portfolio news is currently available "
        "from Yahoo Finance."
    )

st.caption(
    "News is supplied by Yahoo Finance and its underlying "
    "publishers. Headlines are informational only."
)


# FOOTER
# ============================================================

st.divider()

st.caption(
    "Portfolio performance record inception: "
    "28 September 2026. Historical individual-security "
    "statistics are reference information only."
)
