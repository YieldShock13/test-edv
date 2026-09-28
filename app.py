
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
# ============================================================

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

    # Yahoo timestamps
    idx = s.index

    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    local_dates = idx.tz_convert(SAST).date

    unique_dates = sorted(set(local_dates))

    if len(unique_dates) < 2:
        continue

    today = unique_dates[-1]
    prior_day = unique_dates[-2]

    today_mask = local_dates == today
    prior_mask = local_dates == prior_day

    today_s = s.iloc[np.where(today_mask)[0]]
    prior_s = s.iloc[np.where(prior_mask)[0]]

    if today_s.empty or prior_s.empty:
        continue

    latest_raw = float(today_s.iloc[-1])
    prior_raw = float(prior_s.iloc[-1])

    # JSE Yahoo quotes are generally in cents.
    latest_price = latest_raw / 100
    prior_price = prior_raw / 100

    ret = latest_price / prior_price - 1

    opening_value = qty * prior_price
    current_value = qty * latest_price

    pnl = current_value - opening_value

    timestamp = idx[np.where(today_mask)[0][-1]]

    timestamp = timestamp.tz_convert(SAST)

    live_rows.append({
        "Holding": holding,
        "Ticker": ticker,
        "Bucket": bucket,
        "Qty": qty,
        "Prior_Close": prior_price,
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

st.title("Portfolio Monitor")

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
    Ordinary intraday prices versus previous market close.
    Official daily portfolio performance is recorded from
    Adjusted Close after EOD.
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
            mode="lines",
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
# HISTORICAL PERIOD HELPERS
# ============================================================

def endpoint_return(
    series,
    start,
    end,
    tolerance=10
):

    s = (
        series
        .dropna()
        .sort_index()
    )

    if s.empty:
        return np.nan

    starts = s[
        (s.index >= start) &
        (
            s.index <=
            start +
            pd.Timedelta(days=tolerance)
        )
    ]

    ends = s[
        s.index <= end
    ]

    if starts.empty or ends.empty:
        return np.nan

    start_date = starts.index[0]
    end_date = ends.index[-1]

    if end_date <= start_date:
        return np.nan

    return (
        float(ends.iloc[-1])
        /
        float(starts.iloc[0])
        - 1
    )


hist_end = adj_close.index.max().normalize()

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
        pd.DateOffset(years=5)
}


# ============================================================
# BHETNC HISTORICAL ENDPOINT PROXY
# ============================================================

@st.cache_data(ttl=86400)
def download_bhetnc_inputs():

    x = yf.download(
        [
            "BRK-B",
            "USDZAR=X"
        ],
        start="2020-11-20",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    return x


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


def valid_fx_endpoint(
    s,
    date,
    threshold=0.10
):

    s = s.dropna().sort_index()

    if date not in s.index:
        return False

    pos = s.index.get_loc(date)

    if (
        pos == 0
        or
        pos == len(s) - 1
    ):
        return True

    value = float(s.iloc[pos])
    prev = float(s.iloc[pos - 1])
    nxt = float(s.iloc[pos + 1])

    return not (
        abs(value / prev - 1) > threshold
        and
        abs(value / nxt - 1) > threshold
    )


def get_valid_endpoint(
    s,
    target,
    tolerance=10,
    validate_fx=False
):

    s = s.dropna().sort_index()

    candidates = s[
        (s.index >= target) &
        (
            s.index <=
            target +
            pd.Timedelta(days=tolerance)
        )
    ]

    for dt, value in candidates.items():

        if (
            validate_fx
            and
            not valid_fx_endpoint(
                s,
                dt
            )
        ):
            continue

        return dt, float(value)

    return None


def bhetnc_return(start, end):

    if bh_adj.empty:
        return np.nan

    brk = bh_adj["BRK-B"]
    fx = bh_adj["USDZAR=X"]

    bs = get_valid_endpoint(
        brk,
        start
    )

    fs = get_valid_endpoint(
        fx,
        start,
        validate_fx=True
    )

    be = get_valid_endpoint(
        brk,
        end
    )

    fe = get_valid_endpoint(
        fx,
        end,
        validate_fx=True
    )

    if any(
        x is None
        for x in [
            bs,
            fs,
            be,
            fe
        ]
    ):
        return np.nan

    if (
        abs(
            (bs[0] - fs[0]).days
        ) > 3
        or
        abs(
            (be[0] - fe[0]).days
        ) > 3
    ):
        return np.nan

    years = (
        (
            max(be[0], fe[0])
            -
            max(bs[0], fs[0])
        ).days
        /
        365.25
    )

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

    for period, start in periods.items():

        if ticker == "BHETNC.JO":

            result[period] = (
                bhetnc_return(
                    start,
                    hist_end
                )
            )

        elif ticker in adj_close.columns:

            result[period] = (
                endpoint_return(
                    adj_close[ticker],
                    start,
                    hist_end
                )
            )

        else:

            result[period] = np.nan

    reference_rows.append(result)


reference = pd.DataFrame(
    reference_rows
)


# ============================================================
# REFERENCE TABLE DISPLAY
# ============================================================

reference_display = (
    reference.copy()
)


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
    "BHETNC historical reference returns are reconstructed "
    "from Berkshire Hathaway Class B × USD/ZAR less the "
    "1.00% annual fee. Current/intraday BHETNC performance "
    "uses the actual BHETNC.JO market instrument."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Portfolio performance record inception: "
    "28 September 2026. Historical individual-security "
    "statistics are reference information only."
)
