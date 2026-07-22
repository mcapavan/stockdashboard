import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Portfolio Strategy Dashboard", layout="wide")

# Collects caught exceptions so they're surfaced in the UI instead of
# silently swallowed. Reset each script run; cached functions won't
# re-append on a cache hit, so this only reflects fresh fetches.
DATA_WARNINGS = []


def record_warning(source, ticker, err):
    DATA_WARNINGS.append(f"**{source}** ({ticker}): {err}")

# =====================================================================
# DATA FETCHING
# =====================================================================

@st.cache_data(ttl=3600)
def get_data(ticker, period="2y"):
    """Pull daily OHLCV. Retries a few times before giving up.

    Why the retry matters: Yahoo Finance / yfinance occasionally fails
    on the very first request from a freshly-started process (cold-start
    session/rate-limit hiccup - common on cloud hosts). Without a retry,
    that one blip gets cached by @st.cache_data as a permanent `None`
    for the full ttl, since Streamlit caches whatever the function
    returns - including a failure. That's the likely reason the
    DEFAULT ticker (loaded automatically on boot, so it's the one most
    exposed to a cold-start blip) can fail while every other ticker
    (fetched later, after the app has warmed up) works fine.
    """
    df, last_error = None, None
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_error = e
        time.sleep(1.5 * (attempt + 1))

    if df is None or df.empty:
        record_warning("get_data", ticker, last_error or "empty response after 3 attempts")
        return None

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip() for col in df.columns]

    # Defensive check: if Yahoo ever returns a malformed/partial
    # response, fail cleanly here rather than crashing deeper in the
    # indicator pipeline.
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        record_warning("get_data", ticker, f"missing expected columns: got {list(df.columns)}")
        return None

    return df


@st.cache_data(ttl=3600)
def get_ticker_info(ticker):
    """Consolidated fetch for currency, company name, and analyst target.

    These were previously three separate cached functions, each
    independently calling yf.Ticker(ticker).info for the SAME ticker -
    tripling requests to Yahoo's most rate-limit-sensitive endpoint for
    no benefit. One retry-backed fetch here instead of three bare ones.
    """
    result = {"currency_symbol": "$", "company_name": ticker, "analyst_target": None}
    info, last_error = None, None
    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).info
            if info:
                break
        except Exception as e:
            last_error = e
        time.sleep(1.5 * (attempt + 1))

    if not info:
        record_warning("get_ticker_info", ticker, last_error or "empty info after 3 attempts")
        return result

    currency = info.get('currency') or 'USD'
    symbol_map = {'USD': '$', 'INR': '₹', 'EUR': '€', 'GBP': '£', 'JPY': '¥'}
    result["currency_symbol"] = symbol_map.get(currency, currency + ' ')
    result["company_name"] = info.get('longName', ticker)

    try:
        targets = yf.Ticker(ticker).analyst_price_targets
        if isinstance(targets, dict):
            result["analyst_target"] = targets.get("mean")
    except Exception as e:
        record_warning("get_ticker_info(analyst_target)", ticker, e)

    return result


def compute_latest_snapshot(ticker, rsi_low, rsi_high, vol_confirm_mult,
                             momentum_vol_mult, momentum_rsi_cap,
                             atr_stop_mult, atr_target_mult):
    """Latest-day signal snapshot for one ticker, using the same rules
    as the main single-stock view. Used by Portfolio Scan to check many
    tickers at once without duplicating the full historical engine."""
    df = get_data(ticker)
    if df is None or len(df) < 60:
        return None
    df = df.copy()
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['ATR_10'] = ta.atr(df['High'], df['Low'], df['Close'], length=10, mamode="sma")
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['Vol_Avg_30'] = df['Volume'].shift(1).rolling(window=30).mean()
    df['Price Change'] = df['Close'].pct_change() * 100
    df = df.dropna(subset=['EMA_20', 'ATR_10', 'RSI_14'])
    if df.empty:
        return None

    row = df.iloc[-1]
    close, ema, open_ = row['Close'], row['EMA_20'], row['Open']
    rsi, vol, avg_vol, atr = row['RSI_14'], row['Volume'], row['Vol_Avg_30'], row['ATR_10']

    signal, conviction = "😴 WAIT", 10
    if (close < ema and rsi_low <= rsi <= rsi_high
            and pd.notna(avg_vol) and vol > (avg_vol * vol_confirm_mult)
            and close > (open_ * 0.97)):
        signal = "💎 VALUE BUY"
        ema_gap_pct = ((ema - close) / ema) * 100 if ema else 0
        vol_ratio = vol / avg_vol if avg_vol else 1
        conviction = int(np.clip(50 + ema_gap_pct * 5 + (vol_ratio - 1) * 20 + (rsi_high - rsi), 0, 100))
    elif (close > ema and pd.notna(avg_vol) and vol > (avg_vol * momentum_vol_mult) and rsi < momentum_rsi_cap):
        signal = "🚀 MOMENTUM BUY"
        trend_gap_pct = ((close - ema) / ema) * 100 if ema else 0
        vol_ratio = vol / avg_vol if avg_vol else 1
        overbought_penalty = max(0, rsi - (momentum_rsi_cap - 10)) * 2
        conviction = int(np.clip(50 + trend_gap_pct * 3 + (vol_ratio - 1) * 20 - overbought_penalty, 0, 100))

    return {
        "Ticker": ticker,
        "Price": round(float(close), 2),
        "Chg %": round(float(row['Price Change']), 2) if pd.notna(row['Price Change']) else 0.0,
        "RSI": round(float(rsi), 1),
        "Signal": signal,
        "Conviction": conviction,
        "Stop": round(float(close - atr_stop_mult * atr), 2),
        "Target": round(float(close + atr_target_mult * atr), 2),
    }


# =====================================================================
# SIDEBAR: TICKER SELECTION
# =====================================================================

st.sidebar.header("🕹️ Strategy Controls")

@st.cache_data(ttl=300)
def load_portfolio_tiers(csv_path="tickers.csv"):
    """Loads the ticker universe from a CSV sitting alongside app.py, so
    adding/removing/editing tickers or whole tiers is just an edit to that
    file - no code changes needed. Expected columns: Market (US/India),
    Tier (plain label, e.g. 'Tier 1: Core Compounders' or
    'Other / Watchlist'), Name (display label), Ticker (yfinance symbol,
    include .NS for India). Cached for 5 minutes so local edits during
    development show up without a full app restart.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(
            f"Could not find `{csv_path}`. Make sure tickers.csv is committed "
            "in the same folder as app.py."
        )
        return {}

    tiers = {}
    for _, row in df.iterrows():
        market = str(row["Market"]).strip()
        flag = "🇺🇸 US " if market.upper() == "US" else "🇮🇳 India "
        key = f"{flag}{str(row['Tier']).strip()}"
        tiers.setdefault(key, {})[str(row["Name"]).strip()] = str(row["Ticker"]).strip()
    return tiers


PORTFOLIO_TIERS = load_portfolio_tiers()

tier_options = ["-- Enter Manually --"] + list(PORTFOLIO_TIERS.keys())
default_tier_index = tier_options.index("🇺🇸 US Other / Watchlist")

# If a Portfolio Scan row was clicked on the previous run, apply that
# navigation request BEFORE creating the widgets below, so this run
# opens directly on the clicked ticker.
if "pending_jump_tier" in st.session_state:
    st.session_state["tier_select"] = st.session_state.pop("pending_jump_tier")
if "pending_jump_stock" in st.session_state:
    st.session_state["stock_select"] = st.session_state.pop("pending_jump_stock")

tier_choice = st.sidebar.selectbox(
    "Portfolio Tier",
    options=tier_options,
    index=default_tier_index,
    key="tier_select",
)

if tier_choice == "-- Enter Manually --":
    ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", value="RIVN").upper()
    st.sidebar.caption("🇺🇸 US: RIVN, TSLA, NVDA | 🇮🇳 India: INFY.NS, TCS.NS, RELIANCE.NS")
else:
    tier_stocks = PORTFOLIO_TIERS[tier_choice]
    stock_names = sorted(tier_stocks.keys())
    default_stock_index = (
        stock_names.index("Rivian Automotive (RIVN)")
        if "Rivian Automotive (RIVN)" in stock_names else 0
    )
    selected_name = st.sidebar.selectbox(
        "Select Stock", options=stock_names, index=default_stock_index, key="stock_select"
    )
    ticker_symbol = tier_stocks[selected_name]

# Auto-collapse the Portfolio Scan results table the moment the active
# ticker changes - whether that change came from clicking a scan row or
# from manually picking a different ticker in the dropdowns above.
if st.session_state.get("last_ticker_symbol") is not None and st.session_state["last_ticker_symbol"] != ticker_symbol:
    st.session_state["scan_expanded_us"] = False
    st.session_state["scan_expanded_india"] = False
st.session_state["last_ticker_symbol"] = ticker_symbol

st.sidebar.markdown("---")

# --- Strategy parameters, exposed so you can tune without editing code ---
with st.sidebar.expander("⚙️ Signal Thresholds", expanded=False):
    rsi_low = st.slider("Value Buy: RSI lower bound (avoid falling knives)", 10, 40, 25)
    rsi_high = st.slider("Value Buy: RSI upper bound (must be genuinely soft)", 30, 50, 40)
    vol_confirm_mult = st.slider("Value Buy: min volume vs 30d avg", 0.8, 2.0, 1.1, 0.1)
    momentum_vol_mult = st.slider("Momentum Buy: min volume vs 30d avg", 1.0, 2.5, 1.3, 0.1)
    momentum_rsi_cap = st.slider("Momentum Buy: max RSI (avoid buying blow-offs)", 60, 85, 75)
    atr_stop_mult = st.slider("Stop: ATR multiplier", 1.0, 5.0, 3.0, 0.5)
    atr_target_mult = st.slider("Target: ATR multiplier", 2.0, 10.0, 6.0, 0.5)

st.sidebar.markdown("---")

def scan_row_styler(row):
    if "VALUE" in str(row.Signal):
        return ['background-color: rgba(46, 204, 113, 0.2)'] * len(row)
    if "MOMENTUM" in str(row.Signal):
        return ['background-color: rgba(52, 152, 219, 0.2)'] * len(row)
    return [''] * len(row)


def render_portfolio_scan(section_id, section_title, button_label, tier_names):
    """Renders one market's Portfolio Scan: button + results table + click-to-navigate.
    section_id must be unique per market (e.g. 'us', 'india') - it's used to
    namespace all the session_state keys and widget keys so the two scans
    don't collide with each other."""
    cache_key = f"scan_df_cache_{section_id}"
    lookup_key = f"scan_ticker_lookup_{section_id}"
    expanded_key = f"scan_expanded_{section_id}"
    table_widget_key = f"scan_table_select_{section_id}"
    clear_flag_key = f"clear_scan_selection_{section_id}"

    total_tickers = sum(len(PORTFOLIO_TIERS.get(t, {})) for t in tier_names)
    last_scan = st.session_state.get(f"last_scan_time_{section_id}", "not run yet")

    st.subheader(section_title)
    st.caption(f"📦 {total_tickers} tickers tracked · 🕒 Last scan: {last_scan}")
    run_scan = st.button(button_label, key=f"run_scan_btn_{section_id}")

    if run_scan:
        st.session_state[expanded_key] = True
        st.session_state[f"last_scan_time_{section_id}"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_stocks = {}
        ticker_lookup = {}  # ticker -> (tier_name, display_name), for click-to-navigate
        for tier_name in tier_names:
            short_tier = tier_name.replace("🇺🇸 US ", "").replace("🇮🇳 India ", "")
            for name, tkr in PORTFOLIO_TIERS[tier_name].items():
                all_stocks[name] = (tkr, short_tier)
                ticker_lookup[tkr] = (tier_name, name)
        st.session_state[lookup_key] = ticker_lookup

        scan_rows = []
        progress = st.progress(0.0, text="Scanning tickers...")
        tickers_list = list(all_stocks.items())
        for i, (name, (tkr, short_tier)) in enumerate(tickers_list):
            snap = compute_latest_snapshot(
                tkr, rsi_low, rsi_high, vol_confirm_mult,
                momentum_vol_mult, momentum_rsi_cap, atr_stop_mult, atr_target_mult
            )
            if snap:
                snap["Name"] = name
                snap["Tier"] = short_tier
                scan_rows.append(snap)
            progress.progress((i + 1) / len(tickers_list), text=f"Scanning tickers... ({tkr})")
        progress.empty()

        if scan_rows:
            scan_df = pd.DataFrame(scan_rows).sort_values("Conviction", ascending=False).reset_index(drop=True)
            scan_df = scan_df[["Name", "Tier", "Ticker", "Price", "Chg %", "RSI", "Signal", "Conviction", "Stop", "Target"]]
            st.session_state[cache_key] = scan_df

            n_value = (scan_df['Signal'] == "💎 VALUE BUY").sum()
            n_momentum = (scan_df['Signal'] == "🚀 MOMENTUM BUY").sum()
            if n_value or n_momentum:
                st.success(f"💎 {n_value} Value Buy · 🚀 {n_momentum} Momentum Buy flagged today.")
            else:
                st.info("No BUY signals across this portfolio today.")
        else:
            st.warning("No data returned for any ticker in this scan - check network/ticker validity.")

    # Render from cache (not just inside `if run_scan`) so the table and its
    # click-to-navigate behaviour survive the rerun triggered by a row click.
    if cache_key in st.session_state:
        scan_df = st.session_state[cache_key]
        ticker_lookup = st.session_state.get(lookup_key, {})

        # A row selection made on a previous run to trigger navigation needs
        # to be cleared here, BEFORE the widget is recreated - otherwise the
        # checkbox stays ticked forever and fights any later manual change
        # to the "Select Stock" dropdown.
        if st.session_state.pop(clear_flag_key, False):
            st.session_state[table_widget_key] = {"selection": {"rows": [], "columns": []}}

        with st.expander("📋 Scan Results", expanded=st.session_state.get(expanded_key, True)):
            st.caption("👉 Click any row to jump straight to that ticker's full analysis below.")
            display_cols = [c for c in scan_df.columns if c != "Ticker"]
            scan_event = st.dataframe(
                scan_df.style.apply(scan_row_styler, axis=1).format({
                    'Price': '{:.2f}', 'Chg %': '{:+.2f}%', 'RSI': '{:.1f}',
                    'Stop': '{:.2f}', 'Target': '{:.2f}',
                }),
                use_container_width=True, hide_index=True, column_order=display_cols,
                on_select="rerun", selection_mode="single-row", key=table_widget_key,
            )

        selected_rows = []
        if scan_event and hasattr(scan_event, "selection"):
            selected_rows = scan_event.selection.get("rows", [])
        elif isinstance(scan_event, dict):
            selected_rows = scan_event.get("selection", {}).get("rows", [])

        if selected_rows:
            clicked_ticker = scan_df.iloc[selected_rows[0]]["Ticker"]
            if clicked_ticker in ticker_lookup:
                jump_tier, jump_name = ticker_lookup[clicked_ticker]
                # Only trigger a jump if this isn't already the selected ticker
                # (avoids an infinite rerun loop once the click has been applied).
                if st.session_state.get("tier_select") != jump_tier or st.session_state.get("stock_select") != jump_name:
                    st.session_state["pending_jump_tier"] = jump_tier
                    st.session_state["pending_jump_stock"] = jump_name
                    # Schedule the checkbox to be cleared on the very next run,
                    # once it's done its job of triggering this navigation.
                    st.session_state[clear_flag_key] = True
                    st.rerun()


# =====================================================================
# APP HEADER
# =====================================================================
st.title("📊 Portfolio Strategy Dashboard")
st.caption(
    "DCA / scale-in signal scanning across your US and India tiers, plus a full "
    "single-stock deep dive with backtesting and risk-level tracking."
)
st.markdown("---")

# =====================================================================
# PORTFOLIO SCAN - US and India run as two fully independent sections
# =====================================================================
st.header("🔍 Portfolio Scans")
st.caption(
    "Today's signal across your portfolio, using the same thresholds set in the "
    "sidebar. Use this to see which names are flashing a signal before drilling "
    "into any single ticker below - click any result row to jump straight there."
)

render_portfolio_scan(
    section_id="us",
    section_title="🇺🇸 US Portfolio Scan — Tier 1 / 2 / 3",
    button_label="🔍 Run US Portfolio Scan",
    tier_names=["🇺🇸 US Tier 1: Core Compounders", "🇺🇸 US Tier 2: Structural Growth", "🇺🇸 US Tier 3: Higher-Risk / Upside"],
)

st.markdown("---")

render_portfolio_scan(
    section_id="india",
    section_title="🇮🇳 India Portfolio Scan — Tier 1 / 2 / 3 / 4 / 5",
    button_label="🔍 Run India Portfolio Scan",
    tier_names=[
        "🇮🇳 India Tier 1: Core Compounders", "🇮🇳 India Tier 2: Structural Growth",
        "🇮🇳 India Tier 3: Tactical / Cyclical", "🇮🇳 India Tier 4: VRS New Recommendations",
        "🇮🇳 India Tier 5: Green Energy",
    ],
)

st.markdown("---")
st.header("📌 Single Stock Deep Dive")
st.caption("Full signal history, chart, backtest, and tranche tracking for the ticker selected in the sidebar.")

data = get_data(ticker_symbol)
ticker_info = get_ticker_info(ticker_symbol)
currency_symbol = ticker_info["currency_symbol"]

if data is not None and len(data) >= 60:

    # =================================================================
    # 1. INDICATORS
    # =================================================================
    data['EMA_20'] = ta.ema(data['Close'], length=20)
    data['ATR_10'] = ta.atr(data['High'], data['Low'], data['Close'], length=10, mamode="sma")
    data['RSI_14'] = ta.rsi(data['Close'], length=14)
    # shift(1): compare today's volume against the PRECEDING 30 days,
    # not an average that includes today's own (possibly spiking)
    # volume - otherwise a genuine spike partially masks itself.
    data['Vol_Avg_30'] = data['Volume'].shift(1).rolling(window=30).mean()
    data['Price Change'] = data['Close'].pct_change() * 100
    data['VWAP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data = data.dropna(subset=['EMA_20', 'ATR_10', 'RSI_14']).copy()

    # Per-condition diagnostics for VALUE BUY - lets you see WHICH
    # condition is the bottleneck for a given ticker, rather than just
    # seeing "no signal" and guessing why.
    data['cond_below_ema'] = data['Close'] < data['EMA_20']
    data['cond_rsi_band'] = (data['RSI_14'] >= rsi_low) & (data['RSI_14'] <= rsi_high)
    data['cond_vol_confirm'] = data['Volume'] > (data['Vol_Avg_30'] * vol_confirm_mult)
    data['cond_no_crash'] = data['Close'] > (data['Open'] * 0.97)

    temp_latest = data.iloc[-1]
    current_market_price = float(temp_latest['Close'])
    current_atr = float(temp_latest['ATR_10'])

    # =================================================================
    # 2. RISK-REWARD CALCULATOR (unchanged logic, still useful standalone)
    # =================================================================
    with st.sidebar.expander("🧮 Interactive Risk-Reward Calculator", expanded=False):
        calc_entry = st.number_input(
            f"Hypothetical Entry Price ({currency_symbol})",
            value=round(current_market_price, 2), step=0.1
        )
        calc_size = st.number_input("Position Size (Shares)", value=100, step=10)

        risk_mode = st.radio("Stop Loss Metric", ["ATR Multiplier", "Percentage Drop"])
        if risk_mode == "ATR Multiplier":
            atr_mult = st.slider("ATR Multiplier (Risk)", 1.0, 5.0, 3.0, 0.5)
            calculated_risk_per_share = current_atr * atr_mult
        else:
            pct_drop = st.slider("Percent Risk (%)", 1.0, 20.0, 5.0, 0.5)
            calculated_risk_per_share = calc_entry * (pct_drop / 100.0)
        calc_stop = calc_entry - calculated_risk_per_share

        rr_ratio = st.slider("Target Risk-Reward Ratio (R:R)", 1.0, 5.0, 2.0, 0.5)
        calc_target = calc_entry + (calculated_risk_per_share * rr_ratio)

        total_cost = calc_entry * calc_size
        total_risk = calculated_risk_per_share * calc_size
        total_reward = (calc_target - calc_entry) * calc_size

        st.markdown("**Calculator Output:**")
        st.info(
            f"🛑 **Suggested Stop:** {currency_symbol}{calc_stop:.2f}\n\n"
            f"🎯 **Suggested Target:** {currency_symbol}{calc_target:.2f}"
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Total Risk", f"{currency_symbol}{total_risk:.2f}", delta_color="inverse")
        with col_c2:
            st.metric("Total Reward", f"{currency_symbol}{total_reward:.2f}")
        st.caption(f"Total Capital Exposure: {currency_symbol}{total_cost:,.2f}")

    st.sidebar.markdown("---")

    # =================================================================
    # 3. SIGNAL ENGINE
    #
    # Two passes:
    #  Pass 1 - identify raw entry conditions per row (Value Buy /
    #           Momentum Buy) purely from indicators. This does NOT
    #           depend on any stop/target level.
    #  Pass 2 - walk the rows in order tracking a single hypothetical
    #           position. Stop/Target are FIXED at the point of entry
    #           (from that day's Close and ATR) and held constant until
    #           hit - not recalculated every row against themselves,
    #           which was the bug in the original version (Stop/Target
    #           were derived from and compared against the same row,
    #           so they almost never fired).
    # =================================================================

    def raw_entry_signal(row):
        close, ema, open_ = row['Close'], row['EMA_20'], row['Open']
        rsi, vol, avg_vol = row['RSI_14'], row['Volume'], row['Vol_Avg_30']

        # VALUE BUY: price below trend, RSI genuinely soft but not a
        # falling knife, volume ABOVE average (real participation on
        # the dip, not just "not dead"), no single-day crash.
        if (close < ema
                and rsi_low <= rsi <= rsi_high
                and vol > (avg_vol * vol_confirm_mult)
                and close > (open_ * 0.97)):
            return "VALUE BUY", "💎"

        # MOMENTUM BUY: price above trend with a volume thrust, but
        # capped RSI so we're not chasing an already-extended move.
        if (close > ema
                and vol > (avg_vol * momentum_vol_mult)
                and rsi < momentum_rsi_cap):
            return "MOMENTUM BUY", "🚀"

        return None, None

    raw_signals = data.apply(raw_entry_signal, axis=1, result_type='expand')
    data['RawSignal'] = raw_signals[0]
    data['RawIcon'] = raw_signals[1]

    signals, reasons, convictions, stops, targets = [], [], [], [], []

    for _, row in data.iterrows():
        close, ema = row['Close'], row['EMA_20']
        rsi, vol, avg_vol, atr = row['RSI_14'], row['Volume'], row['Vol_Avg_30'], row['ATR_10']
        raw_sig = row['RawSignal']

        if raw_sig == "VALUE BUY":
            stop = close - (atr_stop_mult * atr)
            target = close + (atr_target_mult * atr)
            ema_gap_pct = ((ema - close) / ema) * 100 if ema else 0
            vol_ratio = vol / avg_vol if avg_vol else 1
            conviction = 50 + ema_gap_pct * 5 + (vol_ratio - 1) * 20 + (rsi_high - rsi)
            signals.append("💎 VALUE BUY")
            reasons.append("Value criteria met (soft RSI + volume confirmation)")
            convictions.append(int(np.clip(conviction, 0, 100)))
            stops.append(stop)
            targets.append(target)
            continue

        if raw_sig == "MOMENTUM BUY":
            stop = close - (atr_stop_mult * atr)
            target = close + (atr_target_mult * atr)
            trend_gap_pct = ((close - ema) / ema) * 100 if ema else 0
            vol_ratio = vol / avg_vol if avg_vol else 1
            overbought_penalty = max(0, rsi - (momentum_rsi_cap - 10)) * 2
            conviction = 50 + trend_gap_pct * 3 + (vol_ratio - 1) * 20 - overbought_penalty
            signals.append("🚀 MOMENTUM BUY")
            reasons.append("Momentum criteria met (trend + volume thrust)")
            convictions.append(int(np.clip(conviction, 0, 100)))
            stops.append(stop)
            targets.append(target)
            continue

        signals.append("😴 WAIT")
        reasons.append("No entry criteria met")
        convictions.append(10)
        stops.append(np.nan)
        targets.append(np.nan)

    data['Signal'] = signals
    data['Reason'] = reasons
    data['Conviction'] = convictions
    data['Stop'] = stops
    data['Target'] = targets

    st.caption(
        "Note: every day is judged independently - this reflects a DCA / scale-in "
        "approach (buy again each time conditions are met) rather than a single "
        "buy-once, hold-until-stop trade. Stop/Target shown are suggested risk levels "
        "for that specific tranche, not a portfolio-wide position being tracked."
    )

    # =================================================================
    # TRANCHE OUTCOME TRACKING (retrospective only)
    #
    # Purely informational - this does NOT feed back into the Signal
    # column or suppress future signals (that was the old bug). For
    # every past BUY signal, independently check what happened
    # afterwards: did price ever hit that tranche's own stop or target.
    # Multiple overlapping tranches are each tracked separately.
    # =================================================================
    idx_list = list(data.index)
    tranche_records = []
    buy_signal_rows = data[data['Signal'].isin(["💎 VALUE BUY", "🚀 MOMENTUM BUY"])]

    for ts, row in buy_signal_rows.iterrows():
        i = idx_list.index(ts)
        stop, target = row['Stop'], row['Target']
        outcome, outcome_date, days_to_outcome = "Still Open", None, None
        for j in range(i + 1, len(data)):
            if data['Low'].iloc[j] <= stop:
                outcome, outcome_date, days_to_outcome = "🔴 Stopped Out", idx_list[j], j - i
                break
            if data['High'].iloc[j] >= target:
                outcome, outcome_date, days_to_outcome = "🔵 Target Hit", idx_list[j], j - i
                break
        tranche_records.append({
            "Entry Date": ts.strftime("%Y-%m-%d"),
            "Signal": row['Signal'],
            "Entry Price": round(row['Close'], 2),
            "Stop": round(stop, 2) if pd.notna(stop) else None,
            "Target": round(target, 2) if pd.notna(target) else None,
            "Outcome": outcome,
            "Outcome Date": outcome_date.strftime("%Y-%m-%d") if outcome_date is not None else "-",
            "Trading Days": days_to_outcome if days_to_outcome is not None else "-",
        })

    tranche_df = pd.DataFrame(tranche_records).sort_values("Entry Date", ascending=False) if tranche_records else pd.DataFrame()

    latest_row = data.iloc[-1]
    signal_date = data.index[-2]
    signal_text = data.iloc[-2]["Signal"]
    signal_date_str = signal_date.strftime("%d %b %Y")

    analyst_target = ticker_info["analyst_target"]
    if analyst_target is not None:
        target_upside = ((analyst_target / latest_row['Close']) - 1) * 100
    else:
        target_upside = 0

    # =================================================================
    # 4. BACKTEST: does a BUY signal actually predict anything?
    #    Simple, transparent, forward-return check - not a substitute
    #    for a proper walk-forward backtest, but far better than
    #    trusting a signal on vibes.
    #
    #    Caution baked in: consecutive signal days during the same
    #    drawdown/rally aren't independent trials - RSI staying oversold
    #    for 5 days in a row produces 5 "occurrences" that are really one
    #    underlying event. "Independent Episodes" clusters signal days
    #    that are close together (within CLUSTER_GAP trading days) into
    #    a single event, giving a more honest sense of true sample size.
    # =================================================================
    HORIZON = 20      # trading days forward
    CLUSTER_GAP = 3   # signals within this many trading days count as one episode
    entry_rows = data[data['Signal'].isin(["💎 VALUE BUY", "🚀 MOMENTUM BUY"])].copy()
    fwd_returns = {"💎 VALUE BUY": [], "🚀 MOMENTUM BUY": []}
    signal_positions = {"💎 VALUE BUY": [], "🚀 MOMENTUM BUY": []}

    idx_list = list(data.index)
    for ts, row in entry_rows.iterrows():
        i = idx_list.index(ts)
        signal_positions[row['Signal']].append(i)
        if i + HORIZON < len(data):
            fwd_ret = (data['Close'].iloc[i + HORIZON] / row['Close'] - 1) * 100
            fwd_returns[row['Signal']].append(fwd_ret)

    def count_episodes(positions, max_gap=CLUSTER_GAP):
        if not positions:
            return 0
        positions = sorted(positions)
        episodes = 1
        for prev, curr in zip(positions, positions[1:]):
            if curr - prev > max_gap:
                episodes += 1
        return episodes

    backtest_summary = []
    for sig_name, rets in fwd_returns.items():
        if rets:
            episodes = count_episodes(signal_positions[sig_name])
            backtest_summary.append({
                "Signal": sig_name,
                "Raw Occurrences": len(rets),
                "Independent Episodes": episodes,
                f"Avg {HORIZON}d Fwd Return": f"{np.mean(rets):+.2f}%",
                "Win Rate": f"{(np.array(rets) > 0).mean() * 100:.0f}%",
                "Best": f"{np.max(rets):+.2f}%",
                "Worst": f"{np.min(rets):+.2f}%",
            })

    # =================================================================
    # 5. UI HEADER
    # =================================================================
    st.subheader(f"📊 {ticker_info['company_name']} ({ticker_symbol})")

    if DATA_WARNINGS:
        with st.expander(f"⚠️ {len(DATA_WARNINGS)} data fetch warning(s) - click to view"):
            for w in DATA_WARNINGS:
                st.write(w)

    st.markdown(f"### 📌 Signal ({signal_date_str}): {signal_text}")

    # Simplified stance: purely price/trend/analyst based now that
    # Prophet forecast and news-sentiment scoring have been removed -
    # both added noise (Prophet's yearly seasonality needs several
    # years of history to mean anything; headline sentiment via
    # TextBlob is a weak lexicon-based proxy on financial text) without
    # clear evidence they improved the signal.
    bull_points = 0
    if latest_row['Close'] > latest_row['EMA_20']:
        bull_points += 1
    if latest_row['RSI_14'] > 50:
        bull_points += 1
    if analyst_target and analyst_target > latest_row['Close']:
        bull_points += 1

    if bull_points == 3:
        stance = "🟢 Bullish"
    elif bull_points == 2:
        stance = "🟡 Neutral+"
    elif bull_points == 1:
        stance = "🟠 Neutral"
    else:
        stance = "🔴 Bearish"

    st.markdown("---")
    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("Current Price", f"{currency_symbol}{latest_row['Close']:.2f}",
                   f"{latest_row['Price Change']:.2f}%")
    with top2:
        if analyst_target is not None:
            st.metric("Analyst Target", f"{currency_symbol}{analyst_target:.2f}", f"{target_upside:.1f}%")
        else:
            st.metric("Analyst Target", "N/A")
    with top3:
        st.metric("RSI (14)", f"{latest_row['RSI_14']:.1f}")
    with top4:
        st.metric("Investment View", stance)

    # =================================================================
    # 6. CHART
    # =================================================================
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.85, 0.15])
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                  low=data['Low'], close=data['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], line=dict(color='orange', width=1.5),
                              name="20-Day EMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], line=dict(color='cyan', width=1, dash='dot'),
                              name="VWAP"), row=1, col=1)

    value_rows = data[data['Signal'] == "💎 VALUE BUY"]
    momentum_rows = data[data['Signal'] == "🚀 MOMENTUM BUY"]

    fig.add_trace(go.Scatter(x=value_rows.index, y=value_rows['Low'] * 0.98, mode='markers',
                              marker=dict(symbol='diamond', size=10, color='#2ecc71', line=dict(width=1, color='white')),
                              name="Value Buy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=momentum_rows.index, y=momentum_rows['Low'] * 0.98, mode='markers',
                              marker=dict(symbol='triangle-up', size=11, color='#00e5ff'),
                              name="Momentum Buy"), row=1, col=1)

    if not tranche_df.empty:
        stopped = tranche_df[tranche_df['Outcome'] == "🔴 Stopped Out"]
        targeted = tranche_df[tranche_df['Outcome'] == "🔵 Target Hit"]
        if not stopped.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(stopped['Outcome Date']), y=stopped['Stop'], mode='markers',
                marker=dict(symbol='x', size=9, color='red'), name="Stop Hit"), row=1, col=1)
        if not targeted.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(targeted['Outcome Date']), y=targeted['Target'], mode='markers',
                marker=dict(symbol='star', size=10, color='#3498db'), name="Target Hit"), row=1, col=1)

    colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in data.iterrows()]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color=colors, opacity=0.4),
                  row=2, col=1)

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=550,
                       hovermode="x unified", margin=dict(t=10, b=20, l=20, r=20),
                       xaxis=dict(range=[data.index[-90], data.index[-1]]))

    st.markdown("---")
    st.subheader("📈 Technical Analysis")
    st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # 6a. TRANCHE OUTCOMES - per-signal stop/target result (the "exit" view)
    # =================================================================
    if not tranche_df.empty:
        n_stopped = (tranche_df['Outcome'] == "🔴 Stopped Out").sum()
        n_target = (tranche_df['Outcome'] == "🔵 Target Hit").sum()
        n_open = (tranche_df['Outcome'] == "Still Open").sum()
        st.info(
            f"📋 Of {len(tranche_df)} past BUY signals: **{n_target} hit their target**, "
            f"**{n_stopped} hit their stop**, **{n_open} are still open** (neither level reached yet)."
        )

    with st.expander("🎯 Tranche Outcomes", expanded=False):
        st.caption(
            "What happened after each past BUY signal - did that tranche's own stop or "
            "target get hit, or is it still open. Independent per signal; does not affect "
            "future signals."
        )
        if not tranche_df.empty:
            st.dataframe(tranche_df, use_container_width=True, hide_index=True)
        else:
            st.info("No BUY signals yet in this window to evaluate.")

    # =================================================================
    # 6b. HISTORICAL DATA TABLE
    # =================================================================
    with st.expander("📚 Historical Data", expanded=False):
        st.caption("Strategy History (Latest First)")

        df_disp = data.reset_index()
        df_disp['Date'] = df_disp['Date'].dt.strftime('%Y-%m-%d')
        df_disp = df_disp.sort_values(by='Date', ascending=False)

        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Price Change',
                'EMA_20', 'VWAP', 'ATR_10', 'RSI_14', 'Stop', 'Target',
                'Conviction', 'Signal', 'Reason']

        def row_styler(row):
            if "VALUE" in str(row.Signal):
                return ['background-color: rgba(46, 204, 113, 0.2)'] * len(row)
            if "MOMENTUM" in str(row.Signal):
                return ['background-color: rgba(52, 152, 219, 0.2)'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_disp[cols].head(90).style.apply(row_styler, axis=1).format({
                'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 'Close': '{:.2f}',
                'Price Change': '{:+.2f}%', 'Volume': '{:,.0f}', 'EMA_20': '{:.2f}',
                'VWAP': '{:.2f}', 'ATR_10': '{:.2f}', 'RSI_14': '{:.1f}',
                'Stop': '{:.2f}', 'Target': '{:.2f}',
            }, na_rep="-"),
            use_container_width=True
        )

    # =================================================================
    # 6c. CONDITION DIAGNOSTICS - answers "why no VALUE BUY signal?"
    # =================================================================
    with st.expander("🔍 Why no VALUE BUY signal? Condition breakdown"):
        n = len(data)
        diag_rows = [
            {"Condition": "Below 20-EMA", "Days Passed": int(data['cond_below_ema'].sum()),
             "% of Days": f"{data['cond_below_ema'].mean() * 100:.1f}%"},
            {"Condition": f"RSI in [{rsi_low}, {rsi_high}]", "Days Passed": int(data['cond_rsi_band'].sum()),
             "% of Days": f"{data['cond_rsi_band'].mean() * 100:.1f}%"},
            {"Condition": f"Volume > {vol_confirm_mult}x 30d avg", "Days Passed": int(data['cond_vol_confirm'].sum()),
             "% of Days": f"{data['cond_vol_confirm'].mean() * 100:.1f}%"},
            {"Condition": "No single-day crash (>3%)", "Days Passed": int(data['cond_no_crash'].sum()),
             "% of Days": f"{data['cond_no_crash'].mean() * 100:.1f}%"},
        ]
        all_four = (data['cond_below_ema'] & data['cond_rsi_band']
                    & data['cond_vol_confirm'] & data['cond_no_crash'])
        diag_rows.append({"Condition": "ALL FOUR TOGETHER", "Days Passed": int(all_four.sum()),
                           "% of Days": f"{all_four.mean() * 100:.1f}%"})
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
        st.caption(
            f"Out of {n} trading days in the window. If one condition's pass rate is much "
            "lower than the others, that's your bottleneck - loosen that slider first "
            "rather than all of them at once."
        )

    # =================================================================
    # 7. BACKTEST RESULTS
    # =================================================================
    st.markdown("---")
    st.subheader(f"🧪 Signal Backtest — Forward {HORIZON}-Day Returns")
    st.caption(
        "Simplified check: for every historical BUY signal, what happened to price "
        f"{HORIZON} trading days later. Ignores intra-trade stop-outs, so treat as a "
        "rough sanity check on whether the signal has any edge, not a full backtest. "
        "**Raw Occurrences** counts every signal day; **Independent Episodes** clusters "
        "signals within 3 trading days of each other into one event, since consecutive "
        "days in the same dip/rally aren't truly independent samples - trust the Episodes "
        "count more than Raw Occurrences when judging how much data you actually have."
    )
    if backtest_summary:
        st.dataframe(pd.DataFrame(backtest_summary), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough historical BUY signals yet to backtest.")

elif data is not None:
    st.warning("Not enough historical data for reliable signal generation (need 60+ days).")
else:
    st.error("Error loading data. Check ticker.")

# =========================================================================
# CLEAR CACHE - placed last and unconditionally, so it always renders
# (even if the data fetch above failed) and always sits at the true
# bottom of the sidebar, after both expanders above.
# =========================================================================
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear cache & retry data fetch"):
    get_data.clear()
    get_ticker_info.clear()
    st.rerun()