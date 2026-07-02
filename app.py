import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stocks Strategy Dashboard Pro", layout="wide")

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
    
    if df.empty:
        return None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.columns = [str(col).strip() for col in df.columns]
    return df

# Sidebar Title
st.sidebar.header("🕹️ Strategy Controls")

# --- TICKER DICTIONARY & SELECTION ---
ticker_dict = {
    "Rivian Automotive (RIVN)": "RIVN",
    "Nio Inc. (NIO)": "NIO",
    "XPeng Inc. (XPEV)": "XPEV",
    "Tesla, Inc. (TSLA)": "TSLA",
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp (MSFT)": "MSFT",
    "Alphabet Inc (GOOGL)": "GOOGL",
    "Amazon.com Inc (AMZN)": "AMZN",
    "NVIDIA Corp (NVDA)": "NVDA",
    "Intel Corp (INTC)": "INTC"
}

selected_name = st.sidebar.selectbox(
    "Search or Select Stock",
    options=["-- Enter Manually --"] + list(ticker_dict.keys()),
    index=1
)

if selected_name == "-- Enter Manually --":
    ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", value="RIVN").upper()
else:
    ticker_symbol = ticker_dict[selected_name]

data = get_data(ticker_symbol)

if data is not None:
    # --- 1. CALCULATE EXCEL INDICATORS ---
    data['EMA_20'] = ta.ema(data['Close'], length=20)
    data['ATR_10'] = ta.atr(data['High'], data['Low'], data['Close'], length=10, mamode="sma")
    data['RSI_14'] = ta.rsi(data['Close'], length=14)
    data['Vol_Avg_30'] = data['Volume'].rolling(window=30).mean()
    data['Price Change'] = data['Close'].pct_change() * 100
    data['VWAP'] = (data['High'] + data['Low'] + data['Close']) / 3
    
    data['Stop'] = data['Close'] - (3 * data['ATR_10'])
    data['Target'] = data['Close'] + (6 * data['ATR_10'])
    
    # --- 2. SIDEBAR SETTINGS & INTERACTIVE RISK-REWARD CALCULATOR ---
    temp_latest = data.iloc[-1]
    current_market_price = float(temp_latest['Close'])
    current_atr = float(temp_latest['ATR_10'])
    
    st.sidebar.subheader("💼 Portfolio Settings")
    core_shares = st.sidebar.number_input("Core Shares", value=100)
    core_basis = st.sidebar.number_input("Cost Basis ($)", value=15.63)
    
    stop_loss_val = st.sidebar.number_input("Current Stop Loss ($)", value=float(temp_latest['Stop']))
    target_price_val = st.sidebar.number_input("Current Profit Target ($)", value=float(temp_latest['Target']))

    # NEW: INTERACTIVE RISK-REWARD CALCULATOR SECTION
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧮 Interactive Risk-Reward Calculator")
    
    calc_entry = st.sidebar.number_input("Hypothetical Entry Price ($)", value=round(current_market_price, 2), step=0.1)
    calc_size = st.sidebar.number_input("Position Size (Shares)", value=100, step=10)
    
    # Let users choose risk definition (Multiplier of ATR vs Manual Percent)
    risk_mode = st.sidebar.radio("Stop Loss Metric", ["ATR Multiplier", "Percentage Drop"])
    
    if risk_mode == "ATR Multiplier":
        atr_mult = st.sidebar.slider("ATR Multiplier (Risk)", 1.0, 5.0, 3.0, 0.5)
        calculated_risk_per_share = current_atr * atr_mult
        calc_stop = calc_entry - calculated_risk_per_share
    else:
        pct_drop = st.sidebar.slider("Percent Risk (%)", 1.0, 20.0, 5.0, 0.5)
        calculated_risk_per_share = calc_entry * (pct_drop / 100.0)
        calc_stop = calc_entry - calculated_risk_per_share
        
    rr_ratio = st.sidebar.slider("Target Risk-Reward Ratio (R:R)", 1.0, 5.0, 2.0, 0.5)
    calc_target = calc_entry + (calculated_risk_per_share * rr_ratio)
    
    # Math Calculations
    total_cost = calc_entry * calc_size
    total_risk = calculated_risk_per_share * calc_size
    total_reward = (calc_target - calc_entry) * calc_size
    
    # Display Calculator Results directly in sidebar with styling
    st.sidebar.markdown("**Calculator Output:**")
    st.sidebar.info(f"🛑 **Suggested Stop:** ${calc_stop:.2f}\n\n🎯 **Suggested Target:** ${calc_target:.2f}")
    
    col_c1, col_c2 = st.sidebar.columns(2)
    with col_c1:
        st.metric("Total Risk", f"${total_risk:.2f}", delta_color="inverse")
    with col_c2:
        st.metric("Total Reward", f"${total_reward:.2f}")
        
    st.sidebar.caption(f"Total Capital Exposure: ${total_cost:,.2f}")
    st.sidebar.markdown("---")

    # --- 3. LOGIC ENGINE ---
    def excel_logic_port(row):
        low, close, ema = row['Low'], row['Close'], row['EMA_20']
        high, rsi, vol, avg_vol = row['High'], row['RSI_14'], row['Volume'], row['Vol_Avg_30']
        stop, target = row['Stop'], row['Target']

        is_latest = (row.name == data.index[-1])
        prefix = "⚡ LIVE: " if is_latest else ""
        
        if low <= stop: 
            return f"{prefix}⚠️ EXIT / STOP", "red", "Hit Stop Loss Level", 0
        if high >= target: 
            return f"{prefix}💰 TAKE PROFIT", "blue", "Hit Target Level", 0
        if close < ema and close > (row['Open'] * 0.95) and rsi > 35 and vol > (avg_vol * 0.7):
            return f"{prefix}💎 VALUE BUY", "green", "Value Criteria Met", 5
        if close > ema and vol > (avg_vol * 1.3):
            return f"{prefix}🚀 MOMENTUM BUY", "cyan", "Momentum Criteria Met", 5
        
        return f"{prefix}😴 WAIT", "gray", "Price above SMA or Low Volume", 1

    res = data.apply(excel_logic_port, axis=1)
    data[['Signal', 'Color', 'Reason', 'Conviction']] = pd.DataFrame(res.tolist(), index=data.index)
    latest_row = data.iloc[-1]

    # --- 4. UI HEADER ---
    try:
        company_name = yf.Ticker(ticker_symbol).info.get('longName', ticker_symbol)
        st.title(f"📊 {company_name} Strategy Dashboard")
    except:
        st.title(f"📊 {ticker_symbol} Strategy Dashboard")
    
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Price", f"${latest_row['Close']:.2f}", delta=f"{latest_row['Price Change']:.2f}%")
    with c2: 
        st.markdown(f"### Signal: :{latest_row['Color']}[{latest_row['Signal']}]")
        st.caption(f"Reason: {latest_row['Reason']}")
    with c3:
        pnl = (latest_row['Close'] - core_basis) * core_shares
        st.metric("Portfolio P&L", f"${pnl:,.2f}", delta=f"Basis: ${core_basis}")

    # --- 5. COMPACT CHART ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.85, 0.15])
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], line=dict(color='orange', width=1.5), name="20-Day EMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], line=dict(color='cyan', width=1, dash='dot'), name="VWAP"), row=1, col=1)
    
    fig.add_hline(y=target_price_val, line_dash="dash", line_color="green", annotation_text="TARGET", row=1, col=1)
    fig.add_hline(y=stop_loss_val, line_dash="dash", line_color="red", annotation_text="STOP", row=1, col=1)

    colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in data.iterrows()]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color=colors, opacity=0.4), row=2, col=1)

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=550, hovermode="x unified",
                      xaxis=dict(range=[data.index[-90], data.index[-1]]))
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. DATA TABLE ---
    st.subheader("Strategy History (Latest First)")
    df_disp = data.reset_index()
    df_disp['Date'] = df_disp['Date'].dt.strftime('%Y-%m-%d')
    df_disp = df_disp.sort_values(by='Date', ascending=False)
    
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Price Change', 'EMA_20', 'VWAP', 'ATR_10', 'RSI_14', 'Stop', 'Target', 'Conviction', 'Signal', 'Reason']

    def row_styler(row):
        # Updated styles: Using lighter alpha transparency colors to increase text readability
        if "VALUE" in str(row.Signal): return ['background-color: rgba(46, 204, 113, 0.2)'] * len(row)
        if "MOMENTUM" in str(row.Signal): return ['background-color: rgba(52, 152, 219, 0.2)'] * len(row)
        if "EXIT" in str(row.Signal): return ['background-color: #440e0e'] * len(row)
        return [''] * len(row)

    st.dataframe(
        df_disp[cols].head(60).style.apply(row_styler, axis=1).format({
            'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 'Close': '{:.2f}',
            'Price Change': '{:+.2f}%', 'Volume': '{:,.0f}', 'EMA_20': '{:.2f}', 
            'VWAP': '{:.2f}', 'ATR_10': '{:.2f}', 'RSI_14': '{:.1f}', 'Stop': '{:.2f}', 'Target': '{:.2f}'
        }), use_container_width=True
    )
else:
    st.error("Error loading data. Check ticker.")
