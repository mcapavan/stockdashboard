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
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, multi_level_index=False)
    return df if not df.empty else None

# Sidebar Ticker Input
st.sidebar.header("🕹️ Strategy Controls")
# --- SEARCHABLE DROPDOWN WITH COMPANY NAMES ---
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

# # The selectbox shows the friendly name (keys)
# selected_name = st.sidebar.selectbox(
#     "Search or Select Stock", 
#     options=list(ticker_dict.keys()),
#     index=0  # Defaults to Rivian
# )

# This extracts the actual ticker for yfinance to use
# ticker_symbol = ticker_dict[selected_name]
#ticker_symbol = st.sidebar.text_input("Stock Ticker", value="RIVN").upper()

# --- TICKER SELECTION MODE ---
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
    
    # Dynamic Excel Formulas for Stop and Target
    data['Stop'] = data['Close'] - (3 * data['ATR_10'])
    data['Target'] = data['Close'] + (6 * data['ATR_10'])
    
    # --- 2. LOGIC ENGINE ---
    # We define the sidebar defaults using a temporary latest row
    temp_latest = data.iloc[-1]
    
    st.sidebar.subheader("💼 Portfolio Settings")
    core_shares = st.sidebar.number_input("Core Shares", value=100)
    core_basis = st.sidebar.number_input("Cost Basis ($)", value=15.63)
    
    stop_loss_val = st.sidebar.number_input("Current Stop Loss ($)", value=float(temp_latest['Stop']))
    target_price_val = st.sidebar.number_input("Current Profit Target ($)", value=float(temp_latest['Target']))

    # --- 2. LOGIC ENGINE (Exact Excel Port) ---
    def excel_logic_port(row):
        low, close, ema = row['Low'], row['Close'], row['EMA_20']
        high, rsi, vol, avg_vol = row['High'], row['RSI_14'], row['Volume'], row['Vol_Avg_30']
        stop, target = row['Stop'], row['Target']

        # ADD THIS LINE: It checks if the row being processed is the last one in the data
        is_latest = (row.name == data.index[-1])
        prefix = "⚡ LIVE: " if is_latest else ""
        
        # 1. STOP LOSS CHECK
        if low <= stop: 
            return f"{prefix}⚠️ EXIT / STOP", "red", "Hit Stop Loss Level", 0
        
        # 2. TAKE PROFIT CHECK
        if high >= target: 
            return f"{prefix}💰 TAKE PROFIT", "blue", "Hit Target Level", 0
        
        # 3. VALUE BUY (Nested AND logic)
        # E30 < J30 (Close < EMA) AND E30 > G30*0.95 (Close > Low*0.95) 
        # AND L30 > 35 (RSI > 35) AND F30 > Avg*0.7 (Vol > 0.7x)
        if close < ema and close > (row['Open'] * 0.95) and rsi > 35 and vol > (avg_vol * 0.7):
            return f"{prefix}💎 VALUE BUY", "green", "Value Criteria Met", 5
        
        # 4. MOMENTUM BUY
        if close > ema and vol > (avg_vol * 1.3):
            return f"{prefix}🚀 MOMENTUM BUY", "cyan", "Momentum Criteria Met", 5
        
        # 5. DEFAULT
        return f"{prefix}😴 WAIT", "gray", "Price above SMA or Low Volume", 1


    # Apply Logic to the entire dataframe
    res = data.apply(excel_logic_port, axis=1)
    data[['Signal', 'Color', 'Reason', 'Conviction']] = pd.DataFrame(res.tolist(), index=data.index)

    # CRITICAL FIX: Now capture the latest_row AFTER all columns are created
    latest_row = data.iloc[-1]

    # --- 3. UI HEADER ---
    # Add this inside your 'if data is not None:' block
    try:
        company_name = yf.Ticker(ticker_symbol).info.get('longName', ticker_symbol)
        st.title(f"📊 {company_name} Strategy Dashboard")
    except:
        st.title(f"📊 {ticker_symbol} Strategy Dashboard")
    
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Price", f"${latest_row['Close']:.2f}", delta=f"{latest_row['Price Change']:.2f}%")
    with c2: 
        # Color is now guaranteed to be in latest_row
        st.markdown(f"### Signal: :{latest_row['Color']}[{latest_row['Signal']}]")
        st.caption(f"Reason: {latest_row['Reason']}")
    with c3:
        pnl = (latest_row['Close'] - core_basis) * core_shares
        st.metric("Portfolio P&L", f"${pnl:,.2f}", delta=f"Basis: ${core_basis}")

    # --- 4. COMPACT CHART ---
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

    # --- 5. DATA TABLE ---
    st.subheader("Strategy History (Latest First)")
    df_disp = data.reset_index()
    df_disp['Date'] = df_disp['Date'].dt.strftime('%Y-%m-%d')
    df_disp = df_disp.sort_values(by='Date', ascending=False)
    
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Price Change', 'EMA_20', 'VWAP', 'ATR_10', 'RSI_14', 'Stop', 'Target', 'Conviction', 'Signal', 'Reason']

    def row_styler(row):
        if "VALUE" in str(row.Signal): return ['background-color: #1b4d3e'] * len(row)
        if "MOMENTUM" in str(row.Signal): return ['background-color: #0e2f44'] * len(row)
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
