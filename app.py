import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="RIVN Strategy Dashboard Pro", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("🕹️ Strategy Controls")
ticker_symbol = st.sidebar.text_input("Stock Ticker", value="RIVN").upper()

st.sidebar.subheader("💼 Portfolio Settings")
core_shares = st.sidebar.number_input("Core Shares", value=200)
core_basis = st.sidebar.number_input("Cost Basis ($)", value=15.63)
stop_loss_val = st.sidebar.number_input("Stop Loss ($)", value=12.50)
target_price_val = st.sidebar.number_input("Profit Target ($)", value=24.14)

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data(ticker):
    # 1y ensures indicators like EMA have settled accurately
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, multi_level_index=False)
    return df if not df.empty else None

data = get_data(ticker_symbol)

if data is not None:
    # 1. INDICATORS
    data['EMA_20'] = ta.ema(data['Close'], length=20)
    data['RSI_14'] = ta.rsi(data['Close'], length=14)
    data['ATR_10'] = ta.atr(data['High'], data['Low'], data['Close'], length=10, mamode="sma")
    data['Vol_Avg_30'] = data['Volume'].rolling(window=30).mean()
    data['Price Change'] = data['Close'].pct_change() * 100
    data['VWAP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['Stop'] = stop_loss_val
    data['Target'] = target_price_val

    # 2. LOGIC ENGINE
    def excel_logic_port(row):
        low, close, ema = row['Low'], row['Close'], row['EMA_20']
        rsi, vol, avg_vol = row['RSI_14'], row['Volume'], row['Vol_Avg_30']
        if low <= stop_loss_val: return "🛑 EXIT", "red", "Hit Stop Loss Level", 0
        if close >= target_price_val: return "🎯 EXIT", "blue", "Hit Target Level", 0
        if close < ema and close > (ema * 0.95) and rsi > 35:
            if vol > (avg_vol * 0.7): return "💎 VALUE BUY", "green", "Value Criteria Met", 5
            return "💎 VALUE BUY", "green", "Volume too Low (But Value Met)", 4
        if close > ema and vol > (avg_vol * 1.3): return "🚀 MOMENTUM BUY", "cyan", "Momentum Criteria Met", 5
        if close >= ema: return "😴 WAIT", "gray", "Price above SMA (No Value)", 1
        return "😴 WAIT", "gray", "Neutral Range", 1

    res = data.apply(excel_logic_port, axis=1)
    data[['Signal', 'Color', 'Reason', 'Conviction']] = pd.DataFrame(res.tolist(), index=data.index)

    # 3. HEADER METRICS
    st.title(f"📊 {ticker_symbol} Strategy Dashboard")
    latest = data.iloc[-1]
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Current Price", f"${latest['Close']:.2f}")
    with c2: st.markdown(f"### Signal: :{latest['Color']}[{latest['Signal']}]")
    with c3:
        pnl = (latest['Close'] - core_basis) * core_shares
        st.metric("Portfolio P&L", f"${pnl:,.2f}", delta=f"Basis: ${core_basis}")

    # 4. DUAL-PANE INTERACTIVE CHART
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, row_heights=[0.85, 0.15])

    # --- TOP PANE: PRICE ---
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], 
        name="OHLC"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], line=dict(color='orange', width=1.5), name="20-Day EMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], line=dict(color='cyan', width=1, dash='dot'), name="VWAP"), row=1, col=1)
    
    fig.add_hline(y=target_price_val, line_dash="dash", line_color="green", annotation_text="TARGET", row=1, col=1)
    fig.add_hline(y=stop_loss_val, line_dash="dash", line_color="red", annotation_text="STOP", row=1, col=1)

    # --- BOTTOM PANE: VOLUME ---
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in data.iterrows()]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color=colors, opacity=0.5), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Vol_Avg_30'], line=dict(color='white', width=1), name="30-Day Avg Vol"), row=2, col=1)

    # --- CHART CUSTOMIZATION ---
    # Set 3 Month Initial Zoom
    end_date = data.index[-1]
    start_date = end_date - timedelta(days=90)

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=550,
        hovermode="x unified", # Shows all info for that date in one tooltip
        margin=dict(t=30, b=10),
        xaxis=dict(range=[start_date, end_date]) # Focuses zoom on last 3 months
    )
    
    # Force Y-axis to scale to the 3-month zoom range
    fig.update_yaxes(fixedrange=False)
    
    st.plotly_chart(fig, use_container_width=True)

    # 5. TABLE
    st.subheader("Strategy History (Latest First)")
    df_disp = data.reset_index()
    df_disp['Date'] = df_disp['Date'].dt.strftime('%Y-%m-%d')
    df_disp = df_disp.sort_values(by='Date', ascending=False)
    
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'EMA_20', 'VWAP', 'ATR_10', 'RSI_14', 'Stop', 'Target', 'Conviction', 'Signal', 'Reason']

    def row_styler(row):
        if "VALUE" in str(row.Signal): return ['background-color: #1b4d3e'] * len(row)
        if "MOMENTUM" in str(row.Signal): return ['background-color: #0e2f44'] * len(row)
        if "EXIT" in str(row.Signal): return ['background-color: #440e0e'] * len(row)
        return [''] * len(row)

    st.dataframe(
        df_disp[cols].head(50).style.apply(row_styler, axis=1).format({
            'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 'Close': '{:.2f}',
            'Volume': '{:,.0f}', 'EMA_20': '{:.2f}', 'VWAP': '{:.2f}', 
            'ATR_10': '{:.2f}', 'RSI_14': '{:.1f}', 'Stop': '{:.2f}', 'Target': '{:.2f}'
        }), use_container_width=True
    )
else:
    st.error("Check your connection or Ticker Symbol.")
