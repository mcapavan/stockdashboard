import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from prophet import Prophet
from textblob import TextBlob

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
# Forecast Function

@st.cache_data(ttl=3600)
def get_forecast(data, days=60):

    forecast_df = data.reset_index()[['Date', 'Close']].copy()

    forecast_df.rename(
        columns={
            'Date': 'ds',
            'Close': 'y'
        },
        inplace=True
    )

    forecast_df['ds'] = pd.to_datetime(
        forecast_df['ds']
    ).dt.tz_localize(None)


    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return forecast
# Sentiment Function
@st.cache_data(ttl=1800)
def get_sentiment(ticker):

    try:
        stock = yf.Ticker(ticker)

        news = stock.news

        if not news:
            return 0, "Neutral"

        sentiments = []

        for article in news[:10]:

            title = ""

            if isinstance(article, dict):

                title = article.get("title", "")

                if not title and "content" in article:
                    title = article["content"].get(
                        "title",
                        ""
                    )

            polarity = TextBlob(title).sentiment.polarity
            sentiments.append(polarity)

        avg_sentiment = sum(sentiments) / len(sentiments)

        if avg_sentiment > 0.10:
            label = "🟢 Bullish"
        elif avg_sentiment < -0.10:
            label = "🔴 Bearish"
        else:
            label = "🟡 Neutral"

        return avg_sentiment, label

    except:
        return 0, "Unavailable"

@st.cache_data(ttl=3600)
def get_analyst_target(ticker):

    try:

        stock = yf.Ticker(ticker)

        targets = stock.analyst_price_targets

        if isinstance(targets, dict):

            return targets.get("mean")

        return None

    except:
        return None

@st.cache_data(ttl=3600)
def get_currency_symbol(ticker):
    try:
        info = yf.Ticker(ticker).info
        # currency = info.get('currency', 'USD')
        currency = info.get('currency') or 'USD'
    except:
        currency = 'USD'

    symbol_map = {
        'USD': '$',
        'INR': '₹',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
    }
    return symbol_map.get(currency, currency + ' ')
        
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

    st.sidebar.caption(
        "🇺🇸 US: RIVN, TSLA, NVDA  |  🇮🇳 India: INFY.NS, TCS.NS, RELIANCE.NS"
    )
else:
    ticker_symbol = ticker_dict[selected_name]

st.sidebar.markdown("---")

forecast_days = st.sidebar.slider(
    "🔮 Forecast Days",
    min_value=30,
    max_value=180,
    value=60,
    step=30
)

data = get_data(ticker_symbol)
currency_symbol = get_currency_symbol(ticker_symbol)

if data is not None:
    if len(data) < 60:
        st.warning(
            "Not enough historical data for reliable forecasting."
        )

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
    
    
    # NEW: INTERACTIVE RISK-REWARD CALCULATOR SECTION
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧮 Interactive Risk-Reward Calculator")
    
    calc_entry = st.sidebar.number_input(f"Hypothetical Entry Price ({currency_symbol})", value=round(current_market_price, 2), step=0.1)
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
    st.sidebar.info(f"🛑 **Suggested Stop:** {currency_symbol}{calc_stop:.2f}\n\n🎯 **Suggested Target:** {currency_symbol}{calc_target:.2f}")
    
    col_c1, col_c2 = st.sidebar.columns(2)
    with col_c1:
        st.metric("Total Risk", f"{currency_symbol}{total_risk:.2f}", delta_color="inverse")
    with col_c2:
        st.metric("Total Reward", f"{currency_symbol}{total_reward:.2f}")
        
    st.sidebar.caption(f"Total Capital Exposure: {currency_symbol}{total_cost:,.2f}")
    st.sidebar.markdown("---")

    # Sidebar Inputs for Portfolio Settings 
    st.sidebar.subheader("💼 Portfolio Settings")
    core_shares = st.sidebar.number_input("Core Shares", value=100)
    core_basis = st.sidebar.number_input(f"Cost Basis ({currency_symbol})", value=15.63)
    
    stop_loss_val = st.sidebar.number_input(f"Current Stop Loss ({currency_symbol})", value=float(temp_latest['Stop']))
    target_price_val = st.sidebar.number_input(f"Current Profit Target ({currency_symbol})", value=float(temp_latest['Target']))


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

    signal_date = data.index[-2]
    signal_text = data.iloc[-2]["Signal"]
    signal_date_str = signal_date.strftime("%d %b %Y")

    try:
        # forecast = get_forecast(data)
        forecast = get_forecast(
            data,
            forecast_days
        )
        future_price = forecast.iloc[-1]['yhat']
    except Exception as e:
        forecast = None
        future_price = latest_row['Close']

    sent_score, sent_label = get_sentiment(ticker_symbol)

    # Display-friendly News Sentiment
    if "Bullish" in sent_label:
        display_sentiment = "📰 Bullish"
    elif "Bearish" in sent_label:
        display_sentiment = "📰 Bearish"
    else:
        display_sentiment = "📰 Neutral"

    analyst_target = get_analyst_target(
        ticker_symbol
    )

    forecast_upside = (
        (future_price / latest_row['Close']) - 1
    ) * 100
    
    if analyst_target is not None:
        target_upside = (
            (analyst_target / latest_row['Close']) - 1
        ) * 100
    else:
        target_upside = 0
    # --- QUANT SCORE ---

    quant_score = 0

    if latest_row['Close'] > latest_row['EMA_20']:
        quant_score += 25

    if latest_row['RSI_14'] > 50:
        quant_score += 25

    if sent_score > 0:
        quant_score += 25

    if future_price > latest_row['Close']:
        quant_score += 25

    # --- 4. UI HEADER ---
    try:
        company_name = yf.Ticker(ticker_symbol).info.get('longName', ticker_symbol)
        st.title(f"📊 {company_name}")
    except:
        st.title(f"📊 {ticker_symbol}")
    
    previous_signal = data.iloc[-2]["Signal"]

    st.markdown(
        f"### 📌 Signal ({signal_date_str}): {signal_text}"
    )
    
    # ==================================================
    # EXECUTIVE SUMMARY
    # ==================================================

    bull_points = 0

    if latest_row['Close'] > latest_row['EMA_20']:
        bull_points += 1

    if latest_row['RSI_14'] > 50:
        bull_points += 1

    if sent_score > 0:
        bull_points += 1

    if future_price > latest_row['Close']:
        bull_points += 1

    if analyst_target and analyst_target > latest_row['Close']:
        bull_points += 1


    if bull_points >= 4:
        stance = "🟢 Bullish"

    elif bull_points >= 3:
        stance = "🟡 Neutral++"

    elif bull_points == 2:
        stance = "🟠 Neutral"

    else:
        stance = "🔴 Bearish"

    st.markdown("---")
    # Executive Dashboard

    top1, top2, top3, top4, top5 = st.columns(5)

    with top1:
        st.metric(
            "Current Price",
            f"{currency_symbol}{latest_row['Close']:.2f}",
            f"{latest_row['Price Change']:.2f}%"
        )
   
    with top2:
        st.metric(
            "Forecast Target",
            f"{currency_symbol}{future_price:.2f}",
            f"{forecast_upside:.1f}% ({forecast_days} Days)"
        )
    with top3:
        if analyst_target is not None:
            st.metric(
                "Analyst Target",
                f"{currency_symbol}{analyst_target:.2f}",
                f"{target_upside:.1f}%"
            )
        else:
            st.metric(
                "Analyst Target",
                "N/A"
            )
    with top4:
        st.metric(
            "News Sentiment",
            display_sentiment
        )
    with top5:
        st.metric(
            "Investment View",
            stance
        )    

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
                        margin=dict(t=10,b=20,l=20,r=20),
                        xaxis=dict(range=[data.index[-90], data.index[-1]]))
    
    # show technical chart
    st.markdown("---")
    st.subheader("📈 Technical Analysis")
    st.plotly_chart(fig, use_container_width=True)


    if forecast is not None:
        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name='Forecast',
                line=dict(color='cyan')
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                name='Upper',
                line=dict(color='green', dash='dot')
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                name='Lower',
                line=dict(color='red', dash='dot')
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                name="Actual Price",
                line=dict(color="white")
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                line=dict(width=0),
                showlegend=False
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                fill='tonexty',
                fillcolor='rgba(0,255,255,0.15)',
                line=dict(width=0),
                name='Confidence Band'
            )
        )

        # forecast_fig.update_layout(
        #     template="plotly_dark",
        #     height=450
        # )

        forecast_fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=450, hovermode="x unified",
                        margin=dict(t=10,b=20,l=20,r=20))

        # show forecast chart
        st.markdown("---")
        st.subheader("🔮 Forecast Analysis")
        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

    # --- 6. DATA TABLE ---
    # show Data Table with Conditional Formatting 
    st.markdown("---")
    st.subheader("📚 Historical Data")
    st.caption("Strategy History (Latest First)")
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
