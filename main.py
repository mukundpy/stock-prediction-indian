import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Indian Stock Forecast Pro", layout="wide")

# Title and intro
st.title("📈 Indian Stock Forecast Pro")
st.markdown("Your personal Indian stock market analysis companion")
st.markdown("---")

url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
df = pd.read_csv(url)

# Create (Company Name, Yahoo Finance Symbol) list
indian_stocks = [
    (row["NAME OF COMPANY"], row["SYMBOL"] + ".NS")
    for _, row in df.iterrows()
]

# Sidebar with filters
with st.sidebar:
    st.header("🔍 Search & Filter")
    
    # Create a searchable company list
    stock_names = [s[0] for s in indian_stocks]
    selected_name = st.selectbox(
        "Find your favorite company",
        stock_names,
        help="Search for any Indian company from the list"
    )
    
    # Get ticker
    selected_ticker = [s[1] for s in indian_stocks if s[0] == selected_name][0]
    
    # Time period selection
    st.markdown("### Time Period")
    period = st.selectbox(
        "How far back would you like to look?",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
        help="Select the time range for historical data"
    )
    
    # Chart type selection
    st.markdown("### Chart Style")
    chart_type = st.radio(
        "How do you want to see the price?",
        ["Line Chart", "Candlestick Chart"],
        help="Choose your preferred chart visualization"
    )
    
    # Forecast days
    st.markdown("### Forecasting")
    forecast_days = st.slider(
        "How many days ahead should we predict?",
        1, 90, 30,
        help="Adjust the forecast duration"
    )

# Main content area
st.subheader(f"📊 {selected_name}")
st.caption(f"Ticker: {selected_ticker} | Time Period: {period}")

# Load stock data function
@st.cache_data
def load_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            raise ValueError("No data available for this ticker")
        return data
    except Exception as e:
        return None

# Calculate technical indicators
def calculate_indicators(data):
    df = data.copy()
    
    # Moving Averages - help identify trends
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI - shows if stock is overbought or oversold
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands - volatility indicator
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma + (std * 2)
    df['BB_Lower'] = sma - (std * 2)
    df['BB_Middle'] = sma
    
    return df

# Simple forecast function
def forecast_price(data, days):
    try:
        close_prices = data['Close'].values
        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future
        future_X = np.arange(len(close_prices), len(close_prices) + days).reshape(-1, 1)
        future_prices = model.predict(future_X)
        
        # Calculate volatility
        volatility = np.std(np.diff(close_prices[-30:])) if len(close_prices) >= 30 else 0
        
        # Create dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': future_prices,
            'upper': future_prices + (volatility * 1.96),
            'lower': future_prices - (volatility * 1.96)
        })
        
        return forecast_df
    except Exception as e:
        return None

# Load data
with st.spinner("📥 Fetching the latest stock data..."):
    data = load_stock_data(selected_ticker, period)

if data is None or data.empty:
    st.error("❌ Couldn't fetch data for this stock. Please try another company or check your internet connection.")
else:
    st.success(f"✅ Got {len(data)} days of historical data!")
    
    # Calculate indicators
    data = calculate_indicators(data)
    
    # Display Key Metrics
    st.markdown("### 💰 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data['Close'].iloc[-1]
        st.metric("Current Price", f"₹{current_price:,.0f}")
    
    with col2:
        if len(data) > 1:
            prev_price = data['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            st.metric("Daily Change", f"₹{change:,.0f}", f"{change_pct:.2f}%")
    
    with col3:
        high = data['High'].max()
        st.metric("52-Week High", f"₹{high:,.0f}")
    
    with col4:
        low = data['Low'].min()
        st.metric("52-Week Low", f"₹{low:,.0f}")
    
    # Price Chart
    st.markdown("### 📈 Price Movement")
    
    if chart_type == "Line Chart":
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA20'],
            mode='lines',
            name='20-Day Average',
            line=dict(color='orange', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA50'],
            mode='lines',
            name='50-Day Average',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            mode='lines',
            name='Upper Band',
            line=dict(color='gray', width=1),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            mode='lines',
            name='Lower Band',
            line=dict(color='gray', width=1),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.1)'
        ))
        
    else:  # Candlestick
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC'
        )])
    
    fig.update_layout(
        title=f"{selected_name} - Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (Rupees)",
        height=500,
        template="plotly_white",
        hovermode='x unified',
        font=dict(size=11)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical Analysis Section
    st.markdown("### 🔬 Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### RSI (Momentum Indicator)")
        latest_rsi = data['RSI'].iloc[-1]
        if latest_rsi > 70:
            st.warning(f"⚠️ RSI: {latest_rsi:.1f} - Stock looks overbought (might go down)")
        elif latest_rsi < 30:
            st.info(f"📌 RSI: {latest_rsi:.1f} - Stock looks oversold (might go up)")
        else:
            st.success(f"✅ RSI: {latest_rsi:.1f} - Stock is in normal range")
    
    with col2:
        st.markdown("#### Moving Averages (Trend)")
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        st.write(f"20-Day MA: ₹{ma20:,.0f}")
        st.write(f"50-Day MA: ₹{ma50:,.0f}")
        if ma20 > ma50:
            st.success("📈 **Bullish Trend** - 20-day is above 50-day")
        else:
            st.error("📉 **Bearish Trend** - 20-day is below 50-day")
    
    # RSI Chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI'],
        mode='lines',
        name='RSI(14)',
        line=dict(color='purple', width=2)
    ))
    
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    fig_rsi.update_layout(
        title="RSI (14) - Momentum Indicator",
        xaxis_title="Date",
        yaxis_title="RSI Value",
        height=350,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # Price Forecast Section
    st.markdown("### 🔮 Price Prediction")
    st.info("📌 This forecast uses Linear Regression based on historical price movement")
    
    with st.spinner("Computing forecast..."):
        forecast_df = forecast_price(data, forecast_days)
        
        if forecast_df is not None:
            fig_forecast = go.Figure()
            
            # Historical
            fig_forecast.add_trace(go.Scatter(
                x=data.index[-60:],  # Last 60 days
                y=data['Close'].iloc[-60:],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines',
                name='Predicted Price',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Confidence bands
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Range',
                fillcolor='rgba(255, 0, 0, 0.2)'
            ))
            
            fig_forecast.update_layout(
                title=f"{selected_name} - {forecast_days}-Day Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price (Rupees)",
                height=500,
                template="plotly_white",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Show forecast table
            st.markdown("#### Next Few Days Prediction")
            forecast_table = forecast_df.tail(7).copy()
            forecast_table['date'] = forecast_table['date'].dt.strftime('%Y-%m-%d')
            forecast_table.columns = ['Date', 'Predicted Price', 'Upper Bound', 'Lower Bound']
            forecast_table['Predicted Price'] = forecast_table['Predicted Price'].apply(lambda x: f"₹{x:,.0f}")
            forecast_table['Upper Bound'] = forecast_table['Upper Bound'].apply(lambda x: f"₹{x:,.0f}")
            forecast_table['Lower Bound'] = forecast_table['Lower Bound'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(forecast_table, use_container_width=True, hide_index=True)
    
    # Returns Analysis
    st.markdown("### 📊 Performance Analysis")
    
    if len(data) > 20:
        returns = data['Close'].pct_change().dropna()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_return = returns.mean() * 100
            st.metric("Daily Return (Avg)", f"{avg_return:+.3f}%")
        
        with col2:
            volatility = returns.std() * 100
            st.metric("Daily Volatility", f"{volatility:.3f}%")
        
        with col3:
            total_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            st.metric("Total Return", f"{total_return:+.2f}%")
    
    # Volume Analysis
    st.markdown("### 📈 Trading Volume")
    
    if 'Volume' in data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            avg_volume = data['Volume'].mean()
            st.metric("Average Daily Volume", f"{avg_volume:,.0f}")
        
        with col2:
            recent_volume = data['Volume'].iloc[-1]
            st.metric("Today's Volume", f"{recent_volume:,.0f}")
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        
        fig_vol.update_layout(
            height=350,
            title="Daily Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # Raw Data Section
    with st.expander("📋 View Raw Data"):
        st.dataframe(data.tail(20), use_container_width=True)
    
    # Download Section
    st.markdown("### 💾 Download Data")
    
    csv = data.to_csv()
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"{selected_name.replace(' ', '_')}_stock_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
### About This Tool
**Indian Stock Forecast Pro** - Your free companion for analyzing Indian stock market data

**Features:**
- 📊 Real-time data from 100+ Indian companies
- 🔍 Search and filter stocks easily
- 📈 Multiple chart types (Line & Candlestick)
- 🔬 Advanced technical indicators (RSI, MA, Bollinger Bands)
- 🔮 AI-powered price predictions
- 💹 Performance analytics
- 📥 Download data as CSV

**Data Source:** Yahoo Finance  
**Update Frequency:** Real-time

---
**⚠️ Disclaimer:** This tool is for educational purposes only. Stock market predictions are not guaranteed. 
Always consult a financial advisor before making investment decisions. Past performance doesn't guarantee future results.

*Made for Indian engineering students learning data analysis & stock market investing* ❤️
""")
