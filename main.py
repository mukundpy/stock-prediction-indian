import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Indian Stock Forecast Pro", layout="wide")

# Title
st.title("📈 Indian Stock Forecast Pro")
st.markdown("---")

# Indian stocks
indian_stocks = [
    ("Reliance", "RELIANCE.NS"),
    ("HDFC Bank", "HDFCBANK.NS"),
    ("TCS", "TCS.NS"),
    ("Infosys", "INFY.NS"),
    ("ICICI Bank", "ICICIBANK.NS"),
    ("ITC", "ITC.NS"),
    ("HUL", "HINDUNILVR.NS"),
    ("Axis Bank", "AXISBANK.NS"),
    ("Bajaj Finance", "BAJFINANCE.NS"),
    ("SBI", "SBIN.NS"),
    ("HCL Tech", "HCLTECH.NS"),
    ("Wipro", "WIPRO.NS"),
    ("Sun Pharma", "SUNPHARMA.NS"),
    ("Asian Paints", "ASIANPAINT.NS"),
    ("Titan", "TITAN.NS"),
    ("ONGC", "ONGC.NS"),
    ("Power Grid", "POWERGRID.NS"),
    ("Nestle", "NESTLEIND.NS"),
    ("Coal India", "COALINDIA.NS"),
    ("Tata Motors", "TATAMOTORS.NS")
]

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    stock_names = [s[0] for s in indian_stocks]
    selected_name = st.selectbox("Select Company", stock_names)
    
    selected_ticker = [s[1] for s in indian_stocks if s[0] == selected_name][0]
    
    period = st.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    chart_type = st.radio("Chart Type", ["Line", "Candlestick"])
    
    forecast_days = st.slider("Forecast Days", 1, 90, 30)

# Main content
st.subheader(f"{selected_name} ({selected_ticker})")

# Load data function
@st.cache_data
def load_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            raise ValueError("No data returned from API")
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Calculate technical indicators
def calculate_indicators(data):
    df = data.copy()
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma + (std * 2)
    df['BB_Lower'] = sma - (std * 2)
    df['BB_Middle'] = sma
    
    return df

# Load with spinner
with st.spinner("Loading stock data..."):
    data = load_stock_data(selected_ticker, period)

if data is None or data.empty:
    st.error("Could not load data. Please try again or check ticker symbol.")
else:
    st.success(f"Loaded {len(data)} days of data")
    
    # Calculate indicators
    data = calculate_indicators(data)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_price = data['Close'].iloc[-1]
        st.metric("Current Price", f"₹{current_price:,.2f}")
    with col2:
        if len(data) > 1:
            prev_price = data['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            st.metric("Daily Change", f"₹{change:,.2f}", f"{change_pct:.2f}%")
    with col3:
        st.metric("52 Week High", f"₹{data['High'].max():,.2f}")
    with col4:
        st.metric("52 Week Low", f"₹{data['Low'].min():,.2f}")
    
    # Create chart
    st.subheader("Price Chart with Technical Indicators")
    
    if chart_type == "Line":
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA20'],
            mode='lines',
            name='MA 20',
            line=dict(color='orange', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA50'],
            mode='lines',
            name='MA 50',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            mode='lines',
            name='BB Lower',
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
        title=f"{selected_name} Price with Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=500,
        template="plotly_white",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical Indicators Section
    st.subheader("Technical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**RSI (Relative Strength Index)**")
        latest_rsi = data['RSI'].iloc[-1]
        if latest_rsi > 70:
            st.warning(f"RSI: {latest_rsi:.2f} - Overbought")
        elif latest_rsi < 30:
            st.info(f"RSI: {latest_rsi:.2f} - Oversold")
        else:
            st.success(f"RSI: {latest_rsi:.2f} - Neutral")
    
    with col2:
        st.markdown("**Moving Averages**")
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        st.write(f"MA 20: ₹{ma20:,.2f}")
        st.write(f"MA 50: ₹{ma50:,.2f}")
        if ma20 > ma50:
            st.success("Bullish (MA20 > MA50)")
        else:
            st.error("Bearish (MA20 < MA50)")
    
    # RSI Chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=2)
    ))
    
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    fig_rsi.update_layout(
        title="RSI (14)",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=300,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # Price Forecasting with Prophet
    st.subheader("Price Forecast (Facebook Prophet)")
    
    with st.spinner("Training forecast model..."):
        try:
            forecast_data = data[['Close']].reset_index()
            forecast_data.columns = ['ds', 'y']
            forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
            
            model = Prophet(yearly_seasonality=True, daily_seasonality=False, interval_width=0.95)
            model.fit(forecast_data)
            
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            fig_forecast = go.Figure()
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['y'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            ))
            
            forecast_future = forecast[forecast['ds'] > forecast_data['ds'].max()]
            fig_forecast.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Interval (95%)',
                fillcolor='rgba(255, 0, 0, 0.2)'
            ))
            
            fig_forecast.update_layout(
                title=f"{selected_name} - {forecast_days} Day Forecast",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=500,
                template="plotly_white",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            st.markdown("**Forecast Summary**")
            forecast_summary = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).copy()
            forecast_summary.columns = ['Date', 'Forecast Price', 'Lower Bound', 'Upper Bound']
            forecast_summary['Forecast Price'] = forecast_summary['Forecast Price'].apply(lambda x: f"₹{x:,.2f}")
            forecast_summary['Lower Bound'] = forecast_summary['Lower Bound'].apply(lambda x: f"₹{x:,.2f}")
            forecast_summary['Upper Bound'] = forecast_summary['Upper Bound'].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(forecast_summary, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")
    
    # Returns Analysis
    st.subheader("Returns Analysis")
    
    if len(data) > 20:
        returns = data['Close'].pct_change().dropna()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_return = returns.mean() * 100
            st.metric("Avg Daily Return", f"{avg_return:.3f}%")
        
        with col2:
            volatility = returns.std() * 100
            st.metric("Daily Volatility", f"{volatility:.3f}%")
        
        with col3:
            total_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            st.metric("Total Return", f"{total_return:.2f}%")
    
    # Volume Analysis
    st.subheader("Volume Analysis")
    
    if 'Volume' in data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            avg_volume = data['Volume'].mean()
            st.metric("Average Volume", f"{avg_volume:,.0f}")
        
        with col2:
            recent_volume = data['Volume'].iloc[-1]
            st.metric("Recent Volume", f"{recent_volume:,.0f}")
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        
        fig_vol.update_layout(
            height=300,
            title="Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # Data Table
    with st.expander("View Raw Data"):
        st.dataframe(data.tail(20), use_container_width=True)
    
    # Download buttons
    st.subheader("Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = data.to_csv()
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_name.replace(' ', '_')}_data.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
**Indian Stock Forecast Pro** | Data from Yahoo Finance

Features:
- Real-time Indian stock data (NSE/BSE)
- Price forecasting using Facebook Prophet
- Technical indicators (RSI, Moving Averages, Bollinger Bands)
- Interactive charts with Plotly
- Download data in CSV format
- Beautiful responsive UI

*For educational purposes only. Not financial advice.*
""")