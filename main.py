import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
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
        st.error(f"Error loading data: 
