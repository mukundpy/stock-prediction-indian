# Indian Stock Forecast Pro 📈

A powerful Streamlit web application for analyzing and forecasting Indian stock prices with professional-grade technical analysis.

## ✨ Features

- 📊 **Real-time Indian Stock Data** - Access NSE/BSE data for 20+ major Indian stocks
- 🔮 **AI-Powered Price Forecasting** - Facebook Prophet for accurate price predictions
- 📈 **Technical Indicators** 
  - RSI (Relative Strength Index) - Identify overbought/oversold conditions
  - Moving Averages (20/50) - Spot trends with MA crossovers
  - Bollinger Bands - Volatility analysis and support/resistance levels
- 📱 **Interactive Charts** - Candlestick & line charts with Plotly
- 💹 **Comprehensive Analysis**
  - Returns analysis (daily returns, volatility, total return)
  - Volume analysis with visualization
  - 52-week high/low metrics
- 📥 **Export Data** - Download historical data in CSV format
- 🎨 **Beautiful UI** - Clean, responsive Indian-themed interface

## 📦 Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/mukundpy/starters.git
cd starters
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run main.py
```

4. Open in browser:
Navigate to `http://localhost:8501`

### Cloud Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your repository
4. Choose `main.py` as the entry point
5. App will be live in 2-5 minutes

## 🎯 How to Use

1. **Select a Stock** - Choose from 20+ major Indian companies in the sidebar
2. **Set Time Period** - Select 1-month to 5-year historical data
3. **Choose Chart Type** - View as line chart or candlestick chart
4. **Adjust Forecast** - Change the number of days to forecast (1-90 days)
5. **Analyze Indicators** - Review RSI, Moving Averages, and Bollinger Bands
6. **View Predictions** - See AI-powered price forecasts with confidence intervals
7. **Download Data** - Export your analysis as CSV

## 📚 Technical Stack

- **Frontend:** Streamlit
- **Data:** yfinance (Yahoo Finance API)
- **Forecasting:** Facebook Prophet
- **Visualization:** Plotly
- **Data Processing:** Pandas, NumPy
- **Python Version:** 3.8+

## 📊 Supported Stocks

- **Banking:** HDFC Bank, ICICI Bank, Axis Bank, SBI
- **IT:** TCS, Infosys, Wipro, HCL Tech
- **Energy:** Reliance, ONGC, Power Grid
- **Pharma:** Sun Pharma
- **Consumer:** HUL, Nestle, ITC, Titan
- **Finance:** Bajaj Finance
- **Coal:** Coal India
- **Auto:** Tata Motors
- **Paint:** Asian Paints

## ⚠️ Disclaimer

This tool is for **educational purposes only**. It is not financial advice. Stock market investments carry risk. Always consult with a financial advisor before making investment decisions.

## 🐛 Troubleshooting

**Q: "Error loading data"**
- Check your internet connection
- Verify stock ticker is correct
- Try a different stock symbol

**Q: Forecast not working?**
- Ensure you have at least 60 days of historical data
- Try with a longer time period (e.g., 1 year)

**Q: App running slowly?**
- Reduce forecast days
- Try a shorter time period
- Refresh the page

## 📝 License

MIT License - Feel free to use and modify

## 🤝 Contributing

Contributions welcome! Feel free to submit issues and pull requests.

---

**Made with ❤️ for Indian stock market enthusiasts**