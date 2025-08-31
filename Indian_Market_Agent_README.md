# 🇮🇳 Indian Stock Market AI Agent & Dashboard

A comprehensive AI-powered Indian stock market analysis platform with real-time data, portfolio management, and intelligent insights tailored specifically for Indian markets.

## 🚀 Features

### Core Capabilities
- **Real-time NSE/BSE Stock Monitoring** - Live price tracking with 5-minute intervals during market hours
- **Mutual Funds & ETF Analysis** - Comprehensive support for 20+ popular Indian ETFs including NIFTYBEES, BANKBEES, GOLDBEES
- **AI-Powered Analysis** - Claude-4-powered investment recommendations and market insights
- **Portfolio Management** - Track mixed portfolios with stocks and mutual funds
- **Indian Market Context** - Market hours, currency rates, sector analysis tailored for Indian investors
- **Interactive Dashboard** - Beautiful Streamlit web interface with real-time charts

### Market Coverage
- **Stocks**: 50+ popular Indian stocks across all major sectors
- **Indices**: Nifty 50, Sensex, Bank Nifty, IT Index
- **ETFs**: Equity, Sectoral, International, and Commodity ETFs
- **Currency**: Real-time USD/INR rates and impact analysis

## 📁 Project Structure

```
├── indian_stock_market_agent.py    # Core AI agent with analysis capabilities
├── indian_dashboard.py             # Streamlit web dashboard
└── README.md                       # This file
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Anthropic API key

### Install Dependencies
```bash
pip install streamlit plotly yfinance anthropic pandas numpy pytz requests
```

### Environment Setup
```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"

# Or create a .env file
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

Get your API key from: [https://console.anthropic.com](https://console.anthropic.com)

## 🚦 Quick Start

### Option 1: Web Dashboard (Recommended)
```bash
streamlit run indian_dashboard.py
```
Access at: http://localhost:8501

### Option 2: Python Script
```bash
python indian_stock_market_agent.py
```

## 🏗️ Architecture Overview

### IndianStockMarketAgent Class

The core agent (`indian_stock_market_agent.py`) provides:

#### Key Methods

##### Data Retrieval
```python
# Fetch comprehensive stock/ETF data
get_indian_stock_data(symbols: List[str], period: str = "1d") -> Dict

# Get major indices (Nifty, Sensex, Bank Nifty, IT)
get_nifty_sensex_data() -> Dict

# Current USD/INR exchange rate
get_usd_inr_rate() -> float
```

##### Analysis Functions
```python
# Individual stock analysis
analyze_indian_stock(symbol: str) -> str

# Mutual fund/ETF analysis
analyze_indian_mutual_fund(symbol: str) -> str

# Portfolio analysis
analyze_portfolio_indian(portfolio: Dict[str, Dict]) -> str

# Market outlook
market_outlook_indian() -> str
```

##### Utility Functions
```python
# Check if market is open (9:15 AM - 3:30 PM IST, Mon-Fri)
is_market_open() -> bool

# Get sector for any stock/ETF
get_indian_sector(symbol: str) -> str
```

### Dashboard Components

The Streamlit dashboard (`indian_dashboard.py`) includes:

1. **Market Overview Tab**
   - Real-time indices (Nifty, Sensex)
   - Watchlist tracking
   - Market status and timing

2. **Stock Analysis Tab**
   - Interactive price charts
   - Technical indicators (SMA 20/50)
   - AI-powered stock analysis
   - Fundamental data (P/E, market cap, etc.)

3. **Mutual Funds/ETFs Tab**
   - NAV tracking and charts
   - Category-wise fund analysis
   - Performance comparison
   - AI fund recommendations

4. **Portfolio Management Tab**
   - Mixed portfolio support (stocks + funds)
   - Real-time P&L calculation
   - Sector allocation analysis
   - AI portfolio optimization

5. **AI Assistant Tab**
   - Natural language market queries
   - Quick question templates
   - Context-aware responses

## 💡 Usage Examples

### Basic Stock Analysis
```python
from indian_stock_market_agent import IndianStockMarketAgent

# Initialize agent
agent = IndianStockMarketAgent("your-api-key")

# Analyze a stock
analysis = agent.analyze_indian_stock("RELIANCE")
print(analysis)

# Get live data
data = agent.get_indian_stock_data(["TCS", "INFY", "HDFCBANK"])
```

### Portfolio Analysis
```python
# Define portfolio
portfolio = {
    'RELIANCE': {'qty': 100, 'avg_price': 2400.0},
    'NIFTYBEES': {'qty': 200, 'avg_price': 180.0},  # ETF
    'HDFCBANK': {'qty': 50, 'avg_price': 1500.0}
}

# Get comprehensive analysis
portfolio_analysis = agent.analyze_portfolio_indian(portfolio)
print(portfolio_analysis)
```

### Market Outlook
```python
# Get current market analysis
outlook = agent.market_outlook_indian()
print(outlook)

# Check if market is open
if agent.is_market_open():
    print("Market is currently open for trading")
```

## 📊 Supported Assets

### Stocks (50+ symbols)
- **IT**: TCS, INFY, WIPRO, TECHM, HCLTECH
- **Banking**: HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK
- **Consumer**: RELIANCE, HINDUNILVR, ITC, NESTLEIND, BRITANNIA
- **Auto**: MARUTI, TATAMOTORS, BAJAJ-AUTO, M&M, HEROMOTOCO
- **Pharma**: SUNPHARMA, DRREDDY, CIPLA, BIOCON, AUROPHARMA
- **Oil & Gas**: ONGC, IOC, BPCL, HINDPETRO
- **Metals**: TATASTEEL, HINDALCO, JSWSTEEL, VEDL
- **Infrastructure**: LT, ADANIPORTS, POWERGRID, NTPC

### ETFs & Mutual Funds (20+ symbols)
- **Equity ETFs**: NIFTYBEES, JUNIORBEES, BANKBEES
- **Sectoral ETFs**: ITBEES, PHARMABES, AUTOBEES, FMCGBEES
- **International ETFs**: HNGSNGBEES, NETFLTBEES
- **Commodity ETFs**: GOLDBEES, GOLDSHARE
- **Specialized ETFs**: PSUBEES, METALBEES, REALTYBEES

### Indices
- **NIFTY** (^NSEI) - Nifty 50 Index
- **SENSEX** (^BSESN) - BSE Sensex
- **NIFTY_BANK** (^NSEBANK) - Nifty Bank Index
- **NIFTY_IT** (^CNXIT) - Nifty IT Index

## 🎯 Key Features Explained

### Indian Market Context
- **Market Hours**: Automatically detects IST market hours (9:15 AM - 3:30 PM)
- **Currency Impact**: Real-time USD/INR rates with sector impact analysis
- **Regulatory Awareness**: Tax implications, LTCG/STCG considerations
- **Sectoral Analysis**: Government policy impact on different sectors

### AI Analysis Capabilities
- **Investment Recommendations**: BUY/HOLD/SELL with detailed reasoning
- **Risk Assessment**: Company, sector, and market risk analysis
- **Target Pricing**: Price targets with timelines
- **Portfolio Optimization**: Rebalancing suggestions and sector allocation
- **Market Timing**: Entry/exit strategies based on technical and fundamental analysis

### Real-time Data Features
- **Live Prices**: 5-minute interval updates during market hours
- **Volume Analysis**: Volume spikes and unusual activity detection
- **Technical Indicators**: SMA 20/50, price vs moving averages
- **Fundamental Metrics**: P/E ratios, market cap, dividend yields

### Portfolio Management
- **Mixed Assets**: Support for stocks, ETFs, and mutual funds in single portfolio
- **P&L Tracking**: Real-time profit/loss calculation with percentage returns
- **Sector Allocation**: Automatic sector-wise portfolio breakdown
- **Performance Attribution**: Individual position performance analysis

## 📈 Dashboard Usage Guide

### Setting Up
1. Launch dashboard: `streamlit run indian_dashboard.py`
2. Enter your Anthropic API key in the sidebar
3. Start with the Market Overview tab

### Adding Investments to Portfolio
1. Go to Portfolio tab
2. Select investment type (Stocks or Mutual Funds/ETFs)
3. Choose symbol, quantity, and average price
4. Click "Add Position"

### Getting AI Analysis
1. Select any stock or ETF
2. Click "Generate AI Analysis" 
3. Choose analysis type:
   - Complete analysis
   - Risk assessment
   - Buy/sell recommendation
   - Portfolio review

### Watchlist Management
- Add/remove stocks from sidebar
- View real-time prices in Market Overview
- Set up for quick monitoring

## 🔍 Advanced Features

### Caching System
- **Analysis Cache**: 5-minute cache for API calls to reduce costs
- **Price Cache**: Intelligent caching during market hours
- **Session State**: Persistent portfolio and settings

### Error Handling
- **Fallback Pricing**: Uses average price when live data fails
- **Data Validation**: Handles missing or invalid data gracefully
- **API Limits**: Built-in rate limiting and error recovery

### Performance Optimization
- **Parallel Data Fetching**: Multiple API calls executed simultaneously
- **Selective Updates**: Only refreshes necessary data
- **Memory Management**: Efficient data structures for large portfolios

## 🛠️ Customization

### Adding New Stocks/ETFs
```python
# In indian_stock_market_agent.py, update the dictionaries:

self.popular_indian_stocks['YOUR_SYMBOL'] = 'YOUR_SYMBOL.NS'
self.popular_mutual_funds['YOUR_ETF'] = 'YOUR_ETF.NS'

# Add to appropriate sector
self.indian_sectors['YOUR_SECTOR'].append('YOUR_SYMBOL')
```

### Modifying Analysis Prompts
```python
# In _call_claude_indian method, customize the context:
indian_context = f"""
Your custom market context here...
{prompt}
Your custom instructions here...
"""
```

### Dashboard Customization
```python
# In indian_dashboard.py, modify the CSS styles:
st.markdown("""
<style>
    .your-custom-class {
        /* Your custom styles */
    }
</style>
""", unsafe_allow_html=True)
```

## 🚨 Limitations & Considerations

### Data Source Limitations
- Uses Yahoo Finance API (free but may have delays)
- Some mutual fund data might be limited
- Weekend/holiday data may be stale

### API Usage
- Claude API has rate limits and costs
- Analysis results are cached for 5 minutes
- Large portfolios may require multiple API calls

### Market Data Accuracy
- Real-time data may have 15-20 minute delays
- During high volatility, prices might be outdated
- Always verify with official sources before trading

## 🔒 Security & Privacy

### API Key Security
- Never commit API keys to version control
- Use environment variables or secure config files
- Rotate keys regularly

### Data Privacy
- No trading data is stored permanently
- Session data cleared on browser refresh
- No external data transmission except to required APIs

## 📝 Contributing

### Adding Features
1. Fork the repository
2. Create feature branch
3. Add comprehensive documentation
4. Test thoroughly with Indian market data
5. Submit pull request

### Bug Reports
Include:
- Detailed error messages
- Steps to reproduce
- Market conditions when error occurred
- System and Python version info

## 📜 License

This project is for educational and research purposes. Not financial advice.

### Disclaimer
- **Not Financial Advice**: All analysis is for informational purposes only
- **Market Risk**: Stock and mutual fund investments carry risk of loss
- **Data Accuracy**: Verify all data with official sources before making investment decisions
- **Regulatory Compliance**: Ensure compliance with local regulations

## 🆘 Support & Troubleshooting

### Common Issues

**"No data available" errors:**
- Check internet connection
- Verify symbol spelling
- Try different time periods

**API key errors:**
- Ensure key is correctly set in environment
- Check API key permissions on Anthropic console
- Verify sufficient API credits

**Slow performance:**
- Reduce number of symbols analyzed simultaneously
- Check network connectivity
- Clear browser cache for dashboard

### Getting Help
1. Check the error messages in the dashboard
2. Review the console output for detailed logs
3. Ensure all dependencies are properly installed
4. Verify API key configuration

### Performance Tips
- Use specific time periods for analysis
- Limit portfolio size for faster calculations
- Cache results when possible
- Monitor API usage to avoid rate limits

---

**Made with ❤️ for Indian Stock Market Investors**

*Last Updated: December 2024*