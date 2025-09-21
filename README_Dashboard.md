# 🇮🇳 Indian Stock Market - AI Dashboard

A comprehensive, real-time financial dashboard specifically designed for Indian investors, featuring advanced risk-adjusted performance metrics, AI-powered analysis, and seamless integration with Indian markets.

## 📊 Dashboard Overview

The Indian Stock Market Dashboard is a powerful Streamlit-based web application that provides:
- **Real-time NSE/BSE stock tracking** with live market data
- **Advanced portfolio risk analysis** using professional financial metrics
- **AI-powered investment insights** with Claude integration
- **Comprehensive mutual fund analysis** with AMFI data integration
- **Global market correlation** analysis for Indian investors

## ✨ Key Features

### 🎯 **Core Functionality**
- **Live Market Data**: Real-time prices from Yahoo Finance for 500+ Indian stocks
- **Mutual Fund Integration**: NAV data from AMFI via MFApi for 50+ popular funds
- **Market Hours Detection**: Automatically adjusts data refresh based on Indian market hours
- **Multi-Asset Support**: Stocks, ETFs, Mutual Funds in single unified interface

### 📈 **Advanced Risk Analytics**
- **Professional Risk Metrics**: Sharpe Ratio, Sortino Ratio, Maximum Drawdown
- **Benchmark Analysis**: Alpha, Beta, Information Ratio vs NIFTY 50
- **Value at Risk**: Daily VaR calculations at 95% confidence level
- **Volatility Analysis**: Annualized volatility with risk categorization
- **Historical Data**: 1-year lookback using actual market data (not synthetic)

### 🤖 **AI-Powered Analysis**
- **Claude Integration**: Advanced AI analysis for stocks and portfolio
- **Market Sentiment**: AI-driven market outlook and recommendations
- **Stock Analysis**: Personalized buy/sell/hold recommendations
- **Global Correlation**: AI analysis of global market impact on Indian investments

### 💼 **Portfolio Management**
- **Multi-Asset Tracking**: Stocks, mutual funds, ETFs in single portfolio
- **Real-time P&L**: Live profit/loss tracking with percentage calculations
- **Risk Assessment**: Automated risk scoring (LOW/MEDIUM/HIGH)
- **Diversification Analysis**: Sector allocation and concentration metrics
- **Performance Attribution**: Best/worst performer identification

## 🏗️ **Technical Architecture**

### **Data Sources**
- **Yahoo Finance API**: Real-time stock and ETF prices
- **AMFI API (MFApi)**: Official mutual fund NAV data
- **NIFTY 50 Benchmark**: For alpha, beta, and correlation analysis
- **Global Indices**: International market data for context

### **Technology Stack**
```
Frontend: Streamlit (Python web framework)
Data Processing: pandas, numpy
Visualization: plotly (interactive charts)
AI Integration: Anthropic Claude API
Market Data: yfinance, requests
Risk Calculations: Custom implementations
```

### **Performance Optimizations**
- **Async Data Fetching**: Parallel API calls for faster loading
- **Intelligent Caching**: Reduced API calls during market hours
- **Error Handling**: Graceful fallbacks for data unavailability
- **Mobile Responsive**: Works on desktop, tablet, and mobile

## 🎯 **Risk Metrics Explained**

### **Core Risk Metrics**
| Metric | Formula | Good Range | Interpretation |
|--------|---------|------------|----------------|
| **Sharpe Ratio** | (Return - Risk-free) / Volatility | > 1.0 | Risk-adjusted returns quality |
| **Sortino Ratio** | (Return - Risk-free) / Downside Dev | > 1.5 | Downside risk focus |
| **Max Drawdown** | Largest Peak-to-Trough Decline | < 20% | Maximum loss potential |

### **Risk Assessment Metrics**
| Metric | Calculation | Low Risk | High Risk |
|--------|-------------|----------|-----------|
| **VaR (95%)** | 5th Percentile Daily Return | < 2% | > 5% |
| **Annual Volatility** | Std Dev × √252 | < 15% | > 25% |

### **Benchmark-Relative Metrics**
| Metric | Vs NIFTY 50 | Excellent | Poor |
|--------|-------------|-----------|------|
| **Alpha** | Excess Return over Expected | > 2% | < -2% |
| **Beta** | Market Sensitivity | 0.8-1.2 | < 0.6 or > 1.4 |
| **Information Ratio** | Active Return / Tracking Error | > 0.75 | < 0 |

## 🚀 **Installation & Setup**

### **Prerequisites**
```bash
Python 3.8+
pip (Python package manager)
Git (for cloning repository)
```

### **Step 1: Clone Repository**
```bash
git clone <repository-url>
cd indian-stock-dashboard
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Environment Setup**
```bash
# Optional: For AI features
export ANTHROPIC_API_KEY="your-claude-api-key"
```

### **Step 4: Run Dashboard**
```bash
streamlit run indian_dashboard.py
```

The dashboard will open at `http://localhost:8501`

## 📋 **Required Dependencies**

```txt
streamlit>=1.28.0
plotly>=5.15.0
yfinance>=0.2.50
pandas>=2.0.0
numpy>=1.24.0
anthropic>=0.3.0
requests>=2.31.0
pytz>=2023.3
```

## 🎮 **User Guide**

### **Tab 1: 📊 Market Overview**
- **Live Indices**: NIFTY 50 and SENSEX with real-time updates
- **Watchlist**: Customizable stock watchlist with quick metrics
- **Market Status**: Live market hours and USD/INR rate

### **Tab 2: 📈 Stock Analysis**
- **Stock Selection**: 200+ popular Indian stocks across sectors
- **Interactive Charts**: Candlestick charts with multiple timeframes
- **Key Metrics**: P/E ratio, market cap, day high/low
- **AI Analysis**: Claude-powered investment recommendations

### **Tab 3: 🏦 Mutual Funds/ETFs**
- **Fund Analysis**: 50+ popular Indian mutual funds
- **NAV Tracking**: Official AMFI data with historical performance
- **Benchmark Comparison**: Fund vs benchmark performance charts
- **AI Insights**: Detailed fund analysis with risk assessment

### **Tab 4: 💼 Portfolio Management**
- **Multi-Asset Support**: Add stocks, mutual funds, ETFs
- **Real-time Tracking**: Live P&L with color-coded performance
- **Risk Scoring**: Automated portfolio risk assessment
- **Advanced Analysis**: Integration with performance agent

### **Tab 5: 📈 Risk Metrics** *(New!)*
- **Dedicated Risk Tab**: Comprehensive risk analysis section
- **9 Professional Metrics**: Complete risk-adjusted performance suite
- **Data Transparency**: Shows exactly what data is being used
- **Detailed Explanations**: Professional-grade interpretations

### **Tab 6: 🌍 Global Markets**
- **International Indices**: S&P 500, NASDAQ, Nikkei, FTSE, etc.
- **Regional Grouping**: Organized by North America, Asia, Europe
- **Commodities**: Gold, oil, silver tracking
- **AI Correlation**: Impact analysis on Indian markets

### **Tab 7: 🤖 AI Assistant**
- **Natural Language**: Ask questions about markets and stocks
- **Market Insights**: AI-powered market sentiment analysis
- **Quick Questions**: Pre-built queries for common scenarios
- **Indian Context**: All analysis tailored for Indian investors

## ⚙️ **Configuration Options**

### **API Keys**
- **Claude API**: Enter in sidebar for AI features (optional)
- **No Registration Required**: Core functionality works without API keys

### **Customization**
- **Watchlist**: Add/remove stocks from sidebar
- **Data Refresh**: Manual refresh or auto-refresh during market hours
- **Time Zones**: Automatically handles IST (Indian Standard Time)

### **Data Sources Config**
```python
# Automatic source selection:
# Stocks: {symbol}.NS via yfinance
# Mutual Funds: AMFI scheme codes via MFApi
# ETFs: {symbol}.NS via yfinance
# Benchmark: ^NSEI (NIFTY 50)
```

## 🔍 **Troubleshooting**

### **Common Issues**

**Issue**: Dashboard won't load
```
Solution: Check Python version (need 3.8+) and install requirements.txt
```

**Issue**: No stock data appearing
```
Solution: Check internet connection, Yahoo Finance may be temporarily down
```

**Issue**: Mutual fund data missing
```
Solution: AMFI API may be slow, try refreshing or check scheme codes
```

**Issue**: AI features not working
```
Solution: Add valid Anthropic API key in sidebar
```

### **Performance Tips**
- **Data Refresh**: Use manual refresh during market hours to avoid API limits
- **Portfolio Size**: Keep portfolio under 20 positions for optimal performance
- **Chart Periods**: Use shorter periods (1M) during market hours, longer (1Y) for analysis

## 🔐 **Privacy & Security**

### **Data Handling**
- **No Personal Data Storage**: All data processed in real-time
- **API Keys**: Stored temporarily in session, never saved to disk
- **Market Data**: Public data only, no personal financial information
- **Session-Based**: Portfolio data cleared when browser closes

### **External Connections**
- **Yahoo Finance**: Public market data API
- **AMFI/MFApi**: Official Indian mutual fund data
- **Anthropic**: AI analysis (only if API key provided)
- **No Third-Party Tracking**: No analytics or tracking scripts

## 📊 **Data Accuracy & Disclaimers**

### **Data Sources Reliability**
- **Stock Prices**: Yahoo Finance (15-20 minute delay)
- **Mutual Fund NAV**: Official AMFI data (updated daily)
- **Risk Calculations**: Based on 1-year historical data
- **AI Analysis**: For informational purposes only

### **Investment Disclaimers**
⚠️ **Important Notice**:
- This dashboard is for **educational and informational purposes only**
- **Not investment advice**: All analysis should be supplemented with professional advice
- **Market Risk**: All investments carry risk, past performance doesn't guarantee future results
- **Data Accuracy**: While we use reliable sources, real-time accuracy cannot be guaranteed
- **AI Limitations**: AI analysis is based on historical data and general market patterns

## 🤝 **Contributing**

### **Feature Requests**
- Open GitHub issues for feature requests
- Include detailed use case and expected behavior
- Consider Indian market context for all features

### **Bug Reports**
```
Include:
1. Steps to reproduce
2. Expected vs actual behavior
3. Browser and Python versions
4. Portfolio composition (if relevant)
```

### **Development Setup**
```bash
# Development mode
git clone <repo>
cd dashboard
pip install -r requirements.txt
streamlit run indian_dashboard.py --debug
```

## 📞 **Support & Documentation**

### **Quick Help**
- **In-App Help**: Hover over metrics for explanations
- **Expander Sections**: Detailed guides within each tab
- **Error Messages**: Self-explanatory with suggested fixes

### **Advanced Features**
- **Performance Agent**: Advanced portfolio analysis (if available)
- **Market Agent**: Enhanced AI analysis capabilities
- **Custom Risk Metrics**: Extensible calculation framework

## 🎯 **Roadmap & Future Features**

### **Planned Enhancements**
- [ ] **Options Trading**: Support for Indian options analysis
- [ ] **Sector Analysis**: Detailed sector-wise performance metrics
- [ ] **Alert System**: Price and risk-based alerting
- [ ] **Export Features**: PDF reports and CSV data export
- [ ] **Multi-Language**: Hindi and other regional language support

### **Technical Improvements**
- [ ] **Database Integration**: Portfolio persistence across sessions
- [ ] **Real-time Updates**: WebSocket-based live data
- [ ] **Mobile App**: Native mobile application
- [ ] **API Access**: RESTful API for external integrations

---

## 📝 **Version Information**

**Current Version**: 2.0
**Last Updated**: January 2025
**Python Compatibility**: 3.8+
**Streamlit Version**: 1.28+

**Major Features in v2.0**:
- ✅ Dedicated Risk Metrics tab
- ✅ Real historical data (no synthetic data)
- ✅ 9 professional risk metrics
- ✅ NIFTY 50 benchmark integration
- ✅ Enhanced AI analysis
- ✅ Comprehensive mutual fund support

---

*Built with ❤️ for Indian investors by [Your Name/Team]*

**⭐ If you find this dashboard useful, please consider giving it a star on GitHub!**