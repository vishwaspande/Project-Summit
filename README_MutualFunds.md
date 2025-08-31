# 🏦 Mutual Fund Integration Guide

## 🚀 Enhanced Features

Your Indian Stock Market AI Agent now supports **real mutual fund data** in addition to ETFs! This includes popular funds like:

### 📊 Supported Mutual Funds

#### Large Cap Funds
- **PPFAS Flexi Cap Direct Growth** (`PPFAS_FLEXI_CAP_DIRECT`)
- **Axis Bluechip Direct Growth** (`AXIS_BLUECHIP_DIRECT`)
- **Mirae Asset Large Cap Direct Growth** (`MIRAE_LARGECAP_DIRECT`)
- **HDFC Top 100 Direct Growth** (`HDFC_TOP100_DIRECT`)
- **ICICI Prudential Bluechip Direct Growth** (`ICICI_BLUECHIP_DIRECT`)

#### Mid Cap Funds
- **Axis Midcap Direct Growth** (`AXIS_MIDCAP_DIRECT`)
- **HDFC Mid-Cap Opportunities Direct Growth** (`HDFC_MIDCAP_DIRECT`)
- **Kotak Emerging Equity Direct Growth** (`KOTAK_EMERGING_EQUITY_DIRECT`)

#### Small Cap Funds
- **Axis Small Cap Direct Growth** (`AXIS_SMALLCAP_DIRECT`)
- **HDFC Small Cap Direct Growth** (`HDFC_SMALLCAP_DIRECT`)
- **SBI Small Cap Direct Growth** (`SBI_SMALLCAP_DIRECT`)

#### Flexi Cap Funds
- **Parag Parikh Flexi Cap Direct Growth** (`PARAG_FLEXI_CAP_DIRECT`)
- **Canara Robeco Flexi Cap Direct Growth** (`CANARA_FLEXI_CAP_DIRECT`)
- **Quant Active Fund Direct Growth** (`QUANT_ACTIVE_DIRECT`)

#### International Funds
- **Motilal Oswal Nasdaq 100 Direct Growth** (`MOTILAL_NASDAQ_DIRECT`)
- **Edelweiss US Value Equity Direct Growth** (`EDELWEISS_US_VALUE_DIRECT`)

#### Debt Funds
- **Axis Liquid Direct Growth** (`AXIS_LIQUID_DIRECT`)
- **HDFC Liquid Direct Growth** (`HDFC_LIQUID_DIRECT`)
- **ICICI Prudential Liquid Direct Growth** (`ICICI_LIQUID_DIRECT`)

## 🔧 Setup for Real Mutual Fund Data

### Option 1: Demo Mode (Default)
The system works out-of-the-box with demo data for mutual funds. No additional configuration needed.

### Option 2: Real NAV Data (Recommended)

#### Step 1: Get RapidAPI Key
1. Go to [RapidAPI.com](https://rapidapi.com)
2. Sign up for a free account
3. Subscribe to "Latest Mutual Fund NAV" API
4. Get your API key

#### Step 2: Configure Environment
```bash
# Add to your environment variables
export RAPIDAPI_KEY="your-rapidapi-key-here"

# Or add to your .env file
echo "RAPIDAPI_KEY=your-rapidapi-key-here" >> .env
```

#### Step 3: Alternative - AMFI Direct
The system also tries to fetch data directly from AMFI (Association of Mutual Funds in India) as a fallback.

## 💡 Usage Examples

### Analyze PPFAS Flexi Cap Fund
```python
from indian_stock_market_agent import IndianStockMarketAgent

agent = IndianStockMarketAgent("your-anthropic-key")

# Analyze the popular PPFAS fund
analysis = agent.analyze_indian_mutual_fund("PPFAS_FLEXI_CAP_DIRECT")
print(analysis)
```

### Mixed Portfolio with Stocks and Mutual Funds
```python
# Portfolio with both stocks and mutual funds
mixed_portfolio = {
    'RELIANCE': {'qty': 100, 'avg_price': 2400.0},           # Stock
    'PPFAS_FLEXI_CAP_DIRECT': {'qty': 1000, 'avg_price': 45.0},  # Mutual Fund
    'NIFTYBEES': {'qty': 500, 'avg_price': 180.0},          # ETF
    'AXIS_BLUECHIP_DIRECT': {'qty': 800, 'avg_price': 55.0}     # Mutual Fund
}

# Comprehensive portfolio analysis
portfolio_analysis = agent.analyze_portfolio_indian(mixed_portfolio)
print(portfolio_analysis)
```

### Get NAV Data
```python
# Get live mutual fund data
fund_data = agent.get_indian_stock_data([
    "PPFAS_FLEXI_CAP_DIRECT",
    "AXIS_MIDCAP_DIRECT", 
    "HDFC_SMALLCAP_DIRECT"
])

for fund, data in fund_data.items():
    if not data.get('error'):
        print(f"{data['scheme_name']}: ₹{data['nav']:.2f}")
        print(f"Date: {data['nav_date']}")
        print(f"Source: {data['data_source']}")
```

## 🌐 Web Dashboard Usage

### Adding Mutual Funds to Portfolio
1. Launch: `streamlit run indian_dashboard.py`
2. Go to **Portfolio** tab
3. Select **"Mutual Funds/ETFs"** as investment type
4. Choose from the dropdown (e.g., PPFAS_FLEXI_CAP_DIRECT)
5. Enter units and average NAV
6. Click "Add Position"

### Mutual Fund Analysis Tab
1. Go to **"🏦 Mutual Funds/ETFs"** tab
2. Select any mutual fund from the dropdown
3. View NAV, scheme details, and category
4. Get AI-powered analysis with fund-specific insights

## 🔍 Key Features

### Real NAV Data
- **Live Updates**: Latest NAV from AMFI/RapidAPI
- **Historical Tracking**: NAV date and source information
- **Fallback System**: Multiple data sources for reliability

### Fund-Specific Analysis
- **Performance vs Benchmark**: Compare with category peers
- **Cost Analysis**: Expense ratio and fee structure
- **Risk Assessment**: Volatility and risk metrics
- **SIP Recommendations**: Systematic investment plans

### Portfolio Integration
- **Mixed Assets**: Stocks, ETFs, and mutual funds in one portfolio
- **Category Allocation**: Automatic categorization by fund type
- **Performance Attribution**: Individual fund contribution analysis

## 🎯 Data Sources Hierarchy

1. **RapidAPI** (Primary): Real-time NAV data with API key
2. **AMFI Direct** (Secondary): Free access to official NAV data
3. **Demo Data** (Fallback): Synthetic data for testing/development

## ⚠️ Important Notes

### Mutual Fund vs ETF Differences
- **Mutual Funds**: NAV updated once daily after market close
- **ETFs**: Real-time pricing during market hours
- **Trading**: MFs through AMCs, ETFs through stock exchanges

### Data Limitations
- **NAV Timing**: Mutual fund NAVs are T+1 (published next day)
- **Weekends**: No NAV updates on non-trading days
- **API Limits**: RapidAPI has rate limits based on subscription

### Investment Disclaimer
- **Not Financial Advice**: All analysis for informational purposes
- **Due Diligence**: Verify fund details with official sources
- **Risk Factors**: Mutual funds are subject to market risk

## 🔧 Adding More Funds

### Step 1: Get AMFI Code
Find the AMFI code from:
- Fund fact sheets
- AMFI website
- Registrar websites

### Step 2: Add to Agent
```python
# In indian_stock_market_agent.py
self.popular_mutual_funds.update({
    'YOUR_FUND_NAME': 'AMFI_CODE',
    'SBI_BLUECHIP_DIRECT': '109734',  # Example
})
```

### Step 3: Update Categories
```python
# Add to appropriate sector
self.indian_sectors['Large Cap Funds'].append('YOUR_FUND_NAME')
```

## 🚀 Advanced Features

### Automatic SIP Calculator
```python
# SIP analysis for mutual funds
sip_analysis = agent._call_claude_indian(f"""
Analyze SIP investment in PPFAS_FLEXI_CAP_DIRECT:
- Monthly SIP amount: ₹10,000
- Investment horizon: 10 years
- Current NAV: ₹{nav_data['nav']}
- Risk profile: Moderate

Provide:
1. Expected returns analysis
2. SIP vs lump-sum comparison
3. Goal-based allocation
4. Exit strategy recommendations
""")
```

### Category Comparison
```python
# Compare funds in same category
large_cap_funds = ['PPFAS_FLEXI_CAP_DIRECT', 'AXIS_BLUECHIP_DIRECT', 'MIRAE_LARGECAP_DIRECT']
comparison_data = agent.get_indian_stock_data(large_cap_funds)

# AI comparison analysis
comparison = agent._call_claude_indian(f"""
Compare these Large Cap funds for Indian investor:
{chr(10).join([f"- {fund}: ₹{data['nav']:.2f}" for fund, data in comparison_data.items()])}

Rank them by:
1. Performance consistency
2. Expense ratio efficiency  
3. Risk-adjusted returns
4. Fund manager track record
""")
```

## 📈 Success Metrics

With mutual fund integration, you can now:
- ✅ Track complete investment portfolio (stocks + MFs + ETFs)
- ✅ Analyze 25+ popular Indian mutual fund schemes
- ✅ Get AI-powered fund selection advice
- ✅ Compare direct vs regular plans
- ✅ Monitor SIP performance
- ✅ Optimize asset allocation across categories

---

**Ready to analyze your mutual fund investments with AI power! 🚀**

*For technical support, ensure proper API configuration and check error logs for troubleshooting.*