#!/usr/bin/env python3
"""
Indian Stock Market Dashboard - Complete Working Streamlit Web App

Features for Indian Markets:
- NSE/BSE stock tracking with real-time data
- Nifty 50 & Sensex monitoring
- Indian market hours display and status
- Rupee-denominated analysis
- Sector-wise analysis (IT, Banking, Pharma, Auto, etc.)
- Portfolio management with Indian context
- AI-powered stock analysis
- Real-time alerts and monitoring
- Correct mutual fund data from AMFI

Setup:
pip install streamlit plotly yfinance anthropic pandas numpy pytz requests
export ANTHROPIC_API_KEY="your-key-here"

Run:
streamlit run indian_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
import anthropic
from datetime import datetime, timedelta
import pytz
import time
import json
import os
import numpy as np
from typing import Dict, List
import warnings
import requests
warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Indian Stock Market - AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the agent after page config
try:
    from indian_stock_market_agent import IndianStockMarketAgent
    AGENT_AVAILABLE = True
except ImportError:
    try:
        from indian_stock_agent import IndianStockMarketAgent
        AGENT_AVAILABLE = True
    except ImportError:
        AGENT_AVAILABLE = False

# Import the investment performance agent
try:
    from investment_performance_agent import analyze_portfolio_performance
    PERFORMANCE_AGENT_AVAILABLE = True
except ImportError:
    PERFORMANCE_AGENT_AVAILABLE = False

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E, #138808);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .metric-negative {
        color: #f44336;
        font-weight: bold;
    }
    .indian-flag {
        background: linear-gradient(to bottom, #FF6B35 33%, white 33%, white 66%, #138808 66%);
        height: 15px;
        width: 25px;
        display: inline-block;
        margin-right: 10px;
        border-radius: 3px;
    }
    .status-open {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-closed {
        color: #f44336;
        font-weight: bold;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
        
        .stSelectbox > div > div {
            font-size: 16px;
        }
        
        .stButton > button {
            width: 100%;
            height: 50px;
            font-size: 18px;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class IndianStockDashboard:
    def __init__(self):
        """Initialize the Indian stock dashboard."""
        
        # Indian timezone
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Popular Indian stocks by category
        self.indian_stocks = {
            'Large Cap IT': ['TCS', 'INFY', 'WIPRO', 'TECHM', 'HCLTECH'],
            'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'Consumer': ['RELIANCE', 'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA'],
            'Auto': ['MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'M&M', 'HEROMOTOCO'],
            'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'BIOCON', 'AUROPHARMA'],
            'Oil & Gas': ['ONGC', 'IOC', 'BPCL', 'HINDPETRO'],
            'Metals': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL'],
            'Infrastructure': ['LT', 'ADANIPORTS', 'POWERGRID', 'NTPC'],
            'Consumer Durables': ['ASIANPAINT', 'BERGER', 'TITAN', 'VOLTAS']
        }
        
        # Mutual Funds with correct names and ETFs with NSE symbols
        self.mutual_funds = {
            # Mutual Funds (require API calls to mfapi.in)
            'Flexi Cap Funds': ['Parag Parikh Flexi Cap Fund - Direct Plan'],
            'Small Cap Funds': ['HDFC Small Cap Fund - Direct Plan - Growth'], 
            'Index Funds': ['HDFC Nifty Next50 Index Fund Direct Growth', 'HDFC Nifty 50 Index Fund - Direct Plan'],
            'Sectoral Funds': ['Nippon India Pharma Fund Direct Growth Plan', 'ICICI Prudential Energy Opportunity Fund - Direct Plan'],
            
            # ETFs (can use yfinance with .NS symbols)
            'Commodity ETFs': ['HDFCGOLDETF.NS', 'HDFCSILVERETF.NS']
        }
        
        # AMFI scheme codes for mutual funds (correct mappings)
        self.scheme_codes = {
            'Parag Parikh Flexi Cap Fund - Direct Plan': '122639',
            'HDFC Small Cap Fund - Direct Plan - Growth': '130503',
            'HDFC Nifty Next50 Index Fund Direct Growth': '149288',
            'HDFC Nifty 50 Index Fund - Direct Plan': '119063',
            'Nippon India Pharma Fund Direct Growth Plan': '118759',
            'ICICI Prudential Energy Opportunity Fund - Direct Plan': '152728'
        }
        
        # Flatten stock list
        self.all_stocks = []
        for stocks in self.indian_stocks.values():
            self.all_stocks.extend(stocks)
        self.all_stocks = sorted(list(set(self.all_stocks)))
        
        # Flatten mutual funds list
        self.all_mutual_funds = []
        for funds in self.mutual_funds.values():
            self.all_mutual_funds.extend(funds)
        self.all_mutual_funds = sorted(list(set(self.all_mutual_funds)))
        
        # Combined list for analysis
        self.all_investments = self.all_stocks + self.all_mutual_funds
        
        # Popular stocks for quick access
        self.popular_stocks = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 
            'ITC', 'SBIN', 'ASIANPAINT', 'MARUTI', 'BAJFINANCE'
        ]
        
        # Initialize session state
        self.init_session_state()
        
        # Setup AI services
        self.claude_client = None
        self.agent = None
        self.setup_ai()
        
        # Initialize agent for performance calculation even without API key
        try:
            from indian_stock_market_agent import IndianStockMarketAgent
            self.agent = IndianStockMarketAgent()  # Initialize without API key for basic functionality
        except Exception as e:
            print(f"Warning: Could not initialize agent for performance analysis: {e}")
    
    def init_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'portfolio': {},
            'watchlist': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY','ITC'],
            'live_data': {},
            'indices_data': {},
            'last_refresh': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def setup_ai(self):
        """Setup AI services."""
        api_key = st.sidebar.text_input(
            "🔑 Claude API Key:",
            type="password",
            help="Get from console.anthropic.com",
            key="claude_api_key_input"
        )
        
        if api_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=api_key)
                if AGENT_AVAILABLE:
                    self.agent = IndianStockMarketAgent(api_key)
                st.sidebar.success("✅ AI Services Connected!")
                return True
            except Exception as e:
                st.sidebar.error(f"❌ AI Error: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Enter API key for AI features")
        
        return False
    
    def is_market_open(self):
        """Check if Indian market is open."""
        now = datetime.now(self.ist)
        current_time = now.time()
        weekday = now.weekday()
        
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        
        if weekday >= 5:  # Weekend
            return False
        
        return market_open <= current_time <= market_close
    
    def get_fund_nav_from_mfapi(self, fund_name: str) -> dict:
        """Get mutual fund NAV from MFApi using scheme code."""
        if fund_name not in self.scheme_codes:
            return {'error': f'Scheme code not found for {fund_name}'}
        
        scheme_code = self.scheme_codes[fund_name]
        
        try:
            url = f"https://api.mfapi.in/mf/{scheme_code}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                api_data = response.json()
                
                if api_data and 'data' in api_data and len(api_data['data']) > 0:
                    latest = api_data['data'][0]  # Most recent NAV
                    previous = api_data['data'][1] if len(api_data['data']) > 1 else latest
                    
                    current_nav = float(latest['nav'])
                    prev_nav = float(previous['nav']) if previous['nav'] != latest['nav'] else current_nav
                    
                    change = current_nav - prev_nav
                    change_pct = (change / prev_nav * 100) if prev_nav > 0 else 0
                    
                    return {
                        'scheme_name': api_data['meta']['scheme_name'],
                        'scheme_code': api_data['meta']['scheme_code'],
                        'fund_house': api_data['meta']['fund_house'],
                        'scheme_type': api_data['meta']['scheme_type'],
                        'scheme_category': api_data['meta']['scheme_category'],
                        'nav': current_nav,
                        'nav_date': latest['date'],
                        'prev_nav': prev_nav,
                        'change': change,
                        'change_pct': change_pct,
                        'is_mutual_fund': True,
                        'data_source': 'AMFI via MFApi'
                    }
            
            return {'error': f'API returned status {response.status_code}'}
            
        except Exception as e:
            return {'error': f'Error fetching {fund_name}: {str(e)}'}

    def get_fund_historical_data(self, scheme_code: str, days: int = 365) -> dict:
        """Get historical NAV data for performance analysis."""
        try:
            url = f"https://api.mfapi.in/mf/{scheme_code}"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and 'data' in data and len(data['data']) > 0:
                    # Get historical data
                    historical = data['data'][:days]  # Last 'days' worth of data
                    
                    if len(historical) == 0:
                        return {'error': 'No historical data available'}
                    
                    # Convert to proper format
                    nav_data = []
                    for record in historical:
                        try:
                            nav_data.append({
                                'date': record['date'],
                                'nav': float(record['nav'])
                            })
                        except (ValueError, KeyError):
                            continue
                    
                    if len(nav_data) < 2:
                        return {'error': 'Insufficient historical data'}
                    
                    # Calculate returns for different periods
                    current_nav = nav_data[0]['nav']
                    
                    returns = {}
                    period_mapping = {
                        '1M': 30,
                        '3M': 90, 
                        '6M': 180,
                        '1Y': 365,
                        '2Y': 730,
                        '3Y': 1095,
                        '5Y': 1825
                    }
                    
                    for period, days_back in period_mapping.items():
                        if len(nav_data) > days_back:
                            start_nav = nav_data[min(days_back, len(nav_data) - 1)]['nav']
                            if period in ['2Y', '3Y', '5Y']:
                                years = int(period[0])
                                cagr = ((current_nav / start_nav) ** (1/years) - 1) * 100
                                returns[period] = cagr
                            else:
                                absolute_return = ((current_nav - start_nav) / start_nav) * 100
                                returns[period] = absolute_return
                    
                    return {
                        'scheme_name': data['meta']['scheme_name'],
                        'returns': returns,
                        'current_nav': current_nav,
                        'historical_data': nav_data[:90]  # Last 3 months for charting
                    }
            
            return {'error': 'Could not fetch historical data'}
            
        except Exception as e:
            return {'error': f'Error fetching historical data: {str(e)}'}

    def get_benchmark_returns(self, fund_category: str) -> dict:
        """Get benchmark returns based on fund category."""
        # Define benchmarks for different categories
        benchmark_mapping = {
            'Flexi Cap Funds': '^NSEI',  # Nifty 50
            'Small Cap Funds': '^NSEBANK',  # Nifty Bank (proxy for small cap)
            'Index Funds': '^NSEI',  # Nifty 50
            'Sectoral Funds': '^NSEI',  # Nifty 50
            'Commodity ETFs': 'GC=F'  # Gold
        }
        
        benchmark_symbol = benchmark_mapping.get(fund_category, '^NSEI')
        
        try:
            ticker = yf.Ticker(benchmark_symbol)
            hist = ticker.history(period="5y")  # Get 5 years of data
            
            if hist.empty:
                return {'error': 'No benchmark data available'}
            
            # Calculate returns for different periods
            current_price = hist['Close'].iloc[-1]
            
            returns = {}
            period_mapping = {
                '1M': 30,
                '3M': 90,
                '6M': 180, 
                '1Y': 252,
                '2Y': 504,
                '3Y': 756,
                '5Y': 1260
            }
            
            for period, days_back in period_mapping.items():
                if len(hist) > days_back:
                    start_price = hist['Close'].iloc[-min(days_back, len(hist))]
                    if period in ['2Y', '3Y', '5Y']:
                        years = int(period[0])
                        cagr = ((current_price / start_price) ** (1/years) - 1) * 100
                        returns[period] = cagr
                    else:
                        absolute_return = ((current_price - start_price) / start_price) * 100
                        returns[period] = absolute_return
            
            benchmark_name = {
                '^NSEI': 'Nifty 50',
                '^NSEBANK': 'Nifty Bank',  
                'GC=F': 'Gold'
            }.get(benchmark_symbol, 'Market')
            
            return {
                'benchmark_name': benchmark_name,
                'returns': returns
            }
            
        except Exception as e:
            return {'error': f'Error fetching benchmark data: {str(e)}'}

    def create_performance_comparison_chart(self, fund_returns: dict, benchmark_returns: dict, fund_name: str) -> go.Figure:
        """Create performance comparison chart."""
        if not fund_returns or not benchmark_returns:
            return None
        
        periods = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y']
        fund_values = []
        benchmark_values = []
        valid_periods = []
        
        for period in periods:
            if period in fund_returns and period in benchmark_returns:
                fund_values.append(fund_returns[period])
                benchmark_values.append(benchmark_returns[period])
                valid_periods.append(period)
        
        if not valid_periods:
            return None
        
        fig = go.Figure()
        
        # Add fund performance
        fig.add_trace(go.Bar(
            name=fund_name,
            x=valid_periods,
            y=fund_values,
            marker_color='#1f77b4',
            text=[f'{val:.1f}%' for val in fund_values],
            textposition='outside'
        ))
        
        # Add benchmark performance
        fig.add_trace(go.Bar(
            name=benchmark_returns.get('benchmark_name', 'Benchmark'),
            x=valid_periods,
            y=benchmark_values,
            marker_color='#ff7f0e',
            text=[f'{val:.1f}%' for val in benchmark_values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'{fund_name} vs Benchmark Returns (%)',
            xaxis_title='Time Period',
            yaxis_title='Returns (%)',
            barmode='group',
            height=400,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        """Get mutual fund NAV from MFApi using scheme code."""
        if fund_name not in self.scheme_codes:
            return {'error': f'Scheme code not found for {fund_name}'}
        
        scheme_code = self.scheme_codes[fund_name]
        
        try:
            url = f"https://api.mfapi.in/mf/{scheme_code}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                api_data = response.json()
                
                if api_data and 'data' in api_data and len(api_data['data']) > 0:
                    latest = api_data['data'][0]  # Most recent NAV
                    previous = api_data['data'][1] if len(api_data['data']) > 1 else latest
                    
                    current_nav = float(latest['nav'])
                    prev_nav = float(previous['nav']) if previous['nav'] != latest['nav'] else current_nav
                    
                    change = current_nav - prev_nav
                    change_pct = (change / prev_nav * 100) if prev_nav > 0 else 0
                    
                    return {
                        'scheme_name': api_data['meta']['scheme_name'],
                        'scheme_code': api_data['meta']['scheme_code'],
                        'fund_house': api_data['meta']['fund_house'],
                        'scheme_type': api_data['meta']['scheme_type'],
                        'scheme_category': api_data['meta']['scheme_category'],
                        'nav': current_nav,
                        'nav_date': latest['date'],
                        'prev_nav': prev_nav,
                        'change': change,
                        'change_pct': change_pct,
                        'is_mutual_fund': True,
                        'data_source': 'AMFI via MFApi'
                    }
            
            return {'error': f'API returned status {response.status_code}'}
            
        except Exception as e:
            return {'error': f'Error fetching {fund_name}: {str(e)}'}

    def get_stock_data(self, symbols: List[str]) -> Dict:
        """Fetch stock AND mutual fund data using appropriate sources."""
        data = {}
        
        for symbol in symbols:
            try:
                # Check if it's a mutual fund (in scheme_codes)
                if hasattr(self, 'scheme_codes') and symbol in self.scheme_codes:
                    # It's a mutual fund - use MFApi
                    fund_data = self.get_fund_nav_from_mfapi(symbol)
                    
                    if 'error' not in fund_data:
                        data[symbol] = {
                            'price': fund_data['nav'],
                            'prev_close': fund_data['prev_nav'], 
                            'change': fund_data['change'],
                            'change_pct': fund_data['change_pct'],
                            'high': fund_data['nav'],  # NAV doesn't have intraday high/low
                            'low': fund_data['nav'],
                            'volume': 0,  # MFs don't have volume
                            'market_cap': 0,
                            'pe_ratio': None,
                            'sector': fund_data['scheme_category'],
                            'timestamp': datetime.now(),
                            'scheme_name': fund_data['scheme_name'],
                            'scheme_code': fund_data['scheme_code'],
                            'fund_house': fund_data['fund_house'],
                            'nav_date': fund_data['nav_date'],
                            'is_mutual_fund': True,
                            'data_source': fund_data['data_source']
                        }
                    else:
                        data[symbol] = fund_data  # Contains error info
                        
                elif symbol.endswith('.NS'):
                    # It's an ETF - use yfinance directly
                    ticker = yf.Ticker(symbol)
                    
                    if self.is_market_open():
                        hist = ticker.history(period="1d", interval="5m")
                    else:
                        hist = ticker.history(period="5d")
                    
                    info = ticker.info
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = info.get('previousClose', hist['Close'].iloc[0])
                        change = current_price - prev_close
                        change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                        
                        data[symbol] = {
                            'price': current_price,
                            'prev_close': prev_close,
                            'change': change,
                            'change_pct': change_pct,
                            'high': hist['High'].max(),
                            'low': hist['Low'].min(),
                            'volume': hist['Volume'].iloc[-1] if len(hist) > 0 else 0,
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE'),
                            'sector': 'ETF',
                            'timestamp': datetime.now(),
                            'is_mutual_fund': False
                        }
                    else:
                        data[symbol] = {'error': f'No data available for ETF {symbol}'}
                        
                else:
                    # It's a stock - use yfinance
                    ticker = yf.Ticker(f"{symbol}.NS")
                    
                    if self.is_market_open():
                        hist = ticker.history(period="1d", interval="5m")
                    else:
                        hist = ticker.history(period="5d")
                    
                    info = ticker.info
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = info.get('previousClose', hist['Close'].iloc[0])
                        change = current_price - prev_close
                        change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                        
                        data[symbol] = {
                            'price': current_price,
                            'prev_close': prev_close,
                            'change': change,
                            'change_pct': change_pct,
                            'high': hist['High'].max(),
                            'low': hist['Low'].min(),
                            'volume': hist['Volume'].iloc[-1] if len(hist) > 0 else 0,
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE'),
                            'sector': self.get_sector(symbol),
                            'timestamp': datetime.now(),
                            'is_mutual_fund': False
                        }
                    else:
                        data[symbol] = {'error': f'No data available for stock {symbol}'}
                        
            except Exception as e:
                data[symbol] = {'error': f"Error fetching {symbol}: {str(e)}"}
        
        return data
    
    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        for sector, stocks in self.indian_stocks.items():
            if symbol in stocks:
                return sector
        return 'Other'
    
    def get_indices_data(self) -> Dict:
        """Fetch Nifty and Sensex data."""
        indices = {}
        
        try:
            # Nifty 50
            nifty = yf.Ticker("^NSEI")
            nifty_hist = nifty.history(period="1d")
            
            if not nifty_hist.empty:
                current = nifty_hist['Close'].iloc[-1]
                open_price = nifty_hist['Open'].iloc[0]
                change = current - open_price
                change_pct = (change / open_price) * 100
                
                indices['NIFTY'] = {
                    'price': current,
                    'change': change,
                    'change_pct': change_pct,
                    'high': nifty_hist['High'].max(),
                    'low': nifty_hist['Low'].min()
                }
            
            # Sensex
            sensex = yf.Ticker("^BSESN")
            sensex_hist = sensex.history(period="1d")
            
            if not sensex_hist.empty:
                current = sensex_hist['Close'].iloc[-1]
                open_price = sensex_hist['Open'].iloc[0]
                change = current - open_price
                change_pct = (change / open_price) * 100
                
                indices['SENSEX'] = {
                    'price': current,
                    'change': change,
                    'change_pct': change_pct,
                    'high': sensex_hist['High'].max(),
                    'low': sensex_hist['Low'].min()
                }
        
        except Exception as e:
            st.error(f"Error fetching indices: {e}")
        
        return indices
    
    def get_usd_inr_rate(self) -> float:
        """Get USD/INR rate."""
        try:
            usd_inr = yf.Ticker("USDINR=X")
            hist = usd_inr.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except:
            pass
        return 83.0
    
    def create_stock_chart(self, symbol: str, period: str = "1M") -> go.Figure:
        """Create stock price chart for stocks and ETFs."""
        try:
            # Handle mutual funds separately (no charts available)
            if hasattr(self, 'scheme_codes') and symbol in self.scheme_codes:
                st.info(f"📊 Chart not available for mutual fund: {symbol}. Showing NAV data only.")
                return None
            
            # For ETFs and stocks
            if symbol.endswith('.NS'):
                ticker = yf.Ticker(symbol)
            else:
                ticker = yf.Ticker(f"{symbol}.NS")
            
            period_map = {
                "1D": ("1d", "5m"),
                "1W": ("5d", "30m"), 
                "1M": ("1mo", "1d"),
                "3M": ("3mo", "1d"),
                "6M": ("6mo", "1d"),
                "1Y": ("1y", "1d")
            }
            
            hist_period, interval = period_map.get(period, ("1mo", "1d"))
            data = ticker.history(period=hist_period, interval=interval)
            
            if data.empty:
                return None
            
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol,
                increasing_line_color='#00C851',
                decreasing_line_color='#FF4444'
            ))
            
            # Update title based on type
            if symbol.endswith('.NS'):
                title = f"{symbol} - {period} Chart (₹)"
            else:
                title = f"{symbol} - {period} Chart (₹)"
            
            fig.update_layout(
                title=title,
                yaxis_title="Price (₹)",
                template="plotly_white",
                height=500,
                showlegend=False,
                xaxis_rangeslider_visible=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Chart error for {symbol}: {e}")
            return None
    
    def call_claude_analysis(self, prompt: str) -> str:
        """Call Claude for analysis."""
        if not self.claude_client:
            return "❌ Claude API not configured"
        
        try:
            # Add Indian market context
            context = f"""
            INDIAN MARKET CONTEXT:
            - Current time: {datetime.now(self.ist).strftime('%Y-%m-%d %H:%M IST')}
            - Market status: {'OPEN' if self.is_market_open() else 'CLOSED'}
            - All prices in Indian Rupees (₹)
            - USD/INR: ₹{self.get_usd_inr_rate():.2f}
            
            {prompt}
            
            Provide analysis relevant to Indian investors.
            """
            
            message = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": context}]
            )
            
            return message.content[0].text
        
        except Exception as e:
            return f"❌ Analysis error: {str(e)}"
    
    def render_header(self):
        """Render dashboard header."""
        st.markdown('<h1 class="main-header">Indian Stock Market - AI Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Market status and time
        current_time = datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S IST')
        market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
        usd_inr = self.get_usd_inr_rate()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📅 {current_time}")
        with col2:
            st.info(f"📈 Market: {market_status}")  
        with col3:
            st.info(f"💱 USD/INR: ₹{usd_inr:.2f}")
    
    def render_market_overview(self):
        """Render market overview section."""
        st.header("📊 Market Overview")
        
        # Get indices data
        indices = self.get_indices_data()
        
        if indices:
            col1, col2 = st.columns(2)
            
            if 'NIFTY' in indices:
                nifty = indices['NIFTY']
                with col1:
                    st.metric(
                        "Nifty 50",
                        f"{nifty['price']:.2f}",
                        f"{nifty['change']:+.2f} ({nifty['change_pct']:+.2f}%)"
                    )
            
            if 'SENSEX' in indices:
                sensex = indices['SENSEX']
                with col2:
                    st.metric(
                        "Sensex",
                        f"{sensex['price']:.2f}",
                        f"{sensex['change']:+.2f} ({sensex['change_pct']:+.2f}%)"
                    )
        
        # Watchlist stocks
        if st.session_state.watchlist:
            st.subheader("📋 Your Watchlist")
            
            with st.spinner("Fetching watchlist data..."):
                watchlist_data = self.get_stock_data(st.session_state.watchlist)
            
            if watchlist_data:
                cols = st.columns(min(len(watchlist_data), 4))
                
                for i, (symbol, data) in enumerate(watchlist_data.items()):
                    if 'error' not in data:
                        with cols[i % 4]:
                            color = "normal" if data['change_pct'] >= 0 else "inverse"
                            st.metric(
                                symbol,
                                f"₹{data['price']:.2f}",
                                f"₹{data['change']:+.2f} ({data['change_pct']:+.2f}%)",
                                delta_color=color
                            )
    
    def render_stock_analysis(self):
        """Render stock analysis section."""
        st.header("📈 Stock Analysis")
        
        # Stock selection
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_stock = st.selectbox(
                "Select stock for analysis:",
                self.all_stocks,
                index=0 if self.all_stocks else 0,
                key="stock_analysis_selectbox"
            )
        
        with col2:
            chart_period = st.selectbox(
                "Chart period:",
                ["1D", "1W", "1M", "3M", "6M", "1Y"],
                index=2,
                key="stock_chart_period_selectbox"
            )
        
        with col3:
            if st.button("🔄 Refresh", type="primary"):
                st.rerun()
        
        if selected_stock:
            # Create two columns for chart and data
            chart_col, data_col = st.columns([2, 1])
            
            with chart_col:
                # Stock chart
                fig = self.create_stock_chart(selected_stock, chart_period)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with data_col:
                # Stock data
                stock_data = self.get_stock_data([selected_stock])
                
                if selected_stock in stock_data and 'error' not in stock_data[selected_stock]:
                    data = stock_data[selected_stock]
                    
                    st.subheader(f"📊 {selected_stock} Details")
                    st.metric("Current Price", f"₹{data['price']:.2f}")
                    st.metric("Day Change", f"₹{data['change']:+.2f} ({data['change_pct']:+.1f}%)")
                    st.metric("Day High", f"₹{data['high']:.2f}")
                    st.metric("Day Low", f"₹{data['low']:.2f}")
                    
                    if data.get('market_cap'):
                        market_cap_cr = data['market_cap'] / 10000000
                        st.metric("Market Cap", f"₹{market_cap_cr:.0f} Cr")
                    
                    if data.get('pe_ratio'):
                        st.metric("P/E Ratio", f"{data['pe_ratio']:.1f}")
                else:
                    st.error(f"Could not fetch data for {selected_stock}")
            
            # AI Analysis
            if self.claude_client:
                st.subheader("🤖 AI Analysis")
                
                if st.button("Generate AI Analysis", type="secondary"):
                    stock_data = self.get_stock_data([selected_stock])
                    if selected_stock in stock_data and 'error' not in stock_data[selected_stock]:
                        data = stock_data[selected_stock]
                        
                        prompt = f"""
                        Analyze {selected_stock} for Indian investors:
                        
                        Current Price: ₹{data['price']:.2f}
                        Day Change: {data['change_pct']:+.1f}%
                        Sector: {data['sector']}
                        P/E Ratio: {data.get('pe_ratio', 'N/A')}
                        
                        Provide:
                        1. Investment recommendation (BUY/HOLD/SELL)
                        2. Key reasons for the recommendation
                        3. Target price and timeline
                        4. Risk factors to consider
                        5. Suitability for different investor types
                        
                        Keep response concise and actionable.
                        """
                        
                        with st.spinner("🤖 Analyzing..."):
                            analysis = self.call_claude_analysis(prompt)
                            st.markdown("### Analysis Result")
                            st.markdown(analysis)
    
    def render_mutual_funds_analysis(self):
        """Render mutual funds/ETF analysis section."""
        st.header("🏦 Mutual Funds & ETFs Analysis")
        
        # Fund selection
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_fund = st.selectbox(
                "Select Mutual Fund/ETF for analysis:",
                self.all_mutual_funds,
                index=0 if self.all_mutual_funds else 0,
                key="fund_analysis_selectbox"
            )
        
        with col2:
            chart_period = st.selectbox(
                "Chart period:",
                ["1D", "1W", "1M", "3M", "6M", "1Y"],
                index=2,
                key="fund_chart_period_selectbox"
            )
        
        with col3:
            if st.button("🔄 Refresh Fund Data", type="primary"):
                st.rerun()
        
        if selected_fund:
            # Create two columns for chart and data
            chart_col, data_col = st.columns([2, 1])
            
            with chart_col:
                # Fund chart (only for ETFs)
                fig = self.create_stock_chart(selected_fund, chart_period)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with data_col:
                # Fund data
                fund_data = self.get_stock_data([selected_fund])
                
                if selected_fund in fund_data and 'error' not in fund_data[selected_fund]:
                    data = fund_data[selected_fund]
                    
                    # Display fund name
                    if data.get('is_mutual_fund'):
                        display_name = data.get('scheme_name', selected_fund)
                        st.subheader(f"📊 {display_name}")
                        st.metric("Current NAV", f"₹{data['price']:.2f}")
                        st.caption(f"NAV Date: {data.get('nav_date', 'N/A')}")
                        st.caption(f"Fund House: {data.get('fund_house', 'N/A')}")
                        st.caption(f"AMFI Code: {data.get('scheme_code', 'N/A')}")
                    else:
                        st.subheader(f"📊 {selected_fund}")
                        st.metric("Current Price", f"₹{data['price']:.2f}")
                        st.metric("Day High", f"₹{data['high']:.2f}")
                        st.metric("Day Low", f"₹{data['low']:.2f}")
                    
                    st.metric("Day Change", f"₹{data['change']:+.2f} ({data['change_pct']:+.1f}%)")
                    
                    # Display fund category
                    fund_category = self.get_fund_category(selected_fund)
                    st.info(f"**Category:** {fund_category}")
                    
                    if data.get('data_source'):
                        st.caption(f"Data Source: {data['data_source']}")
                
                else:
                    if selected_fund in fund_data:
                        st.error(f"Error: {fund_data[selected_fund].get('error', 'Unknown error')}")
                    else:
                        st.error("Could not fetch fund data")
            
            # AI Analysis for Mutual Funds
            if self.claude_client:
                st.subheader("🤖 AI Fund Analysis")
                
                if st.button("🚀 Generate Fund Analysis", type="secondary"):
                    fund_data_for_analysis = self.get_stock_data([selected_fund])
                    if selected_fund in fund_data_for_analysis and 'error' not in fund_data_for_analysis[selected_fund]:
                        data = fund_data_for_analysis[selected_fund]
                        
                        if data.get('is_mutual_fund'):
                            prompt = f"""
                            Analyze this Indian Mutual Fund: {data.get('scheme_name', selected_fund)}
                            
                            Fund Details:
                            - Current NAV: ₹{data['price']:.2f}
                            - Day Change: {data['change_pct']:+.1f}%
                            - Fund House: {data.get('fund_house', 'N/A')}
                            - Category: {data.get('scheme_category', self.get_fund_category(selected_fund))}
                            - AMFI Code: {data.get('scheme_code', 'N/A')}
                            - NAV Date: {data.get('nav_date', 'N/A')}
                            
                            Provide comprehensive analysis:
                            1. Fund overview and investment strategy
                            2. Performance assessment
                            3. Risk factors and volatility
                            4. Expense ratio considerations
                            5. Suitability for different investor profiles
                            6. SIP vs lump sum recommendation
                            7. Investment recommendation (BUY/HOLD/AVOID)
                            
                            Keep response detailed but actionable for Indian investors.
                            """
                        else:
                            prompt = f"""
                            Analyze this Indian ETF: {selected_fund}
                            
                            Current Price: ₹{data['price']:.2f}
                            Day Change: {data['change_pct']:+.1f}%
                            
                            Provide:
                            1. ETF overview and tracking index
                            2. Performance vs underlying index
                            3. Liquidity and trading volume
                            4. Expense ratio analysis
                            5. Investment recommendation
                            """
                        
                        with st.spinner("🤖 Analyzing fund..."):
                            analysis = self.call_claude_analysis(prompt)
                            st.markdown("### 🤖 Fund Analysis Result")
                            st.markdown(analysis)
                    else:
                        st.error("Unable to fetch fund data for analysis")
    
    def get_fund_category(self, fund_symbol: str) -> str:
        """Get category for a mutual fund symbol."""
        for category, funds in self.mutual_funds.items():
            if fund_symbol in funds:
                return category
        return 'Other Funds'
    
    def render_portfolio(self):
        """Render portfolio section."""
        st.header("💼 Portfolio Management")
        
        # Add new position
        st.subheader("➕ Add Position")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            investment_type = st.radio("Investment Type:", ["Stocks", "Mutual Funds/ETFs"], horizontal=True, key="portfolio_investment_type_radio")
            if investment_type == "Stocks":
                new_symbol = st.selectbox("Stock:", self.popular_stocks, key="portfolio_stock_selectbox")
            else:
                new_symbol = st.selectbox("Mutual Fund/ETF:", self.all_mutual_funds, key="portfolio_fund_selectbox")
        
        with col2:
            if investment_type == "Stocks":
                new_quantity = st.number_input("Shares:", min_value=1, value=100, key="portfolio_stock_quantity")
            else:
                new_quantity = st.number_input("Units:", min_value=1, value=100, key="portfolio_fund_quantity")
        
        with col3:
            if investment_type == "Stocks":
                new_price = st.number_input("Avg Price (₹):", min_value=0.01, value=100.0, key="portfolio_stock_price")
            else:
                new_price = st.number_input("Avg NAV (₹):", min_value=0.01, value=50.0, key="portfolio_fund_price")
        
        with col4:
            st.write("")  # Empty space for alignment
            st.write("")  # Empty space for alignment
            if st.button("Add Position", type="primary"):
                st.session_state.portfolio[new_symbol] = {
                    'qty': new_quantity,
                    'avg_price': new_price,
                    'type': investment_type
                }
                unit_type = "shares" if investment_type == "Stocks" else "units"
                st.success(f"Added {new_quantity} {unit_type} of {new_symbol}")
                st.rerun()
        
        # Display portfolio
        if st.session_state.portfolio:
            st.subheader("📊 Current Portfolio")
            
            # Get current prices with better error handling
            symbols = list(st.session_state.portfolio.keys())
            
            with st.spinner("📊 Fetching live prices for portfolio..."):
                current_data = self.get_stock_data(symbols)
            
            portfolio_data = []
            total_invested = 0
            total_current = 0
            
            for symbol, position in st.session_state.portfolio.items():
                try:
                    # Get current price with fallback
                    if symbol in current_data and 'error' not in current_data[symbol]:
                        current_price = current_data[symbol]['price']
                        price_source = "Live"
                        is_mf = current_data[symbol].get('is_mutual_fund', False)
                        nav_date = current_data[symbol].get('nav_date', '')
                    else:
                        current_price = position['avg_price']  # Fallback to avg price
                        price_source = "Fallback"
                        is_mf = False
                        nav_date = ''
                    
                    # Calculate values with proper error handling
                    qty = float(position['qty']) if position['qty'] else 0
                    avg_price = float(position['avg_price']) if position['avg_price'] else 0
                    
                    invested = qty * avg_price
                    current_value = qty * current_price
                    pnl = current_value - invested
                    pnl_pct = (pnl / invested * 100) if invested > 0 else 0
                    
                    # Add to totals
                    total_invested += invested
                    total_current += current_value
                    
                    # Color coding for P&L
                    pnl_color = "🟢" if pnl >= 0 else "🔴"
                    
                    # Format current price display
                    if is_mf and nav_date:
                        current_price_display = f"₹{current_price:.2f} ({nav_date})"
                    else:
                        current_price_display = f"₹{current_price:.2f} ({price_source})"
                    
                    portfolio_data.append({
                        'Investment': symbol,
                        'Type': 'MF' if is_mf else 'Stock/ETF',
                        'Qty': int(qty),
                        'Avg Price': f"₹{avg_price:.2f}",
                        'Current Price': current_price_display,
                        'Invested': f"₹{invested:,.0f}",
                        'Current Value': f"₹{current_value:,.0f}",
                        'P&L': f"{pnl_color} ₹{pnl:,.0f}",
                        'P&L %': f"{pnl_color} {pnl_pct:+.1f}%"
                    })
                    
                except Exception as e:
                    st.error(f"Error calculating P&L for {symbol}: {e}")
            
            # Portfolio summary
            total_pnl = total_current - total_invested
            total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
            
            # Portfolio summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Invested", f"₹{total_invested:,.0f}")
            
            with col2:
                st.metric("Current Value", f"₹{total_current:,.0f}")
            
            with col3:
                delta_color = "normal" if total_pnl >= 0 else "inverse"
                st.metric("Total P&L", f"₹{total_pnl:,.0f}", f"{total_pnl_pct:+.1f}%", delta_color=delta_color)
            
            with col4:
                returns_text = f"{total_pnl_pct:+.1f}%"
                if total_pnl >= 0:
                    st.success(f"📈 Profit: {returns_text}")
                else:
                    st.error(f"📉 Loss: {returns_text}")
            
            # Portfolio table
            if portfolio_data:
                df = pd.DataFrame(portfolio_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Portfolio performance chart
                st.subheader("📊 Portfolio Performance")
                
                perf_data = []
                for symbol, position in st.session_state.portfolio.items():
                    if symbol in current_data and 'error' not in current_data[symbol]:
                        current_price = current_data[symbol]['price']
                        qty = float(position['qty'])
                        avg_price = float(position['avg_price'])
                        
                        invested = qty * avg_price
                        current_value = qty * current_price
                        pnl_pct = ((current_value - invested) / invested * 100) if invested > 0 else 0
                        
                        perf_data.append({
                            'Investment': symbol,
                            'P&L %': pnl_pct,
                            'Invested': invested,
                            'Current Value': current_value
                        })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    
                    fig = px.bar(
                        perf_df, 
                        x='Investment', 
                        y='P&L %',
                        title="Portfolio Performance by Investment",
                        color='P&L %',
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Advanced Portfolio Performance Analysis
                if PERFORMANCE_AGENT_AVAILABLE:
                    st.subheader("🎯 Advanced Portfolio Analysis")
                    
                    with st.spinner("🔍 Analyzing portfolio performance..."):
                        # Convert session state portfolio to the format expected by the agent
                        performance_analysis = analyze_portfolio_performance(st.session_state.portfolio)
                    
                    if 'error' not in performance_analysis:
                        # Create tabs for different analysis views
                        perf_tab1, perf_tab2, perf_tab3 = st.tabs([
                            "📊 Performance Summary", 
                            "🎯 Risk Analysis", 
                            "💡 Recommendations"
                        ])
                        
                        with perf_tab1:
                            # Performance metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                diversity_score = performance_analysis['diversity_score']
                                diversity_color = "🟢" if diversity_score > 70 else "🟡" if diversity_score > 40 else "🔴"
                                st.metric("Diversity Score", f"{diversity_score:.1f}/100", help="Higher is better diversified")
                                st.write(f"{diversity_color} Diversification Level")
                            
                            with col2:
                                risk_level = performance_analysis['risk_assessment']['level']
                                risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk_level, "⚪")
                                st.metric("Risk Level", risk_level)
                                st.write(f"{risk_color} Risk Assessment")
                            
                            with col3:
                                if performance_analysis.get('best_performer'):
                                    bp = performance_analysis['best_performer']
                                    st.metric(
                                        "Best Performer", 
                                        bp['symbol'], 
                                        f"{bp['gain_loss_pct']:+.1f}%",
                                        delta_color="normal"
                                    )
                                
                            # Best and Worst Performers
                            st.subheader("🏆 Top Performers")
                            perf_col1, perf_col2 = st.columns(2)
                            
                            with perf_col1:
                                if performance_analysis.get('best_performer'):
                                    bp = performance_analysis['best_performer']
                                    st.success(f"🏆 **Best: {bp['symbol']}**")
                                    st.write(f"Sector: {bp['sector']}")
                                    st.write(f"Gain: ₹{bp['absolute_gain']:,.0f} ({bp['gain_loss_pct']:+.1f}%)")
                                
                            with perf_col2:
                                if performance_analysis.get('worst_performer'):
                                    wp = performance_analysis['worst_performer']
                                    st.error(f"📉 **Worst: {wp['symbol']}**")
                                    st.write(f"Sector: {wp['sector']}")
                                    st.write(f"Loss: ₹{wp['absolute_loss']:,.0f} ({wp['gain_loss_pct']:+.1f}%)")
                            
                            # Sector Allocation Chart
                            if performance_analysis.get('sector_allocation'):
                                st.subheader("🥧 Sector Allocation")
                                sector_data = performance_analysis['sector_allocation']
                                
                                fig_pie = px.pie(
                                    values=list(sector_data.values()),
                                    names=list(sector_data.keys()),
                                    title="Portfolio Sector Allocation"
                                )
                                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                                fig_pie.update_layout(height=400)
                                st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with perf_tab2:
                            # Risk Analysis
                            st.subheader("⚠️ Risk Metrics")
                            
                            risk_metrics = performance_analysis['risk_assessment']['metrics']
                            risk_level = performance_analysis['risk_assessment']['level']
                            
                            # Risk level indicator
                            risk_colors = {"LOW": "#4CAF50", "MEDIUM": "#FF9800", "HIGH": "#F44336"}
                            st.markdown(f"""
                            <div style="padding: 10px; background-color: {risk_colors.get(risk_level, '#gray')}20; 
                                        border-left: 4px solid {risk_colors.get(risk_level, '#gray')}; border-radius: 5px;">
                                <h4 style="color: {risk_colors.get(risk_level, '#gray')};">Risk Level: {risk_level}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Risk metrics display
                            risk_col1, risk_col2 = st.columns(2)
                            
                            with risk_col1:
                                st.metric(
                                    "Max Sector Concentration", 
                                    f"{risk_metrics.get('max_sector_concentration', 0):.1f}%",
                                    help="Percentage of portfolio in largest sector"
                                )
                                st.metric(
                                    "Number of Positions", 
                                    int(risk_metrics.get('number_of_positions', 0)),
                                    help="Total number of different investments"
                                )
                            
                            with risk_col2:
                                st.metric(
                                    "High Risk Allocation", 
                                    f"{risk_metrics.get('high_risk_allocation', 0):.1f}%",
                                    help="Percentage in high-risk investments"
                                )
                                st.metric(
                                    "Diversification Ratio", 
                                    f"{risk_metrics.get('diversification_ratio', 0):.2f}",
                                    help="Sectors per position (higher is better)"
                                )
                            
                            # Risk guidelines
                            st.info("""
                            **Risk Guidelines:**
                            - 🟢 **LOW**: Well diversified, stable investments
                            - 🟡 **MEDIUM**: Moderate concentration, balanced risk  
                            - 🔴 **HIGH**: Concentrated positions, high volatility assets
                            """)
                        
                        with perf_tab3:
                            # Recommendations
                            st.subheader("💡 Investment Recommendations")
                            
                            recommendations = performance_analysis.get('recommendations', [])
                            
                            if recommendations:
                                for i, rec in enumerate(recommendations, 1):
                                    st.markdown(f"**{i}.** {rec}")
                            else:
                                st.info("No specific recommendations at this time. Portfolio looks balanced!")
                            
                            # Additional insights
                            st.subheader("📈 Portfolio Insights")
                            
                            total_gain_loss_pct = performance_analysis['total_gain_loss']['percentage']
                            
                            if total_gain_loss_pct > 10:
                                st.success("🎉 Your portfolio is performing well! Consider reviewing profit-booking strategies.")
                            elif total_gain_loss_pct > 0:
                                st.info("📊 Portfolio showing positive returns. Stay invested for long-term growth.")
                            elif total_gain_loss_pct > -5:
                                st.warning("⚖️ Portfolio slightly negative. Monitor closely and consider rebalancing.")
                            else:
                                st.error("🔍 Portfolio underperforming. Review holdings and consider strategic changes.")
                    
                    else:
                        st.error(f"Performance analysis error: {performance_analysis.get('error', 'Unknown error')}")
                
                else:
                    st.info("💡 Install investment_performance_agent for advanced portfolio analysis")
        
        else:
            st.info("Add some positions to see portfolio analysis")
    
    def get_global_indices_data(self) -> Dict:
        """Fetch major global indices data."""
        indices = {}
        
        # Global indices with their yfinance symbols
        global_indices_map = {
            # US Indices
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC', 
            'Dow Jones': '^DJI',
            
            # Asian Indices
            'Nikkei 225': '^N225',
            'Hang Seng': '^HSI',
            'Shanghai Composite': '000001.SS',
            'KOSPI': '^KS11',
            'Taiwan Weighted': '^TWII',
            
            # European Indices
            'FTSE 100': '^FTSE',
            'DAX': '^GDAXI',
            'CAC 40': '^FCHI',
            'Euro Stoxx 50': '^STOXX50E',
            
            # Other Major Indices
            'TSX (Canada)': '^GSPTSE',
            'ASX 200 (Australia)': '^AXJO',
            'Bovespa (Brazil)': '^BVSP',
            
            # Commodities
            'Gold': 'GC=F',
            'Crude Oil': 'CL=F',
            'Silver': 'SI=F'
        }
        
        for name, symbol in global_indices_map.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 1:
                    current_price = hist['Close'].iloc[-1]
                    
                    # Calculate daily change
                    if len(hist) >= 2:
                        prev_close = hist['Close'].iloc[-2]
                    else:
                        prev_close = hist['Open'].iloc[0]
                    
                    change = current_price - prev_close
                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                    
                    indices[name] = {
                        'symbol': symbol,
                        'price': current_price,
                        'prev_close': prev_close,
                        'change': change,
                        'change_pct': change_pct,
                        'timestamp': datetime.now(),
                        'region': self.get_region_for_index(name)
                    }
                    
            except Exception as e:
                print(f"Error fetching {name}: {e}")
        
        return indices

    def get_region_for_index(self, index_name: str) -> str:
        """Get region for an index."""
        regions = {
            'North America': ['S&P 500', 'NASDAQ', 'Dow Jones', 'TSX (Canada)'],
            'Asia Pacific': ['Nikkei 225', 'Hang Seng', 'Shanghai Composite', 'KOSPI', 'Taiwan Weighted', 'ASX 200 (Australia)'],
            'Europe': ['FTSE 100', 'DAX', 'CAC 40', 'Euro Stoxx 50'],
            'South America': ['Bovespa (Brazil)'],
            'Commodities': ['Gold', 'Crude Oil', 'Silver']
        }
        
        for region, indices in regions.items():
            if index_name in indices:
                return region
        return 'Other'

    def render_global_indices(self):
        """Render global indices section."""
        st.header("🌍 Global Market Indices")
        
        # Add refresh button
        if st.button("🔄 Refresh Global Data", type="primary"):
            st.rerun()
        
        # Fetch global indices data
        with st.spinner("📡 Fetching global market data..."):
            global_data = self.get_global_indices_data()
        
        if not global_data:
            st.error("Could not fetch global indices data. Please try again.")
            return
        
        # Performance overview
        st.subheader("📊 Today's Global Performance")
        
        # Group by regions
        regions = {}
        for name, data in global_data.items():
            region = data['region']
            if region not in regions:
                regions[region] = []
            regions[region].append((name, data))
        
        # Display by regions
        for region, indices in regions.items():
            st.markdown(f"### {region}")
            
            # Calculate number of columns
            num_cols = min(len(indices), 4)
            cols = st.columns(num_cols)
            
            for i, (name, data) in enumerate(indices):
                with cols[i % num_cols]:
                    # Format price based on index type
                    if data['region'] == 'Commodities':
                        if 'Gold' in name or 'Silver' in name:
                            price_text = f"${data['price']:.2f}/oz"
                        else:
                            price_text = f"${data['price']:.2f}/bbl"
                    else:
                        price_text = f"{data['price']:.2f}"
                    
                    color = "normal" if data['change_pct'] >= 0 else "inverse"
                    st.metric(
                        name,
                        price_text,
                        f"{data['change']:+.2f} ({data['change_pct']:+.2f}%)",
                        delta_color=color
                    )
        
        # Global market insights
        if self.claude_client:
            st.subheader("🤖 Global Market Analysis")
            
            if st.button("Generate Global Analysis", type="secondary"):
                # Prepare global market summary
                market_summary = []
                regional_performance = {}
                
                for name, data in global_data.items():
                    region = data['region']
                    change_pct = data['change_pct']
                    
                    market_summary.append(f"- {name}: {change_pct:+.2f}%")
                    
                    if region not in regional_performance:
                        regional_performance[region] = []
                    regional_performance[region].append(change_pct)
                
                # Calculate regional averages
                regional_avg = {region: sum(changes)/len(changes) 
                              for region, changes in regional_performance.items()}
                
                prompt = f"""
                Analyze today's global market performance and its implications for Indian investors:
                
                GLOBAL INDICES PERFORMANCE:
                {chr(10).join(market_summary)}
                
                REGIONAL AVERAGES:
                {chr(10).join([f"- {region}: {avg:+.1f}%" for region, avg in regional_avg.items()])}
                
                Provide analysis covering:
                1. Global market sentiment and risk appetite
                2. Impact on Indian markets (Nifty/Sensex correlation)
                3. Sector implications for Indian stocks
                4. Currency effects on Indian IT/Pharma exporters
                5. Investment strategy adjustments for Indian investors
                6. Opportunities to capitalize on global trends
                
                Keep recommendations specific to Indian market context.
                """
                
                with st.spinner("🤖 Analyzing global markets..."):
                    analysis = self.call_claude_analysis(prompt)
                    st.markdown("### 🌍 Global Market Analysis")
                    st.markdown(analysis)
    
    def render_market_chat(self):
        """Render AI market chat section."""
        st.header("🤖 AI Market Assistant")
        
        if not self.claude_client:
            st.warning("⚠️ Configure Claude API key to use AI assistant")
            return
        
        # Chat interface
        user_question = st.text_area(
            "Ask about Indian stocks or markets:",
            placeholder="e.g., Should I invest in IT stocks right now? What's the outlook for banking sector?",
            height=100,
            key="market_chat_textarea"
        )
        
        if st.button("💬 Get AI Response", type="primary") and user_question:
            with st.spinner("🤖 Thinking..."):
                response = self.call_claude_analysis(user_question)
                
                st.markdown("### 🤖 AI Response")
                st.markdown(response)
        
        # Quick questions
        st.subheader("Quick Questions")
        quick_questions = [
            "What's the current market sentiment?",
            "Which sectors are performing well?", 
            "Should I invest in IT stocks now?",
            "What are the risks in current market?",
            "Best stocks for long-term investment?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(f"💡 {question}", key=f"quick_{i}"):
                    with st.spinner("🤖 Analyzing..."):
                        response = self.call_claude_analysis(question)
                        st.markdown("### 🤖 AI Response")
                        st.markdown(response)
    
    def render_sidebar_controls(self):
        """Render sidebar controls."""
        st.sidebar.header("⚙️ Dashboard Controls")
        
        # Watchlist management
        st.sidebar.subheader("📋 Manage Watchlist")
        
        # Add to watchlist
        available_stocks = [stock for stock in self.popular_stocks if stock not in st.session_state.watchlist]
        if available_stocks:
            new_watchlist_stock = st.sidebar.selectbox(
                "Add to watchlist:",
                available_stocks,
                key="sidebar_add_watchlist_selectbox"
            )
            
            if st.sidebar.button("➕ Add to Watchlist"):
                if new_watchlist_stock:
                    st.session_state.watchlist.append(new_watchlist_stock)
                    st.sidebar.success(f"Added {new_watchlist_stock}")
                    st.rerun()
        
        # Remove from watchlist
        if st.session_state.watchlist:
            remove_stock = st.sidebar.selectbox(
                "Remove from watchlist:",
                st.session_state.watchlist,
                key="sidebar_remove_watchlist_selectbox"
            )
            
            if st.sidebar.button("➖ Remove"):
                if remove_stock in st.session_state.watchlist:
                    st.session_state.watchlist.remove(remove_stock)
                    st.sidebar.success(f"Removed {remove_stock}")
                    st.rerun()
        
        # Data refresh
        st.sidebar.subheader("🔄 Data Controls")
        
        if st.sidebar.button("🔄 Refresh All Data", type="primary"):
            st.session_state.last_refresh = datetime.now()
            st.sidebar.success("Data refreshed!")
            st.rerun()
        
        # Display last refresh time
        if st.session_state.last_refresh:
            st.sidebar.info(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        # Market info
        st.sidebar.subheader("ℹ️ Market Info")
        st.sidebar.info("Market Hours: 9:15 AM - 3:30 PM IST")
        st.sidebar.info("Currency: All prices in ₹")
        
        # Data sources info
        st.sidebar.subheader("📊 Data Sources")
        st.sidebar.info("Stocks & ETFs: Yahoo Finance")
        st.sidebar.info("Mutual Funds: AMFI via MFApi")
        
        # Agent status
        if AGENT_AVAILABLE:
            st.sidebar.success("✅ Market Agent Available")
        else:
            st.sidebar.warning("⚠️ Market Agent Import Failed")
        
        if PERFORMANCE_AGENT_AVAILABLE:
            st.sidebar.success("✅ Performance Agent Available")
        else:
            st.sidebar.warning("⚠️ Performance Agent Import Failed")

    def run(self):
        """Main dashboard runner."""
        try:
            # Render sidebar
            self.render_sidebar_controls()
            
            # Render main content
            self.render_header()
            
            # Main tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Market Overview",
                "📈 Stock Analysis", 
                "🏦 Mutual Funds/ETFs",
                "💼 Portfolio",
                "🌍 Global Markets",
                "🤖 AI Assistant"
            ])
            
            with tab1:
                self.render_market_overview()
            
            with tab2:
                self.render_stock_analysis()
            
            with tab3:
                self.render_mutual_funds_analysis()
            
            with tab4:
                self.render_portfolio()
            
            with tab5:
                self.render_global_indices()
            
            with tab6:
                self.render_market_chat()
            
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            st.info("Please refresh the page or check your setup")

# Main execution
if __name__ == "__main__":
    # Show agent import status
    if not AGENT_AVAILABLE:
        st.error("❌ Could not import IndianStockMarketAgent")
        st.info("Dashboard will work with limited functionality")
    
    if not PERFORMANCE_AGENT_AVAILABLE:
        st.warning("⚠️ Could not import InvestmentPerformanceAgent")
        st.info("Advanced portfolio analysis will not be available")
    
    # Run dashboard
    dashboard = IndianStockDashboard()
    dashboard.run()