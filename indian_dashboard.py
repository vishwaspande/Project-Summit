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

Setup:
pip install streamlit plotly yfinance anthropic pandas numpy pytz
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
warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Indian Stock Market AI Dashboard",
    page_icon="🇮🇳",
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
        
        # Specific Mutual Funds and ETFs
        self.mutual_funds = {
            # Mutual Funds
            'Flexi Cap Funds': ['PPFAS_FLEXICAP_DIRECT'],
            'Small Cap Funds': ['HDFC_SMALLCAP_DIRECT'], 
            'Index Funds': ['HDFC_NIFTY_NEXT50_DIRECT', 'HDFC_NIFTY50_DIRECT'],
            'Sectoral Funds': ['NIPPON_PHARMA_DIRECT', 'ICICI_ENERGY_DIRECT'],
            
            # ETFs
            'Commodity ETFs': ['HDFC_GOLD_ETF', 'HDFC_SILVER_ETF']
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
            'watchlist': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY'],
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
    
    def get_stock_data(self, symbols: List[str]) -> Dict:
        """Fetch stock data using yfinance."""
        data = {}
        
        for symbol in symbols:
            try:
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
                        'timestamp': datetime.now()
                    }
            except Exception as e:
                st.error(f"Error fetching {symbol}: {e}")
        
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
        """Create stock price chart."""
        try:
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
            
            fig.update_layout(
                title=f"{symbol} - {period} Chart (₹)",
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
        st.markdown('<h1 class="main-header">🇮🇳 Indian Stock Market AI Dashboard</h1>', 
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
                
                if selected_stock in stock_data:
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
            
            # AI Analysis
            if self.claude_client:
                st.subheader("🤖 AI Analysis")
                
                if st.button("Generate AI Analysis", type="secondary"):
                    if selected_stock in stock_data:
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
                # Fund chart
                fig = self.create_stock_chart(selected_fund, chart_period)
                if fig:
                    # Update chart title for funds
                    fig.update_layout(title=f"{selected_fund} - {chart_period} NAV Chart (₹)")
                    st.plotly_chart(fig, use_container_width=True)
            
            with data_col:
                # Fund data
                fund_data = self.get_stock_data([selected_fund])
                
                if selected_fund in fund_data:
                    data = fund_data[selected_fund]
                    
                    # Display fund name if available
                    display_name = data.get('scheme_name', selected_fund.replace('_', ' '))
                    st.subheader(f"📊 {display_name}")
                    
                    # Show current NAV
                    if data.get('is_mutual_fund'):
                        st.metric("Current NAV", f"₹{data['price']:.2f}")
                        if data.get('nav_date'):
                            st.caption(f"NAV Date: {data['nav_date']}")
                        if data.get('data_source'):
                            st.caption(f"Data Source: {data['data_source']}")
                    else:
                        st.metric("Current Price", f"₹{data['price']:.2f}")
                    
                    st.metric("Day Change", f"₹{data['change']:+.2f} ({data['change_pct']:+.1f}%)")
                    
                    if not data.get('is_mutual_fund'):
                        st.metric("Day High", f"₹{data['high']:.2f}")
                        st.metric("Day Low", f"₹{data['low']:.2f}")
                    
                    # Display fund type
                    fund_category = self.get_fund_category(selected_fund)
                    st.info(f"**Category:** {fund_category}")
                    
                    # Show additional info for mutual funds
                    if data.get('is_mutual_fund'):
                        if data.get('scheme_code'):
                            st.caption(f"AMFI Code: {data['scheme_code']}")
                        if data.get('notes'):
                            st.warning(data['notes'])
                    else:
                        if data.get('volume'):
                            st.metric("Volume", f"{data['volume']:,.0f} units")
                        
                        # Asset size approximation
                        if data.get('market_cap'):
                            asset_size = data['market_cap'] / 10000000
                            st.metric("Est. AUM", f"₹{asset_size:.0f} Cr")
            
            # Fund Performance Analysis
            st.subheader("📊 Fund Performance Analysis")
            
            # Get performance data
            if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'calculate_fund_performance'):
                with st.spinner("Calculating fund performance..."):
                    try:
                        performance_data = self.agent.calculate_fund_performance(selected_fund)
                        
                        if 'error' not in performance_data:
                            # Display performance table
                            st.subheader("🎯 Returns Comparison")
                            
                            periods = ['1Y', '2Y', '3Y', '5Y', '10Y', 'inception']
                            period_labels = ['1 Year', '2 Years', '3 Years', '5 Years', '10 Years', 'Since Inception']
                            
                            # Create performance comparison table
                            perf_data = []
                            for i, period in enumerate(periods):
                                fund_return = performance_data['returns'].get(period)
                                benchmark_return = performance_data['benchmark_returns'].get(period)
                                alpha = performance_data['alpha'].get(period)
                                
                                if fund_return is not None:
                                    row = {
                                        'Period': period_labels[i],
                                        'Fund Return (%)': f"{fund_return:.1f}%" if fund_return else "N/A",
                                        'Benchmark (%)': f"{benchmark_return:.1f}%" if benchmark_return else "N/A",
                                        'Alpha (%)': f"{alpha:+.1f}%" if alpha else "N/A"
                                    }
                                    perf_data.append(row)
                            
                            if perf_data:
                                perf_df = pd.DataFrame(perf_data)
                                st.dataframe(perf_df, use_container_width=True, hide_index=True)
                                
                                # Performance visualization
                                st.subheader("📈 Performance Chart")
                                
                                # Create bar chart comparing fund vs benchmark
                                chart_periods = []
                                fund_returns = []
                                benchmark_returns = []
                                
                                for period in periods:
                                    fund_ret = performance_data['returns'].get(period)
                                    bench_ret = performance_data['benchmark_returns'].get(period)
                                    
                                    if fund_ret is not None and bench_ret is not None:
                                        chart_periods.append(period)
                                        fund_returns.append(fund_ret)
                                        benchmark_returns.append(bench_ret)
                                
                                if chart_periods:
                                    chart_df = pd.DataFrame({
                                        'Period': chart_periods,
                                        'Fund': fund_returns,
                                        'Benchmark': benchmark_returns
                                    })
                                    
                                    fig = px.bar(
                                        chart_df.melt(id_vars='Period', var_name='Type', value_name='Return'),
                                        x='Period', 
                                        y='Return', 
                                        color='Type',
                                        title=f"{selected_fund.replace('_', ' ')} vs Benchmark Returns (%)",
                                        barmode='group'
                                    )
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Performance insights
                                if performance_data.get('note'):
                                    st.info(f"ℹ️ {performance_data['note']}")
                                
                                # Inception date
                                if performance_data.get('inception_date'):
                                    st.caption(f"Fund Inception Date: {performance_data['inception_date']}")
                        else:
                            st.error(f"Could not calculate performance: {performance_data.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"Performance calculation failed: {str(e)}")
            
            # Similar funds in category
            st.subheader(f"📈 Other {self.get_fund_category(selected_fund)} Funds")
            
            category = self.get_fund_category(selected_fund)
            similar_funds = []
            for cat, funds in self.mutual_funds.items():
                if cat == category:
                    similar_funds = [f for f in funds if f != selected_fund][:3]
                    break
            
            if similar_funds:
                with st.spinner("Fetching similar funds data..."):
                    similar_data = self.get_stock_data(similar_funds)
                
                if similar_data:
                    cols = st.columns(len(similar_funds))
                    
                    for i, (fund, data) in enumerate(similar_data.items()):
                        if 'error' not in data:
                            with cols[i]:
                                display_name = data.get('scheme_name', fund.replace('_', ' '))
                                color = "normal" if data['change_pct'] >= 0 else "inverse"
                                st.metric(
                                    display_name[:20] + "..." if len(display_name) > 20 else display_name,
                                    f"₹{data['price']:.2f}",
                                    f"₹{data['change']:+.2f} ({data['change_pct']:+.2f}%)",
                                    delta_color=color
                                )
            
            # AI Analysis for Mutual Funds
            if self.claude_client and hasattr(self, 'agent') and self.agent:
                st.subheader("🤖 AI Fund Analysis")
                
                analysis_options = [
                    "Comprehensive Fund Analysis",
                    "Performance vs Benchmark", 
                    "Cost & Expense Analysis",
                    "Risk Assessment",
                    "SIP Recommendation",
                    "Fund Comparison"
                ]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    analysis_type = st.selectbox("Analysis Type:", analysis_options, key="fund_analysis_type_selectbox")
                
                with col2:
                    if st.button("🚀 Generate Fund Analysis", type="secondary"):
                        fund_data_for_analysis = self.get_stock_data([selected_fund])
                        if selected_fund in fund_data_for_analysis and 'error' not in fund_data_for_analysis[selected_fund]:
                            with st.spinner("🤖 Analyzing fund..."):
                                try:
                                    # Use the agent's mutual fund analysis method
                                    analysis = self.agent.analyze_indian_mutual_fund(selected_fund)
                                    st.markdown("### 🤖 Fund Analysis Result")
                                    st.markdown(analysis)
                                except Exception as e:
                                    st.error(f"Analysis error: {str(e)}")
                        else:
                            st.error("Unable to fetch fund data for analysis")
            
            elif self.claude_client:
                st.subheader("🤖 Basic AI Fund Analysis")
                
                if st.button("Generate Basic Fund Analysis", type="secondary"):
                    fund_data_for_basic_analysis = self.get_stock_data([selected_fund])
                    if selected_fund in fund_data_for_basic_analysis:
                        data = fund_data_for_basic_analysis[selected_fund]
                        
                        prompt = f"""
                        Analyze this Indian ETF/Mutual Fund: {selected_fund}
                        
                        Current NAV: ₹{data['price']:.2f}
                        Day Change: {data['change_pct']:+.1f}%
                        Category: {self.get_fund_category(selected_fund)}
                        
                        Provide:
                        1. Fund overview and investment objective
                        2. Performance analysis
                        3. Risk factors
                        4. Suitability for different investors
                        5. Investment recommendation
                        
                        Keep response concise and actionable for Indian investors.
                        """
                        
                        with st.spinner("🤖 Analyzing..."):
                            analysis = self.call_claude_analysis(prompt)
                            st.markdown("### Analysis Result")
                            st.markdown(analysis)
    
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
            
            if not current_data:
                st.error("❌ Could not fetch current market data. Please try refreshing.")
                return
            
            portfolio_data = []
            total_invested = 0
            total_current = 0
            portfolio_debug = []  # For debugging
            
            for symbol, position in st.session_state.portfolio.items():
                try:
                    # Get current price with fallback
                    if symbol in current_data and 'error' not in current_data[symbol]:
                        current_price = current_data[symbol]['price']
                        price_source = "Live"
                    else:
                        current_price = position['avg_price']  # Fallback to avg price
                        price_source = "Fallback"
                    
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
                    
                    # Debug info
                    portfolio_debug.append(f"{symbol}: {qty} × ₹{avg_price:.2f} = ₹{invested:,.0f} invested, {qty} × ₹{current_price:.2f} = ₹{current_value:,.0f} current")
                    
                    # Color coding for P&L
                    pnl_color = "🟢" if pnl >= 0 else "🔴"
                    
                    portfolio_data.append({
                        'Stock': symbol,
                        'Qty': int(qty),
                        'Avg Price': f"₹{avg_price:.2f}",
                        'Current Price': f"₹{current_price:.2f} ({price_source})",
                        'Invested': f"₹{invested:,.0f}",
                        'Current Value': f"₹{current_value:,.0f}",
                        'P&L': f"{pnl_color} ₹{pnl:,.0f}",
                        'P&L %': f"{pnl_color} {pnl_pct:+.1f}%"
                    })
                    
                except Exception as e:
                    st.error(f"Error calculating P&L for {symbol}: {e}")
                    portfolio_debug.append(f"ERROR with {symbol}: {e}")
            
            # Portfolio summary with debugging
            total_pnl = total_current - total_invested
            total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
            
            # Debug expander
            with st.expander("🔍 Debug Portfolio Calculations"):
                st.text("Calculation Details:")
                for debug_line in portfolio_debug:
                    st.text(debug_line)
                st.text(f"Total Invested: ₹{total_invested:,.0f}")
                st.text(f"Total Current: ₹{total_current:,.0f}")
                st.text(f"Total P&L: ₹{total_pnl:,.0f} ({total_pnl_pct:+.1f}%)")
            
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
            
            # Portfolio table with better formatting
            if portfolio_data:
                df = pd.DataFrame(portfolio_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Individual stock performance chart
                st.subheader("📊 Individual Stock Performance")
                
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
                            'Stock': symbol,
                            'P&L %': pnl_pct,
                            'Invested': invested,
                            'Current Value': current_value
                        })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    
                    fig = px.bar(
                        perf_df, 
                        x='Stock', 
                        y='P&L %',
                        title="Portfolio Performance by Stock",
                        color='P&L %',
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # AI Portfolio Analysis Section
            if self.claude_client:
                st.subheader("🤖 AI Portfolio Analysis & Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    analysis_type = st.selectbox(
                        "Analysis Type:",
                        [
                            "Complete Portfolio Review",
                            "Buy/Sell Recommendations", 
                            "Risk Assessment",
                            "Rebalancing Suggestions",
                            "Sector Analysis",
                            "Performance vs Market"
                        ],
                        key="portfolio_analysis_type_selectbox"
                    )
                
                with col2:
                    if st.button("🚀 Generate AI Analysis", type="primary", key="portfolio_analysis"):
                        # Prepare detailed portfolio data for Claude
                        portfolio_summary = []
                        sector_allocation = {}
                        total_portfolio_value = 0
                        individual_performance = []
                        
                        for symbol, position in st.session_state.portfolio.items():
                            if symbol in current_data and 'error' not in current_data[symbol]:
                                current_price = current_data[symbol]['price']
                                sector = current_data[symbol].get('sector', 'Unknown')
                                pe_ratio = current_data[symbol].get('pe_ratio', 'N/A')
                                day_change_pct = current_data[symbol].get('change_pct', 0)
                            else:
                                current_price = position['avg_price']
                                sector = 'Unknown'
                                pe_ratio = 'N/A'
                                day_change_pct = 0
                            
                            qty = float(position['qty'])
                            avg_price = float(position['avg_price'])
                            invested = qty * avg_price
                            current_value = qty * current_price
                            pnl = current_value - invested
                            pnl_pct = (pnl / invested * 100) if invested > 0 else 0
                            
                            total_portfolio_value += current_value
                            
                            # Sector allocation
                            if sector not in sector_allocation:
                                sector_allocation[sector] = 0
                            sector_allocation[sector] += current_value
                            
                            portfolio_summary.append(f"""
                            {symbol} ({sector}):
                            - Holdings: {int(qty)} shares @ ₹{avg_price:.2f} avg cost
                            - Current Price: ₹{current_price:.2f} (today: {day_change_pct:+.1f}%)
                            - Investment: ₹{invested:,.0f} → Current Value: ₹{current_value:,.0f}
                            - P&L: ₹{pnl:,.0f} ({pnl_pct:+.1f}%)
                            - P/E Ratio: {pe_ratio}
                            - Portfolio Weight: {(current_value/total_portfolio_value)*100:.1f}%
                            """)
                            
                            individual_performance.append({
                                'symbol': symbol,
                                'sector': sector,
                                'pnl_pct': pnl_pct,
                                'weight': (current_value/total_portfolio_value)*100,
                                'current_price': current_price,
                                'avg_price': avg_price
                            })
                        
                        # Calculate sector allocation percentages
                        sector_percentages = {sector: (value/total_portfolio_value)*100 
                                           for sector, value in sector_allocation.items()}
                        
                        # Get market context
                        indices_data = self.get_indices_data()
                        nifty_change = indices_data.get('NIFTY', {}).get('change_pct', 0)
                        sensex_change = indices_data.get('SENSEX', {}).get('change_pct', 0)
                        
                        # Create comprehensive prompt based on analysis type
                        if analysis_type == "Complete Portfolio Review":
                            prompt = f"""
                            Conduct a comprehensive review of this Indian stock portfolio:
                            
                            PORTFOLIO HOLDINGS:
                            {chr(10).join(portfolio_summary)}
                            
                            PORTFOLIO SUMMARY:
                            - Total Portfolio Value: ₹{total_portfolio_value:,.0f}
                            - Total P&L: ₹{total_pnl:,.0f} ({total_pnl_pct:+.1f}%)
                            - Number of Holdings: {len(st.session_state.portfolio)}
                            
                            SECTOR ALLOCATION:
                            {chr(10).join([f"- {sector}: {pct:.1f}%" for sector, pct in sector_percentages.items()])}
                            
                            MARKET CONTEXT:
                            - Nifty 50: {nifty_change:+.1f}% today
                            - Sensex: {sensex_change:+.1f}% today
                            
                            Provide comprehensive analysis covering:
                            1. **Overall Portfolio Health** - Risk/return profile
                            2. **Diversification Assessment** - Sector and stock concentration
                            3. **Performance Analysis** - vs market benchmarks
                            4. **Individual Stock Review** - Each holding's prospects
                            5. **Risk Factors** - What could hurt the portfolio
                            6. **Opportunities** - What's missing or underweight
                            7. **Action Plan** - Specific next steps for optimization
                            
                            Be specific with stock names, percentages, and actionable recommendations.
                            """
                        
                        elif analysis_type == "Buy/Sell Recommendations":
                            prompt = f"""
                            Provide specific BUY/SELL recommendations for this Indian portfolio:
                            
                            CURRENT HOLDINGS:
                            {chr(10).join(portfolio_summary)}
                            
                            MARKET CONTEXT:
                            - Nifty 50: {nifty_change:+.1f}% today
                            - Sensex: {sensex_change:+.1f}% today
                            - Portfolio Total: ₹{total_portfolio_value:,.0f}
                            
                            For each stock in the portfolio, provide:
                            
                            **INDIVIDUAL STOCK RECOMMENDATIONS:**
                            For each holding, specify:
                            - 🔴 SELL: If you recommend selling (with target exit price)
                            - 🟡 HOLD: If you recommend holding (with price targets)  
                            - 🟢 BUY MORE: If you recommend adding (with entry price)
                            
                            **NEW STOCK RECOMMENDATIONS:**
                            - What NEW Indian stocks to BUY for better diversification
                            - Specific entry prices and position sizes
                            - Rationale for each recommendation
                            
                            **PORTFOLIO ACTIONS:**
                            - Which existing position to REDUCE and by how much
                            - Which existing position to INCREASE and by how much
                            - Timeline for these actions (immediate vs gradual)
                            
                            Be very specific with prices, quantities, and reasoning.
                            """
                        
                        elif analysis_type == "Risk Assessment":
                            prompt = f"""
                            Assess the risk profile of this Indian portfolio:
                            
                            PORTFOLIO DETAILS:
                            {chr(10).join(portfolio_summary)}
                            
                            RISK ANALYSIS REQUIRED:
                            1. **Concentration Risk** - Over-exposure to specific stocks/sectors
                            2. **Market Risk** - Sensitivity to Nifty/Sensex movements
                            3. **Sector Risk** - Regulatory or industry-specific risks
                            4. **Currency Risk** - USD/INR exposure through IT/Pharma stocks
                            5. **Liquidity Risk** - Ability to exit positions quickly
                            6. **Volatility Assessment** - Expected price swings
                            
                            Provide:
                            - Risk score (1-10) with reasoning
                            - Biggest risk factors in the portfolio
                            - Hedging recommendations
                            - Stop-loss suggestions for each holding
                            - Portfolio stress test scenarios
                            """
                        
                        elif analysis_type == "Rebalancing Suggestions":
                            prompt = f"""
                            Suggest portfolio rebalancing for optimal allocation:
                            
                            CURRENT ALLOCATION:
                            {chr(10).join(portfolio_summary)}
                            
                            SECTOR WEIGHTS:
                            {chr(10).join([f"- {sector}: {pct:.1f}%" for sector, pct in sector_percentages.items()])}
                            
                            Provide specific rebalancing plan:
                            1. **Target Allocation** - Ideal sector/stock weights
                            2. **Trades Required** - Exact buy/sell amounts in ₹
                            3. **Priority Order** - Which trades to do first
                            4. **Tax Implications** - LTCG/STCG considerations
                            5. **Execution Timeline** - When to make each trade
                            6. **Alternative Approach** - Gradual vs immediate rebalancing
                            
                            Focus on practical, executable recommendations.
                            """
                        
                        elif analysis_type == "Sector Analysis":
                            prompt = f"""
                            Analyze the sectoral composition of this portfolio:
                            
                            SECTOR BREAKDOWN:
                            {chr(10).join([f"- {sector}: {pct:.1f}% (₹{sector_allocation[sector]:,.0f})" for sector, pct in sector_percentages.items()])}
                            
                            INDIVIDUAL HOLDINGS:
                            {chr(10).join(portfolio_summary)}
                            
                            Analyze:
                            1. **Sector Diversification** - Over/under-weight analysis
                            2. **Sector Outlook** - Which sectors to increase/decrease
                            3. **Missing Sectors** - What's not represented
                            4. **Sector Correlation** - Risk of sectors moving together
                            5. **Policy Impact** - Government policies affecting each sector
                            6. **Cyclical Analysis** - Sector rotation opportunities
                            
                            Recommend specific sector allocation changes.
                            """
                        
                        else:  # Performance vs Market
                            prompt = f"""
                            Compare this portfolio's performance against Indian market benchmarks:
                            
                            PORTFOLIO PERFORMANCE:
                            Total P&L: {total_pnl_pct:+.1f}%
                            
                            INDIVIDUAL STOCK PERFORMANCE:
                            {chr(10).join([f"- {perf['symbol']}: {perf['pnl_pct']:+.1f}% (weight: {perf['weight']:.1f}%)" for perf in individual_performance])}
                            
                            TODAY'S MARKET:
                            - Nifty 50: {nifty_change:+.1f}%
                            - Sensex: {sensex_change:+.1f}%
                            
                            Analyze:
                            1. **Benchmark Comparison** - vs Nifty 50, Sensex performance
                            2. **Alpha Generation** - Which stocks outperformed market
                            3. **Beta Analysis** - Portfolio sensitivity to market moves
                            4. **Attribution Analysis** - What drove performance
                            5. **Risk-Adjusted Returns** - Return per unit of risk taken
                            6. **Improvement Areas** - How to beat benchmarks consistently
                            """
                        
                        # Generate analysis
                        with st.spinner(f"🤖 Generating {analysis_type.lower()}..."):
                            analysis = self.call_claude_analysis(prompt)
                            
                            st.markdown(f"### 🤖 {analysis_type}")
                            st.markdown(analysis)
                            
                            # Save analysis to session state for reference
                            if 'analysis_history' not in st.session_state:
                                st.session_state.analysis_history = []
                            
                            st.session_state.analysis_history.append({
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'type': analysis_type,
                                'analysis': analysis[:500] + "..." if len(analysis) > 500 else analysis
                            })
                
                # Show recent analysis history
                if 'analysis_history' in st.session_state and st.session_state.analysis_history:
                    with st.expander("📚 Recent Analysis History"):
                        for i, hist in enumerate(reversed(st.session_state.analysis_history[-5:])):
                            st.markdown(f"**{hist['timestamp']} - {hist['type']}**")
                            st.text(hist['analysis'])
                            st.markdown("---")
            
            else:
                st.info("🔑 Configure Claude API key in sidebar to get AI portfolio analysis and buy/sell recommendations")
        
        else:
            st.info("Add some positions to see portfolio analysis")
    
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
        new_watchlist_stock = st.sidebar.selectbox(
            "Add to watchlist:",
            [stock for stock in self.popular_stocks if stock not in st.session_state.watchlist],
            key="sidebar_add_watchlist_selectbox"
        )
        
        if st.sidebar.button("➕ Add to Watchlist"):
            if new_watchlist_stock and new_watchlist_stock not in st.session_state.watchlist:
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
        
        # Agent status
        if AGENT_AVAILABLE:
            st.sidebar.success("✅ Agent Available")
        else:
            st.sidebar.warning("⚠️ Agent Import Failed")
    
# Code for Global indices
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
            'FTSE MIB': 'FTSEMIB.MI',
            
            # Other Major Indices
            'TSX (Canada)': '^GSPTSE',
            'ASX 200 (Australia)': '^AXJO',
            'Bovespa (Brazil)': '^BVSP',
            'FTSE/JSE (South Africa)': '^J203.JO',
            
            # Emerging Markets
            'MSCI Emerging Markets': 'EEM',
            'FTSE Emerging': 'VWO',
            
            # Commodities (as indices)
            'Gold': 'GC=F',
            'Crude Oil': 'CL=F',
            'Silver': 'SI=F',
            'Copper': 'HG=F'
        }
        
        for name, symbol in global_indices_map.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")  # Get 2 days to calculate change
                
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
                        'high': hist['High'].iloc[-1] if len(hist) > 0 else current_price,
                        'low': hist['Low'].iloc[-1] if len(hist) > 0 else current_price,
                        'volume': hist['Volume'].iloc[-1] if len(hist) > 0 else 0,
                        'timestamp': datetime.now(),
                        'region': self.get_region_for_index(name)
                    }
                    
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                # Continue to next index instead of failing completely
        
        return indices

    def get_region_for_index(self, index_name: str) -> str:
        """Get region for an index."""
        regions = {
            'North America': ['S&P 500', 'NASDAQ', 'Dow Jones', 'TSX (Canada)'],
            'Asia Pacific': ['Nikkei 225', 'Hang Seng', 'Shanghai Composite', 'KOSPI', 'Taiwan Weighted', 'ASX 200 (Australia)'],
            'Europe': ['FTSE 100', 'DAX', 'CAC 40', 'Euro Stoxx 50', 'FTSE MIB'],
            'Emerging Markets': ['Bovespa (Brazil)', 'FTSE/JSE (South Africa)', 'MSCI Emerging Markets', 'FTSE Emerging'],
            'Commodities': ['Gold', 'Crude Oil', 'Silver', 'Copper']
        }
        
        for region, indices in regions.items():
            if index_name in indices:
                return region
        return 'Other'

    def create_global_heatmap(self, indices_data: Dict) -> go.Figure:
        """Create a heatmap of global market performance."""
        if not indices_data:
            return None
        
        # Prepare data for heatmap
        regions = {}
        for name, data in indices_data.items():
            region = data['region']
            if region not in regions:
                regions[region] = []
            
            regions[region].append({
                'name': name,
                'change_pct': data['change_pct'],
                'price': data['price']
            })
        
        # Create heatmap data
        all_names = []
        all_regions = []
        all_changes = []
        
        for region, indices in regions.items():
            for index in indices:
                all_names.append(index['name'])
                all_regions.append(region)
                all_changes.append(index['change_pct'])
        
        # Create the heatmap
        fig = go.Figure(data=go.Scatter(
            x=all_regions,
            y=all_names,
            mode='markers+text',
            marker=dict(
                size=[abs(change)*3 + 20 for change in all_changes],  # Size based on change magnitude
                color=all_changes,
                colorscale='RdYlGn',
                colorbar=dict(title="Change %"),
                line=dict(width=1, color='white')
            ),
            text=[f"{change:+.1f}%" for change in all_changes],
            textposition="middle center",
            textfont=dict(color='white', size=10, family='Arial Black'),
            hovertemplate='<b>%{y}</b><br>' +
                        'Region: %{x}<br>' +
                        'Change: %{text}<br>' +
                        '<extra></extra>'
        ))
        
        fig.update_layout(
            title="🌍 Global Market Performance Heatmap",
            xaxis_title="Region",
            yaxis_title="Index",
            height=600,
            template="plotly_white"
        )
        
        return fig

    def render_global_indices(self):
        """Render global indices section."""
        st.header("🌍 Global Market Indices")
        
        # Add refresh button
        col1, col2 = st.columns([1, 4])
        with col1:
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
            
            # Calculate number of columns based on number of indices
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
        
        # Global heatmap
        st.subheader("🗺️ Global Performance Heatmap")
        heatmap_fig = self.create_global_heatmap(global_data)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Market correlation analysis
        st.subheader("📈 Market Trends Analysis")
        
        # Winners and losers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Top Performers")
            winners = sorted(global_data.items(), key=lambda x: x[1]['change_pct'], reverse=True)[:5]
            for name, data in winners:
                st.success(f"**{name}**: +{data['change_pct']:.2f}%")
        
        with col2:
            st.markdown("#### 📉 Underperformers") 
            losers = sorted(global_data.items(), key=lambda x: x[1]['change_pct'])[:5]
            for name, data in losers:
                st.error(f"**{name}**: {data['change_pct']:.2f}%")
        
        # Currency impact section
        st.subheader("💱 Currency Impact on Indian Markets")
        
        usd_inr = self.get_usd_inr_rate()
        
        # Get USD index (DXY) if available
        try:
            dxy = yf.Ticker('DX-Y.NYB')
            dxy_hist = dxy.history(period="2d")
            if not dxy_hist.empty:
                dxy_current = dxy_hist['Close'].iloc[-1]
                dxy_prev = dxy_hist['Close'].iloc[-2] if len(dxy_hist) >= 2 else dxy_hist['Open'].iloc[0]
                dxy_change = ((dxy_current - dxy_prev) / dxy_prev * 100) if dxy_prev > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("USD Index (DXY)", f"{dxy_current:.2f}", f"{dxy_change:+.2f}%")
                
                with col2:
                    st.metric("USD/INR", f"₹{usd_inr:.2f}")
                
                with col3:
                    # Show impact on IT stocks (USD exposure)
                    if 'TCS' in current_data or 'INFY' in current_data:
                        it_impact = "Positive" if dxy_change > 0 else "Negative"
                        st.info(f"IT Stock Impact: {it_impact}")
        
        except Exception as e:
            st.warning("Could not fetch currency data")
        
        # Global market insights
        if self.claude_client:
            st.subheader("🤖 Global Market Insights")
            
            if st.button("Generate Global Market Analysis", type="secondary"):
                # Prepare global market summary for AI
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
                
                USD/INR: ₹{usd_inr:.2f}
                
                Provide analysis covering:
                1. **Global Market Sentiment** - Risk-on or risk-off mood
                2. **Impact on Indian Markets** - How global moves affect Nifty/Sensex
                3. **Sector Implications** - Which Indian sectors benefit/suffer
                4. **Currency Effects** - USD/INR impact on IT, Pharma exporters
                5. **Investment Strategy** - Should Indian investors adjust portfolios
                6. **Opportunities** - Any global trends to capitalize on locally
                
                Keep recommendations specific to Indian market context.
                """
                
                with st.spinner("🤖 Analyzing global markets..."):
                    analysis = self.call_claude_analysis(prompt)
                    st.markdown("### 🌍 Global Market Analysis")
                    st.markdown(analysis)
        
        # Time zone display
        st.subheader("🕒 Global Market Hours")
        
        # Define major market hours (in IST)
        market_hours = {
            'Indian Markets': ('09:15', '15:30'),
            'Tokyo': ('05:30', '11:30'),
            'Hong Kong': ('06:45', '13:00'),
            'London': ('13:30', '22:00'),
            'New York': ('19:30', '02:00')  # Next day
        }
        
        current_ist = datetime.now(self.ist)
        current_time = current_ist.time()
        
        cols = st.columns(len(market_hours))
        
        for i, (market, (open_time, close_time)) in enumerate(market_hours.items()):
            with cols[i]:
                open_dt = datetime.strptime(open_time, "%H:%M").time()
                close_dt = datetime.strptime(close_time, "%H:%M").time()
                
                # Handle overnight sessions (like NY)
                if close_dt < open_dt:  # Overnight session
                    is_open = current_time >= open_dt or current_time <= close_dt
                else:
                    is_open = open_dt <= current_time <= close_dt
                
                # Account for weekends
                is_weekend = current_ist.weekday() >= 5
                if is_weekend and market != 'Commodities':
                    is_open = False
                
                status = "🟢 OPEN" if is_open else "🔴 CLOSED"
                
                st.metric(
                    market,
                    f"{open_time}-{close_time} IST",
                    status
                )




    def run(self):
        """Main dashboard runner."""
        try:
            # Render sidebar
            self.render_sidebar_controls()
            
            # Render main content
            self.render_header()
            
            # Main tabs - ADD the new global indices tab
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Market Overview",
                "📈 Stock Analysis", 
                "🏦 Mutual Funds/ETFs",
                "💼 Portfolio",
                "🌍 Global Markets",  # NEW TAB
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
            
            with tab5:  # NEW TAB CONTENT
                self.render_global_indices()
            
            with tab6:  # Updated tab number
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
    
    # Run dashboard
    dashboard = IndianStockDashboard()
    dashboard.run()