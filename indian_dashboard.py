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
        
        # Flatten stock list
        self.all_stocks = []
        for stocks in self.indian_stocks.values():
            self.all_stocks.extend(stocks)
        self.all_stocks = sorted(list(set(self.all_stocks)))
        
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
            help="Get from console.anthropic.com"
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
                index=0 if self.all_stocks else 0
            )
        
        with col2:
            chart_period = st.selectbox(
                "Chart period:",
                ["1D", "1W", "1M", "3M", "6M", "1Y"],
                index=2
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
    
    def render_portfolio(self):
        """Render portfolio section."""
        st.header("💼 Portfolio Management")
        
        # Add new position
        st.subheader("➕ Add Position")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_symbol = st.selectbox("Stock:", self.popular_stocks)
        
        with col2:
            new_quantity = st.number_input("Quantity:", min_value=1, value=100)
        
        with col3:
            new_price = st.number_input("Avg Price (₹):", min_value=0.01, value=100.0)
        
        with col4:
            if st.button("Add Position", type="primary"):
                st.session_state.portfolio[new_symbol] = {
                    'qty': new_quantity,
                    'avg_price': new_price
                }
                st.success(f"Added {new_quantity} shares of {new_symbol}")
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
                    
                    # Calculate values
                    qty = float(position['qty'])
                    avg_price = float(position['avg_price'])
                    
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
                        ]
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
            height=100
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
            [stock for stock in self.popular_stocks if stock not in st.session_state.watchlist]
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
                st.session_state.watchlist
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
    
    def run(self):
        """Main dashboard runner."""
        try:
            # Render sidebar
            self.render_sidebar_controls()
            
            # Render main content
            self.render_header()
            
            # Main tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Market Overview",
                "📈 Stock Analysis", 
                "💼 Portfolio",
                "🤖 AI Assistant"
            ])
            
            with tab1:
                self.render_market_overview()
            
            with tab2:
                self.render_stock_analysis()
            
            with tab3:
                self.render_portfolio()
            
            with tab4:
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