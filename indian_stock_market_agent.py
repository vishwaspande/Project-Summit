#!/usr/bin/env python3
"""
Indian Stock Market Dashboard - Streamlit Web App

Features for Indian Markets:
- NSE/BSE stock tracking
- Nifty 50 & Sensex monitoring  
- Indian market hours display
- Rupee-denominated analysis
- Sector-wise analysis (IT, Banking, Pharma, etc.)
- FII/DII flow impact
- USD/INR impact analysis
- Indian stock recommendations

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
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="Indian Stock Market AI Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Indian theme
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
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #138808;
    }
    .sector-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .alert-bull {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .alert-bear {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class IndianStockDashboard:
    def __init__(self):
        """Initialize the Indian stock dashboard."""
        
        # Indian timezone
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Popular Indian stocks with their sectors
        self.indian_stocks = {
            'Large Cap IT': ['TCS', 'INFY', 'WIPRO', 'TECHM', 'HCLTECH'],
            'Large Cap Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'Large Cap Consumer': ['RELIANCE', 'HINDUNILVR', 'ITC', 'NESTLEIND'],
            'Large Cap Auto': ['MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'M&M'],
            'Mid Cap IT': ['MPHASIS', 'MINDTREE', 'LTI', 'COFORGE'],
            'Mid Cap Banking': ['FEDERALBNK', 'BANDHANBNK', 'IDFCFIRSTB'],
            'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'BIOCON', 'AUROPHARMA'],
            'Infrastructure': ['LT', 'ADANIPORTS', 'POWERGRID', 'NTPC']
        }
        
        # Flatten the stock list
        self.all_stocks = []
        for stocks in self.indian_stocks.values():
            self.all_stocks.extend(stocks)
        
        # Initialize session state
        if 'indian_portfolio' not in st.session_state:
            st.session_state.indian_portfolio = {}
        if 'monitoring_stocks' not in st.session_state:
            st.session_state.monitoring_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY']
        if 'live_data' not in st.session_state:
            st.session_state.live_data = {}
        if 'indices_data' not in st.session_state:
            st.session_state.indices_data = {}
        
        self.claude_client = None
        self.setup_claude()
    
    def setup_claude(self):
        """Setup Claude API client."""
        api_key = st.sidebar.text_input(
            "🔑 Claude API Key:", 
            type="password",
            help="Get your API key from console.anthropic.com"
        )
        
        if api_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=api_key)
                st.sidebar.success("✅ Claude AI connected!")
                return True
            except Exception as e:
                st.sidebar.error(f"❌ Claude API Error: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Enter API key for AI analysis")
        
        return False
    
    def is_indian_market_open(self) -> bool:
        """Check if Indian stock market is open."""
        now = datetime.now(self.ist)
        current_time = now.time()
        weekday = now.weekday()
        
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        
        # Market closed on weekends
        if weekday >= 5:
            return False
        
        return market_open <= current_time <= market_close
    
    def get_indian_stock_data(self, symbols: List[str]) -> Dict:
        """Fetch Indian stock data."""
        data = {}
        
        for symbol in symbols:
            try:
                # Add .NS for NSE stocks
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                
                # Get intraday data if market is open, otherwise daily data
                if self.is_indian_market_open():
                    hist = ticker.history(period="1d", interval="5m")
                else:
                    hist = ticker.history(period="5d", interval="1d")
                
                info = ticker.info
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = info.get('previousClose', hist['Open'].iloc[0])
                    day_change = current_price - prev_close
                    day_change_pct = (day_change / prev_close * 100) if prev_close > 0 else 0
                    
                    data[symbol] = {
                        'price': current_price,
                        'prev_close': prev_close,
                        'day_change': day_change,
                        'day_change_pct': day_change_pct,
                        'volume': hist['Volume'].sum() if len(hist) > 1 else hist['Volume'].iloc[-1],
                        'high': hist['High'].max(),
                        'low': hist['Low'].min(),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE'),
                        'sector': self.get_stock_sector(symbol),
                        'hist_data': hist,
                        'timestamp': datetime.now(self.ist)
                    }
                    
            except Exception as e:
                st.error(f"Error fetching {symbol}: {e}")
        
        return data
    
    def get_stock_sector(self, symbol: str) -> str:
        """Get sector for a stock symbol."""
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
            nifty_hist = nifty.history(period="1d", interval="5m" if self.is_indian_market_open() else "1d")
            
            if not nifty_hist.empty:
                nifty_price = nifty_hist['Close'].iloc[-1]
                nifty_open = nifty_hist['Open'].iloc[0]
                nifty_change = nifty_price - nifty_open
                nifty_change_pct = (nifty_change / nifty_open) * 100
                
                indices['NIFTY'] = {
                    'price': nifty_price,
                    'change': nifty_change,
                    'change_pct': nifty_change_pct,
                    'high': nifty_hist['High'].max(),
                    'low': nifty_hist['Low'].min()
                }
            
            # Sensex
            sensex = yf.Ticker("^BSESN")
            sensex_hist = sensex.history(period="1d", interval="5m" if self.is_indian_market_open() else "1d")
            
            if not sensex_hist.empty:
                sensex_price = sensex_hist['Close'].iloc[-1]
                sensex_open = sensex_hist['Open'].iloc[0]
                sensex_change = sensex_price - sensex_open
                sensex_change_pct = (sensex_change / sensex_open) * 100
                
                indices['SENSEX'] = {
                    'price': sensex_price,
                    'change': sensex_change,
                    'change_pct': sensex_change_pct,
                    'high': sensex_hist['High'].max(),
                    'low': sensex_hist['Low'].min()
                }
                
        except Exception as e:
            st.error(f"Error fetching indices: {e}")
        
        return indices
    
    def get_usd_inr_rate(self) -> float:
        """Get USD/INR exchange rate."""
        try:
            usd_inr = yf.Ticker("USDINR=X")
            hist = usd_inr.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except:
            pass
        return 83.0  # Default fallback
    
    def call_claude_indian(self, prompt: str, max_tokens: int = 2500) -> str:
        """Call Claude with Indian market context."""
        if not self.claude_client:
            return "❌ Claude API not configured. Please add your API key."
        
        current_time = datetime.now(self.ist).strftime('%Y-%m-%d %H:%M IST')
        market_status = 'OPEN' if self.is_indian_market_open() else 'CLOSED'
        usd_inr = self.get_usd_inr_rate()
        
        indian_context = f"""
        INDIAN MARKET CONTEXT:
        - Current time: {current_time}
        - Market status: {market_status}
        - Currency: All prices in Indian Rupees (₹)
        - USD/INR rate: ₹{usd_inr:.2f}
        - Market hours: 9:15 AM - 3:30 PM IST
        
        {prompt}
        
        Provide analysis specifically relevant to Indian investors, considering:
        - Indian market dynamics and regulations
        - Sectoral trends in India
        - Impact of global factors on Indian markets
        - Currency effects on investments
        - Retail investor perspective
        """
        
        try:
            with st.spinner("🤖 AI analyzing..."):
                message = self.claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=max_tokens,
                    temperature=0.3,
                    messages=[{"role": "user", "content": indian_context}]
                )
                return message.content[0].text
        except Exception as e:
            return f"❌ AI Analysis error: {str(e)}"
    
    def create_indian_stock_chart(self, symbol: str, period: str = "1d") -> go.Figure:
        """Create chart for Indian stock."""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            
            if period == "1d":
                data = ticker.history(period="1d", interval="5m")
                title_suffix = "Today"
            elif period == "1w":
                data = ticker.history(period="5d", interval="30m") 
                title_suffix = "5 Days"
            elif period == "1m":
                data = ticker.history(period="1mo", interval="1d")
                title_suffix = "1 Month"
            else:
                data = ticker.history(period=period, interval="1d")
                title_suffix = period.upper()
            
            if data.empty:
                st.error(f"No data available for {symbol}")
                return None
            
            fig = go.Figure()
            
            # Candlestick chart
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
                title=f"{symbol} - {title_suffix} (₹)",
                yaxis_title="Price (₹)",
                xaxis_title="Time",
                template="plotly_white",
                height=400,
                showlegend=False
            )
            
            fig.update_xaxes(rangeslider_visible=False)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating chart for {symbol}: {e}")
            return None
    
    def render_main_dashboard(self):
        """Render the main dashboard."""
        
        # Header with Indian flag colors
        st.markdown('<h1 class="main-header">🇮🇳 Indian Stock Market AI Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Current time and market status
        current_time = datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S IST')
        market_status = "🟢 OPEN" if self.is_indian_market_open() else "🔴 CLOSED"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📅 {current_time}")
        with col2:
            st.info(f"📈 Market: {market_status}")
        with col3:
            usd_inr = self.get_usd_inr_rate()
            st.info(f"💱 USD/INR: ₹{usd_inr:.2f}")
        
        # Sidebar controls
        st.sidebar.header("📊 Dashboard Controls")
        
        # Stock selection
        selected_stocks = st.sidebar.multiselect(
            "Select Indian stocks to monitor:",
            options=self.all_stocks,
            default=st.session_state.monitoring_stocks,
            help="Choose from NSE listed stocks"
        )
        
        if selected_stocks:
            st.session_state.monitoring_stocks = selected_stocks
        
        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh data", value=False)
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 30, 300, 60)
        
        # Manual refresh
        if st.sidebar.button("🔄 Refresh Data Now", type="primary"):
            self.refresh_data()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Market Overview", 
            "📈 Stock Charts", 
            "💼 Portfolio", 
            "🤖 AI Analysis",
            "📰 Market Outlook",
            "⚡ Alerts"
        ])
        
        with tab1:
            self.render_market_overview_tab()
        
        with tab2:
            self.render_charts_tab()
        
        with tab3:
            self.render_portfolio_tab()
        
        with tab4:
            self.render_ai_analysis_tab()
        
        with tab5:
            self.render_market_outlook_tab()
        
        with tab6:
            self.render_alerts_tab()
        
        # Auto-refresh mechanism
        if auto_refresh:
            time.sleep(refresh_interval)
            st.experimental_rerun()
    
    def refresh_data(self):
        """Refresh all market data."""
        with st.spinner("📊 Refreshing Indian market data..."):
            # Get stocks data
            st.session_state.live_data = self.get_indian_stock_data(st.session_state.monitoring_stocks)
            
            # Get indices data
            st.session_state.indices_data = self.get_indices_data()
            
            st.success("✅ Data refreshed!")
    
    def render_market_overview_tab(self):
        """Render market overview tab."""
        st.subheader("📊 Indian Market Overview")
        
        # Get fresh data if not available
        if not st.session_state.live_data:
            self.refresh_data()
        
        # Indices overview
        if st.session_state.indices_data:
            st.subheader("📈 Market Indices")
            
            col1, col2 = st.columns(2)
            
            if 'NIFTY' in st.session_state.indices_data:
                nifty = st.session_state.indices_data['NIFTY']
                with col1:
                    delta_color = "normal" if nifty['change_pct'] >= 0 else "inverse"
                    st.metric(
                        "Nifty 50",
                        f"{nifty['price']:.1f}",
                        f"{nifty['change']:+.1f} ({nifty['change_pct']:+.1f}%)",
                        delta_color=delta_color
                    )
            
            if 'SENSEX' in st.session_state.indices_data:
                sensex = st.session_state.indices_data['SENSEX']
                with col2:
                    delta_color = "normal" if sensex['change_pct'] >= 0 else "inverse"
                    st.metric(
                        "Sensex",
                        f"{sensex['price']:.1f}",
                        f"{sensex['change']:+.1f} ({sensex['change_pct']:+.1f}%)",
                        delta_color=delta_color
                    )
        
        # Stock overview
        if st.session_state.live_data:
            st.subheader("📋 Monitored Stocks")
            
            # Create columns for stock metrics
            cols = st.columns(min(len(st.session_state.live_data), 4))
            
            for i, (symbol, data) in enumerate(st.session_state.live_data.items()):
                col_idx = i % 4
                with cols[col_idx]:
                    delta_color = "normal" if data['day_change_pct'] >= 0 else "inverse"
                    st.metric(
                        f"{symbol}",
                        f"₹{data['price']:.1f}",
                        f"₹{data['day_change']:+.1f} ({data['day_change_pct']:+.1f}%)",
                        delta_color=delta_color
                    )
            
            # Sector performance
            self.render_sector_performance()
    
    def render_sector_performance(self):
        """Render sector performance analysis."""
        st.subheader("🏭 Sector Performance")
        
        sector_data = {}
        
        # Calculate sector averages
        for symbol, data in st.session_state.live_data.items():
            sector = data['sector']
            if sector not in sector_data:
                sector_data[sector] = []
            sector_data[sector].append(data['day_change_pct'])
        
        # Calculate sector averages
        sector_avg = {}
        for sector, changes in sector_data.items():
            sector_avg[sector] = sum(changes) / len(changes)
        
        # Display sector cards
        cols = st.columns(3)
        for i, (sector, avg_change) in enumerate(sector_avg.items()):
            with cols[i % 3]:
                color = "🟢" if avg_change >= 0 else "🔴"
                st.markdown(f"""
                <div class="sector-card">
                    <h4>{color} {sector}</h4>
                    <h2>{avg_change:+.1f}%</h2>
                    <p>Average Change</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_charts_tab(self):
        """Render charts tab."""
        st.subheader("📈 Stock Charts")
        
        if not st.session_state.monitoring_stocks:
            st.info("Please select stocks to monitor from the sidebar")
            return
        
        # Chart controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_stock = st.selectbox(
                "Select stock for detailed chart:",
                st.session_state.monitoring_stocks
            )
        
        with col2:
            time_period = st.selectbox(
                "Time period:",
                ["1d", "1w", "1m", "3m", "6m", "1y"]
            )
        
        if selected_stock:
            # Create and display chart
            fig = self.create_indian_stock_chart(selected_stock, time_period)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Stock details
            if selected_stock in st.session_state.live_data:
                data = st.session_state.live_data[selected_stock]
                
                st.subheader(f"📊 {selected_stock} Details")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"₹{data['price']:.2f}")
                
                with col2:
                    st.metric("Day High", f"₹{data['high']:.2f}")
                
                with col3:
                    st.metric("Day Low", f"₹{data['low']:.2f}")
                
                with col4:
                    market_cap_cr = data['market_cap'] / 10000000 if data['market_cap'] else 0
                    st.metric("Market Cap", f"₹{market_cap_cr:.0f} Cr")
    
    def render_portfolio_tab(self):
        """Render portfolio management tab."""
        st.subheader("💼 Portfolio Management")
        
        # Add/Edit positions
        st.subheader("➕ Add Stock Position")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            portfolio_symbol = st.selectbox("Stock:", self.all_stocks, key="portfolio_stock")
        
        with col2:
            quantity = st.number_input("Quantity:", min_value=1, value=100, step=1)
        
        with col3:
            avg_price = st.number_input("Average Price (₹):", min_value=0.01, value=100.0, step=0.01)
        
        with col4:
            if st.button("➕ Add Position", type="primary"):
                st.session_state.indian_portfolio[portfolio_symbol] = {
                    'qty': quantity,
                    'avg_price': avg_price,
                    'timestamp': datetime.now(self.ist)
                }
                st.success(f"Added {quantity} shares of {portfolio_symbol}")
        
        # Display portfolio
        if st.session_state.indian_portfolio:
            st.subheader("📊 Current Portfolio")
            
            # Get current prices for portfolio stocks
            portfolio_symbols = list(st.session_state.indian_portfolio.keys())
            current_prices = self.get_indian_stock_data(portfolio_symbols)
            
            # Calculate portfolio metrics
            portfolio_data = []
            total_invested = 0
            total_current = 0
            
            for symbol, position in st.session_state.indian_portfolio.items():
                current_price = current_prices.get(symbol, {}).get('price', position['avg_price'])
                invested = position['qty'] * position['avg_price']
                current_value = position['qty'] * current_price
                pnl = current_value - invested
                pnl_pct = (pnl / invested * 100) if invested > 0 else 0
                
                total_invested += invested
                total_current += current_value
                
                portfolio_data.append({
                    'Stock': symbol,
                    'Qty': position['qty'],
                    'Avg Price': f"₹{position['avg_price']:.2f}",
                    'Current Price': f"₹{current_price:.2f}",
                    'Invested': f"₹{invested:,.0f}",
                    'Current Value': f"₹{current_value:,.0f}",
                    'P&L': f"₹{pnl:,.0f}",
                    'P&L %': f"{pnl_pct:+.1f}%"
                })
            
            # Portfolio summary
            total_pnl = total_current - total_invested
            total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Invested", f"₹{total_invested:,.0f}")
            
            with col2:
                st.metric("Current Value", f"₹{total_current:,.0f}")
            
            with col3:
                delta_color = "normal" if total_pnl >= 0 else "inverse"
                st.metric("Total P&L", f"₹{total_pnl:,.0f}", f"{total_pnl_pct:+.1f}%", delta_color=delta_color)
            
            # Portfolio table
            df = pd.DataFrame(portfolio_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # AI Portfolio Analysis
            if st.button("🤖 Get AI Portfolio Analysis", type="secondary") and self.claude_client:
                portfolio_analysis = self.analyze_indian_portfolio()
                st.markdown("### 🤖 AI Portfolio Analysis")
                st.markdown(portfolio_analysis)
    
    def analyze_indian_portfolio(self) -> str:
        """Get AI analysis of Indian portfolio."""
        if not st.session_state.indian_portfolio:
            return "❌ No portfolio positions found"
        
        # Prepare portfolio summary
        portfolio_summary = []
        for symbol, position in st.session_state.indian_portfolio.items():
            current_data = st.session_state.live_data.get(symbol, {})
            current_price = current_data.get('price', position['avg_price'])
            
            portfolio_summary.append(f"""
            {symbol}: {position['qty']} shares @ ₹{position['avg_price']:.2f}
            Current: ₹{current_price:.2f}
            Sector: {current_data.get('sector', 'Unknown')}
            """)
        
        prompt = f"""
        Analyze this Indian stock portfolio:
        
        PORTFOLIO POSITIONS:
        {chr(10).join(portfolio_summary)}
        
        MARKET CONTEXT:
        - Nifty 50: {st.session_state.indices_data.get('NIFTY', {}).get('change_pct', 'N/A')}%
        - Sensex: {st.session_state.indices_data.get('SENSEX', {}).get('change_pct', 'N/A')}%
        
        Provide comprehensive analysis including:
        1. **Portfolio Health** - Overall risk and diversification
        2. **Sector Analysis** - Sector allocation and balance
        3. **Stock Performance** - Individual stock outlook
        4. **Recommendations** - Specific buy/sell/hold suggestions
        5. **Risk Management** - Key risks and mitigation
        
        Focus on actionable advice for Indian retail investors.
        """
        
        return self.call_claude_indian(prompt)
    
    def render_ai_analysis_tab(self):
        """Render AI analysis tab."""
        st.subheader("🤖 AI-Powered Stock Analysis")
        
        if not self.claude_client:
            st.warning("⚠️ Please configure Claude API key in sidebar for AI analysis")
            return
        
        # Stock selection for analysis
        analysis_stock = st.selectbox("Select stock for AI analysis:", st.session_state.monitoring_stocks)
        
        analysis_type = st.selectbox("Analysis Type:", [
            "Comprehensive Stock Analysis",
            "Technical Analysis",
            "Fundamental Analysis", 
            "Sector Comparison",
            "Risk Assessment"
        ])
        
        if st.button("🚀 Generate AI Analysis", type="primary") and analysis_stock:
            
            if analysis_stock in st.session_state.live_data:
                stock_data = st.session_state.live_data[analysis_stock]
                
                if analysis_type == "Comprehensive Stock Analysis":
                    prompt = f"""
                    Provide comprehensive analysis for {analysis_stock}:
                    
                    Current Price: ₹{stock_data['price']:.2f}
                    Day Change: {stock_data['day_change_pct']:+.1f}%
                    Sector: {stock
