#!/usr/bin/env python3
"""
Enhanced Indian Stock Market Dashboard with AI Agent Integration

Features for Indian Markets:
- NSE/BSE stock tracking with real-time data
- Nifty 50 & Sensex monitoring
- AI-powered autonomous agent with tool use
- Agent activity monitoring and settings
- Portfolio management with Indian context
- Real-time alerts and notifications
- Telegram bot integration
- SQLite memory system for analysis history

New Features:
- Agent Activity tab: Monitor agent actions, decisions, alerts
- Agent Settings tab: Configure watchlist, risk tolerance, scheduling
- Enhanced AI Assistant with multi-step reasoning
- Portfolio performance tracking over time
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import time
import json
import os
import numpy as np
from typing import Dict, List, Optional
import asyncio
from dataclasses import asdict
import warnings

warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Indian Stock Market - AI Agent Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🇮🇳"
)

# Import our enhanced modules
try:
    from config import AgentConfig, load_config
    from indian_stock_market_agent import EnhancedIndianStockMarketAgent
    from memory import MemoryManager
    from notifications import NotificationManager
    from tools import StockMarketTools
    ENHANCED_AGENT_AVAILABLE = True
except ImportError as e:
    st.error(f"Enhanced agent modules not found: {e}")
    ENHANCED_AGENT_AVAILABLE = False

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
    .metric-positive { color: #4CAF50; font-weight: bold; }
    .metric-negative { color: #f44336; font-weight: bold; }
    .metric-neutral { color: #2196F3; font-weight: bold; }

    .agent-status {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .decision-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin-bottom: 0.5rem;
    }

    .alert-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 0.5rem;
    }

    .tool-usage {
        background: #e7f3ff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border: 1px solid #b3d9ff;
        margin: 0.2rem 0;
    }

    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'config' not in st.session_state:
    try:
        st.session_state.config = load_config()
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        st.stop()

if 'agent' not in st.session_state and ENHANCED_AGENT_AVAILABLE:
    try:
        st.session_state.agent = EnhancedIndianStockMarketAgent(st.session_state.config)
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        ENHANCED_AGENT_AVAILABLE = False

# Helper functions
def get_indian_time():
    """Get current Indian time."""
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def is_market_open():
    """Check if Indian stock market is open."""
    now = get_indian_time()
    current_time = now.strftime("%H:%M")
    current_day = now.weekday()

    if current_day >= 5:  # Weekend
        return False

    return "09:15" <= current_time <= "15:30"

def format_currency(amount, currency="₹"):
    """Format currency with Indian style."""
    if amount >= 10000000:  # 1 crore
        return f"{currency}{amount/10000000:.1f}Cr"
    elif amount >= 100000:  # 1 lakh
        return f"{currency}{amount/100000:.1f}L"
    elif amount >= 1000:  # 1 thousand
        return f"{currency}{amount/1000:.1f}K"
    else:
        return f"{currency}{amount:.2f}"

# Main Dashboard Header
st.markdown('<h1 class="main-header">🇮🇳 Indian Stock Market AI Agent Dashboard</h1>',
           unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")

# Market status in sidebar
market_status = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
current_time = get_indian_time().strftime("%H:%M:%S IST")
st.sidebar.markdown(f"**Market Status:** {market_status}")
st.sidebar.markdown(f"**Time:** {current_time}")

# Main navigation tabs
tab_options = [
    "📊 Market Overview",
    "📈 Stock Analysis",
    "💼 Portfolio",
    "🤖 Agent Activity",
    "⚙️ Agent Settings",
    "💬 AI Assistant"
]

selected_tab = st.sidebar.radio("Select View", tab_options)

# ===== MARKET OVERVIEW TAB =====
if selected_tab == "📊 Market Overview":
    st.header("📊 Indian Market Overview")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    try:
        # Get Nifty data
        nifty = yf.Ticker("^NSEI")
        nifty_data = nifty.history(period="1d")
        if not nifty_data.empty:
            nifty_current = nifty_data['Close'].iloc[-1]
            nifty_open = nifty_data['Open'].iloc[-1]
            nifty_change = nifty_current - nifty_open
            nifty_change_pct = (nifty_change / nifty_open) * 100

            with col1:
                st.metric(
                    "Nifty 50",
                    f"{nifty_current:.1f}",
                    f"{nifty_change:+.1f} ({nifty_change_pct:+.1f}%)"
                )
    except Exception as e:
        with col1:
            st.metric("Nifty 50", "Loading...", "")

    try:
        # Get Sensex data
        sensex = yf.Ticker("^BSESN")
        sensex_data = sensex.history(period="1d")
        if not sensex_data.empty:
            sensex_current = sensex_data['Close'].iloc[-1]
            sensex_open = sensex_data['Open'].iloc[-1]
            sensex_change = sensex_current - sensex_open
            sensex_change_pct = (sensex_change / sensex_open) * 100

            with col2:
                st.metric(
                    "Sensex",
                    f"{sensex_current:.1f}",
                    f"{sensex_change:+.1f} ({sensex_change_pct:+.1f}%)"
                )
    except Exception as e:
        with col2:
            st.metric("Sensex", "Loading...", "")

    try:
        # USD/INR
        usd_inr = yf.Ticker("USDINR=X")
        usd_data = usd_inr.history(period="2d")
        if len(usd_data) >= 2:
            usd_current = usd_data['Close'].iloc[-1]
            usd_prev = usd_data['Close'].iloc[-2]
            usd_change = usd_current - usd_prev
            usd_change_pct = (usd_change / usd_prev) * 100

            with col3:
                st.metric(
                    "USD/INR",
                    f"₹{usd_current:.2f}",
                    f"{usd_change:+.3f} ({usd_change_pct:+.2f}%)"
                )
    except Exception as e:
        with col3:
            st.metric("USD/INR", "Loading...", "")

    with col4:
        if ENHANCED_AGENT_AVAILABLE:
            agent_status = "🟢 Active" if 'agent' in st.session_state else "🔴 Inactive"
            st.metric("AI Agent", agent_status, "Enhanced Mode")
        else:
            st.metric("AI Agent", "🔴 Unavailable", "Check Config")

    # Watchlist section
    st.subheader("👀 Watchlist Stocks")

    if ENHANCED_AGENT_AVAILABLE and 'agent' in st.session_state:
        watchlist = st.session_state.config.default_watchlist
    else:
        watchlist = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC']

    # Create watchlist dataframe
    watchlist_data = []

    for symbol in watchlist[:10]:  # Limit to 10 for display
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d")
            info = ticker.info

            if not data.empty:
                current = data['Close'].iloc[-1]
                open_price = data['Open'].iloc[-1]
                change = current - open_price
                change_pct = (change / open_price) * 100
                volume = data['Volume'].iloc[-1] if len(data) > 0 else 0

                watchlist_data.append({
                    'Symbol': symbol,
                    'Price (₹)': f"{current:.2f}",
                    'Change': f"{change:+.2f}",
                    'Change %': f"{change_pct:+.1f}%",
                    'Volume': f"{volume:,}" if volume > 0 else "N/A",
                    'Market Cap': format_currency(info.get('marketCap', 0) * 83) if info.get('marketCap') else "N/A"
                })
        except Exception as e:
            watchlist_data.append({
                'Symbol': symbol,
                'Price (₹)': "Loading...",
                'Change': "",
                'Change %': "",
                'Volume': "",
                'Market Cap': ""
            })

    if watchlist_data:
        df = pd.DataFrame(watchlist_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ===== STOCK ANALYSIS TAB =====
elif selected_tab == "📈 Stock Analysis":
    st.header("📈 Stock Analysis")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Select Stock")

        popular_stocks = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC', 'HINDUNILVR',
            'BHARTIARTL', 'KOTAKBANK', 'ASIANPAINT', 'MARUTI', 'LT',
            'WIPRO', 'TECHM', 'HCLTECH', 'ICICIBANK', 'SBIN'
        ]

        selected_stock = st.selectbox(
            "Choose a stock:",
            popular_stocks,
            index=0
        )

        # Analysis type
        analysis_type = st.radio(
            "Analysis Type:",
            ["Quick Analysis", "Comprehensive AI Analysis"] if ENHANCED_AGENT_AVAILABLE else ["Quick Analysis"]
        )

        analyze_button = st.button("🔍 Analyze Stock", type="primary")

    with col2:
        if analyze_button:
            if analysis_type == "Quick Analysis":
                # Traditional quick analysis
                st.subheader(f"📊 Quick Analysis: {selected_stock}")

                try:
                    ticker = yf.Ticker(f"{selected_stock}.NS")
                    data = ticker.history(period="3mo")
                    info = ticker.info

                    if not data.empty:
                        # Price chart
                        fig = go.Figure(data=go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name=selected_stock
                        ))
                        fig.update_layout(
                            title=f"{selected_stock} Price Chart (3 months)",
                            xaxis_title="Date",
                            yaxis_title="Price (₹)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Key metrics
                        current = data['Close'].iloc[-1]
                        prev_close = info.get('previousClose', data['Close'].iloc[-2])
                        change = current - prev_close
                        change_pct = (change / prev_close) * 100

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Current Price", f"₹{current:.2f}", f"{change:+.2f} ({change_pct:+.1f}%)")

                        with col2:
                            if 'marketCap' in info:
                                market_cap_inr = info['marketCap'] * 83  # Approximate USD to INR
                                st.metric("Market Cap", format_currency(market_cap_inr))
                            else:
                                st.metric("Market Cap", "N/A")

                        with col3:
                            pe_ratio = info.get('trailingPE', 'N/A')
                            st.metric("P/E Ratio", f"{pe_ratio:.1f}" if pe_ratio != 'N/A' else "N/A")

                        with col4:
                            volume = data['Volume'].iloc[-1] if len(data) > 0 else 0
                            st.metric("Volume", f"{volume:,.0f}" if volume > 0 else "N/A")

                        # Additional info
                        with st.expander("📋 Additional Information"):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Fundamentals:**")
                                st.write(f"• 52W High: ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
                                st.write(f"• 52W Low: ₹{info.get('fiftyTwoWeekLow', 'N/A')}")
                                st.write(f"• Beta: {info.get('beta', 'N/A')}")
                                st.write(f"• Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "• Dividend Yield: N/A")

                            with col2:
                                st.write("**Business:**")
                                st.write(f"• Sector: {info.get('sector', 'N/A')}")
                                st.write(f"• Industry: {info.get('industry', 'N/A')}")
                                if 'longBusinessSummary' in info:
                                    summary = info['longBusinessSummary'][:200] + "..." if len(info['longBusinessSummary']) > 200 else info['longBusinessSummary']
                                    st.write(f"• Summary: {summary}")
                    else:
                        st.error(f"No data available for {selected_stock}")

                except Exception as e:
                    st.error(f"Error analyzing {selected_stock}: {str(e)}")

            elif analysis_type == "Comprehensive AI Analysis" and ENHANCED_AGENT_AVAILABLE:
                # AI-powered comprehensive analysis
                st.subheader(f"🤖 AI Analysis: {selected_stock}")

                with st.spinner("Agent is analyzing the stock... This may take 30-60 seconds."):
                    try:
                        # Run async analysis
                        async def run_analysis():
                            return await st.session_state.agent.analyze_stock_comprehensive(selected_stock)

                        # Create new event loop if none exists
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                        response = loop.run_until_complete(run_analysis())

                        if response.success:
                            # Display analysis
                            st.markdown("### 📊 AI Analysis Results")
                            st.markdown(response.final_answer)

                            # Show agent reasoning process
                            with st.expander("🧠 Agent Reasoning Process"):
                                st.write(f"**Execution Time:** {response.execution_time:.1f} seconds")
                                st.write(f"**Tools Used:** {', '.join(response.tools_used)}")
                                st.write(f"**Reasoning Steps:** {response.total_steps}")

                                for i, thought in enumerate(response.thoughts, 1):
                                    with st.container():
                                        st.markdown(f"**Step {i}:** {thought.action}")
                                        if thought.tool_name:
                                            st.markdown(f"*Tool:* `{thought.tool_name}`")
                                            st.markdown(f"*Observation:* {thought.observation}")
                        else:
                            st.error(f"AI Analysis failed: {response.error}")

                    except Exception as e:
                        st.error(f"Error running AI analysis: {str(e)}")

# ===== PORTFOLIO TAB =====
elif selected_tab == "💼 Portfolio":
    st.header("💼 Portfolio Management")

    # Portfolio input section
    st.subheader("📝 Portfolio Positions")

    # Use default portfolio if available
    if ENHANCED_AGENT_AVAILABLE and 'agent' in st.session_state:
        default_portfolio = st.session_state.config.default_portfolio
    else:
        default_portfolio = {
            'RELIANCE': {'qty': 100, 'avg_price': 2400.0},
            'HDFCBANK': {'qty': 75, 'avg_price': 1500.0},
            'TCS': {'qty': 50, 'avg_price': 3200.0},
            'INFY': {'qty': 200, 'avg_price': 1400.0}
        }

    # Portfolio editor
    with st.expander("✏️ Edit Portfolio", expanded=False):
        st.write("Add or modify your portfolio positions:")

        # Add new position
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            new_symbol = st.text_input("Stock Symbol", placeholder="e.g., RELIANCE")
        with col2:
            new_qty = st.number_input("Quantity", min_value=1, value=100)
        with col3:
            new_avg_price = st.number_input("Avg Price (₹)", min_value=0.01, value=100.0)
        with col4:
            if st.button("➕ Add Position"):
                if new_symbol:
                    default_portfolio[new_symbol.upper()] = {
                        'qty': new_qty,
                        'avg_price': new_avg_price
                    }
                    st.rerun()

    # Portfolio analysis
    if st.button("📊 Analyze Portfolio", type="primary"):
        if ENHANCED_AGENT_AVAILABLE and 'agent' in st.session_state:
            # AI-powered portfolio analysis
            st.subheader("🤖 AI Portfolio Analysis")

            with st.spinner("Agent is analyzing your portfolio... This may take 30-60 seconds."):
                try:
                    async def run_portfolio_analysis():
                        return await st.session_state.agent.portfolio_health_check(default_portfolio)

                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    response = loop.run_until_complete(run_portfolio_analysis())

                    if response.success:
                        st.markdown("### 📊 Portfolio Analysis")
                        st.markdown(response.final_answer)

                        with st.expander("🔍 Analysis Details"):
                            st.write(f"**Tools Used:** {', '.join(response.tools_used)}")
                            st.write(f"**Analysis Steps:** {response.total_steps}")
                            st.write(f"**Execution Time:** {response.execution_time:.1f} seconds")
                    else:
                        st.error(f"Portfolio analysis failed: {response.error}")

                except Exception as e:
                    st.error(f"Error running portfolio analysis: {str(e)}")
        else:
            # Basic portfolio analysis
            st.subheader("📊 Portfolio Summary")

            portfolio_data = []
            total_invested = 0
            total_current_value = 0

            for symbol, position in default_portfolio.items():
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    data = ticker.history(period="1d")

                    if not data.empty:
                        current_price = data['Close'].iloc[-1]
                        qty = position['qty']
                        avg_price = position['avg_price']

                        invested = qty * avg_price
                        current_value = qty * current_price
                        pnl = current_value - invested
                        pnl_pct = (pnl / invested) * 100

                        total_invested += invested
                        total_current_value += current_value

                        portfolio_data.append({
                            'Symbol': symbol,
                            'Qty': qty,
                            'Avg Price': f"₹{avg_price:.2f}",
                            'Current Price': f"₹{current_price:.2f}",
                            'Invested': f"₹{invested:,.0f}",
                            'Current Value': f"₹{current_value:,.0f}",
                            'P&L': f"₹{pnl:,.0f}",
                            'P&L %': f"{pnl_pct:+.1f}%"
                        })
                except Exception as e:
                    portfolio_data.append({
                        'Symbol': symbol,
                        'Qty': position['qty'],
                        'Avg Price': f"₹{position['avg_price']:.2f}",
                        'Current Price': "Loading...",
                        'Invested': f"₹{position['qty'] * position['avg_price']:,.0f}",
                        'Current Value': "Loading...",
                        'P&L': "Loading...",
                        'P&L %': "Loading..."
                    })

            if portfolio_data:
                df = pd.DataFrame(portfolio_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Portfolio summary
                total_pnl = total_current_value - total_invested
                total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Invested", format_currency(total_invested))
                with col2:
                    st.metric("Current Value", format_currency(total_current_value))
                with col3:
                    st.metric("Total P&L", format_currency(total_pnl), f"{total_pnl_pct:+.1f}%")

# ===== AGENT ACTIVITY TAB =====
elif selected_tab == "🤖 Agent Activity":
    st.header("🤖 Agent Activity Monitor")

    if not ENHANCED_AGENT_AVAILABLE:
        st.warning("Enhanced AI Agent is not available. Please check your configuration.")
        st.stop()

    # Agent status overview
    agent_stats = st.session_state.agent.get_agent_stats()

    st.markdown('<div class="agent-status">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Agent Status", "🟢 Active" if 'agent' in st.session_state else "🔴 Inactive")

    with col2:
        st.metric("Success Rate", f"{agent_stats.get('success_rate', 0):.1f}%")

    with col3:
        st.metric("Total Decisions", agent_stats.get('total_decisions', 0))

    with col4:
        st.metric("Tools Available", agent_stats.get('tools_available', 0))

    st.markdown('</div>', unsafe_allow_html=True)

    # Activity tabs
    activity_tab = st.radio(
        "View Activity:",
        ["Recent Decisions", "Alerts Sent", "Analysis History", "Tool Usage"],
        horizontal=True
    )

    if activity_tab == "Recent Decisions":
        st.subheader("🧠 Recent Agent Decisions")

        try:
            decisions = st.session_state.agent.memory.get_agent_decisions(days_back=7)

            if decisions:
                for decision in decisions[:10]:  # Show last 10
                    with st.container():
                        st.markdown(f'<div class="decision-card">', unsafe_allow_html=True)

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**{decision['decision_type'].title()}**")
                            st.write(f"Context: {decision['context']}")
                            st.write(f"Action: {decision['action_taken']}")

                            if decision.get('reasoning'):
                                with st.expander("🤔 Reasoning"):
                                    st.write(decision['reasoning'])

                        with col2:
                            timestamp = datetime.fromisoformat(decision['timestamp'].replace('Z', '+00:00'))
                            st.write(f"**Time:** {timestamp.strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"**Outcome:** {decision.get('outcome', 'N/A')}")

                            confidence = decision.get('confidence', 0)
                            if confidence > 0:
                                st.progress(confidence, text=f"Confidence: {confidence:.0%}")

                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No recent decisions found.")

        except Exception as e:
            st.error(f"Error loading decisions: {e}")

    elif activity_tab == "Alerts Sent":
        st.subheader("🚨 Recent Alerts")

        try:
            alerts = st.session_state.agent.memory.get_recent_alerts(days_back=7)

            if alerts:
                for alert in alerts[:20]:  # Show last 20
                    with st.container():
                        st.markdown(f'<div class="alert-card">', unsafe_allow_html=True)

                        col1, col2 = st.columns([4, 1])

                        with col1:
                            alert_type = alert.get('alert_type', 'info')
                            emoji = {'info': '📢', 'warning': '⚠️', 'critical': '🚨', 'success': '✅'}.get(alert_type, '📢')

                            st.markdown(f"**{emoji} {alert_type.title()} Alert**")
                            st.write(alert['message'])

                            if alert.get('symbol'):
                                st.write(f"Symbol: {alert['symbol']}")

                        with col2:
                            timestamp = datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))
                            st.write(f"**Time:** {timestamp.strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"**Sent:** {'✅' if alert.get('sent') else '❌'}")

                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No recent alerts found.")

        except Exception as e:
            st.error(f"Error loading alerts: {e}")

    elif activity_tab == "Analysis History":
        st.subheader("📊 Analysis History")

        try:
            analyses = st.session_state.agent.memory.get_recent_analyses(days_back=30)

            if analyses:
                # Group by analysis type
                analysis_types = list(set(a['analysis_type'] for a in analyses))
                selected_type = st.selectbox("Filter by type:", ['All'] + analysis_types)

                filtered_analyses = analyses if selected_type == 'All' else [a for a in analyses if a['analysis_type'] == selected_type]

                for analysis in filtered_analyses[:15]:  # Show last 15
                    with st.expander(f"{analysis['analysis_type'].title()} - {analysis.get('symbol', 'General')} ({datetime.fromisoformat(analysis['timestamp'].replace('Z', '+00:00')).strftime('%m/%d %H:%M')})"):

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Analysis Data:**")
                            if analysis.get('data'):
                                st.json(analysis['data'])

                        with col2:
                            st.write("**Details:**")
                            st.write(f"Type: {analysis['analysis_type']}")
                            st.write(f"Symbol: {analysis.get('symbol', 'N/A')}")
                            st.write(f"Source: {analysis.get('source', 'N/A')}")

                            if analysis.get('reasoning'):
                                st.write("**Reasoning:**")
                                st.write(analysis['reasoning'])

                            if analysis.get('recommendation'):
                                st.write(f"**Recommendation:** {analysis['recommendation']}")
            else:
                st.info("No analysis history found.")

        except Exception as e:
            st.error(f"Error loading analysis history: {e}")

    elif activity_tab == "Tool Usage":
        st.subheader("🔧 Tool Usage Statistics")

        # Show available tools
        st.write("**Available Tools:**")

        tools_info = st.session_state.agent.tools.get_tool_definitions()

        for tool in tools_info:
            with st.container():
                st.markdown(f'<div class="tool-usage">', unsafe_allow_html=True)
                st.markdown(f"**{tool['name']}**")
                st.write(tool['description'])
                st.markdown('</div>', unsafe_allow_html=True)

# ===== AGENT SETTINGS TAB =====
elif selected_tab == "⚙️ Agent Settings":
    st.header("⚙️ Agent Configuration")

    if not ENHANCED_AGENT_AVAILABLE:
        st.warning("Enhanced AI Agent is not available. Please check your configuration.")
        st.stop()

    # Settings sections
    settings_section = st.radio(
        "Settings Category:",
        ["General Settings", "Watchlist", "Risk Management", "Notifications", "Scheduling"],
        horizontal=True
    )

    if settings_section == "General Settings":
        st.subheader("🔧 General Agent Settings")

        with st.form("general_settings"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Model Configuration:**")
                model_name = st.selectbox(
                    "AI Model:",
                    ["claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514"],
                    index=0
                )

                max_tokens = st.slider("Max Tokens:", 1000, 8000, st.session_state.config.max_tokens)
                temperature = st.slider("Temperature:", 0.0, 1.0, st.session_state.config.temperature, 0.1)

            with col2:
                st.write("**Agent Behavior:**")
                max_iterations = st.slider("Max Reasoning Steps:", 5, 25, 15)

                risk_tolerance = st.selectbox(
                    "Risk Tolerance:",
                    ["low", "medium", "high"],
                    index=["low", "medium", "high"].index(st.session_state.config.risk_tolerance)
                )

                alert_threshold = st.slider("Alert Threshold (%):", 1.0, 10.0, st.session_state.config.alert_threshold)

            if st.form_submit_button("💾 Save General Settings"):
                st.success("General settings saved! (Note: Some changes require restart)")

    elif settings_section == "Watchlist":
        st.subheader("👀 Watchlist Management")

        current_watchlist = st.session_state.config.default_watchlist

        # Display current watchlist
        st.write("**Current Watchlist:**")
        watchlist_df = pd.DataFrame({'Symbol': current_watchlist})
        st.dataframe(watchlist_df, hide_index=True)

        # Add/remove stocks
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Add Stock:**")
            new_stock = st.text_input("Stock Symbol:", placeholder="e.g., WIPRO")
            if st.button("➕ Add to Watchlist"):
                if new_stock and new_stock.upper() not in current_watchlist:
                    current_watchlist.append(new_stock.upper())
                    st.success(f"Added {new_stock.upper()} to watchlist")
                    st.rerun()

        with col2:
            st.write("**Remove Stock:**")
            if current_watchlist:
                stock_to_remove = st.selectbox("Select stock to remove:", current_watchlist)
                if st.button("➖ Remove from Watchlist"):
                    current_watchlist.remove(stock_to_remove)
                    st.success(f"Removed {stock_to_remove} from watchlist")
                    st.rerun()

    elif settings_section == "Risk Management":
        st.subheader("⚖️ Risk Management")

        with st.form("risk_settings"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Position Limits:**")
                max_position_size = st.slider(
                    "Max Position Size (% of portfolio):",
                    1.0, 50.0,
                    st.session_state.config.max_position_size
                )

                stop_loss_threshold = st.slider(
                    "Default Stop Loss (%):",
                    1.0, 20.0,
                    st.session_state.config.stop_loss_threshold
                )

            with col2:
                st.write("**Risk Metrics:**")
                portfolio_rebalance_threshold = st.slider(
                    "Rebalancing Threshold (%):",
                    1.0, 20.0,
                    st.session_state.config.portfolio_rebalance_threshold
                )

                # Display risk multipliers
                risk_multipliers = {
                    'low': {'alert': 0.7, 'position': 0.5},
                    'medium': {'alert': 1.0, 'position': 1.0},
                    'high': {'alert': 1.5, 'position': 1.5}
                }

                current_multiplier = risk_multipliers[st.session_state.config.risk_tolerance]

                st.info(f"**Current Risk Profile: {st.session_state.config.risk_tolerance.title()}**\n"
                       f"- Alert Sensitivity: {current_multiplier['alert']}x\n"
                       f"- Position Sizing: {current_multiplier['position']}x")

            if st.form_submit_button("💾 Save Risk Settings"):
                st.success("Risk settings saved!")

    elif settings_section == "Notifications":
        st.subheader("📱 Notification Settings")

        # Check Telegram configuration
        telegram_enabled = st.session_state.config.telegram_enabled

        if telegram_enabled:
            st.success("✅ Telegram notifications are enabled")

            # Test notification
            if st.button("📤 Send Test Notification"):
                try:
                    result = st.session_state.agent.notifications.send_notification(
                        "🧪 Test notification from Indian Stock Market AI Agent Dashboard",
                        "info"
                    )
                    if result:
                        st.success("Test notification sent successfully!")
                    else:
                        st.error("Failed to send test notification")
                except Exception as e:
                    st.error(f"Error sending test notification: {e}")
        else:
            st.warning("❌ Telegram notifications are disabled")
            st.info("""
                To enable Telegram notifications:
                1. Create a Telegram bot with @BotFather
                2. Get your bot token
                3. Set environment variables:
                   - TELEGRAM_BOT_TOKEN=your_bot_token
                   - TELEGRAM_CHAT_ID=your_chat_id
                4. Restart the application
            """)

        # Notification preferences
        with st.form("notification_settings"):
            st.write("**Notification Preferences:**")

            morning_briefing = st.checkbox("Morning Briefing", value=True)
            evening_briefing = st.checkbox("Evening Briefing", value=True)
            price_alerts = st.checkbox("Price Movement Alerts", value=True)
            portfolio_updates = st.checkbox("Portfolio Updates", value=True)

            if st.form_submit_button("💾 Save Notification Settings"):
                st.success("Notification preferences saved!")

    elif settings_section == "Scheduling":
        st.subheader("🕐 Autonomous Scheduling")

        with st.form("schedule_settings"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Daily Schedules:**")

                morning_time = st.time_input(
                    "Morning Run Time:",
                    value=datetime.strptime(st.session_state.config.morning_run_time, "%H:%M").time()
                )

                evening_time = st.time_input(
                    "Evening Run Time:",
                    value=datetime.strptime(st.session_state.config.evening_run_time, "%H:%M").time()
                )

                scheduler_enabled = st.checkbox(
                    "Enable Autonomous Scheduling",
                    value=st.session_state.config.scheduler_enabled
                )

            with col2:
                st.write("**Schedule Info:**")
                st.info(f"""
                    **Current Schedule:**
                    - Morning Analysis: {st.session_state.config.morning_run_time} IST
                    - Evening Analysis: {st.session_state.config.evening_run_time} IST
                    - Status: {'✅ Enabled' if st.session_state.config.scheduler_enabled else '❌ Disabled'}

                    **What happens:**
                    - Morning: Pre-market analysis and briefing
                    - Evening: Post-market review and global outlook
                    - Continuous: Market hours monitoring (if enabled)
                """)

            if st.form_submit_button("💾 Save Schedule Settings"):
                st.success("Schedule settings saved! Restart required to apply changes.")

        # Manual triggers
        st.write("**Manual Triggers:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🌅 Run Morning Analysis Now"):
                with st.spinner("Running morning analysis..."):
                    try:
                        async def run_morning():
                            return await st.session_state.agent.autonomous_morning_analysis()

                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                        response = loop.run_until_complete(run_morning())

                        if response.success:
                            st.success("Morning analysis completed!")
                            with st.expander("📊 View Results"):
                                st.markdown(response.final_answer)
                        else:
                            st.error(f"Morning analysis failed: {response.error}")
                    except Exception as e:
                        st.error(f"Error running morning analysis: {e}")

        with col2:
            if st.button("🌆 Run Evening Analysis Now"):
                with st.spinner("Running evening analysis..."):
                    try:
                        async def run_evening():
                            return await st.session_state.agent.autonomous_evening_analysis()

                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                        response = loop.run_until_complete(run_evening())

                        if response.success:
                            st.success("Evening analysis completed!")
                            with st.expander("📊 View Results"):
                                st.markdown(response.final_answer)
                        else:
                            st.error(f"Evening analysis failed: {response.error}")
                    except Exception as e:
                        st.error(f"Error running evening analysis: {e}")

        with col3:
            if st.button("🔍 Portfolio Health Check"):
                with st.spinner("Analyzing portfolio..."):
                    try:
                        async def run_portfolio():
                            return await st.session_state.agent.portfolio_health_check()

                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                        response = loop.run_until_complete(run_portfolio())

                        if response.success:
                            st.success("Portfolio analysis completed!")
                            with st.expander("📊 View Results"):
                                st.markdown(response.final_answer)
                        else:
                            st.error(f"Portfolio analysis failed: {response.error}")
                    except Exception as e:
                        st.error(f"Error running portfolio analysis: {e}")

# ===== AI ASSISTANT TAB =====
elif selected_tab == "💬 AI Assistant":
    st.header("💬 AI Assistant Chat")

    if not ENHANCED_AGENT_AVAILABLE:
        st.warning("Enhanced AI Agent is not available. Please use the basic assistant.")

        # Basic assistant (fallback)
        user_question = st.text_area(
            "Ask about Indian stocks, markets, or investments:",
            placeholder="e.g., What's your view on RELIANCE stock? or How is the IT sector performing?"
        )

        if st.button("💭 Get Analysis", type="primary"):
            if user_question:
                st.info("Basic analysis mode - Enhanced AI Agent not available")
                # Could implement basic analysis here
            else:
                st.warning("Please enter your question")

    else:
        # Enhanced AI Assistant with agent loop
        st.write("Ask the AI Agent anything about Indian stocks, markets, or investments. The agent will use multiple tools to provide comprehensive analysis.")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for i, (question, response, tools_used) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**🙋 You:** {question}")
                st.markdown(f"**🤖 Agent:** {response}")
                if tools_used:
                    st.markdown(f"*Tools used: {', '.join(tools_used)}*")
                st.divider()

        # User input
        user_question = st.text_area(
            "Your Question:",
            placeholder="e.g., Should I buy RELIANCE stock now? What's the outlook for IT stocks? How is my portfolio performing?",
            height=100
        )

        col1, col2 = st.columns([1, 4])

        with col1:
            if st.button("🚀 Ask Agent", type="primary"):
                if user_question.strip():
                    with st.spinner("Agent is thinking and using tools... This may take 30-60 seconds."):
                        try:
                            async def run_question():
                                return await st.session_state.agent.run_agent_loop(user_question)

                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)

                            response = loop.run_until_complete(run_question())

                            if response.success:
                                # Add to chat history
                                st.session_state.chat_history.append((
                                    user_question,
                                    response.final_answer,
                                    response.tools_used
                                ))

                                # Keep only last 10 conversations
                                if len(st.session_state.chat_history) > 10:
                                    st.session_state.chat_history = st.session_state.chat_history[-10:]

                                st.rerun()
                            else:
                                st.error(f"Agent failed to respond: {response.error}")

                        except Exception as e:
                            st.error(f"Error asking agent: {e}")
                else:
                    st.warning("Please enter your question")

        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

        # Suggested questions
        st.subheader("💡 Suggested Questions")

        suggestions = [
            "What's your analysis of RELIANCE stock right now?",
            "How is the banking sector performing today?",
            "Should I invest in IT stocks considering the current USD/INR rate?",
            "What are the top 3 stocks to watch this week?",
            "How is my portfolio diversification? Any rebalancing needed?",
            "What's the impact of recent FII/DII flows on the market?",
            "Which sectors are looking strong for the next quarter?"
        ]

        suggestion_cols = st.columns(2)

        for i, suggestion in enumerate(suggestions):
            with suggestion_cols[i % 2]:
                if st.button(f"💭 {suggestion[:50]}...", key=f"suggest_{i}"):
                    # Set the suggestion as the question and trigger analysis
                    st.session_state.suggested_question = suggestion
                    st.rerun()

        # Handle suggested question
        if hasattr(st.session_state, 'suggested_question'):
            with st.spinner("Agent is analyzing your suggested question..."):
                try:
                    async def run_suggestion():
                        return await st.session_state.agent.run_agent_loop(st.session_state.suggested_question)

                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    response = loop.run_until_complete(run_suggestion())

                    if response.success:
                        st.session_state.chat_history.append((
                            st.session_state.suggested_question,
                            response.final_answer,
                            response.tools_used
                        ))

                        if len(st.session_state.chat_history) > 10:
                            st.session_state.chat_history = st.session_state.chat_history[-10:]

                    # Clear the suggestion
                    del st.session_state.suggested_question
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing suggestion: {e}")
                    del st.session_state.suggested_question

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🇮🇳 Indian Stock Market AI Agent Dashboard v2.0<br>
        Enhanced with Autonomous Reasoning • Real-time Analysis • Telegram Alerts<br>
        <small>⚠️ For educational purposes only. Not financial advice. Please consult a financial advisor.</small>
    </div>
    """,
    unsafe_allow_html=True
)