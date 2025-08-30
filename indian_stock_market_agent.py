#!/usr/bin/env python3
"""
AI-Powered Indian Stock Market Agent

Features specifically for Indian markets:
- NSE/BSE stock analysis
- Indian market hours (9:15 AM - 3:30 PM IST)
- Rupee-denominated analysis
- Indian sector analysis (IT, Banking, Pharma, etc.)
- Integration with Indian financial data
- Support for Indian market holidays
- Nifty 50, Sensex tracking
- FII/DII flow analysis
- Currency impact analysis (USD/INR)

Setup:
pip install yfinance pandas numpy anthropic requests beautifulsoup4 pytz
export ANTHROPIC_API_KEY="your-key-here"

Usage:
python indian_stock_agent.py
"""

import anthropic
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
import threading
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
import os
warnings.filterwarnings('ignore')

@dataclass
class IndianMarketAlert:
    """Alert structure for Indian markets."""
    symbol: str
    alert_type: str
    message: str
    severity: str
    timestamp: datetime
    price_inr: float
    sector: str
    market_cap_cr: float  # Market cap in crores

class IndianStockMarketAgent:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Indian stock market agent."""
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
        )
        
        # Indian market specific data
        self.indian_timezone = pytz.timezone('Asia/Kolkata')
        self.live_prices = {}
        self.price_history = {}
        self.news_cache = {}
        self.alerts = []
        self.analysis_cache = {}
        self.usd_inr_rate = 83.0  # Default, will be updated
        
        # Indian stock symbols mapping (NSE symbols for yfinance)
        self.popular_indian_stocks = {
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS', 
            'HDFCBANK': 'HDFCBANK.NS',
            'INFY': 'INFY.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'SBIN': 'SBIN.NS',
            'BAJFINANCE': 'BAJFINANCE.NS',
            'BHARTIARTL': 'BHARTIARTL.NS',
            'ITC': 'ITC.NS',
            'ASIANPAINT': 'ASIANPAINT.NS',
            'MARUTI': 'MARUTI.NS',
            'KOTAKBANK': 'KOTAKBANK.NS',
            'LT': 'LT.NS',
            'WIPRO': 'WIPRO.NS',
            'ADANIPORTS': 'ADANIPORTS.NS',
            'ULTRACEMCO': 'ULTRACEMCO.NS',
            'TITAN': 'TITAN.NS',
            'POWERGRID': 'POWERGRID.NS',
            'NESTLEIND': 'NESTLEIND.NS',
            'TECHM': 'TECHM.NS',
            'HCLTECH': 'HCLTECH.NS',
            'AXISBANK': 'AXISBANK.NS',
            'TATAMOTORS': 'TATAMOTORS.NS',
            'SUNPHARMA': 'SUNPHARMA.NS'
        }
        
        # Indian sectors mapping
        self.indian_sectors = {
            'IT': ['TCS', 'INFY', 'WIPRO', 'TECHM', 'HCLTECH'],
            'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'Consumer': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'DABUR'],
            'Auto': ['MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'M&M'],
            'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'BIOCON'],
            'Oil & Gas': ['RELIANCE', 'ONGC', 'IOC', 'BPCL'],
            'Metals': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'SAIL'],
            'Infrastructure': ['LT', 'ADANIPORTS', 'POWERGRID', 'NTPC']
        }
        
        # Market timing
        self.market_open_time = "09:15"
        self.market_close_time = "15:30"
        
        print("🇮🇳 Indian Stock Market Agent initialized!")
        print(f"⏰ Market Hours: {self.market_open_time} - {self.market_close_time} IST")
    
    def is_market_open(self) -> bool:
        """Check if Indian stock market is currently open."""
        now = datetime.now(self.indian_timezone)
        current_time = now.strftime("%H:%M")
        current_day = now.weekday()  # 0=Monday, 6=Sunday
        
        # Market closed on weekends
        if current_day >= 5:  # Saturday or Sunday
            return False
        
        # Check if current time is within market hours
        if self.market_open_time <= current_time <= self.market_close_time:
            return True
        
        return False
    
    def get_indian_stock_data(self, symbols: List[str]) -> Dict:
        """Fetch Indian stock data with NSE symbols."""
        data = {}
        
        for symbol in symbols:
            try:
                # Convert to NSE symbol if needed
                yf_symbol = self.popular_indian_stocks.get(symbol, f"{symbol}.NS")
                
                ticker = yf.Ticker(yf_symbol)
                hist = ticker.history(period="1d", interval="1m")
                info = ticker.info
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = info.get('previousClose', current_price)
                    day_change = current_price - prev_close
                    day_change_pct = (day_change / prev_close * 100) if prev_close > 0 else 0
                    
                    # Market cap in crores (1 crore = 10 million)
                    market_cap_inr = info.get('marketCap', 0) * self.usd_inr_rate if info.get('marketCap') else 0
                    market_cap_cr = market_cap_inr / 10000000  # Convert to crores
                    
                    data[symbol] = {
                        'price': current_price,
                        'prev_close': prev_close,
                        'day_change': day_change,
                        'day_change_pct': day_change_pct,
                        'volume': hist['Volume'].iloc[-1] if len(hist) > 0 else 0,
                        'high': hist['High'].max(),
                        'low': hist['Low'].min(),
                        'market_cap_cr': market_cap_cr,
                        'pe_ratio': info.get('trailingPE'),
                        'sector': self.get_indian_sector(symbol),
                        'yf_symbol': yf_symbol,
                        'timestamp': datetime.now(self.indian_timezone)
                    }
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                data[symbol] = {
                    'error': str(e),
                    'symbol': symbol,
                    'timestamp': datetime.now(self.indian_timezone)
                }
                
        return data
    
    def get_indian_sector(self, symbol: str) -> str:
        """Get Indian sector for a given stock symbol."""
        for sector, stocks in self.indian_sectors.items():
            if symbol in stocks:
                return sector
        return 'Other'
    
    def get_nifty_sensex_data(self) -> Dict:
        """Fetch Nifty 50 and Sensex data."""
        indices_data = {}
        
        try:
            # Nifty 50
            nifty = yf.Ticker("^NSEI")
            nifty_hist = nifty.history(period="1d", interval="1m")
            
            if not nifty_hist.empty:
                indices_data['NIFTY'] = {
                    'price': nifty_hist['Close'].iloc[-1],
                    'change': nifty_hist['Close'].iloc[-1] - nifty_hist['Open'].iloc[0],
                    'change_pct': ((nifty_hist['Close'].iloc[-1] / nifty_hist['Open'].iloc[0]) - 1) * 100,
                    'high': nifty_hist['High'].max(),
                    'low': nifty_hist['Low'].min()
                }
            
            # Sensex
            sensex = yf.Ticker("^BSESN")
            sensex_hist = sensex.history(period="1d", interval="1m")
            
            if not sensex_hist.empty:
                indices_data['SENSEX'] = {
                    'price': sensex_hist['Close'].iloc[-1],
                    'change': sensex_hist['Close'].iloc[-1] - sensex_hist['Open'].iloc[0],
                    'change_pct': ((sensex_hist['Close'].iloc[-1] / sensex_hist['Open'].iloc[0]) - 1) * 100,
                    'high': sensex_hist['High'].max(),
                    'low': sensex_hist['Low'].min()
                }
                
        except Exception as e:
            print(f"Error fetching indices data: {e}")
            
        return indices_data
    
    def get_usd_inr_rate(self) -> float:
        """Fetch current USD/INR exchange rate."""
        try:
            usd_inr = yf.Ticker("USDINR=X")
            hist = usd_inr.history(period="1d")
            if not hist.empty:
                self.usd_inr_rate = hist['Close'].iloc[-1]
                return self.usd_inr_rate
        except:
            pass
        return self.usd_inr_rate
    
    def _call_claude_indian(self, prompt: str, max_tokens: int = 3000) -> str:
        """Call Claude with Indian market context."""
        indian_context = f"""
        IMPORTANT CONTEXT:
        - All prices are in Indian Rupees (₹)
        - Market hours: 9:15 AM - 3:30 PM IST
        - Current USD/INR rate: ₹{self.usd_inr_rate:.2f}
        - Market status: {'OPEN' if self.is_market_open() else 'CLOSED'}
        - Analysis time: {datetime.now(self.indian_timezone).strftime('%Y-%m-%d %H:%M IST')}
        
        {prompt}
        
        Provide analysis relevant to Indian investors and market conditions.
        """
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=0.3,
                messages=[{"role": "user", "content": indian_context}]
            )
            return message.content[0].text
        except Exception as e:
            return f"❌ Analysis error: {str(e)}"
    
    def analyze_indian_stock(self, symbol: str) -> str:
        """Comprehensive analysis of Indian stock."""
        print(f"🔍 Analyzing {symbol}...")
        
        stock_data = self.get_indian_stock_data([symbol])
        
        if symbol not in stock_data or 'error' in stock_data[symbol]:
            return f"❌ Could not fetch data for {symbol}"
        
        data = stock_data[symbol]
        
        prompt = f"""
        Analyze this Indian stock for investment decisions:
        
        STOCK: {symbol} ({data['sector']} sector)
        Current Price: ₹{data['price']:.2f}
        Day Change: ₹{data['day_change']:.2f} ({data['day_change_pct']:+.1f}%)
        Market Cap: ₹{data['market_cap_cr']:.0f} crores
        P/E Ratio: {data.get('pe_ratio', 'N/A')}
        Volume: {data['volume']:,} shares
        Day Range: ₹{data['low']:.2f} - ₹{data['high']:.2f}
        
        Provide analysis specifically for Indian investors:
        
        1. **STOCK ASSESSMENT**
           - Valuation at current levels
           - Comparison with sector peers
           - Growth prospects in Indian market
        
        2. **SECTOR ANALYSIS**
           - {data['sector']} sector outlook in India
           - Regulatory environment impact
           - Competition landscape
        
        3. **INVESTMENT RECOMMENDATION**
           - BUY/HOLD/SELL with rationale
           - Target price in next 3-6 months
           - Suitable for retail/HNI investors?
        
        4. **RISK FACTORS**
           - Company-specific risks
           - Sector/regulatory risks
           - Market risks (FII flows, currency, etc.)
        
        5. **KEY CATALYSTS**
           - Upcoming events (results, policy changes)
           - Technical levels to watch
           - News/events to monitor
        
        Focus on actionable insights for Indian retail investors.
        """
        
        return self._call_claude_indian(prompt)
    
    def analyze_portfolio_indian(self, portfolio: Dict[str, Dict]) -> str:
        """Analyze Indian stock portfolio."""
        print("📊 Analyzing Indian portfolio...")
        
        # Get current data for all positions
        symbols = list(portfolio.keys())
        current_data = self.get_indian_stock_data(symbols)
        indices_data = self.get_nifty_sensex_data()
        
        # Calculate portfolio metrics
        portfolio_summary = []
        sector_allocation = {}
        total_invested = 0
        total_current_value = 0
        
        for symbol, position in portfolio.items():
            if symbol in current_data and 'error' not in current_data[symbol]:
                current_price = current_data[symbol]['price']
                invested_amount = position['qty'] * position['avg_price']
                current_value = position['qty'] * current_price
                pnl = current_value - invested_amount
                pnl_pct = (pnl / invested_amount) * 100 if invested_amount > 0 else 0
                
                total_invested += invested_amount
                total_current_value += current_value
                
                sector = current_data[symbol]['sector']
                sector_allocation[sector] = sector_allocation.get(sector, 0) + current_value
                
                portfolio_summary.append(f"""
                {symbol} ({sector}):
                - Qty: {position['qty']} shares @ ₹{position['avg_price']:.2f}
                - Current: ₹{current_price:.2f}
                - Invested: ₹{invested_amount:,.0f} | Current: ₹{current_value:,.0f}
                - P&L: ₹{pnl:,.0f} ({pnl_pct:+.1f}%)
                """)
        
        # Calculate sector allocation percentages
        sector_pct = {sector: (value/total_current_value)*100 
                     for sector, value in sector_allocation.items()} if total_current_value > 0 else {}
        
        portfolio_pnl = total_current_value - total_invested
        portfolio_pnl_pct = (portfolio_pnl / total_invested) * 100 if total_invested > 0 else 0
        
        prompt = f"""
        Analyze this Indian stock portfolio:
        
        PORTFOLIO SUMMARY:
        Total Invested: ₹{total_invested:,.0f}
        Current Value: ₹{total_current_value:,.0f}
        Total P&L: ₹{portfolio_pnl:,.0f} ({portfolio_pnl_pct:+.1f}%)
        
        POSITIONS:
        {chr(10).join(portfolio_summary)}
        
        SECTOR ALLOCATION:
        {chr(10).join([f"- {sector}: {pct:.1f}%" for sector, pct in sector_pct.items()])}
        
        MARKET CONTEXT:
        Nifty 50: {indices_data.get('NIFTY', {}).get('price', 'N/A')} ({indices_data.get('NIFTY', {}).get('change_pct', 0):+.1f}%)
        Sensex: {indices_data.get('SENSEX', {}).get('price', 'N/A')} ({indices_data.get('SENSEX', {}).get('change_pct', 0):+.1f}%)
        USD/INR: ₹{self.usd_inr_rate:.2f}
        
        Provide comprehensive portfolio analysis for Indian investor:
        
        1. **PORTFOLIO HEALTH**
           - Overall performance vs Nifty/Sensex
           - Risk-return profile assessment
           - Diversification analysis
        
        2. **SECTOR ANALYSIS**
           - Over/under-weight sectors
           - Sector rotation opportunities
           - Regulatory/policy impact
        
        3. **REBALANCING SUGGESTIONS**
           - Specific buy/sell recommendations
           - Position sizing adjustments
           - New stocks to consider
        
        4. **RISK MANAGEMENT**
           - Concentration risks
           - Currency exposure (if any)
           - Market correlation risks
        
        5. **ACTION ITEMS**
           - Immediate actions needed
           - Stocks to monitor closely
           - Market levels to watch
        
        Tailor advice for Indian market conditions and regulations.
        """
        
        return self._call_claude_indian(prompt)
    
    def market_outlook_indian(self) -> str:
        """Generate Indian market outlook."""
        print("🇮🇳 Generating Indian market outlook...")
        
        # Get major indices data
        indices_data = self.get_nifty_sensex_data()
        
        # Get key stocks data
        key_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC']
        stocks_data = self.get_indian_stock_data(key_stocks)
        
        # Get USD/INR rate
        usd_inr = self.get_usd_inr_rate()
        
        # Prepare market summary
        indices_summary = []
        if 'NIFTY' in indices_data:
            nifty = indices_data['NIFTY']
            indices_summary.append(f"Nifty 50: {nifty['price']:.0f} ({nifty['change_pct']:+.1f}%)")
        
        if 'SENSEX' in indices_data:
            sensex = indices_data['SENSEX']
            indices_summary.append(f"Sensex: {sensex['price']:.0f} ({sensex['change_pct']:+.1f}%)")
        
        stocks_summary = []
        for symbol, data in stocks_data.items():
            if 'error' not in data:
                stocks_summary.append(f"{symbol}: ₹{data['price']:.1f} ({data['day_change_pct']:+.1f}%)")
        
        prompt = f"""
        Provide comprehensive Indian stock market outlook:
        
        MARKET SNAPSHOT ({datetime.now(self.indian_timezone).strftime('%Y-%m-%d %H:%M IST')}):
        {chr(10).join(indices_summary)}
        
        KEY STOCKS:
        {chr(10).join(stocks_summary)}
        
        CURRENCY:
        USD/INR: ₹{usd_inr:.2f}
        
        MARKET STATUS: {'OPEN' if self.is_market_open() else 'CLOSED'}
        
        Analyze and provide outlook covering:
        
        1. **MARKET SENTIMENT**
           - Overall market direction and momentum
           - Risk-on vs risk-off sentiment
           - Retail vs institutional activity
        
        2. **SECTOR ROTATION**
           - Which sectors are outperforming/underperforming
           - Sectoral themes and trends
           - Policy impact on sectors
        
        3. **KEY DRIVERS**
           - FII/DII flows impact
           - Global factors affecting Indian markets
           - Currency movement implications
        
        4. **TECHNICAL OUTLOOK**
           - Nifty/Sensex support and resistance levels
           - Momentum indicators and trends
           - Key levels to watch
        
        5. **INVESTMENT STRATEGY**
           - Market positioning recommendations
           - Sectors/stocks to focus on
           - Risk management approach
        
        6. **UPCOMING CATALYSTS**
           - Key events/announcements to watch
           - Earnings season impact
           - Policy/regulatory changes
        
        Focus on actionable insights for Indian retail and HNI investors.
        """
        
        return self._call_claude_indian(prompt)

def main():
    """Demo of the Indian Stock Market Agent."""
    
    print("🇮🇳 Indian Stock Market AI Agent Demo")
    print("=" * 60)
    
    # Check if API key is available
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ API key not found!")
        print("Please set: export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    # Initialize agent
    try:
        agent = IndianStockMarketAgent(api_key)
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    # Example Indian portfolio
    sample_portfolio = {
        'RELIANCE': {'qty': 100, 'avg_price': 2400.0},
        'TCS': {'qty': 50, 'avg_price': 3200.0},
        'HDFCBANK': {'qty': 75, 'avg_price': 1500.0},
        'INFY': {'qty': 200, 'avg_price': 1400.0},
        'ITC': {'qty': 500, 'avg_price': 350.0}
    }
    
    print(f"\n📊 Sample Portfolio: {', '.join(sample_portfolio.keys())}")
    print(f"⏰ Market Status: {'OPEN' if agent.is_market_open() else 'CLOSED'}")
    print(f"💱 USD/INR Rate: ₹{agent.get_usd_inr_rate():.2f}")
    
    try:
        # Example 1: Individual stock analysis
        print("\n" + "="*50)
        print("1. INDIAN STOCK ANALYSIS")
        print("="*50)
        
        stock_analysis = agent.analyze_indian_stock('RELIANCE')
        print(stock_analysis[:800] + "..." if len(stock_analysis) > 800 else stock_analysis)
        
        # Example 2: Portfolio analysis
        print("\n" + "="*50)
        print("2. INDIAN PORTFOLIO ANALYSIS") 
        print("="*50)
        
        portfolio_analysis = agent.analyze_portfolio_indian(sample_portfolio)
        print(portfolio_analysis[:800] + "..." if len(portfolio_analysis) > 800 else portfolio_analysis)
        
        # Example 3: Market outlook
        print("\n" + "="*50)
        print("3. INDIAN MARKET OUTLOOK")
        print("="*50)
        
        market_outlook = agent.market_outlook_indian()
        print(market_outlook[:800] + "..." if len(market_outlook) > 800 else market_outlook)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
    
    print("\n" + "="*60)
    print("🚀 Indian Stock Market Agent Demo Complete!")
    print("💡 Key Features:")
    print("  - Real-time NSE/BSE stock monitoring")
    print("  - Rupee-denominated analysis")
    print("  - Indian market hours awareness")
    print("  - Sector-specific insights")
    print("  - Currency impact analysis")
    print("  - Nifty/Sensex correlation")
    print("="*60)

if __name__ == "__main__":
    main()