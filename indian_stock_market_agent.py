#!/usr/bin/env python3
"""
AI-Powered Indian Stock Market Agent (Complete Version)

Features specifically for Indian markets:
- NSE/BSE stock analysis with real-time monitoring
- Indian market hours (9:15 AM - 3:30 PM IST)
- Rupee-denominated analysis with currency impact
- Indian sector analysis (IT, Banking, Pharma, Auto, Infrastructure)
- Integration with Indian financial data sources
- Support for Indian market holidays and timings
- Nifty 50, Sensex, and sectoral indices tracking
- FII/DII flow analysis and impact assessment
- Currency impact analysis (USD/INR) on different sectors
- Real-time news sentiment analysis for Indian stocks
- Portfolio optimization for Indian market conditions
- Risk assessment tailored for Indian regulatory environment

Setup:
pip install yfinance pandas numpy anthropic requests beautifulsoup4 pytz schedule plotly dash
export ANTHROPIC_API_KEY="your-key-here"
export ALPHA_VANTAGE_KEY="your-alpha-vantage-key" (optional)
export EMAIL_PASSWORD="your-app-password" (optional)

Usage:
python indian_stock_market_agent.py
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
import schedule
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import warnings
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IndianMarketAlert:
    """Alert structure for Indian markets."""
    symbol: str
    alert_type: str  # 'price_change', 'volume_spike', 'news_sentiment', 'technical_breakout'
    message: str
    severity: str    # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    price_inr: float
    sector: str
    market_cap_cr: float  # Market cap in crores
    data: Dict[str, Any]
    processed: bool = False

@dataclass
class PortfolioPosition:
    """Structure for portfolio positions."""
    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    day_change: float
    day_change_pct: float
    sector: str
    weight_pct: float

@dataclass
class MarketSentiment:
    """Market sentiment data structure."""
    overall_score: float  # -1 to +1
    confidence: float     # 0 to 1
    key_drivers: List[str]
    sector_sentiments: Dict[str, float]
    timestamp: datetime

class IndianStockMarketAgent:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the comprehensive Indian stock market agent."""
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
        )
        
        # Indian market specific data
        self.indian_timezone = pytz.timezone('Asia/Kolkata')
        self.live_prices = {}
        self.price_history = {}
        self.news_cache = {}
        self.alerts = []
        self.portfolio_positions = {}
        self.market_sentiment = None
        self.analysis_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
        # Market data
        self.nifty_data = {}
        self.sensex_data = {}
        self.sectoral_indices = {}
        self.fii_dii_data = {}
        self.usd_inr_rate = 83.0  # Default, will be updated
        
        # Database for storing historical data
        self.db_path = Path("indian_market_data.db")
        self.init_database()
        
        # Alert thresholds
        self.alert_thresholds = {
            'price_change_pct': 3.0,      # Alert if stock moves >3%
            'volume_spike': 2.0,          # Alert if volume >2x average
            'news_sentiment_score': -0.3,  # Alert if sentiment very negative
            'technical_breakout': 0.8      # Alert for technical pattern confidence
        }
        
        # Comprehensive Indian stock symbols mapping
        self.popular_indian_stocks = {
            # Large Cap IT
            'TCS': 'TCS.NS', 'INFY': 'INFY.NS', 'WIPRO': 'WIPRO.NS', 
            'TECHM': 'TECHM.NS', 'HCLTECH': 'HCLTECH.NS', 'LTI': 'LTI.NS',
            'MINDTREE': 'MINDTREE.NS', 'MPHASIS': 'MPHASIS.NS',
            
            # Banking & Financial Services
            'HDFCBANK': 'HDFCBANK.NS', 'ICICIBANK': 'ICICIBANK.NS', 
            'SBIN': 'SBIN.NS', 'KOTAKBANK': 'KOTAKBANK.NS', 'AXISBANK': 'AXISBANK.NS',
            'BAJFINANCE': 'BAJFINANCE.NS', 'HDFCLIFE': 'HDFCLIFE.NS',
            'SBILIFE': 'SBILIFE.NS', 'ICICIGI': 'ICICIGI.NS',
            
            # Consumer Goods & Retail
            'RELIANCE': 'RELIANCE.NS', 'HINDUNILVR': 'HINDUNILVR.NS', 
            'ITC': 'ITC.NS', 'NESTLEIND': 'NESTLEIND.NS', 'BRITANNIA': 'BRITANNIA.NS',
            'DABUR': 'DABUR.NS', 'MARICO': 'MARICO.NS', 'GODREJCP': 'GODREJCP.NS',
            
            # Automotive
            'MARUTI': 'MARUTI.NS', 'TATAMOTORS': 'TATAMOTORS.NS', 
            'BAJAJ-AUTO': 'BAJAJ-AUTO.NS', 'M&M': 'M&M.NS', 
            'HEROMOTOCO': 'HEROMOTOCO.NS', 'EICHERMOT': 'EICHERMOT.NS',
            
            # Pharmaceuticals
            'SUNPHARMA': 'SUNPHARMA.NS', 'DRREDDY': 'DRREDDY.NS', 
            'CIPLA': 'CIPLA.NS', 'BIOCON': 'BIOCON.NS', 'AUROPHARMA': 'AUROPHARMA.NS',
            'LUPIN': 'LUPIN.NS', 'CADILAHC': 'CADILAHC.NS', 'TORNTPHARM': 'TORNTPHARM.NS',
            
            # Oil & Gas
            'ONGC': 'ONGC.NS', 'IOC': 'IOC.NS', 'BPCL': 'BPCL.NS', 
            'HINDPETRO': 'HINDPETRO.NS', 'GAIL': 'GAIL.NS',
            
            # Metals & Mining
            'TATASTEEL': 'TATASTEEL.NS', 'HINDALCO': 'HINDALCO.NS', 
            'JSWSTEEL': 'JSWSTEEL.NS', 'SAIL': 'SAIL.NS', 'VEDL': 'VEDL.NS',
            'COALINDIA': 'COALINDIA.NS', 'NMDC': 'NMDC.NS',
            
            # Infrastructure & Construction
            'LT': 'LT.NS', 'ADANIPORTS': 'ADANIPORTS.NS', 'POWERGRID': 'POWERGRID.NS',
            'NTPC': 'NTPC.NS', 'BHARTIARTL': 'BHARTIARTL.NS', 'IDEA': 'IDEA.NS',
            
            # Consumer Durables
            'ASIANPAINT': 'ASIANPAINT.NS', 'BERGER': 'BERGER.NS', 'PIDILITIND': 'PIDILITIND.NS',
            'TITAN': 'TITAN.NS', 'VOLTAS': 'VOLTAS.NS',
            
            # Cement
            'ULTRACEMCO': 'ULTRACEMCO.NS', 'SHREECEM': 'SHREECEM.NS', 
            'ACC': 'ACC.NS', 'AMBUJACEMENT': 'AMBUJACEMENT.NS',
            
            # FMCG
            'COLPAL': 'COLPAL.NS', 'PGHH': 'PGHH.NS', 'EMAMILTD': 'EMAMILTD.NS'
        }
        
        # Enhanced Indian sectors mapping with sub-sectors
        self.indian_sectors = {
            'Large Cap IT': ['TCS', 'INFY', 'WIPRO', 'TECHM', 'HCLTECH'],
            'Mid Cap IT': ['LTI', 'MINDTREE', 'MPHASIS', 'COFORGE'],
            'Private Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK'],
            'Public Banking': ['SBIN', 'PNB', 'CANBK', 'IOB'],
            'NBFC': ['BAJFINANCE', 'BAJAJFINSV', 'M&MFIN', 'LICHSGFIN'],
            'Consumer Staples': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA'],
            'Consumer Discretionary': ['RELIANCE', 'TITAN', 'TRENT', 'AVENUE'],
            'Passenger Vehicles': ['MARUTI', 'TATAMOTORS', 'M&M', 'EICHERMOT'],
            'Two Wheelers': ['BAJAJ-AUTO', 'HEROMOTOCO', 'TVSMOTOR'],
            'Pharmaceuticals': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'BIOCON'],
            'Oil & Gas': ['RELIANCE', 'ONGC', 'IOC', 'BPCL'],
            'Metals & Mining': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL'],
            'Infrastructure': ['LT', 'ADANIPORTS', 'POWERGRID', 'NTPC'],
            'Telecom': ['BHARTIARTL', 'IDEA', 'RCOM'],
            'Cement': ['ULTRACEMCO', 'SHREECEM', 'ACC', 'AMBUJACEMENT'],
            'Paints': ['ASIANPAINT', 'BERGER', 'KANSAINER', 'AKZOINDIA']
        }
        
        # Market timing and holidays
        self.market_open_time = "09:15"
        self.market_close_time = "15:30"
        self.pre_market_time = "09:00"
        self.post_market_time = "16:00"
        
        # Indian market holidays (simplified - would need API for complete list)
        self.market_holidays = [
            "2024-01-26", "2024-03-08", "2024-03-25", "2024-04-11", 
            "2024-04-14", "2024-04-17", "2024-05-01", "2024-08-15",
            "2024-10-02", "2024-11-01", "2024-11-15", "2024-12-25"
        ]
        
        # WebSocket connections for real-time data
        self.websocket_connections = {}
        self.monitoring_symbols = set()
        self.is_monitoring = False
        self.monitoring_threads = []
        
        # Email configuration for alerts
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': os.getenv('SENDER_EMAIL'),
            'sender_password': os.getenv('EMAIL_PASSWORD')
        }
        
        logger.info("🇮🇳 Indian Stock Market Agent initialized!")
        logger.info(f"⏰ Market Hours: {self.market_open_time} - {self.market_close_time} IST")
    
    def init_database(self):
        """Initialize SQLite database for storing historical data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables for storing data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    market_cap_cr REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price_inr REAL,
                    sector TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    analysis_result TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    expiry_time DATETIME NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
    
    def is_market_open(self) -> bool:
        """Check if Indian stock market is currently open."""
        now = datetime.now(self.indian_timezone)
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M")
        current_day = now.weekday()  # 0=Monday, 6=Sunday
        
        # Check if it's a holiday
        if current_date in self.market_holidays:
            return False
        
        # Market closed on weekends
        if current_day >= 5:  # Saturday or Sunday
            return False
        
        # Check if current time is within market hours
        if self.market_open_time <= current_time <= self.market_close_time:
            return True
        
        return False
    
    def is_pre_market(self) -> bool:
        """Check if it's pre-market hours."""
        now = datetime.now(self.indian_timezone)
        current_time = now.strftime("%H:%M")
        current_day = now.weekday()
        
        if current_day >= 5:  # Weekend
            return False
        
        return self.pre_market_time <= current_time < self.market_open_time
    
    def is_post_market(self) -> bool:
        """Check if it's post-market hours."""
        now = datetime.now(self.indian_timezone)
        current_time = now.strftime("%H:%M")
        current_day = now.weekday()
        
        if current_day >= 5:  # Weekend
            return False
        
        return self.market_close_time < current_time <= self.post_market_time
    
    def get_indian_stock_data(self, symbols: List[str], period: str = "1d") -> Dict:
        """Fetch comprehensive Indian stock data with enhanced error handling."""
        data = {}
        
        # Use ThreadPoolExecutor for parallel data fetching
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self._fetch_single_stock_data, symbol, period): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    stock_data = future.result()
                    if stock_data:
                        data[symbol] = stock_data
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    data[symbol] = {
                        'error': str(e),
                        'symbol': symbol,
                        'timestamp': datetime.now(self.indian_timezone)
                    }
        
        return data
    
    def _fetch_single_stock_data(self, symbol: str, period: str = "1d") -> Optional[Dict]:
        """Fetch data for a single stock with comprehensive metrics."""
        try:
            # Convert to NSE symbol if needed
            yf_symbol = self.popular_indian_stocks.get(symbol, f"{symbol}.NS")
            
            ticker = yf.Ticker(yf_symbol)
            
            # Get different types of data based on market status
            if self.is_market_open():
                hist = ticker.history(period="1d", interval="1m")
            else:
                hist = ticker.history(period=period, interval="1d")
            
            info = ticker.info
            
            if hist.empty:
                logger.warning(f"No historical data available for {symbol}")
                return None
            
            # Calculate basic metrics
            current_price = hist['Close'].iloc[-1]
            prev_close = info.get('previousClose', hist['Close'].iloc[0] if len(hist) > 1 else current_price)
            day_change = current_price - prev_close
            day_change_pct = (day_change / prev_close * 100) if prev_close > 0 else 0
            
            # Volume analysis
            current_volume = hist['Volume'].iloc[-1] if len(hist) > 0 else 0
            avg_volume = hist['Volume'].mean() if len(hist) > 1 else current_volume
            volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1
            
            # Price analysis
            day_high = hist['High'].max()
            day_low = hist['Low'].min()
            price_range_pct = ((day_high - day_low) / day_low * 100) if day_low > 0 else 0
            
            # Market cap calculations
            market_cap_usd = info.get('marketCap', 0)
            market_cap_inr = market_cap_usd * self.usd_inr_rate if market_cap_usd else 0
            market_cap_cr = market_cap_inr / 10000000  # Convert to crores
            
            # Technical indicators
            close_prices = hist['Close']
            if len(close_prices) >= 20:
                sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
                price_vs_sma20 = ((current_price - sma_20) / sma_20 * 100) if sma_20 > 0 else 0
            else:
                sma_20 = current_price
                price_vs_sma20 = 0
            
            if len(close_prices) >= 50:
                sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
                price_vs_sma50 = ((current_price - sma_50) / sma_50 * 100) if sma_50 > 0 else 0
            else:
                sma_50 = current_price
                price_vs_sma50 = 0
            
            # Volatility calculation
            if len(close_prices) >= 2:
                returns = close_prices.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            else:
                volatility = 0
            
            # Sector and fundamentals
            sector = self.get_indian_sector(symbol)
            
            return {
                # Basic price data
                'price': current_price,
                'prev_close': prev_close,
                'day_change': day_change,
                'day_change_pct': day_change_pct,
                'high': day_high,
                'low': day_low,
                'price_range_pct': price_range_pct,
                
                # Volume data
                'volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                
                # Market cap and valuation
                'market_cap_cr': market_cap_cr,
                'market_cap_usd': market_cap_usd,
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                
                # Technical indicators
                'sma_20': sma_20,
                'sma_50': sma_50,
                'price_vs_sma20': price_vs_sma20,
                'price_vs_sma50': price_vs_sma50,
                'volatility': volatility,
                
                # Fundamental data
                'beta': info.get('beta'),
                'eps': info.get('trailingEps'),
                'book_value': info.get('bookValue'),
                'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                'profit_margins': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                'debt_to_equity': info.get('debtToEquity'),
                'return_on_equity': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                
                # Metadata
                'sector': sector,
                'industry': info.get('industry'),
                'yf_symbol': yf_symbol,
                'timestamp': datetime.now(self.indian_timezone),
                'business_summary': info.get('longBusinessSummary', '')[:500]  # First 500 chars
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_indian_sector(self, symbol: str) -> str:
        """Get Indian sector for a given stock symbol with enhanced mapping."""
        for sector, stocks in self.indian_sectors.items():
            if symbol in stocks:
                return sector
        return 'Other'
    
    def get_nifty_sensex_data(self) -> Dict:
        """Fetch comprehensive Nifty 50 and Sensex data."""
        indices_data = {}
        
        indices_symbols = {
            'NIFTY': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY_BANK': '^NSEBANK',
            'NIFTY_IT': '^CNXIT',
            'NIFTY_PHARMA': '^CNXPHARMA',
            'NIFTY_AUTO': '^CNXAUTO'
        }
        
        for index_name, symbol in indices_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                if self.is_market_open():
                    hist = ticker.history(period="1d", interval="1m")
                else:
                    hist = ticker.history(period="5d", interval="1d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    open_price = hist['Open'].iloc[0]
                    change = current_price - open_price
                    change_pct = (change / open_price * 100) if open_price > 0 else 0
                    
                    indices_data[index_name] = {
                        'price': current_price,
                        'open': open_price,
                        'change': change,
                        'change_pct': change_pct,
                        'high': hist['High'].max(),
                        'low': hist['Low'].min(),
                        'volume': hist['Volume'].sum() if 'Volume' in hist.columns else 0,
                        'timestamp': datetime.now(self.indian_timezone)
                    }
                    
            except Exception as e:
                logger.error(f"Error fetching {index_name} data: {e}")
        
        return indices_data
    
    def get_usd_inr_rate(self) -> float:
        """Fetch current USD/INR exchange rate with fallback."""
        try:
            # Try multiple sources for USD/INR rate
            sources = ["USDINR=X", "INR=X"]
            
            for source in sources:
                try:
                    usd_inr = yf.Ticker(source)
                    hist = usd_inr.history(period="1d")
                    if not hist.empty:
                        rate = hist['Close'].iloc[-1]
                        if 70 <= rate <= 90:  # Sanity check for reasonable USD/INR rate
                            self.usd_inr_rate = rate
                            return rate
                except:
                    continue
            
            # Fallback to API if yfinance fails
            try:
                api_url = "https://api.exchangerate-api.com/v4/latest/USD"
                response = requests.get(api_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    inr_rate = data.get('rates', {}).get('INR')
                    if inr_rate and 70 <= inr_rate <= 90:
                        self.usd_inr_rate = inr_rate
                        return inr_rate
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error fetching USD/INR rate: {e}")
        
        # Return cached rate if all else fails
        return self.usd_inr_rate
    
    def _call_claude_indian(self, prompt: str, max_tokens: int = 3000, temperature: float = 0.3) -> str:
        """Enhanced Claude API call with Indian market context and caching."""
        
        # Create cache key
        cache_key = hash(prompt[:200])  # Use first 200 chars for cache key
        current_time = time.time()
        
        # Check cache first
        if cache_key in self.analysis_cache:
            cached_result, cached_time = self.analysis_cache[cache_key]
            if current_time - cached_time < self.cache_expiry:
                logger.info("📋 Using cached analysis...")
                return cached_result
        
        # Enhanced Indian market context
        market_status = self.get_market_status_context()
        indian_context = f"""
        COMPREHENSIVE INDIAN MARKET CONTEXT:
        
        MARKET STATUS & TIMING:
        - Current time: {datetime.now(self.indian_timezone).strftime('%Y-%m-%d %H:%M:%S IST')}
        - Market status: {market_status}
        - Market hours: 9:15 AM - 3:30 PM IST (Monday to Friday)
        
        CURRENCY & ECONOMICS:
        - All prices in Indian Rupees (₹)
        - Current USD/INR rate: ₹{self.usd_inr_rate:.2f}
        - Currency impact on IT/Pharma exports and Oil imports
        
        REGULATORY ENVIRONMENT:
        - SEBI regulations and compliance requirements
        - Indian tax implications (LTCG, STCG, STT)
        - FII/DII investment limits and flows
        
        MARKET DYNAMICS:
        - Retail investor behavior patterns in India
        - Institutional investment flows (FII/DII)
        - Government policy impact on sectors
        - Monsoon and seasonal factors
        
        {prompt}
        
        IMPORTANT: Provide analysis specifically relevant to Indian investors considering:
        - Indian market regulations and tax implications
        - Currency hedging for international exposure
        - Sectoral policy impacts and government initiatives
        - Regional economic factors and business cycles
        - Comparison with Indian benchmarks (Nifty, Sensex)
        - Suitability for different investor categories (retail, HNI, institutional)
        """
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": indian_context}]
            )
            
            result = message.content[0].text
            
            # Cache the result
            self.analysis_cache[cache_key] = (result, current_time)
            
            # Store in database for persistence
            self._store_analysis_in_db(cache_key, result, current_time)
            
            return result
            
        except anthropic.AuthenticationError:
            return "❌ Error: Invalid API key. Please check your ANTHROPIC_API_KEY."
        except anthropic.RateLimitError:
            return "⏱️ Rate limit exceeded. Consider upgrading your plan for higher limits."
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"❌ Analysis error: {str(e)}"
    
    def get_market_status_context(self) -> str:
        """Get detailed market status context."""
        if self.is_market_open():
            return "OPEN (Live trading session)"
        elif self.is_pre_market():
            return "PRE-MARKET (Orders being placed)"
        elif self.is_post_market():
            return "POST-MARKET (After hours)"
        else:
            return "CLOSED (Weekend or holiday)"
    
    def _store_analysis_in_db(self, cache_key: str, result: str, timestamp: float):
        """Store analysis result in database for persistence."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            expiry_time = datetime.fromtimestamp(timestamp + self.cache_expiry)
            
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_cache 
                (cache_key, analysis_result, timestamp, expiry_time)
                VALUES