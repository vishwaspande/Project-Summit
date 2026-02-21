#!/usr/bin/env python3
"""
Tool definitions for the Indian Stock Market AI Agent

This module contains all the callable tools that the AI agent can use
to interact with market data, analyze stocks, and perform various operations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
import sqlite3
from memory import MemoryManager
from notifications import NotificationManager
from config import AgentConfig

logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """Standardized result format for all tools."""
    success: bool
    data: Any = None
    error: str = ""
    timestamp: datetime = None
    source: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))

class StockMarketTools:
    """Collection of tools for the Indian Stock Market AI Agent."""

    def __init__(self, config: AgentConfig, memory_manager: MemoryManager,
                 notification_manager: NotificationManager):
        self.config = config
        self.memory = memory_manager
        self.notifications = notification_manager
        self.indian_timezone = pytz.timezone('Asia/Kolkata')
        self.usd_inr_rate = 83.0  # Will be updated dynamically

        # Indian stock symbols mapping
        self.popular_stocks = {
            # Large Cap IT
            'TCS': 'TCS.NS', 'INFY': 'INFY.NS', 'WIPRO': 'WIPRO.NS',
            'TECHM': 'TECHM.NS', 'HCLTECH': 'HCLTECH.NS', 'LTI': 'LTI.NS',

            # Banking & Financial Services
            'HDFCBANK': 'HDFCBANK.NS', 'ICICIBANK': 'ICICIBANK.NS',
            'SBIN': 'SBIN.NS', 'KOTAKBANK': 'KOTAKBANK.NS', 'AXISBANK': 'AXISBANK.NS',
            'BAJFINANCE': 'BAJFINANCE.NS', 'BAJAJFINSV': 'BAJAJFINSV.NS',

            # Consumer Goods & Retail
            'RELIANCE': 'RELIANCE.NS', 'HINDUNILVR': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS', 'NESTLEIND': 'NESTLEIND.NS', 'BRITANNIA': 'BRITANNIA.NS',

            # Automotive
            'MARUTI': 'MARUTI.NS', 'TATAMOTORS': 'TATAMOTORS.NS',
            'BAJAJ-AUTO': 'BAJAJ-AUTO.NS', 'M&M': 'M&M.NS',
            'HEROMOTOCO': 'HEROMOTOCO.NS',

            # Others
            'BHARTIARTL': 'BHARTIARTL.NS', 'ASIANPAINT': 'ASIANPAINT.NS',
            'LT': 'LT.NS', 'POWERGRID': 'POWERGRID.NS', 'NTPC': 'NTPC.NS'
        }

    def get_stock_price(self, symbol: str) -> ToolResult:
        """
        Fetch live price for any NSE/BSE stock.

        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE', 'TCS')

        Returns:
            ToolResult: Contains current price, change, and other data
        """
        try:
            # Convert symbol to Yahoo Finance format
            yf_symbol = self.popular_stocks.get(symbol, f"{symbol}.NS")
            ticker = yf.Ticker(yf_symbol)

            # Get current data
            info = ticker.info
            hist = ticker.history(period="1d", interval="5m")

            if hist.empty:
                hist = ticker.history(period="1d")

            if hist.empty:
                return ToolResult(
                    success=False,
                    error=f"No data available for {symbol}",
                    source="yfinance"
                )

            current_price = hist['Close'].iloc[-1]
            prev_close = info.get('previousClose', hist['Open'].iloc[0])
            change = current_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close > 0 else 0

            data = {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'previous_close': round(prev_close, 2),
                'change': round(change, 2),
                'change_percent': round(change_pct, 2),
                'day_high': round(hist['High'].max(), 2),
                'day_low': round(hist['Low'].min(), 2),
                'volume': int(hist['Volume'].iloc[-1]) if len(hist) > 0 else 0,
                'market_cap': info.get('marketCap'),
                'currency': 'INR'
            }

            return ToolResult(
                success=True,
                data=data,
                source="yfinance"
            )

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                source="yfinance"
            )

    def get_multiple_stocks(self, symbols: List[str]) -> ToolResult:
        """
        Fetch prices for multiple stocks at once.

        Args:
            symbols (List[str]): List of stock symbols

        Returns:
            ToolResult: Contains data for all requested stocks
        """
        try:
            results = {}
            errors = []

            for symbol in symbols:
                result = self.get_stock_price(symbol)
                if result.success:
                    results[symbol] = result.data
                else:
                    errors.append(f"{symbol}: {result.error}")

            return ToolResult(
                success=len(results) > 0,
                data=results,
                error="; ".join(errors) if errors else "",
                source="yfinance"
            )

        except Exception as e:
            logger.error(f"Error fetching multiple stocks: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                source="yfinance"
            )

    def get_stock_fundamentals(self, symbol: str) -> ToolResult:
        """
        Get fundamental data like P/E, market cap, 52-week high/low, volume.

        Args:
            symbol (str): Stock symbol

        Returns:
            ToolResult: Contains fundamental metrics
        """
        try:
            yf_symbol = self.popular_stocks.get(symbol, f"{symbol}.NS")
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info

            # Convert market cap from USD to INR
            market_cap_usd = info.get('marketCap', 0)
            market_cap_inr = market_cap_usd * self.usd_inr_rate if market_cap_usd else 0

            data = {
                'symbol': symbol,
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'market_cap_inr': market_cap_inr,
                'market_cap_crores': market_cap_inr / 10000000 if market_cap_inr else 0,
                'week_52_high': info.get('fiftyTwoWeekHigh'),
                'week_52_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'book_value': info.get('bookValue'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'return_on_equity': info.get('returnOnEquity'),
                'profit_margins': info.get('profitMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'currency': 'INR'
            }

            return ToolResult(
                success=True,
                data=data,
                source="yfinance"
            )

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                source="yfinance"
            )

    def get_technical_indicators(self, symbol: str, period: str = "6mo") -> ToolResult:
        """
        Calculate technical indicators: SMA 20/50/200, RSI, MACD.

        Args:
            symbol (str): Stock symbol
            period (str): Time period for data

        Returns:
            ToolResult: Contains technical indicators
        """
        try:
            yf_symbol = self.popular_stocks.get(symbol, f"{symbol}.NS")
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period=period)

            if len(hist) < 20:
                return ToolResult(
                    success=False,
                    error=f"Insufficient data for technical analysis of {symbol}",
                    source="yfinance"
                )

            # Calculate Simple Moving Averages
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()

            # Calculate RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))

            # Calculate MACD
            exp1 = hist['Close'].ewm(span=12).mean()
            exp2 = hist['Close'].ewm(span=26).mean()
            hist['MACD'] = exp1 - exp2
            hist['MACD_Signal'] = hist['MACD'].ewm(span=9).mean()
            hist['MACD_Histogram'] = hist['MACD'] - hist['MACD_Signal']

            # Get latest values
            latest = hist.iloc[-1]
            current_price = latest['Close']

            data = {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'sma_20': round(latest['SMA_20'], 2) if not pd.isna(latest['SMA_20']) else None,
                'sma_50': round(latest['SMA_50'], 2) if not pd.isna(latest['SMA_50']) else None,
                'sma_200': round(latest['SMA_200'], 2) if not pd.isna(latest['SMA_200']) else None,
                'rsi': round(latest['RSI'], 2) if not pd.isna(latest['RSI']) else None,
                'macd': round(latest['MACD'], 4) if not pd.isna(latest['MACD']) else None,
                'macd_signal': round(latest['MACD_Signal'], 4) if not pd.isna(latest['MACD_Signal']) else None,
                'macd_histogram': round(latest['MACD_Histogram'], 4) if not pd.isna(latest['MACD_Histogram']) else None,

                # Price relative to moving averages
                'price_vs_sma20': round((current_price - latest['SMA_20']) / latest['SMA_20'] * 100, 2) if not pd.isna(latest['SMA_20']) else None,
                'price_vs_sma50': round((current_price - latest['SMA_50']) / latest['SMA_50'] * 100, 2) if not pd.isna(latest['SMA_50']) else None,
                'price_vs_sma200': round((current_price - latest['SMA_200']) / latest['SMA_200'] * 100, 2) if not pd.isna(latest['SMA_200']) else None,

                # Technical signals
                'rsi_signal': 'Oversold' if latest['RSI'] < 30 else 'Overbought' if latest['RSI'] > 70 else 'Neutral',
                'macd_signal_trend': 'Bullish' if latest['MACD'] > latest['MACD_Signal'] else 'Bearish',

                'currency': 'INR'
            }

            return ToolResult(
                success=True,
                data=data,
                source="yfinance"
            )

        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                source="yfinance"
            )

    def get_market_news(self, query: str = "Indian stock market", limit: int = 10) -> ToolResult:
        """
        Fetch latest market news using NewsAPI or similar service.

        Args:
            query (str): Search query for news
            limit (int): Number of articles to fetch

        Returns:
            ToolResult: Contains news articles
        """
        try:
            # Note: For demo purposes, returning mock news data
            # In production, integrate with NewsAPI, Google News, or other news APIs

            mock_news = [
                {
                    'title': 'Nifty 50 hits record high as IT stocks surge',
                    'description': 'Indian benchmark indices reached new peaks today led by strong performance in IT sector stocks.',
                    'url': 'https://example.com/news1',
                    'published_at': datetime.now(self.indian_timezone).isoformat(),
                    'source': 'Economic Times',
                    'sentiment': 'positive'
                },
                {
                    'title': 'RBI monetary policy meeting outcomes',
                    'description': 'Reserve Bank of India announces key policy decisions affecting interest rates and liquidity.',
                    'url': 'https://example.com/news2',
                    'published_at': (datetime.now(self.indian_timezone) - timedelta(hours=2)).isoformat(),
                    'source': 'Business Standard',
                    'sentiment': 'neutral'
                },
                {
                    'title': 'FII selling continues in Indian markets',
                    'description': 'Foreign institutional investors continue net selling streak in Indian equity markets.',
                    'url': 'https://example.com/news3',
                    'published_at': (datetime.now(self.indian_timezone) - timedelta(hours=4)).isoformat(),
                    'source': 'Mint',
                    'sentiment': 'negative'
                }
            ]

            # Filter by limit
            news_data = mock_news[:limit]

            return ToolResult(
                success=True,
                data=news_data,
                source="mock_news_api"
            )

        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                source="news_api"
            )

    def get_fii_dii_data(self) -> ToolResult:
        """
        Get FII/DII flow data for Indian markets.

        Returns:
            ToolResult: Contains FII/DII flow information
        """
        try:
            # Note: This would typically fetch from NSE/BSE APIs or financial data providers
            # For demo purposes, returning mock data

            current_date = datetime.now(self.indian_timezone).strftime('%Y-%m-%d')

            mock_data = {
                'date': current_date,
                'fii_equity_flow': -1250.5,  # Crores INR (negative = selling)
                'fii_debt_flow': 450.2,      # Crores INR
                'dii_flow': 2100.8,          # Crores INR (positive = buying)
                'total_fii_flow': -800.3,    # Combined FII flow
                'net_flow': 1300.5,          # Net of FII + DII

                # Historical trends (last 5 days)
                'fii_5day_trend': [-1250.5, -890.2, -1100.8, -750.3, -980.1],
                'dii_5day_trend': [2100.8, 1800.5, 1950.2, 1650.8, 1875.3],

                # Year-to-date flows
                'fii_ytd_flow': -15230.5,
                'dii_ytd_flow': 18450.8,

                'currency': 'INR (Crores)',
                'source': 'NSE/BSE Data'
            }

            return ToolResult(
                success=True,
                data=mock_data,
                source="nse_bse_api"
            )

        except Exception as e:
            logger.error(f"Error fetching FII/DII data: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                source="fii_dii_api"
            )

    def get_usd_inr_rate(self) -> ToolResult:
        """
        Get current USD/INR exchange rate.

        Returns:
            ToolResult: Contains USD/INR rate and change
        """
        try:
            ticker = yf.Ticker("USDINR=X")
            hist = ticker.history(period="5d")

            if hist.empty:
                return ToolResult(
                    success=False,
                    error="No USD/INR data available",
                    source="yfinance"
                )

            current_rate = hist['Close'].iloc[-1]
            prev_rate = hist['Close'].iloc[-2] if len(hist) > 1 else current_rate
            change = current_rate - prev_rate
            change_pct = (change / prev_rate * 100) if prev_rate > 0 else 0

            # Update internal rate
            self.usd_inr_rate = current_rate

            data = {
                'current_rate': round(current_rate, 4),
                'previous_rate': round(prev_rate, 4),
                'change': round(change, 4),
                'change_percent': round(change_pct, 2),
                'day_high': round(hist['High'].max(), 4),
                'day_low': round(hist['Low'].min(), 4),
                'currency_pair': 'USD/INR'
            }

            return ToolResult(
                success=True,
                data=data,
                source="yfinance"
            )

        except Exception as e:
            logger.error(f"Error fetching USD/INR rate: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                source="yfinance"
            )

    def check_portfolio(self, portfolio: Dict[str, Dict[str, float]]) -> ToolResult:
        """
        Evaluate current portfolio holdings and P&L.

        Args:
            portfolio (Dict): Portfolio positions with qty and avg_price

        Returns:
            ToolResult: Contains portfolio analysis
        """
        try:
            symbols = list(portfolio.keys())
            current_prices_result = self.get_multiple_stocks(symbols)

            if not current_prices_result.success:
                return ToolResult(
                    success=False,
                    error="Could not fetch current prices for portfolio analysis",
                    source="portfolio_analysis"
                )

            current_prices = current_prices_result.data
            total_invested = 0
            total_current_value = 0
            positions = []

            for symbol, position in portfolio.items():
                if symbol in current_prices:
                    qty = position['qty']
                    avg_price = position['avg_price']
                    current_price = current_prices[symbol]['current_price']

                    invested_amount = qty * avg_price
                    current_value = qty * current_price
                    unrealized_pnl = current_value - invested_amount
                    pnl_percent = (unrealized_pnl / invested_amount * 100) if invested_amount > 0 else 0

                    total_invested += invested_amount
                    total_current_value += current_value

                    positions.append({
                        'symbol': symbol,
                        'quantity': qty,
                        'avg_cost': avg_price,
                        'current_price': current_price,
                        'invested_amount': round(invested_amount, 2),
                        'current_value': round(current_value, 2),
                        'unrealized_pnl': round(unrealized_pnl, 2),
                        'pnl_percent': round(pnl_percent, 2),
                        'day_change': current_prices[symbol]['change'],
                        'day_change_percent': current_prices[symbol]['change_percent']
                    })

            portfolio_pnl = total_current_value - total_invested
            portfolio_pnl_percent = (portfolio_pnl / total_invested * 100) if total_invested > 0 else 0

            data = {
                'total_invested': round(total_invested, 2),
                'current_value': round(total_current_value, 2),
                'total_pnl': round(portfolio_pnl, 2),
                'total_pnl_percent': round(portfolio_pnl_percent, 2),
                'positions': positions,
                'currency': 'INR'
            }

            return ToolResult(
                success=True,
                data=data,
                source="portfolio_analysis"
            )

        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                source="portfolio_analysis"
            )

    def send_alert(self, message: str, alert_type: str = "info") -> ToolResult:
        """
        Send notification to user via Telegram.

        Args:
            message (str): Alert message
            alert_type (str): Type of alert (info, warning, critical)

        Returns:
            ToolResult: Contains notification status
        """
        try:
            result = self.notifications.send_notification(message, alert_type)

            # Log the alert
            self.memory.store_alert({
                'message': message,
                'alert_type': alert_type,
                'timestamp': datetime.now(self.indian_timezone),
                'sent': result
            })

            return ToolResult(
                success=result,
                data={'message': message, 'type': alert_type},
                error="" if result else "Failed to send notification",
                source="telegram"
            )

        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                source="telegram"
            )

    def save_to_memory(self, analysis_type: str, data: Dict[str, Any], reasoning: str = "") -> ToolResult:
        """
        Store analysis results, decisions, alerts in database.

        Args:
            analysis_type (str): Type of analysis
            data (Dict): Analysis data
            reasoning (str): AI reasoning/explanation

        Returns:
            ToolResult: Contains storage status
        """
        try:
            result = self.memory.store_analysis(analysis_type, data, reasoning)

            return ToolResult(
                success=result,
                data={'analysis_type': analysis_type, 'stored': result},
                error="" if result else "Failed to store in database",
                source="sqlite"
            )

        except Exception as e:
            logger.error(f"Error saving to memory: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                source="sqlite"
            )

    def read_from_memory(self, analysis_type: str, days_back: int = 7) -> ToolResult:
        """
        Retrieve past analyses and decisions from database.

        Args:
            analysis_type (str): Type of analysis to retrieve
            days_back (int): Number of days to look back

        Returns:
            ToolResult: Contains retrieved data
        """
        try:
            data = self.memory.get_recent_analyses(analysis_type, days_back)

            return ToolResult(
                success=len(data) > 0,
                data=data,
                error="No data found" if len(data) == 0 else "",
                source="sqlite"
            )

        except Exception as e:
            logger.error(f"Error reading from memory: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                source="sqlite"
            )

    # Tool registry for the AI agent
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for the Anthropic API.

        Returns:
            List[Dict]: Tool definitions in Anthropic format
        """
        return [
            {
                "name": "get_stock_price",
                "description": "Fetch live price for any NSE/BSE stock using yfinance, append .NS for NSE",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol (e.g., RELIANCE, TCS)"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_multiple_stocks",
                "description": "Fetch prices for a watchlist of stocks at once",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stock symbols"
                        }
                    },
                    "required": ["symbols"]
                }
            },
            {
                "name": "get_stock_fundamentals",
                "description": "Get P/E, market cap, 52-week high/low, volume and other fundamental data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_technical_indicators",
                "description": "Calculate moving averages (SMA 20, 50, 200), RSI, MACD for technical analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "period": {"type": "string", "description": "Time period (default: 6mo)", "enum": ["1mo", "3mo", "6mo", "1y", "2y"]}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_market_news",
                "description": "Fetch latest market news using news APIs",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for news"},
                        "limit": {"type": "integer", "description": "Number of articles to fetch"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_fii_dii_data",
                "description": "Get FII/DII flow data for Indian markets",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_usd_inr_rate",
                "description": "Get current USD/INR exchange rate",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "check_portfolio",
                "description": "Evaluate current portfolio holdings and P&L",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "portfolio": {
                            "type": "object",
                            "description": "Portfolio with symbol as key and qty/avg_price as values"
                        }
                    },
                    "required": ["portfolio"]
                }
            },
            {
                "name": "send_alert",
                "description": "Send notification to user via Telegram bot integration",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Alert message"},
                        "alert_type": {"type": "string", "description": "Alert type", "enum": ["info", "warning", "critical"]}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "save_to_memory",
                "description": "Store analysis results, decisions, alerts in SQLite database",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_type": {"type": "string", "description": "Type of analysis"},
                        "data": {"type": "object", "description": "Analysis data"},
                        "reasoning": {"type": "string", "description": "AI reasoning/explanation"}
                    },
                    "required": ["analysis_type", "data"]
                }
            },
            {
                "name": "read_from_memory",
                "description": "Retrieve past analyses and decisions from SQLite database",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_type": {"type": "string", "description": "Type of analysis to retrieve"},
                        "days_back": {"type": "integer", "description": "Number of days to look back"}
                    },
                    "required": ["analysis_type"]
                }
            }
        ]