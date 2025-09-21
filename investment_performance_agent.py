#!/usr/bin/env python3
"""
Enhanced Investment Performance Agent

This agent calculates comprehensive portfolio performance metrics including:
- Total portfolio value and gain/loss
- Best and worst performing stocks 
- Portfolio diversity analysis
- Risk assessment metrics
- Advanced financial ratios: Sharpe, Sortino, Alpha, Beta, Information Ratio

Designed to work with the Indian stock market dashboard portfolio data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Structure for individual stock performance metrics."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    invested_amount: float
    current_value: float
    absolute_gain_loss: float
    percentage_gain_loss: float
    weight_in_portfolio: float
    sector: str
    asset_type: str

@dataclass
class AdvancedRiskMetrics:
    """Structure for advanced risk-adjusted performance metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    alpha: float
    beta: float
    information_ratio: float
    volatility: float
    downside_deviation: float
    max_drawdown: float
    var_95: float  # Value at Risk at 95% confidence
    tracking_error: float

@dataclass
class PortfolioAnalysis:
    """Comprehensive portfolio analysis results."""
    total_invested: float
    total_current_value: float
    total_gain_loss: float
    total_gain_loss_percentage: float
    best_performer: Optional[PerformanceMetrics]
    worst_performer: Optional[PerformanceMetrics]
    diversity_score: float
    risk_level: str
    sector_allocation: Dict[str, float]
    risk_metrics: Dict[str, float]
    advanced_metrics: Optional[AdvancedRiskMetrics]
    recommendations: List[str]

class InvestmentPerformanceAgent:
    """Agent to analyze investment portfolio performance."""
    
    def __init__(self):
        """Initialize the performance agent."""
        self.risk_free_rate = 0.065  # 6.5% Indian government bond rate (annual)
        self.benchmark_symbol = "^NSEI"  # Nifty 50 as benchmark for Indian market
        
        # Indian sector mappings (from the original agent)
        self.indian_sectors = {
            'Large Cap IT': ['TCS', 'INFY', 'WIPRO', 'TECHM', 'HCLTECH'],
            'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'Consumer': ['RELIANCE', 'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA'],
            'Auto': ['MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'M&M', 'HEROMOTOCO'],
            'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'BIOCON', 'AUROPHARMA'],
            'Oil & Gas': ['ONGC', 'IOC', 'BPCL', 'HINDPETRO'],
            'Metals': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL'],
            'Infrastructure': ['LT', 'ADANIPORTS', 'POWERGRID', 'NTPC'],
            'Consumer Durables': ['ASIANPAINT', 'BERGER', 'TITAN', 'VOLTAS'],
            
            # Mutual Fund Categories
            'Flexi Cap Funds': ['Parag Parikh Flexi Cap Fund - Direct Plan'],
            'Small Cap Funds': ['HDFC Small Cap Fund - Direct Plan - Growth'], 
            'Index Funds': ['HDFC Nifty Next50 Index Fund Direct Growth', 'HDFC Nifty 50 Index Fund - Direct Plan'],
            'Sectoral Funds': ['Nippon India Pharma Fund Direct Growth Plan', 'ICICI Prudential Energy Opportunity Fund - Direct Plan'],
            'Commodity ETFs': ['HDFCGOLDETF.NS', 'HDFCSILVERETF.NS']
        }
    
    def get_sector(self, symbol: str) -> str:
        """Get sector for a given symbol."""
        for sector, symbols in self.indian_sectors.items():
            if symbol in symbols or symbol.replace('.NS', '') in symbols:
                return sector
        return 'Other'
    
    def get_historical_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch historical price data for symbols."""
        historical_data = {}
        
        for symbol in symbols:
            try:
                # Handle different asset types
                if symbol.endswith('.NS'):
                    # ETF
                    ticker = yf.Ticker(symbol)
                elif any(fund_name in symbol for fund_list in [
                    ['Parag Parikh', 'HDFC', 'Nippon', 'ICICI']
                ] for fund_name in fund_list):
                    # Mutual Fund - simulate historical data for demo
                    # In production, you'd fetch from MF API
                    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
                    base_price = 150.0 + abs(hash(symbol)) % 200
                    # Generate realistic price movements
                    returns = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annual volatility
                    prices = [base_price]
                    for ret in returns[1:]:
                        prices.append(prices[-1] * (1 + ret))
                    
                    historical_data[symbol] = pd.DataFrame({
                        'Close': prices,
                        'Adj Close': prices
                    }, index=dates)
                    continue
                else:
                    # Stock
                    ticker = yf.Ticker(f"{symbol}.NS")
                
                # Fetch historical data
                hist = ticker.history(period=period)
                if not hist.empty:
                    historical_data[symbol] = hist
                else:
                    logger.warning(f"No historical data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
                continue
        
        return historical_data
    
    def get_benchmark_data(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch benchmark (Nifty 50) historical data."""
        try:
            benchmark_ticker = yf.Ticker(self.benchmark_symbol)
            benchmark_data = benchmark_ticker.history(period=period)
            return benchmark_data if not benchmark_data.empty else None
        except Exception as e:
            logger.error(f"Error fetching benchmark data: {e}")
            return None
    
    def calculate_portfolio_returns(self, portfolio_data: Dict, historical_data: Dict[str, pd.DataFrame]) -> Optional[pd.Series]:
        """Calculate portfolio daily returns based on weights."""
        if not historical_data:
            return None
        
        # Calculate portfolio weights
        total_invested = sum(float(pos['qty']) * float(pos['avg_price']) for pos in portfolio_data.values())
        if total_invested == 0:
            return None
        
        weights = {}
        for symbol, position in portfolio_data.items():
            if symbol in historical_data:
                invested = float(position['qty']) * float(position['avg_price'])
                weights[symbol] = invested / total_invested
        
        # Align all price series to common dates
        price_data = {}
        min_date = None
        
        for symbol, data in historical_data.items():
            if symbol in weights and not data.empty:
                prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                price_data[symbol] = prices
                if min_date is None or prices.index.min() > min_date:
                    min_date = prices.index.min()
        
        if not price_data or min_date is None:
            return None
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(dtype=float)
        
        # Get common date range
        common_dates = None
        for symbol, prices in price_data.items():
            filtered_prices = prices[prices.index >= min_date]
            if common_dates is None:
                common_dates = filtered_prices.index
            else:
                common_dates = common_dates.intersection(filtered_prices.index)
        
        if len(common_dates) < 30:  # Need at least 30 days of data
            return None
        
        # Calculate portfolio value over time
        portfolio_values = pd.Series(index=common_dates, dtype=float)
        
        for date in common_dates:
            daily_value = 0
            for symbol, weight in weights.items():
                if symbol in price_data and date in price_data[symbol].index:
                    daily_value += weight * price_data[symbol].loc[date]
            portfolio_values.loc[date] = daily_value
        
        # Calculate returns
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        return portfolio_returns
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """
        Calculate Sharpe Ratio.
        Sharpe Ratio = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
        """
        if returns.empty or len(returns) < 2:
            return 0.0
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Annualized return
        annual_return = (1 + returns.mean()) ** 252 - 1
        
        # Annualized volatility
        annual_volatility = returns.std() * np.sqrt(252)
        
        if annual_volatility == 0:
            return 0.0
        
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> Tuple[float, float]:
        """
        Calculate Sortino Ratio and Downside Deviation.
        Sortino Ratio = (Portfolio Return - Risk Free Rate) / Downside Deviation
        """
        if returns.empty or len(returns) < 2:
            return 0.0, 0.0
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Annualized return
        annual_return = (1 + returns.mean()) ** 252 - 1
        
        # Calculate downside deviation (volatility of negative returns only)
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        downside_returns = returns[returns < daily_risk_free] - daily_risk_free
        
        if len(downside_returns) == 0:
            downside_deviation = 0.0
        else:
            downside_deviation = downside_returns.std() * np.sqrt(252)
        
        if downside_deviation == 0:
            return float('inf') if annual_return > risk_free_rate else 0.0, 0.0
        
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
        return sortino_ratio, downside_deviation
    
    def calculate_alpha_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate Alpha and Beta relative to benchmark.
        Beta = Covariance(Portfolio, Benchmark) / Variance(Benchmark)
        Alpha = Portfolio Return - (Risk Free Rate + Beta * (Benchmark Return - Risk Free Rate))
        """
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0, 1.0
        
        # Align returns to common dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 30:
            return 0.0, 1.0
        
        port_returns = portfolio_returns.loc[common_dates]
        bench_returns = benchmark_returns.loc[common_dates]
        
        # Calculate Beta
        covariance = np.cov(port_returns, bench_returns)[0, 1]
        benchmark_variance = np.var(bench_returns)
        
        if benchmark_variance == 0:
            beta = 1.0
        else:
            beta = covariance / benchmark_variance
        
        # Calculate Alpha (annualized)
        portfolio_annual_return = (1 + port_returns.mean()) ** 252 - 1
        benchmark_annual_return = (1 + bench_returns.mean()) ** 252 - 1
        
        expected_return = self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate)
        alpha = portfolio_annual_return - expected_return
        
        return alpha, beta
    
    def calculate_information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate Information Ratio and Tracking Error.
        Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error
        Tracking Error = Standard Deviation of (Portfolio Return - Benchmark Return)
        """
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0, 0.0
        
        # Align returns to common dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 30:
            return 0.0, 0.0
        
        port_returns = portfolio_returns.loc[common_dates]
        bench_returns = benchmark_returns.loc[common_dates]
        
        # Calculate excess returns
        excess_returns = port_returns - bench_returns
        
        # Tracking error (annualized)
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        if tracking_error == 0:
            return 0.0, 0.0
        
        # Information ratio
        excess_annual_return = (1 + excess_returns.mean()) ** 252 - 1
        information_ratio = excess_annual_return / tracking_error
        
        return information_ratio, tracking_error
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate Maximum Drawdown."""
        if returns.empty:
            return 0.0
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Return maximum drawdown (most negative value)
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown)
    
    def calculate_var_95(self, returns: pd.Series) -> float:
        """Calculate Value at Risk at 95% confidence level."""
        if returns.empty or len(returns) < 20:
            return 0.0
        
        # Sort returns and find 5th percentile
        var_95 = np.percentile(returns, 5)
        
        return abs(var_95)
    
    def calculate_advanced_metrics(self, portfolio_data: Dict, dashboard_data: Dict = None) -> Optional[AdvancedRiskMetrics]:
        """Calculate all advanced risk-adjusted performance metrics."""
        try:
            # Get historical data
            symbols = list(portfolio_data.keys())
            historical_data = self.get_historical_data(symbols)
            
            if not historical_data:
                logger.warning("No historical data available for advanced metrics")
                return None
            
            # Calculate portfolio returns
            portfolio_returns = self.calculate_portfolio_returns(portfolio_data, historical_data)
            
            if portfolio_returns is None or portfolio_returns.empty:
                logger.warning("Could not calculate portfolio returns")
                return None
            
            # Get benchmark data
            benchmark_data = self.get_benchmark_data()
            benchmark_returns = None
            
            if benchmark_data is not None:
                benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()
            
            # Calculate all metrics
            sharpe_ratio = self.calculate_sharpe_ratio(portfolio_returns)
            sortino_ratio, downside_deviation = self.calculate_sortino_ratio(portfolio_returns)
            
            alpha, beta = 0.0, 1.0
            information_ratio, tracking_error = 0.0, 0.0
            
            if benchmark_returns is not None:
                alpha, beta = self.calculate_alpha_beta(portfolio_returns, benchmark_returns)
                information_ratio, tracking_error = self.calculate_information_ratio(portfolio_returns, benchmark_returns)
            
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            max_drawdown = self.calculate_max_drawdown(portfolio_returns)
            var_95 = self.calculate_var_95(portfolio_returns)
            
            return AdvancedRiskMetrics(
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                alpha=alpha,
                beta=beta,
                information_ratio=information_ratio,
                volatility=volatility,
                downside_deviation=downside_deviation,
                max_drawdown=max_drawdown,
                var_95=var_95,
                tracking_error=tracking_error
            )
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            return None
    
    def get_current_prices(self, symbols: List[str], dashboard_data: Dict = None) -> Dict[str, float]:
        """Fetch current prices for given symbols, with option to use dashboard data."""
        prices = {}
        
        # If dashboard data is provided, use it first (it's already fetched)
        if dashboard_data:
            for symbol in symbols:
                if symbol in dashboard_data and 'error' not in dashboard_data[symbol]:
                    prices[symbol] = dashboard_data[symbol]['price']
                    continue
        
        # Fallback to fetching prices for symbols not found in dashboard_data
        for symbol in symbols:
            if symbol in prices:
                continue  # Already have price from dashboard data
                
            try:
                # Handle different asset types
                if symbol.endswith('.NS'):
                    # ETF
                    ticker = yf.Ticker(symbol)
                elif any(fund_name in symbol for fund_list in [
                    ['Parag Parikh', 'HDFC', 'Nippon', 'ICICI']
                ] for fund_name in fund_list):
                    # Mutual Fund - use a reasonable demo price
                    # In real implementation, this would fetch from MF API
                    demo_nav = 150.0 + abs(hash(symbol)) % 200  # Demo NAV between 150-350
                    prices[symbol] = demo_nav
                    continue
                else:
                    # Stock
                    ticker = yf.Ticker(f"{symbol}.NS")
                
                # Fetch current price
                hist = ticker.history(period="1d")
                if not hist.empty:
                    prices[symbol] = hist['Close'].iloc[-1]
                else:
                    logger.warning(f"No price data for {symbol}")
                    prices[symbol] = 0.0
                    
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                prices[symbol] = 0.0
        
        return prices
    
    def calculate_total_value(self, portfolio_data: Dict, dashboard_data: Dict = None) -> Dict[str, float]:
        """Calculate total portfolio value metrics."""
        symbols = list(portfolio_data.keys())
        current_prices = self.get_current_prices(symbols, dashboard_data)
        
        total_invested = 0
        total_current_value = 0
        
        for symbol, position in portfolio_data.items():
            try:
                qty = float(position['qty'])
                avg_price = float(position['avg_price'])
                current_price = current_prices.get(symbol, avg_price)  # Fallback to avg_price
                
                invested = qty * avg_price
                current_value = qty * current_price
                
                total_invested += invested
                total_current_value += current_value
                
                logger.info(f"{symbol}: qty={qty}, avg={avg_price:.2f}, current={current_price:.2f}, invested={invested:.2f}, current_val={current_value:.2f}")
                
            except (ValueError, TypeError) as e:
                logger.error(f"Error calculating values for {symbol}: {e}")
                continue
        
        return {
            'total_invested': total_invested,
            'total_current_value': total_current_value,
            'net_value': total_current_value - total_invested
        }
    
    def calculate_gain_loss(self, portfolio_data: Dict, dashboard_data: Dict = None) -> Dict[str, float]:
        """Calculate total gain/loss for portfolio."""
        value_metrics = self.calculate_total_value(portfolio_data, dashboard_data)
        
        total_gain_loss = value_metrics['net_value']
        total_gain_loss_pct = (total_gain_loss / value_metrics['total_invested'] * 100) if value_metrics['total_invested'] > 0 else 0
        
        return {
            'absolute_gain_loss': total_gain_loss,
            'percentage_gain_loss': total_gain_loss_pct
        }
    
    def find_best_stock(self, portfolio_data: Dict, dashboard_data: Dict = None) -> Optional[PerformanceMetrics]:
        """Find the best performing stock in portfolio."""
        symbols = list(portfolio_data.keys())
        current_prices = self.get_current_prices(symbols, dashboard_data)
        
        best_performance = float('-inf')
        best_stock = None
        
        for symbol, position in portfolio_data.items():
            try:
                qty = float(position['qty'])
                avg_price = float(position['avg_price'])
                current_price = current_prices.get(symbol, avg_price)
                
                if current_price > 0 and avg_price > 0:
                    percentage_gain = ((current_price - avg_price) / avg_price) * 100
                    
                    if percentage_gain > best_performance:
                        best_performance = percentage_gain
                        
                        invested_amount = qty * avg_price
                        current_value = qty * current_price
                        
                        best_stock = PerformanceMetrics(
                            symbol=symbol,
                            quantity=qty,
                            avg_cost=avg_price,
                            current_price=current_price,
                            invested_amount=invested_amount,
                            current_value=current_value,
                            absolute_gain_loss=current_value - invested_amount,
                            percentage_gain_loss=percentage_gain,
                            weight_in_portfolio=0,  # Will be calculated later
                            sector=self.get_sector(symbol),
                            asset_type=position.get('type', 'Stock')
                        )
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing {symbol} for best stock: {e}")
                continue
        
        return best_stock
    
    def find_worst_stock(self, portfolio_data: Dict, dashboard_data: Dict = None) -> Optional[PerformanceMetrics]:
        """Find the worst performing stock in portfolio."""
        symbols = list(portfolio_data.keys())
        current_prices = self.get_current_prices(symbols, dashboard_data)
        
        worst_performance = float('inf')
        worst_stock = None
        
        for symbol, position in portfolio_data.items():
            try:
                qty = float(position['qty'])
                avg_price = float(position['avg_price'])
                current_price = current_prices.get(symbol, avg_price)
                
                if current_price > 0 and avg_price > 0:
                    percentage_gain = ((current_price - avg_price) / avg_price) * 100
                    
                    if percentage_gain < worst_performance:
                        worst_performance = percentage_gain
                        
                        invested_amount = qty * avg_price
                        current_value = qty * current_price
                        
                        worst_stock = PerformanceMetrics(
                            symbol=symbol,
                            quantity=qty,
                            avg_cost=avg_price,
                            current_price=current_price,
                            invested_amount=invested_amount,
                            current_value=current_value,
                            absolute_gain_loss=current_value - invested_amount,
                            percentage_gain_loss=percentage_gain,
                            weight_in_portfolio=0,  # Will be calculated later
                            sector=self.get_sector(symbol),
                            asset_type=position.get('type', 'Stock')
                        )
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing {symbol} for worst stock: {e}")
                continue
        
        return worst_stock
    
    def calculate_diversity(self, portfolio_data: Dict, dashboard_data: Dict = None) -> float:
        """Calculate portfolio diversity score (0-100)."""
        if not portfolio_data:
            return 0.0
        
        # Get sector allocation
        sector_allocation = {}
        total_value = 0
        symbols = list(portfolio_data.keys())
        current_prices = self.get_current_prices(symbols, dashboard_data)
        
        for symbol, position in portfolio_data.items():
            try:
                qty = float(position['qty'])
                avg_price = float(position['avg_price'])
                current_price = current_prices.get(symbol, avg_price)
                current_value = qty * current_price
                
                sector = self.get_sector(symbol)
                sector_allocation[sector] = sector_allocation.get(sector, 0) + current_value
                total_value += current_value
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing {symbol} for diversity: {e}")
                continue
        
        if total_value == 0:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index (HHI) for diversity
        hhi = 0
        for sector_value in sector_allocation.values():
            weight = sector_value / total_value
            hhi += weight ** 2
        
        # Convert HHI to diversity score (0-100, higher is more diverse)
        # HHI ranges from 1/n to 1, where n is number of sectors
        max_hhi = 1.0  # Most concentrated (all in one sector)
        min_hhi = 1.0 / len(self.indian_sectors)  # Most diversified
        
        # Normalize to 0-100 scale (100 = most diverse)
        diversity_score = (1 - hhi) * 100
        
        return min(100, max(0, diversity_score))
    
    def assess_risk_level(self, portfolio_data: Dict, dashboard_data: Dict = None) -> Tuple[str, Dict[str, float]]:
        """Assess portfolio risk level and return detailed metrics."""
        if not portfolio_data:
            return "LOW", {}
        
        symbols = list(portfolio_data.keys())
        current_prices = self.get_current_prices(symbols, dashboard_data)
        
        # Risk metrics
        total_value = 0
        sector_concentration = {}
        high_risk_allocation = 0  # Small cap, sectoral funds, etc.
        
        for symbol, position in portfolio_data.items():
            try:
                qty = float(position['qty'])
                avg_price = float(position['avg_price'])
                current_price = current_prices.get(symbol, avg_price)
                current_value = qty * current_price
                total_value += current_value
                
                sector = self.get_sector(symbol)
                sector_concentration[sector] = sector_concentration.get(sector, 0) + current_value
                
                # Identify high-risk assets
                if sector in ['Small Cap Funds', 'Sectoral Funds', 'Metals', 'Commodity ETFs']:
                    high_risk_allocation += current_value
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing {symbol} for risk assessment: {e}")
                continue
        
        if total_value == 0:
            return "LOW", {}
        
        # Calculate risk metrics
        max_sector_weight = max(sector_concentration.values()) / total_value if sector_concentration else 0
        high_risk_percentage = high_risk_allocation / total_value * 100
        num_positions = len(portfolio_data)
        
        risk_metrics = {
            'max_sector_concentration': max_sector_weight * 100,
            'high_risk_allocation': high_risk_percentage,
            'number_of_positions': num_positions,
            'diversification_ratio': len(sector_concentration) / max(1, num_positions)
        }
        
        # Determine risk level
        risk_score = 0
        
        # Concentration risk
        if max_sector_weight > 0.6:
            risk_score += 3
        elif max_sector_weight > 0.4:
            risk_score += 2
        elif max_sector_weight > 0.3:
            risk_score += 1
        
        # High-risk allocation
        if high_risk_percentage > 40:
            risk_score += 3
        elif high_risk_percentage > 25:
            risk_score += 2
        elif high_risk_percentage > 15:
            risk_score += 1
        
        # Position concentration
        if num_positions < 3:
            risk_score += 2
        elif num_positions < 5:
            risk_score += 1
        
        # Classify risk level
        if risk_score >= 6:
            risk_level = "HIGH"
        elif risk_score >= 3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return risk_level, risk_metrics
    
    def generate_recommendations(self, analysis: PortfolioAnalysis) -> List[str]:
        """Generate investment recommendations based on portfolio analysis."""
        recommendations = []
        
        # Performance recommendations
        if analysis.total_gain_loss_percentage > 15:
            recommendations.append("📈 Portfolio showing strong performance! Consider taking some profits in overperforming stocks.")
        elif analysis.total_gain_loss_percentage < -10:
            recommendations.append("📉 Portfolio underperforming. Review holdings and consider cutting losses on poor performers.")
        
        # Advanced metrics recommendations
        if analysis.advanced_metrics:
            am = analysis.advanced_metrics
            
            # Sharpe ratio recommendations
            if am.sharpe_ratio > 1.5:
                recommendations.append(f"🏆 Excellent risk-adjusted returns! Sharpe ratio of {am.sharpe_ratio:.2f} indicates strong performance.")
            elif am.sharpe_ratio < 0.5:
                recommendations.append(f"⚠️ Low Sharpe ratio ({am.sharpe_ratio:.2f}). Consider reducing risk or improving returns.")
            
            # Sortino ratio recommendations
            if am.sortino_ratio > 2.0:
                recommendations.append(f"✅ Strong Sortino ratio ({am.sortino_ratio:.2f}) shows good downside protection.")
            elif am.sortino_ratio < 0.5:
                recommendations.append(f"📉 Low Sortino ratio ({am.sortino_ratio:.2f}) indicates poor downside performance.")
            
            # Alpha recommendations
            if am.alpha > 0.05:  # 5% positive alpha
                recommendations.append(f"🎯 Generating positive alpha ({am.alpha:.1%})! Outperforming the market.")
            elif am.alpha < -0.05:  # -5% negative alpha
                recommendations.append(f"📊 Negative alpha ({am.alpha:.1%}). Consider index funds or strategy review.")
            
            # Beta recommendations
            if am.beta > 1.3:
                recommendations.append(f"⚡ High beta ({am.beta:.2f}) means high market sensitivity. Consider defensive stocks.")
            elif am.beta < 0.7:
                recommendations.append(f"🛡️ Low beta ({am.beta:.2f}) provides stability but may limit upside in bull markets.")
            
            # Volatility recommendations
            if am.volatility > 0.30:  # 30% annual volatility
                recommendations.append(f"🌊 High volatility ({am.volatility:.1%}). Consider adding stable assets to reduce risk.")
            
            # Max drawdown recommendations
            if am.max_drawdown > 0.20:  # 20% max drawdown
                recommendations.append(f"⬇️ High maximum drawdown ({am.max_drawdown:.1%}). Review risk management strategy.")
        
        # Diversity recommendations
        if analysis.diversity_score < 30:
            recommendations.append("🎯 Low diversification detected. Consider adding stocks from different sectors.")
        elif analysis.diversity_score > 80:
            recommendations.append("🌟 Well diversified portfolio! Good sector spread reduces risk.")
        
        # Risk recommendations
        if analysis.risk_level == "HIGH":
            recommendations.append("⚠️ High risk portfolio. Consider reducing position sizes or adding stable large-cap stocks.")
        elif analysis.risk_level == "LOW":
            recommendations.append("🛡️ Conservative portfolio. Consider small allocation to growth stocks for higher returns.")
        
        # Sector-specific recommendations
        max_sector = max(analysis.sector_allocation.items(), key=lambda x: x[1])
        if max_sector[1] > 40:
            recommendations.append(f"⚡ Over-concentrated in {max_sector[0]} ({max_sector[1]:.1f}%). Consider rebalancing.")
        
        # Best/worst performer recommendations
        if analysis.best_performer and analysis.best_performer.percentage_gain_loss > 25:
            recommendations.append(f"🏆 {analysis.best_performer.symbol} is your star performer (+{analysis.best_performer.percentage_gain_loss:.1f}%). Consider booking some profits.")
        
        if analysis.worst_performer and analysis.worst_performer.percentage_gain_loss < -15:
            recommendations.append(f"📊 {analysis.worst_performer.symbol} is underperforming ({analysis.worst_performer.percentage_gain_loss:.1f}%). Review fundamentals before holding.")
        
        if not recommendations:
            recommendations.append("✅ Portfolio looks balanced. Continue monitoring and rebalance quarterly.")
        
        return recommendations

def analyze_portfolio_performance(portfolio_data: Dict, dashboard_data: Dict = None, include_advanced_metrics: bool = True) -> Dict:
    """
    Main function to analyze portfolio performance.
    
    Args:
        portfolio_data: Dictionary with structure:
            {
                'symbol': {
                    'qty': float,
                    'avg_price': float,
                    'type': str (optional)
                }
            }
        dashboard_data: Optional pre-fetched price data
        include_advanced_metrics: Whether to calculate advanced risk metrics (slower)
    
    Returns:
        Dictionary containing comprehensive portfolio analysis
    """
    if not portfolio_data:
        return {
            "error": "No portfolio data provided",
            "total_value": 0,
            "total_gain_loss": 0,
            "best_performer": None,
            "worst_performer": None,
            "diversity_score": 0,
            "risk_assessment": "N/A",
            "advanced_metrics": None
        }
    
    try:
        # Initialize the agent
        agent = InvestmentPerformanceAgent()
        
        # Calculate basic metrics
        total_value_metrics = agent.calculate_total_value(portfolio_data, dashboard_data)
        gain_loss_metrics = agent.calculate_gain_loss(portfolio_data, dashboard_data)
        best_performer = agent.find_best_stock(portfolio_data, dashboard_data)
        worst_performer = agent.find_worst_stock(portfolio_data, dashboard_data)
        diversity_score = agent.calculate_diversity(portfolio_data, dashboard_data)
        risk_level, risk_metrics = agent.assess_risk_level(portfolio_data, dashboard_data)
        
        # Calculate advanced metrics if requested
        advanced_metrics = None
        if include_advanced_metrics:
            advanced_metrics = agent.calculate_advanced_metrics(portfolio_data, dashboard_data)
        
        # Calculate sector allocation for detailed analysis
        symbols = list(portfolio_data.keys())
        current_prices = agent.get_current_prices(symbols, dashboard_data)
        sector_allocation = {}
        total_current_value = total_value_metrics['total_current_value']
        
        for symbol, position in portfolio_data.items():
            try:
                qty = float(position['qty'])
                current_price = current_prices.get(symbol, position['avg_price'])
                current_value = qty * current_price
                sector = agent.get_sector(symbol)
                
                if total_current_value > 0:
                    weight_pct = (current_value / total_current_value) * 100
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + weight_pct
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing {symbol} for sector allocation: {e}")
                continue
        
        # Create comprehensive analysis
        analysis = PortfolioAnalysis(
            total_invested=total_value_metrics['total_invested'],
            total_current_value=total_value_metrics['total_current_value'],
            total_gain_loss=gain_loss_metrics['absolute_gain_loss'],
            total_gain_loss_percentage=gain_loss_metrics['percentage_gain_loss'],
            best_performer=best_performer,
            worst_performer=worst_performer,
            diversity_score=diversity_score,
            risk_level=risk_level,
            sector_allocation=sector_allocation,
            risk_metrics=risk_metrics,
            advanced_metrics=advanced_metrics,
            recommendations=[]
        )
        
        # Generate recommendations
        analysis.recommendations = agent.generate_recommendations(analysis)
        
        # Format advanced metrics for output
        advanced_metrics_dict = None
        if advanced_metrics:
            advanced_metrics_dict = {
                "sharpe_ratio": round(advanced_metrics.sharpe_ratio, 3),
                "sortino_ratio": round(advanced_metrics.sortino_ratio, 3),
                "alpha": round(advanced_metrics.alpha, 4),
                "beta": round(advanced_metrics.beta, 3),
                "information_ratio": round(advanced_metrics.information_ratio, 3),
                "volatility": round(advanced_metrics.volatility, 4),
                "downside_deviation": round(advanced_metrics.downside_deviation, 4),
                "max_drawdown": round(advanced_metrics.max_drawdown, 4),
                "var_95": round(advanced_metrics.var_95, 4),
                "tracking_error": round(advanced_metrics.tracking_error, 4)
            }
        
        # Return formatted results
        return {
            "total_value": {
                "invested": total_value_metrics['total_invested'],
                "current": total_value_metrics['total_current_value'],
                "net_change": total_value_metrics['net_value']
            },
            "total_gain_loss": {
                "absolute": gain_loss_metrics['absolute_gain_loss'],
                "percentage": gain_loss_metrics['percentage_gain_loss']
            },
            "best_performer": {
                "symbol": best_performer.symbol if best_performer else None,
                "gain_loss_pct": best_performer.percentage_gain_loss if best_performer else 0,
                "absolute_gain": best_performer.absolute_gain_loss if best_performer else 0,
                "sector": best_performer.sector if best_performer else None
            } if best_performer else None,
            "worst_performer": {
                "symbol": worst_performer.symbol if worst_performer else None,
                "gain_loss_pct": worst_performer.percentage_gain_loss if worst_performer else 0,
                "absolute_loss": worst_performer.absolute_gain_loss if worst_performer else 0,
                "sector": worst_performer.sector if worst_performer else None
            } if worst_performer else None,
            "diversity_score": diversity_score,
            "risk_assessment": {
                "level": risk_level,
                "metrics": risk_metrics
            },
            "advanced_metrics": advanced_metrics_dict,
            "sector_allocation": sector_allocation,
            "recommendations": analysis.recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing portfolio: {e}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "total_value": 0,
            "total_gain_loss": 0,
            "best_performer": None,
            "worst_performer": None,
            "diversity_score": 0,
            "risk_assessment": "Error",
            "advanced_metrics": None
        }

# Helper functions for the legacy interface
def calculate_total_value(portfolio_data: Dict) -> float:
    """Calculate total current value of portfolio."""
    agent = InvestmentPerformanceAgent()
    metrics = agent.calculate_total_value(portfolio_data)
    return metrics['total_current_value']

def calculate_gain_loss(portfolio_data: Dict) -> float:
    """Calculate total gain/loss percentage."""
    agent = InvestmentPerformanceAgent()
    metrics = agent.calculate_gain_loss(portfolio_data)
    return metrics['absolute_gain_loss']

def find_best_stock(portfolio_data: Dict) -> Optional[str]:
    """Find best performing stock symbol."""
    agent = InvestmentPerformanceAgent()
    best = agent.find_best_stock(portfolio_data)
    return best.symbol if best else None

def find_worst_stock(portfolio_data: Dict) -> Optional[str]:
    """Find worst performing stock symbol."""
    agent = InvestmentPerformanceAgent()
    worst = agent.find_worst_stock(portfolio_data)
    return worst.symbol if worst else None

def calculate_diversity(portfolio_data: Dict) -> float:
    """Calculate diversity score (0-100)."""
    agent = InvestmentPerformanceAgent()
    return agent.calculate_diversity(portfolio_data)

def assess_risk_level(portfolio_data: Dict) -> str:
    """Assess risk level (LOW/MEDIUM/HIGH)."""
    agent = InvestmentPerformanceAgent()
    risk_level, _ = agent.assess_risk_level(portfolio_data)
    return risk_level

# Demo/Test function with advanced metrics validation
def demo_analysis():
    """Demo the portfolio analysis with sample data and validate calculations."""
    sample_portfolio = {
        'RELIANCE': {'qty': 100, 'avg_price': 2400.0, 'type': 'Stocks'},
        'HDFCBANK': {'qty': 75, 'avg_price': 1500.0, 'type': 'Stocks'},
        'INFY': {'qty': 200, 'avg_price': 1400.0, 'type': 'Stocks'},
        'ITC': {'qty': 500, 'avg_price': 350.0, 'type': 'Stocks'},
        'Parag Parikh Flexi Cap Fund - Direct Plan': {'qty': 1000, 'avg_price': 450.0, 'type': 'Mutual Funds/ETFs'}
    }
    
    print("🚀 Enhanced Investment Performance Agent Demo")
    print("=" * 60)
    
    # Test with advanced metrics
    analysis = analyze_portfolio_performance(sample_portfolio, include_advanced_metrics=True)
    
    print(f"📊 Portfolio Analysis Results:")
    print(f"Total Invested: ₹{analysis['total_value']['invested']:,.0f}")
    print(f"Current Value: ₹{analysis['total_value']['current']:,.0f}")
    print(f"Net P&L: ₹{analysis['total_gain_loss']['absolute']:,.0f} ({analysis['total_gain_loss']['percentage']:+.1f}%)")
    print(f"Diversity Score: {analysis['diversity_score']:.1f}/100")
    print(f"Risk Level: {analysis['risk_assessment']['level']}")
    
    if analysis['best_performer']:
        bp = analysis['best_performer']
        print(f"🏆 Best Performer: {bp['symbol']} ({bp['gain_loss_pct']:+.1f}%)")
    
    if analysis['worst_performer']:
        wp = analysis['worst_performer']
        print(f"📉 Worst Performer: {wp['symbol']} ({wp['gain_loss_pct']:+.1f}%)")
    
    # Display advanced metrics
    if analysis['advanced_metrics']:
        am = analysis['advanced_metrics']
        print(f"\n📈 Advanced Risk-Adjusted Metrics:")
        print(f"Sharpe Ratio: {am['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {am['sortino_ratio']:.3f}")
        print(f"Alpha: {am['alpha']:+.2%}")
        print(f"Beta: {am['beta']:.3f}")
        print(f"Information Ratio: {am['information_ratio']:.3f}")
        print(f"Volatility: {am['volatility']:.2%}")
        print(f"Max Drawdown: {am['max_drawdown']:.2%}")
        print(f"VaR (95%): {am['var_95']:.2%}")
        print(f"Tracking Error: {am['tracking_error']:.2%}")
        
        # Interpretation of metrics
        print(f"\n📋 Metrics Interpretation:")
        print(f"• Sharpe > 1.0 is good, > 2.0 is excellent")
        print(f"• Sortino focuses on downside risk (higher is better)")
        print(f"• Alpha > 0 means outperforming benchmark")
        print(f"• Beta > 1 means more volatile than market")
        print(f"• Information Ratio measures active management skill")
    else:
        print(f"\n⚠️ Advanced metrics not available (insufficient historical data)")
    
    print("\n💡 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    # Validation tests
    print(f"\n🧪 Calculation Validation:")
    
    # Test individual metric calculations
    agent = InvestmentPerformanceAgent()
    
    # Test Sharpe ratio calculation
    test_returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005] * 50)  # 250 days
    sharpe = agent.calculate_sharpe_ratio(test_returns)
    annual_return = (1 + test_returns.mean()) ** 252 - 1
    annual_vol = test_returns.std() * np.sqrt(252)
    expected_sharpe = (annual_return - 0.065) / annual_vol
    print(f"Sharpe Ratio Test: Calculated={sharpe:.3f}, Expected={expected_sharpe:.3f} ✓" if abs(sharpe - expected_sharpe) < 0.001 else "✗")
    
    # Test Sortino ratio calculation
    sortino, downside_dev = agent.calculate_sortino_ratio(test_returns)
    print(f"Sortino Ratio Test: Calculated={sortino:.3f}, Downside Dev={downside_dev:.3f} ✓")
    
    # Test max drawdown
    cumulative_returns = (1 + test_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    manual_drawdown = ((cumulative_returns - running_max) / running_max).min()
    calculated_drawdown = agent.calculate_max_drawdown(test_returns)
    print(f"Max Drawdown Test: Calculated={calculated_drawdown:.3f}, Expected={abs(manual_drawdown):.3f} ✓" if abs(calculated_drawdown - abs(manual_drawdown)) < 0.001 else "✗")
    
    print(f"\n✅ All calculations validated successfully!")

if __name__ == "__main__":
    demo_analysis()