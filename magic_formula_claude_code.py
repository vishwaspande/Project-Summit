import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yfinance as yf
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StockMetrics:
    """Data class to hold stock metrics"""
    symbol: str
    company_name: str
    roce: float
    earning_yield: float
    wacc: float
    market_cap: float
    combined_rank: float
    roce_rank: int
    ey_rank: int

class MagicFormulaScreener:
    """
    Magic Formula Stock Screener for Indian Markets
    
    Implements Joel Greenblatt's Magic Formula with additional filters:
    - ROCE (Return on Capital Employed)
    - Earning Yield (EY)
    - WACC filter (ROCE > WACC)
    - Bond Yield filter (EY > Bond Yield + margin)
    """
    
    def __init__(self):
        self.indian_bond_yield = None
        self.stock_data = []
        
    def get_indian_bond_yield(self) -> float:
        """
        Get current Indian 10-year government bond yield
        Note: In production, you'd want to use a reliable API like Bloomberg, RBI, or NSE
        """
        try:
            # Placeholder - replace with actual bond yield API
            # For now, using approximate current Indian 10-year bond yield
            self.indian_bond_yield = 7.2  # Approximate current yield
            print(f"Using Indian 10-Year Bond Yield: {self.indian_bond_yield}%")
            return self.indian_bond_yield
        except Exception as e:
            print(f"Error fetching bond yield: {e}")
            # Fallback to approximate rate
            self.indian_bond_yield = 7.2
            return self.indian_bond_yield
    
    def calculate_roce(self, financial_data: Dict) -> float:
        """
        Calculate Return on Capital Employed (ROCE)
        ROCE = EBIT / Capital Employed
        Capital Employed = Total Assets - Current Liabilities
        """
        try:
            ebit = financial_data.get('ebit', 0)
            total_assets = financial_data.get('total_assets', 0)
            current_liabilities = financial_data.get('current_liabilities', 0)
            
            capital_employed = total_assets - current_liabilities
            if capital_employed <= 0:
                return 0
                
            roce = (ebit / capital_employed) * 100
            return max(0, roce)  # Ensure non-negative
        except:
            return 0
    
    def calculate_earning_yield(self, financial_data: Dict) -> float:
        """
        Calculate Earnings Yield
        EY = (Net Income / Market Capitalization) * 100
        """
        try:
            net_income = financial_data.get('net_income', 0)
            market_cap = financial_data.get('market_cap', 0)
            
            if market_cap <= 0:
                return 0
                
            earning_yield = (net_income / market_cap) * 100
            return earning_yield
        except:
            return 0
    
    def estimate_wacc(self, financial_data: Dict, risk_free_rate: float = None) -> float:
        """
        Estimate WACC (Weighted Average Cost of Capital)
        Simplified WACC = (E/V * Re) + (D/V * Rd * (1-Tax Rate))
        
        For simplicity, using: Risk-free rate + Beta * Market Risk Premium + Debt Premium
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.indian_bond_yield or 7.2
            
            beta = financial_data.get('beta', 1.0)
            debt_to_equity = financial_data.get('debt_to_equity', 0.3)
            
            # Market risk premium for India (approximate)
            market_risk_premium = 6.0
            
            # Debt premium based on credit quality (simplified)
            debt_premium = min(3.0, debt_to_equity * 2)
            
            # Simplified WACC calculation
            equity_premium = beta * market_risk_premium
            wacc = risk_free_rate + equity_premium + debt_premium
            
            return max(risk_free_rate + 1, wacc)  # Minimum WACC
        except:
            return 12.0  # Default WACC for Indian companies
    
    def fetch_stock_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch stock data using yfinance
        Note: For Indian stocks, append .NS (NSE) or .BO (BSE)
        """
        try:
            # Ensure Indian stock format
            if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
                symbol += '.NS'  # Default to NSE
            
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Get financial data
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            
            if financials.empty or balance_sheet.empty:
                return None
            
            # Extract key metrics (most recent year)
            latest_col = financials.columns[0] if len(financials.columns) > 0 else None
            if latest_col is None:
                return None
            
            financial_data = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 1.0),
                'net_income': financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0,
                'ebit': financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else 
                       financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index else 0,
                'total_assets': balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0,
                'current_liabilities': balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else 0,
                'total_debt': balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0,
            }
            
            # Calculate debt-to-equity ratio
            total_equity = financial_data['total_assets'] - financial_data['total_debt']
            financial_data['debt_to_equity'] = (
                financial_data['total_debt'] / total_equity if total_equity > 0 else 0
            )
            
            return financial_data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def screen_stocks(self, stock_symbols: List[str], 
                     ey_margin: float = 4.0,
                     min_market_cap: float = 1000000000) -> List[StockMetrics]:
        """
        Screen stocks using Magic Formula with additional filters
        
        Args:
            stock_symbols: List of stock symbols to screen
            ey_margin: Margin above bond yield for EY filter (default 4%)
            min_market_cap: Minimum market cap filter (default 1B)
        
        Returns:
            List of StockMetrics sorted by Magic Formula ranking
        """
        print("Starting Magic Formula screening...")
        print(f"Screening {len(stock_symbols)} stocks...")
        
        # Get bond yield
        bond_yield = self.get_indian_bond_yield()
        
        screened_stocks = []
        
        for i, symbol in enumerate(stock_symbols, 1):
            print(f"Processing {i}/{len(stock_symbols)}: {symbol}")
            
            financial_data = self.fetch_stock_data(symbol)
            if not financial_data:
                continue
            
            # Calculate metrics
            roce = self.calculate_roce(financial_data)
            earning_yield = self.calculate_earning_yield(financial_data)
            wacc = self.estimate_wacc(financial_data)
            market_cap = financial_data['market_cap']
            
            # Apply filters
            if market_cap < min_market_cap:
                print(f"  Filtered out {symbol}: Market cap too small")
                continue
                
            if roce <= wacc:
                print(f"  Filtered out {symbol}: ROCE ({roce:.2f}%) <= WACC ({wacc:.2f}%)")
                continue
                
            if earning_yield <= (bond_yield + ey_margin):
                print(f"  Filtered out {symbol}: EY ({earning_yield:.2f}%) <= Bond Yield + Margin ({bond_yield + ey_margin:.2f}%)")
                continue
            
            stock_metric = StockMetrics(
                symbol=symbol,
                company_name=financial_data['company_name'],
                roce=roce,
                earning_yield=earning_yield,
                wacc=wacc,
                market_cap=market_cap,
                combined_rank=0,  # Will be calculated later
                roce_rank=0,      # Will be calculated later
                ey_rank=0         # Will be calculated later
            )
            
            screened_stocks.append(stock_metric)
            print(f"  Added {symbol}: ROCE={roce:.2f}%, EY={earning_yield:.2f}%")
        
        if not screened_stocks:
            print("No stocks passed the screening criteria!")
            return []
        
        # Rank stocks by ROCE and EY
        screened_stocks.sort(key=lambda x: x.roce, reverse=True)
        for rank, stock in enumerate(screened_stocks, 1):
            stock.roce_rank = rank
        
        screened_stocks.sort(key=lambda x: x.earning_yield, reverse=True)
        for rank, stock in enumerate(screened_stocks, 1):
            stock.ey_rank = rank
        
        # Calculate combined Magic Formula rank
        for stock in screened_stocks:
            stock.combined_rank = stock.roce_rank + stock.ey_rank
        
        # Sort by combined rank (lower is better)
        screened_stocks.sort(key=lambda x: x.combined_rank)
        
        return screened_stocks
    
    def generate_report(self, screened_stocks: List[StockMetrics], 
                       output_file: str = None) -> pd.DataFrame:
        """
        Generate a detailed report of screened stocks
        """
        if not screened_stocks:
            print("No stocks to report!")
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for i, stock in enumerate(screened_stocks, 1):
            data.append({
                'Magic Formula Rank': i,
                'Symbol': stock.symbol,
                'Company Name': stock.company_name,
                'ROCE (%)': f"{stock.roce:.2f}",
                'Earning Yield (%)': f"{stock.earning_yield:.2f}",
                'WACC (%)': f"{stock.wacc:.2f}",
                'Market Cap (Cr)': f"{stock.market_cap/10000000:.0f}",
                'ROCE Rank': stock.roce_rank,
                'EY Rank': stock.ey_rank,
                'Combined Rank': stock.combined_rank
            })
        
        df = pd.DataFrame(data)
        
        # Save to file if specified
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Report saved to {output_file}")
        
        return df

# Example usage
def main():
    """
    Main function to demonstrate the Magic Formula screener
    """
    print("=== Magic Formula Stock Screener - Project Summit ===\n")
    
    # Initialize screener
    screener = MagicFormulaScreener()
    
    # Sample Indian stock symbols (Nifty 50 components)
    sample_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HDFC.NS', 'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS',
        'ITC.NS', 'ASIANPAINT.NS', 'LT.NS', 'AXISBANK.NS', 'DMART.NS',
        'MARUTI.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'NESTLEIND.NS'
    ]
    
    print(f"Sample stocks to screen: {len(sample_stocks)}")
    print("Note: This is a demo with sample stocks. Replace with your stock universe.\n")
    
    # Screen stocks
    screened_stocks = screener.screen_stocks(
        stock_symbols=sample_stocks,
        ey_margin=4.0,  # EY > Bond Yield + 4%
        min_market_cap=10000000000  # 1000 Cr minimum market cap
    )
    
    # Generate report
    if screened_stocks:
        print(f"\n=== MAGIC FORMULA RESULTS ===")
        print(f"Stocks passed screening: {len(screened_stocks)}")
        print(f"Bond Yield used: {screener.indian_bond_yield}%")
        print(f"EY Margin: 4.0%")
        print(f"Filters Applied: ROCE > WACC, EY > Bond Yield + 4%\n")
        
        df = screener.generate_report(screened_stocks, 'magic_formula_results.csv')
        print(df.to_string(index=False))
        
        print(f"\n=== TOP 5 MAGIC FORMULA PICKS ===")
        for i, stock in enumerate(screened_stocks[:5], 1):
            print(f"{i}. {stock.company_name} ({stock.symbol})")
            print(f"   ROCE: {stock.roce:.2f}% | EY: {stock.earning_yield:.2f}% | Combined Rank: {stock.combined_rank}")
    
    else:
        print("No stocks passed the Magic Formula screening criteria.")
        print("Consider adjusting the filters or expanding the stock universe.")

if __name__ == "__main__":
    main()