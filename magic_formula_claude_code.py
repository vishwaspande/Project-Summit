import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yfinance as yf
from datetime import datetime, timedelta
import requests
import warnings
import os
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
            self.indian_bond_yield = 6.43  # Current yield as of August 2025
            print(f"Using Indian 10-Year Bond Yield: {self.indian_bond_yield}%")
            return self.indian_bond_yield
        except Exception as e:
            print(f"Error fetching bond yield: {e}")
            # Fallback to approximate rate
            self.indian_bond_yield = 6.43
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
                risk_free_rate = self.indian_bond_yield or 6.43
            
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
                       output_dir: str = "magic_formula_results",
                       file_prefix: str = "magic_formula") -> pd.DataFrame:
        """
        Generate comprehensive reports in multiple formats
        
        Args:
            screened_stocks: List of screened stocks
            output_dir: Directory to save files (default: magic_formula_results)
            file_prefix: Prefix for output files (default: magic_formula)
        
        Returns:
            DataFrame with results
        """
        if not screened_stocks:
            print("No stocks to report!")
            # Create empty report for record keeping
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            empty_file = os.path.join(output_dir, f"{file_prefix}_no_results_{timestamp}.txt")
            with open(empty_file, 'w') as f:
                f.write(f"Magic Formula Screening Results\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Result: No stocks passed the screening criteria\n")
                f.write(f"Bond Yield: {self.indian_bond_yield}%\n")
                f.write(f"EY Threshold: {self.indian_bond_yield + 4.0}%\n")
            print(f"Empty results logged to: {empty_file}")
            return pd.DataFrame()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create main results DataFrame
        data = []
        for i, stock in enumerate(screened_stocks, 1):
            data.append({
                'Magic Formula Rank': i,
                'Symbol': stock.symbol,
                'Company Name': stock.company_name,
                'ROCE (%)': round(stock.roce, 2),
                'Earning Yield (%)': round(stock.earning_yield, 2),
                'WACC (%)': round(stock.wacc, 2),
                'Market Cap (Cr)': round(stock.market_cap/10000000, 0),
                'ROCE Rank': stock.roce_rank,
                'EY Rank': stock.ey_rank,
                'Combined Rank': stock.combined_rank,
                'ROCE vs WACC': f"+{round(stock.roce - stock.wacc, 2)}%",
                'EY vs Bond+4%': f"+{round(stock.earning_yield - (self.indian_bond_yield + 4.0), 2)}%"
            })
        
        df = pd.DataFrame(data)
        
        # Save main results in multiple formats
        csv_file = os.path.join(output_dir, f"{file_prefix}_results_{timestamp}.csv")
        excel_file = os.path.join(output_dir, f"{file_prefix}_results_{timestamp}.xlsx")
        
        # Save CSV
        df.to_csv(csv_file, index=False)
        print(f"✅ Results saved to: {csv_file}")
        
        # Save Excel with multiple sheets (only if xlsxwriter is available)
        try:
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                # Main results sheet
                df.to_excel(writer, sheet_name='Magic Formula Results', index=False)
                
                # Summary statistics sheet
                summary_data = {
                    'Metric': [
                        'Stocks Passed Screening',
                        'Average ROCE (%)',
                        'Average EY (%)',
                        'Average WACC (%)',
                        'Bond Yield Used (%)',
                        'EY Threshold (%)',
                        'Median Market Cap (Cr)',
                        'Top Stock ROCE (%)',
                        'Top Stock EY (%)'
                    ],
                    'Value': [
                        len(screened_stocks),
                        round(df['ROCE (%)'].mean(), 2),
                        round(df['Earning Yield (%)'].mean(), 2),
                        round(df['WACC (%)'].mean(), 2),
                        self.indian_bond_yield,
                        round(self.indian_bond_yield + 4.0, 2),
                        round(df['Market Cap (Cr)'].median(), 0),
                        round(df['ROCE (%)'].max(), 2),
                        round(df['Earning Yield (%)'].max(), 2)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary Stats', index=False)
                
                # Top performers sheet
                top_10 = df.head(10)[['Magic Formula Rank', 'Symbol', 'Company Name', 
                                    'ROCE (%)', 'Earning Yield (%)', 'Combined Rank']]
                top_10.to_excel(writer, sheet_name='Top 10 Picks', index=False)
            
            print(f"✅ Detailed Excel report saved to: {excel_file}")
            
        except ImportError:
            print(f"⚠️  xlsxwriter not available. Install with: pip install xlsxwriter")
            print(f"📊 CSV file saved instead: {csv_file}")
        
        # Create a summary text report
        txt_file = os.path.join(output_dir, f"{file_prefix}_summary_{timestamp}.txt")
        with open(txt_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MAGIC FORMULA STOCK SCREENING REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Bond Yield Used: {self.indian_bond_yield}%\n")
            f.write(f"EY Threshold: {self.indian_bond_yield + 4.0}%\n")
            f.write(f"Filters Applied: ROCE > WACC, EY > Bond Yield + 4%\n\n")
            
            f.write("SCREENING RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Stocks Passed Screening: {len(screened_stocks)}\n")
            f.write(f"Average ROCE: {round(df['ROCE (%)'].mean(), 2)}%\n")
            f.write(f"Average Earning Yield: {round(df['Earning Yield (%)'].mean(), 2)}%\n")
            f.write(f"Average WACC: {round(df['WACC (%)'].mean(), 2)}%\n\n")
            
            f.write("TOP 5 MAGIC FORMULA PICKS:\n")
            f.write("-" * 40 + "\n")
            for i, stock in enumerate(screened_stocks[:5], 1):
                f.write(f"{i}. {stock.company_name} ({stock.symbol})\n")
                f.write(f"   ROCE: {stock.roce:.2f}% | EY: {stock.earning_yield:.2f}% | Rank: {stock.combined_rank}\n")
                f.write(f"   Market Cap: {stock.market_cap/10000000:.0f} Cr\n\n")
            
            if len(screened_stocks) > 5:
                f.write("COMPLETE RANKINGS:\n")
                f.write("-" * 20 + "\n")
                for i, stock in enumerate(screened_stocks, 1):
                    f.write(f"{i:2d}. {stock.symbol:12s} | ROCE: {stock.roce:5.1f}% | EY: {stock.earning_yield:5.1f}%\n")
        
        print(f"✅ Summary report saved to: {txt_file}")
        
        # Create a detailed log file for debugging
        log_file = os.path.join(output_dir, f"{file_prefix}_detailed_log_{timestamp}.txt")
        with open(log_file, 'w') as f:
            f.write("DETAILED SCREENING LOG\n")
            f.write("=" * 50 + "\n\n")
            for stock in screened_stocks:
                f.write(f"Company: {stock.company_name} ({stock.symbol})\n")
                f.write(f"Market Cap: ₹{stock.market_cap/10000000:.0f} Cr\n")
                f.write(f"ROCE: {stock.roce:.2f}% (Rank: {stock.roce_rank})\n")
                f.write(f"Earning Yield: {stock.earning_yield:.2f}% (Rank: {stock.ey_rank})\n")
                f.write(f"WACC: {stock.wacc:.2f}%\n")
                f.write(f"ROCE vs WACC: +{stock.roce - stock.wacc:.2f}%\n")
                f.write(f"EY vs Threshold: +{stock.earning_yield - (self.indian_bond_yield + 4.0):.2f}%\n")
                f.write(f"Combined Magic Formula Rank: {stock.combined_rank}\n")
                f.write("-" * 50 + "\n")
        
        print(f"✅ Detailed log saved to: {log_file}")
        print(f"\n📁 All files saved in directory: {output_dir}")
        
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
        
        # Generate comprehensive reports
        df = screener.generate_report(
            screened_stocks, 
            output_dir="magic_formula_results",
            file_prefix="project_summit_screening"
        )
        
        print(df.to_string(index=False))
        
        print(f"\n=== TOP 5 MAGIC FORMULA PICKS ===")
        for i, stock in enumerate(screened_stocks[:5], 1):
            print(f"{i}. {stock.company_name} ({stock.symbol})")
            print(f"   ROCE: {stock.roce:.2f}% | EY: {stock.earning_yield:.2f}% | Combined Rank: {stock.combined_rank}")
    
    else:
        print("No stocks passed the Magic Formula screening criteria.")
        print("Consider adjusting the filters or expanding the stock universe.")
        
        # Save empty results for record keeping
        screener.generate_report(
            [], 
            output_dir="magic_formula_results",
            file_prefix="project_summit_screening_empty"
        )

if __name__ == "__main__":
    main()