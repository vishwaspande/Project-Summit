# Screen stocks
    screened_stocks, all_stocks = screener.screen_stocks(
        stock_symbols=sample_stocks,
        ey_margin=4.0,  # EY > Bond Yield + 4%
        min_market_cap=10000000000  # 1000 Cr minimum market cap
    )
    
    # Generate comprehensive report with ALL data
    print(f"\n=== COMPLETE ANALYSIS RESULTS ===")
    print(f"Total stocks analyzed: {len(all_stocks)}")
    print(f"Stocks passed screening: {len(screened_stocks)}")
    print(f"Stocks failed screening: {len(all_stocks) - len(screened_stocks)}")
    print(f"Pass rate: {(len(screened_stocks) / len(all_stocks)) * 100:.1f}%" if all_stocks else "0%")
    print(f"Bond Yield used: {screener.indian_bond_yield}%")
    print(f"EY Threshold: {screener.indian_bond_yield + 4.0}%")
    print(f"Filters Applied: ROCE > WACC, EY > Bond Yield + 4%\n")
    
    # Generate comprehensive reports including failed stocks
    df = screener.generate_report(
        screened_stocks, 
        all_stocks,
        output_dir="magic_formula_results",
        file_prefix="project_summit_complete_analysis"
    )
    
    if screened_stocks:
        print(f"\n=== TOP 5 MAGIC FORMULA PICKS ===")
        for i, stock in enumerate(screened_stocks[:5], 1):
            print(f"{i}. {stock.company_name} ({stock.symbol})")
            print(f"   ROCE: {stock.roce:.2f}% | EY: {stock.earning_yield:.2f}% | Combined Rank: {stock.combined_rank}")
    
    # Show failure analysis
    failed_stocks = [stock for stock in all_stocks if not stock.passed_screening]
    if failed_stocks:
        print(f"\n=== FAILURE ANALYSIS ===")
        failure_reasons = {}
        for stock in failed_stocks:
            if stock.filter_failure_reason in failure_reasons:
                failure_reasons[stock.filter_failure_reason] += 1
            else:
                failure_reasons[stock.filter_failure_reason] = 1
        
        print("Most common failure reasons:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {reason}: {count} stocks")
    
    print(f"\n📊 Complete analysis saved with ALL stock data for your analysis!")
    print(f"💡 Check Excel file for detailed breakdown of passed and failed stocks.")

if __name__ == "__main__":
    main()import pandas as pd
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
    # New fields for complete analysis
    passed_screening: bool = False
    filter_failure_reason: str = ""
    net_income: float = 0
    total_assets: float = 0
    current_liabilities: float = 0
    ebit: float = 0
    beta: float = 1.0
    debt_to_equity: float = 0
    pe_ratio: float = 0

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
                     min_market_cap: float = 1000000000) -> Tuple[List[StockMetrics], List[StockMetrics]]:
        """
        Screen stocks using Magic Formula with additional filters
        
        Args:
            stock_symbols: List of stock symbols to screen
            ey_margin: Margin above bond yield for EY filter (default 4%)
            min_market_cap: Minimum market cap filter (default 1B)
        
        Returns:
            Tuple of (passed_stocks, all_stocks_analyzed)
        """
        print("Starting Magic Formula screening...")
        print(f"Screening {len(stock_symbols)} stocks...")
        
        # Get bond yield
        bond_yield = self.get_indian_bond_yield()
        
        screened_stocks = []  # Stocks that passed
        all_stocks = []       # All stocks analyzed
        
        for i, symbol in enumerate(stock_symbols, 1):
            print(f"Processing {i}/{len(stock_symbols)}: {symbol}")
            
            financial_data = self.fetch_stock_data(symbol)
            if not financial_data:
                # Create entry for failed data fetch
                failed_stock = StockMetrics(
                    symbol=symbol,
                    company_name=symbol,
                    roce=0,
                    earning_yield=0,
                    wacc=0,
                    market_cap=0,
                    combined_rank=999,
                    roce_rank=999,
                    ey_rank=999,
                    passed_screening=False,
                    filter_failure_reason="Data fetch failed"
                )
                all_stocks.append(failed_stock)
                print(f"  ❌ {symbol}: Data fetch failed")
                continue
            
            # Calculate metrics
            roce = self.calculate_roce(financial_data)
            earning_yield = self.calculate_earning_yield(financial_data)
            wacc = self.estimate_wacc(financial_data)
            market_cap = financial_data['market_cap']
            
            # Calculate P/E ratio
            pe_ratio = market_cap / financial_data['net_income'] if financial_data['net_income'] > 0 else 0
            
            # Create comprehensive stock metrics
            stock_metric = StockMetrics(
                symbol=symbol,
                company_name=financial_data['company_name'],
                roce=roce,
                earning_yield=earning_yield,
                wacc=wacc,
                market_cap=market_cap,
                combined_rank=0,  # Will be calculated later for passed stocks
                roce_rank=0,      # Will be calculated later for passed stocks
                ey_rank=0,        # Will be calculated later for passed stocks
                passed_screening=False,  # Will be updated if passes
                filter_failure_reason="",
                net_income=financial_data['net_income'],
                total_assets=financial_data['total_assets'],
                current_liabilities=financial_data['current_liabilities'],
                ebit=financial_data['ebit'],
                beta=financial_data['beta'],
                debt_to_equity=financial_data['debt_to_equity'],
                pe_ratio=pe_ratio
            )
            
            # Apply filters and track failure reasons
            failure_reasons = []
            
            if market_cap < min_market_cap:
                failure_reasons.append(f"Market cap too small ({market_cap/10000000:.0f} < {min_market_cap/10000000:.0f} Cr)")
                
            if roce <= wacc:
                failure_reasons.append(f"ROCE ({roce:.2f}%) <= WACC ({wacc:.2f}%)")
                
            if earning_yield <= (bond_yield + ey_margin):
                failure_reasons.append(f"EY ({earning_yield:.2f}%) <= Bond Yield + Margin ({bond_yield + ey_margin:.2f}%)")
            
            # Update stock metrics based on filter results
            if failure_reasons:
                stock_metric.filter_failure_reason = "; ".join(failure_reasons)
                stock_metric.passed_screening = False
                print(f"  ❌ {symbol}: {stock_metric.filter_failure_reason}")
            else:
                stock_metric.passed_screening = True
                screened_stocks.append(stock_metric)
                print(f"  ✅ {symbol}: ROCE={roce:.2f}%, EY={earning_yield:.2f}%")
            
            # Add to complete analysis list
            all_stocks.append(stock_metric)
        
        # Rank only the stocks that passed screening
        if screened_stocks:
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
            
            print(f"\n✅ {len(screened_stocks)} stocks passed screening out of {len(all_stocks)} analyzed")
        else:
            print(f"\n❌ No stocks passed the screening criteria out of {len(all_stocks)} analyzed")
        
        return screened_stocks, all_stocks
    
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
        
        # File paths
        csv_file = os.path.join(output_dir, f"{file_prefix}_results_{timestamp}.csv")
        excel_file = os.path.join(output_dir, f"{file_prefix}_results_{timestamp}.xlsx")
        
        # PRIORITY: Save Excel file first
        excel_saved = False
        
        # Try multiple Excel engines
        excel_engines = ['xlsxwriter', 'openpyxl']
        
        for engine in excel_engines:
            try:
                print(f"Attempting to save Excel using {engine} engine...")
                
                with pd.ExcelWriter(excel_file, engine=engine) as writer:
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
                    
                    # If using xlsxwriter, add formatting
                    if engine == 'xlsxwriter':
                        workbook = writer.book
                        worksheet = writer.sheets['Magic Formula Results']
                        
                        # Add header formatting
                        header_format = workbook.add_format({
                            'bold': True,
                            'text_wrap': True,
                            'valign': 'top',
                            'fg_color': '#D7E4BC',
                            'border': 1
                        })
                        
                        # Apply header formatting
                        for col_num, value in enumerate(df.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                        
                        # Auto-adjust column widths
                        for i, col in enumerate(df.columns):
                            column_len = max(df[col].astype(str).str.len().max(), len(col))
                            worksheet.set_column(i, i, min(column_len + 2, 50))
                
                print(f"✅ EXCEL FILE SAVED SUCCESSFULLY: {excel_file}")
                print(f"📊 Excel engine used: {engine}")
                excel_saved = True
                break
                
            except ImportError as e:
                print(f"⚠️  {engine} not available: {e}")
                continue
            except Exception as e:
                print(f"❌ Error with {engine}: {e}")
                continue
        
        # If Excel failed, save CSV as backup
        if not excel_saved:
            print(f"⚠️  Excel save failed. Installing required packages...")
            print(f"Run: pip install xlsxwriter openpyxl")
            df.to_csv(csv_file, index=False)
            print(f"📄 CSV backup saved: {csv_file}")
        else:
            # Also save CSV for compatibility
            df.to_csv(csv_file, index=False)
            print(f"📄 CSV backup also saved: {csv_file}")
        
        # Create summary text report (minimal)
        txt_file = os.path.join(output_dir, f"{file_prefix}_summary_{timestamp}.txt")
        with open(txt_file, 'w') as f:
            f.write("MAGIC FORMULA SCREENING SUMMARY\n")
            f.write("=" * 40 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Stocks Passed: {len(screened_stocks)}\n")
            f.write(f"Bond Yield: {self.indian_bond_yield}%\n")
            f.write(f"EY Threshold: {self.indian_bond_yield + 4.0}%\n\n")
            
            f.write("TOP 5 PICKS:\n")
            f.write("-" * 20 + "\n")
            for i, stock in enumerate(screened_stocks[:5], 1):
                f.write(f"{i}. {stock.symbol} - ROCE: {stock.roce:.1f}%, EY: {stock.earning_yield:.1f}%\n")
        
        print(f"📝 Summary saved: {txt_file}")
        print(f"\n🎯 MAIN OUTPUT: {excel_file if excel_saved else csv_file}")
        
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