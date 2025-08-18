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
    
    def generate_report(self, screened_stocks: List[StockMetrics], all_stocks: List[StockMetrics],
                       output_dir: str = "magic_formula_results",
                       file_prefix: str = "magic_formula") -> pd.DataFrame:
        """
        Generate comprehensive reports in multiple formats
        
        Args:
            screened_stocks: List of stocks that passed screening
            all_stocks: List of all stocks analyzed (passed + failed)
            output_dir: Directory to save files (default: magic_formula_results)
            file_prefix: Prefix for output files (default: magic_formula)
        
        Returns:
            DataFrame with all results
        """
        if not all_stocks:
            print("No stocks were analyzed!")
            return pd.DataFrame()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive DataFrame with ALL stocks
        all_data = []
        for stock in all_stocks:
            all_data.append({
                'Symbol': stock.symbol,
                'Company Name': stock.company_name,
                'Passed Screening': 'YES' if stock.passed_screening else 'NO',
                'Failure Reason': stock.filter_failure_reason if not stock.passed_screening else 'PASSED',
                'Magic Formula Rank': stock.combined_rank if stock.passed_screening else 'N/A',
                'ROCE (%)': round(stock.roce, 2),
                'Earning Yield (%)': round(stock.earning_yield, 2),
                'WACC (%)': round(stock.wacc, 2),
                'Market Cap (Cr)': round(stock.market_cap/10000000, 0) if stock.market_cap > 0 else 0,
                'P/E Ratio': round(stock.pe_ratio, 2) if stock.pe_ratio > 0 else 'N/A',
                'ROCE Rank': stock.roce_rank if stock.passed_screening else 'N/A',
                'EY Rank': stock.ey_rank if stock.passed_screening else 'N/A',
                'Combined Rank': stock.combined_rank if stock.passed_screening else 'N/A',
                'ROCE vs WACC': f"+{round(stock.roce - stock.wacc, 2)}%" if stock.roce > 0 and stock.wacc > 0 else 'N/A',
                'EY vs Bond+4%': f"+{round(stock.earning_yield - (self.indian_bond_yield + 4.0), 2)}%" if stock.earning_yield > 0 else 'N/A',
                'Net Income (Cr)': round(stock.net_income/10000000, 2) if stock.net_income != 0 else 0,
                'Total Assets (Cr)': round(stock.total_assets/10000000, 2) if stock.total_assets != 0 else 0,
                'Current Liabilities (Cr)': round(stock.current_liabilities/10000000, 2) if stock.current_liabilities != 0 else 0,
                'EBIT (Cr)': round(stock.ebit/10000000, 2) if stock.ebit != 0 else 0,
                'Beta': round(stock.beta, 2),
                'Debt to Equity': round(stock.debt_to_equity, 2)
            })
        
        all_df = pd.DataFrame(all_data)
        
        # Create passed stocks DataFrame
        passed_data = []
        for i, stock in enumerate(screened_stocks, 1):
            passed_data.append({
                'Magic Formula Rank': i,
                'Symbol': stock.symbol,
                'Company Name': stock.company_name,
                'ROCE (%)': round(stock.roce, 2),
                'Earning Yield (%)': round(stock.earning_yield, 2),
                'WACC (%)': round(stock.wacc, 2),
                'Market Cap (Cr)': round(stock.market_cap/10000000, 0),
                'Combined Rank': stock.combined_rank,
                'ROCE vs WACC': f"+{round(stock.roce - stock.wacc, 2)}%",
                'EY vs Bond+4%': f"+{round(stock.earning_yield - (self.indian_bond_yield + 4.0), 2)}%"
            })
        
        passed_df = pd.DataFrame(passed_data) if screened_stocks else pd.DataFrame()
        
        # File paths
        csv_file = os.path.join(output_dir, f"{file_prefix}_complete_analysis_{timestamp}.csv")
        excel_file = os.path.join(output_dir, f"{file_prefix}_complete_analysis_{timestamp}.xlsx")
        
        # PRIORITY: Save Excel file with multiple sheets
        excel_saved = False
        excel_engines = ['xlsxwriter', 'openpyxl']
        
        for engine in excel_engines:
            try:
                print(f"Attempting to save Excel using {engine} engine...")
                
                with pd.ExcelWriter(excel_file, engine=engine) as writer:
                    # Sheet 1: ALL STOCKS ANALYZED
                    all_df.to_excel(writer, sheet_name='All Stocks Analyzed', index=False)
                    
                    # Sheet 2: PASSED STOCKS ONLY (if any)
                    if not passed_df.empty:
                        passed_df.to_excel(writer, sheet_name='Passed Magic Formula', index=False)
                    
                    # Sheet 3: FAILED STOCKS ONLY
                    failed_df = all_df[all_df['Passed Screening'] == 'NO']
                    if not failed_df.empty:
                        failed_df.to_excel(writer, sheet_name='Failed Screening', index=False)
                    
                    # Sheet 4: SUMMARY STATISTICS
                    summary_data = {
                        'Metric': [
                            'Total Stocks Analyzed',
                            'Stocks Passed Screening',
                            'Stocks Failed Screening',
                            'Pass Rate (%)',
                            'Average ROCE - All Stocks (%)',
                            'Average EY - All Stocks (%)',
                            'Average WACC - All Stocks (%)',
                            'Bond Yield Used (%)',
                            'EY Threshold (%)',
                            'Median Market Cap - All (Cr)',
                            'Highest ROCE (%)',
                            'Highest EY (%)',
                            'Main Failure Reason'
                        ],
                        'Value': [
                            len(all_stocks),
                            len(screened_stocks),
                            len(all_stocks) - len(screened_stocks),
                            round((len(screened_stocks) / len(all_stocks)) * 100, 2) if all_stocks else 0,
                            round(all_df['ROCE (%)'].mean(), 2),
                            round(all_df['Earning Yield (%)'].mean(), 2),
                            round(all_df['WACC (%)'].mean(), 2),
                            self.indian_bond_yield,
                            round(self.indian_bond_yield + 4.0, 2),
                            round(all_df['Market Cap (Cr)'].median(), 0),
                            round(all_df['ROCE (%)'].max(), 2),
                            round(all_df['Earning Yield (%)'].max(), 2),
                            all_df['Failure Reason'].value_counts().index[0] if len(all_df[all_df['Passed Screening'] == 'NO']) > 0 else 'N/A'
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
                    
                    # Apply formatting if using xlsxwriter
                    if engine == 'xlsxwriter':
                        workbook = writer.book
                        
                        # Format All Stocks sheet
                        worksheet_all = writer.sheets['All Stocks Analyzed']
                        header_format = workbook.add_format({
                            'bold': True,
                            'text_wrap': True,
                            'valign': 'top',
                            'fg_color': '#D7E4BC',
                            'border': 1
                        })
                        
                        # Color code passed/failed
                        pass_format = workbook.add_format({'fg_color': '#C6EFCE'})  # Light green
                        fail_format = workbook.add_format({'fg_color': '#FFC7CE'})  # Light red
                        
                        # Apply header formatting
                        for col_num, value in enumerate(all_df.columns.values):
                            worksheet_all.write(0, col_num, value, header_format)
                            worksheet_all.set_column(col_num, col_num, 15)  # Set column width
                
                print(f"✅ COMPLETE EXCEL ANALYSIS SAVED: {excel_file}")
                print(f"📊 Excel engine used: {engine}")
                excel_saved = True
                break
                
            except ImportError:
                print(f"⚠️  {engine} not available")
                continue
            except Exception as e:
                print(f"❌ Error with {engine}: {e}")
                continue
        
        # Save CSV backup of all data
        all_df.to_csv(csv_file, index=False)
        print(f"📄 Complete CSV saved: {csv_file}")
        
        if not excel_saved:
            print(f"⚠️  Excel save failed. Install: pip install xlsxwriter openpyxl")
        
        # Print summary
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"Total stocks analyzed: {len(all_stocks)}")
        print(f"Stocks passed screening: {len(screened_stocks)}")
        print(f"Stocks failed screening: {len(all_stocks) - len(screened_stocks)}")
        print(f"Pass rate: {(len(screened_stocks) / len(all_stocks)) * 100:.1f}%")
        
        print(f"\n🎯 MAIN OUTPUT: {excel_file if excel_saved else csv_file}")
        
        return all_df

def get_nifty_100_stocks():
    """
    Get NSE Nifty 100 stock symbols
    Returns list of top 100 NSE stocks by market cap
    """
    nifty_100_stocks = [
        # Nifty 50 stocks
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS',
        'MARUTI.NS', 'LT.NS', 'AXISBANK.NS', 'SUNPHARMA.NS', 'TITAN.NS',
        'ULTRACEMCO.NS', 'NESTLEIND.NS', 'KOTAKBANK.NS', 'NTPC.NS', 'HCLTECH.NS',
        'BAJFINANCE.NS', 'M&M.NS', 'WIPRO.NS', 'JSWSTEEL.NS', 'POWERGRID.NS',
        'TATASTEEL.NS', 'TECHM.NS', 'INDUSINDBK.NS', 'COALINDIA.NS', 'HINDALCO.NS',
        'ADANIPORTS.NS', 'BAJAJFINSV.NS', 'CIPLA.NS', 'DRREDDY.NS', 'GRASIM.NS',
        'APOLLOHOSP.NS', 'EICHERMOT.NS', 'BPCL.NS', 'BRITANNIA.NS', 'TATACONSUM.NS',
        'DIVISLAB.NS', 'SHREECEM.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'ONGC.NS',
        'ADANIENT.NS', 'TATAMOTORS.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'UPL.NS',
        
        # Nifty Next 50 stocks (completing Nifty 100)
        'GODREJCP.NS', 'SIEMENS.NS', 'PIDILITIND.NS', 'COLPAL.NS', 'DLF.NS',
        'MARICO.NS', 'TORNTPHARM.NS', 'BOSCHLTD.NS', 'DABUR.NS', 'LUPIN.NS',
        'BERGEPAINT.NS', 'GAIL.NS', 'BANKBARODA.NS', 'ADANIGREEN.NS', 'MOTHERSON.NS',
        'SRF.NS', 'AMBUJACEM.NS', 'HAVELLS.NS', 'DMART.NS', 'CONCOR.NS',
        'BIOCON.NS', 'PGHH.NS', 'MCDOWELL-N.NS', 'BAJAJHLDNG.NS', 'SAIL.NS',
        'ALKEM.NS', 'OFSS.NS', 'NMDC.NS', 'BANDHANBNK.NS', 'MUTHOOTFIN.NS',
        'ACC.NS', 'PAGEIND.NS', 'GODREJPROP.NS', 'CUMMINSIND.NS', 'CADILAHC.NS',
        'RECLTD.NS', 'PFC.NS', 'AUROPHARMA.NS', 'MPHASIS.NS', 'TORNTPOWER.NS',
        'L&TFH.NS', 'BATAINDIA.NS', 'ASHOKLEY.NS', 'INDIGO.NS', 'ABCAPITAL.NS',
        'TATAPOWER.NS', 'LICHSGFIN.NS', 'MANAPPURAM.NS', 'CHOLAFIN.NS', 'INDUSTOWER.NS'
    ]
    
    return nifty_100_stocks

def get_nifty_200_stocks():
    """
    Extended list including Nifty 200 for broader analysis
    """
    nifty_200_additional = [
        'IDFCFIRSTB.NS', 'FEDERALBNK.NS', 'PEL.NS', 'VOLTAS.NS', 'MINDTREE.NS',
        'JUBLFOOD.NS', 'CROMPTON.NS', 'LALPATHLAB.NS', 'ESCORTS.NS', 'RBLBANK.NS',
        'PERSISTENT.NS', 'IPCALAB.NS', 'RELAXO.NS', 'RAMCOCEM.NS', 'HONAUT.NS',
        'TRENT.NS', 'POLYCAB.NS', 'ASTRAL.NS', 'BALKRISIND.NS', 'STARHEALTH.NS',
        'NAUKRI.NS', 'LAURUSLABS.NS', 'DEEPAKNTR.NS', 'COFORGE.NS', 'PFIZER.NS',
        'SYNGENE.NS', 'CANBK.NS', 'GMRINFRA.NS', 'ZYDUSLIFE.NS', 'JINDALSTEL.NS',
        'APOLLOTYRE.NS', 'CHAMBLFERT.NS', 'BHEL.NS', 'IDEA.NS', 'PNB.NS',
        'IOC.NS', 'ICICIPRULI.NS', 'SRTRANSFIN.NS', 'EXIDEIND.NS', 'FLUOROCHEM.NS',
        'PIIND.NS', 'WHIRLPOOL.NS', 'JKCEMENT.NS', 'ZEEL.NS', 'UNIONBANK.NS',
        'METROPOLIS.NS', 'CUB.NS', 'ADANIPOWER.NS', 'ABBOTINDIA.NS', 'DIXON.NS'
    ]
    
    return get_nifty_100_stocks() + nifty_200_additional

# Example usage
def main():
    """
    Main function to demonstrate the Magic Formula screener on NSE Top 100
    """
    print("=== Magic Formula Stock Screener - NSE Top 100 Analysis ===\n")
    
    # Initialize screener
    screener = MagicFormulaScreener()
    
    # Get NSE Top 100 stocks
    print("📊 Analysis Options:")
    print("1. Nifty 100 stocks (100 stocks)")
    print("2. Nifty 200 stocks (150 stocks)")
    print("3. Custom stock list")
    
    # You can change this to 2 for Nifty 200 analysis
    analysis_choice = 1
    
    if analysis_choice == 1:
        stock_universe = get_nifty_100_stocks()
        analysis_name = "Nifty 100"
        file_prefix = "nifty_100_magic_formula"
    elif analysis_choice == 2:
        stock_universe = get_nifty_200_stocks()
        analysis_name = "Nifty 200"
        file_prefix = "nifty_200_magic_formula"
    else:
        # Custom list - you can modify this
        stock_universe = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS'
        ]
        analysis_name = "Custom Selection"
        file_prefix = "custom_magic_formula"
    
    print(f"\n🎯 Selected: {analysis_name}")
    print(f"📈 Total stocks to analyze: {len(stock_universe)}")
    print(f"⏱️  Estimated time: {len(stock_universe) * 3} seconds")
    print(f"💡 This will analyze ALL stocks regardless of filter results\n")
    
    input("Press Enter to start analysis...")
    
    # Screen stocks with relaxed filters for broader analysis
    screened_stocks, all_stocks = screener.screen_stocks(
        stock_symbols=stock_universe,
        ey_margin=4.0,  # EY > Bond Yield + 4%
        min_market_cap=5000000000  # 500 Cr minimum (lower for broader analysis)
    )
    
    # Generate comprehensive report with ALL data
    print(f"\n=== {analysis_name.upper()} ANALYSIS RESULTS ===")
    print(f"Total stocks analyzed: {len(all_stocks)}")
    print(f"Stocks passed screening: {len(screened_stocks)}")
    print(f"Stocks failed screening: {len(all_stocks) - len(screened_stocks)}")
    print(f"Pass rate: {(len(screened_stocks) / len(all_stocks)) * 100:.1f}%" if all_stocks else "0%")
    print(f"Bond Yield used: {screener.indian_bond_yield}%")
    print(f"EY Threshold: {screener.indian_bond_yield + 4.0}%")
    print(f"Filters Applied: ROCE > WACC, EY > Bond Yield + 4%, Market Cap > 500 Cr\n")
    
    # Generate comprehensive reports including failed stocks
    df = screener.generate_report(
        screened_stocks, 
        all_stocks,
        output_dir="nse_analysis_results",
        file_prefix=file_prefix
    )
    
    if screened_stocks:
        print(f"\n=== TOP 10 {analysis_name.upper()} MAGIC FORMULA PICKS ===")
        for i, stock in enumerate(screened_stocks[:10], 1):
            print(f"{i:2d}. {stock.company_name} ({stock.symbol})")
            print(f"     ROCE: {stock.roce:5.1f}% | EY: {stock.earning_yield:5.1f}% | Rank: {stock.combined_rank}")
    
    # Show failure analysis
    failed_stocks = [stock for stock in all_stocks if not stock.passed_screening]
    if failed_stocks:
        print(f"\n=== FAILURE ANALYSIS ===")
        failure_reasons = {}
        for stock in failed_stocks:
            reason = stock.filter_failure_reason
            if "EY" in reason and "Bond Yield" in reason:
                key = "Low Earnings Yield"
            elif "ROCE" in reason and "WACC" in reason:
                key = "ROCE <= WACC"
            elif "Market cap" in reason:
                key = "Small Market Cap"
            elif "Data fetch failed" in reason:
                key = "Data Unavailable"
            else:
                key = "Other"
            
            failure_reasons[key] = failure_reasons.get(key, 0) + 1
        
        print("Most common failure reasons:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(failed_stocks)) * 100
            print(f"  • {reason}: {count} stocks ({percentage:.1f}%)")
    
    # Sector analysis if enough stocks
    if len(all_stocks) > 20:
        print(f"\n=== SECTOR INSIGHTS ===")
        print("Top performers by ROCE:")
        top_roce = sorted([s for s in all_stocks if s.roce > 0], key=lambda x: x.roce, reverse=True)[:5]
        for i, stock in enumerate(top_roce, 1):
            print(f"  {i}. {stock.symbol}: {stock.roce:.1f}% ROCE")
        
        print("\nTop performers by Earnings Yield:")
        top_ey = sorted([s for s in all_stocks if s.earning_yield > 0], key=lambda x: x.earning_yield, reverse=True)[:5]
        for i, stock in enumerate(top_ey, 1):
            print(f"  {i}. {stock.symbol}: {stock.earning_yield:.1f}% EY")
    
    print(f"\n📊 Complete {analysis_name} analysis saved with ALL stock data!")
    print(f"💡 Check Excel file for detailed breakdown of passed and failed stocks.")
    print(f"📁 Results saved in: nse_analysis_results/ folder")

if __name__ == "__main__":
    main()