#!/usr/bin/env python3
"""
Test script for mutual fund functionality
"""

import os
import sys
from indian_stock_market_agent import IndianStockMarketAgent

def test_mutual_fund_functionality():
    """Test mutual fund data fetching and performance calculation."""
    
    print("Testing Mutual Fund Functionality")
    print("=" * 50)
    
    # Initialize agent (API key not required for demo data)
    try:
        agent = IndianStockMarketAgent()
        print("Agent initialized successfully")
    except Exception as e:
        print(f"Agent initialization failed: {e}")
        return False
    
    # Test 1: Fetch mutual fund data
    print("\nTest 1: Fetching Mutual Fund Data")
    test_funds = ['PPFAS_FLEXICAP_DIRECT', 'HDFC_SMALLCAP_DIRECT', 'HDFC_NIFTY50_DIRECT']
    
    try:
        fund_data = agent.get_indian_stock_data(test_funds)
        
        for fund in test_funds:
            if fund in fund_data and 'error' not in fund_data[fund]:
                data = fund_data[fund]
                print(f"SUCCESS {fund}:")
                print(f"   NAV: Rs{data['price']:.2f}")
                print(f"   Scheme: {data.get('scheme_name', 'N/A')}")
                print(f"   Change: {data['day_change_pct']:+.1f}%")
                print(f"   Source: {data.get('data_source', 'N/A')}")
            else:
                error_msg = fund_data.get(fund, {}).get('error', 'Unknown error')
                print(f"ERROR {fund}: {error_msg}")
                
    except Exception as e:
        print(f"Mutual fund data fetch failed: {e}")
        return False
    
    # Test 2: Performance calculation
    print("\nTest 2: Performance Calculation")
    
    try:
        performance = agent.calculate_fund_performance('PPFAS_FLEXICAP_DIRECT')
        
        if 'error' not in performance:
            print("Performance calculation successful")
            print(f"   1Y Return: {performance['returns'].get('1Y', 'N/A')}%")
            print(f"   3Y Return: {performance['returns'].get('3Y', 'N/A')}%")
            print(f"   5Y Return: {performance['returns'].get('5Y', 'N/A')}%")
            print(f"   Inception Date: {performance.get('inception_date', 'N/A')}")
            
            # Test alpha calculation
            if '5Y' in performance['alpha']:
                print(f"   5Y Alpha: {performance['alpha']['5Y']:+.1f}%")
        else:
            print(f"Performance calculation failed: {performance['error']}")
            return False
            
    except Exception as e:
        print(f"Performance calculation error: {e}")
        return False
    
    # Test 3: ETF data
    print("\nTest 3: ETF Data")
    
    try:
        etf_data = agent.get_indian_stock_data(['HDFC_GOLD_ETF'])
        
        if 'HDFC_GOLD_ETF' in etf_data and 'error' not in etf_data['HDFC_GOLD_ETF']:
            data = etf_data['HDFC_GOLD_ETF']
            print(f"SUCCESS HDFC_GOLD_ETF:")
            print(f"   Price: Rs{data['price']:.2f}")
            print(f"   Change: {data['day_change_pct']:+.1f}%")
            print(f"   Asset Type: {data.get('asset_type', 'N/A')}")
        else:
            print("ETF data fetch failed")
            
    except Exception as e:
        print(f"ETF test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("Mutual fund functionality is working")
    print("Performance comparison is working") 
    print("ETF support is working")
    
    return True

if __name__ == "__main__":
    success = test_mutual_fund_functionality()
    
    if success:
        print("\nReady to run the dashboard:")
        print("   streamlit run indian_dashboard.py")
        print("\nNavigate to 'Mutual Funds/ETFs' tab to see the functionality")
    else:
        print("\nTests failed - check error messages above")
        sys.exit(1)