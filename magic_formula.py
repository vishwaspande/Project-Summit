# (once) install deps
py -m pip install yfinance pandas numpy

# default run (built-in NSE universe)
py magic_formula.py

# recommended: exclude financials from ROCE math, add a ROCE floor, custom output
py magic_formula.py --exclude_financials --min_roce 0.12 --out data\summit_magic.csv

# custom tickers & risk inputs
py magic_formula.py --tickers "TCS.NS,INFY.NS,ITC.NS,LT.NS" --rf 0.065 --mrp 0.065 --ey_spread 0.03 --exclude_financials --debug
