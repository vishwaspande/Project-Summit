#!/usr/bin/env python3
"""
Screener.in Data Fetcher for Super Investor Stocks
====================================================
Scrapes REAL financial data from screener.in for all 360 stocks.

USAGE:
  pip install requests beautifulsoup4
  python fetch_real_data.py

This will:
1. Read your stock list from Comprehensive_Super_Investor_Comparison.csv
2. Scrape screener.in for each stock's financials
3. Save results to data.js (for dashboard) and stock_data.csv
4. Rebuild dashboard_standalone.html with real data

Place this script in the same folder as your CSV file and run it.
"""

import csv
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install requests beautifulsoup4")
    import requests
    from bs4 import BeautifulSoup


# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "Comprehensive_Super_Investor_Comparison.csv"
CACHE_DIR = SCRIPT_DIR / "cache"
OUTPUT_DATA_JS = SCRIPT_DIR / "data.js"
OUTPUT_CSV = SCRIPT_DIR / "stock_data.csv"
OUTPUT_HTML = SCRIPT_DIR / "dashboard_standalone.html"
INDEX_HTML = SCRIPT_DIR / "index.html"

CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

DELAY = 1.5  # seconds between requests to be polite


# ── Screener.in Scraper ────────────────────────────────────────────────────
def fetch_screener_data(code, stock_name):
    """Fetch financial data from screener.in for a given stock code."""
    
    # Try consolidated first, then standalone
    for suffix in ["/consolidated/", "/"]:
        url = f"https://www.screener.in/company/{code}{suffix}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return parse_screener_page(resp.text, code, stock_name)
        except requests.exceptions.RequestException as e:
            pass
    
    # Try with stock name search
    try:
        search_url = f"https://www.screener.in/api/company/search/?q={code}"
        resp = requests.get(search_url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            results = resp.json()
            if results:
                slug = results[0].get("url", "")
                if slug:
                    url = f"https://www.screener.in{slug}"
                    resp2 = requests.get(url, headers=HEADERS, timeout=15)
                    if resp2.status_code == 200:
                        return parse_screener_page(resp2.text, code, stock_name)
    except Exception:
        pass
    
    return None


def safe_float(text):
    """Extract numeric value from text."""
    if not text:
        return None
    text = str(text).strip().replace(",", "").replace("₹", "").replace("%", "")
    text = text.strip()
    if text in ["", "-", "—", "N/A", "NA"]:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_screener_page(html, code, stock_name):
    """Parse screener.in company page and extract key metrics."""
    soup = BeautifulSoup(html, "html.parser")
    data = {"stock": stock_name, "code": code}
    
    # ── Extract from top ratios section ──
    # Look for the key-value pairs in the company ratios
    li_items = soup.select("#top-ratios li, .company-ratios li, .ratios-table li")
    
    ratio_map = {}
    for li in li_items:
        name_el = li.select_one(".name")
        value_el = li.select_one(".value, .number")
        if name_el and value_el:
            name = name_el.get_text(strip=True).lower()
            value = value_el.get_text(strip=True)
            ratio_map[name] = value
    
    # Also try alternate layout
    for span in soup.select("span.name, span.nowrap"):
        text = span.get_text(strip=True).lower()
        next_el = span.find_next_sibling("span") or span.find_next("span")
        if next_el:
            ratio_map[text] = next_el.get_text(strip=True)
    
    # Try to get data from the page text directly using regex
    page_text = soup.get_text()
    
    # Market Cap
    match = re.search(r'Market\s*Cap\s*₹?\s*([\d,]+(?:\.\d+)?)\s*Cr', page_text)
    if match:
        data["market_cap_cr"] = safe_float(match.group(1))
    
    # Current Price
    match = re.search(r'Current\s*Price\s*₹?\s*([\d,]+(?:\.\d+)?)', page_text)
    if match:
        data["current_price"] = safe_float(match.group(1))
    
    # Stock P/E
    match = re.search(r'Stock\s*P/?E\s*([\d,]+(?:\.\d+)?)', page_text)
    if match:
        data["pe_ratio"] = safe_float(match.group(1))
    
    # Book Value
    match = re.search(r'Book\s*Value\s*₹?\s*([\d,]+(?:\.\d+)?)', page_text)
    if match:
        data["book_value"] = safe_float(match.group(1))
    
    # Dividend Yield
    match = re.search(r'Dividend\s*Yield\s*([\d.]+)\s*%', page_text)
    if match:
        data["dividend_yield"] = safe_float(match.group(1))
    
    # ROCE
    match = re.search(r'ROCE\s*([\-\d.]+)\s*%', page_text)
    if match:
        data["roce"] = safe_float(match.group(1))
    
    # ROE
    match = re.search(r'ROE\s*([\-\d.]+)\s*%', page_text)
    if match:
        data["roe"] = safe_float(match.group(1))
    
    # High / Low
    match = re.search(r'High\s*/\s*Low\s*₹?\s*([\d,]+(?:\.\d+)?)\s*/\s*([\d,]+(?:\.\d+)?)', page_text)
    if match:
        data["high_52w"] = safe_float(match.group(1))
        data["low_52w"] = safe_float(match.group(2))
    
    # Debt to Equity
    match = re.search(r'Debt\s*to\s*Equity\s*([\d.]+)', page_text, re.IGNORECASE)
    if match:
        data["debt_to_equity"] = safe_float(match.group(1))
    
    # P/B ratio = price / book_value
    if data.get("current_price") and data.get("book_value") and data["book_value"] > 0:
        data["pb_ratio"] = round(data["current_price"] / data["book_value"], 2)
    
    # EPS = price / PE
    if data.get("current_price") and data.get("pe_ratio") and data["pe_ratio"] > 0:
        data["eps"] = round(data["current_price"] / data["pe_ratio"], 2)
    
    # ── Try to get growth data from tables ──
    # Look for compounded growth rates
    match = re.search(r'Compounded\s*Sales\s*Growth.*?10\s*Years:?\s*([\-\d.]+)\s*%', page_text, re.DOTALL)
    
    # Revenue growth (look for recent quarters)
    match = re.search(r'Revenue\s*Growth.*?([\-\d.]+)\s*%', page_text)
    if match:
        data["revenue_growth"] = safe_float(match.group(1))
    
    # EPS growth
    match = re.search(r'EPS\s*Growth.*?([\-\d.]+)\s*%', page_text)  
    if match:
        data["eps_growth"] = safe_float(match.group(1))
    
    # ── Shareholding ──
    match = re.search(r'Promoters?\s*(\d+\.?\d*)\s*%', page_text)
    if match:
        data["promoter_holding"] = safe_float(match.group(1))
    
    # Pledging
    match = re.search(r'[Pp]ledg\w*\s*([\d.]+)\s*%', page_text)
    if match:
        data["promoter_pledge"] = safe_float(match.group(1))
    else:
        data["promoter_pledge"] = 0
    
    # ── Sector/Industry ──
    breadcrumb = soup.select(".company-links a, .sub-links a, .breadcrumb a")
    sectors = [a.get_text(strip=True) for a in breadcrumb if a.get_text(strip=True) not in ["", "Home", stock_name, code]]
    if sectors:
        data["sector"] = sectors[-1] if sectors else "Unknown"
    
    # Try to get Net Income from P&L tables
    # Look for "Net Profit" row
    match = re.search(r'Net\s*Profit\s*(?:₹?\s*)?([\-\d,]+(?:\.\d+)?)', page_text)
    if match:
        data["net_income_cr"] = safe_float(match.group(1))
    
    # Set defaults for missing fields
    for key in ["market_cap_cr", "current_price", "pe_ratio", "pb_ratio", "book_value",
                 "roe", "roce", "dividend_yield", "debt_to_equity", "eps", "eps_growth",
                 "revenue_growth", "high_52w", "low_52w", "promoter_holding", "promoter_pledge",
                 "fcf_cr", "net_income_cr", "sector", "peg_ratio", "ev_ebitda"]:
        if key not in data:
            data[key] = None
    
    # Calculate PEG if we have P/E and growth
    if data.get("pe_ratio") and data.get("eps_growth") and data["eps_growth"] > 0:
        data["peg_ratio"] = round(data["pe_ratio"] / data["eps_growth"], 2)
    
    return data


# ── Scoring Engine (same as before) ────────────────────────────────────────
def score_stock(m, investor_count):
    scores = {}
    weights = {}
    
    def rate(category, name, value, weight, lower_better=False, lo=None, hi=None):
        if value is None:
            return
        if category not in scores:
            scores[category] = 0
            weights[category] = 0
        weights[category] += weight
        if lower_better:
            if value <= lo: s = weight
            elif value >= hi: s = 0
            else: s = weight * (1 - (value - lo) / (hi - lo))
        else:
            if value >= hi: s = weight
            elif value <= lo: s = 0
            else: s = weight * (value - lo) / (hi - lo)
        scores[category] = scores.get(category, 0) + max(0, min(weight, s))
    
    rate("Valuation", "P/E", m.get("pe_ratio"), 8, lower_better=True, lo=5, hi=50)
    rate("Valuation", "PEG", m.get("peg_ratio"), 6, lower_better=True, lo=0.3, hi=3)
    rate("Valuation", "EV/EBITDA", m.get("ev_ebitda"), 5, lower_better=True, lo=5, hi=30)
    rate("Valuation", "P/B", m.get("pb_ratio"), 4, lower_better=True, lo=0.5, hi=8)
    
    pe = m.get("pe_ratio")
    if pe and pe > 0:
        rate("Valuation", "EY", 100.0 / pe, 2, lo=2, hi=15)
    
    rate("Quality", "ROE", m.get("roe"), 10, lo=5, hi=25)
    rate("Quality", "ROCE", m.get("roce"), 10, lo=8, hi=30)
    fcf = m.get("fcf_cr")
    ni = m.get("net_income_cr")
    if fcf is not None:
        if fcf > 0: scores["Quality"] = scores.get("Quality", 0) + 3
        weights["Quality"] = weights.get("Quality", 0) + 3
        if ni and ni > 0 and fcf > 0:
            rate("Quality", "FCF/PAT", fcf / ni, 2, lo=0.3, hi=1.2)
    
    rate("Growth", "EPS Growth", m.get("eps_growth"), 10, lo=0, hi=30)
    rate("Growth", "Revenue Growth", m.get("revenue_growth"), 10, lo=0, hi=30)
    
    rate("Safety", "D/E", m.get("debt_to_equity"), 8, lower_better=True, lo=0, hi=2)
    rate("Safety", "Promoter", m.get("promoter_holding"), 6, lo=25, hi=70)
    rate("Safety", "Pledge", m.get("promoter_pledge"), 4, lower_better=True, lo=0, hi=25)
    rate("Safety", "Dividend", m.get("dividend_yield"), 2, lo=0, hi=4)
    
    hi52 = m.get("high_52w")
    lo52 = m.get("low_52w")
    cur = m.get("current_price")
    if hi52 and lo52 and cur and hi52 > lo52:
        rate("Momentum", "52W", 1 - (cur - lo52) / (hi52 - lo52), 5, lo=0, hi=0.6)
    
    scores["Conviction"] = min(5, investor_count * 1.25)
    weights["Conviction"] = 5
    
    total_score = sum(scores.values())
    total_weight = sum(weights.values())
    final = (total_score / total_weight) * 100 if total_weight > 0 else 0
    
    return round(min(100, final), 1), {
        "category_scores": {k: round(v, 1) for k, v in scores.items()},
        "category_weights": {k: round(v, 1) for k, v in weights.items()},
        "data_completeness": f"{sum(1 for v in m.values() if v is not None)}/{len(m)}",
    }


def classify_stock(m, score):
    tags = []
    pe = m.get("pe_ratio")
    pb = m.get("pb_ratio")
    roe = m.get("roe")
    roce = m.get("roce")
    epsg = m.get("eps_growth")
    revg = m.get("revenue_growth")
    de = m.get("debt_to_equity")
    dy = m.get("dividend_yield")
    peg = m.get("peg_ratio")
    
    if pe and pe < 15 and pb and pb < 2: tags.append("Deep Value")
    elif pe and pe < 20: tags.append("Value")
    if epsg and epsg > 20 and revg and revg > 15: tags.append("High Growth")
    elif epsg and epsg > 10: tags.append("Growth")
    if roe and roe > 20 and roce and roce > 20: tags.append("High Quality")
    if de is not None and de < 0.3: tags.append("Low Debt")
    if dy and dy > 2: tags.append("Dividend")
    if pe and pe < 0: tags.append("Loss-Making")
    if peg and 0.5 < peg < 1.5 and roe and roe > 15: tags.append("GARP")
    return tags if tags else ["Unclassified"]


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SCREENER.IN DATA FETCHER")
    print("  Fetching REAL data for all stocks")
    print("=" * 60)
    
    # Find CSV file
    csv_path = CSV_FILE
    if not csv_path.exists():
        # Try current directory
        for f in Path(".").glob("*.csv"):
            if "investor" in f.name.lower() or "super" in f.name.lower() or "comprehensive" in f.name.lower():
                csv_path = f
                break
    
    if not csv_path.exists():
        print(f"ERROR: Cannot find CSV file at {CSV_FILE}")
        print("Place this script in the same folder as your CSV file.")
        sys.exit(1)
    
    # Load stocks
    stocks = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stocks.append({
                "stock": row["Stock"].strip(),
                "code": row["Code"].strip(),
                "investor": row["Investor"].strip(),
                "investor_count": int(row["Investor Count"].strip() or 0),
            })
    
    print(f"\n📊 Loaded {len(stocks)} stocks")
    
    # Check cache
    cached = {}
    for fn in CACHE_DIR.glob("*.json"):
        with open(fn) as f:
            try:
                d = json.load(f)
                cached[fn.stem] = d
            except:
                pass
    
    def cache_key(code):
        return re.sub(r'[^a-zA-Z0-9]', '_', code)
    
    already = sum(1 for s in stocks if cache_key(s["code"]) in cached)
    print(f"💾 Cached: {already} | To fetch: {len(stocks) - already}")
    
    # Fetch missing stocks
    fetched = 0
    failed = []
    for i, s in enumerate(stocks):
        ck = cache_key(s["code"])
        if ck in cached:
            continue
        
        # Only fetch for alpha codes (numeric BSE codes won't work on screener.in)
        code = s["code"]
        if code.isdigit():
            print(f"  ⏭ {s['stock']} ({code}) - numeric BSE code, skipping screener.in")
            failed.append(s)
            continue
        
        print(f"  [{i+1}/{len(stocks)}] Fetching {s['stock']} ({code})...", end="", flush=True)
        
        try:
            data = fetch_screener_data(code, s["stock"])
            if data and data.get("current_price"):
                cache_path = CACHE_DIR / f"{ck}.json"
                with open(cache_path, "w") as f:
                    json.dump(data, f, indent=2)
                cached[ck] = data
                fetched += 1
                mcap = data.get("market_cap_cr", "?")
                pe = data.get("pe_ratio", "?")
                print(f" ✓ MCap={mcap} PE={pe}")
            else:
                print(f" ✗ No data found")
                failed.append(s)
        except Exception as e:
            print(f" ✗ Error: {e}")
            failed.append(s)
        
        time.sleep(DELAY)
    
    print(f"\n✅ Fetched: {fetched} | Failed: {len(failed)}")
    if failed:
        print(f"Failed stocks: {', '.join(s['stock'] for s in failed[:20])}")
    
    # Score all stocks
    print("\n📈 Scoring all stocks...")
    results = []
    for s in stocks:
        ck = cache_key(s["code"])
        metrics = cached.get(ck, {})
        
        if metrics and metrics.get("current_price"):
            score, breakdown = score_stock(metrics, s["investor_count"])
            tags = classify_stock(metrics, score)
        else:
            score, breakdown, tags = 0, {}, ["No Data"]
        
        results.append({
            "stock": s["stock"],
            "code": s["code"],
            "score": score,
            "tags": tags,
            "investor": s["investor"][:100],
            "inv_count": s["investor_count"],
            "price": metrics.get("current_price"),
            "mcap": metrics.get("market_cap_cr"),
            "pe": metrics.get("pe_ratio"),
            "pb": metrics.get("pb_ratio"),
            "roe": metrics.get("roe"),
            "roce": metrics.get("roce"),
            "de": metrics.get("debt_to_equity"),
            "dy": metrics.get("dividend_yield"),
            "epsg": metrics.get("eps_growth"),
            "revg": metrics.get("revenue_growth"),
            "hi52": metrics.get("high_52w"),
            "lo52": metrics.get("low_52w"),
            "promo": metrics.get("promoter_holding"),
            "pledge": metrics.get("promoter_pledge"),
            "fcf": metrics.get("fcf_cr"),
            "ni": metrics.get("net_income_cr"),
            "sector": metrics.get("sector"),
            "peg": metrics.get("peg_ratio"),
            "bv": metrics.get("book_value"),
            "breakdown": breakdown,
        })
    
    results.sort(key=lambda x: (-x["score"], -x["inv_count"]))
    
    # ── Save data.js ──
    minified = json.dumps(results, separators=(",", ":"))
    with open(OUTPUT_DATA_JS, "w") as f:
        f.write("const STOCKS_DATA = " + minified + ";")
    print(f"\n📁 Saved: {OUTPUT_DATA_JS}")
    
    # ── Save CSV ──
    fields = ["Rank", "Stock", "Code", "Score", "Tags", "Investor", "InvCount",
              "Price", "MCap_Cr", "PE", "PB", "ROE", "ROCE", "DE", "DivYld",
              "EPS_Gr", "Rev_Gr", "Promoter", "Pledge", "Sector"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(results):
            w.writerow({
                "Rank": i+1, "Stock": r["stock"], "Code": r["code"],
                "Score": r["score"], "Tags": ", ".join(r["tags"]),
                "Investor": r["investor"], "InvCount": r["inv_count"],
                "Price": r["price"] or "", "MCap_Cr": r["mcap"] or "",
                "PE": r["pe"] or "", "PB": r["pb"] or "",
                "ROE": r["roe"] or "", "ROCE": r["roce"] or "",
                "DE": r["de"] or "", "DivYld": r["dy"] or "",
                "EPS_Gr": r["epsg"] or "", "Rev_Gr": r["revg"] or "",
                "Promoter": r["promo"] or "", "Pledge": r["pledge"] or "",
                "Sector": r["sector"] or "",
            })
    print(f"📁 Saved: {OUTPUT_CSV}")
    
    # ── Rebuild standalone HTML ──
    if INDEX_HTML.exists():
        with open(INDEX_HTML) as f:
            html = f.read()
        standalone = html.replace(
            '<script src="data.js"></script>',
            "<script>const STOCKS_DATA = " + minified + ";</script>"
        )
        with open(OUTPUT_HTML, "w") as f:
            f.write(standalone)
        print(f"📁 Saved: {OUTPUT_HTML}")
    
    # ── Print top 20 ──
    scored = [r for r in results if r["score"] > 0]
    print(f"\n{'='*60}")
    print(f"  TOP 20 STOCKS (out of {len(scored)} scored)")
    print(f"{'='*60}")
    fmt = "{:<4} {:<28} {:<7} {:<4} {:<8} {:<7} {:<7} {:<6}"
    print(fmt.format("#", "Stock", "Score", "Inv", "P/E", "ROE%", "ROCE%", "D/E"))
    print("-" * 60)
    for i, r in enumerate(results[:20]):
        def fv(v): return f"{v:.1f}" if isinstance(v, (int, float)) and v is not None else "-"
        print(fmt.format(i+1, r["stock"][:27], r["score"], r["inv_count"],
                         fv(r["pe"]), fv(r["roe"]), fv(r["roce"]), fv(r["de"])))
    
    print(f"\n✅ Done! Open dashboard_standalone.html in your browser.")
    print(f"   Or double-click run_dashboard.bat")


if __name__ == "__main__":
    main()
