#!/usr/bin/env python3
"""
Super Investor Stock Analysis Agent v2
=======================================
Uses Anthropic Claude API + web_search tool to fetch latest financials
for Indian stocks from screener.in, then scores and ranks them.

Usage:
  python3 stock_agent_v2.py               # Run all stocks
  python3 stock_agent_v2.py --batches 5   # Run 5 batches only
  python3 stock_agent_v2.py --score-only  # Score cached data only
"""

import csv, json, os, sys, time, re, math, argparse
from datetime import datetime
from pathlib import Path
import urllib.request

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_CSV = "/mnt/user-data/uploads/Comprehensive_Super_Investor_Comparison.csv"
OUTPUT_DIR = "/mnt/user-data/outputs"
CACHE_DIR = "/home/claude/cache"
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"
BATCH_SIZE = 8  # stocks per API call for efficiency

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load CSV ────────────────────────────────────────────────────────────────
def load_stocks():
    stocks = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stocks.append({
                "stock": row["Stock"].strip(),
                "code": row["Code"].strip(),
                "investor": row["Investor"].strip(),
                "qty_held": row["Qty Held"].strip(),
                "investor_count": int(row["Investor Count"].strip() or 0),
            })
    return stocks


# ── API Call with web search ────────────────────────────────────────────────
def call_api(prompt, max_tokens=8192):
    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [{"role": "user", "content": prompt}],
    }
    
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    texts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return "\n".join(texts)


def fetch_batch_metrics(batch):
    """Fetch financial metrics for a batch of Indian stocks using web search."""
    
    stock_lines = "\n".join(
        f"  {i+1}. {s['stock']} (Code: {s['code']})" for i, s in enumerate(batch)
    )
    
    prompt = f"""Search screener.in for each of these Indian NSE/BSE stocks and extract their key financial metrics.

STOCKS:
{stock_lines}

For each stock, search "screener.in <STOCK_CODE>" or "screener.in <STOCK_NAME> financials" to find:
- Current Price, Market Cap (Cr), P/E ratio, P/B ratio, Book Value
- ROE (%), ROCE (%), Dividend Yield (%)  
- Debt to Equity ratio
- EPS, EPS Growth (%), Revenue Growth (%)
- 52-week High and Low
- Promoter Holding (%), Promoter Pledge (%)
- Free Cash Flow (Cr), Net Income/PAT (Cr)
- Sector/Industry
- PEG ratio, EV/EBITDA

Return ONLY a JSON array. Use null for unavailable data. No markdown fences, no commentary.
Format:
[
  {{
    "stock": "Company Name",
    "code": "NSE_CODE",
    "current_price": 123.45,
    "market_cap_cr": 5000,
    "pe_ratio": 25.5,
    "pb_ratio": 3.2,
    "book_value": 150,
    "roe": 18.5,
    "roce": 22.0,
    "dividend_yield": 1.2,
    "debt_to_equity": 0.5,
    "eps": 45.2,
    "eps_growth": 15.0,
    "revenue_growth": 12.0,
    "high_52w": 200,
    "low_52w": 100,
    "promoter_holding": 55.0,
    "promoter_pledge": 0.0,
    "fcf_cr": 300,
    "net_income_cr": 450,
    "sector": "Financial Services",
    "peg_ratio": 1.5,
    "ev_ebitda": 15.0
  }}
]"""

    try:
        response_text = call_api(prompt)
        # Extract JSON array from response
        json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', response_text)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
        # Try cleaning up common issues
        cleaned = response_text.strip()
        cleaned = re.sub(r'^```json\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        if cleaned.startswith('['):
            return json.loads(cleaned)
        print(f"  ⚠ Could not parse JSON from response (length={len(response_text)})")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


# ── Cache ───────────────────────────────────────────────────────────────────
def cache_key(code):
    return re.sub(r'[^a-zA-Z0-9]', '_', code)

def load_cached(code):
    path = os.path.join(CACHE_DIR, f"{cache_key(code)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_cache(code, data):
    path = os.path.join(CACHE_DIR, f"{cache_key(code)}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Scoring Engine ──────────────────────────────────────────────────────────
def score_stock(m, investor_count):
    """
    Multi-factor scoring model (0-100 scale).
    Categories: Valuation, Quality, Growth, Safety, Momentum, Conviction.
    """
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
    
    # ── VALUATION (25 pts) ──
    rate("Valuation", "P/E", m.get("pe_ratio"), 8, lower_better=True, lo=5, hi=50)
    rate("Valuation", "PEG", m.get("peg_ratio"), 6, lower_better=True, lo=0.3, hi=3)
    rate("Valuation", "EV/EBITDA", m.get("ev_ebitda"), 5, lower_better=True, lo=5, hi=30)
    rate("Valuation", "P/B", m.get("pb_ratio"), 4, lower_better=True, lo=0.5, hi=8)
    
    # Earnings yield (inverse of P/E, higher = cheaper)
    pe = m.get("pe_ratio")
    if pe and pe > 0:
        ey = 100.0 / pe
        rate("Valuation", "Earnings Yield", ey, 2, lo=2, hi=15)
    
    # ── QUALITY (25 pts) ──
    rate("Quality", "ROE", m.get("roe"), 10, lo=5, hi=25)
    rate("Quality", "ROCE", m.get("roce"), 10, lo=8, hi=30)
    
    # FCF quality
    fcf = m.get("fcf_cr")
    ni = m.get("net_income_cr")
    if fcf is not None:
        if fcf > 0:
            scores["Quality"] = scores.get("Quality", 0) + 3
        weights["Quality"] = weights.get("Quality", 0) + 3
        # FCF to earnings conversion
        if ni and ni > 0 and fcf > 0:
            ratio = fcf / ni
            rate("Quality", "FCF/PAT", ratio, 2, lo=0.3, hi=1.2)
    
    # ── GROWTH (20 pts) ──
    rate("Growth", "EPS Growth", m.get("eps_growth"), 10, lo=0, hi=30)
    rate("Growth", "Revenue Growth", m.get("revenue_growth"), 10, lo=0, hi=30)
    
    # ── SAFETY (20 pts) ──
    rate("Safety", "Debt/Equity", m.get("debt_to_equity"), 8, lower_better=True, lo=0, hi=2)
    rate("Safety", "Promoter Hold", m.get("promoter_holding"), 6, lo=25, hi=70)
    rate("Safety", "Low Pledge", m.get("promoter_pledge"), 4, lower_better=True, lo=0, hi=25)
    rate("Safety", "Dividend", m.get("dividend_yield"), 2, lo=0, hi=4)
    
    # ── MOMENTUM/VALUE (5 pts) ──
    hi52 = m.get("high_52w")
    lo52 = m.get("low_52w")
    cur = m.get("current_price")
    if hi52 and lo52 and cur and hi52 > lo52:
        pos = (cur - lo52) / (hi52 - lo52)
        rate("Momentum", "52W Discount", 1 - pos, 5, lo=0, hi=0.6)
    
    # ── CONVICTION (5 pts) ──
    scores["Conviction"] = min(5, investor_count * 1.25)
    weights["Conviction"] = 5
    
    # ── Aggregate ──
    total_score = sum(scores.values())
    total_weight = sum(weights.values())
    
    # Normalize to 100
    if total_weight > 0:
        # Scale up proportionally if some metrics missing
        final = (total_score / total_weight) * 100
    else:
        final = 0
    
    return round(min(100, final), 1), {
        "category_scores": {k: round(v, 1) for k, v in scores.items()},
        "category_weights": {k: round(v, 1) for k, v in weights.items()},
        "data_completeness": f"{sum(1 for v in m.values() if v is not None)}/{len(m)}",
    }


# ── Classification ──────────────────────────────────────────────────────────
def classify_stock(m, score):
    """Classify stock into investment categories."""
    tags = []
    
    pe = m.get("pe_ratio")
    pb = m.get("pb_ratio")
    roe = m.get("roe")
    roce = m.get("roce")
    eps_g = m.get("eps_growth")
    rev_g = m.get("revenue_growth")
    de = m.get("debt_to_equity")
    dy = m.get("dividend_yield")
    
    # Value stock
    if pe and pe < 15 and pb and pb < 2:
        tags.append("Deep Value")
    elif pe and pe < 20:
        tags.append("Value")
    
    # Growth stock
    if eps_g and eps_g > 20 and rev_g and rev_g > 15:
        tags.append("High Growth")
    elif eps_g and eps_g > 10:
        tags.append("Growth")
    
    # Quality
    if roe and roe > 20 and roce and roce > 20:
        tags.append("High Quality")
    
    # Safety
    if de is not None and de < 0.3:
        tags.append("Low Debt")
    if de is not None and de == 0:
        tags.append("Debt Free")
    
    # Income
    if dy and dy > 2:
        tags.append("Dividend")
    
    # Turnaround
    if pe and pe < 0:
        tags.append("Loss-Making")
    
    # GARP
    peg = m.get("peg_ratio")
    if peg and 0.5 < peg < 1.5 and roe and roe > 15:
        tags.append("GARP")
    
    return tags if tags else ["Unclassified"]


# ── Main Pipeline ───────────────────────────────────────────────────────────
def run_agent(max_batches=None, score_only=False):
    print("=" * 72)
    print("  🏦 SUPER INVESTOR STOCK ANALYSIS AGENT v2")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    
    stocks = load_stocks()
    print(f"\n📊 Loaded {len(stocks)} stocks from {len(set(s['investor'] for s in stocks))} super investors")
    
    if not score_only:
        uncached = [s for s in stocks if load_cached(s["code"]) is None]
        cached_count = len(stocks) - len(uncached)
        print(f"💾 Cached: {cached_count} | To fetch: {len(uncached)}")
        
        if uncached:
            # Sort by investor count (highest conviction first)
            uncached.sort(key=lambda x: -x["investor_count"])
            batches = [uncached[i:i+BATCH_SIZE] for i in range(0, len(uncached), BATCH_SIZE)]
            if max_batches:
                batches = batches[:max_batches]
            
            for bi, batch in enumerate(batches):
                names = ", ".join(s["stock"][:20] for s in batch)
                print(f"\n🔍 Batch {bi+1}/{len(batches)}: {names}")
                
                result = fetch_batch_metrics(batch)
                matched = 0
                
                if result:
                    for item in result:
                        # Match result back to stock
                        item_code = item.get("code", "").upper()
                        item_name = item.get("stock", "").lower()
                        
                        target = None
                        for s in batch:
                            if s["code"].upper() == item_code:
                                target = s
                                break
                            if s["stock"].lower() in item_name or item_name in s["stock"].lower():
                                target = s
                                break
                        
                        if not target and len(result) == len(batch):
                            idx = result.index(item)
                            if idx < len(batch):
                                target = batch[idx]
                        
                        if target:
                            save_cache(target["code"], item)
                            matched += 1
                
                print(f"  ✅ Matched {matched}/{len(batch)} stocks")
                time.sleep(1)  # Rate limiting
    
    # ── Score all stocks ──
    print("\n📈 Scoring all stocks...")
    results = []
    
    for s in stocks:
        cached_data = load_cached(s["code"])
        if cached_data:
            final_score, breakdown = score_stock(cached_data, s["investor_count"])
            tags = classify_stock(cached_data, final_score)
            results.append({
                **s,
                "metrics": cached_data,
                "score": final_score,
                "breakdown": breakdown,
                "tags": tags,
                "has_data": True,
            })
        else:
            results.append({
                **s,
                "metrics": {},
                "score": 0,
                "breakdown": {},
                "tags": ["No Data"],
                "has_data": False,
            })
    
    results.sort(key=lambda x: (-x["score"], -x["investor_count"]))
    return results


# ── Output Generation ───────────────────────────────────────────────────────
def generate_outputs(results):
    # ── Full CSV ──
    csv_path = os.path.join(OUTPUT_DIR, "stock_analysis_results.csv")
    fields = [
        "Rank", "Stock", "Code", "Score", "Tags", "Investor", "Inv Count",
        "Price", "Mkt Cap Cr", "P/E", "P/B", "PEG", "EV/EBITDA",
        "ROE%", "ROCE%", "EPS", "EPS Gr%", "Rev Gr%", "D/E",
        "Div Yld%", "Promoter%", "Pledge%", "FCF Cr", "PAT Cr",
        "52W Hi", "52W Lo", "Book Val", "Sector"
    ]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(results):
            m = r.get("metrics", {})
            writer.writerow({
                "Rank": i + 1,
                "Stock": r["stock"],
                "Code": r["code"],
                "Score": r["score"],
                "Tags": ", ".join(r.get("tags", [])),
                "Investor": r["investor"][:80],
                "Inv Count": r["investor_count"],
                "Price": m.get("current_price", ""),
                "Mkt Cap Cr": m.get("market_cap_cr", ""),
                "P/E": m.get("pe_ratio", ""),
                "P/B": m.get("pb_ratio", ""),
                "PEG": m.get("peg_ratio", ""),
                "EV/EBITDA": m.get("ev_ebitda", ""),
                "ROE%": m.get("roe", ""),
                "ROCE%": m.get("roce", ""),
                "EPS": m.get("eps", ""),
                "EPS Gr%": m.get("eps_growth", ""),
                "Rev Gr%": m.get("revenue_growth", ""),
                "D/E": m.get("debt_to_equity", ""),
                "Div Yld%": m.get("dividend_yield", ""),
                "Promoter%": m.get("promoter_holding", ""),
                "Pledge%": m.get("promoter_pledge", ""),
                "FCF Cr": m.get("fcf_cr", ""),
                "PAT Cr": m.get("net_income_cr", ""),
                "52W Hi": m.get("high_52w", ""),
                "52W Lo": m.get("low_52w", ""),
                "Book Val": m.get("book_value", ""),
                "Sector": m.get("sector", ""),
            })
    
    # ── JSON ──
    json_path = os.path.join(OUTPUT_DIR, "stock_analysis_full.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Outputs:")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {json_path}")
    
    return csv_path, json_path


def print_summary(results):
    with_data = [r for r in results if r["has_data"]]
    without_data = [r for r in results if not r["has_data"]]
    
    print(f"\n{'=' * 72}")
    print(f"  📊 ANALYSIS SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Stocks with data: {len(with_data)}/{len(results)}")
    
    if with_data:
        scores = [r["score"] for r in with_data]
        print(f"  Score range: {min(scores):.1f} - {max(scores):.1f}")
        print(f"  Average score: {sum(scores)/len(scores):.1f}")
    
    print(f"\n{'─' * 72}")
    print(f"  TOP 20 STOCKS BY COMPOSITE SCORE")
    print(f"{'─' * 72}")
    print(f"{'#':<4} {'Stock':<28} {'Score':<7} {'Inv':<4} {'P/E':<8} {'ROE%':<7} {'ROCE%':<7} {'D/E':<6} {'Tags'}")
    print(f"{'─' * 72}")
    
    for i, r in enumerate(results[:20]):
        m = r.get("metrics", {})
        
        def fmt(v, decimals=1):
            if isinstance(v, (int, float)):
                return f"{v:.{decimals}f}"
            return "-"
        
        tags_str = ", ".join(r.get("tags", []))[:25]
        print(f"{i+1:<4} {r['stock'][:27]:<28} {r['score']:<7} {r['investor_count']:<4} "
              f"{fmt(m.get('pe_ratio')):<8} {fmt(m.get('roe')):<7} {fmt(m.get('roce')):<7} "
              f"{fmt(m.get('debt_to_equity'), 2):<6} {tags_str}")
    
    # Sector breakdown
    sectors = {}
    for r in with_data[:50]:
        sec = r.get("metrics", {}).get("sector", "Unknown") or "Unknown"
        sectors[sec] = sectors.get(sec, 0) + 1
    
    if sectors:
        print(f"\n{'─' * 72}")
        print(f"  SECTOR DISTRIBUTION (Top 50 scored stocks)")
        print(f"{'─' * 72}")
        for sec, count in sorted(sectors.items(), key=lambda x: -x[1])[:10]:
            bar = "█" * count
            print(f"  {sec[:30]:<30} {count:>3} {bar}")


# ── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=None, help="Max batches to fetch")
    parser.add_argument("--score-only", action="store_true", help="Only score cached data")
    args = parser.parse_args()
    
    results = run_agent(max_batches=args.batches, score_only=args.score_only)
    generate_outputs(results)
    print_summary(results)
    print(f"\n✅ Complete at {datetime.now().strftime('%H:%M:%S')}")
