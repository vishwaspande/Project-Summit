# magic_formula.py
# Project Summit – Magic Formula (ROCE + Earnings Yield) with WACC & Bond filters
# Python 3.9+

import argparse
import math
import os
import sys
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------- Helpers ---------------------------

def latest_nonnull_from_rows(df: pd.DataFrame, row_names: List[str]) -> Optional[float]:
    """
    From a yfinance financials DataFrame (rows=items, cols=dates),
    return the latest non-null value across the first matching row in `row_names`.
    """
    if df is None or df.empty:
        return None
    for rn in row_names:
        if rn in df.index:
            row = df.loc[rn].dropna()
            if not row.empty:
                # yfinance financials usually have newest column first (leftmost)
                return float(row.iloc[0])
    return None


def latest_two_nonnull_series(df: pd.DataFrame, row_name: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (latest, previous) values for a given row if present, else (None, None).
    Useful to average capital employed if you want.
    """
    if df is None or df.empty or row_name not in df.index:
        return None, None
    row = df.loc[row_name].dropna()
    if row.empty:
        return None, None
    latest = float(row.iloc[0])
    prev = float(row.iloc[1]) if len(row) > 1 else None
    return latest, prev


def clamp(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return max(lo, min(hi, x))


# --------------------------- Finance calc ---------------------------

def compute_earnings_yield(price: Optional[float], eps_ttm: Optional[float]) -> Optional[float]:
    if price is None or eps_ttm is None or price == 0:
        return None
    return eps_ttm / price  # ≈ 1/PE


def compute_roce(
    ebit: Optional[float],
    total_assets: Optional[float],
    current_liab: Optional[float],
    total_assets_prev: Optional[float] = None,
    current_liab_prev: Optional[float] = None,
) -> Optional[float]:
    if ebit is None or total_assets is None or current_liab is None:
        return None
    ce_latest = total_assets - current_liab
    if ce_latest is None or ce_latest <= 0:
        return None
    # Use average capital employed if previous period available
    if total_assets_prev is not None and current_liab_prev is not None:
        ce_prev = total_assets_prev - current_liab_prev
        if ce_prev is not None and ce_prev > 0:
            ce_avg = 0.5 * (ce_latest + ce_prev)
            if ce_avg > 0:
                return ebit / ce_avg
    return ebit / ce_latest


def compute_cost_of_equity(rf: float, beta: Optional[float], mrp: float) -> float:
    if beta is None or (isinstance(beta, float) and math.isnan(beta)):
        beta = 1.0
    return rf + beta * mrp


def compute_cost_of_debt(interest_exp: Optional[float], total_debt: Optional[float]) -> Optional[float]:
    if interest_exp is None or total_debt is None or total_debt <= 0:
        return None
    # Interest expense is often negative; use absolute
    return abs(interest_exp) / total_debt


def compute_tax_rate(income_tax_exp: Optional[float], pretax_income: Optional[float]) -> Optional[float]:
    if income_tax_exp is None or pretax_income is None or pretax_income == 0:
        return None
    return clamp(income_tax_exp / pretax_income, 0.0, 0.35)


def compute_wacc(
    cost_of_equity: float,
    cost_of_debt: Optional[float],
    tax_rate: Optional[float],
    market_cap: Optional[float],
    total_debt: Optional[float],
) -> Optional[float]:
    E = market_cap if market_cap is not None and market_cap > 0 else 0.0
    D = total_debt if total_debt is not None and total_debt > 0 else 0.0
    V = E + D
    if V == 0:
        return None
    we = E / V
    wd = D / V if cost_of_debt is not None else 0.0
    t = tax_rate if (tax_rate is not None and not math.isnan(tax_rate)) else 0.25
    cd_after_tax = (cost_of_debt * (1 - t)) if cost_of_debt is not None else 0.0
    return we * cost_of_equity + wd * cd_after_tax


# --------------------------- Data fetch ---------------------------

def fetch_company_block(ticker: str, debug: bool = False) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    info: Dict[str, Any] = {}

    # Price & market cap (fast)
    try:
        fast = t.fast_info
        info["price"] = float(fast.get("last_price")) if fast and fast.get("last_price") else None
        info["market_cap"] = float(fast.get("market_cap")) if fast and fast.get("market_cap") else None
    except Exception:
        info["price"], info["market_cap"] = None, None

    # Fallbacks via .info (also get beta & sector)
    try:
        inf = t.info
        if info.get("price") is None:
            info["price"] = inf.get("currentPrice")
        if info.get("market_cap") is None:
            info["market_cap"] = inf.get("marketCap")
        info["beta"] = inf.get("beta")
        info["sector"] = inf.get("sector")
        # EPS TTM
        info["eps_ttm"] = inf.get("trailingEps")
    except Exception:
        info.setdefault("beta", None)
        info.setdefault("sector", None)
        info.setdefault("eps_ttm", None)

    # Financial statements (annual first, then quarterly fallbacks)
    try:
        fin_a = t.financials            # income stmt (annual)
        fin_q = t.quarterly_financials  # income stmt (quarterly)
    except Exception:
        fin_a, fin_q = None, None

    try:
        bs_a = t.balance_sheet
        bs_q = t.quarterly_balance_sheet
    except Exception:
        bs_a, bs_q = None, None

    # EBIT / Operating Income
    ebit_candidates = ["EBIT", "Ebit", "Operating Income", "OperatingIncome", "Operating Profit"]
    ebit = latest_nonnull_from_rows(fin_a, ebit_candidates)
    if ebit is None:
        ebit = latest_nonnull_from_rows(fin_q, ebit_candidates)
    info["ebit"] = ebit

    # Interest expense
    interest_candidates = ["Interest Expense", "InterestExpense", "Interest Expense Non Operating"]
    interest = latest_nonnull_from_rows(fin_a, interest_candidates)
    if interest is None:
        interest = latest_nonnull_from_rows(fin_q, interest_candidates)
    info["interest_expense"] = interest

    # Tax & Pretax
    tax_candidates = ["Tax Provision", "Income Tax Expense"]
    pretax_candidates = ["Pretax Income", "Income Before Tax", "Earnings Before Tax"]
    tax_exp = latest_nonnull_from_rows(fin_a, tax_candidates) or latest_nonnull_from_rows(fin_q, tax_candidates)
    pretax = latest_nonnull_from_rows(fin_a, pretax_candidates) or latest_nonnull_from_rows(fin_q, pretax_candidates)
    info["income_tax_expense"] = tax_exp
    info["pretax_income"] = pretax

    # Balance Sheet: Total Assets, Current Liabilities, Total Debt
    total_assets = latest_nonnull_from_rows(bs_a, ["Total Assets"]) or latest_nonnull_from_rows(bs_q, ["Total Assets"])
    current_liab = latest_nonnull_from_rows(bs_a, ["Total Current Liabilities"]) or latest_nonnull_from_rows(
        bs_q, ["Total Current Liabilities"]
    )
    total_debt = latest_nonnull_from_rows(bs_a, ["Total Debt"]) or latest_nonnull_from_rows(bs_q, ["Total Debt"])
    if total_debt is None:
        short_a = latest_nonnull_from_rows(bs_a, ["Short Long Term Debt", "Short/Current Long Term Debt"])
        long_a = latest_nonnull_from_rows(bs_a, ["Long Term Debt"])
        short_q = latest_nonnull_from_rows(bs_q, ["Short Long Term Debt", "Short/Current Long Term Debt"])
        long_q = latest_nonnull_from_rows(bs_q, ["Long Term Debt"])
        candidates = []
        if short_a is not None or long_a is not None:
            candidates.append((short_a or 0.0) + (long_a or 0.0))
        if short_q is not None or long_q is not None:
            candidates.append((short_q or 0.0) + (long_q or 0.0))
        total_debt = candidates[0] if candidates and candidates[0] > 0 else None

    info["total_assets"] = total_assets
    info["current_liab"] = current_liab
    info["total_debt"] = total_debt

    # Previous period for averaging capital employed (optional)
    # Prefer annual; fall back to quarterly if annual missing
    base_df = bs_a if (bs_a is not None and not bs_a.empty) else bs_q
    ta_latest, ta_prev = latest_two_nonnull_series(base_df, "Total Assets") if base_df is not None else (None, None)
    tcl_latest, tcl_prev = latest_two_nonnull_series(base_df, "Total Current Liabilities") if base_df is not None else (None, None)
    info["ta_prev"] = ta_prev
    info["tcl_prev"] = tcl_prev

    if debug:
        if info.get("ebit") is None:
            print(f"[DEBUG] {ticker}: missing EBIT/Operating Income", file=sys.stderr)
        if info.get("total_assets") is None or info.get("current_liab") is None:
            print(f"[DEBUG] {ticker}: missing Total Assets/Total Current Liabilities", file=sys.stderr)

    return info


# --------------------------- Pipeline ---------------------------

def rank_desc(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="min")


def build_table(
    tickers: List[str],
    risk_free_rate: float,
    market_risk_premium: float,
    ey_spread_over_bonds: float,
    min_roce: float,
    exclude_financials: bool,
    debug: bool,
) -> pd.DataFrame:

    rows: List[Dict[str, Any]] = []

    for tk in tickers:
        data = fetch_company_block(tk, debug=debug)

        ey = compute_earnings_yield(price=data.get("price"), eps_ttm=data.get("eps_ttm"))
        roce = compute_roce(
            ebit=data.get("ebit"),
            total_assets=data.get("total_assets"),
            current_liab=data.get("current_liab"),
            total_assets_prev=data.get("ta_prev"),
            current_liab_prev=data.get("tcl_prev"),
        )

        # For financials (banks/NBFCs), ROCE is not meaningful; optionally blank it out
        sector = (data.get("sector") or "").lower()
        if exclude_financials and ("financial" in sector or "bank" in sector):
            roce = None

        coe = compute_cost_of_equity(risk_free_rate, data.get("beta"), market_risk_premium)
        cod = compute_cost_of_debt(data.get("interest_expense"), data.get("total_debt"))
        tax_rate = compute_tax_rate(data.get("income_tax_expense"), data.get("pretax_income"))
        wacc = compute_wacc(coe, cod, tax_rate, data.get("market_cap"), data.get("total_debt"))

        if debug and roce is None:
            print(f"[DEBUG] {tk}: ROCE None (ebit={data.get('ebit')}, TA={data.get('total_assets')}, "
                  f"TCL={data.get('current_liab')})", file=sys.stderr)

        rows.append({
            "ticker": tk,
            "sector": data.get("sector"),
            "price": data.get("price"),
            "market_cap": data.get("market_cap"),
            "beta": data.get("beta"),
            "eps_ttm": data.get("eps_ttm"),
            "ey": ey,
            "roce": roce,
            "wacc": wacc,
            "cost_of_equity": coe,
            "cost_of_debt": cod,
            "tax_rate": tax_rate,
            "ebit": data.get("ebit"),
            "total_assets": data.get("total_assets"),
            "current_liab": data.get("current_liab"),
            "total_debt": data.get("total_debt"),
            "risk_free_rate": risk_free_rate,
            "mrp": market_risk_premium,
        })

    df = pd.DataFrame(rows)

    # Filters
    df["passes_ey_vs_bond"] = df["ey"] >= (risk_free_rate + ey_spread_over_bonds)
    df["passes_roce_gt_wacc"] = (df["roce"] > df["wacc"])

    if min_roce > 0:
        df["passes_roce_floor"] = df["roce"] >= min_roce
    else:
        df["passes_roce_floor"] = True

    df["selected"] = df["passes_ey_vs_bond"] & df["passes_roce_gt_wacc"] & df["passes_roce_floor"]

    # Ranks (higher EY/ROCE better)
    df["rank_ey"] = rank_desc(df["ey"].fillna(-np.inf))
    df["rank_roce"] = rank_desc(df["roce"].fillna(-np.inf))
    df["magic_rank_sum"] = df["rank_ey"] + df["rank_roce"]

    # Sort: selected first, then best rank sum, then EY, then ROCE
    df = df.sort_values(
        by=["selected", "magic_rank_sum", "ey", "roce"],
        ascending=[False, True, False, False]
    ).reset_index(drop=True)

    return df


# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Project Summit – Magic Formula (ROCE + EY) with WACC & Bond filters"
    )
    p.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated tickers (for NSE via Yahoo, use suffix .NS e.g., TCS.NS, HDFCBANK.NS)"
    )
    p.add_argument("--rf", type=float, default=0.065, help="Risk-free (10Y), e.g. 0.065 for 6.5%")
    p.add_argument("--mrp", type=float, default=0.065, help="Market risk premium, e.g. 0.065 for 6.5%")
    p.add_argument("--ey_spread", type=float, default=0.025, help="EY must be >= rf + spread (e.g. 0.025 = 2.5%)")
    p.add_argument("--min_roce", type=float, default=0.0, help="Optional ROCE floor (e.g. 0.12 for 12%)")
    p.add_argument("--out", type=str, default="data/magic_formula_output.csv", help="Output CSV path")
    p.add_argument("--exclude_financials", action="store_true", help="Exclude Financials from ROCE calculations")
    p.add_argument("--debug", action="store_true", help="Print debug info")
    args = p.parse_args()

    default_universe = [
        "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "RELIANCE.NS",
        "LT.NS", "ASIANPAINT.NS", "BAJFINANCE.NS", "MARUTI.NS", "HCLTECH.NS"
    ]

    tickers = [t.strip() for t in (args.tickers.split(",") if args.tickers else default_universe)]
    return args, tickers


def main():
    args, tickers = parse_args()

    # Ensure output dir exists
    out_path = args.out.replace("\\", "/")
    out_dir = os.path.dirname(out_path)
    if out_dir and out_dir not in ("", ".", "./"):
        os.makedirs(out_dir, exist_ok=True)

    df = build_table(
        tickers=tickers,
        risk_free_rate=args.rf,
        market_risk_premium=args.mrp,
        ey_spread_over_bonds=args.ey_spread,
        min_roce=args.min_roce,
        exclude_financials=args.exclude_financials,
        debug=args.debug,
    )

    # Save raw numeric data
    df.to_csv(out_path, index=False)

    # Pretty console view
    view_cols = [
        "ticker", "sector", "selected", "magic_rank_sum",
        "ey", "roce", "wacc", "cost_of_equity", "cost_of_debt", "tax_rate",
        "passes_ey_vs_bond", "passes_roce_gt_wacc",
        "price", "market_cap", "beta", "total_debt"
    ]
    for c in view_cols:
        if c not in df.columns:
            df[c] = np.nan

    display = df[view_cols].copy()
    for c in ["ey", "roce", "wacc", "cost_of_equity", "cost_of_debt", "tax_rate"]:
        display[c] = display[c].apply(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "")

    print("\n=== Project Summit – Magic Formula (ROCE + EY) with WACC & Bond Filters ===")
    print(display.head(15).to_string(index=False))
    print(f"\nSaved full results to: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
