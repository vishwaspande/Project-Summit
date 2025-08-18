import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Config:
    # Risk inputs (you can change these via CLI)
    risk_free_rate: float = 0.065  # 6.5% India 10Y by default
    market_risk_premium: float = 0.065  # 6.5% common analyst assumption for India
    ey_spread_over_bonds: float = 0.025  # demand 2.5% over bonds as a risk premium filter
    min_roce: float = 0.0  # optional hard floor
    # Output
    out_csv: str = "data/magic_formula_output.csv"
    # Universe default (NSE tickers on Yahoo end with .NS)
    tickers: List[str] = None


def _safe_latest_value(row_like: pd.Series) -> Optional[float]:
    """
    yfinance financials are a row with date columns. We want the latest non-null value.
    Columns are usually newest->oldest; but we defensively sort by column name (dates) desc.
    """
    if row_like is None or len(row_like) == 0:
        return None
    s = row_like.dropna()
    if s.empty:
        return None
    # Try to sort columns as datetimes (newest first), else keep order
    try:
        cols = pd.to_datetime(s.index)
        s = s.loc[pd.Series(cols, index=s.index).sort_values(ascending=False).index]
    except Exception:
        pass
    return float(s.iloc[0])



def compute_earnings_yield(price: Optional[float], eps_ttm: Optional[float]) -> Optional[float]:
    if price is None or eps_ttm is None or price == 0:
        return None
    return eps_ttm / price  # EY = EPS/Price (≈ 1/PE)


def compute_roce(ebit: Optional[float], total_assets: Optional[float], current_liab: Optional[float]) -> Optional[float]:
    if ebit is None or total_assets is None or current_liab is None:
        return None
    capital_employed = total_assets - current_liab
    if capital_employed is None or capital_employed <= 0:
        return None
    return ebit / capital_employed



def clamp(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    if x is None or math.isnan(x):
        return None
    return max(lo, min(hi, x))


def compute_cost_of_equity(rf: float, beta: Optional[float], mrp: float) -> float:
    if beta is None or math.isnan(beta):
        beta = 1.0
    return rf + beta * mrp


def compute_cost_of_debt(interest_exp: Optional[float], total_debt: Optional[float]) -> Optional[float]:
    if interest_exp is None or total_debt is None or total_debt <= 0:
        return None
    # Interest expense is usually negative in statements; take absolute
    return abs(interest_exp) / total_debt


def compute_tax_rate(income_tax_exp: Optional[float], pretax_income: Optional[float]) -> Optional[float]:
    if income_tax_exp is None or pretax_income is None or pretax_income == 0:
        return None
    # clamp 0..35% (India effective corporate often ~22–25% after surcharges; but keep wide)
    return clamp(income_tax_exp / pretax_income, 0.0, 0.35)


def compute_wacc(cost_of_equity: float,
                 cost_of_debt: Optional[float],
                 tax_rate: Optional[float],
                 market_cap: Optional[float],
                 total_debt: Optional[float]) -> Optional[float]:
    E = market_cap if market_cap is not None and market_cap > 0 else 0.0
    D = total_debt if total_debt is not None and total_debt > 0 else 0.0
    V = E + D
    if V == 0:
        return None
    we = E / V
    wd = D / V
    if cost_of_debt is None:
        wd = 0.0  # if we can't estimate debt cost, treat as equity-only
    t = tax_rate if (tax_rate is not None and not math.isnan(tax_rate)) else 0.25
    cd_after_tax = (cost_of_debt * (1 - t)) if cost_of_debt is not None else 0.0
    return we * cost_of_equity + wd * cd_after_tax


def fetch_company_block(ticker: str) -> Dict[str, Any]:
    """
    Pulls price, EPS TTM, EBIT (latest annual), balance sheet (assets, current liabilities, debt),
    income (for tax, interest), and beta/market cap using yfinance.
    """
    t = yf.Ticker(ticker)

    info = {}
    # price & basic fields
    try:
        fast = t.fast_info  # newer, quicker price snapshot
        info["price"] = float(fast["last_price"]) if "last_price" in fast and fast["last_price"] else None
        info["market_cap"] = float(fast["market_cap"]) if "market_cap" in fast and fast["market_cap"] else None
    except Exception:
        info["price"] = None
        info["market_cap"] = None

    # fallback to .info if needed
    if info["market_cap"] is None or info["price"] is None:
        try:
            inf = t.info
            info["market_cap"] = info["market_cap"] or inf.get("marketCap")
            info["price"] = info["price"] or inf.get("currentPrice")
            info["beta"] = inf.get("beta")
        except Exception:
            pass

    # EPS TTM
    try:
        # yfinance’s .info sometimes has "trailingEps"
        if "beta" not in info or info.get("beta") is None:
            inf = t.info
            info["beta"] = inf.get("beta")
        info["eps_ttm"] = t.info.get("trailingEps")
    except Exception:
        info["eps_ttm"] = None

        # Income statement (annual, with fallbacks)
    try:
        inc = t.income_stmt  # annual
        if inc is None or inc.empty:
            inc = t.income_stmt_quarterly

        ebit = None
        # Common keys across tickers
        for key in ("EBIT", "Ebit", "Operating Income", "OperatingIncome"):
            if isinstance(inc, pd.DataFrame) and key in inc.index:
                ebit = _safe_latest_value(inc.loc[key])
                if ebit is not None:
                    break
        info["ebit"] = ebit

        # interest expense (usually negative)
        interest = None
        for key in ("Interest Expense", "InterestExpense", "Interest Expense Non Operating"):
            if isinstance(inc, pd.DataFrame) and key in inc.index:
                interest = _safe_latest_value(inc.loc[key])
                if interest is not None:
                    break
        info["interest_expense"] = interest

        # tax & pretax for tax-rate
        tax_exp = None
        for key in ("Tax Provision", "Income Tax Expense"):
            if isinstance(inc, pd.DataFrame) and key in inc.index:
                tax_exp = _safe_latest_value(inc.loc[key]); break

        pretax = None
        for key in ("Pretax Income", "Income Before Tax", "Earnings Before Tax"):
            if isinstance(inc, pd.DataFrame) and key in inc.index:
                pretax = _safe_latest_value(inc.loc[key]); break

        info["income_tax_expense"] = tax_exp
        info["pretax_income"] = pretax
    except Exception:
        info.update({"ebit": None, "interest_expense": None, "income_tax_expense": None, "pretax_income": None})

        # Balance sheet (annual, with fallbacks)
    try:
        bs = t.balance_sheet
        if bs is None or bs.empty:
            bs = t.balance_sheet_quarterly

        total_assets = None
        for key in ("Total Assets", "TotalAssets"):
            if isinstance(bs, pd.DataFrame) and key in bs.index:
                total_assets = _safe_latest_value(bs.loc[key]); break

        current_liab = None
        for key in ("Total Current Liabilities", "TotalCurrentLiabilities"):
            if isinstance(bs, pd.DataFrame) and key in bs.index:
                current_liab = _safe_latest_value(bs.loc[key]); break

        total_debt = None
        for key in ("Total Debt", "TotalDebt"):
            if isinstance(bs, pd.DataFrame) and key in bs.index:
                total_debt = _safe_latest_value(bs.loc[key]); break
        if total_debt is None:
            short_lt = bs.loc["Short Long Term Debt"] if isinstance(bs, pd.DataFrame) and "Short Long Term Debt" in bs.index else None
            long_term = bs.loc["Long Term Debt"] if isinstance(bs, pd.DataFrame) and "Long Term Debt" in bs.index else None
            total_debt = ( _safe_latest_value(short_lt) if short_lt is not None else 0.0 ) + \
                         ( _safe_latest_value(long_term) if long_term is not None else 0.0 )
            if total_debt == 0.0:
                total_debt = None

        info["total_assets"] = total_assets
        info["current_liab"] = current_liab
        info["total_debt"] = total_debt
    except Exception:
        info.update({"total_assets": None, "current_liab": None, "total_debt": None})

    return info


def rank_series_desc(s: pd.Series) -> pd.Series:
    """Rank descending (higher is better)."""
    return s.rank(ascending=False, method="min")


def build_magic_formula_table(cfg: Config) -> pd.DataFrame:
    rows = []
    for tk in cfg.tickers:
        data = fetch_company_block(tk)

        ey = compute_earnings_yield(price=data.get("price"), eps_ttm=data.get("eps_ttm"))
        roce = compute_roce(
            ebit=data.get("ebit"),
            total_assets=data.get("total_assets"),
            current_liab=data.get("current_liab"),
        )
        rf = cfg.risk_free_rate
        mrp = cfg.market_risk_premium
        coe = compute_cost_of_equity(rf, data.get("beta"), mrp)
        cod = compute_cost_of_debt(data.get("interest_expense"), data.get("total_debt"))
        tax_rate = compute_tax_rate(data.get("income_tax_expense"), data.get("pretax_income"))
        wacc = compute_wacc(coe, cod, tax_rate, data.get("market_cap"), data.get("total_debt"))

        rows.append({
            "ticker": tk,
            "price": data.get("price"),
            "market_cap": data.get("market_cap"),
            "beta": data.get("beta"),
            "eps_ttm": data.get("eps_ttm"),
            "ey": ey,
            "roce": roce,
            "ebit": data.get("ebit"),
            "total_assets": data.get("total_assets"),
            "current_liab": data.get("current_liab"),
            "total_debt": data.get("total_debt"),
            "interest_expense": data.get("interest_expense"),
            "income_tax_expense": data.get("income_tax_expense"),
            "pretax_income": data.get("pretax_income"),
            "risk_free_rate": rf,
            "market_risk_premium": mrp,
            "cost_of_equity": coe,
            "cost_of_debt": cod,
            "tax_rate": tax_rate,
            "wacc": wacc
        })

    df = pd.DataFrame(rows)

    # Filters
    # 1) EY >= rf + spread
    df["passes_ey_vs_bond"] = df["ey"] >= (cfg.risk_free_rate + cfg.ey_spread_over_bonds)
    # 2) ROCE > WACC (value accretive)
    df["passes_roce_gt_wacc"] = (df["roce"] > df["wacc"])

    # Magic Formula ranks (higher EY/ROCE is better)
    df["rank_ey"] = rank_series_desc(df["ey"].fillna(-np.inf))
    df["rank_roce"] = rank_series_desc(df["roce"].fillna(-np.inf))
    df["magic_rank_sum"] = df["rank_ey"] + df["rank_roce"]

    # Optional hard ROCE floor
    if cfg.min_roce > 0:
        df["passes_roce_floor"] = df["roce"] >= cfg.min_roce
    else:
        df["passes_roce_floor"] = True

    # Final selected flag
    df["selected"] = df["passes_ey_vs_bond"] & df["passes_roce_gt_wacc"] & df["passes_roce_floor"]

    # Sort: primary by magic rank, secondary by EY & ROCE
    df = df.sort_values(by=["selected", "magic_rank_sum", "ey", "roce"],
                        ascending=[False, True, False, False]).reset_index(drop=True)
    return df


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Project Summit: Magic Formula (ROCE + EY) with WACC & bond filters")
    p.add_argument("--tickers", type=str, required=False,
                   help="Comma-separated ticker list. For NSE via Yahoo, append .NS (e.g., TCS.NS, HDFCBANK.NS).")
    p.add_argument("--rf", type=float, default=0.065, help="Risk-free rate (10Y bond), e.g., 0.065 for 6.5%")
    p.add_argument("--mrp", type=float, default=0.065, help="Market risk premium, e.g., 0.065 for 6.5%")
    p.add_argument("--ey_spread", type=float, default=0.025, help="EY minus bond yield threshold (e.g., 0.025 = 2.5%)")
    p.add_argument("--min_roce", type=float, default=0.0, help="Optional ROCE floor (e.g., 0.12 for 12%)")
    p.add_argument("--out", type=str, default="data/magic_formula_output.csv", help="Output CSV path")
    p.add_argument("--debug", action="store_true", help="Print missing-field reasons per ticker")

    args = p.parse_args()

    default_universe = [
        # Add / edit your starter universe here
        "TCS.NS", "INFY.NS", "HDFCBANK.NS", "HDFC.NS", "ICICIBANK.NS", "ITC.NS",
        "RELIANCE.NS", "LT.NS", "ASIANPAINT.NS", "BAJFINANCE.NS", "MARUTI.NS"
    ]

    cfg = Config(
        risk_free_rate=args.rf,
        market_risk_premium=args.mrp,
        ey_spread_over_bonds=args.ey_spread,
        min_roce=args.min_roce,
        out_csv=args.out,
        tickers=[t.strip() for t in (args.tickers.split(",") if args.tickers else default_universe)]
    )
    return cfg


def main():
    cfg = parse_args()

    # Basic IO prep
    out_dir = "/".join(cfg.out_csv.replace("\\", "/").split("/")[:-1])
    if out_dir and out_dir not in ("", ".", "./"):
        import os
        os.makedirs(out_dir, exist_ok=True)

    df = build_magic_formula_table(cfg)

    # nice columns for viewing
    view_cols = [
        "ticker", "selected", "magic_rank_sum",
        "ey", "roce", "wacc", "cost_of_equity", "cost_of_debt", "tax_rate",
        "passes_ey_vs_bond", "passes_roce_gt_wacc",
        "price", "market_cap", "beta", "eps_ttm", "total_debt"
    ]
    for c in view_cols:
        if c not in df.columns:
            df[c] = np.nan

    # formatting
    def pct(x): return x if pd.isna(x) else float(x)

    for c in ["ey", "roce", "wacc", "cost_of_equity", "cost_of_debt", "tax_rate"]:
        df[c] = df[c].astype(float)

    if getattr(cfg, "debug", False):
        if roce is None:
            print(f"[DEBUG] {tk}: ROCE missing — ebit={data.get('ebit')}, "
                f"total_assets={data.get('total_assets')}, current_liab={data.get('current_liab')}")


    # save
    df.to_csv(cfg.out_csv, index=False)

    # print top 10 to console
    display = df[view_cols].copy()
    # Pretty percentage-like columns for console (leave CSV raw)
    for c in ["ey", "roce", "wacc", "cost_of_equity", "cost_of_debt", "tax_rate"]:
        display[c] = display[c].apply(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "")

    print("\n=== Magic Formula (with WACC & Bond Filters) ===")
    print(display.head(10).to_string(index=False))
    print(f"\nSaved full results to: {cfg.out_csv}")


if __name__ == "__main__":
    # Friendlier error surface
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
