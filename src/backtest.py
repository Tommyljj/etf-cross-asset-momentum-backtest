"""Next-open execution; fractional adjusted-price units drift between orders."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import MOMENTUM_WINDOWS, TOP_N, ONE_WAY_COST, INITIAL_CAPITAL

@dataclass
class BacktestResult:
    daily: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    executions: pd.DataFrame
    rankings: pd.DataFrame

def momentum_scores(prices, windows=MOMENTUM_WINDOWS):
    if not windows or any(int(w) != w or w < 1 for w in windows):
        raise ValueError("Windows must be positive integers.")
    return sum(prices.pct_change(w, fill_method=None).where(
        prices.notna().rolling(w + 1).sum().eq(w + 1)) for w in windows) / len(windows)

def build_schedule(close, windows=MOMENTUM_WINDOWS, top_n=TOP_N, cash_if_all_negative=False):
    if int(top_n) != top_n or not 1 <= top_n <= len(close.columns):
        raise ValueError("Invalid top_n.")
    scores = momentum_scores(close, windows)
    targets, rows = {}, []
    for date in close.groupby(close.index.to_period("M")).tail(1).index:
        score = scores.loc[date].dropna().sort_values(ascending=False, kind="stable")
        eligible = len(score) >= top_n
        cash = eligible and cash_if_all_negative and score.max() < 0
        chosen = list(score.head(top_n).index) if eligible and not cash else []
        i = close.index.searchsorted(date, side="right")
        trade_date = close.index[i] if i < len(close) else pd.NaT
        if eligible and pd.notna(trade_date):
            target = pd.Series(0.0, index=close.columns)
            target.loc[chosen] = 1 / top_n if chosen else 0
            targets[trade_date] = target
        ranks = {name: i + 1 for i, name in enumerate(score.index)}
        for name in close.columns:
            rows.append(dict(decision_date=date, trade_date=trade_date, asset=name,
                score=scores.at[date, name], rank=ranks.get(name, np.nan),
                selected=name in chosen, eligible=eligible, cash_signal=cash))
    if not targets:
        raise ValueError("No executable monthly signal; extend history or reduce windows.")
    return targets, pd.DataFrame(rows)

def simulate(open_prices, close_prices, volume, targets, start, cost=ONE_WAY_COST,
             initial_capital=INITIAL_CAPITAL, rankings=None):
    """Skip entire rebalance if held/target assets cannot trade; no delayed retry.
    Adjusted units are synthetic units, not exchange shares. Zero cash interest.
    """
    if not 0 <= cost < 1 or initial_capital <= 0:
        raise ValueError("Invalid cost or capital.")
    for frame in (open_prices, volume):
        if not frame.index.equals(close_prices.index) or not frame.columns.equals(close_prices.columns):
            raise ValueError("Mismatched market data.")
    if not close_prices.index.is_unique or not close_prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted.")
    for frame in (open_prices, close_prices):
        v = frame.to_numpy(dtype=float)
        if np.isinf(v).any() or (v[np.isfinite(v)] <= 0).any():
            raise ValueError("Prices must be positive or missing.")
    if np.isinf(volume.to_numpy(dtype=float)).any() or (volume < 0).any().any():
        raise ValueError("Invalid volume.")
    for date, target in targets.items():
        if date not in close_prices.index or not target.index.equals(close_prices.columns):
            raise ValueError("Target dates/assets mismatch.")
        if not np.isfinite(target).all() or (target < 0).any() or target.sum() > 1 + 1e-12:
            raise ValueError("Targets must be long-only, unlevered.")
    units = pd.Series(0.0, index=close_prices.columns)
    cash, previous_nav = float(initial_capital), float(initial_capital)
    marked = close_prices.ffill()  # valuation only; never execution
    daily, holdings, trades, executions = [], [], [], []
    for date in close_prices.loc[start:].index:
        idx = close_prices.index.get_loc(date)
        prev = marked.iloc[idx - 1] if idx else pd.Series(np.nan, index=units.index)
        op, cl = open_prices.loc[date], marked.loc[date]
        old_units = units.copy()
        trade_pnl = pd.Series(0.0, index=units.index)
        fees, notional, executed = 0.0, 0.0, False
        if date in targets:
            target = targets[date]
            needed = units.gt(0) | target.gt(0)
            valid = op.notna() & volume.loc[date].gt(0)
            if not valid[needed].all():
                executions.append(dict(date=date, status="skipped_untradable",
                    detail="|".join(units.index[needed & ~valid])))
            else:
                current = (units * op).fillna(0)
                nav_open = cash + current.sum()
                lo, hi = 0.0, nav_open
                # Solve post-fee NAV + cost * traded value = pre-fee NAV.
                for _ in range(80):
                    mid = (lo + hi) / 2
                    if mid + cost * (target * mid - current).abs().sum() > nav_open:
                        hi = mid
                    else:
                        lo = mid
                after = (lo + hi) / 2
                delta = target * after - current
                delta = delta.where(delta.abs() > initial_capital * 1e-12, 0.0)
                new_units = units + (delta / op).fillna(0)
                # A fully sold position can retain sub-cent floating-point dust.
                # Force executed zero-weight targets to exact zero for clean audits.
                new_units.loc[target.eq(0)] = 0.0
                fees, notional = float(delta.abs().sum() * cost), float(delta.abs().sum())
                cash -= float(delta.sum()) + fees
                trade_pnl = ((new_units - units) * (cl - op)).fillna(0)
                for name in units.index[delta.ne(0)]:
                    trades.append(dict(date=date, asset=name, side="buy" if delta[name] > 0 else "sell",
                        adjusted_price=op[name], delta_adjusted_units=new_units[name]-units[name],
                        traded_value=delta[name], fee=abs(delta[name])*cost,
                        pre_weight=current[name]/nav_open, target_weight=target[name]))
                units, executed = new_units, True
                executions.append(dict(date=date, status="executed", detail=""))
        if cl[units.gt(0)].isna().any():
            raise ValueError("Held asset has no valuation price.")
        pnl = (old_units * (cl - prev)).fillna(0) + trade_pnl
        values = (units * cl).fillna(0)
        nav = float(values.sum() + cash)
        if cash < -1e-6 or nav <= 0:
            raise ArithmeticError("Cash/NAV invariant violated.")
        if not np.isclose(pnl.sum() - fees, nav - previous_nav, atol=1e-6):
            raise ArithmeticError("P&L reconciliation failed.")
        daily.append(dict(date=date, nav=nav, cash=cash, strategy_return=nav/previous_nav-1,
            gross_return=pnl.sum()/previous_nav, transaction_cost=fees/previous_nav,
            fee_amount=fees, turnover_notional=notional/previous_nav, rebalance_executed=executed))
        for name in units.index:
            holdings.append(dict(date=date, asset=name, adjusted_units=units[name],
                market_value=values[name], weight=values[name]/nav, asset_pnl=pnl[name],
                return_contribution=pnl[name]/previous_nav, observed_close=pd.notna(close_prices.at[date,name]),
                tradable_open=bool(pd.notna(op[name]) and volume.at[date,name] > 0)))
        previous_nav = nav
    if not daily:
        raise ValueError("Empty evaluation period.")
    return BacktestResult(pd.DataFrame(daily).set_index("date"), pd.DataFrame(holdings),
        pd.DataFrame(trades, columns=["date","asset","side","adjusted_price","delta_adjusted_units",
                                     "traded_value","fee","pre_weight","target_weight"]),
        pd.DataFrame(executions, columns=["date","status","detail"]),
        rankings if rankings is not None else pd.DataFrame())

def run_momentum_backtest(open_prices, close_prices, volume, windows=MOMENTUM_WINDOWS,
                          top_n=TOP_N, cash_if_all_negative=False, cost=ONE_WAY_COST):
    targets, rankings = build_schedule(close_prices, windows, top_n, cash_if_all_negative)
    return simulate(open_prices, close_prices, volume, targets, min(targets), cost, rankings=rankings)
