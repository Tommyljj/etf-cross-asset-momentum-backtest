"""Sample-based annualization; drawdown includes the initial capital peak."""
import numpy as np
from .config import TRADING_DAYS_PER_YEAR

def equity_curve(r):
    return (1 + r).cumprod()

def drawdown(r):
    curve = equity_curve(r)
    return curve / curve.cummax().clip(lower=1.0) - 1

def max_drawdown(r):
    return float(drawdown(r).min())

def performance_metrics(daily_returns):
    r = daily_returns.astype(float)
    if r.empty or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Returns must be finite, nonempty, greater than -100%.")
    total = float(equity_curve(r).iloc[-1] - 1)
    annual = (1 + total) ** (TRADING_DAYS_PER_YEAR / len(r)) - 1
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    return dict(total_return=total, annual_return=annual, annual_volatility=vol,
        sharpe_ratio_rf0=float(r.mean()*TRADING_DAYS_PER_YEAR/vol) if vol > 0 else np.nan,
        return_vol_ratio=annual/vol if vol > 0 else np.nan,
        max_drawdown=max_drawdown(r), daily_win_rate=float((r > 0).mean()))
