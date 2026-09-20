"""Auditable outputs for every portfolio and frozen run metadata."""
import json
import os
from pathlib import Path
import pandas as pd
from .config import OUTPUT_DIR, INITIAL_CAPITAL
from .metrics import performance_metrics, equity_curve, drawdown

def save_outputs(results, metadata, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame({name:r.daily.strategy_return for name,r in results.items()})
    if comparison.isna().any().any():
        raise ValueError("All portfolios must use exactly the same evaluation dates.")
    summary = {}
    for name, r in results.items():
        folder = output_dir / name
        folder.mkdir(exist_ok=True)
        for filename, frame, index in (
            ("daily_returns",r.daily,True), ("daily_holdings",r.holdings,False),
            ("trades",r.trades,False), ("executions",r.executions,False),
            ("monthly_rankings",r.rankings,False)):
            frame.to_csv(folder / f"{filename}.csv", index=index, encoding="utf-8-sig")
        r.holdings.pivot(index="date",columns="asset",values="weight").to_csv(
            folder / "daily_weights.csv", encoding="utf-8-sig")
        metrics = performance_metrics(r.daily.strategy_return)
        metrics.update(trade_count=len(r.trades), rebalance_count=int(r.daily.rebalance_executed.sum()),
                       total_fees=float(r.daily.fee_amount.sum()),
                       total_turnover=float(r.daily.turnover_notional.sum()))
        summary[name] = metrics
    summary = pd.DataFrame(summary).T
    summary.to_csv(output_dir / "performance_summary.csv", encoding="utf-8-sig")
    comparison.to_csv(output_dir / "portfolio_returns.csv", encoding="utf-8-sig")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = comparison.apply(equity_curve)
    # Explicit initial NAV anchor includes first-session losses in charts.
    anchor = curves.index[0] - pd.Timedelta(days=1)
    curves = pd.concat([pd.DataFrame(1.0,index=[anchor],columns=curves.columns),curves])
    dd = pd.concat([pd.DataFrame(0.0,index=[anchor],columns=comparison.columns),
                    comparison.apply(drawdown)])
    fig, axes = plt.subplots(2,1,figsize=(12,8),sharex=True)
    curves.plot(ax=axes[0],title="ETF momentum research: net equity")
    dd.plot(ax=axes[1],title="Drawdown from initial capital / running peak")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png",dpi=160)
    plt.close(fig)
    (output_dir / "research_notes.md").write_text(
        "# 回测结果说明\n\n数据及规则见 run_metadata.json；指标见 performance_summary.csv。\n"
        "momentum 为相对动量，momentum_cash 为全部有效分数负时持现金；"
        "benchmark_equal 为初始等权买入持有，benchmark_csi300 为沪深300买入持有。\n"
        "所有组合在相同日期开盘建仓，使用相同单边成本。现金收益和无风险利率均为0。\n"
        "使用同口径前复权开收盘价及可分割的复权单位，不是交易所真实份额；"
        "未模拟整手约束、涨跌停排队或容量冲击。成交量仅作事后可交易代理。\n"
        "各资产 return_contribution 之和减 daily_returns.transaction_cost 等于组合净收益；"
        "跨期归因请累计 asset_pnl 金额，不可直接累计日贡献百分比。\n"
        "基准差异同时包含资产选择、集中度和再平衡效应，不能单独证明动量因果作用。\n",
        encoding="utf-8")
    return summary
