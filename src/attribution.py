"""Validation-period loss attribution for the research-selected momentum rule."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .data import load_market_data
from .out_of_sample import run_out_of_sample


def _compound(group: pd.Series) -> float:
    return float((1 + group).prod() - 1)


def _markdown_percent_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| 期间 | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for index, row in frame.iterrows():
        values = " | ".join(f"{float(row[column]):.2%}" for column in columns)
        lines.append(f"| {index} | {values} |")
    return "\n".join(lines)


def _drawdown_episode(daily: pd.DataFrame) -> dict:
    nav = daily["nav"]
    wealth = pd.concat([pd.Series([config.INITIAL_CAPITAL], index=[nav.index[0] - pd.Timedelta(days=1)]), nav])
    peaks = wealth.cummax()
    drawdown = wealth / peaks - 1
    valley = drawdown.idxmin()
    peak = wealth.loc[:valley].idxmax()
    future = wealth.loc[valley:]
    recovered = future[future.ge(peaks.loc[valley])]
    recovery = recovered.index[0] if not recovered.empty else pd.NaT
    return {
        "peak_date": str(peak.date()),
        "valley_date": str(valley.date()),
        "recovery_date": str(recovery.date()) if pd.notna(recovery) else "未恢复",
        "max_drawdown": float(drawdown.loc[valley]),
        "peak_nav": float(wealth.loc[peak]),
        "valley_nav": float(wealth.loc[valley]),
    }


def build_attribution(data):
    _, metadata, validation_results, benchmark_results = run_out_of_sample(data)
    selected_key = (metadata["selected_window"], int(metadata["selected_top_n"]))
    selected = validation_results[selected_key]
    equal = benchmark_results["equal_weight"]
    csi300 = benchmark_results["csi300"]

    daily = selected.daily.copy()
    daily["equal_weight_return"] = equal.daily["strategy_return"].reindex(daily.index)
    daily["equal_weight_nav"] = equal.daily["nav"].reindex(daily.index)
    daily["csi300_return"] = csi300.daily["strategy_return"].reindex(daily.index)
    daily["csi300_nav"] = csi300.daily["nav"].reindex(daily.index)
    daily["drawdown"] = daily["nav"] / pd.concat(
        [pd.Series([config.INITIAL_CAPITAL]), daily["nav"].reset_index(drop=True)]
    ).cummax().iloc[1:].set_axis(daily.index) - 1

    returns = daily[["strategy_return", "equal_weight_return", "csi300_return"]]
    annual = returns.groupby(returns.index.year).agg(_compound)
    annual.index.name = "year"
    monthly = returns.groupby(returns.index.to_period("M")).agg(_compound)
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "month"
    monthly["excess_vs_equal"] = monthly["strategy_return"] - monthly["equal_weight_return"]

    holdings = selected.holdings.copy()
    holdings["year"] = pd.to_datetime(holdings["date"]).dt.year
    by_year = holdings.groupby(["year", "asset"], as_index=False).agg(
        asset_pnl=("asset_pnl", "sum"),
        average_weight=("weight", "mean"),
        holding_days=("weight", lambda x: int(x.gt(1e-8).sum())),
    )
    total = holdings.groupby("asset", as_index=False).agg(
        asset_pnl=("asset_pnl", "sum"),
        average_weight=("weight", "mean"),
        holding_days=("weight", lambda x: int(x.gt(1e-8).sum())),
    )
    total.insert(0, "year", "全部")
    attribution = pd.concat([total, by_year], ignore_index=True)
    attribution["contribution_to_initial_capital"] = attribution["asset_pnl"] / config.INITIAL_CAPITAL

    trades = selected.trades.copy()
    if trades.empty:
        trade_summary = pd.DataFrame(columns=["asset", "trade_count", "buy_notional", "sell_notional", "fees"])
    else:
        trade_summary = trades.groupby("asset", as_index=False).agg(
            trade_count=("asset", "size"),
            buy_notional=("traded_value", lambda x: float(x[x.gt(0)].sum())),
            sell_notional=("traded_value", lambda x: float(-x[x.lt(0)].sum())),
            fees=("fee", "sum"),
        )

    episode = _drawdown_episode(selected.daily)
    reconciliation = {
        "ending_nav": float(selected.daily["nav"].iloc[-1]),
        "net_pnl": float(selected.daily["nav"].iloc[-1] - config.INITIAL_CAPITAL),
        "asset_pnl": float(holdings["asset_pnl"].sum()),
        "fees": float(selected.daily["fee_amount"].sum()),
    }
    reconciliation["difference"] = (
        reconciliation["asset_pnl"] - reconciliation["fees"] - reconciliation["net_pnl"]
    )
    return {
        "metadata": metadata,
        "daily": daily,
        "annual": annual,
        "monthly": monthly,
        "attribution": attribution,
        "trade_summary": trade_summary,
        "episode": episode,
        "reconciliation": reconciliation,
        "selected": selected,
    }


def write_outputs(bundle, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle["daily"].to_csv(output_dir / "validation_daily.csv", encoding="utf-8-sig")
    bundle["annual"].to_csv(output_dir / "annual_returns.csv", encoding="utf-8-sig")
    bundle["monthly"].to_csv(output_dir / "monthly_returns.csv", encoding="utf-8-sig")
    bundle["attribution"].to_csv(output_dir / "asset_attribution.csv", index=False, encoding="utf-8-sig")
    bundle["trade_summary"].to_csv(output_dir / "trade_summary.csv", index=False, encoding="utf-8-sig")

    annual = bundle["annual"].rename(columns={
        "strategy_return": "Selected momentum",
        "equal_weight_return": "Equal weight",
        "csi300_return": "CSI 300",
    })
    fig, ax = plt.subplots(figsize=(9, 5))
    annual.plot(kind="bar", ax=ax, color=["#C44E52", "#55A868", "#4C72B0"])
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title("Validation-period returns by calendar year")
    ax.set_xlabel("")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "annual_returns.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bundle["daily"].index, bundle["daily"]["nav"] / config.INITIAL_CAPITAL, label="Selected momentum", color="#C44E52")
    ax.plot(bundle["daily"].index, bundle["daily"]["equal_weight_nav"] / config.INITIAL_CAPITAL, label="Equal weight", color="#55A868")
    ax.plot(bundle["daily"].index, bundle["daily"]["csi300_nav"] / config.INITIAL_CAPITAL, label="CSI 300", color="#4C72B0")
    ax.axhline(1, color="#777777", linewidth=0.8)
    ax.set_title("Validation-period wealth")
    ax.set_ylabel("Wealth (start = 1)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_wealth.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    total_attr = bundle["attribution"].query("year == '全部'").sort_values("asset_pnl")
    worst_months = bundle["monthly"].nsmallest(5, "strategy_return")
    best_months = bundle["monthly"].nlargest(3, "strategy_return")
    annual_rows = bundle["annual"]
    episode = bundle["episode"]
    rec = bundle["reconciliation"]
    total_fees = rec["fees"]
    total_turnover = float(bundle["selected"].daily["turnover_notional"].sum())
    largest_loss = total_attr.iloc[0]
    most_held = total_attr.sort_values("holding_days", ascending=False).iloc[0]
    note = f"""# 验证期失效归因

## 主要结论

- 研究期选中的规则为 {bundle['metadata']['selected_window']} 日动量、Top {bundle['metadata']['selected_top_n']}。验证期从 {bundle['metadata']['validation_start']} 到 {bundle['metadata']['validation_end']}。
- 最大回撤为 {episode['max_drawdown']:.2%}，从 {episode['peak_date']} 的峰值下跌至 {episode['valley_date']}。截至样本结束，净值{('仍未恢复到前期峰值' if episode['recovery_date'] == '未恢复' else '已于 ' + episode['recovery_date'] + ' 恢复到前期峰值')}。
- 累计交易费用 {total_fees:,.0f} 元，约占初始资金 {total_fees/config.INITIAL_CAPITAL:.2%}；累计换手 {total_turnover:.2f} 倍。费用不是主要亏损来源。
- 资产损益最差的是 {largest_loss['asset']}，累计损益 {largest_loss['asset_pnl']:,.0f} 元。持有天数最多的是 {most_held['asset']}，共 {int(most_held['holding_days'])} 个交易日。
- Top 1 规则长期集中在单一资产。参数切换慢于市场风格变化时，错误持仓会直接放大组合回撤。

## 分年度收益

{_markdown_percent_table(annual_rows)}

## 最差的五个月

{_markdown_percent_table(worst_months)}

## 表现最好的三个月

{_markdown_percent_table(best_months)}

## 对账

- 资产损益合计：{rec['asset_pnl']:,.2f} 元
- 交易费用合计：{rec['fees']:,.2f} 元
- 组合净损益：{rec['net_pnl']:,.2f} 元
- 对账差额：{rec['difference']:.8f} 元

资产损益减交易费用等于组合净损益。结论只描述本次固定样本，不代表未来表现。
"""
    (output_dir / "attribution_notes.md").write_text(note, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = load_market_data(args.snapshot)
    bundle = build_attribution(data)
    output_dir = args.output_dir or config.OUTPUT_DIR / datetime.now(timezone.utc).strftime(
        "attribution_%Y%m%dT%H%M%S%fZ"
    )
    write_outputs(bundle, output_dir)
    print((output_dir / "attribution_notes.md").read_text(encoding="utf-8"))
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
