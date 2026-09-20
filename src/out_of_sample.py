"""Research-period parameter selection followed by untouched-period validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .backtest import build_schedule, simulate
from .data import load_market_data
from .experiments import TOP_NS, WINDOW_SETS
from .metrics import performance_metrics


SPLIT_DATE = pd.Timestamp("2024-01-01")
SELECTION_COST_BPS = 5


def _metrics_row(case_type, case, period, result, window=None, top_n=None):
    daily = result.daily["strategy_return"]
    return {
        "case_type": case_type,
        "case": case,
        "period": period,
        "window": window,
        "top_n": top_n,
        "cost_bps": SELECTION_COST_BPS,
        **performance_metrics(daily),
        "trade_count": len(result.trades),
        "rebalance_count": int(result.daily["rebalance_executed"].sum()),
        "total_fees": float(result.daily["fee_amount"].sum()),
        "total_turnover": float(result.daily["turnover_notional"].sum()),
    }


def run_out_of_sample(data, split_date=SPLIT_DATE):
    """Select by research Sharpe, then restart every rule from cash in validation."""
    schedules = {}
    for label, windows in WINDOW_SETS.items():
        for top_n in TOP_NS:
            targets, _ = build_schedule(data.close, windows=windows, top_n=top_n)
            schedules[(label, top_n)] = targets

    research_end = data.close.index[data.close.index < split_date].max()
    if pd.isna(research_end):
        raise ValueError("No research observations before the split date.")

    first_research = max(min(targets) for targets in schedules.values())
    common_research_dates = sorted(set.intersection(*(set(t) for t in schedules.values())))
    research_start = next(
        (
            date
            for date in common_research_dates
            if date >= first_research
            and date <= research_end
            and data.open.loc[date].notna().all()
            and data.volume.loc[date].gt(0).all()
        ),
        None,
    )
    first_validation = max(
        min(date for date in targets if date >= split_date) for targets in schedules.values()
    )
    common_validation_dates = sorted(set.intersection(*(set(t) for t in schedules.values())))
    validation_start = next(
        (
            date
            for date in common_validation_dates
            if date >= first_validation
            and data.open.loc[date].notna().all()
            and data.volume.loc[date].gt(0).all()
        ),
        None,
    )
    if research_start is None or validation_start is None:
        raise ValueError("No common executable date in one of the periods.")

    rows = []
    validation_results = {}
    for (label, top_n), targets in schedules.items():
        research_result = simulate(
            data.open.loc[:research_end],
            data.close.loc[:research_end],
            data.volume.loc[:research_end],
            {date: weights for date, weights in targets.items() if date <= research_end},
            research_start,
            cost=SELECTION_COST_BPS / 10_000,
        )
        validation_targets = {
            date: weights for date, weights in targets.items() if date >= validation_start
        }
        validation_result = simulate(
            data.open,
            data.close,
            data.volume,
            validation_targets,
            validation_start,
            cost=SELECTION_COST_BPS / 10_000,
        )
        case = f"window_{label}_top_{top_n}"
        rows.append(_metrics_row("strategy", case, "research", research_result, label, top_n))
        rows.append(_metrics_row("strategy", case, "validation", validation_result, label, top_n))
        validation_results[(label, top_n)] = validation_result

    benchmark_weights = {
        "equal_weight": pd.Series(1 / len(data.close.columns), index=data.close.columns),
        "csi300": pd.Series(
            [float(asset == config.BENCHMARK) for asset in data.close.columns],
            index=data.close.columns,
        ),
    }
    benchmark_results = {}
    for name, weights in benchmark_weights.items():
        research_result = simulate(
            data.open.loc[:research_end],
            data.close.loc[:research_end],
            data.volume.loc[:research_end],
            {research_start: weights},
            research_start,
            cost=SELECTION_COST_BPS / 10_000,
        )
        validation_result = simulate(
            data.open,
            data.close,
            data.volume,
            {validation_start: weights},
            validation_start,
            cost=SELECTION_COST_BPS / 10_000,
        )
        rows.append(_metrics_row("benchmark", name, "research", research_result))
        rows.append(_metrics_row("benchmark", name, "validation", validation_result))
        benchmark_results[name] = validation_result

    results = pd.DataFrame(rows)
    research_candidates = results[
        results["case_type"].eq("strategy") & results["period"].eq("research")
    ]
    selected = research_candidates.sort_values(
        ["sharpe_ratio_rf0", "annual_return", "window", "top_n"],
        ascending=[False, False, True, True],
    ).iloc[0]
    selected_key = (str(selected["window"]), int(selected["top_n"]))
    metadata = {
        "research_start": str(research_start.date()),
        "research_end": str(research_end.date()),
        "validation_start": str(validation_start.date()),
        "validation_end": str(data.close.index.max().date()),
        "split_date": str(split_date.date()),
        "selection_rule": "highest research-period Sharpe; annual return then label/top_n as tie-breakers",
        "candidate_count": len(research_candidates),
        "fixed_cost_bps": SELECTION_COST_BPS,
        "cash_rule": False,
        "selected_window": selected_key[0],
        "selected_top_n": selected_key[1],
        "snapshot_sha256": data.metadata.get("sha256", "unavailable"),
        "snapshot_path": data.metadata.get("snapshot_path", "in-memory fixture"),
    }
    return results, metadata, validation_results, benchmark_results


def write_outputs(results, metadata, validation_results, benchmark_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    results.to_csv(output_dir / "out_of_sample_results.csv", index=False, encoding="utf-8-sig")
    (output_dir / "out_of_sample_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected_case = f"window_{metadata['selected_window']}_top_{metadata['selected_top_n']}"
    baseline_case = "window_20+60+120_top_2"
    cases = [selected_case]
    if baseline_case != selected_case:
        cases.append(baseline_case)
    cases.extend(["equal_weight", "csi300"])
    table = results[results["case"].isin(cases)].pivot(
        index="case", columns="period", values="annual_return"
    ).reindex(cases)
    fig, ax = plt.subplots(figsize=(9, 5))
    table.plot(kind="bar", ax=ax, color=["#5B9BD5", "#ED7D31"])
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_title("Research-period selection and validation performance")
    ax.set_ylabel("Annual return")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "research_vs_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    selected_rows = results[results["case"].eq(selected_case)].set_index("period")
    equal_rows = results[results["case"].eq("equal_weight")].set_index("period")
    validation_selected = selected_rows.loc["validation"]
    validation_equal = equal_rows.loc["validation"]
    validation_strategies = results[
        results["case_type"].eq("strategy") & results["period"].eq("validation")
    ]
    best_validation = validation_strategies.sort_values(
        "sharpe_ratio_rf0", ascending=False
    ).iloc[0]
    positive_validation = int(validation_strategies["annual_return"].gt(0).sum())
    beat_equal_count = int(
        validation_strategies["annual_return"].gt(validation_equal["annual_return"]).sum()
    )
    sharpe_table = results[results["case_type"].eq("strategy")].pivot(
        index="case", columns="period", values="sharpe_ratio_rf0"
    )
    rank_correlation = float(sharpe_table.corr(method="spearman").iloc[0, 1])
    success = validation_selected["annual_return"] > 0
    beat_equal = validation_selected["annual_return"] > validation_equal["annual_return"]
    note = f"""# 样本外验证结果

- 研究期：{metadata['research_start']} 至 {metadata['research_end']}；验证期：{metadata['validation_start']} 至 {metadata['validation_end']}。
- 只根据研究期夏普，从 {metadata['candidate_count']} 组窗口和持仓数量中选出：窗口 {metadata['selected_window']}、Top {metadata['selected_top_n']}。
- 入选参数在研究期年化收益 {selected_rows.loc['research', 'annual_return']:.2%}、夏普 {selected_rows.loc['research', 'sharpe_ratio_rf0']:.2f}。
- 从现金开始进入验证期后，年化收益 {validation_selected['annual_return']:.2%}、夏普 {validation_selected['sharpe_ratio_rf0']:.2f}、最大回撤 {validation_selected['max_drawdown']:.2%}。
- 验证期等权基准年化收益 {validation_equal['annual_return']:.2%}、夏普 {validation_equal['sharpe_ratio_rf0']:.2f}、最大回撤 {validation_equal['max_drawdown']:.2%}。
- 验证期收益是否为正：{'是' if success else '否'}；是否超过等权基准：{'是' if beat_equal else '否'}。
- 验证期 12 组策略中有 {positive_validation} 组年化收益为正，{beat_equal_count} 组超过等权基准。
- 验证期事后最高夏普是 {best_validation['window']} 日、Top {int(best_validation['top_n'])}：年化收益 {best_validation['annual_return']:.2%}、夏普 {best_validation['sharpe_ratio_rf0']:.2f}。它在验证期结束前不能被当作可选择参数。
- 12 组参数的研究期与验证期夏普排名相关系数为 {rank_correlation:.2f}；接近 0 或为负表示参数排序缺乏延续性。

样本外表现比研究期更重要。两个区间都较短，这仍是一次有限样本验证，不能据此推断未来收益。
"""
    (output_dir / "out_of_sample_notes.md").write_text(note, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = load_market_data(args.snapshot)
    results, metadata, validation_results, benchmark_results = run_out_of_sample(data)
    output_dir = args.output_dir or config.OUTPUT_DIR / datetime.now(timezone.utc).strftime(
        "oos_%Y%m%dT%H%M%S%fZ"
    )
    write_outputs(results, metadata, validation_results, benchmark_results, output_dir)
    print((output_dir / "out_of_sample_notes.md").read_text(encoding="utf-8"))
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
