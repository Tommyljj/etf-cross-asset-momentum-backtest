"""Run a pre-declared sensitivity grid on one frozen snapshot and sample."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .backtest import build_schedule, simulate
from .data import load_market_data
from .metrics import performance_metrics


WINDOW_SETS = {
    "20": (20,),
    "60": (60,),
    "120": (120,),
    "20+60+120": (20, 60, 120),
}
TOP_NS = (1, 2, 3)
COSTS_BPS = (5, 10, 20)
CASH_RULES = (False, True)


def run_grid(data) -> tuple[pd.DataFrame, dict]:
    """Return all 72 strategy cases plus two benchmarks on a common sample."""
    schedules: dict[tuple, tuple[dict, pd.DataFrame]] = {}
    for window_label, windows in WINDOW_SETS.items():
        for top_n in TOP_NS:
            for cash_rule in CASH_RULES:
                schedules[(window_label, top_n, cash_rule)] = build_schedule(
                    data.close,
                    windows=windows,
                    top_n=top_n,
                    cash_if_all_negative=cash_rule,
                )

    first_dates = [min(targets) for targets, _ in schedules.values()]
    candidate_start = max(first_dates)
    execution_dates = sorted(set.intersection(*(set(targets) for targets, _ in schedules.values())))
    common_start = next(
        (
            date
            for date in execution_dates
            if date >= candidate_start
            and data.open.loc[date].notna().all()
            and data.volume.loc[date].gt(0).all()
        ),
        None,
    )
    if common_start is None:
        raise ValueError("No common executable start date for the experiment grid.")

    rows: list[dict] = []
    for (window_label, top_n, cash_rule), (targets, _) in schedules.items():
        for cost_bps in COSTS_BPS:
            result = simulate(
                data.open,
                data.close,
                data.volume,
                targets,
                common_start,
                cost=cost_bps / 10_000,
            )
            metrics = performance_metrics(result.daily["strategy_return"])
            rows.append(
                {
                    "case_type": "strategy",
                    "window": window_label,
                    "top_n": top_n,
                    "cost_bps": cost_bps,
                    "cash_rule": cash_rule,
                    **metrics,
                    "trade_count": len(result.trades),
                    "rebalance_count": int(result.daily["rebalance_executed"].sum()),
                    "total_fees": float(result.daily["fee_amount"].sum()),
                    "total_turnover": float(result.daily["turnover_notional"].sum()),
                }
            )

    benchmark_targets = {
        "equal_weight": pd.Series(1 / len(data.close.columns), index=data.close.columns),
        "csi300": pd.Series(
            [float(asset == config.BENCHMARK) for asset in data.close.columns],
            index=data.close.columns,
        ),
    }
    for name, weights in benchmark_targets.items():
        result = simulate(
            data.open,
            data.close,
            data.volume,
            {common_start: weights},
            common_start,
            cost=config.ONE_WAY_COST,
        )
        rows.append(
            {
                "case_type": "benchmark",
                "window": name,
                "top_n": np.nan,
                "cost_bps": config.ONE_WAY_COST * 10_000,
                "cash_rule": False,
                **performance_metrics(result.daily["strategy_return"]),
                "trade_count": len(result.trades),
                "rebalance_count": int(result.daily["rebalance_executed"].sum()),
                "total_fees": float(result.daily["fee_amount"].sum()),
                "total_turnover": float(result.daily["turnover_notional"].sum()),
            }
        )

    metadata = {
        "common_start": str(common_start.date()),
        "common_end": str(data.close.index.max().date()),
        "strategy_case_count": len(WINDOW_SETS) * len(TOP_NS) * len(COSTS_BPS) * len(CASH_RULES),
        "predeclared_grid": {
            "windows": WINDOW_SETS,
            "top_n": TOP_NS,
            "costs_bps": COSTS_BPS,
            "cash_rules": CASH_RULES,
        },
        "snapshot_sha256": data.metadata.get("sha256", "unavailable"),
        "snapshot_path": data.metadata.get("snapshot_path", "in-memory fixture"),
    }
    return pd.DataFrame(rows), metadata


def write_results(results: pd.DataFrame, metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    results.to_csv(output_dir / "experiment_results.csv", index=False, encoding="utf-8-sig")
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    strategy = results[results["case_type"].eq("strategy")].copy()
    base = strategy[
        strategy["cost_bps"].eq(5) & strategy["cash_rule"].eq(False)
    ]
    window_order = list(WINDOW_SETS)
    for metric, title, filename in (
        ("annual_return", "Annual return: 5 bps, no cash rule", "annual_return_heatmap.png"),
        ("sharpe_ratio_rf0", "Sharpe ratio: 5 bps, no cash rule", "sharpe_heatmap.png"),
    ):
        table = base.pivot(index="window", columns="top_n", values=metric).reindex(window_order)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        image = ax.imshow(table.to_numpy(), cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(table.columns)), [f"Top {int(x)}" for x in table.columns])
        ax.set_yticks(range(len(table.index)), table.index)
        ax.set_title(title)
        for row in range(table.shape[0]):
            for col in range(table.shape[1]):
                value = table.iloc[row, col]
                label = f"{value:.1%}" if metric == "annual_return" else f"{value:.2f}"
                ax.text(col, row, label, ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    equal_return = float(
        results.loc[
            results["case_type"].eq("benchmark") & results["window"].eq("equal_weight"),
            "annual_return",
        ].iloc[0]
    )
    profitable = int(strategy["annual_return"].gt(0).sum())
    beat_equal = int(strategy["annual_return"].gt(equal_return).sum())
    paired = strategy.pivot(
        index=["window", "top_n", "cost_bps"], columns="cash_rule", values="annual_return"
    )
    cash_changed = int((paired[True] - paired[False]).abs().gt(1e-12).sum())
    positive_distinct = int(paired[False].gt(0).sum())
    best = strategy.sort_values("sharpe_ratio_rf0", ascending=False).iloc[0]
    baseline = strategy[
        strategy["window"].eq("20+60+120")
        & strategy["top_n"].eq(2)
        & strategy["cost_bps"].eq(5)
        & strategy["cash_rule"].eq(False)
    ].iloc[0]
    note = f"""# 参数敏感性实验结果

共同样本：{metadata['common_start']} 至 {metadata['common_end']}。预先规定的策略组合共 {metadata['strategy_case_count']} 组。

- 基线组合年化收益：{baseline['annual_return']:.2%}；夏普：{baseline['sharpe_ratio_rf0']:.2f}；最大回撤：{baseline['max_drawdown']:.2%}。
- {profitable}/{metadata['strategy_case_count']} 组策略年化收益为正。
- {beat_equal}/{metadata['strategy_case_count']} 组策略年化收益超过同样本等权买入持有基准（{equal_return:.2%}）。
- 现金规则在 {cash_changed}/{len(paired)} 个非重复参数组合中改变了结果；本样本实际未触发，因此 72 行相当于 36 个非重复组合，其中 {positive_distinct} 个为正收益。
- 按样本内夏普最高的组合：窗口 {best['window']}、Top {int(best['top_n'])}、成本 {int(best['cost_bps'])} bps、现金规则 {bool(best['cash_rule'])}；年化收益 {best['annual_return']:.2%}，夏普 {best['sharpe_ratio_rf0']:.2f}。

最高值仅用于描述这张完整参数表，不能作为未来最优参数。下一步应按时间切分样本，检验该组合是否在未参与选择的时期继续成立。
"""
    (output_dir / "sensitivity_notes.md").write_text(note, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = load_market_data(args.snapshot)
    results, metadata = run_grid(data)
    output_dir = args.output_dir or config.OUTPUT_DIR / datetime.now(timezone.utc).strftime(
        "sensitivity_%Y%m%dT%H%M%S%fZ"
    )
    write_results(results, metadata, output_dir)
    print((output_dir / "sensitivity_notes.md").read_text(encoding="utf-8"))
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
