"""python -m src.run [--refresh] [--snapshot PATH] [--output-dir PATH]"""
import argparse
from datetime import datetime, timezone
import hashlib
import platform
from pathlib import Path
import pandas as pd
import numpy as np
from . import config
from .data import load_market_data, import_csv_snapshot
from .backtest import build_schedule, simulate
from .report import save_outputs

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--refresh",action="store_true",help="Download a new immutable snapshot.")
    group.add_argument("--snapshot",type=Path)
    group.add_argument("--import-dir",type=Path,help="Import same-basis adjusted Open/Close CSVs with Volume.")
    parser.add_argument("--output-dir",type=Path)
    args = parser.parse_args()
    if args.import_dir:
        args.snapshot = import_csv_snapshot(args.import_dir)
    data = load_market_data(args.snapshot, args.refresh)
    base, rankings = build_schedule(data.close)
    defensive, defensive_rankings = build_schedule(data.close, cash_if_all_negative=True)
    # Common start is a scheduled session when every benchmark asset has an open and volume.
    start = next((d for d in base if data.open.loc[d].notna().all()
                  and data.volume.loc[d].gt(0).all()), None)
    if start is None:
        raise ValueError("No common tradable starting session.")
    portfolios = {}
    for name, targets, ranks in (("momentum",base,rankings),
                                 ("momentum_cash",defensive,defensive_rankings)):
        portfolios[name] = simulate(data.open,data.close,data.volume,targets,start,rankings=ranks)
    for name, weights in (
        ("benchmark_equal",pd.Series(1/len(data.close.columns),index=data.close.columns)),
        ("benchmark_csi300",pd.Series([float(n==config.BENCHMARK) for n in data.close.columns],
                                     index=data.close.columns))):
        if weights.sum() == 0:
            raise ValueError("Benchmark absent from universe.")
        portfolios[name] = simulate(data.open,data.close,data.volume,{start:weights},start)
    settings = {k:getattr(config,k) for k in ("START_DATE","END_DATE","MOMENTUM_WINDOWS",
                 "TOP_N","ONE_WAY_COST","INITIAL_CAPITAL","TRADING_DAYS_PER_YEAR","UNIVERSE")}
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(__file__).parent.glob("*.py")}
    metadata = dict(data=data.metadata,settings=settings,code_sha256=hashes,
        python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,
        evaluation_start=str(start),cash_rate=0,risk_free_rate=0,
        execution="next open; adjusted fractional units; skip whole rebalance if untradable",
        selected_baseline="momentum")
    folder = args.output_dir or config.OUTPUT_DIR / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    summary = save_outputs(portfolios,metadata,folder)
    print(summary.to_string())
    print(f"Outputs: {folder.resolve()}")

if __name__ == "__main__":
    main()
