"""Single source of truth for research assumptions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Codes use Yahoo Finance suffixes. Names are retained in every output for presentation use.
UNIVERSE = {
    "沪深300ETF": "510300.SS",
    "创业板50ETF": "159949.SZ",
    "中证500ETF": "510500.SS",
    "红利ETF": "510880.SS",
    "黄金ETF": "518880.SS",
    "国债ETF": "511010.SS",
    "纳斯达克100ETF": "513100.SS",
}

BENCHMARK = "沪深300ETF"
START_DATE = "2021-01-01"
END_DATE = "2026-09-17"  # Fixed, completed session; change explicitly for new research.
MOMENTUM_WINDOWS = (20, 60, 120)
TOP_N = 2
ONE_WAY_COST = 0.0005  # 5 bps for each unit of traded notional
TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 1_000_000.0
