"""Immutable JSON snapshots with hashes; retain adjusted open/close and volume."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from .config import RAW_DATA_DIR, START_DATE, END_DATE, UNIVERSE

@dataclass
class MarketData:
    open: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    metadata: dict

def _hash(content):
    return hashlib.sha256(content).hexdigest()

def load_market_data(snapshot=None, refresh=False):
    root = RAW_DATA_DIR / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / "active.json"
    if snapshot is None and not refresh and pointer.exists():
        snapshot = root / json.loads(pointer.read_text(encoding="utf-8"))["snapshot"]
    if snapshot is None:
        end = END_DATE
        if end is None or pd.Timestamp(end).date() >= datetime.now().date():
            raise ValueError("Set END_DATE to a completed session, before today.")
        frames = {}
        for name, ticker in UNIVERSE.items():
            code, suffix = ticker.split(".")
            params = dict(secid=f"{'1' if suffix == 'SS' else '0'}.{code}", klt=101, fqt=1,
                beg=START_DATE.replace("-",""), end=end.replace("-",""), lmt=10000,
                fields1="f1,f2,f3,f4,f5,f6", fields2="f51,f52,f53,f54,f55,f56,f57,f58")
            response = requests.get("https://push2his.eastmoney.com/api/qt/stock/kline/get",
                                    params=params, timeout=30)
            response.raise_for_status()
            bars = (response.json().get("data") or {}).get("klines", [])
            if not bars:
                raise ValueError(f"No data for {ticker}.")
            frames[name] = [x.split(",") for x in bars]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        metadata = dict(source="Eastmoney public daily kline", adjustment="fqt=1 forward-adjusted",
            downloaded_at_utc=stamp, requested_start=START_DATE, requested_end=end, universe=UNIVERSE,
            fields=["date","open","close","high","low","volume","amount","amplitude"])
        payload = json.dumps(dict(metadata=metadata, bars=frames), ensure_ascii=False).encode("utf-8")
        snapshot = root / f"market_{stamp}.json"
        # Validate first; failed downloads or validation never replace the active snapshot.
        market = decode_snapshot(payload)
        snapshot.write_bytes(payload)
        snapshot.with_suffix(".sha256").write_text(_hash(payload), encoding="ascii")
        pointer.write_text(json.dumps(dict(snapshot=snapshot.name)), encoding="utf-8")
    else:
        snapshot = Path(snapshot)
        payload = snapshot.read_bytes()
        checksum = snapshot.with_suffix(".sha256")
        if checksum.exists() and checksum.read_text(encoding="ascii").strip() != _hash(payload):
            raise ValueError("Snapshot checksum mismatch; data may have changed.")
        market = decode_snapshot(payload)
    market.metadata.update(snapshot_path=str(Path(snapshot).resolve()), sha256=_hash(payload))
    return market

def decode_snapshot(payload):
    doc = json.loads(payload)
    meta = dict(doc["metadata"])
    if meta["adjustment"] != "fqt=1 forward-adjusted":
        raise ValueError("Open and close must share the documented adjustment basis.")
    if meta["universe"] != UNIVERSE:
        raise ValueError("Snapshot universe differs from config; choose a matching snapshot.")
    if meta["requested_start"] != START_DATE or meta["requested_end"] != END_DATE:
        raise ValueError("Snapshot date settings differ; use --refresh or restore config.")
    frames = {}
    for name in UNIVERSE:
        f = pd.DataFrame(doc["bars"][name], columns=meta["fields"])
        f["date"] = pd.to_datetime(f["date"])
        if f.date.isna().any():
            raise ValueError(f"Missing dates: {name}")
        if f.date.duplicated().any():
            raise ValueError(f"Duplicate dates: {name}")
        f = f.set_index("date").sort_index()
        for col in ("open","close","volume"):
            f[col] = pd.to_numeric(f[col], errors="raise")
            if not np.isfinite(f[col]).all() or (f[col] < 0).any():
                raise ValueError(f"Invalid {col}: {name}")
        if (f[["open","close"]] <= 0).any().any():
            raise ValueError(f"Nonpositive prices: {name}")
        f = f.loc[START_DATE:END_DATE]
        if f.empty:
            raise ValueError(f"No rows in requested range: {name}")
        frames[name] = f
    matrices = {col: pd.concat({n:f[col] for n,f in frames.items()}, axis=1, sort=True).sort_index()
                for col in ("open","close","volume")}
    # Missing bars remain missing. Signal needs a complete window; marking may carry forward.
    meta["actual_start"] = str(matrices["close"].index.min().date())
    meta["actual_end"] = str(matrices["close"].index.max().date())
    meta["missing_bars"] = {n:int(v) for n,v in matrices["close"].isna().sum().items()}
    return MarketData(matrices["open"], matrices["close"], matrices["volume"], meta)

def import_csv_snapshot(directory):
    """Explicit import of same-basis adjusted Open/Close and unadjusted Volume."""
    frames, hashes = {}, {}
    for name, ticker in UNIVERSE.items():
        path = Path(directory) / f"{ticker}.csv"
        hashes[ticker] = _hash(path.read_bytes())
        frame = pd.read_csv(path)
        frame.columns = [c.strip().lower() for c in frame.columns]
        required = ["date","open","close","volume"]
        if not set(required).issubset(frame.columns):
            raise ValueError(f"{path}: requires Date, Open, Close, Volume; close-only input is unsafe.")
        frames[name] = frame[required].astype(str).values.tolist()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    meta = dict(source="User CSV import; adjustment basis asserted by importer",
                adjustment="fqt=1 forward-adjusted", imported_at_utc=stamp,
                requested_start=START_DATE, requested_end=END_DATE, universe=UNIVERSE,
                fields=["date","open","close","volume"], input_sha256=hashes)
    payload = json.dumps(dict(metadata=meta,bars=frames),ensure_ascii=False).encode("utf-8")
    decode_snapshot(payload)
    root = RAW_DATA_DIR / "snapshots"
    root.mkdir(parents=True,exist_ok=True)
    path = root / f"import_{stamp}.json"
    path.write_bytes(payload)
    path.with_suffix(".sha256").write_text(_hash(payload),encoding="ascii")
    (root / "active.json").write_text(json.dumps(dict(snapshot=path.name)),encoding="utf-8")
    return path
