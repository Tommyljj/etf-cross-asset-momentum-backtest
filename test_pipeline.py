"""Offline pipeline tests use synthetic data and isolated temporary directories."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import config
from src.data import decode_snapshot, load_market_data
from src.run import main

def fixture_payload():
    dates = pd.bdate_range("2021-01-01",periods=160)
    bars = {}
    for i,name in enumerate(config.UNIVERSE):
        bars[name] = [[str(d.date()),str(100+j*(i+1)/100),str(100+j*(i+1)/100+.05),"1000"]
                      for j,d in enumerate(dates)]
    return dict(metadata=dict(source="SYNTHETIC TEST ONLY",adjustment="fqt=1 forward-adjusted",
        requested_start=config.START_DATE,requested_end=config.END_DATE,universe=config.UNIVERSE,
        fields=["date","open","close","volume"]),bars=bars)

class PipelineTests(unittest.TestCase):
    def test_duplicate_dates_rejected(self):
        p=fixture_payload()
        name=next(iter(config.UNIVERSE))
        p["bars"][name].append(p["bars"][name][0])
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            decode_snapshot(json.dumps(p))
    def test_missing_dates_preserved(self):
        p=fixture_payload()
        name=next(iter(config.UNIVERSE))
        p["bars"][name].pop(10)
        data=decode_snapshot(json.dumps(p))
        self.assertTrue(pd.isna(data.close.iloc[10,0]))
        self.assertTrue(pd.isna(data.open.iloc[10,0]))
    def test_offline_snapshot_and_checksum(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"snapshot.json"
            content=json.dumps(fixture_payload()).encode()
            path.write_bytes(content)
            path.with_suffix(".sha256").write_text(hashlib.sha256(content).hexdigest())
            with patch("src.data.RAW_DATA_DIR",Path(directory)), patch("src.data.requests.get") as request:
                data=load_market_data(path)
                request.assert_not_called()
                self.assertEqual(data.metadata["sha256"],hashlib.sha256(content).hexdigest())
                path.write_bytes(content+b" ")
                with self.assertRaisesRegex(ValueError,"checksum"):
                    load_market_data(path)
    def test_failed_download_does_not_activate_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.data.RAW_DATA_DIR",Path(directory)), patch(
                    "src.data.requests.get",side_effect=RuntimeError("test network failure")):
                with self.assertRaises(RuntimeError):
                    load_market_data(refresh=True)
            self.assertFalse((Path(directory)/"snapshots"/"active.json").exists())
    def test_end_to_end_four_portfolios_report(self):
        data=decode_snapshot(json.dumps(fixture_payload()))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/"report"
            with patch("src.run.load_market_data",return_value=data), patch(
                    "sys.argv",["run","--output-dir",str(folder)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            summary=pd.read_csv(folder/"performance_summary.csv",index_col=0)
            self.assertEqual(len(summary),4)
            self.assertGreater((folder/"equity_curve.png").stat().st_size,1000)
            for name in summary.index:
                daily=pd.read_csv(folder/name/"daily_returns.csv")
                holdings=pd.read_csv(folder/name/"daily_holdings.csv")
                np.testing.assert_allclose(holdings.groupby("date").market_value.sum().to_numpy()
                    +daily.cash,daily.nav,atol=1e-6)
            ranks=pd.read_csv(folder/"momentum"/"monthly_rankings.csv")
            self.assertTrue(ranks.groupby("decision_date").size().eq(7).all())
            self.assertTrue((folder/"run_metadata.json").exists())

if __name__=="__main__":
    unittest.main()
