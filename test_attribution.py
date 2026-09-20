import unittest

import numpy as np

from src.attribution import build_attribution
from src.data import load_market_data


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = build_attribution(load_market_data())

    def test_pnl_reconciles(self):
        self.assertAlmostEqual(self.bundle["reconciliation"]["difference"], 0.0, places=6)

    def test_monthly_returns_match_daily_compounding(self):
        monthly = self.bundle["monthly"]["strategy_return"]
        daily = self.bundle["daily"]["strategy_return"]
        self.assertAlmostEqual(float((1 + monthly).prod()), float((1 + daily).prod()), places=12)

    def test_drawdown_is_valid(self):
        episode = self.bundle["episode"]
        self.assertLessEqual(episode["max_drawdown"], 0)
        self.assertTrue(np.isfinite(episode["max_drawdown"]))


if __name__ == "__main__":
    unittest.main()
