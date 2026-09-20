"""Sensitivity-grid coverage and sample-consistency tests."""

import json
import unittest

from src.data import decode_snapshot
from src.experiments import CASH_RULES, COSTS_BPS, TOP_NS, WINDOW_SETS, run_grid
from test_pipeline import fixture_payload


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.metadata = run_grid(decode_snapshot(json.dumps(fixture_payload())))

    def test_every_predeclared_strategy_case_runs_once(self):
        strategy = self.results[self.results["case_type"].eq("strategy")]
        expected = len(WINDOW_SETS) * len(TOP_NS) * len(COSTS_BPS) * len(CASH_RULES)
        self.assertEqual(len(strategy), expected)
        self.assertEqual(len(strategy.drop_duplicates(["window", "top_n", "cost_bps", "cash_rule"])), expected)

    def test_benchmarks_are_present(self):
        benchmark = self.results[self.results["case_type"].eq("benchmark")]
        self.assertEqual(set(benchmark["window"]), {"equal_weight", "csi300"})

    def test_all_results_share_one_sample(self):
        self.assertEqual(self.metadata["common_start"], "2021-07-01")
        self.assertTrue(self.results["annual_return"].notna().all())


if __name__ == "__main__":
    unittest.main()
