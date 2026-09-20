"""Out-of-sample selection protocol tests."""

import json
import unittest

import pandas as pd

from src.data import decode_snapshot
from src.out_of_sample import run_out_of_sample
from test_pipeline import fixture_payload


def long_fixture():
    payload = fixture_payload()
    dates = pd.bdate_range("2021-01-01", "2024-06-28")
    for index, name in enumerate(payload["metadata"]["universe"]):
        payload["bars"][name] = [
            [str(date.date()), str(100 + i * (index + 1) / 200), str(100.05 + i * (index + 1) / 200), "1000"]
            for i, date in enumerate(dates)
        ]
    return payload


class OutOfSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.metadata, _, _ = run_out_of_sample(
            decode_snapshot(json.dumps(long_fixture()))
        )

    def test_research_selection_uses_twelve_candidates(self):
        research = self.results[
            self.results["case_type"].eq("strategy")
            & self.results["period"].eq("research")
        ]
        self.assertEqual(len(research), 12)
        self.assertEqual(self.metadata["candidate_count"], 12)

    def test_validation_restarts_after_split(self):
        self.assertLess(self.metadata["research_end"], self.metadata["validation_start"])
        validation = self.results[self.results["period"].eq("validation")]
        self.assertEqual(len(validation), 14)
        self.assertTrue(validation["total_fees"].gt(0).all())

    def test_selected_case_is_research_sharpe_maximum(self):
        research = self.results[
            self.results["case_type"].eq("strategy")
            & self.results["period"].eq("research")
        ]
        selected = research.sort_values(
            ["sharpe_ratio_rf0", "annual_return", "window", "top_n"],
            ascending=[False, False, True, True],
        ).iloc[0]
        self.assertEqual(str(selected["window"]), self.metadata["selected_window"])
        self.assertEqual(int(selected["top_n"]), self.metadata["selected_top_n"])


if __name__ == "__main__":
    unittest.main()
