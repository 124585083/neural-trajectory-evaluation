"""Mathematical and missing-identity checks for the descriptive audit only."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("descriptive_audit", ROOT / "descriptive_audit.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class DescriptiveAuditTests(unittest.TestCase):
    def test_exact_zero_not_tolerance(self):
        x = mod.endpoint_statistics([-1e-20, 0, 1e-20])
        self.assertEqual((x["n_down"], x["n_exact_zero"], x["n_up"]), (1, 1, 1))

    def test_opposing_errors_cancel_but_variance_remains(self):
        x = mod.endpoint_statistics([-3, -1, 1, 3])
        self.assertEqual(x["mean_error"], 0)
        self.assertEqual(x["mean_abs_error"], 2)
        self.assertEqual(x["cancellation_fraction"], 1)
        self.assertEqual(x["within_variance_ddof0"], 5)

    def test_variance_identity_shifted(self):
        x = mod.endpoint_statistics(np.array([-2., -1., 3., 8.]))
        self.assertAlmostEqual(x["mean_square_error"], x["square_mean_error"] + x["within_variance_ddof0"], places=12)

    def test_saved_members_and_unknown_identity(self):
        d = pd.read_csv(ROOT / "results/descriptive/endpoint_condition_audit.csv")
        for animal, n in [("mahler", 7407), ("perle", 84873)]:
            a = d[d.animal == animal]
            self.assertEqual(len(a), 79)
            self.assertEqual(int(a.n_members.sum()), n)
            self.assertEqual(int(a.loc[a.condition_id == 59920, "n_members"].iloc[0]), 0)
            self.assertTrue(np.isnan(a.loc[a.condition_id == 59920, "mean_error"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
