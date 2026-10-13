"""Isolated fixture integration, explicitly NOT the real-data smoke/benchmark.

Run separately: python -m unittest tests.test_pipeline_integration -v
All generated fixture artifacts remain in ignored tmp/, never results/metrics.csv.
"""
from contextlib import redirect_stdout
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.common as common
import src.data as data
import src.evaluate as evaluation
import src.train_ft_transformer as ft
import src.train_lightgbm as lgb


class FixtureIntegration(unittest.TestCase):
    def test_saved_models_and_paired_evaluation(self):
        if not ft.torch.cuda.is_available():
            self.skipTest("CUDA is required; no CPU substitution")
        root = common.ROOT
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        temporary = (root / "tmp" / ("fixture_integration_" + stamp)).resolve()
        self.assertTrue(temporary.is_relative_to(root / "tmp"))
        temporary.mkdir(parents=True)
        (temporary / "configs").mkdir()
        config = common.read_json(root / "configs" / "mvp.json")
        config["lightgbm"]["n_estimators"] = 4
        config["lightgbm"]["min_child_samples"] = 10
        config["lightgbm"]["n_jobs"] = 1
        # The smoke training code fixes its own estimator count at 80.
        config["lightgbm"]["early_stopping_rounds"] = 3
        common.write_json(temporary / "configs" / "mvp.json", config)
        n = 400
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({name: rng.uniform(0, 10, n) for name in data.NUMERIC})
        for name in data.CATEGORICAL:
            frame[name] = np.where(np.arange(n) % 3 == 0, "a", "b")
        frame.loc[::7, "dist1"] = np.nan
        frame["TransactionID"] = np.arange(n)
        frame["TransactionDT"] = np.arange(n)
        frame["isFraud"] = (np.arange(n) % 5 == 0).astype(int)
        csv_path = temporary / "HANDCRAFTED_FIXTURE_NOT_KAGGLE.csv"
        frame.to_csv(csv_path, index=False)
        # Patch all storage roots; normal benchmark paths cannot be touched.
        with patch.object(common, "ROOT", temporary), patch.object(data, "ROOT", temporary), \
             patch.object(lgb, "ROOT", temporary), patch.object(ft, "ROOT", temporary), \
             patch.object(evaluation, "ROOT", temporary), \
             (temporary / "FIXTURE_ONLY_stdout.log").open("w", encoding="utf-8") as log, \
             redirect_stdout(log):
            data.prepare(csv_path, "smoke", config)
            a = lgb.train("smoke")
            b = ft.train("smoke")
            self.assertEqual(a["status"], "COMPLETED")
            self.assertEqual(b["status"], "COMPLETED")
            table = evaluation.evaluate("smoke")
            self.assertEqual(len(table), 2)
            self.assertTrue((table.n_test == 60).all())
            self.assertEqual(common.read_json(temporary / "results" / "smoke" / "metric_recomputation.json")["status"], "PASS")
            for filename in ["precision_recall.png", "roc.png", "recall_budget.png", "model_comparison.png"]:
                self.assertGreater((temporary / "results" / "smoke" / filename).stat().st_size, 1000)
            # Fingerprint guard must reject tampering before evaluating test.
            model_path = temporary / "models" / "smoke" / "lightgbm.txt"
            with model_path.open("a", encoding="utf-8") as file:
                file.write("\nTAMPER_PROBE\n")
            with self.assertRaisesRegex(ValueError, "model fingerprint mismatch"):
                evaluation.evaluate("smoke")
        marker = {"status": "PASS", "kind": "HANDCRAFTED_FIXTURE_ONLY_NOT_BENCHMARK",
                  "real_data_smoke": False, "artifact_root": str(temporary),
                  "benchmark_source_sha256": common.source_fingerprints(),
                  "test_source_sha256": common.sha256(Path(__file__)),
                  "checks": ["train preprocessing", "LightGBM training/serialization", "FT CUDA epoch/checkpoint",
                             "identical test rows", "all plots", "metric disk recomputation", "tamper rejection"]}
        common.write_json(root / "reports" / "fixture_integration.json", marker)


if __name__ == "__main__":
    unittest.main(verbosity=2)
