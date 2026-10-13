"""Tiny handcrafted fixtures validate contracts, not fraud benchmark performance."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.common import ROOT, probability_check, read_json
from src.data import CATEGORICAL, NUMERIC, Preprocessor, chronological_split, lightgbm_frame
from src.evaluate import metrics, rank_indices, top_budget


def fixture():
    return pd.DataFrame({**{n: [1.0, np.nan, 3.0, 5.0] for n in NUMERIC},
                         **{n: ["a", "b", None, "a"] for n in CATEGORICAL}})


class Contracts(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(read_json(ROOT / "configs" / "mvp.json"))
        self.config["numeric_log1p"] = []

    def test_chronology_and_proportions(self):
        frame = pd.DataFrame({"TransactionID": range(20), "TransactionDT": list(range(19, -1, -1)),
                              "isFraud": [0, 1] * 10})
        train, validation, test = chronological_split(frame)
        self.assertEqual([len(train), len(validation), len(test)], [14, 3, 3])
        self.assertEqual(train.TransactionDT.tolist(), list(range(14)))
        self.assertLess(train.TransactionDT.max(), validation.TransactionDT.min())
        self.assertLess(validation.TransactionDT.max(), test.TransactionDT.min())
        self.assertEqual(len(set(train.TransactionID) & set(test.TransactionID)), 0)

    def test_training_only_state_and_missing_unknown(self):
        train = fixture()
        preprocessor = Preprocessor(self.config).fit(train)
        before = copy.deepcopy(preprocessor.state)
        heldout = fixture()
        heldout["TransactionAmt"] = [100000.0, np.nan, 300000.0, 500000.0]
        heldout["ProductCD"] = ["future", None, "a", "b"]
        arrays = preprocessor.transform(heldout)
        self.assertEqual(before, preprocessor.state)
        self.assertEqual(preprocessor.state["medians"][0], 3.0)
        column = preprocessor.state["categorical"].index("ProductCD")
        self.assertEqual(arrays["x_cat"][:, column].tolist(), [0, 1, 2, 0])
        self.assertEqual(arrays["x_lgb_num"][1, 0], 3.0)
        self.assertEqual(arrays["x_num"][1, len(NUMERIC)], 1.0)
        self.assertTrue(np.isfinite(arrays["x_num"]).all())

    def test_training_only_constant_filter(self):
        frame = fixture()
        frame["dist2"] = np.nan
        frame["card4"] = "fixed"
        state = Preprocessor(self.config).fit(frame).state
        self.assertIn("dist2", state["dropped_training_constant_or_missing"])
        self.assertIn("card4", state["dropped_training_constant_or_missing"])
        self.assertNotIn("TransactionID", state["numeric"] + state["categorical"])
        self.assertNotIn("TransactionDT", state["numeric"] + state["categorical"])

    def test_category_cap_and_matching_lightgbm_codes(self):
        self.config["categorical_max_values"] = 1
        self.config["categorical_min_count"] = 1
        preprocessor = Preprocessor(self.config).fit(fixture())
        arrays = preprocessor.transform(fixture())
        frame = lightgbm_frame(arrays, preprocessor.state)
        self.assertEqual(preprocessor.state["cat_cardinalities"], [3] * len(CATEGORICAL))
        self.assertEqual(frame.ProductCD.astype(int).tolist(), arrays["x_cat"][:, 0].tolist())

    def test_hand_calculated_budget_and_metric_values(self):
        y = np.zeros(100, dtype=int)
        y[[0, 50]] = 1
        probability = np.linspace(1, 0, 100)
        result = metrics(y, probability, np.arange(100))
        self.assertEqual(result["recall_top_1pct"], 0.5)
        self.assertEqual(result["precision_top_1pct"], 1.0)
        self.assertEqual(result["recall_top_5pct"], 0.5)
        self.assertEqual(result["top_1pct_count"], 1)
        self.assertEqual(result["top_5pct_count"], 5)
        small = metrics([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.7], [1, 2, 3, 4])
        self.assertAlmostEqual(small["average_precision"], 1.0)
        self.assertAlmostEqual(small["roc_auc"], 1.0)
        self.assertAlmostEqual(small["brier_score"], 0.075)

    def test_ties_and_ceil_are_label_independent(self):
        ids = np.array([30, 10, 20])
        self.assertEqual(rank_indices([0.5, 0.5, 0.5], ids).tolist(), [1, 2, 0])
        result = top_budget(np.array([0, 1, 0]), np.array([0.5] * 3), ids, 0.01)
        self.assertEqual(result["k"], 1)
        self.assertEqual(result["recall"], 1.0)

    def test_invalid_predictions_and_single_class_rejected(self):
        for probability in [[np.nan], [-0.01], [1.01], [[0.3]]]:
            with self.assertRaises(ValueError):
                probability_check(probability)
        with self.assertRaises(ValueError):
            metrics([0, 0], [0.1, 0.2], [1, 2])
        with self.assertRaises(ValueError):
            metrics([0, 1], [0.1, 0.2], [1, 1])

    def test_no_cuda_never_falls_back(self):
        from src.train_ft_transformer import require_cuda
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is forbidden"):
                require_cuda()

    def test_lightgbm_native_categorical_library_integration(self):
        import lightgbm as lgb
        preprocessor = Preprocessor(self.config).fit(fixture())
        arrays = preprocessor.transform(fixture())
        frame = lightgbm_frame(arrays, preprocessor.state)
        labels = np.array([0, 1, 0, 1])
        model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1, num_leaves=3,
                                  verbosity=-1, random_state=42, n_jobs=1)
        model.fit(frame, labels, categorical_feature=preprocessor.state["categorical"])
        probability_check(model.predict_proba(frame)[:, 1])
        self.assertEqual(model.n_features_in_, frame.shape[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
