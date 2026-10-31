import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
pca = importlib.import_module("07_pca")


def example_features():
    values = np.random.default_rng(42).normal(size=(16, 4))
    values = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    result = pd.DataFrame(values, columns=pca.FEATURES)
    result.insert(0, "year", 2024)
    result.insert(0, "municipality", [f"Municipality {number}" for number in range(len(result))])
    return result


class PCATests(unittest.TestCase):
    def setUp(self):
        self.features = example_features()

    def test_variance_matches_covariance_eigenvalues(self):
        _, _, variance = pca.reduce_dimensions(self.features)
        covariance = np.cov(self.features[pca.FEATURES].to_numpy(), rowvar=False)
        expected = np.linalg.eigvalsh(covariance)[::-1]
        np.testing.assert_allclose(variance.explained_variance, expected, atol=1e-12)
        self.assertAlmostEqual(variance.explained_variance_ratio.sum(), 1)

    def test_smallest_component_count_meeting_threshold(self):
        scores, _, variance = pca.reduce_dimensions(self.features, 0.90)
        retained = int(variance.retained.sum())
        self.assertGreaterEqual(variance.cumulative_variance_ratio.iloc[retained - 1], 0.90)
        if retained > 1:
            self.assertLess(variance.cumulative_variance_ratio.iloc[retained - 2], 0.90)
        self.assertEqual(len(scores.columns) - len(pca.IDS), retained)
        pd.testing.assert_frame_equal(scores[pca.IDS], self.features[pca.IDS])

    def test_full_reconstruction_and_orthonormal_weights(self):
        scores, components, _ = pca.reduce_dimensions(self.features, 1)
        names = [column for column in components if column.startswith("PC")]
        weights = components[names].to_numpy()
        restored = scores[names].to_numpy() @ weights.T + components.center.to_numpy()
        np.testing.assert_allclose(restored, self.features[pca.FEATURES], atol=1e-12)
        np.testing.assert_allclose(weights.T @ weights, np.eye(len(names)), atol=1e-12)
        largest = weights[np.abs(weights).argmax(axis=0), np.arange(len(names))]
        self.assertTrue((largest > 0).all())

    def test_reconstruction_loss_matches_discarded_variance(self):
        scores, components, variance = pca.reduce_dimensions(self.features, 0.01)
        names = list(scores.columns[2:])
        restored = scores[names].to_numpy() @ components[names].to_numpy().T + components.center.to_numpy()
        loss = ((self.features[pca.FEATURES].to_numpy() - restored) ** 2).sum()
        expected = variance.loc[~variance.retained, "explained_variance"].sum() * (len(scores) - 1)
        self.assertAlmostEqual(loss, expected)

    def test_rank_deficiency_and_constant_zero_column(self):
        values = self.features[pca.FEATURES[0]].copy()
        for column in pca.FEATURES[:3]:
            self.features[column] = values
        self.features[pca.FEATURES[3]] = 0.0
        scores, components, variance = pca.reduce_dimensions(self.features)
        self.assertEqual(list(scores.columns), pca.IDS + ["PC1"])
        self.assertEqual(len(variance), 1)
        self.assertAlmostEqual(variance.explained_variance_ratio.iloc[0], 1)
        self.assertAlmostEqual(components.PC1.iloc[-1], 0)

    def test_invalid_thresholds_are_rejected(self):
        for threshold in [0, -1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                pca.reduce_dimensions(self.features, threshold)

    def test_missing_and_nonfinite_values_are_rejected(self):
        for value in [float("nan"), float("inf")]:
            changed = self.features.copy()
            changed.loc[0, pca.FEATURES[0]] = value
            with self.assertRaises(ValueError):
                pca.reduce_dimensions(changed)

    def test_unscaled_and_near_zero_columns_are_rejected(self):
        for factor in [2, 1e-10]:
            changed = self.features.copy()
            changed[pca.FEATURES[0]] *= factor
            with self.assertRaises(ValueError):
                pca.reduce_dimensions(changed)

    def test_all_constant_features_are_rejected(self):
        self.features[pca.FEATURES] = 0.0
        with self.assertRaises(ValueError):
            pca.reduce_dimensions(self.features)

    def test_duplicate_identifiers_and_mixed_years_are_rejected(self):
        duplicate = pd.concat([self.features, self.features.iloc[[0]]], ignore_index=True)
        mixed = self.features.copy()
        mixed.loc[0, "year"] = 2011
        for invalid in [duplicate, mixed]:
            with self.assertRaises(ValueError):
                pca.reduce_dimensions(invalid)


if __name__ == "__main__":
    unittest.main()
