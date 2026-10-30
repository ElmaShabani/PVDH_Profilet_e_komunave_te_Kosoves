import importlib
import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
scaling = importlib.import_module("06_feature_scaling")


def example_indicators():
    records = []
    examples = [
        ("A", 100, 40, 20, 10, 100, "no_specific_coverage_warning"),
        ("B", 200, 150, 30, 20, 110, "no_specific_coverage_warning"),
        ("C", 80, 0, 10, 30, 90, "incomplete_participation"),
    ]
    for name, total, urban, children, older, sex_ratio, coverage in examples:
        for settlement, count in [("all", total), ("urban", urban), ("rural", total - urban)]:
            records.append({
                "municipality": name, "year": 2024, "settlement_type": settlement,
                "total_population": count, "children_share_pct": children,
                "older_share_pct": older, "working_age_sex_ratio": sex_ratio,
                "coverage_status": coverage, "population_cells_complete": True,
            })
    return pd.DataFrame(records)


class FeatureScalingTests(unittest.TestCase):
    def setUp(self):
        self.indicators = example_indicators()

    def test_one_row_per_municipality_and_urban_share(self):
        selected, excluded = scaling.select_features(self.indicators, 2024)
        self.assertEqual(selected.municipality.tolist(), ["A", "B"])
        self.assertEqual(selected.urban_share_pct.tolist(), [40, 75])
        self.assertEqual(excluded.municipality.tolist(), ["C"])
        self.assertEqual(excluded.exclusion_reason.iloc[0], "coverage_warning")

    def test_standardization_and_inverse_transform(self):
        selected, _ = scaling.select_features(self.indicators, 2024)
        standardized, parameters = scaling.standardize_features(selected)
        values = standardized[scaling.FEATURES]
        self.assertTrue(values.mean().abs().lt(1e-12).all())
        self.assertTrue(values.std(ddof=0).sub(1).abs().lt(1e-12).all())
        parameters = parameters.set_index("feature")
        restored = values.mul(parameters.scale).add(parameters["mean"])
        self.assertTrue(restored.sub(selected[scaling.FEATURES]).abs().lt(1e-12).all().all())

    def test_constant_feature_has_zero_output(self):
        self.indicators["working_age_sex_ratio"] = 100
        selected, _ = scaling.select_features(self.indicators, 2024)
        standardized, parameters = scaling.standardize_features(selected)
        self.assertTrue(standardized.working_age_sex_ratio.eq(0).all())
        parameter = parameters.set_index("feature").loc["working_age_sex_ratio"]
        self.assertEqual(parameter["std"], 0)
        self.assertEqual(parameter.scale, 1)
        self.assertTrue(parameter.is_constant)

    def test_missing_feature_is_excluded_without_imputation(self):
        self.indicators.loc[self.indicators.municipality.eq("A"), "older_share_pct"] = float("nan")
        selected, excluded = scaling.select_features(self.indicators, 2024)
        self.assertEqual(selected.municipality.tolist(), ["B"])
        reason = excluded.set_index("municipality").loc["A", "exclusion_reason"]
        self.assertIn("missing_or_invalid_feature", reason)
        with self.assertRaises(ValueError):
            scaling.standardize_features(selected)

    def test_zero_population_is_excluded(self):
        self.indicators.loc[self.indicators.municipality.eq("A"), "total_population"] = 0
        selected, excluded = scaling.select_features(self.indicators, 2024)
        self.assertNotIn("A", selected.municipality.tolist())
        self.assertIn("undefined_population_total", excluded.set_index("municipality").loc["A", "exclusion_reason"])

    def test_incomplete_input_cells_are_excluded(self):
        mask = self.indicators.municipality.eq("A") & self.indicators.settlement_type.eq("urban")
        self.indicators.loc[mask, "population_cells_complete"] = False
        selected, excluded = scaling.select_features(self.indicators, 2024)
        self.assertNotIn("A", selected.municipality.tolist())
        self.assertIn("missing_source_cells", excluded.set_index("municipality").loc["A", "exclusion_reason"])

    def test_inconsistent_totals_are_rejected(self):
        self.indicators.loc[0, "total_population"] += 1
        with self.assertRaises(ValueError):
            scaling.select_features(self.indicators, 2024)

    def test_missing_settlement_and_duplicate_rows_are_rejected(self):
        for invalid in [self.indicators.iloc[1:], pd.concat([self.indicators, self.indicators.iloc[[0]]])]:
            with self.assertRaises(ValueError):
                scaling.select_features(invalid, 2024)

    def test_scaler_does_not_mix_census_years(self):
        selected, _ = scaling.select_features(self.indicators, 2024)
        selected.loc[0, "year"] = 2011
        with self.assertRaises(ValueError):
            scaling.standardize_features(selected)


if __name__ == "__main__":
    unittest.main()
