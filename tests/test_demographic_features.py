import importlib
import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
features = importlib.import_module("05_demographic_features")


def example_population():
    records = []
    for settlement, factor in [("urban", 1), ("rural", 2)]:
        for sex, count in [("male", 1), ("female", 2)]:
            for age in features.AGE_GROUPS:
                records.append({
                    "municipality": "Example",
                    "year": 2024,
                    "settlement_type": settlement,
                    "sex": sex,
                    "age_group": age,
                    "age_band": features.AGE_BANDS[age],
                    "population": factor * count,
                    "coverage_status": "no_specific_coverage_warning",
                })
    return pd.DataFrame(records).astype(features.DTYPES | {
        "age_band": "string", "coverage_status": "string"
    })


class DemographicFeatureTests(unittest.TestCase):
    def setUp(self):
        self.population = example_population()

    def indicators(self):
        _, result = features.build_demographic_tables(self.population)
        return result.set_index("settlement_type")

    def test_known_formulas(self):
        urban = self.indicators().loc["urban"]
        self.assertEqual(urban.children_population, 9)
        self.assertEqual(urban.working_age_population, 30)
        self.assertEqual(urban.older_population, 15)
        self.assertEqual(urban.total_population, 54)
        self.assertAlmostEqual(urban.children_share_pct, 100 / 6)
        self.assertAlmostEqual(urban.ageing_index, 15 / 9 * 100)
        self.assertEqual(urban.dependency_ratio, 80)
        self.assertEqual(urban.sex_ratio, 50)
        self.assertEqual(urban.working_age_sex_ratio, 50)

    def test_municipality_total_reconciles(self):
        result = self.indicators()
        for column in ["total_population", "children_population", "working_age_population", "older_population"]:
            self.assertEqual(result.loc["all", column], result.loc[["urban", "rural"], column].sum())
        bands, _ = features.build_demographic_tables(self.population)
        self.assertEqual(len(bands), 18)
        self.assertEqual(result.loc["all", "population_cells_expected"], 72)

    def test_partial_missingness_does_not_become_partial_total(self):
        self.population.loc[0, "population"] = pd.NA
        result = self.indicators()
        for settlement in ["urban", "all"]:
            self.assertTrue(pd.isna(result.loc[settlement, "children_population"]))
            self.assertTrue(pd.isna(result.loc[settlement, "total_population"]))
            self.assertTrue(pd.isna(result.loc[settlement, "dependency_ratio"]))
            self.assertFalse(result.loc[settlement, "population_cells_complete"])
        self.assertEqual(result.loc["urban", "working_age_population"], 30)
        self.assertEqual(result.loc["urban", "working_age_sex_ratio"], 50)
        self.assertEqual(result.loc["rural", "total_population"], 108)

    def test_zero_counts_remain_zero_and_ratios_are_undefined(self):
        self.population["population"] = pd.Series(0, index=self.population.index, dtype="Int64")
        result = self.indicators()
        self.assertTrue(result.total_population.eq(0).all())
        self.assertTrue(result.population_cells_complete.all())
        ratios = [column for column in result if column.endswith("_pct") or "ratio" in column or column == "ageing_index"]
        self.assertTrue(result[ratios].isna().all().all())

    def test_zero_children_only_makes_ageing_index_undefined(self):
        self.population.loc[self.population.age_band.eq("0-14"), "population"] = 0
        result = self.indicators()
        self.assertTrue(result.ageing_index.isna().all())
        self.assertTrue(result.children_share_pct.eq(0).all())
        self.assertTrue(result.dependency_ratio.eq(50).all())

    def test_all_missing_counts_stay_missing(self):
        self.population["population"] = pd.Series(pd.NA, index=self.population.index, dtype="Int64")
        result = self.indicators()
        self.assertTrue(result.total_population.isna().all())
        self.assertTrue(result.population_cells_present.eq(0).all())

    def test_coverage_warnings_survive_aggregation(self):
        self.population["coverage_status"] = "incomplete_participation"
        bands, indicators = features.build_demographic_tables(self.population)
        self.assertTrue(bands.coverage_status.eq("incomplete_participation").all())
        self.assertTrue(indicators.coverage_status.eq("incomplete_participation").all())

    def test_incomplete_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            features.build_demographic_tables(self.population.iloc[1:])

    def test_wrong_age_band_is_rejected(self):
        self.population.loc[0, "age_band"] = "65+"
        with self.assertRaises(ValueError):
            features.build_demographic_tables(self.population)


if __name__ == "__main__":
    unittest.main()
