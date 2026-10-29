from pathlib import Path

import pandas as pd

from validate_data import AGE_GROUPS, DTYPES, KEY, validate_population


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/analysis/population_analysis_ready.csv"
OUTPUT = ROOT / "data/processed/analysis"
BASE = ["municipality", "year", "settlement_type"]
AGE_BANDS = {
    **dict.fromkeys(AGE_GROUPS[:3], "0-14"),
    **dict.fromkeys(AGE_GROUPS[3:13], "15-64"),
    **dict.fromkeys(AGE_GROUPS[13:], "65+"),
}
BAND_SIZES = {"0-14": 3, "15-64": 10, "65+": 5}


def load_population():
    if not INPUT.is_file():
        raise FileNotFoundError("Run phase1/04_feature_preparation.py first.")
    columns = KEY + ["population", "coverage_status", "age_band"]
    population = pd.read_csv(
        INPUT,
        usecols=columns,
        dtype=DTYPES | {"coverage_status": "string", "age_band": "string"},
    )
    coverage = population[["municipality", "year", "coverage_status"]].drop_duplicates()
    raw = pd.read_excel(ROOT / "data/raw/askdata.xlsx", sheet_name="census2024_13", header=None)
    validate_population(population[KEY + ["population"]], coverage, raw)
    return population


def validate_groups(population):
    if population[KEY + ["coverage_status"]].isna().any().any():
        raise ValueError("Grouping keys and coverage status must not be missing.")
    if population.duplicated(KEY).any():
        raise ValueError("Duplicate population records found.")
    expected_bands = population["age_group"].map(AGE_BANDS).astype("string")
    if expected_bands.isna().any() or population["age_band"].isna().any():
        raise ValueError("Missing or unknown age groups/bands.")
    if not population["age_band"].eq(expected_bands).all():
        raise ValueError("Age bands do not match the original age groups.")
    if not population.groupby(BASE + ["sex"]).size().eq(18).all():
        raise ValueError("Each sex/settlement group must have all 18 age groups.")
    if set(population["sex"]) != {"male", "female"}:
        raise ValueError("Expected male and female population records.")
    if not population.groupby(BASE)["sex"].nunique().eq(2).all():
        raise ValueError("Each settlement group must contain both sexes.")
    if set(population["settlement_type"]) != {"urban", "rural"}:
        raise ValueError("Expected urban and rural population records.")
    if not population.groupby(["municipality", "year"])["settlement_type"].nunique().eq(2).all():
        raise ValueError("Each municipality/year must contain urban and rural records.")
    if not population.groupby(["municipality", "year"])["coverage_status"].nunique().eq(1).all():
        raise ValueError("Inconsistent municipality/year coverage status.")
    if population["population"].lt(0).any():
        raise ValueError("Population counts must be nonnegative.")


def per_100(numerator, denominator):
    return numerator.astype("Float64").div(denominator.where(denominator.gt(0))).mul(100)


def aggregate_bands(population, dimensions):
    groups = dimensions + ["sex", "age_band"]
    bands = population.groupby(groups, observed=True, sort=True).agg(
        population=("population", lambda values: values.sum(min_count=1)),
        population_cells_present=("population", "count"),
        population_cells_expected=("population", "size"),
        coverage_status=("coverage_status", "first"),
    ).reset_index()
    expected = bands["age_band"].map(BAND_SIZES)
    if "settlement_type" not in dimensions:
        expected *= 2
    if not bands["population_cells_expected"].eq(expected).all():
        raise ValueError("Incomplete age-band category grid.")
    bands["population_cells_complete"] = bands["population_cells_present"].eq(expected)
    # A partial age band is not a complete population count.
    bands["population"] = bands["population"].where(bands["population_cells_complete"]).astype("Int64")
    return bands


def calculate_indicators(population, bands, dimensions):
    counts = bands.groupby(dimensions + ["age_band"], observed=True)["population"].sum(
        min_count=2
    ).unstack("age_band").rename(columns={
        "0-14": "children_population", "15-64": "working_age_population", "65+": "older_population"
    })
    sex_counts = bands.groupby(dimensions + ["sex"], observed=True)["population"].sum(
        min_count=3
    ).unstack("sex").rename(columns={"male": "male_population", "female": "female_population"})
    working_sexes = bands.loc[bands["age_band"].eq("15-64")].pivot(
        index=dimensions, columns="sex", values="population"
    )
    metadata = population.groupby(dimensions, observed=True).agg(
        coverage_status=("coverage_status", "first"),
        population_cells_present=("population", "count"),
        population_cells_expected=("population", "size"),
    )
    result = metadata.join(counts).join(sex_counts)
    result["population_cells_complete"] = result["population_cells_present"].eq(
        result["population_cells_expected"]
    )
    result["total_population"] = counts.sum(axis=1, min_count=3).astype("Int64")
    for name, column in [
        ("children_share_pct", "children_population"),
        ("working_age_share_pct", "working_age_population"),
        ("older_share_pct", "older_population"),
    ]:
        result[name] = per_100(result[column], result["total_population"])
    result["ageing_index"] = per_100(result["older_population"], result["children_population"])
    result["dependency_ratio"] = per_100(
        result["children_population"] + result["older_population"], result["working_age_population"]
    )
    result["sex_ratio"] = per_100(result["male_population"], result["female_population"])
    result["working_age_sex_ratio"] = per_100(working_sexes["male"], working_sexes["female"])
    return result.reset_index()


def build_demographic_tables(population):
    validate_groups(population)
    band_tables, indicator_tables = [], []
    for dimensions in [BASE, ["municipality", "year"]]:
        bands = aggregate_bands(population, dimensions)
        indicators = calculate_indicators(population, bands, dimensions)
        if "settlement_type" not in dimensions:
            bands["settlement_type"] = "all"
            indicators["settlement_type"] = "all"
        band_tables.append(bands)
        indicator_tables.append(indicators)
    bands = pd.concat(band_tables, ignore_index=True)
    indicators = pd.concat(indicator_tables, ignore_index=True)
    bands = bands[BASE + [column for column in bands if column not in BASE]]
    indicators = indicators[BASE + [column for column in indicators if column not in BASE]]
    return bands.sort_values(BASE + ["sex", "age_band"]), indicators.sort_values(BASE)


def main():
    bands, indicators = build_demographic_tables(load_population())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bands.to_csv(OUTPUT / "age_band_population.csv", index=False, encoding="utf-8")
    indicators.to_csv(OUTPUT / "demographic_indicators.csv", index=False, encoding="utf-8")
    report = (
        "ASKdata - Demographic Feature Report\n\n"
        f"Age-band records: {len(bands)}\n"
        f"Indicator records: {len(indicators)}\n"
        f"Indicator groups with missing input cells: {(~indicators.population_cells_complete).sum()}\n"
        f"Indicator groups with coverage warnings: {indicators.coverage_status.ne('no_specific_coverage_warning').sum()}\n\n"
        "FORMULAS\n"
        "Age shares (%) = age-band population / total population * 100.\n"
        "Ageing index = population 65+ / population 0-14 * 100.\n"
        "Dependency ratio = (population 0-14 + population 65+) / population 15-64 * 100.\n"
        "Sex ratio = male population / female population * 100.\n"
        "Working-age sex ratio uses only ages 15-64.\n\n"
        "RULES\n"
        "settlement_type='all' is the municipality total; do not add it to urban/rural.\n"
        "Every age-band count requires all of its source cells. Partial bands remain missing.\n"
        "Totals require every constituent band; missing counts are never replaced with zero.\n"
        "Ratios require both inputs and a positive denominator; otherwise they remain missing.\n"
        "Complete input cells do not establish complete census participation.\n"
        "Coverage warnings remain attached; limited-coverage indicators describe reported counts.\n"
        "No specific coverage warning does not guarantee year-to-year comparability.\n"
        "These indicators do not directly measure migration.\n"
    )
    report_path = ROOT / "results/demographic_features_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    preview = BASE + ["total_population", "ageing_index", "dependency_ratio", "sex_ratio"]
    print(f"Indicator records: {len(indicators)}")
    print("\n=== FIRST 10 INDICATOR ROWS ===")
    print(indicators[preview].head(10).to_string(index=False, float_format="{:.2f}".format))
    print(f"Coverage warnings and complete results saved to {OUTPUT}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
