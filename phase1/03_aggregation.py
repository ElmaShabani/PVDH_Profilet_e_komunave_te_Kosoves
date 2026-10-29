from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/population_validated.csv"
OUTPUT = ROOT / "data/processed/aggregates"

KEY = ["municipality", "year", "settlement_type", "sex", "age_group"]
INPUT_COLUMNS = KEY + ["population", "coverage_status"]

AGE_GROUPS = [f"{age}-{age + 4}" for age in range(0, 85, 5)] + ["85+"]
EXPECTED_SETTLEMENTS = {"urban", "rural"}
EXPECTED_SEXES = {"male", "female"}
EXPECTED_YEARS = {2011, 2024}
ALLOWED_COVERAGE = {
    "no_specific_coverage_warning",
    "not_enumerated",
    "incomplete_participation",
}

AGGREGATION_LEVELS = {
    "municipality_year_settlement_sex": [
        "municipality", "year", "settlement_type", "sex"
    ],
    "municipality_year_settlement": [
        "municipality", "year", "settlement_type"
    ],
    "municipality_year_sex": [
        "municipality", "year", "sex"
    ],
    "municipality_year": ["municipality", "year"],
}


def load_population():
    if not INPUT.is_file():
        raise FileNotFoundError(
            "Validated data not found. Run phase1/02_preprocessing.py "
            "and phase1/validate_data.py first."
        )

    population = pd.read_csv(
        INPUT,
        dtype={
            "municipality": "string",
            "year": "int64",
            "settlement_type": "string",
            "sex": "string",
            "age_group": "string",
            "population": "Int64",
            "coverage_status": "string",
        },
    )

    if list(population.columns) != INPUT_COLUMNS:
        raise ValueError("Unexpected columns in population_validated.csv.")
    if population[KEY].isna().any().any():
        raise ValueError("Population grouping keys must not be missing.")
    if population["coverage_status"].isna().any():
        raise ValueError("Coverage status must not be missing.")
    if population.duplicated(KEY).any():
        raise ValueError("Duplicate population records found.")
    if population["population"].lt(0).fillna(False).any():
        raise ValueError("Negative population values found.")

    expected_categories = {
        "year": EXPECTED_YEARS,
        "settlement_type": EXPECTED_SETTLEMENTS,
        "sex": EXPECTED_SEXES,
        "age_group": set(AGE_GROUPS),
    }
    for column, expected in expected_categories.items():
        if set(population[column].dropna().tolist()) != expected:
            raise ValueError(f"Unexpected or missing categories in {column}.")

    coverage_values = set(population["coverage_status"].dropna().tolist())
    if not coverage_values.issubset(ALLOWED_COVERAGE):
        raise ValueError("Unexpected coverage status found.")

    if population["municipality"].nunique() != 38:
        raise ValueError("Expected 38 municipalities in the validated dataset.")

    rows_per_municipality_year = population.groupby(
        ["municipality", "year"], observed=True
    ).size()
    expected_rows_per_pair = (
        len(EXPECTED_SETTLEMENTS) * len(EXPECTED_SEXES) * len(AGE_GROUPS)
    )
    if (
        len(rows_per_municipality_year) != 38 * len(EXPECTED_YEARS)
        or not rows_per_municipality_year.eq(expected_rows_per_pair).all()
    ):
        raise ValueError("The municipality/year category grid is incomplete.")

    status_count = population.groupby(
        ["municipality", "year"], observed=True
    )["coverage_status"].nunique()
    if not status_count.eq(1).all():
        raise ValueError("Coverage status changes within a municipality/year.")

    return population


def aggregate_level(population, group_columns):
    expected_cells = len(AGE_GROUPS)

    if "settlement_type" not in group_columns:
        expected_cells *= len(EXPECTED_SETTLEMENTS)
    if "sex" not in group_columns:
        expected_cells *= len(EXPECTED_SEXES)

    grouped = population.groupby(
        group_columns,
        observed=True,
        sort=True,
        dropna=False,
    )

    result = grouped.agg(
        population_cells_expected=("population", "size"),
        population_cells_present=("population", "count"),
        reported_population_sum=(
            "population",
            lambda values: values.sum(min_count=1),
        ),
        coverage_status=("coverage_status", "first"),
    ).reset_index()

    if not result["population_cells_expected"].eq(expected_cells).all():
        raise ValueError("An aggregation group has an unexpected number of cells.")

    result["population_cells_complete"] = result[
        "population_cells_present"
    ].eq(result["population_cells_expected"])

    result["reported_population_sum"] = result[
        "reported_population_sum"
    ].astype("Int64")

    return result[
        group_columns
        + [
            "coverage_status",
            "reported_population_sum",
            "population_cells_present",
            "population_cells_expected",
            "population_cells_complete",
        ]
    ]


def main():
    population = load_population()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "ASKdata - Population Aggregation Report",
        "",
        "population_cells_complete describes input-cell completeness only.",
        "It does not confirm that census participation or coverage was complete.",
        "reported_population_sum is the sum of reported values, not necessarily",
        "the total population when cells are missing or census coverage is limited.",
        "no_specific_coverage_warning does not guarantee full comparability.",
        "",
    ]

    for level, group_columns in AGGREGATION_LEVELS.items():
        result = aggregate_level(population, group_columns)
        output_path = OUTPUT / f"{level}.csv"
        result.to_csv(output_path, index=False, encoding="utf-8")

        incomplete_cells = result.loc[~result["population_cells_complete"]]
        limited_coverage = result.loc[
            result["coverage_status"].isin(
                {"not_enumerated", "incomplete_participation"}
            )
        ]

        report_lines.extend(
            [
                f"=== {level} ===",
                f"Groups: {len(result)}",
                f"Groups with missing input cells: {len(incomplete_cells)}",
                f"Groups with a coverage limitation: {len(limited_coverage)}",
                f"Output: {output_path.relative_to(ROOT)}",
            ]
        )

        if not incomplete_cells.empty:
            report_lines.append("Groups with missing input cells:")
            report_lines.append(
                incomplete_cells[
                    group_columns
                    + [
                        "coverage_status",
                        "population_cells_present",
                        "population_cells_expected",
                    ]
                ].to_string(index=False)
            )

        if not limited_coverage.empty:
            report_lines.append("Groups with a coverage limitation:")
            report_lines.append(
                limited_coverage[
                    group_columns
                    + [
                        "coverage_status",
                        "population_cells_complete",
                        "reported_population_sum",
                    ]
                ].to_string(index=False)
            )

        report_lines.append("")

    report_path = ROOT / "results/population_aggregation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Aggregation outputs saved to {OUTPUT}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()