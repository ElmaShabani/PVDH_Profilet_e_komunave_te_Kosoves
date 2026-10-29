from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/population_validated.csv"
OUTPUT = ROOT / "data/processed/analysis/population_analysis_ready.csv"
REPORT = ROOT / "results/feature_preparation_report.txt"

KEY = ["municipality", "year", "settlement_type", "sex", "age_group"]
INPUT_COLUMNS = KEY + ["population", "coverage_status"]

AGE_GROUPS = [f"{age}-{age + 4}" for age in range(0, 85, 5)] + ["85+"]
EXPECTED_YEARS = {2011, 2024}
EXPECTED_SETTLEMENTS = {"urban", "rural"}
EXPECTED_SEXES = {"male", "female"}
ALLOWED_COVERAGE = {
    "no_specific_coverage_warning",
    "not_enumerated",
    "incomplete_participation",
}
EXPECTED_MUNICIPALITIES = 38

AGE_BAND_MAP = {
    **dict.fromkeys(AGE_GROUPS[:3], "0-14"),
    **dict.fromkeys(AGE_GROUPS[3:13], "15-64"),
    **dict.fromkeys(AGE_GROUPS[13:], "65+"),
}

OUTPUT_COLUMNS = [
    "municipality",
    "year",
    "settlement_type",
    "sex",
    "age_group",
    "age_band",
    "population",
    "coverage_status",
    "is_child_0_14",
    "is_working_age_15_64",
    "is_older_65_plus",
]


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
        actual = set(population[column].dropna().tolist())
        if actual != expected:
            raise ValueError(f"Unexpected or missing categories in {column}.")

    coverage_values = set(population["coverage_status"].dropna().tolist())
    if not coverage_values.issubset(ALLOWED_COVERAGE):
        raise ValueError("Unexpected coverage status found.")

    municipality_count = population["municipality"].nunique()
    if municipality_count != EXPECTED_MUNICIPALITIES:
        raise ValueError(
            f"Expected {EXPECTED_MUNICIPALITIES} municipalities, "
            f"found {municipality_count}."
        )

    expected_rows = (
        EXPECTED_MUNICIPALITIES
        * len(EXPECTED_YEARS)
        * len(EXPECTED_SETTLEMENTS)
        * len(EXPECTED_SEXES)
        * len(AGE_GROUPS)
    )
    if len(population) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} validated records, found {len(population)}."
        )

    rows_per_municipality_year = population.groupby(
        ["municipality", "year"], observed=True
    ).size()
    expected_rows_per_pair = (
        len(EXPECTED_SETTLEMENTS)
        * len(EXPECTED_SEXES)
        * len(AGE_GROUPS)
    )
    if not rows_per_municipality_year.eq(expected_rows_per_pair).all():
        raise ValueError("The municipality/year category grid is incomplete.")

    status_count = population.groupby(
        ["municipality", "year"], observed=True
    )["coverage_status"].nunique()
    if not status_count.eq(1).all():
        raise ValueError("Coverage status changes within a municipality/year.")

    return population


def prepare_features(population):
    prepared = population.copy()

    prepared["age_band"] = prepared["age_group"].map(AGE_BAND_MAP).astype("string")
    if prepared["age_band"].isna().any():
        raise ValueError("Some age groups could not be mapped to an age band.")

    prepared["is_child_0_14"] = prepared["age_band"].eq("0-14")
    prepared["is_working_age_15_64"] = prepared["age_band"].eq("15-64")
    prepared["is_older_65_plus"] = prepared["age_band"].eq("65+")

    if not (
        prepared[
            [
                "is_child_0_14",
                "is_working_age_15_64",
                "is_older_65_plus",
            ]
        ].sum(axis=1).eq(1).all()
    ):
        raise ValueError("Each record must belong to exactly one age band.")

    return prepared[OUTPUT_COLUMNS]


def write_report(population, prepared):
    missing_population = int(prepared["population"].isna().sum())

    report_lines = [
        "ASKdata - Phase 1 Feature Preparation Report",
        "",
        f"Input: {INPUT.relative_to(ROOT)}",
        f"Output: {OUTPUT.relative_to(ROOT)}",
        f"Records retained: {len(prepared)}",
        f"Municipalities: {prepared['municipality'].nunique()}",
        f"Years: {', '.join(map(str, sorted(prepared['year'].unique())))}",
        f"Missing population values retained: {missing_population}",
        "",
        "PHASE 1 DECISIONS",
        "Sampling: no sample was drawn; all validated source records were retained.",
        "Coverage: coverage_status was preserved for every record.",
        "Missing values: population missing values were not imputed or changed to zero.",
        "Feature selection: the output uses an explicit analytical column list.",
        "All original analytical fields were retained because they are needed",
        "for the project's municipality, year, age, sex and settlement analyses.",
        "Dimension reduction: no statistical reduction was applied; the source",
        "dimensions remain meaningful and are needed for later analyses.",
        "Discretization: the source five-year age groups were mapped to 0-14,",
        "15-64 and 65+; the original age_group column remains unchanged.",
        "Binarization: three mutually exclusive Boolean age-band flags were added.",
        "Transformation: the validated long-format dataset was copied to a new output.",
        "Aggregation: summary tables remain separate outputs from phase1/03_aggregation.py.",
        "",
        "OUTPUT COLUMNS",
        ", ".join(OUTPUT_COLUMNS),
        "",
        "AGE BAND COUNTS",
        prepared["age_band"].value_counts().sort_index().to_string(),
        "",
        "COVERAGE STATUS COUNTS",
        prepared["coverage_status"].value_counts().sort_index().to_string(),
        "",
        "The output is prepared for later analysis; it does not calculate or",
        "interpret migration, ageing, dependency or other analytical results.",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main():
    population = load_population()
    prepared = prepare_features(population)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(OUTPUT, index=False, encoding="utf-8")
    write_report(population, prepared)

    print(f"Prepared records: {len(prepared)}")
    print(f"Analysis-ready data saved to {OUTPUT}")
    print(f"Preparation report saved to {REPORT}")


if __name__ == "__main__":
    main()