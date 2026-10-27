from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["municipality", "year", "settlement_type", "sex", "age_group"]
DTYPES = {column: "string" for column in KEY}
DTYPES.update(year="int64", population="Int64")
NORTH = {"Leposaviq", "Zubin Potok", "Zveqan", "Mitrovicë e Veriut"}
AGE_GROUPS = [f"{age}-{age + 4}" for age in range(0, 85, 5)] + ["85+"]


def validate_population(population, coverage, raw):
    if list(population.columns) != KEY + ["population"]:
        raise ValueError("Unexpected population columns.")
    if list(coverage.columns) != ["municipality", "year", "coverage_status"]:
        raise ValueError("Unexpected coverage columns.")
    if population[KEY].isna().any().any() or population.duplicated(KEY).any():
        raise ValueError("Population keys must be complete and unique.")
    for column, dtype in DTYPES.items():
        if str(population[column].dtype) != dtype:
            raise ValueError(f"Expected {dtype} for {column}.")
    if population["population"].lt(0).any():
        raise ValueError("Negative population counts found.")

    # Reconstruct the original layout to compare every population cell.
    source = raw.loc[raw[1].isin(AGE_GROUPS)].copy()
    source[0] = source[0].ffill()
    if len(source) != 684 or source[0].nunique() != 38:
        raise ValueError("Expected 38 municipalities with 18 age groups in this source.")
    source = source.set_index([0, 1])
    source.index.names = ["municipality", "age_group"]
    if source.index.has_duplicates:
        raise ValueError("Duplicate source municipality and age-group rows.")
    source = source.loc[:, 2:9].replace(":", pd.NA)
    source = source.apply(pd.to_numeric, errors="raise").astype("Int64")
    source.columns = pd.MultiIndex.from_product(
        [[2011, 2024], ["urban", "rural"], ["male", "female"]],
        names=["year", "settlement_type", "sex"],
    )
    northern = source.index.get_level_values("municipality").isin(NORTH)
    source.loc[northern, (2011, slice(None), slice(None))] = pd.NA

    restored = population.pivot(
        index=["municipality", "age_group"],
        columns=["year", "settlement_type", "sex"],
        values="population",
    )
    if set(restored.index) != set(source.index) or set(restored.columns) != set(source.columns):
        raise ValueError("Population categories do not match the source.")
    if len(population) != source.size:
        raise ValueError("Population records are missing from the full category grid.")
    restored = restored.reindex(index=source.index, columns=source.columns)
    try:
        pd.testing.assert_frame_equal(restored, source, check_index_type=False)
    except AssertionError as error:
        raise ValueError("Population values or missingness differ from the source rules.") from error

    pairs = ["municipality", "year"]
    if coverage[pairs + ["coverage_status"]].isna().any().any():
        raise ValueError("Coverage keys and status must not be missing.")
    if coverage.duplicated(pairs).any():
        raise ValueError("Duplicate municipality/year coverage records.")
    expected_pairs = set(population[pairs].itertuples(index=False, name=None))
    if set(coverage[pairs].itertuples(index=False, name=None)) != expected_pairs:
        raise ValueError("Coverage must match all population municipality/year pairs.")

    joined = population.merge(coverage, on=pairs, how="left", validate="many_to_one")
    expected_status = pd.Series("no_specific_coverage_warning", index=joined.index)
    northern = joined["municipality"].isin(NORTH)
    expected_status.loc[northern & joined["year"].eq(2011)] = "not_enumerated"
    expected_status.loc[northern & joined["year"].eq(2024)] = "incomplete_participation"
    if not joined["coverage_status"].eq(expected_status).all():
        raise ValueError("Coverage warnings do not match the source rules.")
    return joined


def main():
    processed = ROOT / "data/processed"
    required = [processed / "population.csv", processed / "population_coverage.csv"]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("Run phase1/02_preprocessing.py before validation.")

    population = pd.read_csv(required[0], dtype=DTYPES)
    coverage = pd.read_csv(required[1], dtype={
        "municipality": "string", "year": "int64", "coverage_status": "string"
    })
    raw = pd.read_excel(ROOT / "data/raw/askdata.xlsx", sheet_name="census2024_13", header=None)
    validated = validate_population(population, coverage, raw)

    validated.to_csv(processed / "population_validated.csv", index=False, encoding="utf-8")
    report = (
        "ASKdata - Validation Report\n\n"
        "PASS: analytical column types and unique, complete record keys\n"
        "PASS: all source municipalities, age groups, years, sexes and settlement types\n"
        "PASS: every population cell matches Excel after the documented coverage adjustment\n"
        "PASS: missingness matches source rules and genuine zeros are preserved\n"
        "PASS: coverage records match exactly, with no duplicate or missing joins\n"
        "PASS: northern municipality warnings match the source rules\n\n"
        f"Validated records: {len(validated)}\n"
        f"Missing population counts: {validated['population'].isna().sum()}\n\n"
        "The source's coverage limitations still apply; passing validation does not\n"
        "establish full census comparability. Missing counts must remain missing\n"
        "during aggregation; zero denominators must produce undefined ratios.\n"
    )
    report_path = ROOT / "results/validation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print("Validation passed.")
    print("\n=== FIRST 10 VALIDATED ROWS ===")
    print(validated.head(10).to_string(index=False))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
