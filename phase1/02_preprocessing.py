from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AGE_GROUPS = [f"{age}-{age + 4}" for age in range(0, 85, 5)] + ["85+"]
KEY = ["municipality", "year", "settlement_type", "sex", "age_group"]
NORTH = {"Leposaviq", "Zubin Potok", "Zveqan", "Mitrovicë e Veriut"}
HEADERS = [
    (year, settlement, sex)
    for year in (2011, 2024)
    for settlement in ("Urban", "Rural")
    for sex in ("Meshkuj", "Femra")
]


def prepare_population(raw):
    if raw.shape[1] != 10:
        raise ValueError("Expected two label columns and eight population columns.")

    raw = raw.map(lambda value: value.strip() if isinstance(value, str) else value)
    raw = raw.replace("", pd.NA)
    headers = raw.iloc[2:5, 2:].ffill(axis=1)
    actual_headers = [(int(year), settlement, sex) for year, settlement, sex in zip(*headers.to_numpy())]
    if actual_headers != HEADERS:
        raise ValueError("Unexpected year, settlement or sex headers.")

    age_rows = raw.index[raw[1].isin(AGE_GROUPS)]
    if age_rows.empty:
        raise ValueError("No recognized age groups found.")

    # Fill municipality labels only inside the population table.
    body = raw.loc[age_rows.min():age_rows.max()].dropna(how="all").copy()
    if not body[1].isin(AGE_GROUPS).all():
        raise ValueError("Unexpected age group or non-data row inside the table.")
    body[0] = body[0].ffill()
    if body[0].isna().any():
        raise ValueError("Some population rows have no municipality.")
    if body.duplicated([0, 1]).any():
        raise ValueError("Duplicate municipality and age-group rows found.")
    if not body.groupby(0)[1].agg(set).map(lambda ages: ages == set(AGE_GROUPS)).all():
        raise ValueError("Each municipality must have all 18 age groups.")

    parts = []
    for column, (year, settlement, sex) in enumerate(HEADERS, start=2):
        part = body[[0, 1, column]].copy()
        part.columns = ["municipality", "age_group", "population"]
        part["year"] = year
        part["settlement_type"] = settlement.lower()
        part["sex"] = {"Meshkuj": "male", "Femra": "female"}[sex]
        parts.append(part)
    population = pd.concat(parts, ignore_index=True)[KEY + ["population"]]

    values = population["population"].replace(":", pd.NA)
    numeric = pd.to_numeric(values, errors="coerce")
    if (values.notna() & numeric.isna()).any():
        raise ValueError("Unrecognized text in population cells.")
    if (numeric.notna() & ((numeric < 0) | (numeric % 1 != 0))).any():
        raise ValueError("Population must contain nonnegative whole numbers.")
    population["population"] = numeric.astype("Int64")
    population["year"] = population["year"].astype("int64")
    for column in ("municipality", "settlement_type", "sex", "age_group"):
        population[column] = population[column].astype("string")
    if population.duplicated(KEY).any():
        raise ValueError("Duplicate population record keys found.")

    missing_before = int(population["population"].isna().sum())
    # Source notes: no 2011 enumeration in these four municipalities.
    not_enumerated = population["municipality"].isin(NORTH) & population["year"].eq(2011)
    replaced = int(population.loc[not_enumerated, "population"].notna().sum())
    population.loc[not_enumerated, "population"] = pd.NA
    population = population.sort_values(
        KEY, key=lambda values: values.map({age: i for i, age in enumerate(AGE_GROUPS)})
        if values.name == "age_group" else values
    ).reset_index(drop=True)

    coverage = population[["municipality", "year"]].drop_duplicates().copy()
    coverage["coverage_status"] = "no_specific_coverage_warning"
    northern = coverage["municipality"].isin(NORTH)
    coverage.loc[northern & coverage["year"].eq(2011), "coverage_status"] = "not_enumerated"
    coverage.loc[northern & coverage["year"].eq(2024), "coverage_status"] = "incomplete_participation"

    metrics = {
        "Raw worksheet rows": len(raw),
        "Population rows before reshaping": len(body),
        "Excluded header, blank and note rows": len(raw) - len(body),
        "Municipalities": population["municipality"].nunique(),
        "Age groups": population["age_group"].nunique(),
        "Processed records": len(population),
        "Missing counts before coverage adjustment": missing_before,
        "Counts set to missing for absent 2011 enumeration": replaced,
        "Missing counts after coverage adjustment": int(population["population"].isna().sum()),
        "Duplicate record keys": int(population.duplicated(KEY).sum()),
    }
    return population, coverage.reset_index(drop=True), metrics


def main():
    source = ROOT / "data/raw/askdata.xlsx"
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    raw = pd.read_excel(source, sheet_name="census2024_13", header=None, keep_default_na=False)
    population, coverage, metrics = prepare_population(raw)

    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    population.to_csv(output / "population.csv", index=False, encoding="utf-8")
    coverage.to_csv(output / "population_coverage.csv", index=False, encoding="utf-8")

    report = "ASKdata - Preprocessing Report\n\n"
    report += "\n".join(f"{label}: {value}" for label, value in metrics.items())
    report += "\n\nCOLUMN TYPES\n" + population.dtypes.to_string()
    report += "\n\nMISSING COUNTS BY MUNICIPALITY AND YEAR\n"
    report += population.groupby(["municipality", "year"])["population"].agg(
        lambda values: int(values.isna().sum())
    ).to_string()
    report += (
        "\n\nRULES\n"
        "Empty data cells and ':' remain missing; no counts are imputed.\n"
        "2011 counts in the four northern municipalities are set to missing because\n"
        "the source states they were not enumerated. Other source zeros are preserved.\n"
        "2024 counts in these municipalities are retained with an incomplete-participation warning.\n"
        "Join population_coverage.csv by municipality and year before comparing years.\n"
        "No specific coverage warning does not guarantee full comparability.\n"
        "Missing population values are empty CSV fields; load them as nullable Int64.\n"
        "No population records were sampled or automatically deduplicated.\n"
        "The original workbook is unchanged.\n"
    )
    report_path = ROOT / "results/preprocessing_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Processed records: {len(population)}")
    print("\n=== FIRST 10 PROCESSED ROWS ===")
    print(population.head(10).to_string(index=False))
    print(f"Processed data saved to {output}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
