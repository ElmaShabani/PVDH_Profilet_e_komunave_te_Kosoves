import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Create a small municipal sample for demonstration purposes.")
    parser.add_argument("--year", type=int, default=2024, choices=[2011, 2024])
    args = parser.parse_args()

    year = args.year
    source = ROOT / "data" / "processed" / "modeling" / f"selected_features_{year}.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Run 06_feature_scaling.py first: {source}")

    population = pd.read_csv(source)

    if "municipality" not in population.columns:
        raise ValueError("The dataset must contain a 'municipality' column.")

    if population["municipality"].isna().any():
        raise ValueError("Municipality names must not be missing.")

    if population["municipality"].duplicated().any():
        raise ValueError("Expected one row per municipality.")

    if year not in population["year"].unique():
        raise ValueError(f"Unexpected census year: {year}")

    if len(population) < 10:
        raise ValueError("At least 10 municipalities are required.")

    population = population.loc[population["municipality"].notna(), :].copy()
    population["municipality"] = population["municipality"].astype(str)
    population = population.sort_values("municipality").reset_index(drop=True)

    sample = population.sample(
        n=10,
        replace=False,
        random_state=42,
    ).sort_values("municipality")

    output = ROOT / "data" / "processed" / "sampling"
    output.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output / f"municipality_sample_{year}.csv", index=False, encoding="utf-8")

    print(f"Eligible municipalities: {len(population)}")
    print(f"Sampled municipalities: {len(sample)}")
    print(sample.to_string(index=False))
    print("Demonstration only: the main dataset remains unchanged.")


if __name__ == "__main__":
    main()
