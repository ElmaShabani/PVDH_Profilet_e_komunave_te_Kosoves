import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/analysis/demographic_indicators.csv"
OUTPUT = ROOT / "data/processed/modeling"
IDS = ["municipality", "year"]
FEATURES = [
    "children_share_pct", "older_share_pct", "working_age_sex_ratio", "urban_share_pct"
]
COVERAGE = {"no_specific_coverage_warning", "not_enumerated", "incomplete_participation"}


def select_features(indicators, year):
    required = IDS + [
        "settlement_type", "total_population", "coverage_status", "population_cells_complete"
    ] + FEATURES[:-1]
    if not set(required).issubset(indicators.columns):
        raise ValueError("Required demographic indicator columns are missing.")
    if indicators[IDS + ["settlement_type", "coverage_status", "population_cells_complete"]].isna().any().any():
        raise ValueError("Identifiers, coverage and completeness must not be missing.")
    if indicators.duplicated(IDS + ["settlement_type"]).any():
        raise ValueError("Duplicate municipality/year/settlement indicator records.")
    if not set(indicators.coverage_status).issubset(COVERAGE):
        raise ValueError("Unknown coverage status.")
    source = indicators.loc[indicators.year.eq(year)].copy()
    if source.empty:
        raise ValueError(f"No indicator records for {year}.")
    settlements = source.groupby("municipality").settlement_type.agg(set)
    if not settlements.map(lambda values: values == {"all", "urban", "rural"}).all():
        raise ValueError("Each municipality must have all, urban and rural records.")
    if not source.groupby("municipality").coverage_status.nunique().eq(1).all():
        raise ValueError("Coverage differs within a municipality/year.")

    counts = source.pivot(index="municipality", columns="settlement_type", values="total_population")
    if counts.lt(0).any().any():
        raise ValueError("Negative population totals found.")
    known = counts.notna().all(axis=1)
    if (known & counts["all"].ne(counts["urban"] + counts["rural"])).any():
        raise ValueError("Urban and rural counts do not match the municipality total.")

    candidates = source.loc[source.settlement_type.eq("all")].set_index("municipality")
    candidates["urban_share_pct"] = counts["urban"].astype("Float64").div(
        counts["all"].where(counts["all"].gt(0))
    ).mul(100)
    complete = source.groupby("municipality").population_cells_complete.all()
    numeric = candidates[FEATURES].replace([float("inf"), -float("inf")], pd.NA)
    valid_features = numeric.notna().all(axis=1) & numeric.ge(0).all(axis=1)
    valid_features &= numeric[["children_share_pct", "older_share_pct", "urban_share_pct"]].le(100).all(axis=1)
    valid_features &= numeric.children_share_pct.add(numeric.older_share_pct).le(100 + 1e-8).fillna(False)

    candidates["exclusion_reason"] = ""
    conditions = {
        "coverage_warning": candidates.coverage_status.ne("no_specific_coverage_warning"),
        "missing_source_cells": ~complete,
        "undefined_population_total": ~known | counts["all"].le(0).fillna(True),
        "missing_or_invalid_feature": ~valid_features,
    }
    for reason, mask in conditions.items():
        candidates.loc[mask.reindex(candidates.index), "exclusion_reason"] += reason + "; "
    candidates["exclusion_reason"] = candidates.exclusion_reason.str.rstrip("; ")
    candidates = candidates.reset_index().sort_values(IDS).reset_index(drop=True)
    eligible = candidates.exclusion_reason.eq("")
    selected = candidates.loc[eligible, IDS + FEATURES].reset_index(drop=True)
    excluded = candidates.loc[~eligible, IDS + ["coverage_status", "exclusion_reason"]].reset_index(drop=True)
    return selected, excluded


def standardize_features(selected):
    if len(selected) < 2:
        raise ValueError("At least two eligible municipalities are needed for standardization.")
    values = selected[FEATURES].astype("float64")
    if values.replace([float("inf"), -float("inf")], pd.NA).isna().any().any():
        raise ValueError("Standardization requires finite, nonmissing features.")
    if selected.year.nunique() != 1:
        raise ValueError("Fit the scaler to one census year at a time.")
    mean = values.mean()
    constant = values.nunique().eq(1)
    std = values.std(ddof=0).mask(constant, 0.0)
    scale = std.mask(constant, 1.0)
    standardized = values.sub(mean).div(scale)
    standardized.loc[:, constant] = 0.0
    standardized = pd.concat([selected[IDS].reset_index(drop=True), standardized.reset_index(drop=True)], axis=1)
    parameters = pd.DataFrame({
        "mean": mean, "std": std, "scale": scale, "is_constant": constant
    }).rename_axis("feature").reset_index()
    parameters["year"] = int(selected.year.iloc[0])
    parameters["sample_size"] = len(selected)
    return standardized, parameters


def main():
    parser = argparse.ArgumentParser(description="Select and standardize municipality demographic features.")
    parser.add_argument("--year", type=int, choices=[2011, 2024], default=2024)
    year = parser.parse_args().year
    if not INPUT.is_file():
        raise FileNotFoundError("Run phase1/05_demographic_features.py first.")
    indicators = pd.read_csv(INPUT, dtype={
        "municipality": "string", "year": "int64", "settlement_type": "string",
        "total_population": "Int64", "coverage_status": "string", "population_cells_complete": "boolean",
        **{column: "Float64" for column in FEATURES[:-1]},
    })
    selected, excluded = select_features(indicators, year)
    standardized, parameters = standardize_features(selected)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "selected_features": selected, "standardized_features": standardized,
        "scaler_parameters": parameters, "excluded_municipalities": excluded,
    }.items():
        table.to_csv(OUTPUT / f"{name}_{year}.csv", index=False, encoding="utf-8")
    report = (
        "ASKdata - Feature Selection and Standardization Report\n\n"
        f"Census year: {year}\n"
        f"Selected municipalities: {len(selected)}\n"
        f"Excluded municipalities: {len(excluded)}\n"
        f"Features: {', '.join(FEATURES)}\n\n"
        "SELECTION\n"
        "One municipality-level row per year; urban/rural rows are not separate model observations.\n"
        "Urban share is urban population / municipality population * 100.\n"
        "The two age shares describe age composition; working-age share is omitted\n"
        "because all three age shares sum to 100. Ageing and dependency indicators\n"
        "are omitted because they are derived from the same age shares.\n"
        "Working-age sex ratio describes sex balance; urban share describes settlement structure.\n"
        "Absolute counts are omitted so municipality size does not dominate demographic profiles.\n"
        "Identifiers, coverage labels and Boolean age flags are not numerical model features.\n"
        "Only complete, finite records without a specific coverage warning are selected.\n"
        "Excluded municipalities and reasons are retained separately; no values are imputed.\n\n"
        "STANDARDIZATION\n"
        "z = (value - mean) / scale; parameters are fitted only on selected municipalities\n"
        "from the requested census year. Standard deviation uses ddof=0.\n"
        "Constant features use scale=1 and become zero; they should be omitted from later models.\n"
        "Original units are retained in selected_features; fitted parameters are saved separately.\n"
        "Separate yearly fits do not put changes between censuses on a common reference scale.\n"
        "For future held-out evaluation, fit parameters on training observations only.\n"
        "No specific coverage warning does not guarantee full census comparability.\n"
        "Feature selection is a documented starting point, not evidence of predictive importance.\n"
        "PCA and clustering have not been applied in this step.\n"
    )
    report_path = ROOT / f"results/feature_scaling_report_{year}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Year: {year} | Selected: {len(selected)} | Excluded: {len(excluded)}")
    print("\n=== FIRST 10 STANDARDIZED ROWS ===")
    print(standardized.head(10).to_string(index=False, float_format="{:.3f}".format))
    print(f"Features, exclusions and scaler parameters saved to {OUTPUT}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
