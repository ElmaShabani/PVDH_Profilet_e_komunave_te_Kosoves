import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Run and verify the complete phase 1 workflow.")
    parser.add_argument("--year", type=int, choices=[2011, 2024], default=2024)
    year = parser.parse_args().year
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    summary_path = results / f"phase1_summary_{year}.txt"
    summary_path.write_text("INCOMPLETE: phase 1 verification is running.\n", encoding="utf-8")
    source = json.loads((ROOT / "data/raw/source.json").read_text(encoding="utf-8"))
    workbook = ROOT / "data/raw/askdata.xlsx"
    source_hash = hashlib.sha256(workbook.read_bytes()).hexdigest()
    if source_hash != source["sha256"]:
        summary_path.write_text("FAILED: the workbook does not match the recorded source checksum.\n", encoding="utf-8")
        raise SystemExit("Dataset changed. Verify its provenance before running phase 1.")

    steps = [
        ("Initial quality audit", ["phase1/01_data_quality.py"]),
        ("Preprocessing", ["phase1/02_preprocessing.py"]),
        ("Source validation", ["phase1/validate_data.py"]),
        ("Aggregation", ["phase1/03_aggregation.py"]),
        ("Age categories and Boolean features", ["phase1/04_feature_preparation.py"]),
        ("Demographic indicators", ["phase1/05_demographic_features.py"]),
        ("Feature selection and scaling", ["phase1/06_feature_scaling.py", "--year", str(year)]),
        ("PCA", ["phase1/07_pca.py", "--year", str(year)]),
        ("Automated tests", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
    ]
    log_path = results / f"phase1_run_{year}.txt"
    tests_run = 0
    with log_path.open("w", encoding="utf-8") as log:
        for label, arguments in steps:
            completed = subprocess.run(
                [sys.executable, *arguments], cwd=ROOT, capture_output=True, text=True
            )
            output = completed.stdout + completed.stderr
            log.write(f"=== {label} ===\n{output}\n")
            log.flush()
            if label == "Automated tests":
                match = re.search(r"Ran (\d+) tests?", output)
                tests_run = int(match.group(1)) if match else 0
            failed = completed.returncode != 0 or (label == "Automated tests" and tests_run == 0)
            if failed:
                summary_path.write_text(f"FAILED: {label}. See {log_path.name}.\n", encoding="utf-8")
                print(output, file=sys.stderr)
                raise SystemExit(f"FAILED: {label}. Full log: {log_path}")
            print(f"PASS: {label}", flush=True)

    if hashlib.sha256(workbook.read_bytes()).hexdigest() != source_hash:
        summary_path.write_text("FAILED: the raw workbook changed during execution.\n", encoding="utf-8")
        raise SystemExit("The original workbook changed during execution.")

    processed = ROOT / "data/processed"
    validated = pd.read_csv(processed / "population_validated.csv")
    indicators = pd.read_csv(processed / "analysis/demographic_indicators.csv")
    modeling = processed / "modeling"
    selected = pd.read_csv(modeling / f"selected_features_{year}.csv")
    excluded = pd.read_csv(modeling / f"excluded_municipalities_{year}.csv")
    variance = pd.read_csv(modeling / "pca" / f"pca_variance_{year}.csv")
    scores = pd.read_csv(modeling / "pca" / f"pca_scores_{year}.csv")
    retained = int(variance.retained.sum())
    explained = float(variance.loc[variance.retained, "explained_variance_ratio"].sum())
    summary = (
        "ASKdata - Phase 1 Final Verification\n\n"
        "Status: PASS\n"
        f"Model preparation year: {year}\n"
        f"Workflow stages completed: {len(steps)}\n"
        f"Automated tests passed: {tests_run}\n"
        f"Validated population records: {len(validated)}\n"
        f"Missing population counts retained: {validated.population.isna().sum()}\n"
        f"Demographic indicator groups: {len(indicators)}\n"
        f"Selected municipalities: {len(selected)}\n"
        f"Excluded municipalities: {len(excluded)}\n"
        f"PCA retained components: {retained}\n"
        f"PCA retained variance: {explained:.2%}\n\n"
        "SOURCE\n"
        f"Publisher: {source['publisher']}\n"
        f"Table: {source['table_id']}\n"
        f"URL: {source['url']}\n"
        f"Source last updated: {source['source_last_updated']}\n"
        f"Source page verified: {source['source_verified_on']}\n"
        "Original download date: not recorded\n"
        f"Raw workbook SHA-256: {source_hash}\n"
        "Raw workbook unchanged: yes\n\n"
        "PREPROCESSING DECISIONS\n"
        "Collection: the supplied ASK Excel export is preserved unchanged.\n"
        "Types: population uses nullable integers; years are integers; labels are strings.\n"
        "Quality: source values, unique keys, category coverage and missingness are validated.\n"
        "Integration: census years use one long-format schema with coverage metadata.\n"
        "Aggregation: municipality, settlement, sex and broad age-band totals are available.\n"
        "Sampling: all validated records are retained; ten displayed rows are only previews.\n"
        "Cleaning: report headers, blank rows and source notes are excluded from data records.\n"
        "Missingness: no imputation; absent 2011 northern enumeration is represented as missing.\n"
        "Feature creation: age shares, ageing/dependency indices and sex ratios are calculated.\n"
        "Discretization: five-year age categories map to 0-14, 15-64 and 65+.\n"
        "Binarization: three mutually exclusive Boolean age-band indicators are available.\n"
        "Selection: four demographic features are retained without identifiers or redundant age shares.\n"
        "Transformation: selected features are standardized with saved means and scales.\n"
        "Dimension reduction: SVD-based PCA retains at least 90% of standardized variance.\n\n"
        "LIMITS\n"
        "PASS verifies technical execution and tests, not complete census comparability.\n"
        "Coverage warnings remain in analytical outputs; exclusions and reasons remain in modeling outputs.\n"
        "Missing counts are not zero; zero-denominator ratios remain undefined.\n"
        "Municipality totals marked 'all' must not be added again to urban/rural rows.\n"
        "Separately fitted yearly scalers/PCA axes are not a common basis for census comparisons.\n"
        "Cohort changes do not directly measure emigration; urban/rural data do not identify villages.\n"
        "Clustering, cohort analysis and the interactive dashboard are future project work.\n"
    )
    summary_path.write_text(summary, encoding="utf-8")
    print(f"\nPHASE 1 PASSED: {tests_run} tests; PCA retains {explained:.2%} variance.")
    print("\n=== FIRST 10 FINAL ROWS ===")
    print(scores.head(10).to_string(index=False, float_format="{:.3f}".format))
    print(f"Final summary: {summary_path}")
    print(f"Full execution log: {log_path}")


if __name__ == "__main__":
    main()
