from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    file_path = ROOT / "data/raw/askdata.xlsx"
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Preserve the source layout for this initial audit.
    df = pd.read_excel(file_path, sheet_name=0, header=None)
    notes = (
        "This report describes the raw worksheet, not a cleaned dataset.\n"
        "Headers, merged cells, blank rows and footnotes affect these counts.\n"
        "Raw column types are not the final analytical data types.\n"
        "The source uses ':' for unavailable values; these are counted separately.\n"
        "Repeated raw rows are not necessarily duplicate population records.\n"
        "The source warns of different census coverage in 2011 and 2024, including\n"
        "missing 2011 coverage and incomplete 2024 participation in northern municipalities."
    )
    sections = [
        ("DIMENSIONS (RAW WORKSHEET)", f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}"),
        ("FIRST 10 ROWS", df.head(10).to_string(index=False, header=False)),
        ("RAW COLUMN TYPES", df.dtypes.to_string()),
        ("EMPTY CELLS BY COLUMN", df.isna().sum().to_string()),
        ("UNAVAILABLE VALUES (:) BY COLUMN", df.eq(":").sum().to_string()),
        ("DUPLICATE RAW ROWS", str(df.duplicated().sum())),
        ("RAW SUMMARY", df.describe(include="all").to_string()),
        ("INTERPRETATION", notes),
    ]
    report = "ASKdata - Initial Data Quality Report\n"
    report += "\n\n".join(f"=== {title} ===\n{body}" for title, body in sections)
    report += "\n"

    report_path = ROOT / "results/data_quality.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("\n=== FIRST 10 ROWS (RAW WORKSHEET) ===")
    print(df.head(10).to_string(index=False, header=False))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
