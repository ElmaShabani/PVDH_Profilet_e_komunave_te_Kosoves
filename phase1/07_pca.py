import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELING = ROOT / "data/processed/modeling"
IDS = ["municipality", "year"]
FEATURES = [
    "children_share_pct", "older_share_pct", "working_age_sex_ratio", "urban_share_pct"
]


def reduce_dimensions(standardized, variance_threshold=0.90):
    if not 0 < variance_threshold <= 1:
        raise ValueError("Variance threshold must be greater than 0 and at most 1.")
    if list(standardized.columns) != IDS + FEATURES:
        raise ValueError("Unexpected standardized feature columns or order.")
    if len(standardized) < 2:
        raise ValueError("PCA requires at least two municipality records.")
    if standardized[IDS].isna().any().any() or standardized.duplicated(IDS).any():
        raise ValueError("Municipality/year identifiers must be present and unique.")
    if standardized.year.nunique() != 1:
        raise ValueError("Fit PCA to one census year at a time.")

    values = standardized[FEATURES].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("PCA requires finite, nonmissing features.")
    center = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    if not np.allclose(center, 0, atol=1e-8):
        raise ValueError("Features are not centered. Run phase1/06_feature_scaling.py first.")
    constant_zero = (values == 0).all(axis=0)
    if not np.all(np.isclose(std, 1, rtol=1e-8, atol=1e-8) | constant_zero):
        raise ValueError("Features must have unit variance or be constant zero columns.")

    centered = values - center
    _, singular_values, weights = np.linalg.svd(centered, full_matrices=False)
    tolerance = np.finfo(float).eps * max(centered.shape) * singular_values[0]
    positive = singular_values > tolerance
    singular_values, weights = singular_values[positive], weights[positive]
    if not len(singular_values):
        raise ValueError("All features are constant; PCA has no variance to explain.")

    # Fix arbitrary SVD signs so the largest absolute weight is positive.
    signs = np.sign(weights[np.arange(len(weights)), np.abs(weights).argmax(axis=1)])
    weights *= signs[:, None]
    eigenvalues = singular_values ** 2 / (len(values) - 1)
    ratios = eigenvalues / eigenvalues.sum()
    cumulative = ratios.cumsum()
    retained = min(int(np.searchsorted(cumulative, variance_threshold)) + 1, len(weights))
    names = [f"PC{number}" for number in range(1, len(weights) + 1)]

    scores = pd.DataFrame(centered @ weights[:retained].T, columns=names[:retained])
    scores = pd.concat([standardized[IDS].reset_index(drop=True), scores], axis=1)
    components = pd.DataFrame(weights.T, columns=names, index=FEATURES)
    components.insert(0, "center", center)
    components = components.rename_axis("feature").reset_index()
    variance = pd.DataFrame({
        "component": names,
        "explained_variance": eigenvalues,
        "explained_variance_ratio": ratios,
        "cumulative_variance_ratio": cumulative,
        "retained": np.arange(len(weights)) < retained,
    })
    return scores, components, variance


def main():
    parser = argparse.ArgumentParser(description="Reduce standardized demographic features with PCA.")
    parser.add_argument("--year", type=int, choices=[2011, 2024], default=2024)
    parser.add_argument("--variance-threshold", type=float, default=0.90)
    args = parser.parse_args()
    source = MODELING / f"standardized_features_{args.year}.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Run phase1/06_feature_scaling.py --year {args.year} first.")
    standardized = pd.read_csv(source, dtype={"municipality": "string", "year": "int64"})
    if not standardized.year.eq(args.year).all():
        raise ValueError("Input census year does not match the requested year.")
    scores, components, variance = reduce_dimensions(standardized, args.variance_threshold)

    output = MODELING / "pca"
    output.mkdir(parents=True, exist_ok=True)
    for name, table in {"scores": scores, "components": components, "variance": variance}.items():
        table.to_csv(output / f"pca_{name}_{args.year}.csv", index=False, encoding="utf-8")
    retained = int(variance.retained.sum())
    explained = float(variance.loc[variance.retained, "explained_variance_ratio"].sum())
    report = (
        "ASKdata - Principal Component Analysis Report\n\n"
        f"Census year: {args.year}\n"
        f"Municipalities: {len(scores)}\n"
        f"Original features: {len(FEATURES)}\n"
        f"Retained components: {retained}\n"
        f"Variance threshold: {args.variance_threshold:.2%}\n"
        f"Retained variance: {explained:.2%}\n\n"
        "EXPLAINED VARIANCE\n" + variance.to_string(index=False) + "\n\n"
        "METHOD\n"
        "PCA is fitted using NumPy SVD on the centered standardized feature matrix.\n"
        "The smallest number of components meeting the variance threshold is retained.\n"
        "Numerically zero-variance directions are excluded.\n"
        "Component weights are orthonormal coefficients, not feature correlations.\n"
        "Each component's largest absolute weight is oriented positively.\n"
        "The components file saves the fitted center and all nonzero-variance directions.\n"
        "Scores contain only the retained components and municipality/year identifiers.\n"
        "The variance threshold refers to variation in the selected standardized features,\n"
        "not accuracy, population coverage or proof that clustering will be useful.\n"
        "PCA inherits the exclusions from step 06; no missing observations are imputed.\n"
        "Original features, scaler parameters and exclusion reasons remain unchanged.\n"
        "Separately fitted yearly PCA axes are not directly comparable across censuses.\n"
        "For future held-out evaluation, fit the scaler and PCA on training observations only.\n"
        "Clustering and causal interpretation have not been performed.\n"
    )
    report_path = ROOT / f"results/pca_report_{args.year}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Year: {args.year} | Municipalities: {len(scores)}")
    print(f"Features: {len(FEATURES)} -> Components: {retained} | Retained variance: {explained:.2%}")
    print("\n=== FIRST 10 PCA ROWS ===")
    print(scores.head(10).to_string(index=False, float_format="{:.3f}".format))
    print(f"PCA scores, components and variance saved to {output}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
