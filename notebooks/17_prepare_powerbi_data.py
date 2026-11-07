
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"
OUTPUT_DIR = PROJECT_ROOT / "powerbi"

OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Model performance
model_results = pd.read_csv(
    REPORTS_DIR / "final_model_comparison.csv"
)

model_performance = model_results[
    [
        "rank",
        "model",
        "feature_representation",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "training_time_seconds",
    ]
].copy()

model_performance.to_csv(
    OUTPUT_DIR / "model_performance.csv",
    index=False
)

# 2. Cross-validation summary
cv_results = pd.read_csv(
    REPORTS_DIR / "advanced_model_cv_results.csv"
)

cv_summary = (
    cv_results
    .groupby(["model", "parameters"], as_index=False)
    .agg(
        mean_f1=("f1_score", "mean"),
        std_f1=("f1_score", "std"),
        mean_training_time=("training_time_seconds", "mean"),
        fold_count=("fold", "nunique"),
    )
)

cv_summary.to_csv(
    OUTPUT_DIR / "cv_summary.csv",
    index=False
)

# 3. Error summary
error_data = pd.read_csv(
    REPORTS_DIR / "error_analysis.csv"
)

error_summary = (
    error_data
    .groupby("error_type")
    .size()
    .reset_index(name="count")
)

error_summary.to_csv(
    OUTPUT_DIR / "error_summary.csv",
    index=False
)

# 4. Error details (without long text fields)
error_details = error_data[
    [
        "title",
        "actual_label",
        "predicted_label",
        "error_type",
    ]
].copy()

error_details.to_csv(
    OUTPUT_DIR / "error_details.csv",
    index=False
)

# 5. Dataset overview and class distribution
dataset = pd.read_csv(DATA_PATH)

total_postings = len(dataset)
class_counts = dataset["fraudulent"].value_counts().sort_index()

real_count = int(class_counts.get(0, 0))
fake_count = int(class_counts.get(1, 0))

dataset_overview = pd.DataFrame(
    [{
        "total_postings": total_postings,
        "real_postings": real_count,
        "fake_postings": fake_count,
        "real_percentage": real_count / total_postings * 100,
        "fake_percentage": fake_count / total_postings * 100,
        "feature_count": len(dataset.columns),
    }]
)

dataset_overview.to_csv(
    OUTPUT_DIR / "dataset_overview.csv",
    index=False
)

class_distribution = pd.DataFrame(
    [
        {
            "class": "Real",
            "label": 0,
            "count": real_count,
            "percentage": real_count / total_postings * 100,
        },
        {
            "class": "Fake",
            "label": 1,
            "count": fake_count,
            "percentage": fake_count / total_postings * 100,
        },
    ]
)

class_distribution.to_csv(
    OUTPUT_DIR / "class_distribution.csv",
    index=False
)

print("Power BI export files created successfully.")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Total postings: {total_postings}")
print(f"Real postings: {real_count}")
print(f"Fake postings: {fake_count}")
print("\nCreated files:")

for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"- {file.name}")
