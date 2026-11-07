
from pathlib import Path
import pandas as pd

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input and output files
INPUT_FILE = PROJECT_ROOT / "powerbi" / "cv_summary.csv"
OUTPUT_FILE = PROJECT_ROOT / "powerbi" / "cv_summary_best_configs.csv"


def main():
    # Load cross-validation summary
    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "model",
        "parameters",
        "mean_f1",
        "std_f1",
        "mean_training_time",
        "fold_count",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Select the best configuration for each model
    best_configs = (
        df.loc[df.groupby("model")["mean_f1"].idxmax()]
        .sort_values("mean_f1", ascending=False)
        .reset_index(drop=True)
    )

    # Save the cleaned dataset
    best_configs.to_csv(OUTPUT_FILE, index=False)

    print("Best configurations saved successfully!")
    print(best_configs.to_string(index=False))
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
