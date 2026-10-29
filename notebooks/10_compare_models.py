
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# 1. Dosya yolları
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# 2. Önceki model çalışmalarından elde edilen sonuçlar
# Metrikler sahte ilan sınıfına (fraudulent = 1) aittir.
results = [
    {
        "model": "Logistic Regression\n(Description)",
        "accuracy": 0.9667,
        "precision": 0.6957,
        "recall": 0.5549,
        "f1_score": 0.6174,
    },
    {
        "model": "Linear SVM\n(Description)",
        "accuracy": 0.9746,
        "precision": 0.9457,
        "recall": 0.5029,
        "f1_score": 0.6566,
    },
    {
        "model": "Logistic Regression\n(3 Text Fields)",
        "accuracy": 0.9782,
        "precision": 0.7914,
        "recall": 0.7457,
        "f1_score": 0.7679,
    },
    {
        "model": "Logistic Regression\n(Text + Structured)",
        "accuracy": 0.9765,
        "precision": 0.7214,
        "recall": 0.8382,
        "f1_score": 0.7754,
    },
]

df = pd.DataFrame(results)


# 3. Sonuç tablosunu terminalde göster
print("\n" + "=" * 78)
print("JOBSHIELD AI - MODEL COMPARISON")
print("Metrics for the fake-job class (fraudulent = 1)")
print("=" * 78)

print(
    df.assign(
        model=df["model"].str.replace("\n", " ", regex=False)
    ).to_string(
        index=False,
        formatters={
            "accuracy": "{:.4f}".format,
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1_score": "{:.4f}".format,
        },
    )
)


# 4. CSV olarak kaydet
csv_path = PROJECT_ROOT / "reports" / "model_comparison.csv"
df.to_csv(csv_path, index=False)
print(f"\nComparison table saved to: {csv_path}")


# 5. Precision, recall ve F1-score karşılaştırma grafiği
metrics = ["precision", "recall", "f1_score"]
labels = ["Precision", "Recall", "F1-score"]

plot_df = df.set_index("model")[metrics]
ax = plot_df.plot(
    kind="bar",
    figsize=(12, 7),
    width=0.8,
)

ax.set_title("JobShield AI - Fake Job Detection Model Comparison")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xticklabels(
    [
        "Logistic Regression\n(Description)",
        "Linear SVM\n(Description)",
        "Logistic Regression\n(3 Text Fields)",
        "Logistic Regression\n(Text + Structured)",
    ],
    rotation=0,
)
ax.legend(labels, loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

figure_path = FIGURES_DIR / "model_comparison.png"
plt.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Comparison chart saved to: {figure_path}")
print("\nModel comparison completed.")
