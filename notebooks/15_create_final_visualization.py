
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(REPORTS / "final_model_comparison.csv")
df = df.sort_values("f1_score", ascending=True)

plt.figure(figsize=(11, 6))

bars = plt.barh(df["model"], df["f1_score"])

plt.xlabel("F1-score")
plt.title("JobShield AI - Final Model Comparison")
plt.xlim(0, 1)

for bar, value in zip(bars, df["f1_score"]):
    plt.text(
        value + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.4f}",
        va="center"
    )

plt.tight_layout()

output_path = FIGURES / "final_model_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("Grafik başarıyla oluşturuldu:")
print(output_path)
