
from pathlib import Path
import pandas as pd

# Proje ana dizini
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

# Mevcut sonuçları oku
basic = pd.read_csv(REPORTS / "model_comparison.csv")
advanced = pd.read_csv(REPORTS / "advanced_model_comparison.csv")

# Model adlarını ve özellik temsillerini düzenle
basic["model"] = basic["model"].str.replace("\n", " ", regex=False)
basic["feature_representation"] = [
    "Description TF-IDF",
    "Description TF-IDF",
    "Three text fields TF-IDF",
    "Text + structured features"
]

advanced["feature_representation"] = (
    "Three text fields (SVD) + structured features"
)

# Gelişmiş modellerdeki parametre ve eğitim süresi sütunları
basic["parameters"] = "Default / existing configuration"
basic["training_time_seconds"] = None

# Ortak sütunları seç
columns = [
    "model",
    "feature_representation",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "parameters",
    "training_time_seconds"
]

# Tabloları birleştir
final_results = pd.concat(
    [basic[columns], advanced[columns]],
    ignore_index=True
)

# F1-score'a göre büyükten küçüğe sırala
final_results = final_results.sort_values(
    by="f1_score",
    ascending=False
).reset_index(drop=True)

# Sıralama numarası ekle
final_results.insert(
    0,
    "rank",
    range(1, len(final_results) + 1)
)

# CSV olarak kaydet
output_path = REPORTS / "final_model_comparison.csv"
final_results.to_csv(output_path, index=False)

print("=" * 75)
print("JOBSHIELD AI - NİHAİ MODEL KARŞILAŞTIRMASI")
print("=" * 75)

print(
    final_results[
        ["rank", "model", "accuracy", "precision", "recall", "f1_score"]
    ].to_string(index=False)
)

print("\nDosya kaydedildi:")
print(output_path)
