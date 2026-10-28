
from pathlib import Path

import pandas as pd


# 1. Dosya yolları
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"


# 2. Veri setini yükle
df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("JOBSHIELD AI - YAPISAL ÖZELLİKLERİ İNCELEME")
print("=" * 65)

# 3. Sütun adları ve veri tipleri
print("\n1. TÜM SÜTUNLAR VE VERİ TİPLERİ")
print("-" * 65)

for column in df.columns:
    print(f"{column:25} | {str(df[column].dtype):10} | {df[column].nunique(dropna=True)} farklı değer")


# 4. Eksik değer raporu
print("\n2. EKSİK DEĞERLER")
print("-" * 65)

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2),
})

missing = missing[missing["missing_count"] > 0].sort_values(
    "missing_percent",
    ascending=False,
)

print(missing.to_string())


# 5. İlk model için değerlendirilecek özellikler
candidate_features = [
    "company_profile",
    "requirements",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

print("\n3. ADAY ÖZELLİKLER")
print("-" * 65)

for column in candidate_features:
    if column not in df.columns:
        print(f"\n{column}: Sütun bulunamadı.")
        continue

    print(f"\n{column}")
    print(f"Eksik değer: {df[column].isna().sum()} ({df[column].isna().mean() * 100:.2f}%)")
    print(f"Farklı değer sayısı: {df[column].nunique(dropna=True)}")

    if df[column].nunique(dropna=True) <= 15:
        print("Değer dağılımı:")
        print(df[column].value_counts(dropna=False).head(15).to_string())


# 6. Sınıf oranlarıyla birlikte kategorik özellikleri incele
print("\n4. SAHTE İLAN ORANLARI - KATEGORİK ÖZELLİKLER")
print("-" * 65)

categorical_features = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

for column in categorical_features:
    if column not in df.columns:
        continue

    summary = (
        df.groupby(column, dropna=False)["fraudulent"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .head(10)
    )

    summary["fraudulent_percent"] = (summary["mean"] * 100).round(2)
    summary = summary.drop(columns="mean")

    print(f"\n{column}")
    print(summary.to_string())


print("\n" + "=" * 65)
print("YAPISAL ÖZELLİK İNCELEMESİ TAMAMLANDI")
print("=" * 65)
