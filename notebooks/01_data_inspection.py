
from pathlib import Path

import pandas as pd

# Proje kök dizinini ve veri dosyasını belirle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"

print("=" * 60)
print("JOBSHIELD AI - VERİ SETİ İNCELEMESİ")
print("=" * 60)

# CSV dosyasını oku
df = pd.read_csv(DATA_PATH)

# 1. Veri boyutu
print("\n1. VERİ BOYUTU")
print(f"Satır sayısı: {df.shape[0]}")
print(f"Sütun sayısı: {df.shape[1]}")

# 2. Sütun adları ve veri tipleri
print("\n2. SÜTUNLAR VE VERİ TİPLERİ")
print(df.dtypes.to_string())

# 3. İlk beş satır
print("\n3. İLK 5 SATIR")
print(df.head().to_string())

# 4. Eksik değerler
print("\n4. EKSİK DEĞER ANALİZİ")
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2),
})
missing = missing.sort_values("missing_percent", ascending=False)
print(missing.to_string())

# 5. Hedef değişken
print("\n5. HEDEF DEĞİŞKEN: fraudulent")
if "fraudulent" in df.columns:
    print("Sınıf sayıları:")
    print(df["fraudulent"].value_counts(dropna=False).to_string())
    print("\nSınıf yüzdeleri (%):")
    print(
        (df["fraudulent"].value_counts(normalize=True, dropna=False) * 100)
        .round(2)
        .to_string()
    )
else:
    print("UYARI: 'fraudulent' sütunu bulunamadı.")


# 6. Tekrarlanan kayıt ve açıklama kontrolü
print("\n6. TEKRARLANAN KAYIT KONTROLÜ")

print(
    "Tamamen aynı satır sayısı:",
    df.duplicated().sum()
)

print(
    "Tekrarlanan ilan açıklaması sayısı:",
    df["description"].dropna().duplicated().sum()
)

print(
    "Tekrarlanan ilan başlığı sayısı:",
    df["title"].dropna().duplicated().sum()
)


# 7. Aynı açıklamaya sahip ilanların etiket kontrolü
print("\n7. AYNI AÇIKLAMADAKİ ETİKET ÇAKIŞMASI")

description_groups = (
    df.dropna(subset=["description"])
    .groupby("description")["fraudulent"]
    .agg(["size", "nunique", "min", "max"])
)

repeated_groups = description_groups[
    description_groups["size"] > 1
]

conflicting_groups = repeated_groups[
    repeated_groups["nunique"] > 1
]

print(
    "Tekrarlanan açıklama grubu sayısı:",
    len(repeated_groups)
)
print(
    "Farklı etiket içeren açıklama grubu sayısı:",
    len(conflicting_groups)
)

print("\nİlk 5 çakışmalı grup:")
print(conflicting_groups.head().to_string())

print("\nİnceleme tamamlandı.")
