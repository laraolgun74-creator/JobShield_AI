
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Proje ve veri dosyası
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"

df = pd.read_csv(DATA_PATH)

# Açıklaması boş kayıtlar için her satıra ayrı bir grup ver.
# Böylece bütün boş açıklamalar yanlışlıkla tek grup sayılmaz.
description = df["description"].fillna("").astype(str)
groups = description.where(
    description.str.strip().ne(""),
    "__missing_description_row_" + df.index.astype(str)
)

# Veriyi %80 eğitim, %20 test olacak şekilde böl
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, y=df["fraudulent"], groups=groups)
)

train = df.iloc[train_idx]
test = df.iloc[test_idx]

# Aynı açıklamanın iki tarafta birden bulunup bulunmadığını kontrol et
train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])
overlap = train_groups.intersection(test_groups)

print("=" * 50)
print("GRUP BAZLI VERİ BÖLME KONTROLÜ")
print("=" * 50)

print(f"Toplam kayıt: {len(df)}")
print(f"Eğitim kayıtları: {len(train)} ({len(train) / len(df):.1%})")
print(f"Test kayıtları: {len(test)} ({len(test) / len(df):.1%})")

print("\nEğitim sınıf dağılımı:")
print(train["fraudulent"].value_counts().sort_index())

print("\nTest sınıf dağılımı:")
print(test["fraudulent"].value_counts().sort_index())

print(f"\nİki kümede ortak açıklama grubu: {len(overlap)}")
print("\nKontrol tamamlandı.")
