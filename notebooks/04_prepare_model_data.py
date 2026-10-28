
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# 1. Dosya yolları
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"


# 2. Veri setini yükle
df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("JOBSHIELD AI - MODEL VERİSİ HAZIRLAMA")
print("=" * 55)
print(f"Toplam kayıt: {len(df)}")


# 3. Açıklama metinlerinden grup oluştur
# Aynı açıklamalar eğitim ve test kümelerine dağılmasın.
groups = df["description"].fillna("").astype(str)

# Boş açıklamaları birbirinden bağımsız gruplar olarak ele al.
empty_mask = groups.str.strip().eq("")
groups = groups.where(
    ~empty_mask,
    "__missing_description_row_" + df.index.astype(str),
)


# 4. Eğitim ve test kümelerini ayır
y = df["fraudulent"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(df, y=y, groups=groups)
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()


# 5. Bölme sonuçlarını kontrol et
train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])
overlap = train_groups.intersection(test_groups)

print(f"\nEğitim kayıtları: {len(train_df)}")
print(f"Test kayıtları: {len(test_df)}")
print(f"Eğitimdeki sahte ilan sayısı: {train_df['fraudulent'].sum()}")
print(f"Testteki sahte ilan sayısı: {test_df['fraudulent'].sum()}")
print(f"Ortak açıklama grubu: {len(overlap)}")

assert len(overlap) == 0, "Veri sızıntısı kontrolü başarısız!"
assert set(train_df.index).isdisjoint(set(test_df.index))

print("\nVeri hazırlama ve kontroller başarıyla tamamlandı.")
