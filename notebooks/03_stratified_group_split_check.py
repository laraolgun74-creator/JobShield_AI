
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"

df = pd.read_csv(DATA_PATH)

# Aynı boş olmayan açıklamalar aynı grupta kalır.
# Boş açıklamalar ayrı gruplar olarak ele alınır.
description = df["description"].fillna("").astype(str)

groups = description.where(
    description.str.strip().ne(""),
    "__missing_description_row_" + df.index.astype(str)
)

y = df["fraudulent"]

# 5 katlı grup ve sınıf dağılımı gözeten bölme.
# İlk kat yaklaşık %20 test, kalan katlar eğitim için kullanılacak.
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, y=y, groups=groups)
)

train = df.iloc[train_idx]
test = df.iloc[test_idx]

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])
overlap = train_groups.intersection(test_groups)

def show_distribution(name, data):
    counts = data["fraudulent"].value_counts().sort_index()
    percentages = (
        data["fraudulent"].value_counts(normalize=True)
        .sort_index() * 100
    ).round(2)

    print(f"\n{name}")
    print(f"Kayıt sayısı: {len(data)} ({len(data) / len(df):.1%})")
    print("Sınıf sayıları:")
    print(counts.to_string())
    print("Sınıf yüzdeleri (%):")
    print(percentages.to_string())

print("=" * 55)
print("STRATIFIED GROUP K-FOLD BÖLME KONTROLÜ")
print("=" * 55)

show_distribution("EĞİTİM", train)
show_distribution("TEST", test)

print(f"\nOrtak açıklama grubu: {len(overlap)}")
print("Kontrol tamamlandı.")
