
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"

df = pd.read_csv(DATA_PATH)

# Önceden kullandığımız grup oluşturma yöntemi
description = df["description"].fillna("").astype(str)

groups = description.where(
    description.str.strip().ne(""),
    "__missing_description_row_" + df.index.astype(str)
)

y = df["fraudulent"]

# Mevcut eğitim/test ayrımını aynen koru
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, y=y, groups=groups)
)

test = df.iloc[test_idx].copy()

# Sunumda kullanmak için bir gerçek ve bir sahte ilan seç
real_example = test[test["fraudulent"] == 0].iloc[0]
fake_example = test[test["fraudulent"] == 1].iloc[0]

columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "fraudulent"
]

print("=" * 65)
print("JOBSHIELD AI - SUNUM DEMO İLANLARI")
print("=" * 65)

for name, example in [
    ("GERÇEK İLAN ÖRNEĞİ", real_example),
    ("SAHTE İLAN ÖRNEĞİ", fake_example)
]:
    print(f"\n{'=' * 65}")
    print(name)
    print("=" * 65)

    for column in columns:
        value = example[column]

        if pd.isna(value):
            value = "Bilgi bulunmuyor"

        print(f"\n{column}:")
        print(value)

    print("\nVeri setindeki gerçek etiket:", int(example["fraudulent"]))

print("\nDemo ilanları mevcut test kümesinden seçildi.")
