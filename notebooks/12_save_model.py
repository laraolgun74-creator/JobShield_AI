
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Dosya yolları
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "combined_model.joblib"


# 2. Veri setini yükle
df = pd.read_csv(DATA_PATH)

text_features = [
    "description",
    "company_profile",
    "requirements",
]

categorical_features = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

binary_features = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

feature_columns = (
    text_features + categorical_features + binary_features
)

X = df[feature_columns].copy()
y = df["fraudulent"]

# Metin alanlarındaki eksik değerleri boş metne dönüştür.
for column in text_features:
    X[column] = X[column].fillna("").astype(str)

# Aynı açıklamaları eğitim ve test kümeleri arasında ayır.
groups = df["description"].fillna("").astype(str)
empty_mask = groups.str.strip().eq("")
groups = groups.where(
    ~empty_mask,
    "__missing_description_row_" + df.index.astype(str),
)


# 3. Önceki modelle aynı eğitim kümesini oluştur
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y=y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
y_train = y.iloc[train_idx]

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

assert not train_groups.intersection(test_groups), (
    "Eğitim ve test kümelerinde ortak açıklama grubu bulundu!"
)


# 4. Ön işleme adımları
preprocessor = ColumnTransformer(
    transformers=[
        (
            "description_tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=100_000,
                sublinear_tf=True,
            ),
            "description",
        ),
        (
            "company_profile_tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=30_000,
                sublinear_tf=True,
            ),
            "company_profile",
        ),
        (
            "requirements_tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=50_000,
                sublinear_tf=True,
            ),
            "requirements",
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Missing",
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            categorical_features,
        ),
        (
            "binary",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                ("scaler", StandardScaler()),
            ]),
            binary_features,
        ),
    ],
    remainder="drop",
)


# 5. Model pipeline'ı
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    ),
])


# 6. Modeli eğit
print("JobShield AI birleşik modeli eğitiliyor...")
model.fit(X_train, y_train)


# 7. Eğitilmiş modeli kaydet
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel başarıyla kaydedildi.")
print(f"Model yolu: {MODEL_PATH}")
print(f"Eğitim kayıt sayısı: {len(X_train)}")
print(f"Model dosyası mevcut: {MODEL_PATH.exists()}")
