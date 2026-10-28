
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Dosya yolları
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"


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

feature_columns = text_features + categorical_features + binary_features

X = df[feature_columns].copy()
y = df["fraudulent"]

# Metin sütunlarındaki eksik değerleri boş metne dönüştür.
for column in text_features:
    X[column] = X[column].fillna("").astype(str)

# Aynı açıklamalar eğitim ve test kümelerine dağılmasın.
groups = df["description"].fillna("").astype(str)
empty_mask = groups.str.strip().eq("")
groups = groups.where(
    ~empty_mask,
    "__missing_description_row_" + df.index.astype(str),
)


# 3. Eğitim ve test kümelerini oluştur
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y=y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Grupların kümeler arasında paylaşılmadığını doğrula.
train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])
overlap = train_groups.intersection(test_groups)

assert len(overlap) == 0, "Eğitim ve test arasında ortak açıklama grubu var!"


# 4. Metin sütunları için TF-IDF
# Her metin sütunu ayrı vektörleştirilir.
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


# 5. Tam model pipeline'ı
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


# 6. Eğit
print("Birleşik model eğitiliyor...")
model.fit(X_train, y_train)

# 7. Test kümesinde tahmin yap
y_pred = model.predict(X_test)


# 8. Sonuçları yazdır
print("\n" + "=" * 60)
print("JOBSHIELD AI - METİN + YAPISAL ÖZELLİKLER MODELİ")
print("=" * 60)

print(f"Toplam kayıt: {len(df)}")
print(f"Eğitim kayıtları: {len(X_train)}")
print(f"Test kayıtları: {len(X_test)}")
print(f"Ortak açıklama grubu: {len(overlap)}")

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Gerçek ilan (0)", "Sahte ilan (1)"],
        zero_division=0,
    )
)

print("\nBirleşik model değerlendirmesi tamamlandı.")
