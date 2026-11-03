
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# 1. Dosya yolları
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"
RESULT_PATH = PROJECT_ROOT / "reports" / "advanced_model_comparison.csv"


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

for column in text_features:
    X[column] = X[column].fillna("").astype(str)


# 3. Grupları oluştur
groups = df["description"].fillna("").astype(str)

empty_mask = groups.str.strip().eq("")

groups = groups.where(
    ~empty_mask,
    "__missing_description_row_" + df.index.astype(str),
)


# 4. Eğitim-test ayrımı
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

groups_train = groups.iloc[train_idx]

assert not (
    set(groups.iloc[train_idx])
    & set(groups.iloc[test_idx])
)

print("=" * 60)
print("JOBSHIELD AI - GELİŞMİŞ MODEL DENEYLERİ")
print("=" * 60)

print(f"Eğitim kayıtları: {len(X_train)}")
print(f"Test kayıtları: {len(X_test)}")


# 5. Metin ön işleme pipeline'ı
text_preprocessor = ColumnTransformer(
    transformers=[
        (
            "description",
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
            "company_profile",
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
            "requirements",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=50_000,
                sublinear_tf=True,
            ),
            "requirements",
        ),
    ],
    remainder="drop",
)

text_pipeline = Pipeline([
    ("tfidf", text_preprocessor),
    (
        "svd",
        TruncatedSVD(
            n_components=100,
            random_state=42,
        ),
    ),
])


# 6. Yapısal özelliklerin ön işlemesi
categorical_pipeline = Pipeline([
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
])

binary_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent"),
    ),
    ("scaler", StandardScaler()),
])

structured_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
        (
            "binary",
            binary_pipeline,
            binary_features,
        ),
    ],
    remainder="drop",
)


# 7. Özellikleri hazırlayan fonksiyon
def prepare_features(X_fit, X_transform):
    text_processor = clone(text_pipeline)
    structured_processor = clone(structured_preprocessor)

    fit_text = text_processor.fit_transform(
        X_fit[text_features]
    )
    transform_text = text_processor.transform(
        X_transform[text_features]
    )

    fit_structured = structured_processor.fit_transform(
        X_fit
    )
    transform_structured = structured_processor.transform(
        X_transform
    )

    if hasattr(fit_structured, "toarray"):
        fit_structured = fit_structured.toarray()
        transform_structured = transform_structured.toarray()

    fit_final = np.hstack([
        fit_text,
        fit_structured,
    ])

    transform_final = np.hstack([
        transform_text,
        transform_structured,
    ])

    return fit_final, transform_final


# 8. Hiperparametre seçenekleri
model_configs = {
    "Random Forest": [
        {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_leaf": 2,
        },
        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 2,
        },
    ],
    "XGBoost": [
        {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
    ],
}


# 9. Model oluşturma fonksiyonu
def create_model(model_name, parameters):

    if model_name == "Random Forest":
        return RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            **parameters,
        )

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(
            (y_train == 0).sum()
            / (y_train == 1).sum()
        ),
        random_state=42,
        n_jobs=4,
        tree_method="hist",
        **parameters,
    )


# 10. Eğitim verisi içinde çapraz doğrulama
cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

cv_results = []

for fold, (fit_idx, validation_idx) in enumerate(
    cv.split(X_train, y_train, groups_train),
    start=1,
):
    print(f"\n{'=' * 20} FOLD {fold} {'=' * 20}")

    X_fit = X_train.iloc[fit_idx]
    X_validation = X_train.iloc[validation_idx]

    y_fit = y_train.iloc[fit_idx]
    y_validation = y_train.iloc[validation_idx]

    # Her fold için ön işleme yalnızca eğitim bölümünde öğrenilir.
    X_fit_final, X_validation_final = prepare_features(
        X_fit,
        X_validation,
    )

    print("Özellik matrisi:", X_fit_final.shape)

    for model_name, parameter_options in model_configs.items():

        for parameters in parameter_options:

            print(
                f"\n{model_name} | "
                f"{parameters}"
            )

            model = create_model(
                model_name,
                parameters,
            )

            start_time = time.time()

            model.fit(X_fit_final, y_fit)

            predictions = model.predict(
                X_validation_final
            )

            elapsed = time.time() - start_time

            score = f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )

            cv_results.append({
                "model": model_name,
                "parameters": str(parameters),
                "fold": fold,
                "f1_score": score,
                "training_time_seconds": round(
                    elapsed,
                    2,
                ),
            })

            print(
                f"Validation F1-score: {score:.4f}"
            )


# 11. En iyi hiperparametreleri seç
cv_df = pd.DataFrame(cv_results)

summary = (
    cv_df.groupby(
        ["model", "parameters"],
        as_index=False,
    )
    .agg(
        mean_cv_f1=("f1_score", "mean"),
        std_cv_f1=("f1_score", "std"),
        mean_training_time=(
            "training_time_seconds",
            "mean",
        ),
    )
)

summary = summary.sort_values(
    "mean_cv_f1",
    ascending=False,
)

print("\n" + "=" * 60)
print("ÇAPRAZ DOĞRULAMA SONUÇLARI")
print("=" * 60)

print(summary.to_string(index=False))


# 12. Her model için en iyi parametreleri seç
best_parameters = {}

for model_name in model_configs:

    model_summary = summary[
        summary["model"] == model_name
    ]

    best_row = model_summary.iloc[0]

    best_parameters[model_name] = eval(
        best_row["parameters"]
    )

    print(
        f"\n{model_name} için seçilen parametreler:"
    )
    print(best_parameters[model_name])


# 13. Ön işleme yalnızca tam eğitim verisinde öğrenilir
print("\nNihai modeller hazırlanıyor...")

X_train_final, X_test_final = prepare_features(
    X_train,
    X_test,
)


# 14. Seçilen modelleri eğit ve test kümesinde değerlendir
final_results = []

for model_name, parameters in best_parameters.items():

    print(f"\nNihai model: {model_name}")

    model = create_model(
        model_name,
        parameters,
    )

    start_time = time.time()

    model.fit(X_train_final, y_train)

    predictions = model.predict(X_test_final)

    elapsed = time.time() - start_time

    final_results.append({
        "model": model_name,
        "parameters": str(parameters),
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "training_time_seconds": round(
            elapsed,
            2,
        ),
    })


# 15. Sonuçları kaydet
final_df = pd.DataFrame(final_results)

RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

final_df.to_csv(
    RESULT_PATH,
    index=False,
)

cv_path = (
    PROJECT_ROOT
    / "reports"
    / "advanced_model_cv_results.csv"
)

cv_df.to_csv(
    cv_path,
    index=False,
)

print("\n" + "=" * 60)
print("NİHAİ TEST SONUÇLARI")
print("=" * 60)

print(final_df.to_string(index=False))

print(f"\nCV sonuçları: {cv_path}")
print(f"Nihai sonuçlar: {RESULT_PATH}")
print("\nGelişmiş model deneyleri tamamlandı.")
