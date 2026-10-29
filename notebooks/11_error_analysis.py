
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Dosya yolları
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# 2. Veriyi yükle ve özellikleri tanımla
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

for column in text_features:
    X[column] = X[column].fillna("").astype(str)


# 3. Aynı açıklama gruplarını eğitim ve test arasında ayırma
groups = df["description"].fillna("").astype(str)
empty_mask = groups.str.strip().eq("")

groups = groups.where(
    ~empty_mask,
    "__missing_description_row_" + df.index.astype(str),
)

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

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])
overlap = train_groups.intersection(test_groups)

assert not overlap, "Eğitim ve test arasında ortak açıklama grubu var!"


# 4. Modelin ön işleme adımları
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
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
        (
            "binary",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler()),
            ]),
            binary_features,
        ),
    ],
    remainder="drop",
)


# 5. Modeli eğit ve tahminleri üret
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

print("Birleşik model yeniden eğitiliyor...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# 6. Hata türlerini belirle
analysis = df.iloc[test_idx].copy()

analysis["actual_label"] = y_test.to_numpy()
analysis["predicted_label"] = y_pred
analysis["correct_prediction"] = (
    analysis["actual_label"] == analysis["predicted_label"]
)

analysis["error_type"] = "Correct"
analysis.loc[
    (analysis["actual_label"] == 1)
    & (analysis["predicted_label"] == 0),
    "error_type",
] = "False Negative"

analysis.loc[
    (analysis["actual_label"] == 0)
    & (analysis["predicted_label"] == 1),
    "error_type",
] = "False Positive"

analysis.loc[
    (analysis["actual_label"] == 1)
    & (analysis["predicted_label"] == 1),
    "error_type",
] = "True Positive"

analysis.loc[
    (analysis["actual_label"] == 0)
    & (analysis["predicted_label"] == 0),
    "error_type",
] = "True Negative"


# Hatalı ve doğru tahminlerde eksik alan oranlarını karşılaştır.
missing_columns = [
    "company_profile",
    "requirements",
    "description",
]

print("\nEksik alan oranları - tüm test kümesi:")
missing_summary = (
    analysis.groupby("error_type")[missing_columns]
    .apply(lambda group: group.isna().mean().mul(100).round(1))
)

print(missing_summary.to_string())

# 7. Yanlış sınıflandırılan ilanları CSV'ye kaydet
errors = analysis.loc[~analysis["correct_prediction"]].copy()

columns_to_export = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "fraudulent",
    "actual_label",
    "predicted_label",
    "error_type",
]

# CSV'de bulunan sütunları seç.
columns_to_export = [
    column for column in columns_to_export
    if column in errors.columns
]

errors_path = REPORTS_DIR / "error_analysis.csv"
errors[columns_to_export].to_csv(errors_path, index=False)


# Eksik alan durumuna göre yanlış sınıflandırma oranları
print("\nEksik alan durumuna göre hata oranları:")

analysis["has_company_profile"] = (
    analysis["company_profile"].notna()
)
analysis["has_requirements"] = (
    analysis["requirements"].notna()
)

for feature, label in [
    ("has_company_profile", "Şirket profili"),
    ("has_requirements", "Gereksinimler"),
]:
    print(f"\n{label}:")

    summary = (
        analysis.groupby(feature, dropna=False)
        .agg(
            ilan_sayisi=("correct_prediction", "size"),
            hata_sayisi=("correct_prediction", lambda s: (~s).sum()),
            hata_orani=("correct_prediction", lambda s: (~s).mean() * 100),
        )
        .round(2)
    )

    summary.index = summary.index.map({
        True: "Dolu",
        False: "Eksik",
    })

    print(summary.to_string())

# 8. Sonuç sayılarını yazdır
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 60)
print("JOBSHIELD AI - ERROR ANALYSIS")
print("=" * 60)
print(f"Test kayıtları: {len(X_test)}")
print(f"Ortak açıklama grubu: {len(overlap)}")
print(f"True Negative:  {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"True Positive:  {tp}")
print(f"Toplam yanlış tahmin: {len(errors)}")

print("\nHata türlerinin dağılımı:")
print(errors["error_type"].value_counts())

print(f"\nHata kayıtları yerel CSV'ye kaydedildi: {errors_path}")


# 9. Confusion matrix görseli
fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(cm, interpolation="nearest")
ax.figure.colorbar(image, ax=ax)

ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["Predicted real", "Predicted fake"],
    yticklabels=["Actual real", "Actual fake"],
    title="JobShield AI - Confusion Matrix",
    ylabel="Actual label",
    xlabel="Predicted label",
)

threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black",
        )

plt.tight_layout()

figure_path = FIGURES_DIR / "error_analysis.png"
plt.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Confusion matrix görseli kaydedildi: {figure_path}")
print("\nHata analizi tamamlandı.")
