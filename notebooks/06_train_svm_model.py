
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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


# 1. Dosya yolları
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"


# 2. Veri setini yükle
df = pd.read_csv(DATA_PATH)

X = df["description"].fillna("").astype(str)
y = df["fraudulent"]

# Aynı açıklamalar eğitim ve test kümelerine dağılmasın.
groups = X.copy()
empty_mask = groups.str.strip().eq("")
groups = groups.where(
    ~empty_mask,
    "__missing_description_row_" + df.index.astype(str),
)


# 3. Aynı eğitim/test bölmesini oluştur
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y=y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# 4. TF-IDF + Linear SVM pipeline'ı
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=100_000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        ),
    ),
])


# 5. Modeli eğit
print("Linear SVM modeli eğitiliyor...")
model.fit(X_train, y_train)

# 6. Test kümesinde tahmin yap
y_pred = model.predict(X_test)


# 7. Sonuçları değerlendir
print("\n" + "=" * 55)
print("JOBSHIELD AI - LINEAR SVM SONUÇLARI")
print("=" * 55)

print(f"Toplam kayıt: {len(df)}")
print(f"Eğitim kayıtları: {len(X_train)}")
print(f"Test kayıtları: {len(X_test)}")

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

print("\nLinear SVM değerlendirmesi tamamlandı.")
