
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# 1. Sayfa ayarları
st.set_page_config(
    page_title="JobShield AI",
    page_icon="🛡️",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "combined_model.joblib"


# 2. Modeli yükle
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# 3. Başlık ve açıklama
st.title("🛡️ JobShield AI")
st.subheader("Yapay zekâ destekli sahte iş ilanı analizi")
st.write(
    "İş ilanının bilgilerini girerek modelin sahte ilan riskine "
    "ilişkin tahminini inceleyebilirsin."
)

st.info(
    "Bu araç bir karar destek prototipidir. Tahminler kesin kanıt "
    "değildir; sonuçları tek başına karar vermek için kullanma."
)

if not MODEL_PATH.exists():
    st.error(
        "Eğitilmiş model bulunamadı. Önce "
        "notebooks/12_save_model.py dosyasını çalıştır."
    )
    st.stop()

model = load_model()


# 4. İlan bilgileri
st.header("İş ilanı bilgileri")

with st.form("job_analysis_form"):
    description = st.text_area(
        "İş ilanının açıklaması *",
        placeholder="İşin görevlerini ve sorumluluklarını yaz...",
        height=220,
    )

    company_profile = st.text_area(
        "Şirket profili",
        placeholder="Şirket hakkında verilen bilgiler...",
        height=120,
    )

    requirements = st.text_area(
        "İş için aranan nitelikler",
        placeholder="Deneyim, beceri ve eğitim gereksinimleri...",
        height=120,
    )

    st.subheader("Yapısal özellikler")

    col1, col2 = st.columns(2)

    with col1:
        employment_type = st.selectbox(
            "İstihdam türü",
            [
                "Belirtilmemiş",
                "Full-time",
                "Part-time",
                "Contract",
                "Temporary",
                "Other",
            ],
        )

        required_experience = st.selectbox(
            "Gerekli deneyim",
            [
                "Belirtilmemiş",
                "Internship",
                "Entry level",
                "Associate",
                "Mid-Senior level",
                "Director",
                "Executive",
            ],
        )

        required_education = st.selectbox(
            "Gerekli eğitim",
            [
                "Belirtilmemiş",
                "High School or equivalent",
                "Some College Coursework Completed",
                "Vocational",
                "Associate Degree",
                "Bachelor's Degree",
                "Master's Degree",
                "Doctorate",
                "Professional",
                "Unspecified",
            ],
        )

    with col2:
        industry = st.text_input(
            "Sektör",
            placeholder="Örn. Information Technology",
        )

        function = st.text_input(
            "İş fonksiyonu",
            placeholder="Örn. Engineering",
        )

        telecommuting = st.checkbox(
            "Uzaktan çalışma seçeneği var",
        )

        has_company_logo = st.checkbox(
            "İlan şirket logosu içeriyor",
        )

        has_questions = st.checkbox(
            "Başvuru sürecinde sorular bulunuyor",
        )

    submitted = st.form_submit_button(
        "🔎 İlanı analiz et",
        use_container_width=True,
    )


# 5. Tahmin
if submitted:
    if not description.strip():
        st.warning("Lütfen iş ilanının açıklamasını gir.")
    else:
        input_data = pd.DataFrame(
            [
                {
                    "description": description,
                    "company_profile": company_profile,
                    "requirements": requirements,
                    "employment_type": (
                        None
                        if employment_type == "Belirtilmemiş"
                        else employment_type
                    ),
                    "required_experience": (
                        None
                        if required_experience == "Belirtilmemiş"
                        else required_experience
                    ),
                    "required_education": (
                        None
                        if required_education == "Belirtilmemiş"
                        else required_education
                    ),
                    "industry": industry.strip() or None,
                    "function": function.strip() or None,
                    "telecommuting": int(telecommuting),
                    "has_company_logo": int(has_company_logo),
                    "has_questions": int(has_questions),
                }
            ]
        )

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)
        fake_probability = float(
            probabilities[classes.index(1)]
        )

        st.divider()
        st.header("Analiz sonucu")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("Modelin tahmini: Sahte ilan")
            else:
                st.success("Modelin tahmini: Gerçek ilan")

        with col2:
            st.metric(
                "Sahte ilan sınıfı için model olasılığı",
                f"%{fake_probability * 100:.1f}",
            )

        st.progress(fake_probability)

        st.caption(
            "Bu yüzde, modelin sahte sınıfına atadığı olasılıktır; "
            "gerçek dünyadaki sahte olma olasılığıyla aynı olmak "
            "zorunda değildir."
        )

        st.warning(
            "Model tahmini tek başına doğrulama sağlamaz. Şirketin "
            "resmî web sitesini, iletişim bilgilerini ve başvuru "
            "bağlantısını ayrıca kontrol et."
        )
