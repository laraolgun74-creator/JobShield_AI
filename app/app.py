
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="J/SHIELD — Trust Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "combined_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Trained model not found. Run notebooks/12_save_model.py first.")
    st.stop()

model = load_model()


# =====================================================
# 2. DESIGN SYSTEM
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --ink: #171D1C;
    --panel: #202B26;
    --panel2: #26342E;
    --moss: #293B36;
    --lime: #C5D86D;
    --coral: #E77D61;
    --ivory: #EAE9DF;
    --sage: #89968C;
    --line: #46534B;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--ink);
    color: var(--ivory);
}

.block-container {
    max-width: 1160px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

h1, h2, h3, p, label {
    color: var(--ivory);
}

/* BRAND */

.brandbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0 20px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 42px;
}

.brand-left {
    display: flex;
    align-items: center;
    gap: 13px;
}

.brand-symbol {
    width: 43px;
    height: 43px;
    flex-shrink: 0;
}

.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
    letter-spacing: -1.3px;
    color: var(--ivory);
}

.brand-name span {
    color: var(--lime);
}

.brand-sub {
    color: var(--sage);
    font: 10px 'DM Mono', monospace;
    letter-spacing: 1.4px;
    margin-top: 2px;
}

.brand-status {
    color: var(--lime);
    border: 1px solid #596847;
    padding: 8px 12px;
    font: 10px 'DM Mono', monospace;
    letter-spacing: 1px;
}

/* HERO */

.hero {
    display: grid;
    grid-template-columns: 1.55fr 0.75fr;
    gap: 42px;
    align-items: center;
    padding: 12px 0 42px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 38px;
}

.eyebrow, .mono {
    font-family: 'DM Mono', monospace;
    letter-spacing: 1.5px;
    font-size: 10px;
}

.eyebrow {
    color: var(--lime);
    margin-bottom: 21px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--ivory);
    font-size: clamp(42px, 6vw, 72px);
    font-weight: 700;
    line-height: 0.99;
    letter-spacing: -3.5px;
    margin-bottom: 22px;
}

.hero-title em {
    color: var(--lime);
    font-style: normal;
}

.hero-description {
    color: #AEB8AD;
    font-size: 14px;
    line-height: 1.85;
    max-width: 590px;
}

.hero-art {
    background: var(--moss);
    border: 1px solid #526157;
    padding: 25px;
    min-height: 205px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}

.art-top {
    display: flex;
    justify-content: space-between;
    color: var(--sage);
    font: 10px 'DM Mono', monospace;
}

.art-mark {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--lime);
    font-size: 72px;
    font-weight: 600;
    line-height: 1;
    letter-spacing: -5px;
}

.art-bottom {
    border-top: 1px solid #607064;
    padding-top: 12px;
    color: var(--ivory);
    font: 10px 'DM Mono', monospace;
    letter-spacing: 1px;
}

/* SECTION HEADINGS */

.section-heading {
    display: flex;
    align-items: baseline;
    gap: 13px;
    margin: 36px 0 8px;
}

.section-number {
    color: var(--lime);
    font: 12px 'DM Mono', monospace;
}

.section-title {
    color: var(--ivory);
    font: 600 20px 'Space Grotesk', sans-serif;
}

.section-description {
    color: var(--sage);
    font-size: 12px;
    margin: 0 0 23px 29px;
}

.subsection {
    color: var(--lime);
    font: 11px 'DM Mono', monospace;
    letter-spacing: 1.3px;
    padding: 0 0 13px;
    border-bottom: 1px solid var(--line);
    margin: 12px 0 18px;
}

/* FORM */

div[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

div[data-testid="stTextArea"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label {
    color: #C9CEC2 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: var(--panel) !important;
    color: var(--ivory) !important;
    border: 1px solid #526057 !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
}

div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: var(--lime) !important;
    box-shadow: 0 0 0 1px var(--lime) !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: var(--panel) !important;
    color: var(--ivory) !important;
    border: 1px solid #526057 !important;
    border-radius: 2px !important;
}

div[data-testid="stCheckbox"] label {
    color: #C9CEC2 !important;
    font-size: 12px !important;
}

div[data-testid="stCheckbox"] input:checked + div {
    background: var(--lime) !important;
    border-color: var(--lime) !important;
}

.form-divider {
    height: 1px;
    background: var(--line);
    margin: 23px 0;
}

.signal-note {
    color: var(--sage);
    font: 10px 'DM Mono', monospace;
    letter-spacing: 1px;
    margin: 13px 0;
}

/* BUTTON */

div[data-testid="stFormSubmitButton"] button {
    background: var(--lime) !important;
    color: var(--ink) !important;
    border: 1px solid var(--lime) !important;
    border-radius: 2px !important;
    min-height: 50px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
}

div[data-testid="stFormSubmitButton"] button p,
div[data-testid="stFormSubmitButton"] button span {
    color: var(--ink) !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: var(--ivory) !important;
    border-color: var(--ivory) !important;
}

/* RESULT REPORT */

.report-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--line);
    padding-bottom: 15px;
    margin: 32px 0 20px;
}

.report-id {
    color: var(--lime);
    font: 10px 'DM Mono', monospace;
    letter-spacing: 1.4px;
}

.report-status {
    color: var(--sage);
    font: 10px 'DM Mono', monospace;
}

.result-panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 5px solid var(--lime);
    padding: 26px;
    min-height: 185px;
}

.result-panel.fake {
    border-left-color: var(--coral);
}

.result-label {
    color: var(--sage);
    font: 10px 'DM Mono', monospace;
    letter-spacing: 1.4px;
    margin-bottom: 23px;
}

.result-value {
    color: var(--ivory);
    font: 700 clamp(25px, 3vw, 35px) 'Space Grotesk', sans-serif;
    letter-spacing: -1.4px;
}

.result-note {
    color: #AEB8AD;
    font-size: 12px;
    margin-top: 12px;
    line-height: 1.6;
}

.score-panel {
    background: var(--ivory);
    padding: 26px;
    min-height: 185px;
}

.score-label {
    color: #626B60;
    font: 10px 'DM Mono', monospace;
    letter-spacing: 1.4px;
    margin-bottom: 23px;
}

.score-value {
    color: var(--ink);
    font: 700 49px 'Space Grotesk', sans-serif;
    letter-spacing: -2px;
}

.score-note {
    color: #626B60;
    font-size: 12px;
    margin-top: 12px;
}

.report-explanation {
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    padding: 17px 0;
    color: #B7C0B4;
    font-size: 12px;
    line-height: 1.8;
    margin-top: 20px;
}

.risk-warning {
    background: #342B25;
    border-left: 4px solid var(--coral);
    padding: 16px 18px;
    color: #F0D7CA;
    font-size: 12px;
    line-height: 1.8;
    margin-top: 18px;
}

.risk-warning strong {
    color: var(--coral);
    font: 11px 'DM Mono', monospace;
    letter-spacing: 1px;
}

/* CUSTOM FAKE SCORE BAR */

.score-track {
    width: 100%;
    height: 8px;
    background: #46534B;
    margin-top: 0;
    overflow: hidden;
}

.score-fill {
    height: 8px;
    transition: width 0.3s ease;
}

/* FOOTER */

.footer {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    border-top: 1px solid var(--line);
    margin-top: 55px;
    padding-top: 18px;
    color: var(--sage);
    font: 9px 'DM Mono', monospace;
    letter-spacing: 0.8px;
}

@media (max-width: 760px) {
    .hero {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 43px;
        letter-spacing: -2px;
    }

    .hero-art {
        min-height: 150px;
    }

    .brand-status {
        font-size: 8px;
    }

    .footer {
        flex-direction: column;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. BRAND HEADER
# =====================================================

st.markdown("""
<div class="brandbar">
    <div class="brand-left">
        <svg class="brand-symbol" viewBox="0 0 64 64"
             xmlns="http://www.w3.org/2000/svg">
            <path d="M32 3 L56 15 L53 43 L32 61 L11 43 L8 15 Z"
                  fill="none" stroke="#C5D86D" stroke-width="2.5"/>
            <path d="M32 13 L46 20 L44 40 L32 50 L20 40 L18 20 Z"
                  fill="#C5D86D" fill-opacity=".12"/>
            <path d="M23 32 L29 38 L42 23"
                  fill="none" stroke="#EAE9DF" stroke-width="3"
                  stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div>
            <div class="brand-name">J<span>/</span>SHIELD</div>
            <div class="brand-sub">TRUST INTELLIGENCE / 001</div>
        </div>
    </div>
    <div class="brand-status">● SYSTEM ONLINE</div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 4. HERO
# =====================================================

st.markdown("""
<div class="hero">
    <div>
        <div class="eyebrow">01 / JOB POSTING INVESTIGATION</div>
        <div class="hero-title">
            Not every opportunity<br>
            is what it <em>seems.</em>
        </div>
        <div class="hero-description">
            Examine the signals behind a job posting.
            Identify suspicious patterns with machine learning
            and make more informed decisions before you apply.
        </div>
    </div>
    <div class="hero-art">
        <div class="art-top">
            <span>J/SHIELD</span>
            <span>MODULE 001</span>
        </div>
        <div class="art-mark">◈</div>
        <div class="art-bottom">
            FAKE JOB DETECTION &nbsp; / &nbsp; ML CLASSIFICATION
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 5. FORM HEADING
# =====================================================

st.markdown("""
<div class="section-heading">
    <span class="section-number">02 /</span>
    <span class="section-title">Examine a posting</span>
</div>
<div class="section-description">
    Enter the available information. Required fields are marked.
</div>
""", unsafe_allow_html=True)


# =====================================================
# 6. INPUT FORM
# =====================================================

with st.form("job_analysis_form"):

    st.markdown(
        '<div class="subsection">A — PRIMARY INFORMATION</div>',
        unsafe_allow_html=True,
    )

    description = st.text_area(
        "Job Description *",
        placeholder="Paste the full job description...",
        height=180,
    )

    col1, col2 = st.columns(2)

    with col1:
        company_profile = st.text_area(
            "Company Profile",
            placeholder="What does the company say about itself?",
            height=115,
        )

    with col2:
        requirements = st.text_area(
            "Job Requirements",
            placeholder="Skills, qualifications and experience...",
            height=115,
        )

    st.markdown('<div class="form-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="subsection">B — STRUCTURED SIGNALS</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        employment_type = st.selectbox(
            "Employment Type",
            [
                "Not specified",
                "Full-time",
                "Part-time",
                "Contract",
                "Temporary",
                "Other",
            ],
        )

        required_experience = st.selectbox(
            "Required Experience",
            [
                "Not specified",
                "Internship",
                "Entry level",
                "Associate",
                "Mid-Senior level",
                "Director",
                "Executive",
            ],
        )

        required_education = st.selectbox(
            "Required Education",
            [
                "Not specified",
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
            "Industry",
            placeholder="e.g. Information Technology",
        )

        function = st.text_input(
            "Job Function",
            placeholder="e.g. Engineering",
        )

        st.markdown(
            '<div class="signal-note">ADDITIONAL SIGNALS</div>',
            unsafe_allow_html=True,
        )

        telecommuting = st.checkbox("Remote work available")
        has_company_logo = st.checkbox("Company logo included")
        has_questions = st.checkbox("Screening questions included")

    st.markdown('<div class="form-divider"></div>', unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "↗  RUN JOB ANALYSIS",
        use_container_width=True,
    )


# =====================================================
# 7. PREDICTION AND REPORT
# =====================================================

if submitted:

    if not description.strip():
        st.warning("Please enter a job description.")

    else:
        input_data = pd.DataFrame(
            [
                {
                    "description": description,
                    "company_profile": company_profile,
                    "requirements": requirements,
                    "employment_type": (
                        None if employment_type == "Not specified"
                        else employment_type
                    ),
                    "required_experience": (
                        None if required_experience == "Not specified"
                        else required_experience
                    ),
                    "required_education": (
                        None if required_education == "Not specified"
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

        is_fake = prediction == 1
        result_class = "fake" if is_fake else "real"

        result_text = (
            "Fake job posting" if is_fake else "Real job posting"
        )

        result_note = (
            "The model classified this posting as fake."
            if is_fake
            else "The model classified this posting as real."
        )

        bar_color = "#E77D61" if is_fake else "#C5D86D"
        bar_width = fake_probability * 100

        st.markdown("""
        <div class="section-heading">
            <span class="section-number">03 /</span>
            <span class="section-title">Investigation report</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="report-header">
            <span class="report-id">REPORT / CLASSIFICATION OUTPUT</span>
            <span class="report-status">MODEL ASSESSMENT</span>
        </div>
        """, unsafe_allow_html=True)

        result_col, score_col = st.columns([1.15, 0.85])

        with result_col:
            st.markdown(
                f"""
                <div class="result-panel {result_class}">
                    <div class="result-label">MODEL CLASSIFICATION</div>
                    <div class="result-value">{result_text}</div>
                    <div class="result-note">{result_note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with score_col:
            st.markdown(
                f"""
                <div class="score-panel">
                    <div class="score-label">FAKE CLASS SCORE</div>
                    <div class="score-value">
                        {fake_probability * 100:.1f}%
                    </div>
                    <div class="score-note">
                        Model-assigned score for the fake class
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="score-track">
                <div class="score-fill"
                     style="width: {bar_width:.2f}%;
                            background: {bar_color};">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="report-explanation">
                <strong>HOW TO READ THIS RESULT</strong><br>
                The score represents the probability assigned by the
                model to the fake class. It is not necessarily the
                actual probability that the posting is fraudulent.
                A lower score means the model assigned less likelihood
                to the fake class.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="risk-warning">
                <strong>IMPORTANT / INDEPENDENT VERIFICATION REQUIRED</strong><br>
                This is a machine learning decision-support prototype,
                not a verification service. Always independently check
                the company's official website, contact information
                and application link before sharing personal information.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =====================================================
# 8. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    <span>J/SHIELD — TRUST INTELLIGENCE</span>
    <span>PYTHON / SCIKIT-LEARN / STREAMLIT</span>
    <span>MODEL OUTPUT IS INDICATIVE</span>
</div>
""", unsafe_allow_html=True)
