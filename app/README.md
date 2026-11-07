# Streamlit Application

JobShield AI provides an interactive web interface for analyzing job postings using the trained machine learning model.

## Features

- Job description, company profile, and requirements input
- Structured job posting feature selection
- Real or fake job posting classification
- Fake-class probability score visualization
- Independent verification guidance

## Run the Application

From the project root directory, activate your virtual environment and run:

```bash
streamlit run app/app.py
```

The application requires the trained model at `models/combined_model.joblib`.

See the main [README](../README.md) for installation instructions and project details.
