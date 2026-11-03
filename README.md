# JobShield AI

**An AI-powered machine learning project for detecting potentially fraudulent job postings.**

JobShield AI investigates job posting information using natural language processing and structured features to classify postings as real or potentially fraudulent.

## Project Overview

Online job platforms can contain misleading or fraudulent job advertisements. These postings may expose job seekers to financial risks, identity theft, and misuse of personal information.

The goal of JobShield AI is to explore how machine learning can help identify suspicious job postings by analyzing their textual content and available structured information.

**Research Question**

Does combining structured job posting features with textual information improve the detection of fraudulent job postings?

## Dataset

**Source:** [Real or Fake? Fake Job Posting Prediction — Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

The project uses the `fake_job_postings.csv` dataset.

- Total postings: 17,880
- Real postings: 17,014 (95.16%)
- Fake postings: 866 (4.84%)

The dataset is highly imbalanced, making accuracy alone an insufficient evaluation metric.

Download the dataset from Kaggle and place the CSV file in the data directory expected by the project notebooks.

Raw and processed data files are excluded from Git through `.gitignore`.

Please check the dataset's licensing and usage conditions on Kaggle before redistribution.

## Methodology

### 1. Data Exploration and Preparation

- Explored dataset structure, feature distributions, and missing values.
- Examined the distribution of real and fraudulent postings.
- Investigated repeated job descriptions to reduce the risk of data leakage.
- Used a stratified group split to keep identical job descriptions within the same data partition.

### 2. Feature Engineering

**Textual features**
- Job description
- Company profile
- Job requirements

Text data was transformed using TF-IDF.

**Structured features**
- Employment type
- Required experience
- Required education
- Industry
- Job function
- Telecommuting
- Company logo availability
- Screening questions

### 3. Model Development

The project compares multiple machine learning approaches:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Random Forest
- XGBoost

The main application uses the combined Logistic Regression model, which incorporates textual and structured features.

### 4. Evaluation Strategy

Because fraudulent postings represent a small proportion of the dataset, the project focuses on:

- Precision
- Recall
- F1-score
- Accuracy (as a supplementary metric)

F1-score is particularly useful for evaluating the balance between precision and recall in this imbalanced classification problem.

## Model Results

The following results were obtained on the held-out test set for the primary model comparisons.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Description-only Logistic Regression | 0.9667 | 0.6957 | 0.5549 | 0.6174 |
| Description-only Linear SVM | 0.9746 | 0.9457 | 0.5029 | 0.6566 |
| Three-text-field Logistic Regression | 0.9782 | 0.7914 | 0.7457 | 0.7679 |
| Combined Logistic Regression | 0.9765 | 0.7214 | 0.8382 | **0.7754** |
| Random Forest | 0.9712 | 0.7692 | 0.5780 | 0.6601 |
| XGBoost | 0.9787 | 0.8212 | 0.7168 | 0.7654 |

**Key finding**

The combined Logistic Regression model achieved the highest F1-score (0.7754) and the highest recall (0.8382) among the evaluated models.

This suggests that adding structured features can improve the model's ability to identify fraudulent postings, although the results should be interpreted within the limitations of the dataset and experimental setup.

**Comparison note:** The Random Forest and XGBoost experiments used dimensionality-reduced text features (Truncated SVD), while the Logistic Regression experiments used a different text representation. Therefore, these results should not be interpreted as a perfectly controlled algorithm-only comparison.

## Streamlit Demo

JobShield AI includes an interactive Streamlit application where users can enter job posting information and receive a model prediction.

The application displays:

- Predicted class: Real or Fake
- Model-assigned score for the fake class
- A visual classification report
- An independent verification warning

### Run Locally

1. Clone the repository:

```bash
git clone https://github.com/laraolgun74-creator/JobShield_AI.git
cd JobShield_AI
```

2. Create and activate a virtual environment:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Download the dataset and prepare the project data.

5. Train and save the model using the project notebooks. The Streamlit application expects the trained model at:

```text
models/combined_model.joblib
```

6. Launch the application:

```bash
streamlit run app/app.py
```

## Project Structure

```text
JobShield_AI/
│
├── app/
│   ├── app.py
│   └── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_inspection.py
│   ├── ...
│   └── 16_prepare_demo_examples.py
│
├── reports/
│   ├── model_comparison.csv
│   ├── advanced_model_comparison.csv
│   ├── final_model_comparison.csv
│   └── progress_log.md
│
├── src/
├── tests/
├── requirements.txt
└── README.md
```

Raw data, processed data, and trained model artifacts are excluded from version control by default.

## Limitations

- The dataset is historical and does not necessarily represent current job market patterns.
- The model relies on the information supplied in a job posting and cannot independently verify a company or vacancy.
- Some fraudulent postings may appear legitimate based on their textual content.
- Model predictions are not proof of fraud.
- The model should be treated as a decision-support prototype, not a replacement for independent verification.

Always verify the employer's official website, contact information, and application process before sharing personal or financial information.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Linear SVM
- Random Forest
- XGBoost
- Streamlit
- Git and GitHub

## Project Status

**Completed — Machine Learning Capstone Project**

Developed as part of a Data Science and AI training program.
