# 📡 ChurnSight — Customer Churn Prediction & Analytics Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade ML system to predict customer churn in telecom — featuring a real-time interactive analytics dashboard.**

[Live Demo](#) · [Report Bug](#) · [Request Feature](#)

</div>

---

## 📌 Problem Statement

Customer churn — when a subscriber cancels their service — is one of the **most costly business problems** in the telecom industry. Acquiring a new customer costs **5–7× more** than retaining an existing one.

This project builds an end-to-end machine learning pipeline that:
- Identifies customers at high risk of churning **before** they leave
- Quantifies which features drive churn (contract type, tenure, charges)
- Serves predictions in real time via an interactive dashboard
- Provides actionable retention recommendations per customer

---

## 📊 Dataset

**Source:** IBM Telco Customer Churn Dataset (synthetic version included)

| Property | Value |
|----------|-------|
| Rows | ~7,043 customers |
| Features | 20 (demographics, services, billing) |
| Target | `Churn` (Yes / No) |
| Class balance | ~73% No / 27% Yes |

**Key features:**
- Demographics: Gender, SeniorCitizen, Partner, Dependents
- Services: InternetService, PhoneService, StreamingTV, OnlineSecurity
- Account: Tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML** | scikit-learn (Logistic Regression, Random Forest) |
| **Data** | pandas, numpy |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **Dashboard** | Streamlit |
| **Serialization** | joblib |
| **Language** | Python 3.10+ |

---

## ✨ Features

### 📊 Page 1 — Executive Overview
- KPI cards: total customers, churn rate, retention rate, avg charges, avg tenure
- Churn distribution donut chart
- Contract type breakdown
- Key business insights with data-backed callouts

### 📈 Page 2 — Interactive Data Analysis
- Filter customers by contract type, internet service, tenure range
- Tenure vs churn histogram
- Monthly charges box plot
- Payment method breakdown
- Full feature correlation heatmap

### 🧠 Page 3 — Prediction System
- Form-based real-time churn prediction
- Churn probability score (0–100%)
- Risk level classification (Low / Moderate / High / Very High)
- Human-readable AI explanation of prediction drivers
- Personalised retention recommendations for at-risk customers

### 🔬 Page 4 — Model Insights
- 5 performance metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Top 15 feature importance bar chart
- Confusion matrix heatmap
- Full classification report
- Model architecture details

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Customer-Churn-Project.git
cd Customer-Churn-Project
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (first time only)
```bash
python train.py
```
This generates `model.pkl`, `scaler.pkl`, and `feature_names.pkl` in the root directory.

### 5. Launch the dashboard
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
Customer-Churn-Project/
│
├── app.py                          # Streamlit dashboard (4 pages)
├── train.py                        # Standalone training script
├── model.pkl                       # Trained Random Forest (generated)
├── scaler.pkl                      # StandardScaler (generated)
├── feature_names.pkl               # Feature list (generated)
├── requirements.txt
├── README.md
│
├── data/
│   └── dataset.csv                 # Synthetic Telco dataset (auto-generated)
│
├── notebooks/
│   └── EDA_and_Model_Training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Loading, cleaning, encoding, splitting
│   ├── model_training.py           # Model definitions, training, evaluation
│   └── prediction.py               # Single-customer prediction logic
│
├── utils/
│   ├── __init__.py
│   └── helper_functions.py         # Chart builders, KPI stats, color theme
│
└── visuals/
    └── (saved chart images)
```

---

## 🧠 ML Pipeline

```
Raw CSV
  │
  ▼
data_preprocessing.py
  ├── Drop customerID
  ├── Impute TotalCharges (median)
  ├── LabelEncode binary cols
  └── OneHot encode categorical cols
        │
        ▼
      StandardScaler (numeric cols)
        │
        ▼
     80/20 Train-Test Split (stratified)
        │
        ▼
  model_training.py
  ├── Logistic Regression (baseline)
  └── Random Forest (n=200, balanced weights)
        │
        ▼
  Best model selected by F1 Score
        │
        ▼
  model.pkl + scaler.pkl saved
```

---

## 📈 Model Performance (Sample Results)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | ~79% | ~63% | ~55% | ~59% | ~84% |
| **Random Forest** ✅ | **~82%** | **~68%** | **~62%** | **~65%** | **~87%** |

*Results vary slightly by random seed and dataset sample.*

---

## 🖼️ Screenshots

| Overview Dashboard | Prediction System |
|-------------------|------------------|
| *(screenshot placeholder)* | *(screenshot placeholder)* |

| Data Analysis | Model Insights |
|--------------|---------------|
| *(screenshot placeholder)* | *(screenshot placeholder)* |

---

## 🔮 Future Improvements

- [ ] **XGBoost / LightGBM** — gradient boosting for higher AUC
- [ ] **SHAP values** — per-prediction explainability breakdown
- [ ] **MLflow integration** — experiment tracking and model registry
- [ ] **PostgreSQL backend** — replace CSV with production database
- [ ] **FastAPI REST endpoint** — serve predictions via HTTP API
- [ ] **Docker containerisation** — deploy anywhere with single command
- [ ] **A/B testing module** — compare model versions on live traffic
- [ ] **Email alerts** — notify CRM when high-risk customers detected

---

## 👤 Author

**[Your Name]**
- 🌐 Portfolio: [yourwebsite.com](#)
- 💼 LinkedIn: [linkedin.com/in/yourprofile](#)
- 🐙 GitHub: [github.com/yourusername](#)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
⭐ Star this repo if it helped your learning journey!
</div>
