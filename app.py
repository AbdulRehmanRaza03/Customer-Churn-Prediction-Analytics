"""
app.py
Customer Churn Prediction & Analytics Dashboard
Run: streamlit run app.py
"""

import os
import sys
import warnings
import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import load_data, get_train_test_split, generate_sample_dataset
from src.model_training import (
    train_all_models, save_artifacts, load_artifacts,
    get_feature_importance, evaluate_model, MODEL_PATH
)
from src.prediction import predict_churn
from utils.helper_functions import (
    get_kpi_stats, format_metric_card,
    plot_churn_distribution, plot_tenure_vs_churn,
    plot_monthly_charges, plot_contract_vs_churn,
    plot_correlation_heatmap, plot_internet_vs_churn,
    plot_payment_vs_churn, plot_feature_importance,
    plot_confusion_matrix, COLORS
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight | ML Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0F0F1A; color: #E0E0E0; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%);
        border-right: 1px solid #2A2A4A;
    }

    .block-container { padding-top: 1.5rem; }

    div[data-testid="metric-container"] {
        background: rgba(108, 99, 255, 0.08);
        border: 1px solid rgba(108, 99, 255, 0.25);
        border-radius: 12px;
        padding: 12px 16px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.6rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

    .stSelectbox > div, .stNumberInput > div { border-radius: 8px !important; }

    h1, h2, h3 { color: #FFFFFF !important; }

    .page-header {
        background: linear-gradient(135deg, #6C63FF22, #FF658422);
        border: 1px solid #6C63FF44;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .result-box-churn {
        background: linear-gradient(135deg, #FF658433, #FF658411);
        border: 2px solid #FF6584;
        border-radius: 16px; padding: 28px;
        text-align: center;
    }
    .result-box-stay {
        background: linear-gradient(135deg, #43D78733, #43D78711);
        border: 2px solid #43D787;
        border-radius: 16px; padding: 28px;
        text-align: center;
    }
    .insight-card {
        background: rgba(23, 195, 178, 0.08);
        border-left: 4px solid #17C3B2;
        border-radius: 8px; padding: 14px 18px;
        margin: 8px 0;
    }
    .stTabs [data-baseweb="tab"] { color: #aaa; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #6C63FF !important; border-bottom-color: #6C63FF !important; }
</style>
""", unsafe_allow_html=True)


# ── Data & model loading (cached) ──────────────────────────────────────────────
DATA_PATH = "data/dataset.csv"

@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        generate_sample_dataset(DATA_PATH)
    return load_data(DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_trained_model():
    df = get_data()
    X_tr, X_te, y_tr, y_te, scaler, _, feat = get_train_test_split(df)
    if os.path.exists(MODEL_PATH):
        model, scaler, feat = load_artifacts()
    else:
        model, _, results = train_all_models(X_tr, X_te, y_tr, y_te)
        save_artifacts(model, scaler, feat)
        results_cache = results
    return model, scaler, feat, X_tr, X_te, y_tr, y_te


# ── Sidebar nav ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size: 36px;">📡</div>
        <div style="font-size: 20px; font-weight: 700; color: #6C63FF;">ChurnSight</div>
        <div style="font-size: 11px; color: #888; letter-spacing: 2px;">ML ANALYTICS DASHBOARD</div>
    </div>
    <hr style="border-color: #2A2A4A; margin: 16px 0;">
    """, unsafe_allow_html=True)

    pages = {
        "📊 Overview": "overview",
        "📈 Data Analysis": "analysis",
        "🧠 Prediction": "prediction",
        "🔬 Model Insights": "insights",
    }
    page = st.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
    active = pages[page]

    st.markdown("<hr style='border-color: #2A2A4A; margin-top: 30px;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 11px; color: #555; text-align: center; padding: 10px;">
        Built with Streamlit + scikit-learn<br>
        Customer Churn Prediction v1.0
    </div>
    """, unsafe_allow_html=True)


# ── Load resources ─────────────────────────────────────────────────────────────
with st.spinner("Loading data & model …"):
    df = get_data()
    model, scaler, feat_names, X_tr, X_te, y_tr, y_te = get_trained_model()

kpi = get_kpi_stats(df)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if active == "overview":
    st.markdown("""
    <div class="page-header">
        <h1 style="margin:0; font-size:28px;">📊 Executive Overview</h1>
        <p style="margin:6px 0 0; color:#aaa;">Real-time customer churn intelligence for business leaders</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{kpi['total']:,}")
    c2.metric("Churned",         f"{kpi['churned']:,}",  delta=f"-{kpi['churn_rate']}%", delta_color="inverse")
    c3.metric("Retained",        f"{kpi['retained']:,}", delta=f"+{kpi['retain_rate']}%")
    c4.metric("Avg Monthly $",   f"${kpi['avg_monthly']}")
    c5.metric("Avg Tenure",      f"{kpi['avg_tenure']} mo")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.plotly_chart(plot_churn_distribution(df), use_container_width=True)
    with col_r:
        st.plotly_chart(plot_contract_vs_churn(df), use_container_width=True)

    # Business insights
    st.markdown("### 💡 Key Business Insights")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(f"""<div class="insight-card">
            <b>🚨 Churn Rate:</b> <span style="color:#FF6584; font-size:18px; font-weight:700;">{kpi['churn_rate']}%</span>
            of customers have churned. Industry average is ~15–20%. Requires immediate attention.
        </div>""", unsafe_allow_html=True)
    with i2:
        m2m_churn = df[df["Contract"] == "Month-to-month"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        st.markdown(f"""<div class="insight-card">
            <b>📋 Contract Risk:</b> Month-to-month customers churn at
            <span style="color:#FFC107; font-weight:700;">{m2m_churn:.0f}%</span>
            — push annual contracts via loyalty discounts.
        </div>""", unsafe_allow_html=True)
    with i3:
        fiber_churn = df[df["InternetService"] == "Fiber optic"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        st.markdown(f"""<div class="insight-card">
            <b>📶 Fiber Alert:</b> Fiber optic customers churn at
            <span style="color:#FF6584; font-weight:700;">{fiber_churn:.0f}%</span>.
            Investigate service quality and pricing competitiveness.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DATA ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif active == "analysis":
    st.markdown("""
    <div class="page-header">
        <h1 style="margin:0; font-size:28px;">📈 Interactive Data Analysis</h1>
        <p style="margin:6px 0 0; color:#aaa;">Explore churn patterns across customer segments</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar filters
    with st.sidebar:
        st.markdown("#### 🎛️ Filters")
        contract_filter = st.multiselect(
            "Contract Type",
            df["Contract"].unique().tolist(),
            default=df["Contract"].unique().tolist(),
        )
        internet_filter = st.multiselect(
            "Internet Service",
            df["InternetService"].unique().tolist(),
            default=df["InternetService"].unique().tolist(),
        )
        tenure_range = st.slider("Tenure (months)", 0, 72, (0, 72))

    filtered = df[
        df["Contract"].isin(contract_filter) &
        df["InternetService"].isin(internet_filter) &
        df["tenure"].between(*tenure_range)
    ]

    st.caption(f"Showing **{len(filtered):,}** customers after filters")

    tab1, tab2, tab3, tab4 = st.tabs(["📦 Distribution", "💳 Charges", "🌐 Services", "🔥 Correlation"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_churn_distribution(filtered), use_container_width=True)
        with c2:
            st.plotly_chart(plot_tenure_vs_churn(filtered), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_monthly_charges(filtered), use_container_width=True)
        with c2:
            st.plotly_chart(plot_payment_vs_churn(filtered), use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_contract_vs_churn(filtered), use_container_width=True)
        with c2:
            st.plotly_chart(plot_internet_vs_churn(filtered), use_container_width=True)

    with tab4:
        st.plotly_chart(plot_correlation_heatmap(filtered), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
elif active == "prediction":
    st.markdown("""
    <div class="page-header">
        <h1 style="margin:0; font-size:28px;">🧠 Churn Prediction System</h1>
        <p style="margin:6px 0 0; color:#aaa;">Enter customer details to predict churn probability in real time</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("churn_form"):
        st.markdown("#### 👤 Customer Demographics")
        d1, d2, d3, d4 = st.columns(4)
        gender         = d1.selectbox("Gender",           ["Male", "Female"])
        senior         = d2.selectbox("Senior Citizen",   ["No", "Yes"])
        partner        = d3.selectbox("Partner",          ["No", "Yes"])
        dependents     = d4.selectbox("Dependents",       ["No", "Yes"])

        st.markdown("#### 📋 Account Details")
        a1, a2, a3 = st.columns(3)
        tenure         = a1.number_input("Tenure (months)", 0, 72, 12)
        contract       = a2.selectbox("Contract Type",    ["Month-to-month", "One year", "Two year"])
        paperless      = a3.selectbox("Paperless Billing", ["Yes", "No"])

        st.markdown("#### 💳 Financial Details")
        f1, f2, f3 = st.columns(3)
        monthly        = f1.number_input("Monthly Charges ($)", 10.0, 150.0, 65.0, step=0.5)
        total          = f2.number_input("Total Charges ($)",   0.0, 10000.0, float(monthly * tenure), step=10.0)
        payment        = f3.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

        st.markdown("#### 📡 Services")
        s1, s2, s3 = st.columns(3)
        internet       = s1.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        phone          = s2.selectbox("Phone Service",    ["Yes", "No"])
        multi_lines    = s3.selectbox("Multiple Lines",   ["Yes", "No", "No phone service"])

        s4, s5, s6, s7 = st.columns(4)
        online_sec     = s4.selectbox("Online Security",  ["Yes", "No", "No internet service"])
        online_bk      = s5.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
        device_prot    = s6.selectbox("Device Protection",["Yes", "No", "No internet service"])
        tech_sup       = s7.selectbox("Tech Support",     ["Yes", "No", "No internet service"])

        s8, s9 = st.columns(2)
        stream_tv      = s8.selectbox("Streaming TV",     ["Yes", "No", "No internet service"])
        stream_mv      = s9.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        raw = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bk,
            "DeviceProtection": device_prot, "TechSupport": tech_sup,
            "StreamingTV": stream_tv, "StreamingMovies": stream_mv,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }

        with st.spinner("Analyzing customer profile …"):
            result = predict_churn(raw)

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            css_class = "result-box-churn" if result["churn"] else "result-box-stay"
            label_color = "#FF6584" if result["churn"] else "#43D787"
            st.markdown(f"""
            <div class="{css_class}">
                <div style="font-size: 42px;">&nbsp;</div>
                <div style="font-size: 28px; font-weight: 800; color: {label_color};">
                    {result['label']}
                </div>
                <div style="font-size: 48px; font-weight: 900; color: {label_color}; margin: 12px 0;">
                    {result['probability']}%
                </div>
                <div style="font-size: 14px; color: #ccc;">Churn Probability</div>
                <div style="margin-top: 12px; padding: 8px 20px; background: rgba(255,255,255,0.05);
                            border-radius: 20px; display: inline-block; font-size: 13px; color: #aaa;">
                    Risk Level: <b style="color:{label_color};">{result['confidence']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div class="insight-card">
            🤖 <b>AI Explanation:</b><br>{result['explanation']}
        </div>""", unsafe_allow_html=True)

        # Retention recommendations
        if result["churn"]:
            st.markdown("### 💼 Recommended Retention Actions")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("""<div class="insight-card">
                    🎁 <b>Loyalty Offer</b><br>
                    Offer 20% discount on upgrade to annual contract
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("""<div class="insight-card">
                    📞 <b>Proactive Outreach</b><br>
                    Assign customer success rep to contact within 24h
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown("""<div class="insight-card">
                    🔒 <b>Value Add</b><br>
                    Bundle Online Security + Tech Support at no extra cost
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
elif active == "insights":
    st.markdown("""
    <div class="page-header">
        <h1 style="margin:0; font-size:28px;">🔬 Model Insights & Performance</h1>
        <p style="margin:6px 0 0; color:#aaa;">Understand how the model works and what drives predictions</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    metrics = evaluate_model(model, X_te, y_te)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{metrics['accuracy']*100:.1f}%")
    m2.metric("Precision", f"{metrics['precision']*100:.1f}%")
    m3.metric("Recall",    f"{metrics['recall']*100:.1f}%")
    m4.metric("F1 Score",  f"{metrics['f1']*100:.1f}%")
    m5.metric("ROC-AUC",   f"{metrics['roc_auc']*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        fi_df = get_feature_importance(model, feat_names)
        if not fi_df.empty:
            st.plotly_chart(plot_feature_importance(fi_df), use_container_width=True)

    with col_r:
        cm = metrics["confusion_matrix"]
        st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

        # Interpretation
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        st.markdown(f"""<div class="insight-card" style="margin-top:8px;">
            <b>Matrix Breakdown</b><br>
            ✅ True Negatives: {tn:,}  (correctly predicted stay)<br>
            ✅ True Positives: {tp:,}  (correctly predicted churn)<br>
            ⚠️ False Positives: {fp:,}  (predicted churn, stayed)<br>
            🚨 False Negatives: {fn:,}  (missed churners)
        </div>""", unsafe_allow_html=True)

    # Classification report
    with st.expander("📄 Full Classification Report"):
        st.code(metrics["classification_report"], language="text")

    # Model info
    with st.expander("ℹ️ Model Architecture"):
        model_type = type(model).__name__
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Model Type | `{model_type}` |
        | Training Samples | {len(X_tr):,} |
        | Test Samples | {len(X_te):,} |
        | Feature Count | {len(feat_names)} |
        | Target Variable | Churn (Binary) |
        """)
