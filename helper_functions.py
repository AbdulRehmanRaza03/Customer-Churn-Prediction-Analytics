"""
helper_functions.py
Shared utility functions used across the project.
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#6C63FF",
    "secondary": "#FF6584",
    "success":   "#43D787",
    "warning":   "#FFC107",
    "info":      "#17C3B2",
    "dark":      "#1E1E2E",
    "light":     "#F8F9FA",
    "churn_yes": "#FF6584",
    "churn_no":  "#43D787",
}

CHURN_COLOR_MAP = {"Yes": COLORS["churn_yes"], "No": COLORS["churn_no"]}


def set_plot_theme():
    """Apply consistent dark-minimal theme dict for Plotly."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0", family="Inter, sans-serif"),
        title_font=dict(size=16, color="#FFFFFF"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#333"),
        margin=dict(l=40, r=20, t=50, b=40),
    )


# ── Chart builders ─────────────────────────────────────────────────────────────

def plot_churn_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["Churn"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=[COLORS["churn_yes"], COLORS["churn_no"]],
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(title="Customer Churn Distribution", **set_plot_theme())
    return fig


def plot_tenure_vs_churn(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="tenure", color="Churn",
        color_discrete_map=CHURN_COLOR_MAP,
        barmode="overlay", nbins=40,
        labels={"tenure": "Tenure (months)", "count": "Customers"},
        title="Tenure Distribution by Churn Status",
        opacity=0.75,
    )
    fig.update_layout(**set_plot_theme())
    return fig


def plot_monthly_charges(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x="Churn", y="MonthlyCharges",
        color="Churn",
        color_discrete_map=CHURN_COLOR_MAP,
        labels={"MonthlyCharges": "Monthly Charges ($)"},
        title="Monthly Charges vs Churn",
        points="outliers",
    )
    fig.update_layout(**set_plot_theme())
    return fig


def plot_contract_vs_churn(df: pd.DataFrame) -> go.Figure:
    ct = df.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
    fig = px.bar(
        ct, x="Contract", y="Count", color="Churn",
        color_discrete_map=CHURN_COLOR_MAP,
        barmode="group",
        title="Contract Type vs Churn",
        labels={"Contract": "Contract Type", "Count": "# Customers"},
    )
    fig.update_layout(**set_plot_theme())
    return fig


def plot_internet_vs_churn(df: pd.DataFrame) -> go.Figure:
    ct = df.groupby(["InternetService", "Churn"]).size().reset_index(name="Count")
    fig = px.bar(
        ct, x="InternetService", y="Count", color="Churn",
        color_discrete_map=CHURN_COLOR_MAP,
        barmode="group",
        title="Internet Service Type vs Churn",
    )
    fig.update_layout(**set_plot_theme())
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    # Encode for correlation
    df_enc = df.copy()
    for col in df_enc.select_dtypes("object").columns:
        df_enc[col] = df_enc[col].astype("category").cat.codes

    corr = df_enc.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont_size=8,
        hovertemplate="x=%{x}<br>y=%{y}<br>r=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=600,
        xaxis=dict(tickangle=-45),
        **set_plot_theme()
    )
    return fig


def plot_payment_vs_churn(df: pd.DataFrame) -> go.Figure:
    ct = df.groupby(["PaymentMethod", "Churn"]).size().reset_index(name="Count")
    fig = px.bar(
        ct, x="Count", y="PaymentMethod", color="Churn",
        color_discrete_map=CHURN_COLOR_MAP,
        orientation="h",
        title="Payment Method vs Churn",
    )
    fig.update_layout(**set_plot_theme())
    return fig


def plot_feature_importance(fi_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        fi_df.head(15), x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#17C3B2", "#6C63FF"],
        title="Top 15 Feature Importances",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), **set_plot_theme())
    return fig


def plot_confusion_matrix(cm: list) -> go.Figure:
    labels = ["Not Churn", "Churn"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels, y=labels,
        colorscale=[[0, "#1E1E2E"], [1, COLORS["primary"]]],
        text=cm,
        texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=350,
        **set_plot_theme()
    )
    return fig


# ── Metric helpers ─────────────────────────────────────────────────────────────

def format_metric_card(label: str, value: str, delta: str = "", color: str = "#6C63FF") -> str:
    """Return an HTML string for a KPI card (used in Streamlit markdown)."""
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border: 1px solid {color}55;
        border-radius: 12px; padding: 20px 16px; text-align: center;
        box-shadow: 0 4px 15px {color}22;
    ">
        <div style="font-size: 13px; color: #aaa; letter-spacing: 1px; text-transform: uppercase;">
            {label}
        </div>
        <div style="font-size: 32px; font-weight: 700; color: {color}; margin: 6px 0;">
            {value}
        </div>
        <div style="font-size: 12px; color: #888;">{delta}</div>
    </div>
    """


def get_kpi_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    churned = (df["Churn"] == "Yes").sum()
    churn_rate = churned / total * 100
    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()

    return {
        "total": total,
        "churned": int(churned),
        "retained": int(total - churned),
        "churn_rate": round(churn_rate, 1),
        "retain_rate": round(100 - churn_rate, 1),
        "avg_monthly": round(avg_monthly, 2),
        "avg_tenure": round(avg_tenure, 1),
    }
