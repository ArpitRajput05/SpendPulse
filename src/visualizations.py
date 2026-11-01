"""Visualizations module for SpendPulse.
Provides interactive Plotly charts for web dashboards as well as
publication-ready Matplotlib and Seaborn figures for statistical EDA reports.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for web servers
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. INTERACTIVE PLOTLY CHARTS (Streamlit UI)
# ==========================================

PLOTLY_TEMPLATE = "plotly_white"
COLOR_INCOME = "#10B981"    # Emerald Green
COLOR_EXPENSE = "#EF4444"   # Crimson Red
COLOR_SAVINGS = "#3B82F6"   # Royal Blue
COLOR_ANOMALY = "#F59E0B"   # Amber Warning
COLOR_CRITICAL = "#DC2626"  # Bright Red


def create_monthly_trend_chart(monthly_df: pd.DataFrame) -> go.Figure:
    """Creates a dual-axis interactive chart comparing monthly Income, Expense, and Net Savings."""
    fig = go.Figure()

    if monthly_df.empty:
        fig.update_layout(title="No Monthly Data Available")
        return fig

    # Bar: Income
    fig.add_trace(go.Bar(
        x=monthly_df["year_month"],
        y=monthly_df["income"],
        name="Income",
        marker_color=COLOR_INCOME,
        opacity=0.85,
    ))

    # Bar: Expenses
    fig.add_trace(go.Bar(
        x=monthly_df["year_month"],
        y=monthly_df["expense"],
        name="Expenses",
        marker_color=COLOR_EXPENSE,
        opacity=0.85,
    ))

    # Line: Net Savings
    fig.add_trace(go.Scatter(
        x=monthly_df["year_month"],
        y=monthly_df["net_savings"],
        name="Net Savings",
        mode="lines+markers",
        line=dict(color=COLOR_SAVINGS, width=3),
        marker=dict(size=7),
    ))

    # 3-Month Moving Average
    if "expense_3m_avg" in monthly_df.columns:
        fig.add_trace(go.Scatter(
            x=monthly_df["year_month"],
            y=monthly_df["expense_3m_avg"],
            name="3-Month Expense Trend",
            mode="lines",
            line=dict(color="#6B7280", width=2, dash="dot"),
        ))

    fig.update_layout(
        title="Monthly Cash Flow: Income vs. Expenses & Net Savings",
        barmode="group",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def create_category_pie_chart(cat_df: pd.DataFrame) -> go.Figure:
    """Creates a clean interactive Donut chart showing category breakdown."""
    if cat_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Category Data Available")
        return fig

    fig = px.pie(
        cat_df,
        names="category",
        values="total_amount",
        title="Spending by Category",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        hovertemplate="<b>%{label}</b><br>Spent: $%{value:,.2f}<br>Share: %{percent}<extra></extra>",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.1),
    )
    return fig


def create_budget_utilization_chart(budget_status_df: pd.DataFrame) -> go.Figure:
    """Creates a horizontal comparative bar chart for Budget vs Actual Spent."""
    if budget_status_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Budget Data Available")
        return fig

    sorted_df = budget_status_df.sort_values("actual_spent", ascending=True)

    fig = go.Figure()

    # Budget Target Bar
    fig.add_trace(go.Bar(
        y=sorted_df["category"],
        x=sorted_df["monthly_budget"],
        name="Budget Limit",
        orientation="h",
        marker_color="#CBD5E1",
        opacity=0.6,
    ))

    # Actual Spent Bar (Color coded by status)
    colors = [
        COLOR_CRITICAL if s == "Over Budget" else (COLOR_ANOMALY if s == "Near Limit" else COLOR_INCOME)
        for s in sorted_df["status"]
    ]

    fig.add_trace(go.Bar(
        y=sorted_df["category"],
        x=sorted_df["actual_spent"],
        name="Actual Spent",
        orientation="h",
        marker_color=colors,
        text=[f"${x:,.0f} ({pct:.0f}%)" for x, pct in zip(sorted_df["actual_spent"], sorted_df["utilization_pct"])],
        textposition="auto",
    ))

    fig.update_layout(
        title="Budget Utilization by Category (Spent vs. Limit)",
        barmode="overlay",
        xaxis_title="Amount ($)",
        yaxis_title="Category",
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def create_forecast_chart(historical_monthly: pd.DataFrame, forecast_results: Dict[str, Any]) -> go.Figure:
    """Plots historical expenses alongside ML forecasted future trajectory and confidence bounds."""
    fig = go.Figure()

    if not historical_monthly.empty:
        fig.add_trace(go.Scatter(
            x=historical_monthly["year_month"],
            y=historical_monthly["expense"],
            name="Historical Actuals",
            mode="lines+markers",
            line=dict(color="#1F2937", width=3),
            marker=dict(size=6),
        ))

    forecasts = forecast_results.get("forecasts", [])
    if forecasts:
        f_dates = [f["date"] for f in forecasts]
        f_preds = [f["predicted_expense"] for f in forecasts]
        f_lowers = [f["lower_bound"] for f in forecasts]
        f_uppers = [f["upper_bound"] for f in forecasts]

        # Connect last historical point to first forecast
        if not historical_monthly.empty:
            last_hist_date = historical_monthly["year_month"].iloc[-1]
            last_hist_val = historical_monthly["expense"].iloc[-1]
            line_dates = [last_hist_date] + f_dates
            line_preds = [last_hist_val] + f_preds
            ci_dates = [last_hist_date] + f_dates
            ci_lowers = [last_hist_val] + f_lowers
            ci_uppers = [last_hist_val] + f_uppers
        else:
            line_dates, line_preds = f_dates, f_preds
            ci_dates, ci_lowers, ci_uppers = f_dates, f_lowers, f_uppers

        # Upper bound bound (for fill)
        fig.add_trace(go.Scatter(
            x=ci_dates,
            y=ci_uppers,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=ci_dates,
            y=ci_lowers,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.18)",
            name="90% Confidence Interval",
        ))

        # Forecast Line
        fig.add_trace(go.Scatter(
            x=line_dates,
            y=line_preds,
            name="ML Projected Expense",
            mode="lines+markers",
            line=dict(color="#2563EB", width=3, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ))

    fig.update_layout(
        title="Machine Learning Expense Forecast (Next Horizon)",
        xaxis_title="Month",
        yaxis_title="Expense Amount ($)",
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def create_anomaly_scatter_chart(anomalies_df: pd.DataFrame) -> go.Figure:
    """Generates an interactive scatter chart highlighting unusual transactions detected by IsolationForest."""
    fig = go.Figure()

    if anomalies_df.empty:
        fig.update_layout(title="No Anomaly Data Available")
        return fig

    anomalies_df_copy = anomalies_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(anomalies_df_copy["date"]):
        anomalies_df_copy["date"] = pd.to_datetime(anomalies_df_copy["date"])

    normal = anomalies_df_copy[~anomalies_df_copy["is_anomaly"]]
    flagged = anomalies_df_copy[anomalies_df_copy["is_anomaly"]]

    # Normal Transactions
    fig.add_trace(go.Scatter(
        x=normal["date"],
        y=normal["amount"],
        mode="markers",
        name="Normal Transaction",
        marker=dict(color="#94A3B8", size=6, opacity=0.6),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Amount:</b> $%{y:,.2f}<br><b>Desc:</b> " + normal["description"] + "<extra></extra>",
    ))

    # Flagged Anomalies
    if not flagged.empty:
        fig.add_trace(go.Scatter(
            x=flagged["date"],
            y=flagged["amount"],
            mode="markers+text",
            name="Flagged Anomaly (ML)",
            marker=dict(
                color=COLOR_CRITICAL,
                size=12,
                symbol="triangle-up",
                line=dict(color="#7F1D1D", width=1.5),
            ),
            text=flagged["category"],
            textposition="top center",
            hovertemplate=(
                "🚨 <b>SUSPICIOUS TRANSACTION</b><br>"
                "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                "<b>Amount:</b> $%{y:,.2f}<br>"
                "<b>Category:</b> " + flagged["category"] + "<br>"
                "<b>Risk Score:</b> " + flagged["anomaly_score"].astype(str) + "/100<br>"
                "<b>Reason:</b> " + flagged["explanation"] + "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Transaction Outlier Detection (Isolation Forest)",
        xaxis_title="Transaction Date",
        yaxis_title="Amount ($)",
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ==========================================
# 2. SEABORN & MATPLOTLIB CHARTS (Deep-Dive EDA)
# ==========================================

def plot_seaborn_category_boxplots(df: pd.DataFrame) -> plt.Figure:
    """Generates a Seaborn boxplot illustrating spending distributions and outlier spreads per category."""
    expense_df = df[df["type"] == "expense"].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.set_theme(style="whitegrid")

    sns.boxplot(
        data=expense_df,
        x="category",
        y="amount",
        palette="Spectral",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": "5"},
        ax=ax,
    )
    ax.set_title("Category-wise Spending Distributions & Outliers (Boxplot)", fontsize=13, weight="bold")
    ax.set_xlabel("Expense Category", fontsize=11)
    ax.set_ylabel("Amount ($)", fontsize=11)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    return fig


def plot_seaborn_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Generates a Seaborn correlation matrix between numerical expenditure and temporal features."""
    expense_df = df[df["type"] == "expense"].copy()
    if not pd.api.types.is_datetime64_any_dtype(expense_df["date"]):
        expense_df["date"] = pd.to_datetime(expense_df["date"])

    expense_df["day"] = expense_df["date"].dt.day
    expense_df["month"] = expense_df["date"].dt.month
    expense_df["day_of_week"] = expense_df["date"].dt.day_of_week
    expense_df["is_weekend"] = (expense_df["day_of_week"] >= 5).astype(int)

    num_cols = ["amount", "day", "month", "day_of_week", "is_weekend"]
    corr = expense_df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=0.5,
        linewidths=1,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Temporal Attributes vs. Spend", fontsize=12, weight="bold")
    plt.tight_layout()
    return fig


def plot_seaborn_density_distribution(df: pd.DataFrame) -> plt.Figure:
    """Generates a Kernel Density Estimation (KDE) and histogram of transaction amounts."""
    expense_df = df[df["type"] == "expense"].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(
        expense_df["amount"],
        kde=True,
        color="#2563EB",
        bins=35,
        ax=ax,
    )
    ax.set_title("Expense Amount Distribution (Histogram & KDE)", fontsize=13, weight="bold")
    ax.set_xlabel("Transaction Amount ($)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    return fig
