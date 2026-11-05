"""SpendPulse India | Smart Personal Finance & AI Expense Analytics.
Tailored for the Indian market with UPI insights, INR (₹) formatting,
interactive EDA, and machine learning forecasting & anomaly detection.
"""

import os
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go

from src.database import (
    init_db,
    seed_synthetic_data,
    get_transactions_df,
    get_budgets_df,
    add_transaction,
    delete_transaction,
    set_budget,
)
from src.analytics import (
    get_kpis,
    get_financial_health_score,
    format_inr,
    monthly_spending_summary,
    category_spending_summary,
    daily_spending_trend,
    day_of_week_analysis,
    payment_method_summary,
    current_month_budget_status,
)
from src.ml.forecaster import ExpenseForecaster
from src.ml.anomaly_detector import TransactionAnomalyDetector
from src.visualizations import (
    create_monthly_trend_chart,
    create_category_pie_chart,
    create_budget_utilization_chart,
    create_forecast_chart,
    create_anomaly_scatter_chart,
    plot_seaborn_category_boxplots,
    plot_seaborn_correlation_heatmap,
    plot_seaborn_density_distribution,
)

# Page configuration
st.set_page_config(
    page_title="SpendPulse India | Smart Khata & AI Expense Analytics",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration with graceful fallback to Direct DB
API_BASE_URL = os.environ.get("SPENDPULSE_API_URL", "http://127.0.0.1:8000")


def check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/api/health", timeout=1.0)
        return resp.status_code == 200
    except Exception:
        return False


# Modern Custom CSS Styling
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 14px;
        transition: transform 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-title {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .metric-value {
        color: #0F172A;
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 5px;
    }
    .metric-subtitle {
        color: #10B981;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 3px;
    }
    /* Health Meter Banner */
    .health-banner {
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
        border: 1px solid #BBF7D0;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .health-title {
        color: #166534;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .health-desc {
        color: #15803D;
        font-size: 0.9rem;
        margin-top: 4px;
    }
    .health-score-badge {
        font-size: 1.7rem;
        font-weight: 800;
        padding: 8px 18px;
        border-radius: 12px;
        background: #FFFFFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    /* Quick Insight Pill */
    .insight-pill {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-high {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ensure database is initialized
init_db()
seed_synthetic_data(overwrite=False)

# Check API status
api_online = check_api_health()

# ==========================================
# SIDEBAR NAVIGATION & FILTERS
# ==========================================
with st.sidebar:
    st.title("🇮🇳 SpendPulse")
    st.caption("AI-Powered Smart Khata & Financial Analytics")

    # Architecture Connection Badge
    if api_online:
        st.success("🟢 FastAPI Service: Connected")
    else:
        st.info("⚡ Direct SQLite Mode (FastAPI Offline)")

    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Financial Pulse & Health",
            "📊 Spending Trends & Habits",
            "🔮 AI Spending Predictor",
            "🛡️ AI Fraud & Anomaly Radar",
            "💰 Khata & Budget Manager",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Filter Transactions")

    # Global filter parameters
    all_df = get_transactions_df()
    categories = ["All"] + sorted(all_df["category"].unique().tolist()) if not all_df.empty else ["All"]
    selected_category = st.selectbox("Category", categories, index=0)

    selected_type = st.selectbox("Transaction Type", ["All", "Expense", "Income"], index=0)

    st.markdown("---")
    st.subheader("Indian Market Demo Data")
    if st.button("🔄 Reset Indian Demo Dataset"):
        seed_synthetic_data(overwrite=True)
        st.success("Database re-seeded with realistic Indian financial data!")
        st.rerun()

    st.caption("SpendPulse v2.0 • Tailored for Indian Consumers • Python & Scikit-Learn")


# Helper to fetch filtered transactions
def load_data():
    cat_filter = None if selected_category == "All" else selected_category
    type_filter = None if selected_type == "All" else selected_type
    df = get_transactions_df(category=cat_filter, trans_type=type_filter)
    budgets = get_budgets_df()
    return df, budgets


df, budgets_df = load_data()


# ==========================================
# PAGE 1: FINANCIAL PULSE & HEALTH
# ==========================================
if page == "🏠 Financial Pulse & Health":
    st.header("🏠 Your Financial Pulse & Health")
    st.markdown("A simple, clear snapshot of your earnings, spending, savings, and financial discipline in Indian Rupees (₹).")

    kpis = get_kpis(df, budgets_df)
    health = get_financial_health_score(kpis)

    # Cool Financial Health Banner
    st.markdown(
        f"""
        <div class="health-banner">
            <div>
                <div class="health-title">{health['badge']}</div>
                <div class="health-desc">💡 {health['advice']}</div>
            </div>
            <div class="health-score-badge" style="color: {health['color']};">
                {health['score']} <span style="font-size: 0.9rem; color: #64748B;">/ 100</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Key Financial Metric Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Income</div>
                <div class="metric-value" style="color: #0F172A;">{format_inr(kpis['total_income'])}</div>
                <div class="metric-subtitle">Salary & consulting credits</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Expenses</div>
                <div class="metric-value" style="color: #EF4444;">{format_inr(kpis['total_expenses'])}</div>
                <div class="metric-subtitle" style="color: #64748B;">Daily burn: {format_inr(kpis['avg_daily_expense'])} / day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        savings_color = "#10B981" if kpis["net_savings"] >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Net Savings</div>
                <div class="metric-value" style="color: {savings_color};">{format_inr(kpis['net_savings'])}</div>
                <div class="metric-subtitle" style="color: {savings_color};">Savings Rate: {kpis['savings_rate_pct']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        burn_color = (
            "#EF4444"
            if kpis["overall_budget_utilization_pct"] > 100
            else ("#F59E0B" if kpis["overall_budget_utilization_pct"] > 85 else "#10B981")
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Monthly Budget Burn</div>
                <div class="metric-value" style="color: {burn_color};">{kpis['overall_budget_utilization_pct']}%</div>
                <div class="metric-subtitle" style="color: #64748B;">Top Spend: {kpis['top_expense_category']} ({format_inr(kpis['top_category_amount'], decimals=0)})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Monthly Cash Flow Interactive Chart
    st.subheader("📈 Monthly Cash Flow (Income vs. Spending vs. Net Savings)")
    monthly_df = monthly_spending_summary(df)
    trend_chart = create_monthly_trend_chart(monthly_df)
    st.plotly_chart(trend_chart, use_container_width=True)

    # Two column layout: Budget Utilization + Recent Transactions
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🎯 Current Month Category Budget Status")
        budget_status_df = current_month_budget_status(df, budgets_df)
        if not budget_status_df.empty:
            budget_chart = create_budget_utilization_chart(budget_status_df)
            st.plotly_chart(budget_chart, use_container_width=True)
        else:
            st.info("No budget data available.")

    with col_right:
        st.subheader("🕒 Recent Transactions")
        if not df.empty:
            display_df = df.copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df["amount"] = display_df["amount"].apply(lambda x: format_inr(x))
            st.dataframe(
                display_df[["date", "category", "type", "amount", "description", "payment_method"]].head(10),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No transactions match current filters.")


# ==========================================
# PAGE 2: SPENDING TRENDS & DESI HABITS
# ==========================================
elif page == "📊 Spending Trends & Habits":
    st.header("📊 Spending Trends & Indian Consumer Habits")
    st.markdown("Understand your daily UPI habits, quick-commerce orders (Blinkit/Zepto), weekend food runs, and category shares.")

    # High-level Indian Market Quick Insights
    expense_df = df[df["type"] == "expense"].copy()
    if not expense_df.empty:
        # Calculate Quick Commerce & Swiggy/Zomato stats
        qcom_mask = expense_df["description"].str.contains("Blinkit|Zepto|Instamart", case=False, na=False)
        dining_mask = expense_df["description"].str.contains("Swiggy|Zomato|Chai|Haldiram|Biryani", case=False, na=False)
        upi_mask = expense_df["payment_method"].str.contains("UPI", case=False, na=False)

        qcom_total = expense_df[qcom_mask]["amount"].sum()
        dining_total = expense_df[dining_mask]["amount"].sum()
        upi_total = expense_df[upi_mask]["amount"].sum()
        total_exp = expense_df["amount"].sum()
        upi_share = (upi_total / total_exp * 100.0) if total_exp > 0 else 0.0

        i1, i2, i3 = st.columns(3)
        with i1:
            st.markdown(
                f"""
                <div class="insight-pill">
                    <span style="font-size: 0.8rem; color: #64748B; font-weight: 700;">⚡ QUICK COMMERCE (BLINKIT/ZEPTO)</span>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; margin-top: 4px;">{format_inr(qcom_total, decimals=0)}</div>
                    <span style="font-size: 0.8rem; color: #3B82F6;">{expense_df[qcom_mask].shape[0]} instant delivery orders</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with i2:
            st.markdown(
                f"""
                <div class="insight-pill">
                    <span style="font-size: 0.8rem; color: #64748B; font-weight: 700;">🍔 FOOD & CHAI (SWIGGY/ZOMATO)</span>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; margin-top: 4px;">{format_inr(dining_total, decimals=0)}</div>
                    <span style="font-size: 0.8rem; color: #F59E0B;">{expense_df[dining_mask].shape[0]} restaurant & cafe treats</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with i3:
            st.markdown(
                f"""
                <div class="insight-pill">
                    <span style="font-size: 0.8rem; color: #64748B; font-weight: 700;">📱 UPI SHARE (GPAY/PHONEPE/PAYTM)</span>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #10B981; margin-top: 4px;">{upi_share:.1f}%</div>
                    <span style="font-size: 0.8rem; color: #10B981;">{format_inr(upi_total, decimals=0)} paid via UPI</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    tab_overview, tab_patterns, tab_statistical = st.tabs([
        "🍰 Category Breakdown",
        "📅 Weekly & Payment Habits",
        "📈 Statistical Deep-Dive (EDA)",
    ])

    with tab_overview:
        c1, c2 = st.columns([1, 1])
        with c1:
            cat_summary = category_spending_summary(df, budgets_df)
            donut_chart = create_category_pie_chart(cat_summary)
            st.plotly_chart(donut_chart, use_container_width=True)

        with c2:
            st.subheader("Category Expenditure Table")
            if not cat_summary.empty:
                display_cat = cat_summary[["category", "total_amount", "percentage_of_total", "tx_count", "avg_amount"]].copy()
                display_cat.columns = ["Category", "Total Spent", "Share (%)", "Transactions", "Avg Ticket"]
                display_cat["Total Spent"] = display_cat["Total Spent"].apply(lambda x: format_inr(x))
                display_cat["Share (%)"] = display_cat["Share (%)"].apply(lambda x: f"{x:.1f}%")
                display_cat["Avg Ticket"] = display_cat["Avg Ticket"].apply(lambda x: format_inr(x))
                st.dataframe(display_cat, use_container_width=True, hide_index=True)
            else:
                st.info("No expense transactions to display.")

        st.markdown("---")
        st.subheader("Daily Spending Activity & 7-Day / 30-Day Moving Averages")
        daily_df = daily_spending_trend(df)
        if not daily_df.empty:
            daily_fig = go.Figure()
            daily_fig.add_trace(go.Bar(
                x=daily_df["date"],
                y=daily_df["amount"],
                name="Daily Spend",
                marker_color="#93C5FD",
                opacity=0.6,
            ))
            daily_fig.add_trace(go.Scatter(
                x=daily_df["date"],
                y=daily_df["rolling_7d_avg"],
                name="7-Day Moving Avg",
                line=dict(color="#2563EB", width=2.5),
            ))
            daily_fig.add_trace(go.Scatter(
                x=daily_df["date"],
                y=daily_df["rolling_30d_avg"],
                name="30-Day Moving Avg",
                line=dict(color="#DC2626", width=2, dash="dash"),
            ))
            daily_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Amount (₹)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(daily_fig, use_container_width=True)

    with tab_patterns:
        col_dow, col_pm = st.columns(2)
        with col_dow:
            st.subheader("📅 Spending by Day of Week")
            dow_df = day_of_week_analysis(df)
            if not dow_df.empty:
                dow_fig = px.bar(
                    dow_df,
                    x="day_name",
                    y="total_amount",
                    color="total_amount",
                    color_continuous_scale="Teal",
                    labels={"day_name": "Day", "total_amount": "Total (₹)"},
                )
                dow_fig.update_layout(template="plotly_white", coloraxis_showscale=False)
                st.plotly_chart(dow_fig, use_container_width=True)

        with col_pm:
            st.subheader("💳 Preferred Payment Methods (UPI vs. Cards)")
            pm_df = payment_method_summary(df)
            if not pm_df.empty:
                pm_fig = px.pie(
                    pm_df,
                    names="payment_method",
                    values="total_amount",
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                pm_fig.update_layout(template="plotly_white")
                st.plotly_chart(pm_fig, use_container_width=True)

    with tab_statistical:
        st.markdown("Visual statistical analysis of spending variance and distributions using **Matplotlib** and **Seaborn**.")

        col_eda_1, col_eda_2 = st.columns(2)
        with col_eda_1:
            st.write("##### 1. Category-wise Spread & Outlier Boxplot")
            fig_box = plot_seaborn_category_boxplots(df)
            st.pyplot(fig_box)

        with col_eda_2:
            st.write("##### 2. Calendar Timing Correlation Heatmap")
            fig_corr = plot_seaborn_correlation_heatmap(df)
            st.pyplot(fig_corr)

        st.write("##### 3. Expense Amount Probability Density (Histogram & KDE)")
        fig_density = plot_seaborn_density_distribution(df)
        st.pyplot(fig_density)


# ==========================================
# PAGE 3: AI SPENDING PREDICTOR
# ==========================================
elif page == "🔮 AI Spending Predictor":
    st.header("🔮 AI Spending Predictor")
    st.markdown(
        "Forecasts how much you are likely to spend over the next few months using **Machine Learning** "
        "(trained on past bills, festival seasonality, and spending trends)."
    )

    fc_col1, fc_col2 = st.columns([1, 1])
    with fc_col1:
        horizon = st.slider("Forecast Horizon (Months Ahead)", min_value=1, max_value=6, value=3)
    with fc_col2:
        model_name = st.selectbox(
            "AI Algorithm",
            ["Ridge Regression (Recommended)", "Linear Regression (OLS)", "Random Forest Regressor"],
            index=0,
        )
        model_type_code = (
            "random_forest"
            if "Random Forest" in model_name
            else ("linear" if "Linear" in model_name else "ridge")
        )

    forecaster = ExpenseForecaster(model_type=model_type_code)
    eval_metrics = forecaster.train_and_evaluate(df)
    forecast_results = forecaster.forecast(df, horizon_months=horizon)

    # Plain-English Accuracy Explainer
    if "mae" in eval_metrics:
        mae_val = eval_metrics["mae"]
        st.info(
            f"🎯 **Model Accuracy Explained**: On average, our AI forecast is accurate within **±{format_inr(mae_val, decimals=0)}** "
            f"of your actual monthly bills (R² Fit: {eval_metrics['r2_score']} across {eval_metrics['historical_months_count']} historical months)."
        )

    # Interactive Forecast Plot
    monthly_df = monthly_spending_summary(df)
    forecast_fig = create_forecast_chart(monthly_df, forecast_results)
    st.plotly_chart(forecast_fig, use_container_width=True)

    # Forecasted Projections Table & Category Breakdown
    col_fc_left, col_fc_right = st.columns([1, 1])

    with col_fc_left:
        st.subheader("📅 Projected Monthly Spending")
        fc_list = forecast_results.get("forecasts", [])
        if fc_list:
            display_fc = pd.DataFrame(fc_list)
            display_fc.columns = ["Month", "Projected Total", "Lower Range (Safe)", "Upper Range (Max)"]
            display_fc["Projected Total"] = display_fc["Projected Total"].apply(lambda x: format_inr(x))
            display_fc["Lower Range (Safe)"] = display_fc["Lower Range (Safe)"].apply(lambda x: format_inr(x))
            display_fc["Upper Range (Max)"] = display_fc["Upper Range (Max)"].apply(lambda x: format_inr(x))
            st.dataframe(display_fc, use_container_width=True, hide_index=True)

    with col_fc_right:
        st.subheader("📊 Next Month Expected Category Share")
        projections = forecast_results.get("category_projections_next_month", [])
        if projections:
            proj_df = pd.DataFrame(projections)
            proj_fig = px.bar(
                proj_df,
                x="category",
                y="projected_amount",
                color="projected_amount",
                color_continuous_scale="Teal",
                labels={"category": "Category", "projected_amount": "Projected (₹)"},
            )
            proj_fig.update_layout(template="plotly_white", coloraxis_showscale=False)
            st.plotly_chart(proj_fig, use_container_width=True)


# ==========================================
# PAGE 4: AI FRAUD & ANOMALY RADAR
# ==========================================
elif page == "🛡️ AI Fraud & Anomaly Radar":
    st.header("🛡️ AI Fraud & Unusual Transaction Radar")
    st.markdown(
        "Detects accidental double charges, suspicious UPI payments, and abnormal purchase spikes "
        "using **Scikit-learn's Isolation Forest**. The AI automatically explains *why* each item was flagged."
    )

    c_param1, _ = st.columns([1, 2])
    with c_param1:
        contamination = st.slider(
            "Anomaly Detection Sensitivity",
            min_value=0.01,
            max_value=0.08,
            value=0.03,
            step=0.005,
            help="Higher sensitivity flags more subtle outliers; lower flags only extreme spikes.",
        )

    # Run Isolation Forest
    detector = TransactionAnomalyDetector(contamination=contamination)
    annotated_df = detector.fit_and_detect(df)
    summary = detector.get_summary_stats(annotated_df)

    # Summary metric cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Total Transactions Audited", f"{summary['total_transactions']}")
    with sc2:
        st.metric(
            "Flagged Anomalies",
            f"{summary['anomalies_detected']}",
            delta=f"{summary['anomaly_rate_pct']}% of total",
            delta_color="inverse",
        )
    with sc3:
        st.metric("Total Flagged Value", format_inr(summary["flagged_amount"]))
    with sc4:
        st.metric("Critical Risk Items", f"{summary['critical_risk_count']}")

    st.markdown("---")

    # Interactive Outlier Scatter Plot
    scatter_fig = create_anomaly_scatter_chart(annotated_df)
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Flagged Anomalies Detail Table
    st.subheader("🚩 Suspicious Transactions Requiring Review")
    flagged = annotated_df[annotated_df["is_anomaly"]].copy()

    if not flagged.empty:
        flagged["date"] = flagged["date"].dt.strftime("%Y-%m-%d")
        flagged["formatted_amount"] = flagged["amount"].apply(lambda x: format_inr(x))

        for _, row in flagged.iterrows():
            badge_class = (
                "badge-critical"
                if row["risk_level"] == "Critical"
                else ("badge-high" if row["risk_level"] == "High" else "badge-normal")
            )
            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="{badge_class}">{row['risk_level'].upper()} RISK ({row['anomaly_score']}/100)</span>
                            <strong style="margin-left: 10px; font-size: 1.05rem; color: #0F172A;">{row['description']}</strong>
                            <span style="color: #64748B; margin-left: 8px;">({row['category']})</span>
                        </div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #DC2626;">
                            {row['formatted_amount']}
                        </div>
                    </div>
                    <div style="margin-top: 6px; font-size: 0.85rem; color: #475569;">
                        📅 <strong>Date:</strong> {row['date']} | 💳 <strong>Payment:</strong> {row['payment_method']} | 💡 <strong>AI Explanation:</strong> {row['explanation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("🎉 No suspicious or anomalous transactions detected at the current sensitivity setting.")


# ==========================================
# PAGE 5: KHATA & BUDGET MANAGER
# ==========================================
elif page == "💰 Khata & Budget Manager":
    st.header("💰 Khata & Budget Manager")
    st.markdown("Easily record everyday UPI payments, update monthly category limits, or export your financial ledger.")

    tab_add, tab_budgets, tab_raw = st.tabs(["➕ Record Transaction", "🎯 Manage Monthly Budgets", "📂 Raw Ledger Data"])

    with tab_add:
        st.subheader("Add a New Transaction (UPI / Card / Cash)")
        with st.form("add_transaction_form", clear_on_submit=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                t_date = st.date_input("Date", value=datetime.today())
                t_type = st.selectbox("Type", ["expense", "income"])
            with f_col2:
                cat_options = [
                    "Rent & Housing",
                    "Groceries & Kirana",
                    "Food & Dining",
                    "Utilities & Bills",
                    "Travel & Commute",
                    "Shopping & E-Com",
                    "Entertainment & OTT",
                    "Healthcare & Fitness",
                    "Subscriptions",
                    "Investments & SIP",
                    "Income",
                    "Other",
                ]
                t_cat = st.selectbox("Category", cat_options)
                t_amt = st.number_input("Amount (₹)", min_value=1.0, step=50.0, value=250.0)
            with f_col3:
                t_pm = st.selectbox(
                    "Payment Method",
                    ["UPI (GPay / PhonePe / Paytm)", "Credit Card", "Debit Card", "Net Banking", "Cash"],
                )
                t_desc = st.text_input("Description / Notes", placeholder="e.g., Blinkit groceries, Swiggy dinner, Metro recharge")

            submit = st.form_submit_button("💾 Save to Khata")
            if submit:
                add_transaction(
                    date=t_date.strftime("%Y-%m-%d"),
                    category=t_cat,
                    trans_type=t_type,
                    amount=t_amt,
                    description=t_desc,
                    payment_method=t_pm,
                )
                st.success(f"Saved {format_inr(t_amt)} under '{t_cat}'!")
                st.rerun()

    with tab_budgets:
        st.subheader("Category Monthly Budgets")
        b_df = get_budgets_df()
        if not b_df.empty:
            b_display = b_df.copy()
            b_display["Monthly Budget (₹)"] = b_display["monthly_budget"].apply(lambda x: format_inr(x, decimals=0))
            st.dataframe(b_display[["category", "Monthly Budget (₹)"]], use_container_width=True, hide_index=True)

        with st.form("update_budget_form"):
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                b_cat = st.selectbox(
                    "Select Category to Calibrate",
                    b_df["category"].tolist() if not b_df.empty else ["Groceries & Kirana"],
                )
            with b_col2:
                b_val = st.number_input("New Monthly Budget Limit (₹)", min_value=500.0, step=500.0, value=10000.0)

            b_submit = st.form_submit_button("🎯 Update Budget Limit")
            if b_submit:
                set_budget(b_cat, b_val)
                st.success(f"Budget for '{b_cat}' updated to {format_inr(b_val, decimals=0)}/month!")
                st.rerun()

    with tab_raw:
        st.subheader("Financial Ledger (All Historical Records)")
        if not df.empty:
            raw_display = df.copy()
            raw_display["date"] = raw_display["date"].dt.strftime("%Y-%m-%d")
            raw_display["formatted_amount"] = raw_display["amount"].apply(lambda x: format_inr(x))
            st.dataframe(
                raw_display[["date", "category", "type", "formatted_amount", "description", "payment_method"]],
                use_container_width=True,
                hide_index=True,
            )

            csv = raw_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Khata as CSV",
                data=csv,
                file_name="spendpulse_india_transactions.csv",
                mime="text/csv",
            )
        else:
            st.info("No transaction records available.")
