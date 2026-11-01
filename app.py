"""SpendPulse | Personal Expense Analytics & Prediction
Interactive Streamlit Dashboard with Exploratory Data Analysis & Machine Learning.
"""

import os
from datetime import datetime
import pandas as pd
import streamlit as st
import requests

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
    page_title="SpendPulse | Personal Expense Analytics & Prediction",
    page_icon="💸",
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


# Custom CSS Styling
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-subtitle {
        color: #10B981;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 2px;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-high {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
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
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("💸 SpendPulse")
    st.caption("Personal Expense Analytics & Prediction")

    # Architecture Connection Badge
    if api_online:
        st.success("🟢 FastAPI Backend: Connected")
    else:
        st.info("⚡ Direct SQLite Mode (FastAPI Offline)")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "🔍 Spending Analytics & EDA",
            "🤖 ML Expense Forecaster",
            "🚨 ML Anomaly Detection",
            "💳 Manage Data & Budgets",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Filter Transactions")

    # Global filter parameters
    all_df = get_transactions_df()
    categories = ["All"] + sorted(all_df["category"].unique().tolist()) if not all_df.empty else ["All"]
    selected_category = st.selectbox("Category", categories, index=0)

    selected_type = st.selectbox("Type", ["All", "Expense", "Income"], index=0)

    st.markdown("---")
    st.subheader("Data Management")
    if st.button("🔄 Reset Demo Dataset"):
        seed_synthetic_data(overwrite=True)
        st.success("Database re-seeded with realistic financial history!")
        st.rerun()

    st.caption("SpendPulse v1.0 • Python & Scikit-Learn")


# Helper to fetch filtered transactions
def load_data():
    cat_filter = None if selected_category == "All" else selected_category
    type_filter = None if selected_type == "All" else selected_type
    df = get_transactions_df(category=cat_filter, trans_type=type_filter)
    budgets = get_budgets_df()
    return df, budgets


df, budgets_df = load_data()


# ==========================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==========================================
if page == "📊 Executive Dashboard":
    st.header("📊 Executive Financial Dashboard")
    st.markdown(
        "Real-time overview of cash flow, category distributions, savings metrics, and budget utilization."
    )

    kpis = get_kpis(df, budgets_df)

    # 4 Metric Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Income</div>
                <div class="metric-value">${kpis['total_income']:,.2f}</div>
                <div class="metric-subtitle">Active timeline earnings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Expenses</div>
                <div class="metric-value" style="color: #EF4444;">${kpis['total_expenses']:,.2f}</div>
                <div class="metric-subtitle" style="color: #64748B;">Avg ${kpis['avg_daily_expense']:,.2f} / day</div>
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
                <div class="metric-value" style="color: {savings_color};">${kpis['net_savings']:,.2f}</div>
                <div class="metric-subtitle" style="color: {savings_color};">Savings Rate: {kpis['savings_rate_pct']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        burn_color = "#EF4444" if kpis["overall_budget_utilization_pct"] > 100 else ("#F59E0B" if kpis["overall_budget_utilization_pct"] > 85 else "#10B981")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Monthly Budget Burn</div>
                <div class="metric-value" style="color: {burn_color};">{kpis['overall_budget_utilization_pct']}%</div>
                <div class="metric-subtitle" style="color: #64748B;">Top: {kpis['top_expense_category']} (${kpis['top_category_amount']:,.0f})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("###")

    # Monthly Trend Interactive Chart
    monthly_df = monthly_spending_summary(df)
    trend_chart = create_monthly_trend_chart(monthly_df)
    st.plotly_chart(trend_chart, use_container_width=True)

    # Two column layout: Budget Utilization + Recent Transactions
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🎯 Current Month Budget Utilization")
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
            display_df["amount"] = display_df["amount"].apply(lambda x: f"${x:,.2f}")
            st.dataframe(
                display_df[["date", "category", "type", "amount", "description", "payment_method"]].head(10),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No transactions match current filters.")


# ==========================================
# PAGE 2: SPENDING ANALYTICS & EDA
# ==========================================
elif page == "🔍 Spending Analytics & EDA":
    st.header("🔍 Exploratory Data Analysis & Spending Analytics")
    st.markdown(
        "Statistical decomposition of spending distributions, temporal heatmaps, and category breakdowns."
    )

    tab_interactive, tab_statistical = st.tabs(["📊 Interactive Visuals", "📈 Statistical EDA (Seaborn & Matplotlib)"])

    with tab_interactive:
        c1, c2 = st.columns([1, 1])
        with c1:
            cat_summary = category_spending_summary(df, budgets_df)
            donut_chart = create_category_pie_chart(cat_summary)
            st.plotly_chart(donut_chart, use_container_width=True)

        with c2:
            st.subheader("Category Expenditure Breakdown")
            if not cat_summary.empty:
                display_cat = cat_summary[["category", "total_amount", "percentage_of_total", "tx_count", "avg_amount"]].copy()
                display_cat.columns = ["Category", "Total Spent ($)", "Share (%)", "Tx Count", "Avg Ticket ($)"]
                display_cat["Total Spent ($)"] = display_cat["Total Spent ($)"].apply(lambda x: f"${x:,.2f}")
                display_cat["Share (%)"] = display_cat["Share (%)"].apply(lambda x: f"{x:.1f}%")
                display_cat["Avg Ticket ($)"] = display_cat["Avg Ticket ($)"].apply(lambda x: f"${x:,.2f}")
                st.dataframe(display_cat, use_container_width=True, hide_index=True)
            else:
                st.info("No expense transactions to display.")

        st.markdown("---")
        st.subheader("Daily Spending Volume & Rolling Moving Averages")
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
                name="7-Day Rolling Avg",
                line=dict(color="#2563EB", width=2.5),
            ))
            daily_fig.add_trace(go.Scatter(
                x=daily_df["date"],
                y=daily_df["rolling_30d_avg"],
                name="30-Day Rolling Avg",
                line=dict(color="#DC2626", width=2, dash="dash"),
            ))
            daily_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(daily_fig, use_container_width=True)

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
                    color_continuous_scale="Blues",
                    labels={"day_name": "Day", "total_amount": "Total ($)"},
                )
                dow_fig.update_layout(template="plotly_white", coloraxis_showscale=False)
                st.plotly_chart(dow_fig, use_container_width=True)

        with col_pm:
            st.subheader("💳 Spending by Payment Method")
            pm_df = payment_method_summary(df)
            if not pm_df.empty:
                pm_fig = px.pie(
                    pm_df,
                    names="payment_method",
                    values="total_amount",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                pm_fig.update_layout(template="plotly_white")
                st.plotly_chart(pm_fig, use_container_width=True)

    with tab_statistical:
        st.markdown(
            "Publication-grade exploratory figures generated via **Matplotlib** and **Seaborn** "
            "for statistical analysis of variance, correlation, and probability densities."
        )

        col_eda_1, col_eda_2 = st.columns(2)
        with col_eda_1:
            st.write("##### 1. Category-wise Boxplot & Outlier Spread")
            fig_box = plot_seaborn_category_boxplots(df)
            st.pyplot(fig_box)

        with col_eda_2:
            st.write("##### 2. Feature Correlation Heatmap")
            fig_corr = plot_seaborn_correlation_heatmap(df)
            st.pyplot(fig_corr)

        st.write("##### 3. Expense Amount Probability Density (Histogram & KDE)")
        fig_density = plot_seaborn_density_distribution(df)
        st.pyplot(fig_density)


# ==========================================
# PAGE 3: ML EXPENSE FORECASTER
# ==========================================
elif page == "🤖 ML Expense Forecaster":
    st.header("🤖 Machine Learning Expense Forecaster")
    st.markdown(
        "Forecasts upcoming monthly spending trajectories using **Scikit-learn** regression "
        "trained on lagged features, seasonality indicators, and rolling expenditure metrics."
    )

    fc_col1, fc_col2, fc_col3 = st.columns([1, 1, 2])
    with fc_col1:
        horizon = st.slider("Forecast Horizon (Months)", min_value=1, max_value=6, value=3)
    with fc_col2:
        model_name = st.selectbox(
            "ML Algorithm",
            ["Ridge Regression (L2)", "Linear Regression (OLS)", "Random Forest Regressor"],
            index=0,
        )
        model_type_code = "random_forest" if "Random Forest" in model_name else ("linear" if "Linear" in model_name else "ridge")

    forecaster = ExpenseForecaster(model_type=model_type_code)
    eval_metrics = forecaster.train_and_evaluate(df)
    forecast_results = forecaster.forecast(df, horizon_months=horizon)

    st.markdown("---")

    # Evaluation Scorecard
    if "mae" in eval_metrics:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Mean Absolute Error (MAE)", f"${eval_metrics['mae']:,.2f}")
        with m2:
            st.metric("Root Mean Squared Error (RMSE)", f"${eval_metrics['rmse']:,.2f}")
        with m3:
            st.metric("R² Goodness of Fit", f"{eval_metrics['r2_score']}")
        with m4:
            st.metric("Training Months", f"{eval_metrics['historical_months_count']} months")

    # Interactive Forecast Plot
    monthly_df = monthly_spending_summary(df)
    forecast_fig = create_forecast_chart(monthly_df, forecast_results)
    st.plotly_chart(forecast_fig, use_container_width=True)

    # Forecasted Projections Table & Category Breakdown
    col_fc_left, col_fc_right = st.columns([1, 1])

    with col_fc_left:
        st.subheader("📅 Projected Month-by-Month Expenses")
        if "forecasts" in forecast_results:
            fc_table = pd.DataFrame(forecast_results["forecasts"])
            fc_table.rename(
                columns={
                    "date": "Projected Month",
                    "predicted_expense": "Forecasted Total ($)",
                    "lower_bound": "90% CI Lower ($)",
                    "upper_bound": "90% CI Upper ($)",
                },
                inplace=True,
            )
            for c in ["Forecasted Total ($)", "90% CI Lower ($)", "90% CI Upper ($)"]:
                fc_table[c] = fc_table[c].apply(lambda x: f"${x:,.2f}")
            st.dataframe(fc_table[["Projected Month", "Forecasted Total ($)", "90% CI Lower ($)", "90% CI Upper ($)"]], use_container_width=True, hide_index=True)

    with col_fc_right:
        st.subheader("🎯 Month 1 Category Allocation Projection")
        projections = forecast_results.get("category_projections_next_month", [])
        if projections:
            proj_df = pd.DataFrame(projections)
            proj_fig = px.bar(
                proj_df,
                x="category",
                y="projected_amount",
                color="projected_amount",
                color_continuous_scale="Teal",
                labels={"category": "Category", "projected_amount": "Projected ($)"},
            )
            proj_fig.update_layout(template="plotly_white", coloraxis_showscale=False)
            st.plotly_chart(proj_fig, use_container_width=True)

    # Feature Importance / Model Coefficients
    if "feature_importances" in eval_metrics and eval_metrics["feature_importances"]:
        st.markdown("#### 🔬 Model Explainability & Feature Weights")
        fi_df = pd.DataFrame(
            list(eval_metrics["feature_importances"].items()),
            columns=["Feature", "Weight / Coefficient"],
        )
        st.dataframe(fi_df, use_container_width=True, hide_index=True)


# ==========================================
# PAGE 4: ML ANOMALY DETECTION
# ==========================================
elif page == "🚨 ML Anomaly Detection":
    st.header("🚨 ML Unusual Transaction Detection")
    st.markdown(
        "Detects suspicious, abnormal, or duplicate transactions using **Scikit-learn's Isolation Forest**. "
        "Evaluates transaction magnitude against category baselines, variance distributions, and calendar timings."
    )

    c_param1, c_param2 = st.columns([1, 2])
    with c_param1:
        contamination = st.slider(
            "Anomaly Contamination Rate",
            min_value=0.01,
            max_value=0.08,
            value=0.03,
            step=0.005,
            help="Expected proportion of outliers in the transaction dataset.",
        )

    # Run Isolation Forest
    detector = TransactionAnomalyDetector(contamination=contamination)
    annotated_df = detector.fit_and_detect(df)
    summary = detector.get_summary_stats(annotated_df)

    # Summary metric cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Transactions Evaluated", f"{summary['total_transactions']}")
    with sc2:
        st.metric("Flagged Anomalies", f"{summary['anomalies_detected']}", delta=f"{summary['anomaly_rate_pct']}% of total", delta_color="inverse")
    with sc3:
        st.metric("Total Flagged Amount", f"${summary['flagged_amount']:,.2f}")
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
        flagged["amount"] = flagged["amount"].apply(lambda x: f"${x:,.2f}")
        flagged["anomaly_score"] = flagged["anomaly_score"].apply(lambda x: f"{x:.1f} / 100")

        st.dataframe(
            flagged[["date", "category", "amount", "risk_level", "anomaly_score", "description", "explanation"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No anomalies detected at the current sensitivity setting.")


# ==========================================
# PAGE 5: MANAGE DATA & BUDGETS
# ==========================================
elif page == "💳 Manage Data & Budgets":
    st.header("💳 Data Management & Budget Configuration")
    st.markdown("Record new transactions, calibrate monthly budget caps, or export clean financial datasets.")

    tab_add, tab_budgets, tab_raw = st.tabs(["➕ Add Transaction", "🎯 Set Monthly Budgets", "📂 Raw Data Explorer"])

    with tab_add:
        st.subheader("Record a Transaction")
        with st.form("add_transaction_form", clear_on_submit=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                t_date = st.date_input("Date", value=datetime.today())
                t_type = st.selectbox("Type", ["expense", "income"])
            with f_col2:
                cat_options = [
                    "Housing & Rent", "Groceries", "Dining Out", "Utilities",
                    "Transportation", "Entertainment", "Shopping", "Healthcare",
                    "Subscriptions", "Travel", "Income", "Freelance", "Other"
                ]
                t_cat = st.selectbox("Category", cat_options)
                t_amt = st.number_input("Amount ($)", min_value=0.01, step=1.0, value=25.0)
            with f_col3:
                t_pm = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Bank Transfer", "Cash", "UPI"])
                t_desc = st.text_input("Description / Memo", placeholder="e.g., Weekly grocery run")

            submit = st.form_submit_button("💾 Save Transaction")
            if submit:
                add_transaction(
                    date=t_date.strftime("%Y-%m-%d"),
                    category=t_cat,
                    trans_type=t_type,
                    amount=t_amt,
                    description=t_desc,
                    payment_method=t_pm,
                )
                st.success(f"Transaction of ${t_amt:,.2f} recorded in {t_cat}!")
                st.rerun()

    with tab_budgets:
        st.subheader("Category Budget Targets")
        b_df = get_budgets_df()
        st.dataframe(b_df, use_container_width=True, hide_index=True)

        with st.form("update_budget_form"):
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                b_cat = st.selectbox("Category to Update", b_df["category"].tolist() if not b_df.empty else ["Groceries"])
            with b_col2:
                b_val = st.number_input("New Monthly Budget Limit ($)", min_value=10.0, step=25.0, value=300.0)

            b_submit = st.form_submit_button("🎯 Update Budget Limit")
            if b_submit:
                set_budget(b_cat, b_val)
                st.success(f"Budget for '{b_cat}' updated to ${b_val:,.2f}/month!")
                st.rerun()

    with tab_raw:
        st.subheader("Historical Transactions Database")
        if not df.empty:
            raw_display = df.copy()
            raw_display["date"] = raw_display["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(raw_display, use_container_width=True, hide_index=True)

            csv = raw_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name="spendpulse_transactions.csv",
                mime="text/csv",
            )
        else:
            st.info("No transaction data available.")
