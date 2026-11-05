"""Exploratory Data Analysis (EDA) and Financial Analytics module.
Leverages Pandas and NumPy to compute KPIs, trend aggregates, and budget metrics.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures that the 'date' column is in datetime format."""
    df_copy = df.copy()
    if not df_copy.empty and "date" in df_copy.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_copy["date"]):
            df_copy["date"] = pd.to_datetime(df_copy["date"])
    return df_copy


def format_inr(number: float, include_symbol: bool = True, decimals: int = 2) -> str:
    """Formats number in standard Indian Numbering system (Lakhs & Crores).
    E.g. 150000.5 -> '₹1,50,000.50'
    """
    if pd.isna(number):
        return "₹0.00" if include_symbol else "0.00"
    prefix = "₹" if include_symbol else ""
    sign = "-" if number < 0 else ""
    number = abs(float(number))
    if decimals > 0:
        formatted_dec = f"{number:.{decimals}f}"
        s, d = formatted_dec.split(".")
        dec_part = "." + d
    else:
        s = f"{int(round(number))}"
        dec_part = ""

    if len(s) <= 3:
        res = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        res = ",".join(groups) + "," + last3
    return f"{sign}{prefix}{res}{dec_part}"


def get_financial_health_score(kpis: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates a plain-English 0-100 Financial Health Score tailored for Indian consumers."""
    savings_rate = kpis.get("savings_rate_pct", 0.0)
    burn_pct = kpis.get("overall_budget_utilization_pct", 100.0)

    # 1. Savings score (0 - 50 points)
    if savings_rate >= 35.0:
        savings_score = 50.0
    elif savings_rate >= 20.0:
        savings_score = 35.0 + (savings_rate - 20.0) * 1.0
    elif savings_rate >= 0.0:
        savings_score = (savings_rate / 20.0) * 35.0
    else:
        savings_score = 0.0

    # 2. Budget burn control (0 - 50 points)
    if burn_pct <= 75.0:
        burn_score = 50.0
    elif burn_pct <= 95.0:
        burn_score = 50.0 - (burn_pct - 75.0) * 1.5
    elif burn_pct <= 110.0:
        burn_score = 20.0 - (burn_pct - 95.0) * 1.0
    else:
        burn_score = 5.0

    total_score = int(round(min(100.0, max(0.0, savings_score + burn_score))))

    if total_score >= 80:
        status = "Excellent"
        color = "#10B981"
        badge = "🌟 Solid Financial Health"
        advice = "Fantastic financial discipline! Your savings rate and monthly budget burn are in great shape."
    elif total_score >= 60:
        status = "Good"
        color = "#3B82F6"
        badge = "👍 Healthy & On Track"
        advice = "Good financial stability. Keep an eye on dining & quick commerce orders to boost your savings rate further."
    elif total_score >= 45:
        status = "Moderate"
        color = "#F59E0B"
        badge = "⚠️ Needs Minor Attention"
        advice = "Your monthly spend is catching up to your income. Consider curbing impulse shopping and dining out."
    else:
        status = "Critical"
        color = "#EF4444"
        badge = "🚨 High Burn Rate"
        advice = "Your expenses are exceeding your budget. Review recurring subscriptions and large discretionary purchases."

    return {
        "score": total_score,
        "status": status,
        "color": color,
        "badge": badge,
        "advice": advice,
    }


def get_kpis(df: pd.DataFrame, budgets_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Calculates executive financial summary KPIs."""
    if df.empty:
        return {
            "total_income": 0.0,
            "total_expenses": 0.0,
            "net_savings": 0.0,
            "savings_rate_pct": 0.0,
            "avg_daily_expense": 0.0,
            "total_transactions": 0,
            "top_expense_category": "N/A",
            "top_category_amount": 0.0,
            "overall_budget_utilization_pct": 0.0,
        }

    df = ensure_datetime(df)

    income_df = df[df["type"] == "income"]
    expense_df = df[df["type"] == "expense"]

    total_income = float(income_df["amount"].sum())
    total_expenses = float(expense_df["amount"].sum())
    net_savings = total_income - total_expenses
    savings_rate_pct = (net_savings / total_income * 100.0) if total_income > 0 else 0.0

    # Number of days active in dataframe
    min_date = df["date"].min()
    max_date = df["date"].max()
    num_days = max(1, (max_date - min_date).days + 1)
    avg_daily_expense = total_expenses / num_days

    # Top category
    if not expense_df.empty:
        cat_sums = expense_df.groupby("category")["amount"].sum()
        top_cat = cat_sums.idxmax()
        top_cat_amt = float(cat_sums.max())
    else:
        top_cat = "N/A"
        top_cat_amt = 0.0

    # Overall budget utilization (last 30 days or active month)
    budget_util_pct = 0.0
    if budgets_df is not None and not budgets_df.empty and not expense_df.empty:
        total_monthly_budget = budgets_df["monthly_budget"].sum()
        # Look at the most recent month for budget comparison
        latest_month_str = df["date"].dt.to_period("M").max()
        curr_month_expenses = expense_df[expense_df["date"].dt.to_period("M") == latest_month_str]["amount"].sum()
        if total_monthly_budget > 0:
            budget_util_pct = (curr_month_expenses / total_monthly_budget) * 100.0

    return {
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "net_savings": round(net_savings, 2),
        "savings_rate_pct": round(savings_rate_pct, 1),
        "avg_daily_expense": round(avg_daily_expense, 2),
        "total_transactions": int(len(df)),
        "top_expense_category": top_cat,
        "top_category_amount": round(top_cat_amt, 2),
        "overall_budget_utilization_pct": round(budget_util_pct, 1),
    }


def monthly_spending_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates income, expense, and savings on a monthly level."""
    if df.empty:
        return pd.DataFrame()

    df = ensure_datetime(df)
    df["year_month"] = df["date"].dt.strftime("%Y-%m")

    # Pivot table with year_month as rows and type ('income', 'expense') as columns
    monthly = df.pivot_table(
        index="year_month",
        columns="type",
        values="amount",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()

    for col in ["income", "expense"]:
        if col not in monthly.columns:
            monthly[col] = 0.0

    monthly["net_savings"] = monthly["income"] - monthly["expense"]
    monthly["savings_rate_pct"] = np.where(
        monthly["income"] > 0,
        (monthly["net_savings"] / monthly["income"]) * 100.0,
        0.0,
    )
    monthly["cumulative_savings"] = monthly["net_savings"].cumsum()
    monthly["expense_3m_avg"] = monthly["expense"].rolling(window=3, min_periods=1).mean()

    return monthly.sort_values("year_month").reset_index(drop=True)


def category_spending_summary(df: pd.DataFrame, budgets_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Computes category-wise expense totals, distribution shares, and budget variances."""
    if df.empty:
        return pd.DataFrame()

    df = ensure_datetime(df)
    expense_df = df[df["type"] == "expense"]

    if expense_df.empty:
        return pd.DataFrame()

    cat_stats = expense_df.groupby("category")["amount"].agg(
        total_amount="sum",
        tx_count="count",
        avg_amount="mean",
        max_amount="max",
    ).reset_index()

    total_spent = cat_stats["total_amount"].sum()
    cat_stats["percentage_of_total"] = (cat_stats["total_amount"] / total_spent) * 100.0

    if budgets_df is not None and not budgets_df.empty:
        cat_stats = pd.merge(cat_stats, budgets_df, on="category", how="left")
        cat_stats["monthly_budget"] = cat_stats["monthly_budget"].fillna(0.0)

    cat_stats = cat_stats.sort_values("total_amount", ascending=False).reset_index(drop=True)
    return cat_stats


def daily_spending_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Computes daily expense volume and rolling moving averages."""
    if df.empty:
        return pd.DataFrame()

    df = ensure_datetime(df)
    expense_df = df[df["type"] == "expense"].copy()
    if expense_df.empty:
        return pd.DataFrame()

    daily = expense_df.groupby(expense_df["date"].dt.date)["amount"].sum().reset_index()
    daily.columns = ["date", "amount"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # Continuous date range reindexing to accurately account for zero-spend days
    full_range = pd.date_range(start=daily["date"].min(), end=daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_range, fill_value=0.0).rename_axis("date").reset_index()

    daily["rolling_7d_avg"] = daily["amount"].rolling(window=7, min_periods=1).mean()
    daily["rolling_30d_avg"] = daily["amount"].rolling(window=30, min_periods=1).mean()

    return daily


def day_of_week_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes average and total spending across days of the week."""
    if df.empty:
        return pd.DataFrame()

    df = ensure_datetime(df)
    expense_df = df[df["type"] == "expense"].copy()
    if expense_df.empty:
        return pd.DataFrame()

    expense_df["day_name"] = expense_df["date"].dt.day_name()
    expense_df["day_num"] = expense_df["date"].dt.day_of_week

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats = expense_df.groupby(["day_num", "day_name"])["amount"].agg(
        total_amount="sum",
        tx_count="count",
        avg_amount="mean",
    ).reset_index()

    dow_stats = dow_stats.sort_values("day_num").reset_index(drop=True)
    return dow_stats


def payment_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes distribution across payment methods."""
    if df.empty or "payment_method" not in df.columns:
        return pd.DataFrame()

    expense_df = df[df["type"] == "expense"]
    if expense_df.empty:
        return pd.DataFrame()

    pm_stats = expense_df.groupby("payment_method")["amount"].agg(
        total_amount="sum",
        tx_count="count",
    ).reset_index()

    pm_stats["percentage"] = (pm_stats["total_amount"] / pm_stats["total_amount"].sum()) * 100.0
    return pm_stats.sort_values("total_amount", ascending=False).reset_index(drop=True)


def current_month_budget_status(
    df: pd.DataFrame, budgets_df: pd.DataFrame, target_month: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluates current (or selected) month category expenses against budgeted limits.
    Returns budget utilization %, remaining balance, and alert status.
    """
    if df.empty or budgets_df.empty:
        return pd.DataFrame()

    df = ensure_datetime(df)
    expense_df = df[df["type"] == "expense"].copy()

    if target_month is None:
        target_month = expense_df["date"].dt.strftime("%Y-%m").max()

    month_df = expense_df[expense_df["date"].dt.strftime("%Y-%m") == target_month]

    month_spent = month_df.groupby("category")["amount"].sum().reset_index()
    month_spent.rename(columns={"amount": "actual_spent"}, inplace=True)

    status_df = pd.merge(budgets_df, month_spent, on="category", how="left")
    status_df["actual_spent"] = status_df["actual_spent"].fillna(0.0)
    status_df["remaining"] = status_df["monthly_budget"] - status_df["actual_spent"]
    status_df["utilization_pct"] = np.where(
        status_df["monthly_budget"] > 0,
        (status_df["actual_spent"] / status_df["monthly_budget"]) * 100.0,
        0.0,
    )

    def classify_status(row):
        util = row["utilization_pct"]
        if util > 100.0:
            return "Over Budget"
        elif util >= 85.0:
            return "Near Limit"
        else:
            return "On Track"

    status_df["status"] = status_df.apply(classify_status, axis=1)
    status_df["target_month"] = target_month
    return status_df.sort_values("utilization_pct", ascending=False).reset_index(drop=True)
