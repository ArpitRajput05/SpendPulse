"""Automated Test Suite for SpendPulse.
Covers database, analytics, ML forecasting, anomaly detection, and FastAPI endpoints.
"""

import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.database import (
    init_db,
    seed_synthetic_data,
    add_transaction,
    delete_transaction,
    get_transactions_df,
    set_budget,
    get_budgets,
    get_budgets_df,
)
from src.analytics import (
    get_kpis,
    monthly_spending_summary,
    category_spending_summary,
    daily_spending_trend,
    day_of_week_analysis,
    current_month_budget_status,
)
from src.ml.forecaster import ExpenseForecaster
from src.ml.anomaly_detector import TransactionAnomalyDetector
from src.api import app


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Ensure database has clean schema and seed data for testing."""
    init_db()
    seed_synthetic_data(num_months=12, overwrite=True)


def test_database_crud():
    # 1. Add transaction
    new_id = add_transaction(
        date="2026-02-15",
        category="Dining Out",
        trans_type="expense",
        amount=42.50,
        description="Test Lunch",
        payment_method="Credit Card",
    )
    assert new_id > 0

    # 2. Query transactions
    df = get_transactions_df(start_date="2026-02-15", end_date="2026-02-15")
    assert not df.empty
    found = df[df["id"] == new_id]
    assert len(found) == 1
    assert float(found.iloc[0]["amount"]) == 42.50

    # 3. Delete transaction
    deleted = delete_transaction(new_id)
    assert deleted is True

    # 4. Set & query budget
    set_budget("Test Category", 550.0)
    budgets = get_budgets()
    assert "Test Category" in budgets
    assert budgets["Test Category"] == 550.0


def test_analytics_kpis():
    df = get_transactions_df()
    budgets_df = get_budgets_df()
    kpis = get_kpis(df, budgets_df)

    assert "total_income" in kpis
    assert "total_expenses" in kpis
    assert "net_savings" in kpis
    assert "savings_rate_pct" in kpis
    assert kpis["total_income"] > 0
    assert kpis["total_expenses"] > 0
    assert kpis["total_transactions"] > 100


def test_monthly_and_category_summaries():
    df = get_transactions_df()
    budgets_df = get_budgets_df()

    monthly = monthly_spending_summary(df)
    assert not monthly.empty
    assert "year_month" in monthly.columns
    assert "net_savings" in monthly.columns

    cats = category_spending_summary(df, budgets_df)
    assert not cats.empty
    assert "total_amount" in cats.columns
    assert "percentage_of_total" in cats.columns
    assert round(cats["percentage_of_total"].sum(), 0) == 100.0


def test_ml_forecaster():
    df = get_transactions_df()
    forecaster = ExpenseForecaster(model_type="ridge")

    # Training and metrics
    metrics = forecaster.train_and_evaluate(df)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2_score" in metrics
    assert metrics["historical_months_count"] >= 5

    # Multi-step forecasting
    results = forecaster.forecast(df, horizon_months=3)
    forecasts = results["forecasts"]
    assert len(forecasts) == 3

    for fc in forecasts:
        assert fc["predicted_expense"] > 0
        assert fc["lower_bound"] <= fc["predicted_expense"]
        assert fc["upper_bound"] >= fc["predicted_expense"]

    # Category projections
    cat_projections = results["category_projections_next_month"]
    assert len(cat_projections) > 0
    assert sum(p["percentage"] for p in cat_projections) > 95.0


def test_ml_anomaly_detector():
    df = get_transactions_df()
    detector = TransactionAnomalyDetector(contamination=0.03)

    annotated = detector.fit_and_detect(df)
    assert not annotated.empty
    assert "is_anomaly" in annotated.columns
    assert "anomaly_score" in annotated.columns
    assert "risk_level" in annotated.columns
    assert "explanation" in annotated.columns

    # Verify outlier flags and summary
    summary = detector.get_summary_stats(annotated)
    assert summary["total_transactions"] > 100
    assert summary["anomalies_detected"] > 0
    assert summary["flagged_amount"] > 0

    # Ensure deliberate anomalies are caught with High or Critical risk
    anomalies = annotated[annotated["is_anomaly"]]
    high_critical = anomalies[anomalies["risk_level"].isin(["High", "Critical"])]
    assert len(high_critical) > 0


def test_fastapi_endpoints():
    client = TestClient(app)

    # 1. Health & Root
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert r_root.json()["status"] == "online"

    # 2. KPIs
    r_kpis = client.get("/api/analytics/kpis")
    assert r_kpis.status_code == 200
    assert "total_income" in r_kpis.json()

    # 3. Monthly & Categories
    r_monthly = client.get("/api/analytics/monthly")
    assert r_monthly.status_code == 200
    assert isinstance(r_monthly.json(), list)

    r_cats = client.get("/api/analytics/categories")
    assert r_cats.status_code == 200
    assert isinstance(r_cats.json(), list)

    # 4. ML Forecast Endpoint
    r_fc = client.post("/api/ml/forecast", json={"horizon_months": 3, "model_type": "ridge"})
    assert r_fc.status_code == 200
    fc_data = r_fc.json()
    assert "forecasts" in fc_data
    assert len(fc_data["forecasts"]) == 3

    # 5. ML Anomaly Endpoint
    r_anom = client.post("/api/ml/anomalies", json={"contamination": 0.03})
    assert r_anom.status_code == 200
    anom_data = r_anom.json()
    assert "summary" in anom_data
    assert anom_data["summary"]["anomalies_detected"] > 0
