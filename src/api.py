"""FastAPI REST Service for SpendPulse.
Exposes endpoints for transaction management, budgets, analytics, and machine learning models.
"""

from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.database import (
    init_db,
    seed_synthetic_data,
    add_transaction,
    delete_transaction,
    get_transactions_df,
    set_budget,
    get_budgets_df,
    get_budgets,
)
from src.analytics import (
    get_kpis,
    get_financial_health_score,
    monthly_spending_summary,
    category_spending_summary,
    current_month_budget_status,
)
from src.ml.forecaster import ExpenseForecaster
from src.ml.anomaly_detector import TransactionAnomalyDetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB and ensure demo data exists on startup
    init_db()
    seed_synthetic_data(overwrite=False)
    yield


app = FastAPI(
    title="SpendPulse API",
    description="Personal Expense Analytics & Machine Learning Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Pydantic Schemas
# ==========================================

class TransactionCreate(BaseModel):
    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    category: str = Field(..., description="Expense or income category")
    type: str = Field(..., description="'expense' or 'income'")
    amount: float = Field(..., gt=0, description="Positive transaction amount")
    description: Optional[str] = Field("", description="Optional memo or description")
    payment_method: Optional[str] = Field("Debit Card", description="Payment method used")


class BudgetSet(BaseModel):
    category: str = Field(..., description="Category name")
    monthly_budget: float = Field(..., gt=0, description="Monthly allocated budget amount")


class ForecastRequest(BaseModel):
    horizon_months: int = Field(3, ge=1, le=12, description="Number of months to forecast ahead")
    model_type: str = Field("ridge", description="'ridge', 'linear', or 'random_forest'")


class AnomalyRequest(BaseModel):
    contamination: float = Field(0.03, ge=0.01, le=0.15, description="Expected proportion of anomalies")


# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
def root():
    return {
        "service": "SpendPulse API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
        "description": "Personal Expense Analytics & Machine Learning Prediction Service",
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.get("/api/transactions")
def list_transactions(
    start_date: Optional[str] = Query(None, description="Filter from YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Filter to YYYY-MM-DD"),
    category: Optional[str] = Query(None, description="Filter by category"),
    trans_type: Optional[str] = Query(None, description="'expense' or 'income'"),
    limit: int = Query(1000, ge=1, le=5000),
):
    df = get_transactions_df(
        start_date=start_date,
        end_date=end_date,
        category=category,
        trans_type=trans_type,
    )
    if not df.empty:
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df.head(limit).to_dict(orient="records")
    return []


@app.post("/api/transactions", status_code=201)
def create_transaction(item: TransactionCreate):
    new_id = add_transaction(
        date=item.date,
        category=item.category,
        trans_type=item.type,
        amount=item.amount,
        description=item.description or "",
        payment_method=item.payment_method or "Debit Card",
    )
    return {"status": "success", "id": new_id, "message": "Transaction recorded"}


@app.delete("/api/transactions/{transaction_id}")
def remove_transaction(transaction_id: int):
    success = delete_transaction(transaction_id)
    if not success:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return {"status": "success", "message": f"Transaction {transaction_id} deleted"}


@app.get("/api/budgets")
def list_budgets():
    return get_budgets()


@app.post("/api/budgets")
def update_budget(item: BudgetSet):
    set_budget(item.category, item.monthly_budget)
    return {"status": "success", "category": item.category, "monthly_budget": item.monthly_budget}


@app.get("/api/analytics/kpis")
def get_analytics_kpis():
    df = get_transactions_df()
    budgets_df = get_budgets_df()
    return get_kpis(df, budgets_df)


@app.get("/api/analytics/health-score")
def get_health_score_endpoint():
    df = get_transactions_df()
    budgets_df = get_budgets_df()
    kpis = get_kpis(df, budgets_df)
    return get_financial_health_score(kpis)


@app.get("/api/analytics/monthly")
def get_monthly_analytics():
    df = get_transactions_df()
    monthly = monthly_spending_summary(df)
    if monthly.empty:
        return []
    return monthly.to_dict(orient="records")


@app.get("/api/analytics/categories")
def get_category_analytics():
    df = get_transactions_df()
    budgets_df = get_budgets_df()
    cats = category_spending_summary(df, budgets_df)
    if cats.empty:
        return []
    return cats.to_dict(orient="records")


@app.get("/api/analytics/budget-status")
def get_budget_status(month: Optional[str] = Query(None, description="Target month in YYYY-MM")):
    df = get_transactions_df()
    budgets_df = get_budgets_df()
    status_df = current_month_budget_status(df, budgets_df, target_month=month)
    if status_df.empty:
        return []
    return status_df.to_dict(orient="records")


@app.post("/api/ml/forecast")
def run_expense_forecast(req: ForecastRequest):
    df = get_transactions_df()
    if df.empty:
        raise HTTPException(status_code=400, detail="No transactions available for training")

    forecaster = ExpenseForecaster(model_type=req.model_type)
    forecaster.train_and_evaluate(df)
    results = forecaster.forecast(df, horizon_months=req.horizon_months)
    return results


@app.post("/api/ml/anomalies")
def detect_transaction_anomalies(req: AnomalyRequest):
    df = get_transactions_df()
    if df.empty:
        raise HTTPException(status_code=400, detail="No transactions available for anomaly detection")

    detector = TransactionAnomalyDetector(contamination=req.contamination)
    annotated = detector.fit_and_detect(df)
    summary = detector.get_summary_stats(annotated)

    if not annotated.empty:
        annotated["date"] = annotated["date"].dt.strftime("%Y-%m-%d")
        records = annotated.to_dict(orient="records")
    else:
        records = []

    return {
        "summary": summary,
        "anomalies": [r for r in records if r["is_anomaly"]],
        "all_annotated": records,
    }


@app.post("/api/admin/reset-seed")
def reset_database(overwrite: bool = True):
    count = seed_synthetic_data(overwrite=overwrite)
    return {"status": "success", "message": f"Database seeded with {count} records"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
