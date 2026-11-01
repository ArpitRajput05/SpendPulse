"""Machine Learning Expense Forecasting module.
Uses Scikit-learn to train regression models on historical time-series features
to forecast future expenses with confidence intervals and evaluation metrics.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ExpenseForecaster:
    """Time-series ML Forecaster for monthly personal expenses."""

    def __init__(self, model_type: str = "ridge"):
        self.model_type = model_type.lower()
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=4)
        elif self.model_type == "linear":
            self.model = LinearRegression()
        else:
            self.model = Ridge(alpha=1.0)

        self.feature_names = ["lag_1", "lag_2", "rolling_mean_2", "month", "trend_idx"]
        self.metrics: Dict[str, float] = {}
        self.last_known_data: Optional[pd.DataFrame] = None
        self.residual_std: float = 100.0

    def prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates transactions into monthly totals and creates lagged feature set."""
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy["date"]):
            df_copy["date"] = pd.to_datetime(df_copy["date"])

        expense_df = df_copy[df_copy["type"] == "expense"].copy()
        if expense_df.empty:
            return pd.DataFrame()

        # Monthly total expenses
        monthly = expense_df.set_index("date").resample("MS")["amount"].sum().reset_index()
        monthly.columns = ["date", "actual_expense"]

        # If few months, interpolate or handle safely
        monthly["trend_idx"] = np.arange(len(monthly))
        monthly["month"] = monthly["date"].dt.month
        monthly["lag_1"] = monthly["actual_expense"].shift(1)
        monthly["lag_2"] = monthly["actual_expense"].shift(2)
        monthly["rolling_mean_2"] = monthly["actual_expense"].shift(1).rolling(window=2, min_periods=1).mean()

        # Drop NaNs created by lagging
        return monthly

    def train_and_evaluate(self, df: pd.DataFrame, test_size: int = 3) -> Dict[str, Any]:
        """Trains regression model on historical monthly expenses and computes test metrics."""
        dataset = self.prepare_dataset(df)
        if len(dataset) < 5:
            # Fallback for very small datasets
            return {
                "status": "error",
                "message": f"Insufficient historical months ({len(dataset)}). Need at least 5 months.",
            }

        valid_data = dataset.dropna().reset_index(drop=True)
        self.last_known_data = valid_data

        X = valid_data[self.feature_names]
        y = valid_data["actual_expense"]

        # Train / Test split on time-series (latest months as holdout test)
        if len(valid_data) > test_size + 2:
            train_idx = len(valid_data) - test_size
            X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]
            y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]
        else:
            X_train, X_test = X, X
            y_train, y_test = y, y

        # Fit model
        self.model.fit(X_train, y_train)

        # Predictions & metrics
        y_pred = self.model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else 1.0

        # Refit on all available data for future forecasting
        self.model.fit(X, y)
        all_pred = self.model.predict(X)
        residuals = y - all_pred
        self.residual_std = float(np.std(residuals)) if len(residuals) > 1 else 100.0

        self.metrics = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2_score": round(max(-1.0, min(1.0, r2)), 3),
            "historical_months_count": len(valid_data),
            "test_months_evaluated": len(X_test),
            "model_type": self.model_type,
        }

        # Feature importances / coefficients
        if hasattr(self.model, "coef_"):
            importances = dict(zip(self.feature_names, [round(float(c), 3) for c in self.model.coef_]))
        elif hasattr(self.model, "feature_importances_"):
            importances = dict(zip(self.feature_names, [round(float(c), 3) for c in self.model.feature_importances_]))
        else:
            importances = {}
        self.metrics["feature_importances"] = importances

        return self.metrics

    def forecast(self, df: pd.DataFrame, horizon_months: int = 3) -> Dict[str, Any]:
        """Generates future month-by-month expense forecasts with confidence intervals."""
        if not self.metrics:
            self.train_and_evaluate(df)

        dataset = self.prepare_dataset(df)
        if dataset.empty or len(dataset.dropna()) < 3:
            return {"error": "Not enough historical data to generate forecast"}

        last_row = dataset.dropna().iloc[-1]
        last_date = last_row["date"]
        current_trend = int(last_row["trend_idx"])
        last_val = float(last_row["actual_expense"])
        prev_val = float(last_row["lag_1"]) if not np.isnan(last_row["lag_1"]) else last_val

        forecasts: List[Dict[str, Any]] = []

        curr_lag_1 = last_val
        curr_lag_2 = prev_val

        for step in range(1, horizon_months + 1):
            next_date = last_date + pd.DateOffset(months=step)
            next_month = next_date.month
            next_trend = current_trend + step
            rolling_2 = (curr_lag_1 + curr_lag_2) / 2.0

            x_input = pd.DataFrame([{
                "lag_1": curr_lag_1,
                "lag_2": curr_lag_2,
                "rolling_mean_2": rolling_2,
                "month": next_month,
                "trend_idx": next_trend,
            }])[self.feature_names]

            pred_val = float(self.model.predict(x_input)[0])
            pred_val = max(100.0, pred_val)  # Floor at reasonable positive spend

            # 90% confidence interval bound: 1.645 * residual_std * sqrt(step)
            margin = float(1.645 * self.residual_std * np.sqrt(step))
            lower_bound = max(0.0, pred_val - margin)
            upper_bound = pred_val + margin

            forecasts.append({
                "date": next_date.strftime("%Y-%m"),
                "predicted_expense": round(pred_val, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "step": step,
            })

            # Update auto-regressive state for next iteration
            curr_lag_2 = curr_lag_1
            curr_lag_1 = pred_val

        # Category-level breakdown forecast for Month 1
        category_breakdown = self._forecast_category_projections(df, forecasts[0]["predicted_expense"])

        return {
            "forecasts": forecasts,
            "category_projections_next_month": category_breakdown,
            "metrics": self.metrics,
        }

    def _forecast_category_projections(self, df: pd.DataFrame, next_total_spend: float) -> List[Dict[str, Any]]:
        """Projects category proportions for the next month based on recent expenditure weights."""
        expense_df = df[df["type"] == "expense"].copy()
        if expense_df.empty:
            return []

        # Use recent 3 months to capture active category weights
        if not pd.api.types.is_datetime64_any_dtype(expense_df["date"]):
            expense_df["date"] = pd.to_datetime(expense_df["date"])

        recent_cutoff = expense_df["date"].max() - pd.DateOffset(months=3)
        recent_df = expense_df[expense_df["date"] >= recent_cutoff]
        if recent_df.empty:
            recent_df = expense_df

        cat_totals = recent_df.groupby("category")["amount"].sum()
        total_recent = cat_totals.sum()

        projections = []
        if total_recent > 0:
            for cat, amt in cat_totals.items():
                ratio = amt / total_recent
                proj_amt = ratio * next_total_spend
                projections.append({
                    "category": str(cat),
                    "projected_amount": round(proj_amt, 2),
                    "percentage": round(ratio * 100.0, 1),
                })

        projections.sort(key=lambda x: x["projected_amount"], reverse=True)
        return projections
