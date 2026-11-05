"""Machine Learning Anomaly Detection module.
Uses Scikit-learn IsolationForest to flag unusual or suspicious transactions
and generates interpretable reasons based on category statistics and risk scoring.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


class TransactionAnomalyDetector:
    """Detects unusual financial expenditures using IsolationForest and statistical profiling."""

    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        self.category_stats: Dict[str, Dict[str, float]] = {}

    def fit_and_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trains IsolationForest on expense transactions and annotates anomalies."""
        if df.empty:
            return pd.DataFrame()

        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy["date"]):
            df_copy["date"] = pd.to_datetime(df_copy["date"])

        # Filter strictly for expenses (income anomalies are typically positive bonuses)
        expense_df = df_copy[df_copy["type"] == "expense"].copy()
        if len(expense_df) < 10:
            # Fallback if too few rows
            expense_df["is_anomaly"] = False
            expense_df["anomaly_score"] = 0.0
            expense_df["risk_level"] = "Normal"
            expense_df["explanation"] = "Insufficient data for ML anomaly detection"
            return expense_df

        # Calculate category statistics
        self.category_stats = {}
        for cat, group in expense_df.groupby("category"):
            mean_val = float(group["amount"].mean())
            std_val = float(group["amount"].std()) if len(group) > 1 else 1.0
            std_val = max(std_val, 1.0)
            median_val = float(group["amount"].median())
            self.category_stats[cat] = {
                "mean": mean_val,
                "std": std_val,
                "median": median_val,
                "p95": float(np.percentile(group["amount"], 95)),
            }

        # Feature Engineering
        def compute_cat_zscore(row):
            stats = self.category_stats.get(row["category"], {"mean": 50.0, "std": 20.0})
            return (row["amount"] - stats["mean"]) / stats["std"]

        def compute_cat_ratio(row):
            stats = self.category_stats.get(row["category"], {"median": 50.0})
            return row["amount"] / max(stats["median"], 1.0)

        expense_df["cat_zscore"] = expense_df.apply(compute_cat_zscore, axis=1)
        expense_df["cat_ratio"] = expense_df.apply(compute_cat_ratio, axis=1)
        expense_df["day_of_month"] = expense_df["date"].dt.day
        expense_df["day_of_week"] = expense_df["date"].dt.day_of_week
        expense_df["is_weekend"] = (expense_df["day_of_week"] >= 5).astype(int)

        feature_cols = ["amount", "cat_zscore", "cat_ratio", "day_of_month", "day_of_week"]
        X = expense_df[feature_cols].fillna(0.0)

        # Train IsolationForest
        self.model.fit(X)

        # Scikit-learn outputs -1 for anomalies and 1 for inliers
        predictions = self.model.predict(X)
        raw_scores = self.model.decision_function(X)  # Lower values indicate higher anomaly likelihood

        # Normalize score into intuitive 0 - 100 risk scale (higher = more anomalous)
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        score_range = max_score - min_score if max_score != min_score else 1.0
        normalized_risk = 100.0 * (1.0 - (raw_scores - min_score) / score_range)

        expense_df["is_anomaly"] = predictions == -1
        expense_df["anomaly_score"] = np.round(normalized_risk, 1)

        # Assign Risk Level and Interpretability Explanations
        def assign_risk(row):
            score = row["anomaly_score"]
            if not row["is_anomaly"] and score < 50:
                return "Normal"
            elif score < 65:
                return "Moderate"
            elif score < 85:
                return "High"
            else:
                return "Critical"

        def explain_anomaly(row):
            if not row["is_anomaly"]:
                return "Within typical category patterns"

            cat = row["category"]
            stats = self.category_stats.get(cat, {})
            mean = stats.get("mean", 0.0)
            median = stats.get("median", 0.0)
            amount = row["amount"]
            ratio = row["cat_ratio"]
            z = row["cat_zscore"]

            reasons = []
            if ratio >= 3.0:
                reasons.append(f"{ratio:.1f}x higher than category median (₹{median:,.0f})")
            if z >= 3.0:
                reasons.append(f"{z:.1f}σ above average spend (₹{mean:,.0f})")
            if amount > 25000.0:
                reasons.append("High-ticket single transaction (>₹25,000)")

            if not reasons:
                reasons.append(f"Unusual timing and amount deviation for {cat}")

            return " | ".join(reasons)

        expense_df["risk_level"] = expense_df.apply(assign_risk, axis=1)
        expense_df["explanation"] = expense_df.apply(explain_anomaly, axis=1)

        # Sort by anomaly score descending so highest risk items appear first
        output_cols = [
            "id", "date", "category", "amount", "description",
            "payment_method", "is_anomaly", "anomaly_score", "risk_level", "explanation"
        ]
        available_cols = [c for c in output_cols if c in expense_df.columns]
        result = expense_df[available_cols].sort_values("anomaly_score", ascending=False).reset_index(drop=True)
        return result

    def get_summary_stats(self, annotated_df: pd.DataFrame) -> Dict[str, Any]:
        """Returns aggregate anomaly counts, flagged spend volume, and category distribution."""
        if annotated_df.empty or "is_anomaly" not in annotated_df.columns:
            return {"total_transactions": 0, "anomalies_detected": 0, "flagged_amount": 0.0}

        anomalies = annotated_df[annotated_df["is_anomaly"]]
        return {
            "total_transactions": len(annotated_df),
            "anomalies_detected": len(anomalies),
            "anomaly_rate_pct": round((len(anomalies) / len(annotated_df)) * 100.0, 1),
            "flagged_amount": round(float(anomalies["amount"].sum()), 2),
            "critical_risk_count": int((annotated_df["risk_level"] == "Critical").sum()),
            "high_risk_count": int((annotated_df["risk_level"] == "High").sum()),
        }
