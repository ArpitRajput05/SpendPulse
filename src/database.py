"""Database access and seeding module for SpendPulse.
Uses SQLite for persistent storage of transactions and budgets.
"""

import os
import sqlite3
import random
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Any, Optional

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.environ.get("SPENDPULSE_DB_PATH", os.path.join(DB_DIR, "spendpulse.db"))


def get_db_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes tables for transactions and budgets if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('income', 'expense')),
            amount REAL NOT NULL,
            description TEXT,
            payment_method TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS budgets (
            category TEXT PRIMARY KEY,
            monthly_budget REAL NOT NULL
        );
        """
    )

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_date ON transactions(date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_cat ON transactions(category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_type ON transactions(type);")

    conn.commit()
    conn.close()


def add_transaction(
    date: str,
    category: str,
    trans_type: str,
    amount: float,
    description: str = "",
    payment_method: str = "Debit Card",
) -> int:
    """Inserts a new transaction into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO transactions (date, category, type, amount, description, payment_method)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (date, category, trans_type.lower(), float(amount), description, payment_method),
    )
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id


def delete_transaction(transaction_id: int) -> bool:
    """Deletes a transaction by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def set_budget(category: str, monthly_budget: float) -> None:
    """Sets or updates the monthly budget limit for a category."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO budgets (category, monthly_budget)
        VALUES (?, ?)
        ON CONFLICT(category) DO UPDATE SET monthly_budget = excluded.monthly_budget
        """,
        (category, float(monthly_budget)),
    )
    conn.commit()
    conn.close()


def get_budgets() -> Dict[str, float]:
    """Retrieves all category budgets as a dictionary."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT category, monthly_budget FROM budgets ORDER BY category")
    rows = cursor.fetchall()
    conn.close()
    return {row["category"]: row["monthly_budget"] for row in rows}


def get_budgets_df() -> pd.DataFrame:
    """Retrieves all budgets as a pandas DataFrame."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT category, monthly_budget FROM budgets ORDER BY category", conn)
    conn.close()
    return df


def get_transactions_df(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    trans_type: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieves transactions with optional filtering as a pandas DataFrame."""
    query = "SELECT id, date, category, type, amount, description, payment_method, created_at FROM transactions WHERE 1=1"
    params = []

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    if category and category != "All":
        query += " AND category = ?"
        params.append(category)
    if trans_type and trans_type != "All":
        query += " AND type = ?"
        params.append(trans_type.lower())

    query += " ORDER BY date DESC, id DESC"

    conn = get_db_connection()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


DEFAULT_BUDGETS = {
    "Housing & Rent": 1250.0,
    "Groceries": 450.0,
    "Dining Out": 250.0,
    "Utilities": 190.0,
    "Transportation": 180.0,
    "Entertainment": 140.0,
    "Shopping": 220.0,
    "Healthcare": 110.0,
    "Subscriptions": 45.0,
    "Travel": 160.0,
}


def seed_synthetic_data(num_months: int = 14, overwrite: bool = False) -> int:
    """
    Populates SQLite database with rich, realistic financial transactions
    and default category budgets.
    
    Includes realistic recurring salaries, seasonal spending (holiday surges),
    periodic bills, daily grocery/dining expenses, and deliberate anomalies
    for machine learning detection.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM transactions")
    existing_count = cursor.fetchone()[0]

    if existing_count > 0 and not overwrite:
        conn.close()
        return existing_count

    if overwrite:
        cursor.execute("DELETE FROM transactions")
        cursor.execute("DELETE FROM budgets")
        conn.commit()

    # Seed default budgets
    for cat, amount in DEFAULT_BUDGETS.items():
        cursor.execute(
            "INSERT OR REPLACE INTO budgets (category, monthly_budget) VALUES (?, ?)",
            (cat, amount),
        )

    # Base reference date: End of Feb 2026
    end_date = datetime(2026, 2, 28)
    start_date = end_date - timedelta(days=num_months * 30.5)

    random.seed(42)  # Deterministic seed for reproducible data science demos

    transactions = []

    # Monthly salary dates and recurring expenses
    cur_month_start = datetime(start_date.year, start_date.month, 1)

    while cur_month_start <= end_date:
        year = cur_month_start.year
        month = cur_month_start.month

        # Income: Bi-monthly Salary on 1st and 15th
        day_1 = datetime(year, month, 1)
        day_15 = datetime(year, month, 15)

        base_salary = 2400.0 + random.uniform(-50, 50)
        transactions.append((
            day_1.strftime("%Y-%m-%d"),
            "Income",
            "income",
            round(base_salary, 2),
            "Monthly Salary - Part 1",
            "Bank Transfer",
        ))
        transactions.append((
            day_15.strftime("%Y-%m-%d"),
            "Income",
            "income",
            round(base_salary, 2),
            "Monthly Salary - Part 2",
            "Bank Transfer",
        ))

        # Freelance Income (occasional)
        if random.random() < 0.45:
            freelance_day = datetime(year, month, random.randint(5, 25))
            transactions.append((
                freelance_day.strftime("%Y-%m-%d"),
                "Income",
                "income",
                round(random.uniform(350, 950), 2),
                "Consulting & Freelance Web Project",
                "Bank Transfer",
            ))

        # Year-end performance bonus in December
        if month == 12:
            bonus_day = datetime(year, month, 20)
            transactions.append((
                bonus_day.strftime("%Y-%m-%d"),
                "Income",
                "income",
                1500.0,
                "Annual Performance Bonus",
                "Bank Transfer",
            ))

        # Recurring Fixed Expenses
        # 1. Rent / Housing on the 2nd
        rent_date = datetime(year, month, 2)
        transactions.append((
            rent_date.strftime("%Y-%m-%d"),
            "Housing & Rent",
            "expense",
            1250.0,
            "Apartment Monthly Rent",
            "Bank Transfer",
        ))

        # 2. Utilities (Power, Water, Internet)
        util_date = datetime(year, month, 5)
        # Power is higher in winter/summer
        seasonal_util_mult = 1.25 if month in [12, 1, 7, 8] else 1.0
        util_amount = round(random.uniform(160, 210) * seasonal_util_mult, 2)
        transactions.append((
            util_date.strftime("%Y-%m-%d"),
            "Utilities",
            "expense",
            util_amount,
            "Electricity, Gas & Fiber Internet",
            "Debit Card",
        ))

        # 3. Subscriptions (Streaming, Cloud, Gym)
        sub_date = datetime(year, month, 8)
        transactions.append((
            sub_date.strftime("%Y-%m-%d"),
            "Subscriptions",
            "expense",
            44.99,
            "Netflix, Spotify & Gym Membership",
            "Credit Card",
        ))

        # Move to next month
        if month == 12:
            cur_month_start = datetime(year + 1, 1, 1)
        else:
            cur_month_start = datetime(year, month + 1, 1)

    # Variable Everyday Expenses across the timeline
    days_range = (end_date - start_date).days
    grocery_stores = ["Trader Joe's", "Whole Foods", "Costco Wholesale", "Kroger", "Farmers Market"]
    dining_places = ["Chipotle Mexican Grill", "Starbucks Coffee", "Olive Garden", "Local Pizzeria", "Thai Bistro", "Sushi Bar"]
    transport_vendors = ["Shell Gas Station", "Metro Card Refill", "Uber Ride", "Chevron Fuel"]
    shopping_vendors = ["Amazon", "Target", "IKEA", "Uniqlo", "Nike Store"]
    entertainment_vendors = ["Cinema Movie Tickets", "Steam Games", "Concert Ticket", "Bowling Night"]
    health_vendors = ["CVS Pharmacy", "Walgreens", "Dental Cleaning Co-Pay", "Optometry Exam"]

    for d in range(days_range):
        current_day = start_date + timedelta(days=d)
        date_str = current_day.strftime("%Y-%m-%d")
        month = current_day.month
        is_weekend = current_day.weekday() >= 5

        # Groceries: 2-3 times a week
        if random.random() < 0.35:
            store = random.choice(grocery_stores)
            amount = round(random.uniform(32, 125), 2)
            transactions.append((
                date_str,
                "Groceries",
                "expense",
                amount,
                f"Groceries at {store}",
                random.choice(["Credit Card", "Debit Card"]),
            ))

        # Dining Out: frequent on weekends
        dining_prob = 0.55 if is_weekend else 0.22
        if random.random() < dining_prob:
            place = random.choice(dining_places)
            amount = round(random.uniform(15, 65), 2)
            transactions.append((
                date_str,
                "Dining Out",
                "expense",
                amount,
                f"Dinner / Coffee at {place}",
                "Credit Card",
            ))

        # Transportation: fuel or transit
        if random.random() < 0.20:
            vendor = random.choice(transport_vendors)
            amount = round(random.uniform(22, 58), 2)
            transactions.append((
                date_str,
                "Transportation",
                "expense",
                amount,
                f"Fuel / Transit ({vendor})",
                random.choice(["Credit Card", "Debit Card"]),
            ))

        # Shopping: occasional, surges in November / December (Black Friday / Holidays)
        shopping_prob = 0.25 if month in [11, 12] else 0.12
        if random.random() < shopping_prob:
            vendor = random.choice(shopping_vendors)
            base_amt = random.uniform(25, 110)
            if month in [11, 12]:
                base_amt *= 1.4
            transactions.append((
                date_str,
                "Shopping",
                "expense",
                round(base_amt, 2),
                f"Online / Retail Order at {vendor}",
                "Credit Card",
            ))

        # Entertainment
        if random.random() < 0.10:
            vendor = random.choice(entertainment_vendors)
            transactions.append((
                date_str,
                "Entertainment",
                "expense",
                round(random.uniform(20, 75), 2),
                f"Recreation - {vendor}",
                "Credit Card",
            ))

        # Healthcare: occasional prescriptions or co-pays
        if random.random() < 0.05:
            vendor = random.choice(health_vendors)
            transactions.append((
                date_str,
                "Healthcare",
                "expense",
                round(random.uniform(25, 80), 2),
                f"Healthcare - {vendor}",
                "Debit Card",
            ))

        # Travel: summer & holiday season trips
        if month in [6, 7, 12] and random.random() < 0.04:
            transactions.append((
                date_str,
                "Travel",
                "expense",
                round(random.uniform(180, 420), 2),
                "Flight / Hotel Booking",
                "Credit Card",
            ))

    # DELIBERATE ANOMALIES (for Scikit-Learn Anomaly Detection / IsolationForest validation)
    deliberate_anomalies = [
        ("2025-04-18", "Transportation", "expense", 1480.00, "Emergency Transmission & Brake Overhaul", "Credit Card"),
        ("2025-07-22", "Travel", "expense", 1350.00, "Luxury Island Resort Booking & Flight Package", "Credit Card"),
        ("2025-10-14", "Healthcare", "expense", 780.00, "Emergency Dental Surgery & Crown Replacement", "Debit Card"),
        ("2025-11-28", "Shopping", "expense", 1899.99, "Black Friday UltraWide Monitor & Gaming Setup", "Credit Card"),
        ("2026-01-19", "Dining Out", "expense", 345.50, "Catering Event / Accidental Double Charge", "Credit Card"),
        ("2026-02-11", "Groceries", "expense", 412.80, "Bulk Annual Pantry & Organic Meat Stockup", "Credit Card"),
    ]

    for anomaly in deliberate_anomalies:
        transactions.append(anomaly)

    # Insert all transactions in batch
    cursor.executemany(
        """
        INSERT INTO transactions (date, category, type, amount, description, payment_method)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        transactions,
    )

    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM transactions")
    total_count = cursor.fetchone()[0]
    conn.close()

    return total_count


if __name__ == "__main__":
    init_db()
    count = seed_synthetic_data(overwrite=True)
    print(f"Successfully seeded SpendPulse database with {count} transactions.")
