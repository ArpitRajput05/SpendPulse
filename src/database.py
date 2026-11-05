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
    "Rent & Housing": 25000.0,
    "Groceries & Kirana": 12000.0,
    "Food & Dining": 7500.0,
    "Utilities & Bills": 4500.0,
    "Travel & Commute": 5000.0,
    "Shopping & E-Com": 8000.0,
    "Entertainment & OTT": 3000.0,
    "Healthcare & Fitness": 3500.0,
    "Subscriptions": 1500.0,
    "Investments & SIP": 15000.0,
}


def seed_synthetic_data(num_months: int = 14, overwrite: bool = False) -> int:
    """
    Populates SQLite database with rich, realistic Indian financial transactions
    and default category budgets.
    
    Includes realistic monthly tech/corporate salaries in INR (₹), quarterly bonuses,
    Diwali festival surges, everyday UPI transactions (Blinkit, Swiggy, Zomato, Zepto),
    and deliberate anomalies for machine learning detection.
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

        # Income: Monthly Salary credited on 1st of month
        day_1 = datetime(year, month, 1)
        base_salary = 125000.0 + random.uniform(-2000, 3000)
        transactions.append((
            day_1.strftime("%Y-%m-%d"),
            "Income",
            "income",
            round(base_salary, 2),
            "Monthly Salary Credit - Tech Corp India",
            "Net Banking",
        ))

        # Freelance Consulting / Side Gig (occasional, around 15th-20th)
        if random.random() < 0.45:
            freelance_day = datetime(year, month, random.randint(12, 24))
            transactions.append((
                freelance_day.strftime("%Y-%m-%d"),
                "Income",
                "income",
                round(random.uniform(15000, 32000), 2),
                "Consulting & UI Design Freelance Project",
                "UPI (GPay / PhonePe / Paytm)",
            ))

        # Diwali / Festive Performance Bonus (Oct/Nov)
        if month in [10, 11] and random.random() < 0.6:
            bonus_day = datetime(year, month, 22)
            transactions.append((
                bonus_day.strftime("%Y-%m-%d"),
                "Income",
                "income",
                45000.0,
                "Diwali Festive Performance Bonus",
                "Net Banking",
            ))

        # Fixed Monthly Expenses
        # 1. House Rent & Maintenance on 2nd of month
        rent_date = datetime(year, month, 2)
        transactions.append((
            rent_date.strftime("%Y-%m-%d"),
            "Rent & Housing",
            "expense",
            25000.0,
            "Apartment Rent & Society Maintenance",
            "Net Banking",
        ))

        # 2. Monthly SIP & Mutual Fund Auto-debit on 5th
        sip_date = datetime(year, month, 5)
        transactions.append((
            sip_date.strftime("%Y-%m-%d"),
            "Investments & SIP",
            "expense",
            15000.0,
            "Zerodha Coin / Groww Nifty 50 Index Fund SIP",
            "Net Banking",
        ))

        # 3. Utilities & Broadband Bills on 7th
        util_date = datetime(year, month, 7)
        seasonal_ac_mult = 1.35 if month in [4, 5, 6, 7] else 1.0  # Summer AC electricity bills
        util_amount = round(random.uniform(3800, 4800) * seasonal_ac_mult, 2)
        transactions.append((
            util_date.strftime("%Y-%m-%d"),
            "Utilities & Bills",
            "expense",
            util_amount,
            "Electricity (BESCOM/Tata Power), Gas & JioFiber Wi-Fi",
            "UPI (GPay / PhonePe / Paytm)",
        ))

        # 4. OTT & Digital Subscriptions on 10th
        sub_date = datetime(year, month, 10)
        transactions.append((
            sub_date.strftime("%Y-%m-%d"),
            "Subscriptions",
            "expense",
            1499.0,
            "Netflix, Disney+ Hotstar, Spotify & YouTube Premium",
            "Credit Card",
        ))

        # Move to next month
        if month == 12:
            cur_month_start = datetime(year + 1, 1, 1)
        else:
            cur_month_start = datetime(year, month + 1, 1)

    # Everyday Indian consumer spending vendors
    grocery_vendors = ["Blinkit 10-Min Delivery", "Zepto Quick Delivery", "Instamart", "DMart Supermarket", "Local Subzi & Kirana Mandi", "BigBasket"]
    dining_vendors = ["Swiggy Gourmet", "Zomato Delivery", "Chai Point & Samosa", "Haldiram's Sweets", "Biryani by Kilo", "Third Wave Coffee Roasters", "Local Dosa & Darshini Corner"]
    transport_vendors = ["Uber Auto / Premier", "Ola Cabs", "Metro Card Smart Recharge", "Indian Oil Fuel / Petrol", "FASTag Toll Auto-debit"]
    shopping_vendors = ["Amazon India", "Flipkart Sale", "Myntra Fashion", "Ajio Trends", "Zara Store", "Nykaa Cosmetics"]
    entertainment_vendors = ["BookMyShow Movie Tickets (PVR Inox)", "Gaming & Steam Pass", "Weekend Stand-up Comedy Show", "Bowling & Smaaash"]
    health_vendors = ["Apollo Pharmacy 24/7", "Tata 1mg Medicine Order", "Cult.fit Gym & Fitness", "Practo Specialist Consult"]

    days_range = (end_date - start_date).days

    for d in range(days_range):
        current_day = start_date + timedelta(days=d)
        date_str = current_day.strftime("%Y-%m-%d")
        month = current_day.month
        is_weekend = current_day.weekday() >= 5

        # Groceries & Kirana: Quick commerce is very frequent in India (3-4 times a week)
        if random.random() < 0.40:
            vendor = random.choice(grocery_vendors)
            amount = round(random.uniform(220, 1850), 2)
            payment = random.choices(["UPI (GPay / PhonePe / Paytm)", "Credit Card", "Debit Card"], weights=[0.70, 0.20, 0.10])[0]
            transactions.append((
                date_str,
                "Groceries & Kirana",
                "expense",
                amount,
                f"Groceries & Essentials ({vendor})",
                payment,
            ))

        # Food & Dining: Swiggy/Zomato/Chai
        dining_prob = 0.55 if is_weekend else 0.30
        if random.random() < dining_prob:
            vendor = random.choice(dining_vendors)
            amount = round(random.uniform(150, 1250), 2)
            payment = random.choices(["UPI (GPay / PhonePe / Paytm)", "Credit Card"], weights=[0.75, 0.25])[0]
            transactions.append((
                date_str,
                "Food & Dining",
                "expense",
                amount,
                f"Meal / Chai at {vendor}",
                payment,
            ))

        # Commute & Travel (Auto, Metro, Petrol)
        if random.random() < 0.35:
            vendor = random.choice(transport_vendors)
            amount = round(random.uniform(80, 850), 2)
            transactions.append((
                date_str,
                "Travel & Commute",
                "expense",
                amount,
                f"Commute ({vendor})",
                "UPI (GPay / PhonePe / Paytm)",
            ))

        # Shopping: Spikes during festive season (Diwali / Great Indian Festival in Oct-Nov)
        shop_prob = 0.28 if month in [10, 11] else 0.12
        if random.random() < shop_prob:
            vendor = random.choice(shopping_vendors)
            base_amt = random.uniform(850, 4200)
            if month in [10, 11]:
                base_amt *= 1.5
            transactions.append((
                date_str,
                "Shopping & E-Com",
                "expense",
                round(base_amt, 2),
                f"Order at {vendor}",
                random.choices(["Credit Card", "UPI (GPay / PhonePe / Paytm)"], weights=[0.60, 0.40])[0],
            ))

        # Entertainment
        if random.random() < 0.10:
            vendor = random.choice(entertainment_vendors)
            transactions.append((
                date_str,
                "Entertainment & OTT",
                "expense",
                round(random.uniform(400, 1800), 2),
                f"Recreation - {vendor}",
                "UPI (GPay / PhonePe / Paytm)",
            ))

        # Healthcare & Fitness
        if random.random() < 0.08:
            vendor = random.choice(health_vendors)
            transactions.append((
                date_str,
                "Healthcare & Fitness",
                "expense",
                round(random.uniform(350, 2200), 2),
                f"Wellness - {vendor}",
                random.choice(["UPI (GPay / PhonePe / Paytm)", "Credit Card"]),
            ))

        # Weekend Gateways / Outstation Trips (Goa, Coorg, Rishikesh)
        if month in [5, 6, 12, 1] and random.random() < 0.03:
            transactions.append((
                date_str,
                "Travel & Commute",
                "expense",
                round(random.uniform(4500, 14000), 2),
                "MakeMyTrip - Weekend Getaway Resort & IRCTC Express",
                "Credit Card",
            ))

    # REALISTIC DELIBERATE ANOMALIES (for Scikit-Learn Anomaly Detection / IsolationForest validation)
    deliberate_anomalies = [
        ("2025-05-14", "Travel & Commute", "expense", 46500.00, "MakeMyTrip - Goa Flight & 5-Star Beach Villa Package", "Credit Card"),
        ("2025-08-19", "Healthcare & Fitness", "expense", 34800.00, "Apollo Hospitals - Emergency Dental Surgery & MRI Scan", "Credit Card"),
        ("2025-10-24", "Shopping & E-Com", "expense", 68500.00, "Tanishq Jewellery - Festive Dhanteras 24K Gold Coin & Gift", "Credit Card"),
        ("2025-11-12", "Shopping & E-Com", "expense", 84999.00, "Flipkart Big Billion Days - Apple iPhone 15 Pro Max", "Credit Card"),
        ("2026-01-18", "Food & Dining", "expense", 5850.00, "Swiggy - Team Milestone Dinner / Suspected Accidental Double Bill", "UPI (GPay / PhonePe / Paytm)"),
        ("2026-02-10", "Travel & Commute", "expense", 24500.00, "Maruti Suzuki Arena - Full Engine Clutch & Brake Overhaul", "Debit Card"),
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
