"""Standalone CLI utility to populate or reset SpendPulse SQLite database
with realistic financial data, recurring salaries, and deliberate anomalies.
"""

import argparse
import sys
from src.database import init_db, seed_synthetic_data, get_transactions_df


def main():
    parser = argparse.ArgumentParser(description="SpendPulse Synthetic Data Seeder")
    parser.add_argument(
        "--months",
        type=int,
        default=14,
        help="Number of months of historical financial records to generate (default: 14)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Wipe existing transactions and seed fresh data",
    )
    args = parser.parse_args()

    print(f"Initializing SpendPulse SQLite database...")
    init_db()

    print(f"Generating financial records (Months: {args.months}, Overwrite: {args.overwrite})...")
    total_records = seed_synthetic_data(num_months=args.months, overwrite=args.overwrite)

    df = get_transactions_df()
    print("=" * 60)
    print(f"Database successfully ready!")
    print(f"Total transactions in database: {total_records}")
    if not df.empty:
        print(f"Timeline: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        income_sum = df[df['type'] == 'income']['amount'].sum()
        expense_sum = df[df['type'] == 'expense']['amount'].sum()
        print(f"Total Recorded Income:   ${income_sum:,.2f}")
        print(f"Total Recorded Expenses: ${expense_sum:,.2f}")
        print(f"Net Cumulative Savings:  ${(income_sum - expense_sum):,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
