"""
Script to debug October 2025 data discrepancies
"""
import sys
sys.path.insert(0, '/workspaces/TourDashboardVer2')

import pandas as pd
from data_generator import load_or_generate_data
from utils import filter_data_by_date, filter_confirmed_bookings

# Load data
print("Loading data...")
DEFAULT_DATANET_URL = 'https://docs.google.com/spreadsheets/d/1Mmx7FS-BjHcnitmfiT4xRQ7CFZmBrl8tsV5IySdNYOw/edit?gid=29056776#gid=29056776'
DEFAULT_PLAN_URL = 'https://docs.google.com/spreadsheets/d/1mQYyJpdarm50syGxtJ6dLLJw99CAT4wCnZqy8qCp4JI/edit?gid=322447784#gid=322447784'

tours_df, plans_df, historical_df, data_meta = load_or_generate_data(DEFAULT_DATANET_URL, DEFAULT_PLAN_URL)

print(f"\nTotal rows in tours_df: {len(tours_df)}")
print(f"Columns: {tours_df.columns.tolist()}")

# Filter for October 2025
start_date = '2025-10-01'
end_date = '2025-10-31'

print(f"\n{'='*80}")
print(f"Filtering for October 2025: {start_date} to {end_date}")
print(f"{'='*80}")

# Check booking_date column
if 'booking_date' in tours_df.columns:
    tours_df['booking_date'] = pd.to_datetime(tours_df['booking_date'], errors='coerce')
    print(f"\nDate range in data: {tours_df['booking_date'].min()} to {tours_df['booking_date'].max()}")

# Filter by date
current_data = filter_data_by_date(tours_df, start_date, end_date)
print(f"\nRows after date filter: {len(current_data)}")

# Check status distribution
if 'status' in current_data.columns:
    print(f"\nStatus distribution:")
    print(current_data['status'].value_counts())

# Filter confirmed bookings
confirmed_data = filter_confirmed_bookings(current_data)
print(f"\nRows after confirmed filter: {len(confirmed_data)}")

# Calculate metrics using different methods
print(f"\n{'='*80}")
print("CALCULATION COMPARISON")
print(f"{'='*80}")

# Method 1: Using effective columns (what dashboard uses)
if 'revenue_effective' in confirmed_data.columns:
    revenue_effective = confirmed_data['revenue_effective'].sum()
    print(f"\n1. Revenue (effective, post-cancellation): {revenue_effective:,.0f} VND = {revenue_effective/1e9:,.1f} billion")
else:
    print("\nNo revenue_effective column found")

if 'gross_profit_effective' in confirmed_data.columns:
    profit_effective = confirmed_data['gross_profit_effective'].sum()
    print(f"   Gross Profit (effective): {profit_effective:,.0f} VND = {profit_effective/1e9:,.1f} billion")
else:
    print("No gross_profit_effective column found")

if 'num_customers_effective' in confirmed_data.columns:
    customers_effective = confirmed_data['num_customers_effective'].sum()
    print(f"   Customers (effective): {customers_effective:,.0f}")
else:
    print("No num_customers_effective column found")

# Method 2: Using raw columns (what user might be calculating)
if 'revenue' in confirmed_data.columns:
    revenue_raw = confirmed_data['revenue'].sum()
    print(f"\n2. Revenue (raw, pre-cancellation): {revenue_raw:,.0f} VND = {revenue_raw/1e9:,.1f} billion")

if 'gross_profit' in confirmed_data.columns:
    profit_raw = confirmed_data['gross_profit'].sum()
    print(f"   Gross Profit (raw): {profit_raw:,.0f} VND = {profit_raw/1e9:,.1f} billion")

if 'num_customers' in confirmed_data.columns:
    customers_raw = confirmed_data['num_customers'].sum()
    print(f"   Customers (raw): {customers_raw:,.0f}")

# Check cancellation impact
if 'cancel_count' in confirmed_data.columns:
    total_cancels = confirmed_data['cancel_count'].sum()
    print(f"\n3. Cancellation Impact:")
    print(f"   Total cancelled customers: {total_cancels:,.0f}")
    if 'num_customers' in confirmed_data.columns:
        cancel_rate = (total_cancels / confirmed_data['num_customers'].sum() * 100) if confirmed_data['num_customers'].sum() > 0 else 0
        print(f"   Overall cancellation rate: {cancel_rate:.2f}%")

# Method 3: All data (including non-confirmed)
print(f"\n4. All October data (including cancelled/postponed):")
all_revenue = current_data['revenue'].sum() if 'revenue' in current_data.columns else 0
all_profit = current_data['gross_profit'].sum() if 'gross_profit' in current_data.columns else 0
all_customers = current_data['num_customers'].sum() if 'num_customers' in current_data.columns else 0
print(f"   Revenue: {all_revenue:,.0f} VND = {all_revenue/1e9:,.1f} billion")
print(f"   Gross Profit: {all_profit:,.0f} VND = {all_profit/1e9:,.1f} billion")
print(f"   Customers: {all_customers:,.0f}")

# Show sample of data
print(f"\n{'='*80}")
print("SAMPLE DATA (first 5 rows)")
print(f"{'='*80}")
cols_to_show = ['booking_id', 'booking_date', 'status', 'num_customers', 'cancel_count', 
                'num_customers_effective', 'revenue', 'revenue_effective', 'gross_profit', 'gross_profit_effective']
available_cols = [c for c in cols_to_show if c in confirmed_data.columns]
print(confirmed_data[available_cols].head(5).to_string())

print(f"\n{'='*80}")
print("SUMMARY: Expected vs Actual")
print(f"{'='*80}")
print(f"User's calculation (expected):")
print(f"  Revenue: 682 billion")
print(f"  Gross Profit: 57.4 billion")
print(f"  Customers: 81,607")
print(f"\nDashboard showing (actual):")
print(f"  Revenue: 677.1 billion")
print(f"  Gross Profit: 53.6 billion")
print(f"  Customers: 81,610")
print(f"\nThis script's calculation:")
if 'revenue_effective' in confirmed_data.columns:
    print(f"  Revenue (effective): {revenue_effective/1e9:,.1f} billion")
    print(f"  Gross Profit (effective): {profit_effective/1e9:,.1f} billion")
    print(f"  Customers (effective): {customers_effective:,.0f}")
