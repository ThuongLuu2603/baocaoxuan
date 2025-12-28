"""
Debug script to check what data is being passed to the chart
"""
import sys
sys.path.insert(0, '/workspaces/TourDashboardVer2')

import pandas as pd
from data_generator import load_or_generate_data
from utils import filter_data_by_date, filter_confirmed_bookings, get_top_routes

# Load data
print("Loading data...")
DEFAULT_DATANET_URL = 'https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=903527778#gid=903527778'
DEFAULT_PLAN_URL = 'https://docs.google.com/spreadsheets/d/1mQYyJpdarm50syGxtJ6dLLJw99CAT4wCnZqy8qCp4JI/edit?gid=322447784#gid=322447784'

tours_df, plans_df, historical_df, data_meta = load_or_generate_data(DEFAULT_DATANET_URL, DEFAULT_PLAN_URL)

# Simulate app.py filtering
start_date = '2025-10-01'
end_date = '2025-10-31'

# Filter by date (like in app.py line 466)
filtered_tours = filter_data_by_date(tours_df, start_date, end_date)

print(f"\n{'='*80}")
print(f"SIMULATING APP.PY DATA FLOW")
print(f"{'='*80}")
print(f"Period: {start_date} to {end_date}")
print(f"Filtered tours: {len(filtered_tours)} rows")

# Get top routes like in app.py (line 910-912)
top_revenue = get_top_routes(filtered_tours, 10, 'revenue')

print(f"\n{'='*80}")
print(f"TOP 10 ROUTES BY REVENUE")
print(f"{'='*80}")
print(top_revenue[['route', 'revenue', 'num_customers', 'gross_profit']].to_string(index=False))

# Use top_revenue directly (NEW LOGIC - matching app.py fix)
if not top_revenue.empty:
    df_merged_top10 = top_revenue[['route', 'revenue', 'num_customers', 'gross_profit']].copy()
    df_merged_top10['route'] = df_merged_top10['route'].fillna('').astype(str).str.strip()
    df_merged_top10 = df_merged_top10[df_merged_top10['route'] != ''].copy()
    df_merged_top10 = df_merged_top10.sort_values('revenue', ascending=False)

print(f"\n{'='*80}")
print(f"MERGED DATA (df_merged_top10) - PASSED TO CHART")
print(f"{'='*80}")
print(df_merged_top10.to_string(index=False))

print(f"\n{'='*80}")
print(f"CHECK SPECIFIC ROUTES")
print(f"{'='*80}")

target_routes = ['Singapore - Malaysia', 'Hàn Quốc', 'Châu Âu']
for route in target_routes:
    row = df_merged_top10[df_merged_top10['route'] == route]
    if not row.empty:
        print(f"\n{route}:")
        print(f"   Revenue: {row['revenue'].values[0]:,.0f} ({row['revenue'].values[0]/1e9:,.1f} tỷ)")
        print(f"   Gross Profit: {row['gross_profit'].values[0]:,.0f} ({row['gross_profit'].values[0]/1e9:,.1f} tỷ)")
        print(f"   Customers: {row['num_customers'].values[0]:,.0f}")
    else:
        print(f"\n{route}: NOT FOUND in top 10")
