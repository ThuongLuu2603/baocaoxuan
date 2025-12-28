"""
Script to debug route-level data for October 2025
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

# Filter for October 2025
start_date = '2025-10-01'
end_date = '2025-10-31'

print(f"\n{'='*80}")
print(f"ROUTE-LEVEL ANALYSIS FOR OCTOBER 2025")
print(f"{'='*80}")

# Filter by date
current_data = filter_data_by_date(tours_df, start_date, end_date)
confirmed_data = filter_confirmed_bookings(current_data)

print(f"\nTotal confirmed bookings in October: {len(confirmed_data)}")

# Check specific routes mentioned by user
target_routes = ['Singapore - Malaysia', 'Hàn Quốc', 'Châu Âu']

# First, let's see all unique route names
print(f"\n{'='*80}")
print("ALL UNIQUE ROUTES IN OCTOBER DATA:")
print(f"{'='*80}")
all_routes = sorted(confirmed_data['route'].unique())
for i, route in enumerate(all_routes, 1):
    count = len(confirmed_data[confirmed_data['route'] == route])
    print(f"{i}. {route} ({count} bookings)")

print(f"\n{'='*80}")
print("DETAILED ANALYSIS FOR SPECIFIC ROUTES:")
print(f"{'='*80}")

# Analyze each target route with different calculation methods
for route_name in target_routes:
    # Try to find matching route (case-insensitive, flexible matching)
    matching_routes = [r for r in all_routes if route_name.lower() in r.lower() or r.lower() in route_name.lower()]
    
    if not matching_routes:
        print(f"\n❌ Route '{route_name}' not found in data")
        continue
    
    for matched_route in matching_routes:
        route_data = confirmed_data[confirmed_data['route'] == matched_route].copy()
        
        print(f"\n{'─'*80}")
        print(f"Route: {matched_route}")
        print(f"{'─'*80}")
        print(f"Number of bookings: {len(route_data)}")
        
        # Method 1: RAW values (what user is calculating)
        revenue_raw = route_data['revenue'].sum() if 'revenue' in route_data.columns else 0
        profit_raw = route_data['gross_profit'].sum() if 'gross_profit' in route_data.columns else 0
        customers_raw = route_data['num_customers'].sum() if 'num_customers' in route_data.columns else 0
        
        print(f"\n📊 RAW VALUES (pre-cancellation):")
        print(f"   Doanh thu: {revenue_raw:,.0f} VND = {revenue_raw/1e9:,.1f} tỷ")
        print(f"   Lãi gộp: {profit_raw:,.0f} VND = {profit_raw/1e9:,.1f} tỷ")
        print(f"   Lượt khách: {customers_raw:,.0f}")
        
        # Method 2: EFFECTIVE values (after cancellation)
        if 'revenue_effective' in route_data.columns:
            revenue_eff = route_data['revenue_effective'].sum()
            profit_eff = route_data['gross_profit_effective'].sum() if 'gross_profit_effective' in route_data.columns else 0
            customers_eff = route_data['num_customers_effective'].sum() if 'num_customers_effective' in route_data.columns else 0
            
            print(f"\n📉 EFFECTIVE VALUES (post-cancellation):")
            print(f"   Doanh thu: {revenue_eff:,.0f} VND = {revenue_eff/1e9:,.1f} tỷ")
            print(f"   Lãi gộp: {profit_eff:,.0f} VND = {profit_eff/1e9:,.1f} tỷ")
            print(f"   Lượt khách: {customers_eff:,.0f}")
        
        # Cancellation info
        if 'cancel_count' in route_data.columns:
            total_cancels = route_data['cancel_count'].sum()
            print(f"\n🚫 Cancellations: {total_cancels} customers")
            
        # Show sample bookings
        print(f"\n📋 Sample bookings (first 5):")
        cols_show = ['booking_id', 'booking_date', 'num_customers', 'cancel_count', 'revenue', 'gross_profit']
        available = [c for c in cols_show if c in route_data.columns]
        print(route_data[available].head(5).to_string(index=False))

# User's expected values
print(f"\n{'='*80}")
print("USER'S EXPECTED VALUES:")
print(f"{'='*80}")
print("""
Singapore - Malaysia:
   Doanh thu: 20 tỷ
   Lãi gộp: 1.4 tỷ
   Lượt khách: 1,250

Hàn Quốc:
   Doanh thu: 30 tỷ
   Lãi gộp: 2.8 tỷ
   Lượt khách: 1,683

Châu Âu:
   Doanh thu: 103 tỷ
   Lãi gộp: 9 tỷ
   Lượt khách: 1,491
""")

# Summary by all routes
print(f"\n{'='*80}")
print("SUMMARY: ALL ROUTES (October 2025)")
print(f"{'='*80}")

route_summary = confirmed_data.groupby('route').agg({
    'revenue': 'sum',
    'gross_profit': 'sum',
    'num_customers': 'sum',
    'booking_id': 'count'
}).reset_index()

route_summary.columns = ['Route', 'Revenue', 'Gross Profit', 'Customers', 'Bookings']
route_summary['Revenue (B)'] = route_summary['Revenue'] / 1e9
route_summary['Profit (B)'] = route_summary['Gross Profit'] / 1e9
route_summary = route_summary.sort_values('Revenue', ascending=False)

print(route_summary[['Route', 'Revenue (B)', 'Profit (B)', 'Customers', 'Bookings']].to_string(index=False))
