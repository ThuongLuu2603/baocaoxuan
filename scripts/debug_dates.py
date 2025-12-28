from data_generator import load_or_generate_data
import pandas as pd

DEFAULT_DATANET_URL = 'https://docs.google.com/spreadsheets/d/1CljNuZ4WVNXGL7J111ZhVT9FPCVZDQsB6L5UHMgYeAc/edit?gid=158175105#gid='

print('Loading data from Google Sheet...')
try:
    tours_df, plans_df, historical_df, meta = load_or_generate_data(DEFAULT_DATANET_URL)
except Exception as e:
    try:
        tours_df, plans_df, historical_df = load_or_generate_data(DEFAULT_DATANET_URL)
        meta = {}
    except Exception as e2:
        print('Failed to load data:', e2)
        raise

print('Loaded tours rows:', len(tours_df))

if 'booking_date' in tours_df.columns:
    tours_df['booking_date'] = pd.to_datetime(tours_df['booking_date'], errors='coerce')
if 'end_date' in tours_df.columns:
    tours_df['end_date'] = pd.to_datetime(tours_df['end_date'], errors='coerce')

print('booking_date column present:', 'booking_date' in tours_df.columns)
print('booking_date min/max:', tours_df['booking_date'].min(), tours_df['booking_date'].max())
print('end_date min/max:', tours_df['end_date'].min() if 'end_date' in tours_df.columns else None, tours_df['end_date'].max() if 'end_date' in tours_df.columns else None)

start = pd.to_datetime('2025-11-01')
end = pd.to_datetime('2025-11-30 23:59:59')
mask_nov2025 = (tours_df['booking_date'] >= start) & (tours_df['booking_date'] <= end)
print('Bookings with start date in Nov 2025:', int(mask_nov2025.sum()))

mask_nov2024 = (tours_df['booking_date'] >= pd.to_datetime('2024-11-01')) & (tours_df['booking_date'] <= pd.to_datetime('2024-11-30 23:59:59'))
print('Bookings with start date in Nov 2024:', int(mask_nov2024.sum()))

print('\nSample rows for Nov 2024/Nov 2025 (booking_date, end_date, route, revenue):')
print(tours_df.loc[mask_nov2024 | mask_nov2025, ['booking_date','end_date','route','revenue']].head(20).to_string(index=False))

if 'end_date' in tours_df.columns:
    mask_end_nov2025 = (tours_df['end_date'] >= start) & (tours_df['end_date'] <= end)
    print('Bookings with END date in Nov 2025:', int(mask_end_nov2025.sum()))

    # Rows where END date is in Nov 2025 but START date is NOT (these would be included
    # by an overlap-based inclusion rule but are excluded by start-date-only logic)
    mask_end_only = mask_end_nov2025 & (~mask_nov2025)
    print('Bookings with END in Nov 2025 but START NOT in Nov 2025 (end-only):', int(mask_end_only.sum()))
    if mask_end_only.any():
        print('\nSample of bookings included by end-date overlap but excluded by start-date-only filter:')
        print(tours_df.loc[mask_end_only, ['booking_id','booking_date','end_date','route','revenue']].head(10).to_string(index=False))

print('\nmeta keys:', list(meta.keys()) if isinstance(meta, dict) else meta)
print('parsed_rows (meta):', meta.get('parsed_rows') if isinstance(meta, dict) else None)
