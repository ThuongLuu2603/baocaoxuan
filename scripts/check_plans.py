from data_generator import load_or_generate_data
import pandas as pd

plan_url = 'https://docs.google.com/spreadsheets/d/1CljNuZ4WVNXGL7J111ZhVT9FPCVZDQsB6L5UHMgYeAc/edit?gid=322447784#gid=322447784'

print('Loading plan sheet...')
try:
    tours_df, plans_df, historical_df, meta = load_or_generate_data(spreadsheet_url=None, plan_spreadsheet_url=plan_url)
    print('Meta:', meta)
    print('\nplans_df shape:', None if plans_df is None else plans_df.shape)
    if plans_df is None or plans_df.empty:
        print('plans_df is empty')
    else:
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.max_colwidth', 200)
        print('\nSample of plans_df (top 30):')
        print(plans_df.head(30).to_string(index=False))
        print('\nRows where segment is not null (top 20):')
        segs = plans_df[plans_df['segment'].notna()]
        print(segs.head(20).to_string(index=False))
        print('\nCount per segment:')
        print(plans_df['segment'].value_counts(dropna=False))
        print('\nCompany-level rows containing TOÀN/TOAN:')
        comp = plans_df[plans_df['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN', na=False)]
        print(comp.head(10).to_string(index=False))
except Exception as e:
    print('Error loading or parsing plan sheet:', e)
    raise
