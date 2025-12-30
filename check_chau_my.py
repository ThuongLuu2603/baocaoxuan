"""
Script de kiem tra chi tiet du lieu ke hoach cho 'Chau My'
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

import requests
import re
import pandas as pd
import unicodedata

PLAN_XUAN_URL = 'https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=212301737#gid=212301737'

def normalize_unicode(text):
    """Normalize Unicode"""
    if pd.isna(text):
        return ''
    text_str = str(text).strip()
    return unicodedata.normalize('NFC', text_str)

def load_plan_data(sheet_url):
    """Doc du lieu tu Google Sheet"""
    try:
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return None
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else None
        
        if not gid:
            return None
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        
        text = resp.content.decode('utf-8', errors='replace')
        lines = [line.rstrip('\r') for line in text.split('\n')]
        
        header_idx = None
        for i, line in enumerate(lines[:10]):
            line_upper = line.upper()
            if 'NHOM TUYEN' in line_upper or 'NHOM TUYẾN' in line_upper:
                if 'TUYẾN TOUR' in line_upper or 'TUYEN TOUR' in line_upper:
                    header_idx = i
                    break
        
        if header_idx is None:
            header_idx = 4 if len(lines) > 4 else 0
        
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])), skipinitialspace=True)
        
        # Tim cot route
        route_col = None
        for col in df.columns:
            col_upper = str(col).upper()
            if 'TUYẾN TOUR' in col_upper or 'TUYEN TOUR' in col_upper:
                route_col = col
                break
        
        if route_col is None and len(df.columns) > 1:
            route_col = df.columns[1]
        
        # Tim cot DT (tr.d) - cot C (index 2) cho Cong ty
        dt_col = None
        if len(df.columns) > 3:
            dt_col = df.columns[3]  # Cot D (index 3): DT (tr.d)
        
        return df, route_col, dt_col
    
    except Exception as e:
        print(f"Loi: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    print("="*80)
    print("KIEM TRA DU LIEU KE HOACH CHO 'CHAU MY'")
    print("="*80)
    
    df, route_col, dt_col = load_plan_data(PLAN_XUAN_URL)
    
    if df is None or route_col is None:
        print("Khong the doc du lieu")
        return
    
    print(f"\nCot route: {route_col}")
    print(f"Cot DT: {dt_col}")
    
    # Tim tat ca route co chua "My" hoac "Mỹ"
    print(f"\n{'='*80}")
    print("TAT CA CAC ROUTE CO CHUA 'MY' HOAC 'MỸ':")
    print(f"{'='*80}\n")
    
    routes = df[route_col].dropna().astype(str).str.strip()
    
    for route in routes.unique():
        route_upper = route.upper()
        if 'MY' in route_upper or 'MỸ' in route_upper or 'Mỹ' in route_upper:
            print(f"\nRoute: '{route}'")
            print(f"  - Normalized: '{normalize_unicode(route)}'")
            print(f"  - Len: {len(route)}, Repr: {repr(route)}")
            
            # Tim dong chua route nay
            route_rows = df[df[route_col].astype(str).str.strip() == route]
            if not route_rows.empty:
                for idx, row in route_rows.iterrows():
                    print(f"  - Dong {idx}:")
                    if dt_col:
                        dt_value = row.get(dt_col, 'N/A')
                        print(f"    DT (tr.d): {dt_value} (type: {type(dt_value)})")
                    # In tat ca cac cot
                    print(f"    Tat ca cac cot:")
                    for col in df.columns:
                        val = row.get(col, 'N/A')
                        if pd.notna(val) and str(val).strip() != '':
                            print(f"      {col}: {val}")
    
    # Tim route "Chau My" chinh xac
    print(f"\n{'='*80}")
    print("TIM ROUTE 'CHAU MY' CHINH XAC:")
    print(f"{'='*80}\n")
    
    target_routes = ['Châu Mỹ', 'Châu My', 'Mỹ', 'My']
    for target in target_routes:
        target_normalized = normalize_unicode(target)
        print(f"\nTim: '{target}' (normalized: '{target_normalized}')")
        
        # Tim chinh xac
        exact_matches = df[df[route_col].astype(str).str.strip() == target]
        if not exact_matches.empty:
            print(f"  [OK] Tim thay chinh xac!")
            for idx, row in exact_matches.iterrows():
                print(f"  Dong {idx}:")
                if dt_col:
                    dt_value = row.get(dt_col, 'N/A')
                    print(f"    DT (tr.d): {dt_value}")
        else:
            print(f"  [FAIL] Khong tim thay chinh xac")
            
            # Tim gan dung
            similar = df[df[route_col].astype(str).str.upper().str.contains(target.upper(), na=False)]
            if not similar.empty:
                print(f"  [WARN] Tim thay gan dung:")
                for idx, row in similar.iterrows():
                    route_val = row.get(route_col, 'N/A')
                    print(f"    - '{route_val}'")

if __name__ == "__main__":
    main()

