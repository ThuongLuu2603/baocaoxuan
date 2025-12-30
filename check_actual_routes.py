"""
Script de xem tat ca cac route trong du lieu thuc te
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

ROUTE_PERFORMANCE_URL = 'https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=903527778#gid=903527778'

def load_actual_routes(sheet_url):
    """Doc ten route tu du lieu thuc te"""
    try:
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return []
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else '903527778'
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        
        df = pd.read_csv(io.StringIO(resp.content.decode('utf-8', errors='replace')), skipinitialspace=True)
        
        # Tim cot route
        route_col = None
        for col in df.columns:
            col_upper = str(col).upper()
            if 'TUYẾN TOUR' in col_upper or 'TUYEN TOUR' in col_upper:
                route_col = col
                break
        
        if route_col is None and len(df.columns) > 4:
            route_col = df.columns[4]  # Cot E
        
        if route_col is None:
            return []
        
        routes = df[route_col].dropna().astype(str).str.strip()
        return routes.unique().tolist()
    
    except Exception as e:
        print(f"Loi: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    print("="*80)
    print("TAT CA CAC ROUTE TRONG DU LIEU THUC TE")
    print("="*80)
    
    routes = load_actual_routes(ROUTE_PERFORMANCE_URL)
    
    print(f"\nTong so route: {len(routes)}")
    print(f"\nDanh sach route:")
    print("-" * 80)
    for i, route in enumerate(sorted(routes), 1):
        print(f"{i:3d}. '{route}' (len={len(route)}, repr={repr(route)})")
    
    # Tim cac route co chua "Chau", "Tay", "Nam", "My"
    print(f"\n{'='*80}")
    print("TIM CAC ROUTE CO CHUA: 'Chau', 'Tay', 'Nam', 'My'")
    print(f"{'='*80}\n")
    
    keywords = ['Chau', 'Châu', 'Tay', 'Tây', 'Nam', 'My', 'Mỹ', 'Úc', 'Á']
    for keyword in keywords:
        matches = [r for r in routes if keyword.upper() in r.upper()]
        if matches:
            print(f"\n'{keyword}':")
            for match in matches:
                print(f"  - '{match}'")

if __name__ == "__main__":
    main()

