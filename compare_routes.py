"""
Script de so sanh ten route giua du lieu thuc te va du lieu ke hoach
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

# URL cua Google Sheet ke hoach Xuan
PLAN_XUAN_URL = 'https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=212301737#gid=212301737'

# URL cua du lieu thuc te (can lay tu app.py)
ROUTE_PERFORMANCE_URL = 'https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=903527778#gid=903527778'

# Cac tuyen can kiem tra
TARGET_ROUTES = ['Châu Úc', 'Tây Á', 'Châu Mỹ', 'Nam Á', 'Châu Âu']

def load_plan_routes(sheet_url):
    """Doc ten route tu file ke hoach"""
    try:
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return []
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else None
        
        if not gid:
            return []
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        
        text = resp.content.decode('utf-8', errors='replace')
        lines = [line.rstrip('\r') for line in text.split('\n')]
        
        # Tim dong header
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
        
        if route_col is None:
            return []
        
        routes = df[route_col].dropna().astype(str).str.strip()
        return routes.unique().tolist()
    
    except Exception as e:
        print(f"Loi khi doc du lieu ke hoach: {e}")
        return []

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
        print(f"Loi khi doc du lieu thuc te: {e}")
        return []

def compare_routes(plan_routes, actual_routes, target_routes):
    """So sanh ten route"""
    print("="*80)
    print("SO SANH TEN ROUTE")
    print("="*80)
    
    print(f"\nTong so route trong ke hoach: {len(plan_routes)}")
    print(f"Tong so route trong du lieu thuc te: {len(actual_routes)}")
    
    print(f"\n{'='*80}")
    print("KIEM TRA CAC TUYEN:")
    print(f"{'='*80}\n")
    
    for target in target_routes:
        print(f"\nTuyen: '{target}'")
        print("-" * 80)
        
        # Tim trong ke hoach
        plan_match = None
        for route in plan_routes:
            if route.strip() == target.strip():
                plan_match = route
                break
        
        # Tim trong thuc te
        actual_match = None
        for route in actual_routes:
            if route.strip() == target.strip():
                actual_match = route
                break
        
        if plan_match and actual_match:
            print(f"[OK] Tim thay trong CA HAI:")
            print(f"   Ke hoach: '{plan_match}' (len={len(plan_match)}, repr={repr(plan_match)})")
            print(f"   Thuc te:  '{actual_match}' (len={len(actual_match)}, repr={repr(actual_match)})")
            if plan_match == actual_match:
                print(f"   [MATCH] Ten giong nhau!")
            else:
                print(f"   [NO MATCH] Ten khac nhau!")
        elif plan_match:
            print(f"[WARN] Chi tim thay trong KE HOACH:")
            print(f"   Ke hoach: '{plan_match}'")
            print(f"   Thuc te:  KHONG TIM THAY")
        elif actual_match:
            print(f"[WARN] Chi tim thay trong THUC TE:")
            print(f"   Ke hoach: KHONG TIM THAY")
            print(f"   Thuc te:  '{actual_match}'")
        else:
            print(f"[FAIL] KHONG tim thay trong CA HAI")
            
            # Tim gan dung
            print(f"\n   Tim gan dung trong ke hoach:")
            for route in plan_routes:
                if target.upper() in route.upper() or route.upper() in target.upper():
                    print(f"      - '{route}'")
            
            print(f"\n   Tim gan dung trong thuc te:")
            for route in actual_routes:
                if target.upper() in route.upper() or route.upper() in target.upper():
                    print(f"      - '{route}'")

def main():
    print("="*80)
    print("SO SANH TEN ROUTE GIUA DU LIEU THUC TE VA KE HOACH")
    print("="*80)
    
    # Doc du lieu
    print("\nDang doc du lieu ke hoach...")
    plan_routes = load_plan_routes(PLAN_XUAN_URL)
    
    print("\nDang doc du lieu thuc te...")
    actual_routes = load_actual_routes(ROUTE_PERFORMANCE_URL)
    
    # So sanh
    compare_routes(plan_routes, actual_routes, TARGET_ROUTES)
    
    print(f"\n{'='*80}")
    print("HOAN THANH")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

