"""
Script de kiem tra du lieu ke hoach Xuan tu Google Sheet
Xem cac tuyen duoc luu voi ten nhu the nao
"""
import requests
import io
import re
import pandas as pd
import sys

# Set UTF-8 encoding for output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# URL cua Google Sheet ke hoach Xuan
PLAN_XUAN_URL = 'https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=212301737#gid=212301737'

# Cac tuyen can kiem tra
TARGET_ROUTES = ['Châu Âu', 'Châu Úc', 'Sing - Mã', 'Tây Á', 'Châu Mỹ']

def load_plan_data(sheet_url):
    """Doc du lieu tu Google Sheet"""
    try:
        # Chuyen doi URL thanh CSV export URL
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            print("Khong tim thay Sheet ID trong URL")
            return None
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else None
        
        if not gid:
            print("Khong tim thay GID trong URL")
            return None
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        # Tai CSV
        print(f"Dang tai du lieu tu: {csv_url}")
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        
        # Doc CSV
        text = resp.content.decode('utf-8', errors='replace')
        lines = [line.rstrip('\r') for line in text.split('\n')]
        
        # Tim dong header (dong 5, index 4) - chua "Nhom tuyen" va "Tuyen Tour"
        header_idx = None
        for i, line in enumerate(lines[:10]):
            line_upper = line.upper()
            if 'NHOM TUYEN' in line_upper or 'NHOM TUYẾN' in line_upper:
                if 'TUYẾN TOUR' in line_upper or 'TUYEN TOUR' in line_upper:
                    header_idx = i
                    break
        
        if header_idx is None:
            header_idx = 4 if len(lines) > 4 else 0
        
        print(f"Dong header: {header_idx + 1}")
        
        # Doc tu dong header
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])), skipinitialspace=True)
        
        if df.empty:
            print("DataFrame rong")
            return None
        
        print(f"\nTong so dong du lieu: {len(df)}")
        print(f"Cac cot: {list(df.columns[:10])}...")  # In 10 cot dau
        
        return df
    
    except Exception as e:
        print(f"Loi khi tai du lieu: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_route_column(df):
    """Tim cot chua ten tuyen"""
    route_col = None
    for col in df.columns:
        col_upper = str(col).upper()
        if 'TUYẾN TOUR' in col_upper or 'TUYEN TOUR' in col_upper:
            route_col = col
            break
    
    # Neu khong tim thay bang ten, dung vi tri cot (cot E, index 4)
    if route_col is None and len(df.columns) > 4:
        route_col = df.columns[1]  # Cot B (index 1): Tuyen Tour
    
    return route_col

def check_routes(df, target_routes):
    """Kiem tra cac tuyen trong du lieu"""
    route_col = find_route_column(df)
    
    if route_col is None:
        print("Khong tim thay cot Tuyen Tour")
        return
    
    print(f"\n{'='*80}")
    print(f"Cot Tuyen Tour: {route_col}")
    print(f"{'='*80}\n")
    
    # Lay tat ca cac gia tri route (loai bo NaN)
    all_routes = df[route_col].dropna().astype(str).str.strip()
    
    print(f"Tong so tuyen trong file: {len(all_routes.unique())}")
    print(f"\nTat ca cac tuyen trong file:")
    print("-" * 80)
    for i, route in enumerate(sorted(all_routes.unique()), 1):
        print(f"{i:3d}. {route}")
    
    print(f"\n{'='*80}")
    print("KIEM TRA CAC TUYEN CAN TIM:")
    print(f"{'='*80}\n")
    
    # Kiem tra tung tuyen
    for target_route in target_routes:
        print(f"\nTim tuyen: '{target_route}'")
        print("-" * 80)
        
        # Tim chinh xac
        exact_matches = all_routes[all_routes == target_route]
        if len(exact_matches) > 0:
            print(f"[OK] Tim thay CHINH XAC: '{target_route}'")
            print(f"   So lan xuat hien: {len(exact_matches)}")
            # Hien thi cac dong chua tuyen nay
            indices = exact_matches.index.tolist()
            print(f"   O cac dong: {indices[:10]}...")  # Hien thi 10 dong dau
            
            # Hien thi thong tin chi tiet
            for idx in indices[:5]:  # Chi hien thi 5 dong dau
                row = df.loc[idx]
                print(f"\n   Dong {idx}:")
                print(f"   - Nhom tuyen: {row.get('Nhom tuyen', 'N/A')}")
                print(f"   - Tuyen Tour: {row.get(route_col, 'N/A')}")
                # Tim cac cot LK, DT, LG
                for col in df.columns:
                    if str(col).upper() == 'LK' or (str(col).upper().startswith('LK') and '.' not in str(col)):
                        print(f"   - LK: {row.get(col, 'N/A')}")
                        break
                for col in df.columns:
                    if 'DT (TR.D)' in str(col).upper() or 'DT(TR.D)' in str(col).upper():
                        if '.' not in str(col) or str(col).endswith('.1') == False:
                            print(f"   - DT: {row.get(col, 'N/A')}")
                            break
                for col in df.columns:
                    if 'LG (TR.D)' in str(col).upper() or 'LG(TR.D)' in str(col).upper():
                        if '.' not in str(col) or str(col).endswith('.1') == False:
                            print(f"   - LG: {row.get(col, 'N/A')}")
                            break
        else:
            print(f"[FAIL] KHONG tim thay chinh xac: '{target_route}'")
        
        # Tim gan dung (case-insensitive)
        target_upper = target_route.upper()
        similar_matches = all_routes[all_routes.str.upper().str.contains(target_upper, na=False, regex=False)]
        if len(similar_matches) > 0 and len(exact_matches) == 0:
            print(f"[WARN] Tim thay TUONG TU (case-insensitive):")
            for match in similar_matches.unique():
                print(f"   - '{match}'")
        
        # Tim voi cac bien the
        variants = []
        if 'CHÂU ÂU' in target_upper or 'CHAU AU' in target_upper:
            variants = ['CHÂU ÂU', 'CHAU AU', 'EUROPE', 'CHÂU ÂU', 'CHAUAU']
        elif 'CHÂU ÚC' in target_upper or 'CHAU UC' in target_upper:
            variants = ['CHÂU ÚC', 'CHAU UC', 'AUSTRALIA', 'CHÂU ÚC', 'CHAUUC']
        elif 'SING' in target_upper and ('MÃ' in target_upper or 'MA' in target_upper):
            variants = ['SING - MÃ', 'SING - MA', 'SING-MÃ', 'SING-MA', 'SINGAPORE MALAYSIA', 'SINGAPORE - MALAYSIA']
        elif 'TÂY Á' in target_upper or 'TAY A' in target_upper:
            variants = ['TÂY Á', 'TAY A', 'WEST ASIA', 'TÂY Á', 'TAYA']
        elif 'CHÂU MỸ' in target_upper or 'CHAU MY' in target_upper:
            variants = ['CHÂU MỸ', 'CHAU MY', 'AMERICA', 'CHÂU MỸ', 'CHAUMY']
        
        if variants:
            print(f"\n   Tim voi cac bien the:")
            for variant in variants:
                variant_matches = all_routes[all_routes.str.upper().str.contains(variant.upper(), na=False, regex=False)]
                if len(variant_matches) > 0:
                    for match in variant_matches.unique():
                        print(f"   [OK] '{match}' (bien the: {variant})")

def main():
    print("="*80)
    print("KIEM TRA DU LIEU KE HOACH XUAN")
    print("="*80)
    print(f"\nURL: {PLAN_XUAN_URL}")
    print(f"Cac tuyen can kiem tra: {', '.join(TARGET_ROUTES)}")
    print("\n")
    
    # Tai du lieu
    df = load_plan_data(PLAN_XUAN_URL)
    
    if df is None:
        print("Khong the tai du lieu")
        return
    
    # Kiem tra cac tuyen
    check_routes(df, TARGET_ROUTES)
    
    print(f"\n{'='*80}")
    print("HOAN THANH")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
