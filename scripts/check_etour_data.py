"""
Script để kiểm tra dữ liệu từ Google Sheet ETOUR
Kiểm tra sum cột F (Doanh số đã bán) cho:
- KHU VỰC KINH DOANH chứa "Mien Bac"
- Tuyến tour = "Miền Bắc"
- GIAI ĐOẠN = "KM XUÂN"
"""

import pandas as pd
import requests
import io
import re

def load_etour_data(sheet_url):
    """Load dữ liệu từ Google Sheet"""
    try:
        # Chuyển đổi URL thành CSV export URL
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return pd.DataFrame()
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else '2069863260'
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        # Tải CSV
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        
        text = resp.content.decode('utf-8', errors='replace')
        lines = text.split('\n')
        
        # Header ở dòng 5 (index 4)
        header_idx = 4
        if len(lines) <= header_idx:
            return pd.DataFrame()
        
        # Đọc từ dòng header trở đi
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])), skipinitialspace=True)
        
        if df.empty:
            return pd.DataFrame()
        
        # Chuẩn hóa tên cột
        df.columns = [col.strip() for col in df.columns]
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def check_mien_bac_data(df):
    """Kiểm tra dữ liệu Miền Bắc"""
    print("=" * 80)
    print("KIEM TRA DU LIEU MIEN BAC - KM XUAN")
    print("=" * 80)
    
    # Tìm các cột
    region_col = None
    route_group_col = None
    period_col = None
    revenue_col = None
    
    for col in df.columns:
        col_upper = str(col).upper()
        if 'KHU VỰC KINH DOANH' in col_upper or 'KHU VUC KINH DOANH' in col_upper:
            region_col = col
        elif 'TUYẾN TOUR' in col_upper or 'TUYEN TOUR' in col_upper:
            if len(df.columns) > 10 and col == df.columns[10]:
                route_group_col = col
        elif 'GIAI ĐOẠN' in col_upper or 'GIAI DOAN' in col_upper:
            period_col = col
        elif 'DOANH SỐ ĐÃ BÁN' in col_upper or 'DOANH SO DA BAN' in col_upper:
            revenue_col = col
    
    # Nếu không tìm thấy bằng tên, dùng vị trí cột
    if region_col is None and len(df.columns) > 9:
        region_col = df.columns[9]  # Cột J
    if route_group_col is None and len(df.columns) > 10:
        route_group_col = df.columns[10]  # Cột K
    if period_col is None and len(df.columns) > 12:
        period_col = df.columns[12]  # Cột M
    if revenue_col is None and len(df.columns) > 5:
        revenue_col = df.columns[5]  # Cột F
    
    print(f"\nCac cot duoc su dung:")
    print(f"  - KHU VUC KINH DOANH: {region_col}")
    print(f"  - Tuyen tour: {route_group_col}")
    print(f"  - GIAI DOAN: {period_col}")
    print(f"  - Doanh so da ban: {revenue_col}")
    
    if not all([region_col, route_group_col, period_col, revenue_col]):
        print("\nERROR: KHONG TIM THAY DU CAC COT CAN THIET!")
        return
    
    # Parse giá trị revenue
    def parse_value(val):
        if pd.isna(val) or val == '' or str(val).strip() == '-' or str(val).strip().upper() == 'NAN':
            return 0
        val_str = str(val).strip().replace(',', '').replace('"', '')
        # Xử lý số có dấu chấm làm dấu phân cách hàng nghìn (ví dụ: "6.995" = 6995)
        # Nếu có dấu chấm và không có dấu phẩy, coi dấu chấm là dấu phân cách hàng nghìn
        if '.' in val_str and ',' not in val_str:
            # Đếm số dấu chấm - nếu có nhiều hơn 1, có thể là dấu phân cách hàng nghìn
            if val_str.count('.') > 1:
                val_str = val_str.replace('.', '')
            elif val_str.count('.') == 1:
                # Nếu chỉ có 1 dấu chấm, kiểm tra xem có phải là số thập phân không
                parts = val_str.split('.')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    # Có thể là số thập phân (ví dụ: "123.45")
                    pass
                else:
                    # Có thể là dấu phân cách hàng nghìn (ví dụ: "123.456")
                    val_str = val_str.replace('.', '')
        try:
            return float(val_str)
        except:
            return 0
    
    df['revenue_parsed'] = df[revenue_col].apply(parse_value)
    
    # Filter theo KHU VUC KINH DOANH chua "Mien Bac"
    print(f"\n1. Filter theo KHU VUC KINH DOANH chua 'Mien Bac':")
    df['region_normalized'] = df[region_col].astype(str).str.strip().str.upper()
    mien_bac_mask = df['region_normalized'].str.contains('MIEN BAC', na=False)
    df_filtered = df[mien_bac_mask].copy()
    print(f"   - Tong so dong sau filter region: {len(df_filtered)}")
    print(f"   - Cac gia tri region unique: {df_filtered[region_col].unique()[:10]}")
    
    # Filter theo Tuyen tour = "Mien Bac" (có thể có dấu hoặc không)
    print(f"\n2. Filter theo Tuyen tour = 'Mien Bac' (co dau hoac khong):")
    df_filtered['route_normalized'] = df_filtered[route_group_col].astype(str).str.strip().str.upper()
    # Tìm các giá trị có chứa "MIEN BAC" hoặc "MIỀN BẮC"
    mien_bac_route_mask = df_filtered['route_normalized'].str.contains('MIEN BAC', na=False) | df_filtered['route_normalized'].str.contains('MIỀN BẮC', na=False)
    df_filtered = df_filtered[mien_bac_route_mask].copy()
    print(f"   - Tong so dong sau filter route: {len(df_filtered)}")
    print(f"   - Cac gia tri route unique: {df_filtered[route_group_col].unique()[:10]}")
    
    # Filter theo GIAI DOAN = "KM XUAN" (có thể có dấu hoặc không)
    print(f"\n3. Filter theo GIAI DOAN = 'KM XUAN' (co dau hoac khong):")
    df_filtered['period_normalized'] = df_filtered[period_col].astype(str).str.strip().str.upper()
    # Tìm các giá trị có chứa "KM XUAN" hoặc "KM XUÂN"
    km_xuan_mask = df_filtered['period_normalized'].str.contains('KM XUAN', na=False) | df_filtered['period_normalized'].str.contains('KM XUÂN', na=False)
    df_filtered = df_filtered[km_xuan_mask].copy()
    print(f"   - Tong so dong sau filter period: {len(df_filtered)}")
    print(f"   - Cac gia tri period unique: {df_filtered[period_col].unique()}")
    
    # Sum cot F (Doanh so da ban)
    print(f"\n4. Tinh tong Doanh so da ban (cot F):")
    total_revenue = df_filtered['revenue_parsed'].sum()
    print(f"   - Tong Doanh so da ban: {total_revenue:,.0f} VND")
    print(f"   - Tong Doanh so da ban (Tr.d): {total_revenue / 1_000_000:,.2f} Tr.d")
    
    # Hien thi chi tiet cac dong
    print(f"\n5. Chi tiet cac dong duoc sum:")
    print(f"   - So dong: {len(df_filtered)}")
    print(f"\n   Cac gia tri Doanh so da ban:")
    print(f"\n   DEBUG: Một vài giá trị gốc và parsed:")
    for idx, (row_idx, row) in enumerate(df_filtered.head(5).iterrows(), 1):
        original_val = row[revenue_col]
        parsed_val = row['revenue_parsed']
        print(f"     Row {row_idx+1} (index {row_idx}): Original='{original_val}' (type: {type(original_val)}) -> Parsed={parsed_val:,.0f} VND")
    
    print(f"\n   Tất cả các giá trị:")
    for idx, (row_idx, row) in enumerate(df_filtered.iterrows(), 1):
        original_val = row[revenue_col]
        parsed_val = row['revenue_parsed']
        print(f"     Row {row_idx+1}: {parsed_val:,.0f} VND")
    
    print(f"\n   Tong: {total_revenue:,.0f} VND = {total_revenue / 1_000_000:,.2f} Tr.d")
    
    # Kiem tra xem co du lieu tu period/region khac khong
    print(f"\n6. Kiem tra du lieu tu cac period/region khac:")
    all_mien_bac = df[df['region_normalized'].str.contains('MIEN BAC', na=False)].copy()
    all_mien_bac['route_normalized'] = all_mien_bac[route_group_col].astype(str).str.strip().str.upper()
    all_mien_bac_route = all_mien_bac[
        (all_mien_bac['route_normalized'].str.contains('MIEN BAC', na=False)) |
        (all_mien_bac['route_normalized'].str.contains('MIỀN BẮC', na=False))
    ].copy()
    all_mien_bac_route['period_normalized'] = all_mien_bac_route[period_col].astype(str).str.strip().str.upper()
    all_mien_bac_route['revenue_parsed'] = all_mien_bac_route[revenue_col].apply(parse_value)
    
    print(f"   - Tong so dong co KHU VUC KINH DOANH chua 'Mien Bac' va Tuyen tour = 'Mien Bac': {len(all_mien_bac_route)}")
    print(f"   - Cac period unique: {all_mien_bac_route['period_normalized'].unique()}")
    
    for period in all_mien_bac_route['period_normalized'].unique():
        period_data = all_mien_bac_route[all_mien_bac_route['period_normalized'] == period]
        period_total = period_data['revenue_parsed'].sum()
        print(f"     - Period '{period}': {len(period_data)} dong, Tong: {period_total:,.0f} VND = {period_total / 1_000_000:,.2f} Tr.d")
    
    return df_filtered

if __name__ == "__main__":
    import sys
    import io
    
    # Fix encoding for Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    sheet_url = "https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=2069863260#gid=2069863260"
    
    print("Dang tai du lieu tu Google Sheet...")
    df = load_etour_data(sheet_url)
    
    if df.empty:
        print("ERROR: Khong the tai du lieu!")
    else:
        print(f"OK: Da tai {len(df)} dong du lieu")
        print(f"   Cac cot: {list(df.columns)}")
        
        result = check_mien_bac_data(df)
        
        if result is not None and not result.empty:
            print(f"\nOK: Ket qua: Tong Doanh so da ban = {result['revenue_parsed'].sum() / 1_000_000:,.2f} Tr.d")
        else:
            print("\nERROR: Khong tim thay du lieu phu hop!")

