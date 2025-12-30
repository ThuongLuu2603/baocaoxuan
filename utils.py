"""
Utility functions for Vietravel Business Intelligence Dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def format_currency(value):
    """Format number as Vietnamese currency (VND)"""
    if pd.isna(value) or value is None:
        return "0 ₫"
    
    # Convert to billions for readability if > 1 billion
    if abs(value) >= 1_000_000_000:
        # show billions with one decimal
        return f"{value / 1_000_000_000:,.1f} tỷ ₫"
    elif abs(value) >= 1_000_000:
        # for million-range numbers show up to 3 decimals (e.g. 44.596 triệu)
        return f"{value / 1_000_000:,.3f} triệu ₫"
    else:
        return f"{value:,.0f} ₫"


def format_number(value):
    """Format number with thousand separators"""
    if pd.isna(value) or value is None:
        return "0"
    # Use rounding rather than truncation to avoid off-by-one from float imprecision
    try:
        return f"{int(round(value)):,}"
    except Exception:
        return f"{int(value):,}"


def format_percentage(value):
    """Format number as percentage"""
    if pd.isna(value) or value is None:
        return "0%"
    # Sửa: Bảo vệ giá trị âm hoặc NaN
    return f"{max(0, value):.1f}%" 


def calculate_completion_rate(actual, planned):
    """Calculate completion rate percentage"""
    if planned == 0 or pd.isna(planned) or planned is None:
        return 0
    return (actual / planned) * 100


def get_growth_rate(current, previous):
    """Calculate growth rate percentage"""
    if previous == 0 or pd.isna(previous) or previous is None:
        return 0
    return ((current - previous) / previous) * 100


def filter_data_by_date(df, start_date, end_date, date_column='booking_date'):
    """Filter dataframe by date range using the tour START date (Từ ngày / 'booking_date').

    NOTE: Per recent requirement, reporting inclusion is determined only by the tour
    start date (Google Sheet column E). This helper will therefore always filter
    on the 'booking_date' column when present. The `date_column` parameter is
    accepted for compatibility but ignored to ensure consistent start-date-only
    behavior across the app.
    """

    if df is None or df.empty:
        # Preserve columns if possible
        try:
            return pd.DataFrame(columns=df.columns)
        except Exception:
            return pd.DataFrame()

    # Use canonical start date column name if present
    start_col_candidates = ['booking_date', 'start_date', 'from_date']
    col = None
    for c in start_col_candidates:
        if c in df.columns:
            col = c
            break

    if col is None:
        # Fallback: if no booking/start column exists, try to use provided date_column
        if date_column in df.columns:
            col = date_column
        else:
            # Nothing to filter on, return empty frame with same columns
            return pd.DataFrame(columns=df.columns)

    # Parse start/end dates robustly. Try default parsing first; if no rows match,
    # attempt parsing with dayfirst=True (Vietnam locale) as a fallback.
    try:
        col_dates = pd.to_datetime(df[col], errors='coerce')
    except Exception:
        col_dates = pd.to_datetime(df[col].astype(str), errors='coerce')

    def make_mask(start_dt, end_dt):
        return (col_dates >= pd.to_datetime(start_dt)) & (col_dates <= pd.to_datetime(end_dt))

    # First attempt: parse start/end with default settings
    try:
        mask = make_mask(start_date, end_date)
    except Exception:
        mask = pd.Series([False] * len(df), index=df.index)

    # If no rows selected, try parsing start/end with dayfirst=True
    if mask.sum() == 0:
        try:
            start_dt = pd.to_datetime(start_date, dayfirst=True, errors='coerce')
            end_dt = pd.to_datetime(end_date, dayfirst=True, errors='coerce')
            if not pd.isna(start_dt) and not pd.isna(end_dt):
                mask = (col_dates >= start_dt) & (col_dates <= end_dt)
        except Exception:
            pass

    return df.loc[mask.fillna(False)].copy()


def filter_confirmed_bookings(df):
    """Filter only confirmed bookings (exclude cancelled/postponed)
    
    For Kỳ Báo Cáo data: all records are already confirmed, no status column needed.
    For Datanet data: filter by status == 'Đã xác nhận'
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # If no status column, assume all bookings are confirmed (Kỳ Báo Cáo case)
    if 'status' not in df.columns:
        return df.copy()
    
    # Filter by confirmed status
    return df[df['status'] == 'Đã xác nhận'].copy()


def calculate_kpis(tours_df, plans_df, start_date, end_date, plans_daily_df=None, plans_weekly_df=None, period_type=None, selected_segment=None):
    """
    Calculate key performance indicators for the dashboard
    
    Args:
        period_type: Type of period ("Tháng", "Quý", "Năm", etc.) to determine correct plan calculation
    """
    # Filter data for current period
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    # Calculate actual metrics using RAW values (before cancellation adjustments)
    # User preference: display booked values (revenue, gross_profit, num_customers)
    # rather than effective values which have cancellations subtracted
    actual_revenue = confirmed_data['revenue'].sum() if 'revenue' in confirmed_data.columns else 0
    actual_gross_profit = confirmed_data['gross_profit'].sum() if 'gross_profit' in confirmed_data.columns else 0
    actual_customers = confirmed_data['num_customers'].sum() if 'num_customers' in confirmed_data.columns else 0
    
    # Determine plan sums according to period type selected by user
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    current_date = pd.to_datetime(datetime.now())
    
    planned_revenue = 0
    planned_gross_profit = 0
    planned_customers = 0

    # Calculate plans based on period type for correct business logic
    # IMPORTANT: For Tháng/Quý/Năm, we want the FULL period plan (not just up to end_dt from sidebar)
    # because users expect to see the full month/quarter/year target even if they selected a partial date range
    if period_type == "Tháng":
        # For monthly reports: use the FULL month from start_date
        current_month = start_dt.month
        current_year = start_dt.year
        plan_mask = (plans_df['year'] == current_year) & (plans_df['month'] == current_month)
    elif period_type == "Quý":
        # For quarterly reports: use the FULL quarter that contains start_date
        current_quarter = (start_dt.month - 1) // 3 + 1
        current_year = start_dt.year
        quarter_months = list(range(3 * current_quarter - 2, 3 * current_quarter + 1))
        plan_mask = (plans_df['year'] == current_year) & (plans_df['month'].isin(quarter_months))
    elif period_type == "Năm":
        # For yearly reports: PREFER annual TOTAL row (month==0) if it exists
        # If not, fall back to summing all 12 months
        current_year = start_dt.year
        annual_rows = plans_df[(plans_df['year'] == current_year) & (plans_df['month'] == 0)] if not plans_df.empty else pd.DataFrame()
        if not annual_rows.empty:
            # Use annual TOTAL row directly (no mask needed)
            plan_mask = (plans_df['year'] == current_year) & (plans_df['month'] == 0)
        else:
            # Fall back to all months in the year
            plan_mask = (plans_df['year'] == current_year)
    else:
        # For custom periods or week: prefer daily breakdown when available.
        # If daily plans exist, sum exact days inside the window. Otherwise fall back to
        # prorating monthly plans by overlap days in each month.
        period_length_days = (end_dt - start_dt).days + 1

        if plans_daily_df is not None and not plans_daily_df.empty:
            mask = (plans_daily_df['date'] >= pd.to_datetime(start_dt).normalize()) & (plans_daily_df['date'] <= pd.to_datetime(end_dt).normalize())
            df_slice = plans_daily_df.loc[mask].copy()
            # If the sheet contains a company-level total (TOÀN CÔNG TY), prefer those rows to avoid double-counting
            try:
                # Only prefer the company-level totals when user hasn't filtered to a specific segment
                seg = None if selected_segment is None else str(selected_segment).strip().upper()
                prefer_company = seg in (None, '') or seg in ('TẤT CẢ', 'TAT CA', 'ALL')
                if prefer_company and not plans_df.empty and plans_df['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN').any():
                    # Prefer only the company-level total rows (segment is empty/None/'TOTAL') to avoid double-counting
                    comp_mask = df_slice['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN') & (
                        df_slice['segment'].isna() | df_slice['segment'].astype(str).str.strip().eq('') | df_slice['segment'].astype(str).str.upper().eq('TOTAL')
                    )
                    if comp_mask.any():
                        df_slice = df_slice[comp_mask]
            except Exception:
                # if anything goes wrong, fall back to using all units
                pass
            planned_revenue = df_slice['planned_revenue_daily'].sum()
            planned_gross_profit = df_slice['planned_gross_profit_daily'].sum()
            planned_customers = df_slice['planned_customers_daily'].sum()
        else:
            # No daily breakdown available: prorate monthly plans by overlap days.
            try:
                from calendar import monthrange
                # Build list of months between start_dt and end_dt
                s = start_dt.replace(day=1)
                e = end_dt.replace(day=1)
                months = []
                cur = s
                while cur <= e:
                    months.append((int(cur.year), int(cur.month)))
                    # advance month
                    if cur.month == 12:
                        cur = cur.replace(year=cur.year+1, month=1)
                    else:
                        cur = cur.replace(month=cur.month+1)

                total_rev = 0.0
                total_profit = 0.0
                total_cust = 0.0
                for (y,m) in months:
                    # days in month
                    dim = monthrange(y, m)[1]
                    month_start = pd.to_datetime(datetime(y, m, 1))
                    month_end = pd.to_datetime(datetime(y, m, dim))
                    # overlap between [start_dt, end_dt] and [month_start, month_end]
                    overlap_start = max(pd.to_datetime(start_dt), month_start)
                    overlap_end = min(pd.to_datetime(end_dt), month_end)
                    overlap_days = (overlap_end - overlap_start).days + 1
                    if overlap_days <= 0:
                        continue
                    pmask = (plans_df['year'] == y) & (plans_df['month'] == m)
                    month_plans = plans_df[pmask] if not plans_df.empty else pd.DataFrame()
                    if month_plans.empty:
                        continue
                    # If company-level monthly totals exist, prefer those rows for prorating
                    try:
                        # Only prefer company-level monthly totals when user hasn't filtered by a segment
                        seg = None if selected_segment is None else str(selected_segment).strip().upper()
                        prefer_company = seg in (None, '') or seg in ('TẤT CẢ', 'TAT CA', 'ALL')
                        if prefer_company and month_plans['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN').any():
                            comp_mask = month_plans['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN') & (
                                month_plans['segment'].isna() | month_plans['segment'].astype(str).str.strip().eq('') | month_plans['segment'].astype(str).str.upper().eq('TOTAL')
                            )
                            if comp_mask.any():
                                month_plans = month_plans[comp_mask]
                    except Exception:
                        pass
                    # Sum plans for that month and prorate by overlap_days/dim
                    rev_month = month_plans['planned_revenue'].sum()
                    prof_month = month_plans['planned_gross_profit'].sum()
                    cust_month = month_plans['planned_customers'].sum()
                    frac = float(overlap_days) / float(dim)
                    total_rev += rev_month * frac
                    total_profit += prof_month * frac
                    total_cust += cust_month * frac

                planned_revenue = total_rev
                planned_gross_profit = total_profit
                planned_customers = total_cust
            except Exception:
                # As a last resort, fall back to naive monthly sum covering start-end months
                plan_mask = (plans_df['year'] == start_dt.year) & \
                            (plans_df['month'] >= start_dt.month) & \
                            (plans_df['month'] <= end_dt.month)
    
    # If we have a plan_mask (for Tháng/Quý/Năm), apply it to get the plans
    if 'plan_mask' in locals():
        period_plans = plans_df[plan_mask] if not plans_df.empty else pd.DataFrame()
        if not period_plans.empty:
            # If user selected a specific segment, filter the period_plans accordingly
            try:
                seg = None if selected_segment is None else str(selected_segment).strip()
                if seg and seg.upper() not in ('TẤT CẢ', 'TAT CA', 'ALL'):
                    seg_up = seg.upper()
                    seg_mask = period_plans['segment'].fillna('').astype(str).str.upper().str.contains(seg_up)
                    # Fallback: if no segment rows matched, try matching business_unit (some sheets encode segment names differently)
                    if not seg_mask.any():
                        seg_mask = period_plans['business_unit'].fillna('').astype(str).str.upper().str.contains(seg_up)
                    period_plans = period_plans[seg_mask]
            except Exception:
                pass

            # For Tháng/Quý/Năm: take FULL period plans (no prorating)
            # For Tuần/Tùy chỉnh: prorate by overlap days
            if period_type in ["Tháng", "Quý", "Năm"]:
                # No prorating — sum all monthly rows in the period mask
                planned_revenue = period_plans['planned_revenue'].sum()
                planned_gross_profit = period_plans['planned_gross_profit'].sum()
                planned_customers = period_plans['planned_customers'].sum()
            else:
                # Prorate month-level plans by overlap with the selected date window (for Tuần/Tùy chỉnh only)
                try:
                    from calendar import monthrange
                    total_rev = 0.0
                    total_prof = 0.0
                    total_cust = 0.0
                    for _, prow in period_plans.iterrows():
                        y = int(prow.get('year', start_dt.year))
                        m = int(prow.get('month', start_dt.month) or start_dt.month)
                        # skip annual rows here (month==0) — they are handled elsewhere
                        if m == 0:
                            continue
                        dim = monthrange(y, m)[1]
                        month_start = pd.to_datetime(datetime(y, m, 1))
                        month_end = pd.to_datetime(datetime(y, m, dim))
                        overlap_start = max(start_dt, month_start)
                        overlap_end = min(end_dt, month_end)
                        overlap_days = (overlap_end - overlap_start).days + 1
                        if overlap_days <= 0:
                            continue
                        frac = float(overlap_days) / float(dim)
                        total_rev += float(prow.get('planned_revenue', 0.0)) * frac
                        total_prof += float(prow.get('planned_gross_profit', 0.0)) * frac
                        total_cust += float(prow.get('planned_customers', 0.0)) * frac

                    planned_revenue = total_rev
                    planned_gross_profit = total_prof
                    planned_customers = int(round(total_cust))
                except Exception:
                    # Fallback to naive sum if anything goes wrong
                    planned_revenue = period_plans['planned_revenue'].sum()
                    planned_gross_profit = period_plans['planned_gross_profit'].sum()
                    planned_customers = period_plans['planned_customers'].sum()
        else:
            # If there are no monthly rows for the requested period, try using an annual TOTAL row
            try:
                annual_rows = plans_df[(plans_df['year'] == start_dt.year) & (plans_df['month'] == 0)] if not plans_df.empty else pd.DataFrame()
                if not annual_rows.empty:
                    seg = None if selected_segment is None else str(selected_segment).strip()
                    # If user selected a specific segment, try to match it in the annual rows first
                    matched = pd.DataFrame()
                    if seg and seg.upper() not in ('TẤT CẢ', 'TAT CA', 'ALL'):
                        seg_mask = annual_rows['segment'].fillna('').astype(str).str.upper().str.contains(seg.upper())
                        if seg_mask.any():
                            matched = annual_rows[seg_mask]
                    # Fallback: prefer company TOTAL annual row
                    if matched.empty:
                        comp_mask = annual_rows['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN')
                        if comp_mask.any():
                            matched = annual_rows[comp_mask]
                    # If still empty, use all annual rows
                    if matched.empty:
                        matched = annual_rows

                    if not matched.empty:
                        rev_ann = matched['planned_revenue'].sum()
                        prof_ann = matched['planned_gross_profit'].sum()
                        cust_ann = matched['planned_customers'].sum()
                        
                        # For period_type "Năm": use full annual total (no prorating)
                        # For "Tháng"/"Quý": prorate by months in the period
                        if period_type == "Năm":
                            planned_revenue = rev_ann
                            planned_gross_profit = prof_ann
                            planned_customers = int(cust_ann)
                        else:
                            # Prorate by number of months in the period
                            months_in_period = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
                            frac = float(months_in_period) / 12.0 if months_in_period > 0 else 0
                            planned_revenue = rev_ann * frac
                            planned_gross_profit = prof_ann * frac
                            planned_customers = int(round(cust_ann * frac))
            except Exception:
                # ignore and leave planned values as-is
                pass

    # Prefer "TOÀN CÔNG TY" rows from Google Sheet when available for period-based reports
    try:
        # Prefer "TOÀN CÔNG TY" rows from Google Sheet when available for period-based reports
        # but only when the user hasn't selected a specific segment.
        seg = None if selected_segment is None else str(selected_segment).strip().upper()
        prefer_company = seg in (None, '') or seg in ('TẤT CẢ', 'TAT CA', 'ALL')
        if prefer_company and not plans_df.empty and period_type in ["Tháng", "Quý", "Năm"]:
            # Look for company-level TOTAL rows (segment empty/None/'TOTAL') within the period mask
            company_mask = plans_df['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN')
            if company_mask.any():
                # Apply the same period mask to company totals and prefer rows that look like the total row
                if 'plan_mask' in locals():
                    cand = plans_df[plan_mask & company_mask].copy()
                    comp_mask = cand['segment'].isna() | cand['segment'].astype(str).str.strip().eq('') | cand['segment'].astype(str).str.upper().eq('TOTAL')
                    company_plans = cand[comp_mask] if comp_mask.any() else cand
                    if not company_plans.empty:
                        # For Tháng/Quý/Năm: take FULL period totals (no prorating)
                        planned_revenue = company_plans['planned_revenue'].sum()
                        planned_gross_profit = company_plans['planned_gross_profit'].sum()
                        planned_customers = company_plans['planned_customers'].sum()
    except Exception:
        # if anything goes wrong in this override logic, ignore and keep existing planned values
        pass
    
    # Calculate same period last year
    last_year_start = start_dt - timedelta(days=365)
    last_year_end = end_dt - timedelta(days=365)
    last_year_data = filter_data_by_date(tours_df, last_year_start, last_year_end)
    last_year_confirmed = filter_confirmed_bookings(last_year_data)
    
    # Safe column access with fallback to 0
    ly_revenue = last_year_confirmed['revenue'].sum() if 'revenue' in last_year_confirmed.columns else 0
    ly_gross_profit = last_year_confirmed['gross_profit'].sum() if 'gross_profit' in last_year_confirmed.columns else 0
    ly_customers = last_year_confirmed['num_customers'].sum() if 'num_customers' in last_year_confirmed.columns else 0
    
    # Completion rates
    revenue_completion = calculate_completion_rate(actual_revenue, planned_revenue)
    customer_completion = calculate_completion_rate(actual_customers, planned_customers)
    
    # Growth rates
    revenue_growth = get_growth_rate(actual_revenue, ly_revenue)
    profit_growth = get_growth_rate(actual_gross_profit, ly_gross_profit)
    customer_growth = get_growth_rate(actual_customers, ly_customers)
    
    # Normalize customer counts to integers to avoid display truncation from floating imprecision
    try:
        if planned_customers is not None:
            planned_customers = int(round(planned_customers))
    except Exception:
        pass

    return {
        'actual_revenue': actual_revenue,
        'actual_gross_profit': actual_gross_profit,
        'actual_customers': actual_customers,
        'planned_revenue': planned_revenue,
        'planned_gross_profit': planned_gross_profit,
        'planned_customers': planned_customers,
        'ly_revenue': ly_revenue,
        'ly_gross_profit': ly_gross_profit,
        'ly_customers': ly_customers,
        'revenue_completion': revenue_completion,
        'customer_completion': customer_completion,
        'revenue_growth': revenue_growth,
        'profit_growth': profit_growth,
        'customer_growth': customer_growth
    }




def get_top_routes(tours_df, n=10, metric='revenue'):
    """
    Get top N routes by specified metric
    """
    confirmed = filter_confirmed_bookings(tours_df)
    
    if confirmed.empty: 
        return pd.DataFrame(columns=['route', 'revenue', 'num_customers', 'gross_profit', 'profit_margin'])

    if metric == 'revenue':
        grouped = confirmed.groupby('route').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        grouped = grouped.sort_values('revenue', ascending=False).head(n)
    elif metric == 'customers':
        grouped = confirmed.groupby('route').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        grouped = grouped.sort_values('num_customers', ascending=False).head(n)
    else:  # gross_profit
        grouped = confirmed.groupby('route').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        grouped = grouped.sort_values('gross_profit', ascending=False).head(n)
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    grouped['profit_margin'] = np.where(
        grouped['revenue'] > 0,
        (grouped['gross_profit'] / grouped['revenue'] * 100).round(2),
        0
    )
    
    return grouped


def calculate_operational_metrics(tours_df):
    """
    Calculate operational metrics
    """
    # Average occupancy rate - ONLY for FIT segment as requested
    confirmed = filter_confirmed_bookings(tours_df)
    try:
        fit_confirmed = confirmed[confirmed['segment'].fillna('').astype(str).str.strip().str.upper() == 'FIT']
    except Exception:
        # Fallback: if segmentation column missing or unexpected, treat as empty
        fit_confirmed = pd.DataFrame(columns=confirmed.columns)

    total_booked = fit_confirmed['num_customers'].sum()
    total_capacity = fit_confirmed['tour_capacity'].sum()
    # Protect divide-by-zero: occupancy = booked / capacity
    avg_occupancy = (total_booked / total_capacity * 100) if total_capacity > 0 else 0
    
    # Cancellation/postponement rate
    total_bookings = len(tours_df)
    if 'status' in tours_df.columns:
        cancelled_postponed = len(tours_df[tours_df['status'].isin(['Đã hủy', 'Hoãn'])])
    elif 'cancel_count' in tours_df.columns:
        # For Kỳ Báo Cáo: sum cancel_count
        cancelled_postponed = pd.to_numeric(tours_df['cancel_count'], errors='coerce').fillna(0).sum()
    else:
        cancelled_postponed = 0
    # Hàm này đã có bảo vệ chia cho 0
    cancel_rate = (cancelled_postponed / total_bookings * 100) if total_bookings > 0 else 0
    
    # Returning customer rate
    customer_counts = tours_df.groupby('customer_id').size()
    returning_customers = len(customer_counts[customer_counts >= 2])
    total_unique_customers = len(customer_counts)
    # Hàm này đã có bảo vệ chia cho 0
    returning_rate = (returning_customers / total_unique_customers * 100) if total_unique_customers > 0 else 0
    
    return {
        'avg_occupancy': avg_occupancy,
        'cancel_rate': cancel_rate,
        'returning_rate': returning_rate
    }



def load_unit_completion_data(sheet_url):
    """
    Đọc dữ liệu mức độ hoàn thành kế hoạch đơn vị từ Google Sheet
    Sheet URL: https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=614149511#gid=614149511
    
    Cấu trúc:
    - Header ở dòng 26: Khu vuc,DT Kế hoạch,DT đã bán,Tỷ lệ đạt,LG Kế hoạch,LG đã bán,Tỷ lệ đạt
    - Dữ liệu bắt đầu từ dòng 27
    
    Returns: DataFrame với columns: business_unit, revenue_completion, profit_completion
    """
    import requests
    import io
    import re
    
    try:
        # Chuyển đổi URL thành CSV export URL
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return pd.DataFrame()
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else '614149511'  # Default gid từ URL
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        # Tải CSV
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        
        # Đọc CSV, bỏ qua các dòng trống ở đầu, header ở dòng 26 (index 25)
        text = resp.content.decode('utf-8', errors='replace')
        lines = text.split('\n')
        
        # Tìm dòng header (chứa "Khu vuc" hoặc "Khu vực" và "DT Kế hoạch")
        header_idx = None
        for i, line in enumerate(lines):
            line_upper = line.upper()
            if ('KHU VUC' in line_upper or 'KHU VỰC' in line_upper) and 'DT KẾ HOẠCH' in line_upper:
                header_idx = i
                break
        
        if header_idx is None:
            # Fallback: tìm dòng có "DT Kế hoạch"
            for i, line in enumerate(lines):
                if 'DT Kế hoạch' in line.upper() or 'DT KE HOACH' in line.upper():
                    header_idx = i
                    break
        
        if header_idx is None:
            return pd.DataFrame()
        
        # Đọc từ header_idx, skipinitialspace để xử lý khoảng trắng
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])), dtype=str, skipinitialspace=True)
        
        # Tìm các cột cần thiết
        unit_col = None
        dt_plan_col = None
        dt_actual_col = None
        dt_completion_col = None
        lg_plan_col = None
        lg_actual_col = None
        lg_completion_col = None
        
        for col in df.columns:
            col_upper = str(col).upper()
            if 'KHU VUC' in col_upper or 'KHU VỰC' in col_upper or 'KHU VUC' in col_upper:
                unit_col = col
            elif 'DT KẾ HOẠCH' in col_upper or 'DT KE HOACH' in col_upper:
                dt_plan_col = col
            elif 'DT ĐÃ BÁN' in col_upper or 'DT DA BAN' in col_upper:
                dt_actual_col = col
            elif 'TỶ LỆ ĐẠT' in col_upper and dt_completion_col is None:
                dt_completion_col = col
            elif 'LG KẾ HOẠCH' in col_upper or 'LG KE HOACH' in col_upper:
                lg_plan_col = col
            elif 'LG ĐÃ BÁN' in col_upper or 'LG DA BAN' in col_upper:
                lg_actual_col = col
            elif 'TỶ LỆ ĐẠT' in col_upper and lg_completion_col is None and dt_completion_col is not None:
                lg_completion_col = col
        
        # Nếu không tìm thấy bằng tên, dùng vị trí cột (theo CSV mẫu)
        if unit_col is None and len(df.columns) > 0:
            unit_col = df.columns[0]
        if dt_completion_col is None and len(df.columns) > 3:
            dt_completion_col = df.columns[3]
        if lg_completion_col is None and len(df.columns) > 6:
            lg_completion_col = df.columns[6]
        
        if unit_col is None or dt_completion_col is None or lg_completion_col is None:
            return pd.DataFrame()
        
        # Lấy tất cả các cột cần thiết
        cols_to_keep = [unit_col]
        if dt_plan_col:
            cols_to_keep.append(dt_plan_col)
        if dt_actual_col:
            cols_to_keep.append(dt_actual_col)
        if dt_completion_col:
            cols_to_keep.append(dt_completion_col)
        if lg_plan_col:
            cols_to_keep.append(lg_plan_col)
        if lg_actual_col:
            cols_to_keep.append(lg_actual_col)
        if lg_completion_col:
            cols_to_keep.append(lg_completion_col)
        
        # Lọc dữ liệu (bỏ dòng "Grand Total" và các dòng trống)
        result_df = df[cols_to_keep].copy()
        
        # Đặt tên cột
        col_names = ['business_unit']
        if dt_plan_col:
            col_names.append('revenue_plan')
        if dt_actual_col:
            col_names.append('revenue_actual')
        if dt_completion_col:
            col_names.append('revenue_completion')
        if lg_plan_col:
            col_names.append('profit_plan')
        if lg_actual_col:
            col_names.append('profit_actual')
        if lg_completion_col:
            col_names.append('profit_completion')
        
        result_df.columns = col_names
        
        # Loại bỏ dòng trống và "Grand Total"
        result_df = result_df[
            (result_df['business_unit'].notna()) & 
            (result_df['business_unit'].astype(str).str.strip() != '') &
            (~result_df['business_unit'].astype(str).str.contains('Grand Total', case=False, na=False))
        ].copy()
        
        # Xử lý số liệu: parse các giá trị
        def parse_numeric(val):
            if pd.isna(val) or val == '':
                return 0
            val_str = str(val).strip()
            # Loại bỏ dấu phẩy (nếu có trong số)
            val_str = val_str.replace(',', '')
            try:
                return float(val_str)
            except:
                return 0
        
        def parse_percentage(val):
            if pd.isna(val) or val == '':
                return 0
            val_str = str(val).strip()
            # Loại bỏ dấu %
            val_str = val_str.replace('%', '')
            # Loại bỏ dấu phẩy (nếu có trong số)
            val_str = val_str.replace(',', '')
            try:
                return float(val_str)
            except:
                return 0
        
        # Parse các cột số
        if 'revenue_plan' in result_df.columns:
            result_df['revenue_plan'] = result_df['revenue_plan'].apply(parse_numeric)
        if 'revenue_actual' in result_df.columns:
            result_df['revenue_actual'] = result_df['revenue_actual'].apply(parse_numeric)
        if 'revenue_completion' in result_df.columns:
            result_df['revenue_completion'] = result_df['revenue_completion'].apply(parse_percentage)
        if 'profit_plan' in result_df.columns:
            result_df['profit_plan'] = result_df['profit_plan'].apply(parse_numeric)
        if 'profit_actual' in result_df.columns:
            result_df['profit_actual'] = result_df['profit_actual'].apply(parse_numeric)
        if 'profit_completion' in result_df.columns:
            result_df['profit_completion'] = result_df['profit_completion'].apply(parse_percentage)
        
        # Làm sạch tên đơn vị
        result_df['business_unit'] = result_df['business_unit'].astype(str).str.strip()
        
        # Phân loại khu vực và đơn vị
        # Các dòng bắt đầu bằng "KV" là khu vực, còn lại là đơn vị
        result_df['is_region'] = result_df['business_unit'].str.startswith('KV ')
        result_df['region'] = None
        
        # Gán khu vực cho từng đơn vị
        current_region = None
        for idx, row in result_df.iterrows():
            if row['is_region']:
                current_region = row['business_unit']
                result_df.at[idx, 'region'] = current_region
            else:
                result_df.at[idx, 'region'] = current_region
        
        return result_df
        
    except Exception as e:
        # Trả về DataFrame rỗng nếu có lỗi
        return pd.DataFrame()


def load_unit_completion_data_tet(sheet_url):
    """
    Đọc dữ liệu mức độ hoàn thành kế hoạch đơn vị từ Google Sheet TẾT
    Sheet URL: https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=20510042#gid=20510042
    
    Cấu trúc:
    - Dòng 3: Header cho TẾT (cột J, K, L, M)
    - Cột J: Quốc gia/ tuyến tour
    - Cột K: Doanh thu kế hoạch (Tỷ đồng)
    - Cột L: Doanh thu đã bán (Tỷ đồng)
    - Cột M: Tốc độ đạt kế hoạch (%)
    
    Returns: DataFrame với columns: business_unit, revenue_completion, profit_completion
    """
    import requests
    import io
    import re
    
    try:
        # Chuyển đổi URL thành CSV export URL
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return pd.DataFrame()
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else '20510042'  # Default gid từ URL
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        # Tải CSV
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        
        # Đọc CSV
        text = resp.content.decode('utf-8', errors='replace')
        lines = text.split('\n')
        
        # Tìm dòng header (dòng 3, index 2) - chứa "Quốc gia/ tuyến tour" cho TẾT
        header_idx = None
        for i, line in enumerate(lines[:10]):
            line_upper = line.upper()
            if 'QUỐC GIA' in line_upper or 'QUOC GIA' in line_upper:
                if 'TUYẾN TOUR' in line_upper or 'TUYEN TOUR' in line_upper:
                    header_idx = i
                    break
        
        if header_idx is None:
            header_idx = 2  # Fallback: dòng 3 (index 2)
        
        # Đọc từ header_idx
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])), dtype=str, skipinitialspace=True)
        
        if df.empty:
            return pd.DataFrame()
        
        # Tìm các cột cho TẾT (cột J, K, L, M - index 9, 10, 11, 12)
        # Cột J (index 9): Quốc gia/ tuyến tour
        # Cột K (index 10): Doanh thu kế hoạch (Tỷ đồng)
        # Cột L (index 11): Doanh thu đã bán (Tỷ đồng)
        # Cột M (index 12): Tốc độ đạt kế hoạch (%)
        
        unit_col = None
        dt_plan_col = None
        dt_actual_col = None
        dt_completion_col = None
        
        # Tìm cột bằng tên
        for col in df.columns:
            col_upper = str(col).upper()
            if ('QUỐC GIA' in col_upper or 'QUOC GIA' in col_upper) and ('TUYẾN TOUR' in col_upper or 'TUYEN TOUR' in col_upper):
                unit_col = col
            elif 'DOANH THU KẾ HOẠCH' in col_upper or 'DOANH THU KE HOACH' in col_upper:
                dt_plan_col = col
            elif 'DOANH THU ĐÃ BÁN' in col_upper or 'DOANH THU DA BAN' in col_upper:
                dt_actual_col = col
            elif 'TỐC ĐỘ ĐẠT KẾ HOẠCH' in col_upper or 'TOC DO DAT KE HOACH' in col_upper:
                dt_completion_col = col
        
        # Fallback: dùng vị trí cột (cột J, K, L, M - index 9, 10, 11, 12)
        if unit_col is None and len(df.columns) > 9:
            unit_col = df.columns[9]  # Cột J
        if dt_plan_col is None and len(df.columns) > 10:
            dt_plan_col = df.columns[10]  # Cột K
        if dt_actual_col is None and len(df.columns) > 11:
            dt_actual_col = df.columns[11]  # Cột L
        if dt_completion_col is None and len(df.columns) > 12:
            dt_completion_col = df.columns[12]  # Cột M
        
        if unit_col is None or dt_completion_col is None:
            return pd.DataFrame()
        
        # Lấy các cột cần thiết
        cols_to_keep = [unit_col]
        if dt_plan_col:
            cols_to_keep.append(dt_plan_col)
        if dt_actual_col:
            cols_to_keep.append(dt_actual_col)
        if dt_completion_col:
            cols_to_keep.append(dt_completion_col)
        
        # Lọc dữ liệu
        result_df = df[cols_to_keep].copy()
        
        # Đặt tên cột
        col_names = ['business_unit']
        if dt_plan_col:
            col_names.append('revenue_plan')
        if dt_actual_col:
            col_names.append('revenue_actual')
        if dt_completion_col:
            col_names.append('revenue_completion')
        
        result_df.columns = col_names
        
        # Loại bỏ dòng trống
        result_df = result_df[
            (result_df['business_unit'].notna()) & 
            (result_df['business_unit'].astype(str).str.strip() != '')
        ].copy()
        
        # Xử lý số liệu
        def parse_numeric(val):
            if pd.isna(val) or val == '':
                return 0
            val_str = str(val).strip().replace(',', '')
            try:
                return float(val_str)
            except:
                return 0
        
        def parse_percentage(val):
            if pd.isna(val) or val == '':
                return 0
            val_str = str(val).strip().replace('%', '').replace(',', '')
            try:
                return float(val_str)
            except:
                return 0
        
        # Parse các giá trị
        if 'revenue_plan' in result_df.columns:
            result_df['revenue_plan'] = result_df['revenue_plan'].apply(parse_numeric)
        if 'revenue_actual' in result_df.columns:
            result_df['revenue_actual'] = result_df['revenue_actual'].apply(parse_numeric)
        if 'revenue_completion' in result_df.columns:
            result_df['revenue_completion'] = result_df['revenue_completion'].apply(parse_percentage)
        
        # Thêm cột profit_completion = revenue_completion (vì sheet TẾT không có LG riêng)
        result_df['profit_completion'] = result_df['revenue_completion']
        
        # Thêm cột is_region (giả sử tất cả là đơn vị, không phải khu vực)
        result_df['is_region'] = False
        
        return result_df
        
    except Exception as e:
        return pd.DataFrame()


def load_route_performance_data(sheet_url):
    """
    Đọc dữ liệu tốc độ đạt kế hoạch theo tuyến từ Google Sheet
    Sheet URL: https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=903527778#gid=903527778
    
    Cấu trúc CSV:
    - Header ở dòng 1: Khu vực Đơn Vị, Đơn Vị, Dom/Out..., Nhóm tuyến, Tuyến Tour, GIÁ TRỊ LK, GIÁ TRỊ DT(tr.đ), GIÁ TRỊ LG(tr.đ), Giai đoạn
    - Dữ liệu bắt đầu từ dòng 2
    
    Returns: DataFrame với columns: route, route_type, num_customers, revenue, gross_profit
    """
    import requests
    import io
    import re
    
    try:
        # Chuyển đổi URL thành CSV export URL
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return pd.DataFrame()
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else '903527778'  # Default gid từ URL
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        # Tải CSV
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        
        # Đọc CSV, header ở dòng 1 (index 0)
        df = pd.read_csv(io.StringIO(resp.content.decode('utf-8', errors='replace')), skipinitialspace=True)
        
        if df.empty:
            return pd.DataFrame()
        
        # Tìm các cột cần thiết (case-insensitive)
        route_col = None
        dom_out_col = None
        customers_col = None
        revenue_col = None
        profit_col = None
        period_col = None  # Cột "Giai đoạn"
        region_unit_col = None  # Cột "Khu vực Đơn Vị"
        unit_col = None  # Cột "Đơn Vị"
        
        for col in df.columns:
            col_upper = str(col).upper()
            if 'TUYẾN TOUR' in col_upper or 'TUYEN TOUR' in col_upper:
                route_col = col
            elif 'DOM/OUT' in col_upper or 'DOMOUT' in col_upper:
                dom_out_col = col
            elif 'GIÁ TRỊ LK' in col_upper or 'GIA TRI LK' in col_upper or 'LK' in col_upper:
                customers_col = col
            elif 'GIÁ TRỊ DT' in col_upper or 'GIA TRI DT' in col_upper or ('DT' in col_upper and 'TR.Đ' in col_upper):
                revenue_col = col
            elif 'GIÁ TRỊ LG' in col_upper or 'GIA TRI LG' in col_upper or ('LG' in col_upper and 'TR.Đ' in col_upper):
                profit_col = col
            elif 'GIAI ĐOẠN' in col_upper or 'GIAI DOAN' in col_upper:
                period_col = col
            elif 'KHU VỰC ĐƠN VỊ' in col_upper or 'KHU VUC DON VI' in col_upper or ('KHU VUC' in col_upper and 'DON VI' in col_upper):
                region_unit_col = col
            elif 'ĐƠN VỊ' in col_upper or 'DON VI' in col_upper:
                if unit_col is None:  # Chỉ lấy cột "Đơn Vị" đầu tiên (không phải "Khu vực Đơn Vị")
                    unit_col = col
        
        # Nếu không tìm thấy bằng tên, dùng vị trí cột (theo CSV mẫu)
        if route_col is None and len(df.columns) > 4:
            route_col = df.columns[4]  # Cột E: Tuyến Tour
        if dom_out_col is None and len(df.columns) > 2:
            dom_out_col = df.columns[2]  # Cột C: Dom/Out...
        if customers_col is None and len(df.columns) > 5:
            customers_col = df.columns[5]  # Cột F: GIÁ TRỊ LK
        if revenue_col is None and len(df.columns) > 6:
            revenue_col = df.columns[6]  # Cột G: GIÁ TRỊ DT(tr.đ)
        if profit_col is None and len(df.columns) > 7:
            profit_col = df.columns[7]  # Cột H: GIÁ TRỊ LG(tr.đ)
        if period_col is None and len(df.columns) > 8:
            period_col = df.columns[8]  # Cột I: Giai đoạn
        if region_unit_col is None and len(df.columns) > 0:
            region_unit_col = df.columns[0]  # Cột A: Khu vực Đơn Vị
        if unit_col is None and len(df.columns) > 1:
            unit_col = df.columns[1]  # Cột B: Đơn Vị
        
        if route_col is None or dom_out_col is None:
            return pd.DataFrame()
        
        # Tạo mask để lọc dữ liệu (trước khi tạo result_df)
        # GIỮ LẠI các dòng subtotal (có "LK" trong Đơn Vị) và các dòng chi tiết
        # Chỉ loại bỏ các dòng có "Total" hoặc "Grand Total" trong route (trừ khi là dòng subtotal hợp lệ)
        route_mask = df[route_col].astype(str).str.strip() != ''
        
        # Xác định các dòng subtotal hợp lệ (có "LK" trong Đơn Vị)
        # Các dòng subtotal: Mien Bac LK, Mien Trung LK, Mien Tay LK, TPHCM & DNB LK, Total LK
        if unit_col:
            is_subtotal = df[unit_col].astype(str).str.contains('LK', case=False, na=False)
            # Giữ lại các dòng subtotal và các dòng không có "Total" trong route
            route_mask = route_mask & (
                is_subtotal |  # Giữ lại các dòng subtotal
                (~df[route_col].astype(str).str.contains('Total', case=False, na=False))  # Giữ lại các dòng không có "Total"
            )
        else:
            route_mask = route_mask & (~df[route_col].astype(str).str.contains('Total', case=False, na=False))
        
        # Loại bỏ các dòng có "Total" trong dom_out (trừ khi là dòng subtotal)
        if dom_out_col:
            if unit_col:
                is_subtotal = df[unit_col].astype(str).str.contains('LK', case=False, na=False)
                dom_out_mask = is_subtotal | (~df[dom_out_col].astype(str).str.contains('Total', case=False, na=False))
            else: 
                dom_out_mask = ~df[dom_out_col].astype(str).str.contains('Total', case=False, na=False)
            route_mask = route_mask & dom_out_mask
        
        # Áp dụng mask
        df_filtered = df[route_mask].copy()
        
        if df_filtered.empty:
            return pd.DataFrame()
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame()
        result_df['route'] = df_filtered[route_col].astype(str).str.strip()
        result_df['dom_out'] = df_filtered[dom_out_col].astype(str).str.strip() if dom_out_col else ''
        
        # Thêm cột Khu vực Đơn Vị
        if region_unit_col:
            result_df['region_unit'] = df_filtered[region_unit_col].astype(str).str.strip()
        else:
            result_df['region_unit'] = ''
        
        # Thêm cột Đơn Vị
        if unit_col:
            result_df['unit'] = df_filtered[unit_col].astype(str).str.strip()
        else:
            result_df['unit'] = ''
        
        # Thêm cột Giai đoạn
        if period_col:
            result_df['period'] = df_filtered[period_col].astype(str).str.strip()
        else:
            result_df['period'] = ''
        
        if customers_col:
            result_df['num_customers'] = df_filtered[customers_col]
        else:
            result_df['num_customers'] = 0
        if revenue_col:
            result_df['revenue'] = df_filtered[revenue_col]
        else:
            result_df['revenue'] = 0
        if profit_col:
            result_df['gross_profit'] = df_filtered[profit_col]
        else:
            result_df['gross_profit'] = 0
        
        # Loại bỏ dòng trống route
        result_df = result_df[result_df['route'] != ''].copy()
        
        # Phân loại route_type: "Dom" = Nội địa, "Out" = Outbound
        result_df['route_type'] = result_df['dom_out'].astype(str).str.strip().str.upper()
        result_df['route_type'] = result_df['route_type'].apply(
            lambda x: 'Nội địa' if 'DOM' in x else ('Outbound' if 'OUT' in x else 'Nội địa')
        )
        
        # Làm sạch tên tuyến
        result_df['route'] = result_df['route'].astype(str).str.strip()
        
        # Xử lý số liệu: chuyển đổi sang số
        def parse_numeric(val):
            if pd.isna(val) or val == '':
                return 0
            val_str = str(val).strip()
            val_str = val_str.replace(',', '')
            try:
                return float(val_str)
            except:
                return 0
        
        result_df['num_customers'] = result_df['num_customers'].apply(parse_numeric)
        # LƯU Ý: Dữ liệu từ DEFAULT_ROUTE_PERFORMANCE_URL có đơn vị DT và LG là triệu đồng (tr.đ)
        # Chuyển đổi sang VND để tính toán thống nhất với các nguồn dữ liệu khác
        result_df['revenue'] = result_df['revenue'].apply(parse_numeric) * 1_000_000  # Chuyển từ triệu đồng sang VND
        result_df['gross_profit'] = result_df['gross_profit'].apply(parse_numeric) * 1_000_000  # Chuyển từ triệu đồng sang VND
        
        # Loại bỏ các dòng có route trống, NaN, hoặc "nan" sau khi làm sạch
        result_df = result_df[
            (result_df['route'] != '') & 
            (result_df['route'].notna()) &
            (~result_df['route'].astype(str).str.upper().str.contains('NAN', na=False))
        ].copy()
        
        # Nhóm theo route, route_type, period, region_unit, unit để tổng hợp (nếu có nhiều dòng cho cùng một route)
        result_df = result_df.groupby(['route', 'route_type', 'period', 'region_unit', 'unit']).agg({
            'num_customers': 'sum',
            'revenue': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        
        return result_df
        
    except Exception as e:
        # Trả về DataFrame rỗng nếu có lỗi
        return pd.DataFrame()


def classify_route_type(route_name):
    """
    Phân loại route thành Nội địa hoặc Outbound dựa vào tên route
    """
    if pd.isna(route_name) or route_name == '':
        return 'Nội địa'
    
    route_str = str(route_name).upper()
    
    # Danh sách các từ khóa cho Outbound (quốc tế)
    outbound_keywords = [
        'TRUNG QUỐC', 'CHINA', 'THÁI LAN', 'THAILAND', 'HÀN QUỐC', 'KOREA', 
        'SINGAPORE', 'NHẬT BẢN', 'JAPAN', 'CHÂU ÂU', 'EUROPE', 'ĐÀI LOAN', 'TAIWAN',
        'ÚC', 'AUSTRALIA', 'NEW ZEALAND', 'MỸ', 'USA', 'CANADA', 'CAMPUCHIA', 'CAMBODIA',
        'LÀO', 'LAOS', 'MYANMAR', 'INDONESIA', 'MALAYSIA', 'PHILIPPINES', 'INDIA',
        'DUBAI', 'UAE', 'TURKEY', 'EGYPT', 'SOUTH AFRICA', 'BRAZIL', 'ARGENTINA'
    ]
    
    # Kiểm tra nếu route chứa từ khóa Outbound
    for keyword in outbound_keywords:
        if keyword in route_str:
            return 'Outbound'
    
    # Mặc định là Nội địa
    return 'Nội địa'



def create_route_performance_chart(route_data, metric='revenue', title=''):
    """
    Tạo horizontal bar chart cho route performance (không cần week_type)
    metric: 'revenue', 'num_customers', 'gross_profit'
    """
    if route_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="Không có dữ liệu", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig
    
    # Chọn metric và label tương ứng
    if metric == 'revenue':
        value_col = 'revenue'
        label = 'Doanh Thu'
        unit = 'tr.đ'
        # Chuyển đổi từ VND sang triệu đồng
        route_data = route_data.copy()
        route_data['value_display'] = route_data[value_col] / 1e6
        color = '#636EFA'  # Xanh dương
    elif metric == 'num_customers':
        value_col = 'num_customers'
        label = 'Lượt Khách'
        unit = 'LK'
        route_data = route_data.copy()
        route_data['value_display'] = route_data[value_col]
        color = '#FFA15A'  # Cam
    else:  # gross_profit
        value_col = 'gross_profit'
        label = 'Lãi Gộp'
        unit = 'tr.đ'
        # Chuyển đổi từ VND sang triệu đồng
        route_data = route_data.copy()
        route_data['value_display'] = route_data[value_col] / 1e6
        color = '#00CC96'  # Xanh lá
    
    # Loại bỏ các dòng có route là NaN hoặc "nan"
    route_data = route_data[
        (route_data['route'].notna()) &
        (route_data['route'].astype(str).str.strip() != '') &
        (~route_data['route'].astype(str).str.upper().str.contains('NAN', na=False))
    ].copy()
    
    # Loại bỏ các tuyến có giá trị = 0
    route_data = route_data[
        (route_data['value_display'].fillna(0) != 0)
    ].copy()
    
    if route_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="Không có dữ liệu", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig
    
    # Sắp xếp theo value giảm dần và lấy top routes
    # Sắp xếp giảm dần theo value_display, nếu bằng nhau thì sắp xếp theo route để đảm bảo thứ tự ổn định
    data = route_data.sort_values(
        by=['value_display', 'route'], 
        ascending=[False, True],  # value_display giảm dần, route tăng dần (alphabetical)
        kind='mergesort'
    ).head(20).copy()  # Top 20
    data = data.reset_index(drop=True)
    
    # Với horizontal bar chart trong Plotly:
    # - Phần tử đầu tiên trong list y sẽ ở DƯỚI cùng
    # - Phần tử cuối cùng trong list y sẽ ở TRÊN cùng
    # Data đã sắp xếp giảm dần (giá trị lớn nhất ở đầu), cần đảo ngược để giá trị lớn nhất ở cuối (hiển thị ở trên)
    data = data.iloc[::-1].reset_index(drop=True)
    
    # Tạo biểu đồ
    fig = go.Figure()
    
    # Format text trên bar - không có số thập phân và có dấu phẩy phân cách hàng nghìn
    if metric == 'num_customers':
        text_values = [f"{int(v):,}" for v in data['value_display']]
    else:
        text_values = [f"{int(v):,}" for v in data['value_display']]
    
    # Lấy lists - data đã được đảo ngược (giá trị lớn nhất ở cuối)
    x_values = data['value_display'].tolist()
    y_values = data['route'].tolist()
    
    # Vẽ biểu đồ với thứ tự đã được sắp xếp (giá trị lớn nhất ở cuối list y, sẽ hiển thị ở trên)
    # Format giá trị trong hovertemplate với dấu phẩy phân cách hàng nghìn
    if metric == 'num_customers':
        hover_values = [f"{int(v):,}" for v in x_values]
    else:
        hover_values = [f"{int(v):,}" for v in x_values]
    
    fig.add_trace(go.Bar(
        x=x_values,
        y=y_values,
        orientation='h',
        marker_color=color,
        text=text_values,
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Giá trị: %{customdata} ' + unit + '<extra></extra>',
        customdata=hover_values
    ))
    
    # Cập nhật layout - dùng categoryorder='total descending' để Plotly tự động sắp xếp từ cao xuống thấp
    # Với horizontal bar chart, 'total descending' sẽ đặt giá trị lớn nhất ở TRÊN cùng
    xaxis_title = f"ĐVT: {unit}"
    fig.update_layout(
        title=title if title else f"{label} theo Tuyến",
        xaxis_title=xaxis_title,
        yaxis_title="",
        height=max(400, len(data) * 25),
        margin=dict(l=10, r=100, t=40, b=20),
        showlegend=False,
        yaxis=dict(
            categoryorder='total descending'  # Tự động sắp xếp từ giá trị lớn nhất (ở trên) xuống nhỏ nhất (ở dưới)
        )
    )
    
    return fig


def load_route_plan_data(sheet_url, period_name='TẾT', region_filter=None):
    """
    Đọc dữ liệu kế hoạch tuyến từ Google Sheet
    Sheet URL Tết: https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=1651160424#gid=1651160424
    Sheet URL Xuân: https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=212301737#gid=212301737
    
    Cấu trúc CSV:
    - Dòng 1-2: Tiêu đề
    - Dòng 3: Số thứ tự
    - Dòng 4: Tên các đơn vị/khu vực (Công ty, Mien Bac, Mien Trung, Mien Nam, ...)
    - Dòng 5 (index 4): Header: Nhom tuyen, Tuyến Tour, LK, DT (tr.d), LG (tr.d)...
    - Dòng 6 trở đi: Dữ liệu
    
    Args:
        sheet_url: URL của Google Sheet
        period_name: Tên giai đoạn ('TẾT' hoặc 'KM XUÂN')
        region_filter: Tên khu vực để filter ('Tất cả', 'Mien Bac', 'Mien Trung', 'Mien Nam', hoặc None)
                      Nếu None hoặc 'Tất cả', sẽ lấy tổng Công ty (cột C, D, E)
                      Nếu có region_filter cụ thể, sẽ lấy cột tương ứng với khu vực đó
    
    Returns: DataFrame với columns: route, route_type, plan_customers, plan_revenue, plan_profit, period
    """
    import requests
    import io
    import re
    
    try:
        # Chuyển đổi URL thành CSV export URL
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return pd.DataFrame()
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else None
        
        if not gid:
            return pd.DataFrame()
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        # Tải CSV
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        
        # Đọc CSV
        text = resp.content.decode('utf-8', errors='replace')
        lines = [line.rstrip('\r') for line in text.split('\n')]  # Loại bỏ \r để tránh lỗi
        
        # Tìm dòng header (dòng 5, index 4) - chứa "Nhom tuyen" và "Tuyến Tour"
        header_idx = None
        for i, line in enumerate(lines[:10]):  # Chỉ tìm trong 10 dòng đầu
            line_upper = line.upper()
            if 'NHOM TUYEN' in line_upper or 'NHOM TUYẾN' in line_upper:
                if 'TUYẾN TOUR' in line_upper or 'TUYEN TOUR' in line_upper:
                    header_idx = i
                    break
        
        if header_idx is None:
            # Fallback: dùng dòng 5 (index 4)
            header_idx = 4 if len(lines) > 4 else 0
        
        # Đọc dòng 4 (index 3) TRƯỚC khi đọc DataFrame để có region_headers
        region_row_idx = 3  # Dòng 4 (index 3)
        region_headers = []
        if len(lines) > region_row_idx:
            try:
                # Đọc dòng 4 bằng pandas để parse CSV chính xác
                region_df = pd.read_csv(io.StringIO(lines[region_row_idx]), header=None, nrows=1)
                if not region_df.empty:
                    region_headers = [str(col).strip() for col in region_df.iloc[0].values]
            except Exception as e:
                # Fallback: dùng split đơn giản với CSV parser
                import csv
                try:
                    reader = csv.reader([lines[region_row_idx]])
                    region_headers = [col.strip().strip('"').strip("'") for col in next(reader)]
                except:
                    region_line = lines[region_row_idx]
                    region_headers = [col.strip().strip('"').strip("'") for col in region_line.split(',')]
        
        # Đọc từ dòng header
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])), skipinitialspace=True)
        
        if df.empty:
            return pd.DataFrame()
        
        # Tìm các cột cần thiết
        nhom_tuyen_col = None
        route_col = None
        
        for col in df.columns:
            col_upper = str(col).upper()
            if 'NHOM TUYEN' in col_upper or 'NHOM TUYẾN' in col_upper:
                nhom_tuyen_col = col
            elif 'TUYẾN TOUR' in col_upper or 'TUYEN TOUR' in col_upper:
                route_col = col
        
        # Nếu không tìm thấy bằng tên, dùng vị trí cột (theo CSV mẫu)
        # Cấu trúc CSV: A=Khu vực Đơn Vị, B=Đơn Vị, C=Dom/Out..., D=Nhóm tuyến, E=Tuyến Tour
        if nhom_tuyen_col is None and len(df.columns) > 3:
            nhom_tuyen_col = df.columns[3]  # Cột D (index 3): Nhóm tuyến
        if route_col is None and len(df.columns) > 4:
            route_col = df.columns[4]  # Cột E (index 4): Tuyến Tour
        
        if route_col is None:
            return pd.DataFrame()
        
        # region_headers đã được đọc ở trên (dòng 1146-1163), không cần đọc lại
        
        # Xác định cột cần lấy dựa trên region_filter
        customers_col = None
        revenue_col = None
        profit_col = None
        
        # Chuẩn hóa region_filter
        if region_filter and region_filter != 'Tất cả':
            region_filter_upper = str(region_filter).upper().strip()
            # Map tên khu vực
            region_mapping = {
                'MIEN BAC': 'MIEN BAC',
                'MIỀN BẮC': 'MIEN BAC',
                'MIEN TRUNG': 'MIEN TRUNG',
                'MIỀN TRUNG': 'MIEN TRUNG',
                'MIEN NAM': 'MIEN NAM',
                'MIỀN NAM': 'MIEN NAM',
                'TPHCM & DNB': 'TPHCM & DNB',
                'TPHCM & DNB': 'TPHCM & DNB',
                'TPHCM DNB': 'TPHCM & DNB',
                'TPHCM VÀ DNB': 'TPHCM & DNB',
                'MIEN TAY': 'MIEN TAY',
                'MIỀN TÂY': 'MIEN TAY',
                'MIENTAY': 'MIEN TAY'
            }
            target_region = region_mapping.get(region_filter_upper, region_filter_upper)
        else:
            target_region = None  # Dùng tổng Công ty
        
        # Tìm cột LK, DT, LG dựa trên region_filter
        if target_region is None:
            # Dùng tổng Công ty: Cột C, D, E (index 2, 3, 4)
            if len(df.columns) > 2:
                customers_col = df.columns[2]  # Cột C: LK
            if len(df.columns) > 3:
                revenue_col = df.columns[3]  # Cột D: DT (tr.d)
            if len(df.columns) > 4:
                profit_col = df.columns[4]  # Cột E: LG (tr.d)
        else:
            # Tìm cột của khu vực cụ thể
            # Cách tiếp cận: Tìm trực tiếp các cột LK, DT, LG của khu vực trong DataFrame
            # Cấu trúc: Cột C, D, E (index 2, 3, 4) = Công ty
            # Cột F, G, H (index 5, 6, 7) = Miền Bắc (nếu có)
            # Cột I, J, K (index 8, 9, 10) = Miền Trung (nếu có)
            # ...
            
            # Tìm vị trí của khu vực trong region_headers (dòng 4) để tính offset
            # Từ file CSV thực tế: dòng 4 có ",,Công ty,Công ty,Công ty,Mien Bac,Mien Bac,Mien Bac,Mien Trung,..."
            # Vậy index trong region_headers:
            # - Index 0, 1: trống (cột A, B)
            # - Index 2, 3, 4: "Công ty" (3 lần) -> region_idx_in_headers = 1 (vì "Công ty" xuất hiện đầu tiên ở index 2)
            # - Index 5, 6, 7: "Mien Bac" (3 lần) -> region_idx_in_headers = 2
            # - Index 8, 9, 10: "Mien Trung" (3 lần) -> region_idx_in_headers = 3
            # Nhưng thực tế, mỗi khu vực xuất hiện 3 lần (cho LK, DT, LG), nên cần tìm index đầu tiên của khu vực đó
            
            region_idx_in_headers = None
            if region_headers:
                # Tìm index đầu tiên của khu vực trong region_headers
                # Mỗi khu vực xuất hiện 3 lần liên tiếp (cho LK, DT, LG)
                for i, header in enumerate(region_headers):
                    header_upper = str(header).upper().strip()
                    # Bỏ qua các cột trống hoặc không phải tên khu vực
                    if not header_upper or header_upper in ['', 'NAN', 'NONE']:
                        continue
                    
                    # Kiểm tra match với target_region
                    if target_region == 'MIEN BAC':
                        if ('MIEN BAC' in header_upper or 'MIỀN BẮC' in header_upper or 
                            'MIENBAC' in header_upper or 
                            (header_upper.startswith('MIEN') and 'BAC' in header_upper and 'TRUNG' not in header_upper) or
                            (header_upper.startswith('MIỀN') and 'BẮC' in header_upper and 'TRUNG' not in header_upper)):
                            # Tìm index đầu tiên của khu vực này (bỏ qua các cột trống ở đầu)
                            # Trong region_headers, "Công ty" bắt đầu từ index 2, "Mien Bac" từ index 5
                            # Nhưng cần tính lại dựa trên vị trí thực tế
                            region_idx_in_headers = i
                            break
                    elif target_region == 'MIEN TRUNG':
                        if ('MIEN TRUNG' in header_upper or 'MIỀN TRUNG' in header_upper or 
                            'MIENTRUNG' in header_upper or
                            (header_upper.startswith('MIEN') and 'TRUNG' in header_upper) or
                            (header_upper.startswith('MIỀN') and 'TRUNG' in header_upper)):
                            region_idx_in_headers = i
                            break
                    elif target_region == 'MIEN NAM':
                        if ('MIEN NAM' in header_upper or 'MIỀN NAM' in header_upper or 
                            'MIENNAM' in header_upper or
                            (header_upper.startswith('MIEN') and 'NAM' in header_upper and 'BAC' not in header_upper and 'TRUNG' not in header_upper) or
                            (header_upper.startswith('MIỀN') and 'NAM' in header_upper and 'BẮC' not in header_upper and 'TRUNG' not in header_upper)):
                            region_idx_in_headers = i
                            break
                    elif target_region == 'TPHCM & DNB':
                        if ('TPHCM' in header_upper and 'DNB' in header_upper) or \
                           ('TPHCM' in header_upper and ('&' in header_upper or 'VÀ' in header_upper or 'VA' in header_upper)) or \
                           ('TPHCM' in header_upper and 'DNB' in header_upper) or \
                           ('HO CHI MINH' in header_upper and 'DNB' in header_upper):
                            region_idx_in_headers = i
                            break
                    elif target_region == 'MIEN TAY':
                        if ('MIEN TAY' in header_upper or 'MIỀN TÂY' in header_upper or 
                            'MIENTAY' in header_upper or
                            (header_upper.startswith('MIEN') and 'TAY' in header_upper and 'BAC' not in header_upper and 'TRUNG' not in header_upper and 'NAM' not in header_upper) or
                            (header_upper.startswith('MIỀN') and 'TÂY' in header_upper and 'BẮC' not in header_upper and 'TRUNG' not in header_upper and 'NAM' not in header_upper)):
                            region_idx_in_headers = i
                            break
            
            # Tính offset dựa trên vị trí trong region_headers
            # Cấu trúc: Dòng 4 có "Công ty", "Miền Bắc", "Miền Trung", ...
            # Dòng 5 có "Nhom tuyen", "Tuyến Tour", "LK", "DT (tr.d)", "LG (tr.d)", "LK", "DT (tr.d)", "LG (tr.d)", ...
            # Cột A (index 0): Nhom tuyen
            # Cột B (index 1): Tuyến Tour  
            # Cột C, D, E (index 2, 3, 4): Công ty (LK, DT, LG) - region_idx_in_headers = 0
            # Cột F, G, H (index 5, 6, 7): Miền Bắc (LK, DT, LG) - region_idx_in_headers = 1
            # Cột I, J, K (index 8, 9, 10): Miền Trung (LK, DT, LG) - region_idx_in_headers = 2
            # Công thức: col_offset = 2 + region_idx_in_headers * 3
            # Miền Bắc (region_idx_in_headers = 1): col_offset = 2 + 1 * 3 = 5
            # Miền Trung (region_idx_in_headers = 2): col_offset = 2 + 2 * 3 = 8
            # Từ file CSV thực tế:
            # region_headers[5] = "Mien Bac" (index đầu tiên của Mien Bac)
            # DataFrame.columns[5] = "LK" hoặc "LK.1" (Mien Bac)
            # Vậy col_offset = region_idx_in_headers
            if region_idx_in_headers is not None:
                col_offset = region_idx_in_headers
                
                # Kiểm tra xem có đủ cột không
                if len(df.columns) > col_offset:
                    customers_col = df.columns[col_offset]  # LK
                if len(df.columns) > col_offset + 1:
                    revenue_col = df.columns[col_offset + 1]  # DT
                if len(df.columns) > col_offset + 2:
                    profit_col = df.columns[col_offset + 2]  # LG
                
                # Debug: In ra để kiểm tra
                # print(f"DEBUG: region_idx_in_headers={region_idx_in_headers}, col_offset={col_offset}")
                # print(f"DEBUG: customers_col={customers_col}, revenue_col={revenue_col}, profit_col={profit_col}")
                # print(f"DEBUG: df.columns={list(df.columns)}")
                # print(f"DEBUG: region_headers={region_headers}")
            
            # Fallback: Tìm bằng cách duyệt qua các cột và tìm cột LK, DT, LG
            # QUAN TRỌNG: Khi pandas đọc CSV có nhiều cột trùng tên, nó sẽ tự động thêm suffix
            # Ví dụ: "LK" (Công ty), "LK.1" (Miền Bắc), "LK.2" (Miền Trung)
            if customers_col is None or revenue_col is None or profit_col is None:
                # Tìm các cột LK, DT, LG theo thứ tự và vị trí
                lk_cols = []
                dt_cols = []
                lg_cols = []
                
                for idx, col in enumerate(df.columns):
                    if col == nhom_tuyen_col or col == route_col:
                        continue
                    col_str = str(col).strip()
                    col_upper = col_str.upper()
                    col_idx = idx
                    
                    # Tìm cột LK (có thể là "LK", "LK.1", "LK.2", ...)
                    if col_upper == 'LK' or col_upper.startswith('LK.') or 'LƯỢT KHÁCH' in col_upper:
                        lk_cols.append((col_idx, col))
                    # Tìm cột DT (có thể là "DT (tr.d)", "DT (tr.d).1", "DT (tr.d).2", ...)
                    elif 'DT (TR.D)' in col_upper or 'DT(TR.D)' in col_upper or (col_upper.startswith('DT') and ('TR.D' in col_upper or 'TRD' in col_upper)):
                        dt_cols.append((col_idx, col))
                    # Tìm cột LG (có thể là "LG (tr.d)", "LG (tr.d).1", "LG (tr.d).2", ...)
                    elif 'LG (TR.D)' in col_upper or 'LG(TR.D)' in col_upper or (col_upper.startswith('LG') and ('TR.D' in col_upper or 'TRD' in col_upper)):
                        lg_cols.append((col_idx, col))
                
                # Xác định cột nào thuộc về khu vực đã chọn dựa trên vị trí
                # Công ty: cột đầu tiên (index 2, 3, 4) - không có suffix
                # Miền Bắc: cột thứ hai (index 5, 6, 7) - có suffix .1 hoặc là cột thứ 2
                # Miền Trung: cột thứ ba (index 8, 9, 10) - có suffix .2 hoặc là cột thứ 3
                if target_region == 'MIEN BAC':
                    # Lấy cột LK, DT, LG thứ hai (sau Công ty)
                    # Ưu tiên: cột có suffix .1, nếu không có thì lấy cột có index trong khoảng 5-7
                    for col_idx, col_name in lk_cols:
                        col_name_str = str(col_name)
                        if '.1' in col_name_str:
                            customers_col = col_name
                            break
                        elif col_idx >= 5 and col_idx < 8 and customers_col is None:
                            customers_col = col_name
                    for col_idx, col_name in dt_cols:
                        col_name_str = str(col_name)
                        if '.1' in col_name_str:
                            revenue_col = col_name
                            break
                        elif col_idx >= 5 and col_idx < 8 and revenue_col is None:
                            revenue_col = col_name
                    for col_idx, col_name in lg_cols:
                        col_name_str = str(col_name)
                        if '.1' in col_name_str:
                            profit_col = col_name
                            break
                        elif col_idx >= 5 and col_idx < 8 and profit_col is None:
                            profit_col = col_name
                elif target_region == 'MIEN TRUNG':
                    # Lấy cột LK, DT, LG thứ ba
                    for col_idx, col_name in lk_cols:
                        col_name_str = str(col_name)
                        if '.2' in col_name_str:
                            customers_col = col_name
                            break
                        elif col_idx >= 8 and col_idx < 11 and customers_col is None:
                            customers_col = col_name
                    for col_idx, col_name in dt_cols:
                        col_name_str = str(col_name)
                        if '.2' in col_name_str:
                            revenue_col = col_name
                            break
                        elif col_idx >= 8 and col_idx < 11 and revenue_col is None:
                            revenue_col = col_name
                    for col_idx, col_name in lg_cols:
                        col_name_str = str(col_name)
                        if '.2' in col_name_str:
                            profit_col = col_name
                            break
                        elif col_idx >= 8 and col_idx < 11 and profit_col is None:
                            profit_col = col_name
                elif target_region == 'MIEN NAM':
                    # Lấy cột LK, DT, LG thứ tư
                    for col_idx, col_name in lk_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str:
                            customers_col = col_name
                            break
                        elif col_idx >= 11 and customers_col is None:
                            customers_col = col_name
                    for col_idx, col_name in dt_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str:
                            revenue_col = col_name
                            break
                        elif col_idx >= 11 and revenue_col is None:
                            revenue_col = col_name
                    for col_idx, col_name in lg_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str:
                            profit_col = col_name
                            break
                        elif col_idx >= 11 and profit_col is None:
                            profit_col = col_name
                elif target_region == 'TPHCM & DNB':
                    # Lấy cột LK, DT, LG thứ năm (index 11, 12, 13) - có suffix .3 hoặc là cột thứ 4
                    for col_idx, col_name in lk_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str and col_idx >= 11:
                            customers_col = col_name
                            break
                        elif col_idx >= 11 and col_idx < 14 and customers_col is None:
                            customers_col = col_name
                    for col_idx, col_name in dt_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str and col_idx >= 12:
                            revenue_col = col_name
                            break
                        elif col_idx >= 12 and col_idx < 14 and revenue_col is None:
                            revenue_col = col_name
                    for col_idx, col_name in lg_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str and col_idx >= 13:
                            profit_col = col_name
                            break
                        elif col_idx >= 13 and col_idx < 14 and profit_col is None:
                            profit_col = col_name
                elif target_region == 'MIEN TAY':
                    # Lấy cột LK, DT, LG thứ sáu (index 14, 15, 16) - có suffix .4 hoặc là cột thứ 5
                    for col_idx, col_name in lk_cols:
                        col_name_str = str(col_name)
                        if '.4' in col_name_str:
                            customers_col = col_name
                            break
                        elif col_idx >= 14 and col_idx < 17 and customers_col is None:
                            customers_col = col_name
                    for col_idx, col_name in dt_cols:
                        col_name_str = str(col_name)
                        if '.4' in col_name_str:
                            revenue_col = col_name
                            break
                        elif col_idx >= 15 and col_idx < 17 and revenue_col is None:
                            revenue_col = col_name
                    for col_idx, col_name in lg_cols:
                        col_name_str = str(col_name)
                        if '.4' in col_name_str:
                            profit_col = col_name
                            break
                        elif col_idx >= 16 and col_idx < 17 and profit_col is None:
                            profit_col = col_name
        
        # Fallback cuối cùng: CHỈ dùng cột C, D, E (Công ty) nếu KHÔNG có region_filter
        # Nếu có region_filter nhưng không tìm thấy cột, KHÔNG fallback về Công ty
        # (vì điều này sẽ làm sai dữ liệu khi filter theo khu vực)
        if target_region is None:
            # Chỉ fallback về Công ty nếu không có region_filter
            if customers_col is None and len(df.columns) > 2:
                customers_col = df.columns[2]
            if revenue_col is None and len(df.columns) > 3:
                revenue_col = df.columns[3]
            if profit_col is None and len(df.columns) > 4:
                profit_col = df.columns[4]
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame()
        result_df['nhom_tuyen'] = df[nhom_tuyen_col].astype(str).str.strip() if nhom_tuyen_col else ''
        result_df['route'] = df[route_col].astype(str).str.strip()
        
        # QUAN TRỌNG: "Dom total", "Out Total", "Grand total" nằm ở cột D (Nhóm tuyến), KHÔNG PHẢI cột E (Tuyến Tour)
        # Cột E (Tuyến Tour) của các dòng này là TRỐNG
        # Nếu route trống hoặc không có giá trị, nhưng nhom_tuyen có "Dom total" hoặc "Out Total", 
        # dùng nhom_tuyen làm route
        if nhom_tuyen_col and len(result_df) > 0:
            # Đảm bảo nhom_tuyen là Series hợp lệ
            nhom_tuyen_series = result_df['nhom_tuyen'].fillna('').astype(str).str.strip()
            route_series = result_df['route'].fillna('').astype(str).str.strip()
            
            # Tìm các dòng có nhom_tuyen chứa "Dom total", "Out Total", hoặc "Grand total"
            mask_total = nhom_tuyen_series.str.contains('Dom total|Out Total|Grand total', case=False, na=False)
            
            # Nếu route trống HOẶC route không chứa "Dom total"/"Out Total"/"Grand total", 
            # nhưng nhom_tuyen có, thì dùng nhom_tuyen làm route
            mask_use_nhom = (
                mask_total & 
                ((route_series == '') | (~route_series.str.contains('Dom total|Out Total|Grand total', case=False, na=False)))
            )
            
            if mask_use_nhom.any():
                result_df.loc[mask_use_nhom, 'route'] = nhom_tuyen_series[mask_use_nhom]
        
        # Parse số liệu
        def parse_value(val):
            if pd.isna(val) or val == '' or str(val).strip() == '-' or str(val).strip() == 'nan':
                return 0
            val_str = str(val).strip().replace('"', '')
            
            # Xử lý các trường hợp:
            # 1. Dấu phẩy làm dấu phân cách hàng nghìn: "30,580" -> "30580"
            # 2. Dấu chấm làm dấu phân cách hàng nghìn: "30.580" -> "30580"
            # 3. Dấu chấm làm dấu thập phân: "123.45" -> "123.45"
            
            # Nếu có cả dấu phẩy và dấu chấm: dấu phẩy là phân cách hàng nghìn, dấu chấm là thập phân
            if ',' in val_str and '.' in val_str:
                # Ví dụ: "30,580.50" -> "30580.50"
                val_str = val_str.replace(',', '')
            # Nếu chỉ có dấu phẩy: dấu phẩy là phân cách hàng nghìn
            elif ',' in val_str:
                val_str = val_str.replace(',', '')
            # Nếu chỉ có dấu chấm: cần kiểm tra xem là phân cách hàng nghìn hay thập phân
            elif '.' in val_str:
                parts = val_str.split('.')
                if len(parts) == 2:
                    # Nếu phần sau dấu chấm có <= 2 chữ số -> là số thập phân
                    if len(parts[1]) <= 2:
                        # Giữ nguyên (ví dụ: "123.45")
                        pass
                    else:
                        # Phần sau dấu chấm có > 2 chữ số -> là dấu phân cách hàng nghìn
                        # Ví dụ: "30.580" -> "30580"
                        val_str = val_str.replace('.', '')
                else:
                    # Có nhiều dấu chấm -> tất cả đều là phân cách hàng nghìn
                    # Ví dụ: "1.234.567" -> "1234567"
                    val_str = val_str.replace('.', '')
            
            try:
                return float(val_str)
            except:
                return 0
        
        # Lấy giá trị từ cột tổng Công ty (cột C, D, E) - đã là tổng của tất cả khu vực
        if customers_col:
            result_df['plan_customers'] = df[customers_col].apply(parse_value)
        else:
            result_df['plan_customers'] = 0
        
        # LƯU Ý: Dữ liệu từ DEFAULT_PLAN_TET_URL và DEFAULT_PLAN_XUAN_URL có đơn vị DT và LG là triệu đồng (tr.đ)
        # Chuyển đổi sang VND để tính toán thống nhất với các nguồn dữ liệu khác
        if revenue_col:
            result_df['plan_revenue'] = df[revenue_col].apply(parse_value) * 1_000_000  # Chuyển từ triệu đồng sang VND
        else:
            result_df['plan_revenue'] = 0
        
        if profit_col:
            result_df['plan_profit'] = df[profit_col].apply(parse_value) * 1_000_000  # Chuyển từ triệu đồng sang VND
        else:
            result_df['plan_profit'] = 0
        
        # Loại bỏ dòng trống, nhưng GIỮ LẠI "Dom Total" và "Out Total" để tính phần trăm
        # Chỉ loại bỏ "Grand total" và các dòng Total khác
        # Tìm "Dom total" và "Out Total" trong cả route và nhom_tuyen
        has_dom_total = result_df['route'].astype(str).str.contains('Dom total|Dom Total', case=False, na=False)
        has_out_total = result_df['route'].astype(str).str.contains('Out Total|Out total', case=False, na=False)
        has_dom_total_nhom = result_df['nhom_tuyen'].astype(str).str.contains('Dom total|Dom Total', case=False, na=False)
        has_out_total_nhom = result_df['nhom_tuyen'].astype(str).str.contains('Out Total|Out total', case=False, na=False)
        
        # Giữ lại nếu route hoặc nhom_tuyen có "Dom total" hoặc "Out Total"
        keep_total_rows = has_dom_total | has_out_total | has_dom_total_nhom | has_out_total_nhom
        
        # Sau khi chuyển nhom_tuyen sang route ở trên, route sẽ chứa "Dom total" hoặc "Out Total"
        # Nhưng cần đảm bảo route không trống (nếu route trống nhưng nhom_tuyen có "Dom total"/"Out Total", 
        # thì route đã được set = nhom_tuyen ở trên)
        route_str = result_df['route'].fillna('').astype(str).str.strip()
        nhom_tuyen_str = result_df['nhom_tuyen'].fillna('').astype(str).str.strip()
        
        result_df = result_df[
            # Route không trống (sau khi đã chuyển từ nhom_tuyen nếu cần)
            (route_str != '') &
            (
                # Giữ lại "Dom total" và "Out Total" (từ route hoặc nhom_tuyen)
                keep_total_rows |
                # Loại bỏ các dòng Total khác (nhưng không phải "Dom total" và "Out Total")
                (
                    (~route_str.str.contains('Total', case=False, na=False)) &
                    (~route_str.str.contains('Grand total', case=False, na=False)) &
                    (~nhom_tuyen_str.str.contains('Total', case=False, na=False))
                )
            )
        ].copy()
        
        # Phân loại route_type: Nội địa vs Outbound
        domestic_keywords = ['MIEN BAC', 'MIEN TRUNG', 'MIEN NAM', 'DOM TOTAL', 'BẮC TRUNG BỘ', 'TÂY NGUYÊN', 'NHA TRANG', 'QUY NHƠN', 'NAM TRUNG BỘ', 'PHÚ QUỐC', 'MIỀN TÂY', 'ĐÔNG NAM BỘ', 'LIÊN TUYẾN']
        outbound_keywords = [
            # Châu Á
            'CHAU A', 'DONG BAC A', 'DONG NAM A', 'TAY A', 'NAM A', 
            'CHÂU Á', 'ĐÔNG BẮC Á', 'ĐÔNG NAM Á', 'TÂY Á', 'NAM Á',
            # Châu Âu, Úc, Mỹ, Phi
            'CHAU AU', 'CHÂU ÂU', 'CHAU UC', 'CHÂU ÚC', 'CHAU MY', 'CHÂU MỸ', 'CHAU PHI', 'CHÂU PHI',
            'EUROPE', 'AUSTRALIA', 'AMERICA',
            # Các nước cụ thể
            'OUT TOTAL', 'TRUNG QUỐC', 'NHẬT BẢN', 'ĐÀI LOAN', 'HÀN QUỐC', 'HONG KONG', 
            'THÁI LAN', 'SINGAPORE', 'MALAYSIA', 'SING - MÃ', 'SING - MA', 'SING-MÃ', 'SING-MA',
            'INDONESIA', 'LÀO', 'CAMPUCHIA', 'BRUNEI', 'MỸ', 'CANADA'
        ]
        
        def classify_route_type(nhom_tuyen, route):
            nhom_upper = str(nhom_tuyen).upper()
            route_upper = str(route).upper()
            
            # Chuẩn hóa để so sánh (loại bỏ dấu và khoảng trắng)
            def normalize_for_match(text):
                text = text.replace('Â', 'A').replace('Á', 'A').replace('À', 'A').replace('Ạ', 'A').replace('Ã', 'A')
                text = text.replace('Ê', 'E').replace('É', 'E').replace('È', 'E').replace('Ẹ', 'E')
                text = text.replace('Ô', 'O').replace('Ố', 'O').replace('Ồ', 'O').replace('Ọ', 'O')
                text = text.replace('Ơ', 'O').replace('Ớ', 'O').replace('Ờ', 'O').replace('Ợ', 'O')
                text = text.replace('Ư', 'U').replace('Ứ', 'U').replace('Ừ', 'U').replace('Ự', 'U')
                text = text.replace('Đ', 'D').replace('Ý', 'Y').replace('Ỳ', 'Y').replace('Ỵ', 'Y')
                text = text.replace('-', ' ').replace('_', ' ')
                text = ' '.join(text.split())  # Loại bỏ khoảng trắng thừa
                return text
            
            nhom_normalized = normalize_for_match(nhom_upper)
            route_normalized = normalize_for_match(route_upper)
            
            # Kiểm tra Outbound trước
            for keyword in outbound_keywords:
                keyword_normalized = normalize_for_match(keyword.upper())
                if keyword_normalized in nhom_normalized or keyword_normalized in route_normalized:
                    return 'Outbound'
                # Kiểm tra với keyword gốc (có dấu)
                if keyword in nhom_upper or keyword in route_upper:
                    return 'Outbound'
            
            # Kiểm tra Nội địa
            for keyword in domestic_keywords:
                keyword_normalized = normalize_for_match(keyword.upper())
                if keyword_normalized in nhom_normalized or keyword_normalized in route_normalized:
                    return 'Nội địa'
                # Kiểm tra với keyword gốc (có dấu)
                if keyword in nhom_upper or keyword in route_upper:
                    return 'Nội địa'
            
            # Mặc định là Nội địa
            return 'Nội địa'
        
        result_df['route_type'] = result_df.apply(lambda row: classify_route_type(row['nhom_tuyen'], row['route']), axis=1)
        result_df['period'] = period_name
        
        # QUAN TRỌNG: Loại bỏ các dòng có route là "Dom Total", "Out Total", "Grand Total" trước khi groupby
        # Vì các dòng này không phải là route cụ thể và có thể gây nhầm lẫn
        result_df = result_df[
            ~result_df['route'].astype(str).str.contains('Dom Total|Out Total|Grand Total', case=False, na=False)
        ].copy()
        
        # QUAN TRỌNG: Nếu có nhiều dòng cho cùng một route (có thể từ các nguồn khác nhau),
        # lấy giá trị lớn nhất để đảm bảo lấy đúng giá trị từ tổng Công ty
        # (giá trị tổng Công ty thường lớn hơn giá trị từ các khu vực cụ thể)
        result_df = result_df.groupby(['route', 'route_type', 'period']).agg({
            'plan_customers': 'max',  # Lấy giá trị lớn nhất
            'plan_revenue': 'max',    # Lấy giá trị lớn nhất (thường là tổng Công ty)
            'plan_profit': 'max',     # Lấy giá trị lớn nhất
            'nhom_tuyen': 'first'     # Giữ lại nhom_tuyen (lấy giá trị đầu tiên)
        }).reset_index()
        
        # Chỉ giữ lại các cột cần thiết (sau khi groupby)
        result_df = result_df[['route', 'route_type', 'plan_customers', 'plan_revenue', 'plan_profit', 'period', 'nhom_tuyen']].copy()
        
        return result_df
        
    except Exception as e:
        return pd.DataFrame()


def load_total_plan_data(sheet_url, period_name='TẾT', region_name=None):
    """
    Đọc tổng kế hoạch từ Google Sheet (Dom Total và Out Total)
    Sheet URL Tết: https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=1651160424#gid=1651160424
    Sheet URL Xuân: https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=212301737#gid=212301737
    
    Cấu trúc CSV:
    - Dòng 5 (index 4): Header: Nhom tuyen, Tuyến Tour, LK, DT (tr.d), LG (tr.d)...
    - Dòng 17: "Dom Total" - tổng Nội địa 
      - Cột C: LK (Công ty), Cột D: DT (Công ty)
      - Cột F: LK (Miền Bắc), Cột G: DT (Miền Bắc)
      - Cột I: LK (Miền Trung), Cột J: DT (Miền Trung)
      - Cột L: LK (TPHCM & DNB), Cột M: DT (TPHCM & DNB)
      - Cột O: LK (Miền Tây), Cột P: DT (Miền Tây)
    - Dòng 39: "Out Total" - tổng Outbound (tương tự)
    
    Args:
        sheet_url: URL của Google Sheet
        period_name: Tên giai đoạn ('TẾT' hoặc 'KM XUÂN')
        region_name: Tên khu vực ('Miền Bắc', 'Miền Trung', 'TPHCM & DNB', 'Miền Tây') hoặc None để lấy tổng Công ty
    
    Returns: Dictionary với keys: 'dom_lk', 'dom_dt', 'out_lk', 'out_dt' (đơn vị: LK và tr.d)
    """
    import requests
    import io
    import re
    
    try:
        # Chuyển đổi URL thành CSV export URL
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return {'dom_lk': 0, 'dom_dt': 0, 'out_lk': 0, 'out_dt': 0}
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else None
        
        if not gid:
            return {'dom_lk': 0, 'dom_dt': 0, 'out_lk': 0, 'out_dt': 0}
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        # Tải CSV
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        
        # Đọc CSV
        text = resp.content.decode('utf-8', errors='replace')
        lines = [line.rstrip('\r') for line in text.split('\n')]
        
        # Tìm dòng header (dòng 5, index 4)
        header_idx = None
        for i, line in enumerate(lines[:10]):
            line_upper = line.upper()
            if 'NHOM TUYEN' in line_upper or 'NHOM TUYẾN' in line_upper:
                if 'TUYẾN TOUR' in line_upper or 'TUYEN TOUR' in line_upper:
                    header_idx = i
                    break
        
        if header_idx is None:
            header_idx = 4 if len(lines) > 4 else 0
        
        # Đọc từ dòng header
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])), skipinitialspace=True)
        
        if df.empty:
            return {'dom_lk': 0, 'dom_dt': 0, 'out_lk': 0, 'out_dt': 0}
        
        # Tìm cột Nhóm tuyến (cột A) và Tuyến Tour (cột B)
        nhom_tuyen_col = None
        route_col = None
        for col in df.columns:
            col_upper = str(col).upper()
            if 'NHOM TUYEN' in col_upper or 'NHOM TUYẾN' in col_upper:
                nhom_tuyen_col = col
            elif 'TUYẾN TOUR' in col_upper or 'TUYEN TOUR' in col_upper:
                route_col = col
        
        if nhom_tuyen_col is None and len(df.columns) > 0:
            nhom_tuyen_col = df.columns[0]  # Cột A
        if route_col is None and len(df.columns) > 1:
            route_col = df.columns[1]  # Cột B
        
        # Map tên khu vực với chỉ số cột (LK, DT)
        # Cột C (index 2): LK Công ty, Cột D (index 3): DT Công ty
        # Cột F (index 5): LK Miền Bắc, Cột G (index 6): DT Miền Bắc
        # Cột I (index 8): LK Miền Trung, Cột J (index 9): DT Miền Trung
        # Cột L (index 11): LK TPHCM & DNB, Cột M (index 12): DT TPHCM & DNB
        # Cột O (index 14): LK Miền Tây, Cột P (index 15): DT Miền Tây
        region_col_map = {
            'Miền Bắc': {'lk': 5, 'dt': 6},  # Cột F, G
            'Miền Trung': {'lk': 8, 'dt': 9},  # Cột I, J
            'TPHCM & DNB': {'lk': 11, 'dt': 12},  # Cột L, M
            'Miền Tây': {'lk': 14, 'dt': 15},  # Cột O, P
        }
        
        # Xác định cột cần lấy
        if region_name and region_name in region_col_map:
            # Lấy từ cột khu vực
            lk_col_idx = region_col_map[region_name]['lk']
            dt_col_idx = region_col_map[region_name]['dt']
        else:
            # Lấy từ cột Công ty (mặc định)
            lk_col_idx = 2  # Cột C
            dt_col_idx = 3  # Cột D
        
        lk_col = df.columns[lk_col_idx] if len(df.columns) > lk_col_idx else None
        dt_col = df.columns[dt_col_idx] if len(df.columns) > dt_col_idx else None
        
        if not lk_col or not dt_col:
            return {'dom_lk': 0, 'dom_dt': 0, 'out_lk': 0, 'out_dt': 0}
        
        # Parse số liệu
        def parse_value(val):
            if pd.isna(val) or val == '' or val is None:
                return 0
            val_str = str(val).strip().replace(',', '').replace(' ', '')
            try:
                return float(val_str)
            except:
                return 0
        
        # Tìm dòng "Dom Total" và "Out Total"
        # Ưu tiên tìm trong cột B (Tuyến Tour) vì "Dom Total" và "Out Total" thường nằm ở đó
        dom_lk = 0
        dom_dt = 0
        out_lk = 0
        out_dt = 0
        
        for idx, row in df.iterrows():
            # Kiểm tra trong cột Tuyến Tour (cột B) trước - ưu tiên cao hơn
            route = ''
            if route_col:
                route = str(row[route_col]).strip().upper() if pd.notna(row[route_col]) else ''
            
            # Kiểm tra trong cột Nhóm tuyến (cột A)
            nhom_tuyen = ''
            if nhom_tuyen_col:
                nhom_tuyen = str(row[nhom_tuyen_col]).strip().upper() if pd.notna(row[nhom_tuyen_col]) else ''
            
            # Tìm "Dom Total" - ưu tiên cột B (Tuyến Tour)
            is_dom_total = False
            if route and ('DOM TOTAL' in route or 'DOMTOTAL' in route):
                is_dom_total = True
            elif nhom_tuyen and ('DOM TOTAL' in nhom_tuyen or 'DOMTOTAL' in nhom_tuyen):
                is_dom_total = True
            
            if is_dom_total and dom_lk == 0 and dom_dt == 0:
                dom_lk = parse_value(row[lk_col])
                dom_dt = parse_value(row[dt_col])
                continue  # Đã tìm thấy, không cần kiểm tra Out Total nữa
            
            # Tìm "Out Total" - ưu tiên cột B (Tuyến Tour)
            is_out_total = False
            if route and ('OUT TOTAL' in route or 'OUTTOTAL' in route):
                is_out_total = True
            elif nhom_tuyen and ('OUT TOTAL' in nhom_tuyen or 'OUTTOTAL' in nhom_tuyen):
                is_out_total = True
            
            if is_out_total and out_lk == 0 and out_dt == 0:
                out_lk = parse_value(row[lk_col])
                out_dt = parse_value(row[dt_col])
        
        return {
            'dom_lk': dom_lk,
            'dom_dt': dom_dt,  # Đơn vị: tr.d (triệu đồng)
            'out_lk': out_lk,
            'out_dt': out_dt   # Đơn vị: tr.d (triệu đồng)
        }
        
    except Exception as e:
        return {'dom_lk': 0, 'dom_dt': 0, 'out_lk': 0, 'out_dt': 0}


def create_completion_progress_chart(completion_data, title=''):
    """
    Tạo biểu đồ bar chart hiển thị tiến độ hoàn thành kế hoạch
    completion_data: DataFrame với columns: route, completion_customers, completion_revenue, completion_profit
    """
    if completion_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="Không có dữ liệu", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig

    # Loại bỏ các dòng "Out Total", "Dom Total", "Grand Total" khỏi biểu đồ
    completion_data = completion_data[
        ~completion_data['route'].astype(str).str.contains('Grand Total|Dom Total|Out Total', case=False, na=False)
    ].copy()
    
    # Loại bỏ các tuyến có 0% cho cả 3 chỉ số (Lượt khách, Doanh thu, Lãi Gộp)
    completion_data = completion_data[
        ~((completion_data['completion_customers'].fillna(0) == 0) & 
          (completion_data['completion_revenue'].fillna(0) == 0) & 
          (completion_data['completion_profit'].fillna(0) == 0))
    ].copy()
    
    if completion_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="Không có dữ liệu", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig

    # Sắp xếp theo completion_revenue giảm dần (từ cao xuống thấp)
    completion_data = completion_data.sort_values('completion_revenue', ascending=False).copy()
    
    routes = completion_data['route'].tolist()
    
    fig = go.Figure()
    
    # Cột Lượt khách (màu nâu/cam)
    fig.add_trace(go.Bar(
        x=routes,
        y=completion_data['completion_customers'].values,
        name='Lượt khách',
        marker_color='#FFA15A',  # Màu cam
        text=[f"{v:.1f}%" for v in completion_data['completion_customers'].values],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='<b>%{x}</b><br>Lượt khách: %{y:.1f}%<extra></extra>'
    ))
    
    # Cột Doanh thu (màu xanh dương)
    fig.add_trace(go.Bar(
        x=routes,
        y=completion_data['completion_revenue'].values,
        name='Doanh thu',
        marker_color='#636EFA',  # Màu xanh dương
        text=[f"{v:.1f}%" for v in completion_data['completion_revenue'].values],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='<b>%{x}</b><br>Doanh thu: %{y:.1f}%<extra></extra>'
    ))
    
    # Cột Lãi Gộp (màu xanh lá)
    fig.add_trace(go.Bar(
        x=routes,
        y=completion_data['completion_profit'].values,
        name='Lãi Gộp',
        marker_color='#00CC96',  # Màu xanh lá
        text=[f"{v:.1f}%" for v in completion_data['completion_profit'].values],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='<b>%{x}</b><br>Lãi Gộp: %{y:.1f}%<extra></extra>'
    ))
    
    # Đường Mức mục tiêu (100%) - màu đỏ, nét đứt (vẫn dùng line để hiển thị rõ)
    fig.add_trace(go.Scatter(
        x=routes,
        y=[100] * len(routes),
        name='Mức mục tiêu',
        mode='lines',
        line=dict(color='#EF553B', width=2, dash='dash'),  # Màu đỏ, nét đứt
        hovertemplate='<b>%{x}</b><br>Mức mục tiêu: 100%<extra></extra>'
    ))
    
    # Tính tổng doanh thu đã đạt
    # Loại bỏ các dòng "Grand Total", "Dom Total", "Out Total" trước khi tính tổng
    filtered_data = completion_data[
        ~completion_data['route'].astype(str).str.contains('Grand Total|Dom Total|Out Total', case=False, na=False)
    ].copy()
    
    if filtered_data.empty:
        total_completion = 0
    else:
        total_revenue_actual = filtered_data['revenue'].sum()
        total_revenue_plan = filtered_data['plan_revenue'].sum()
        total_completion = (total_revenue_actual / total_revenue_plan * 100) if total_revenue_plan > 0 else 0
    
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Phần trăm hoàn thành (%)",
        height=500,
        barmode='group',  # Grouped bar chart
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(tickangle=-45, categoryorder='array', categoryarray=routes),
        yaxis=dict(range=[0, 200], dtick=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                x=1,
                y=1,
                xref="paper",
                yref="paper",
                text=f"TỔNG DOANH THU ĐÃ ĐẠT {total_completion:.0f}% MỤC TIÊU ĐƯA RA",
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=12, color="gray", weight="bold")
            )
        ],
        margin=dict(l=50, r=50, t=80, b=100)
    )
    
    return fig


def load_etour_seats_data(sheet_url):
    """
    Đọc dữ liệu theo dõi chỗ bán từ Google Sheet etour
    Sheet URL: https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=2069863260#gid=2069863260
    
    Cấu trúc CSV:
    - Header ở dòng 5 (index 4): Tuyến tour, SL Dự kiến, SL đã bán, SL còn lại, Tỷ lệ, Doanh số đã bán, Doanh số dự kiến, Tỷ lệ, ĐẦU KHỞI HÀNH, KHU VỰC KINH DOANH, TUYẾN TOUR MỚI, DOM/OUT, GIAI ĐOẠN
    - Dữ liệu bắt đầu từ dòng 6
    
    Returns: DataFrame với columns: route, route_type, plan_seats, actual_seats, remaining_seats, plan_revenue, actual_revenue, route_group
    """
    import requests
    import io
    import re
    
    try:
        # Chuyển đổi URL thành CSV export URL
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return pd.DataFrame()
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else '2069863260'  # Default gid từ URL
        
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
        
        # Tìm các cột cần thiết
        route_col = None
        plan_seats_col = None
        actual_seats_col = None
        remaining_seats_col = None
        actual_revenue_col = None
        plan_revenue_col = None
        dom_out_col = None
        route_group_col = None
        region_col = None  # Cột "KHU VỰC KINH DOANH"
        period_col = None  # Cột "GIAI ĐOẠN"
        
        for col in df.columns:
            col_upper = str(col).upper()
            # Tìm cột "Tour" (cột 0) - tên tour cụ thể
            if ('TOUR' in col_upper and 'TUYẾN' not in col_upper and 'TUYEN' not in col_upper) or (len(df.columns) > 0 and col == df.columns[0]):
                if route_col is None:  # Chỉ lấy cột đầu tiên nếu có "Tour"
                    route_col = col
            # Tìm cột "Tuyến tour" (cột 10) - dùng để group
            elif 'TUYẾN TOUR' in col_upper or 'TUYEN TOUR' in col_upper:
                # Nếu là cột 10, đây là route_group
                if len(df.columns) > 10 and col == df.columns[10]:
                    route_group_col = col
                else:
                    # Nếu không phải cột 10, có thể là route_col (nếu chưa có)
                    if route_col is None:
                        route_col = col
            elif 'SL DỰ KIẾN' in col_upper or 'SL DU KIEN' in col_upper:
                plan_seats_col = col
            elif 'SL ĐÃ BÁN' in col_upper or 'SL DA BAN' in col_upper:
                actual_seats_col = col
            elif 'SL CÒN LẠI' in col_upper or 'SL CON LAI' in col_upper:
                remaining_seats_col = col
            elif 'DOANH SỐ ĐÃ BÁN' in col_upper or 'DOANH SO DA BAN' in col_upper:
                actual_revenue_col = col
            elif 'DOANH SỐ DỰ KIẾN' in col_upper or 'DOANH SO DU KIEN' in col_upper:
                plan_revenue_col = col
            elif 'DOM/OUT' in col_upper or 'DOMOUT' in col_upper:
                dom_out_col = col
            elif 'TUYẾN TOUR MỚI' in col_upper or 'TUYEN TOUR MOI' in col_upper:
                route_group_col = col
            elif 'KHU VỰC KINH DOANH' in col_upper or 'KHU VUC KINH DOANH' in col_upper:
                region_col = col
            elif 'GIAI ĐOẠN' in col_upper or 'GIAI DOAN' in col_upper:
                period_col = col
        
        # Nếu không tìm thấy bằng tên, dùng vị trí cột (theo CSV mẫu)
        # Cấu trúc CSV: Tour(0), SL Dự kiến(1), SL đã bán(2), SL còn lại(3), Tỷ lệ(4), 
        # Doanh số đã bán(5), Doanh số dự kiến(6), Tỷ lệ(7), ĐẦU KHỞI HÀNH(8), 
        # KHU VỰC KINH DOANH(9), Tuyến tour(10), DOM/OUT(11), GIAI ĐOẠN(12)
        if route_col is None and len(df.columns) > 0:
            route_col = df.columns[0]  # Cột 0: Tour (tên tour cụ thể)
        if plan_seats_col is None and len(df.columns) > 1:
            plan_seats_col = df.columns[1]  # Cột 1: SL Dự kiến
        if actual_seats_col is None and len(df.columns) > 2:
            actual_seats_col = df.columns[2]  # Cột 2: SL đã bán
        if remaining_seats_col is None and len(df.columns) > 3:
            remaining_seats_col = df.columns[3]  # Cột 3: SL còn lại
        if actual_revenue_col is None and len(df.columns) > 5:
            actual_revenue_col = df.columns[5]  # Cột 5: Doanh số đã bán
        if plan_revenue_col is None and len(df.columns) > 6:
            plan_revenue_col = df.columns[6]  # Cột 6: Doanh số dự kiến
        if dom_out_col is None and len(df.columns) > 11:
            dom_out_col = df.columns[11]  # Cột 11: DOM/OUT
        if route_group_col is None and len(df.columns) > 10:
            route_group_col = df.columns[10]  # Cột 10: Tuyến tour (dùng để group)
        if region_col is None and len(df.columns) > 9:
            region_col = df.columns[9]  # Cột 9: KHU VỰC KINH DOANH
        if period_col is None and len(df.columns) > 12:
            period_col = df.columns[12]  # Cột 12: GIAI ĐOẠN
        
        if route_col is None:
            return pd.DataFrame()
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame()
        result_df['route'] = df[route_col].astype(str).str.strip()
        
        # Thêm cột region_unit (Khu vực Đơn Vị)
        if region_col:
            result_df['region_unit'] = df[region_col].astype(str).str.strip()
        else:
            result_df['region_unit'] = ''
        
        # Thêm cột period (Giai đoạn)
        if period_col:
            result_df['period'] = df[period_col].astype(str).str.strip()
        else:
            result_df['period'] = ''
        
        # Parse số liệu
        def parse_value(val):
            if pd.isna(val) or val == '' or str(val).strip() == '-' or str(val).strip().upper() == 'NAN':
                    return 0
            val_str = str(val).strip().replace(',', '').replace('"', '')
            # Xử lý số có dấu chấm làm dấu phân cách hàng nghìn (ví dụ: "6.995" = 6995, "333.460.000" = 333460000)
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

        if plan_seats_col:
            result_df['plan_seats'] = df[plan_seats_col].apply(parse_value)
        else:
            result_df['plan_seats'] = 0
        
        if actual_seats_col:
            result_df['actual_seats'] = df[actual_seats_col].apply(parse_value)
        else:
            result_df['actual_seats'] = 0
        
        if remaining_seats_col:
            result_df['remaining_seats'] = df[remaining_seats_col].apply(parse_value)
        else:
            result_df['remaining_seats'] = 0
        
        # LƯU Ý: Dữ liệu từ DEFAULT_ETOUR_SEATS_URL có đơn vị Doanh số là VNĐ (không cần chuyển đổi)
        if actual_revenue_col:
            result_df['actual_revenue'] = df[actual_revenue_col].apply(parse_value)  # Đơn vị: VNĐ
        else:
            result_df['actual_revenue'] = 0
        
        if plan_revenue_col:
            result_df['plan_revenue'] = df[plan_revenue_col].apply(parse_value)  # Đơn vị: VNĐ
        else:
            result_df['plan_revenue'] = 0
        
        if route_group_col:
            result_df['route_group'] = df[route_group_col].astype(str).str.strip()
        else:
            result_df['route_group'] = ''
        
        # Xác định route_type từ cột DOM/OUT
        if dom_out_col:
            result_df['route_type'] = df[dom_out_col].astype(str).str.strip().str.upper()
            # Chuẩn hóa: DOM -> Nội địa, OUT -> Outbound
            result_df['route_type'] = result_df['route_type'].apply(
                lambda x: 'Nội địa' if 'DOM' in str(x).upper() else ('Outbound' if 'OUT' in str(x).upper() else 'Nội địa')
            )
        else:
            result_df['route_type'] = 'Nội địa'
        
        # Loại bỏ dòng trống, Total, Grand total, nan
        result_df = result_df[
            (result_df['route'].notna()) &
            (result_df['route'].astype(str).str.strip() != '') &
            (~result_df['route'].astype(str).str.upper().str.contains('TOTAL', na=False)) &
            (~result_df['route'].astype(str).str.upper().str.contains('GRAND TOTAL', na=False)) &
            (~result_df['route'].astype(str).str.upper().str.contains('NAN', na=False))
        ].copy()
        
        # Nhóm theo route, route_group, route_type, region_unit và period để tổng hợp
        # QUAN TRỌNG: Phải groupby theo cả region_unit và period để tránh sum các dòng từ các region/period khác
        # Nhưng giữ lại route, region_unit và period để hiển thị chi tiết
        groupby_cols = ['route', 'route_type', 'region_unit', 'period']
        if 'route_group' in result_df.columns and not result_df['route_group'].isna().all():
            groupby_cols.insert(1, 'route_group')  # Chèn route_group vào vị trí thứ 2
        
        result_df = result_df.groupby(groupby_cols).agg({
            'plan_seats': 'sum',
            'actual_seats': 'sum',
            'remaining_seats': 'sum',
            'plan_revenue': 'sum',
            'actual_revenue': 'sum'
        }).reset_index()
        
        return result_df
        
    except Exception as e:
        return pd.DataFrame()


def create_seats_tracking_chart(data, title=''):
    """
    Tạo biểu đồ kết hợp bar và line chart để theo dõi số chỗ bán
    - Stacked bar: "LK Đã thực hiện" (actual_seats) và "LK kế hoạch còn" (remaining_seats) - trục Y trái (LK)
    - Line chart: "DT Kế hoạch" (plan_revenue) và "DS đã thực hiện" (actual_revenue) - trục Y phải (Tr đồng)
    
    data: DataFrame với columns: route, route_group, plan_seats, actual_seats, remaining_seats, plan_revenue, actual_revenue
    """
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="Không có dữ liệu", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=500)
        return fig

    # Nhóm theo route_group để tổng hợp (nếu có nhiều route trong cùng một route_group)
    # QUAN TRỌNG: plan_revenue và plan_seats dùng 'first' vì mỗi route_group chỉ có 1 giá trị kế hoạch duy nhất
    # actual_revenue và actual_seats dùng 'sum' để sum các dòng từ cùng route_group
    # remaining_seats KHÔNG sum, mà tính lại sau khi groupby: remaining_seats = plan_seats - actual_seats
    if 'route_group' in data.columns and not data['route_group'].isna().all():
        # Sử dụng route_group làm x-axis nếu có
        grouped_data = data.groupby('route_group').agg({
            'plan_seats': 'first',  # Mỗi route_group chỉ có 1 giá trị kế hoạch
            'actual_seats': 'sum',  # Sum các dòng từ cùng route_group
            'plan_revenue': 'first',  # Mỗi route_group chỉ có 1 giá trị kế hoạch
            'actual_revenue': 'sum'  # Sum các dòng từ cùng route_group
        }).reset_index()
    else:
        # Sử dụng route làm x-axis
        grouped_data = data.groupby('route').agg({
            'plan_seats': 'first',  # Mỗi route chỉ có 1 giá trị kế hoạch
            'actual_seats': 'sum',  # Sum các dòng từ cùng route
            'plan_revenue': 'first',  # Mỗi route chỉ có 1 giá trị kế hoạch
            'actual_revenue': 'sum'  # Sum các dòng từ cùng route
        }).reset_index()
        grouped_data['route_group'] = grouped_data['route']
    
    # QUAN TRỌNG: Tính lại remaining_seats sau khi groupby theo công thức: số kế hoạch - đã thực hiện
    # Đảm bảo công thức nhất quán với bảng chi tiết
    grouped_data['remaining_seats'] = (grouped_data['plan_seats'] - grouped_data['actual_seats']).clip(lower=0)
    
    # Tính phần trăm đạt kế hoạch
    grouped_data['completion_seats_pct'] = (grouped_data['actual_seats'] / grouped_data['plan_seats'].replace(0, np.nan) * 100).fillna(0)
    grouped_data['completion_revenue_pct'] = (grouped_data['actual_revenue'] / grouped_data['plan_revenue'].replace(0, np.nan) * 100).fillna(0)
    
    # Sắp xếp theo plan_seats (Lượt khách kế hoạch) giảm dần (giá trị lớn nhất ở bên trái)
    grouped_data = grouped_data.sort_values('plan_seats', ascending=False).head(15).copy()
    grouped_data = grouped_data.reset_index(drop=True)
    
    x_axis = grouped_data['route_group'].tolist()
    
    # Tạo figure với secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Stacked bar chart: Đã thực hiện (actual_seats) - màu xanh dương
    fig.add_trace(
        go.Bar(
            x=x_axis,
            y=grouped_data['actual_seats'].values,
            name='LK Đã thực hiện',
            marker_color='#1f77b4',  # Xanh dương
            hovertemplate='<b>%{x}</b><br>LK Đã thực hiện: %{y:,.0f} LK<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Stacked bar chart: Số chỗ có thể khai thác thêm (remaining_seats) - màu cam
    fig.add_trace(
        go.Bar(
            x=x_axis,
            y=grouped_data['remaining_seats'].values,
            name='LK kế hoạch còn lại',
            marker_color='#ff7f0e',  # Cam
            hovertemplate='<b>%{x}</b><br>LK kế hoạch còn lại: %{y:,.0f} LK<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Line chart: DT Kế hoạch (plan_revenue) - màu xám
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=(grouped_data['plan_revenue'].values / 1_000_000),  # Chuyển từ VND sang triệu đồng
            name='DT Kế hoạch',
            mode='lines+markers',
            line=dict(color='#808080', width=2),  # Xám
            marker=dict(size=8, color='#808080'),
            hovertemplate='<b>%{x}</b><br>DT Kế hoạch: %{y:,.0f} Tr đồng<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Line chart: DT đã thực hiện (actual_revenue) - màu vàng
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=(grouped_data['actual_revenue'].values / 1_000_000),  # Chuyển từ VND sang triệu đồng
            name='DS đã thực hiện',
            mode='lines+markers',
            line=dict(color='#FFD700', width=2),  # Vàng
            marker=dict(size=8, color='#FFD700'),
            hovertemplate='<b>%{x}</b><br>DS đã thực hiện: %{y:,.0f} Tr đồng<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Thêm annotation hiển thị % đạt kế hoạch LK trên đỉnh của stacked bar (bên trái)
    for i, route in enumerate(x_axis):
        total_seats = grouped_data.iloc[i]['plan_seats']
        completion_pct = grouped_data.iloc[i]['completion_seats_pct']
        if total_seats > 0 and not pd.isna(completion_pct):
            fig.add_annotation(
                x=route,
                y=total_seats,
                text=f"LK: <b>{completion_pct:.1f}%</b>",
                showarrow=False,
                font=dict(size=10, color='#dc3545', family='Arial Black'),  # Màu đỏ cho LK
                yshift=15,
                xshift=-25,  # Đẩy sang trái để tách khỏi DT
                yref='y'
            )
    
    # Thêm annotation hiển thị % đạt kế hoạch DT trên các marker của line chart (bên phải)
    for i, route in enumerate(x_axis):
        actual_revenue_tr = grouped_data.iloc[i]['actual_revenue'] / 1_000_000
        completion_pct = grouped_data.iloc[i]['completion_revenue_pct']
        if actual_revenue_tr > 0 and not pd.isna(completion_pct):
            fig.add_annotation(
                x=route,
                y=actual_revenue_tr,
                text=f"DT: {completion_pct:.1f}%",
                showarrow=False,
                font=dict(size=10, color='#FFD700', family='Arial Black'),  # Màu vàng cho DT
                yshift=20,
                xshift=25,  # Đẩy sang phải để tách khỏi LK
                yref='y2'
            )
    
    # Cập nhật layout
    fig.update_layout(
        title=title,
        xaxis_title="",
        height=500,
        barmode='stack',  # Stacked bar
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=100),
        xaxis=dict(tickangle=-45)
    )
    
    # Cập nhật trục Y trái (LK)
    fig.update_yaxes(
        title_text="LK",
        secondary_y=False
    )
    
    # Cập nhật trục Y phải (Tr đồng)
    fig.update_yaxes(
        title_text="Tr đồng",
        secondary_y=True
    )
    
    return fig


def load_completion_progress_actual_data(sheet_url):
    """
    Đọc dữ liệu thực tế cho phần Tiến độ hoàn thành kế hoạch từ Google Sheet.
    URL: https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=903527778#gid=903527778
    
    Cấu trúc:
    - Cột A: Khu vực Đơn Vị (Mien Bac LK, Mien Tay LK, Mien Trung LK, TPHCM & DNB LK, Total LK)
    - Cột D: Nhóm tuyến (Dom Total, Out Total, Grand Total) - KHÔNG lấy từ cột E (Tuyến tour)
    - Cột F: GIÁ TRỊ LK (Lượt khách đã thực hiện)
    - Cột G: GIÁ TRỊ DT (DT đã thực hiện)
    - Cột H: GIÁ TRỊ LG (LG đã thực hiện)
    - Cột I: Giai đoạn
    
    Returns: DataFrame với columns: region_unit, nhom_tuyen, num_customers, revenue, gross_profit, period
    Chỉ lấy các dòng có nhom_tuyen là "Dom Total", "Out Total", "Grand Total"
    """
    import requests
    import io
    import re
    
    try:
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return pd.DataFrame()
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else '903527778'

        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.content.decode('utf-8', errors='replace')), skipinitialspace=True)

        if df.empty:
            return pd.DataFrame()

        # Chuẩn hóa tên cột
        df.columns = [col.strip() for col in df.columns]

        # Tìm các cột cần thiết theo mô tả của người dùng
        region_unit_col = None # Cột A
        nhom_tuyen_col = None  # Cột D
        customers_col = None   # Cột F
        revenue_col = None     # Cột G
        profit_col = None      # Cột H
        period_col = None      # Cột I

        for col in df.columns:
            col_upper = str(col).upper()
            if 'KHU VỰC ĐƠN VỊ' in col_upper or 'KHU VUC DON VI' in col_upper:
                region_unit_col = col
            elif 'NHÓM TUYẾN' in col_upper or 'NHOM TUYEN' in col_upper:
                nhom_tuyen_col = col
            elif 'GIÁ TRỊ LK' in col_upper or 'GIA TRI LK' in col_upper:
                customers_col = col
            elif 'GIÁ TRỊ DT' in col_upper or 'GIA TRI DT' in col_upper:
                revenue_col = col
            elif 'GIÁ TRỊ LG' in col_upper or 'GIA TRI LG' in col_upper:
                profit_col = col
            elif 'GIAI ĐOẠN' in col_upper or 'GIAI DOAN' in col_upper:
                period_col = col

        # Fallback nếu không tìm thấy bằng tên - dùng vị trí cột
        if region_unit_col is None and len(df.columns) > 0: 
            region_unit_col = df.columns[0]  # Cột A
        if nhom_tuyen_col is None and len(df.columns) > 3: 
            nhom_tuyen_col = df.columns[3]  # Cột D
        if customers_col is None and len(df.columns) > 5: 
            customers_col = df.columns[5]  # Cột F
        if revenue_col is None and len(df.columns) > 6: 
            revenue_col = df.columns[6]  # Cột G
        if profit_col is None and len(df.columns) > 7: 
            profit_col = df.columns[7]  # Cột H
        if period_col is None and len(df.columns) > 8: 
            period_col = df.columns[8]  # Cột I

        if nhom_tuyen_col is None:
            return pd.DataFrame()

        # Parse số liệu
        def parse_numeric(val):
            if pd.isna(val) or val == '' or str(val).strip() == '-' or str(val).strip().upper() == 'NAN':
                return 0
            val_str = str(val).strip().replace(',', '').replace('"', '')
            # Xử lý số có dấu chấm làm dấu phân cách hàng nghìn
            if '.' in val_str and ',' not in val_str:
                if val_str.count('.') > 1:
                    val_str = val_str.replace('.', '')
                elif val_str.count('.') == 1:
                    parts = val_str.split('.')
                    if len(parts) == 2 and len(parts[1]) > 2:
                        val_str = val_str.replace('.', '')
            try:
                return float(val_str)
            except:
                return 0

        result_df = pd.DataFrame()
        result_df['region_unit'] = df[region_unit_col].astype(str).str.strip() if region_unit_col else ''
        result_df['nhom_tuyen'] = df[nhom_tuyen_col].astype(str).str.strip()
        # LK (Lượt khách) đã là số thực tế, KHÔNG nhân với 1,000,000
        result_df['num_customers'] = df[customers_col].apply(lambda x: parse_numeric(x) if customers_col else 0) if customers_col else 0
        # DT và LG đơn vị là triệu đồng (tr.đ), cần nhân với 1,000,000 để chuyển sang VND
        result_df['revenue'] = df[revenue_col].apply(lambda x: parse_numeric(x) * 1_000_000 if revenue_col else 0) if revenue_col else 0  # Convert from million VND to VND
        result_df['gross_profit'] = df[profit_col].apply(lambda x: parse_numeric(x) * 1_000_000 if profit_col else 0) if profit_col else 0  # Convert from million VND to VND
        result_df['period'] = df[period_col].astype(str).str.strip() if period_col else ''

        # Chỉ giữ lại các dòng có Nhóm tuyến là "Dom Total", "Out Total", "Grand Total"
        result_df = result_df[
            result_df['nhom_tuyen'].astype(str).str.contains('Dom Total|Out Total|Grand Total', case=False, na=False)
        ].copy()

        return result_df

    except Exception as e:
        return pd.DataFrame()


def load_completion_progress_plan_data(sheet_url, period_name='TẾT', region_filter=None):
    """
    Đọc dữ liệu kế hoạch cho phần Tiến độ hoàn thành kế hoạch từ Google Sheet.
    URL Tết: https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=1651160424#gid=1651160424
    URL Xuân: https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=212301737#gid=212301737
    
    Cấu trúc:
    - Cột A: Nhóm tuyến (Dom Total, Out Total, Grand Total) - KHÔNG lấy từ cột B (Tuyến tour)
    - Cột C, D, E: LK, DT, LG Kế hoạch công ty (nếu region_filter=None hoặc 'Tất cả')
    - Cột F, G, H: LK, DT, LG Kế hoạch Miền Bắc (nếu region_filter='Mien Bac')
    - Cột I, J, K: LK, DT, LG Kế hoạch Miền Trung (nếu region_filter='Mien Trung')
    - ...
    
    Args:
        sheet_url: URL của Google Sheet
        period_name: Tên giai đoạn ('TẾT' hoặc 'KM XUÂN')
        region_filter: Tên khu vực để filter ('Tất cả', 'Mien Bac', 'Mien Trung', 'Mien Nam', hoặc None)
                      Nếu None hoặc 'Tất cả', sẽ lấy tổng Công ty (cột C, D, E)
                      Nếu có region_filter cụ thể, sẽ lấy cột tương ứng với khu vực đó
    
    Returns: DataFrame với columns: nhom_tuyen, plan_customers, plan_revenue, plan_profit, period
    Chỉ lấy các dòng có nhom_tuyen là "Dom Total", "Out Total", "Grand Total"
    """
    import requests
    import io
    import re
    
    try:
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return pd.DataFrame()
        
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else None
        
        if not gid:
            return pd.DataFrame()
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        # Tải CSV
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        
        # Đọc CSV
        text = resp.content.decode('utf-8', errors='replace')
        lines = [line.rstrip('\r') for line in text.split('\n')]  # Loại bỏ \r để tránh lỗi
        
        # Tìm dòng header (dòng 5, index 4) - chứa "Nhom tuyen" và "Tuyến Tour"
        header_idx = None
        for i, line in enumerate(lines[:10]):
            line_upper = line.upper()
            if 'NHOM TUYEN' in line_upper or 'NHOM TUYẾN' in line_upper:
                if 'TUYẾN TOUR' in line_upper or 'TUYEN TOUR' in line_upper:
                    header_idx = i
                    break
        
        if header_idx is None:
            header_idx = 4 if len(lines) > 4 else 0
        
        # Đọc dòng 4 (index 3) TRƯỚC khi đọc DataFrame để có region_headers
        region_row_idx = 3  # Dòng 4 (index 3)
        region_headers = []
        if len(lines) > region_row_idx:
            try:
                # Đọc dòng 4 bằng pandas để parse CSV chính xác
                region_df = pd.read_csv(io.StringIO(lines[region_row_idx]), header=None, nrows=1)
                if not region_df.empty:
                    region_headers = [str(col).strip() for col in region_df.iloc[0].values]
            except Exception as e:
                # Fallback: dùng split đơn giản với CSV parser
                import csv
                try:
                    reader = csv.reader([lines[region_row_idx]])
                    region_headers = [col.strip().strip('"').strip("'") for col in next(reader)]
                except:
                    region_line = lines[region_row_idx]
                    region_headers = [col.strip().strip('"').strip("'") for col in region_line.split(',')]
        
        # Đọc từ dòng header
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])), skipinitialspace=True)
        
        if df.empty:
            return pd.DataFrame()
        
        # Tìm cột Nhóm tuyến (cột A)
        nhom_tuyen_col = None
        for col in df.columns:
            col_upper = str(col).upper()
            if 'NHOM TUYEN' in col_upper or 'NHOM TUYẾN' in col_upper:
                nhom_tuyen_col = col
                break
        
        # Fallback: dùng vị trí cột (cột A = index 0)
        if nhom_tuyen_col is None and len(df.columns) > 0:
            nhom_tuyen_col = df.columns[0]
        
        if nhom_tuyen_col is None:
            return pd.DataFrame()
        
        # Xác định cột cần lấy dựa trên region_filter
        customers_col = None
        revenue_col = None
        profit_col = None
        
        # Chuẩn hóa region_filter
        if region_filter and region_filter != 'Tất cả':
            region_filter_upper = str(region_filter).upper().strip()
            # Map tên khu vực
            region_mapping = {
                'MIEN BAC': 'MIEN BAC',
                'MIỀN BẮC': 'MIEN BAC',
                'MIEN TRUNG': 'MIEN TRUNG',
                'MIỀN TRUNG': 'MIEN TRUNG',
                'MIEN NAM': 'MIEN NAM',
                'MIỀN NAM': 'MIEN NAM',
                'TPHCM & DNB': 'TPHCM & DNB',
                'TPHCM DNB': 'TPHCM & DNB',
                'TPHCM VÀ DNB': 'TPHCM & DNB',
                'MIEN TAY': 'MIEN TAY',
                'MIỀN TÂY': 'MIEN TAY',
                'MIENTAY': 'MIEN TAY'
            }
            target_region = region_mapping.get(region_filter_upper, region_filter_upper)
        else:
            target_region = None  # Dùng tổng Công ty
        
        # Tìm cột LK, DT, LG dựa trên region_filter
        if target_region is None:
            # Dùng tổng Công ty: Cột C, D, E (index 2, 3, 4)
            if len(df.columns) > 2:
                customers_col = df.columns[2]  # Cột C: LK
            if len(df.columns) > 3:
                revenue_col = df.columns[3]  # Cột D: DT (tr.d)
            if len(df.columns) > 4:
                profit_col = df.columns[4]  # Cột E: LG (tr.d)
        else:
            # Tìm cột của khu vực cụ thể
            # Tìm vị trí của khu vực trong region_headers (dòng 4) để tính offset
            region_idx_in_headers = None
            if region_headers:
                # Tìm index đầu tiên của khu vực trong region_headers
                for i, header in enumerate(region_headers):
                    header_upper = str(header).upper().strip()
                    # Bỏ qua các cột trống hoặc không phải tên khu vực
                    if not header_upper or header_upper in ['', 'NAN', 'NONE']:
                        continue
                    
                    # Kiểm tra match với target_region
                    if target_region == 'MIEN BAC':
                        if ('MIEN BAC' in header_upper or 'MIỀN BẮC' in header_upper or 
                            'MIENBAC' in header_upper or 
                            (header_upper.startswith('MIEN') and 'BAC' in header_upper and 'TRUNG' not in header_upper) or
                            (header_upper.startswith('MIỀN') and 'BẮC' in header_upper and 'TRUNG' not in header_upper)):
                            region_idx_in_headers = i
                            break
                    elif target_region == 'MIEN TRUNG':
                        if ('MIEN TRUNG' in header_upper or 'MIỀN TRUNG' in header_upper or 
                            'MIENTRUNG' in header_upper or
                            (header_upper.startswith('MIEN') and 'TRUNG' in header_upper) or
                            (header_upper.startswith('MIỀN') and 'TRUNG' in header_upper)):
                            region_idx_in_headers = i
                            break
                    elif target_region == 'MIEN NAM':
                        if ('MIEN NAM' in header_upper or 'MIỀN NAM' in header_upper or 
                            'MIENNAM' in header_upper or
                            (header_upper.startswith('MIEN') and 'NAM' in header_upper and 'BAC' not in header_upper and 'TRUNG' not in header_upper) or
                            (header_upper.startswith('MIỀN') and 'NAM' in header_upper and 'BẮC' not in header_upper and 'TRUNG' not in header_upper)):
                            region_idx_in_headers = i
                            break
                    elif target_region == 'TPHCM & DNB':
                        if ('TPHCM' in header_upper and 'DNB' in header_upper) or \
                           ('TPHCM' in header_upper and ('&' in header_upper or 'VÀ' in header_upper or 'VA' in header_upper)) or \
                           ('TPHCM' in header_upper and 'DNB' in header_upper) or \
                           ('HO CHI MINH' in header_upper and 'DNB' in header_upper):
                            region_idx_in_headers = i
                            break
                    elif target_region == 'MIEN TAY':
                        if ('MIEN TAY' in header_upper or 'MIỀN TÂY' in header_upper or 
                            'MIENTAY' in header_upper or
                            (header_upper.startswith('MIEN') and 'TAY' in header_upper and 'BAC' not in header_upper and 'TRUNG' not in header_upper and 'NAM' not in header_upper) or
                            (header_upper.startswith('MIỀN') and 'TÂY' in header_upper and 'BẮC' not in header_upper and 'TRUNG' not in header_upper and 'NAM' not in header_upper)):
                            region_idx_in_headers = i
                            break
            
            # Tính offset dựa trên vị trí trong region_headers
            if region_idx_in_headers is not None:
                col_offset = region_idx_in_headers
                
                # Kiểm tra xem có đủ cột không
                if len(df.columns) > col_offset:
                    customers_col = df.columns[col_offset]  # LK
                if len(df.columns) > col_offset + 1:
                    revenue_col = df.columns[col_offset + 1]  # DT
                if len(df.columns) > col_offset + 2:
                    profit_col = df.columns[col_offset + 2]  # LG
            
            # Fallback: Tìm bằng cách duyệt qua các cột và tìm cột LK, DT, LG
            # QUAN TRỌNG: Khi pandas đọc CSV có nhiều cột trùng tên, nó sẽ tự động thêm suffix
            # Ví dụ: "LK" (Công ty), "LK.1" (Miền Bắc), "LK.2" (Miền Trung)
            if customers_col is None or revenue_col is None or profit_col is None:
                # Tìm các cột LK, DT, LG theo thứ tự và vị trí
                lk_cols = []
                dt_cols = []
                lg_cols = []
                
                for idx, col in enumerate(df.columns):
                    if col == nhom_tuyen_col:
                        continue
                    col_str = str(col).strip()
                    col_upper = col_str.upper()
                    col_idx = idx
                    
                    # Tìm cột LK (có thể là "LK", "LK.1", "LK.2", ...)
                    if col_upper == 'LK' or col_upper.startswith('LK.') or 'LƯỢT KHÁCH' in col_upper:
                        lk_cols.append((col_idx, col))
                    # Tìm cột DT (có thể là "DT (tr.d)", "DT (tr.d).1", "DT (tr.d).2", ...)
                    elif 'DT (TR.D)' in col_upper or 'DT(TR.D)' in col_upper or (col_upper.startswith('DT') and ('TR.D' in col_upper or 'TRD' in col_upper)):
                        dt_cols.append((col_idx, col))
                    # Tìm cột LG (có thể là "LG (tr.d)", "LG (tr.d).1", "LG (tr.d).2", ...)
                    elif 'LG (TR.D)' in col_upper or 'LG(TR.D)' in col_upper or (col_upper.startswith('LG') and ('TR.D' in col_upper or 'TRD' in col_upper)):
                        lg_cols.append((col_idx, col))
                
                # Xác định cột nào thuộc về khu vực đã chọn dựa trên vị trí
                if target_region == 'MIEN BAC':
                    # Lấy cột LK, DT, LG thứ hai (sau Công ty)
                    # Ưu tiên: cột có suffix .1, nếu không có thì lấy cột có index trong khoảng 5-7
                    for col_idx, col_name in lk_cols:
                        col_name_str = str(col_name)
                        if '.1' in col_name_str:
                            customers_col = col_name
                            break
                        elif col_idx >= 5 and col_idx < 8 and customers_col is None:
                            customers_col = col_name
                    for col_idx, col_name in dt_cols:
                        col_name_str = str(col_name)
                        if '.1' in col_name_str:
                            revenue_col = col_name
                            break
                        elif col_idx >= 5 and col_idx < 8 and revenue_col is None:
                            revenue_col = col_name
                    for col_idx, col_name in lg_cols:
                        col_name_str = str(col_name)
                        if '.1' in col_name_str:
                            profit_col = col_name
                            break
                        elif col_idx >= 5 and col_idx < 8 and profit_col is None:
                            profit_col = col_name
                elif target_region == 'MIEN TRUNG':
                    # Lấy cột LK, DT, LG thứ ba
                    for col_idx, col_name in lk_cols:
                        col_name_str = str(col_name)
                        if '.2' in col_name_str:
                            customers_col = col_name
                            break
                        elif col_idx >= 8 and col_idx < 11 and customers_col is None:
                            customers_col = col_name
                    for col_idx, col_name in dt_cols:
                        col_name_str = str(col_name)
                        if '.2' in col_name_str:
                            revenue_col = col_name
                            break
                        elif col_idx >= 8 and col_idx < 11 and revenue_col is None:
                            revenue_col = col_name
                    for col_idx, col_name in lg_cols:
                        col_name_str = str(col_name)
                        if '.2' in col_name_str:
                            profit_col = col_name
                            break
                        elif col_idx >= 8 and col_idx < 11 and profit_col is None:
                            profit_col = col_name
                elif target_region == 'MIEN NAM':
                    # Lấy cột LK, DT, LG thứ tư
                    for col_idx, col_name in lk_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str:
                            customers_col = col_name
                            break
                        elif col_idx >= 11 and customers_col is None:
                            customers_col = col_name
                    for col_idx, col_name in dt_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str:
                            revenue_col = col_name
                            break
                        elif col_idx >= 11 and revenue_col is None:
                            revenue_col = col_name
                    for col_idx, col_name in lg_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str:
                            profit_col = col_name
                            break
                        elif col_idx >= 11 and profit_col is None:
                            profit_col = col_name
                elif target_region == 'TPHCM & DNB':
                    # Lấy cột LK, DT, LG thứ năm (index 11, 12, 13)
                    for col_idx, col_name in lk_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str and col_idx >= 11:
                            customers_col = col_name
                            break
                        elif col_idx >= 11 and col_idx < 14 and customers_col is None:
                            customers_col = col_name
                    for col_idx, col_name in dt_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str and col_idx >= 12:
                            revenue_col = col_name
                            break
                        elif col_idx >= 12 and col_idx < 14 and revenue_col is None:
                            revenue_col = col_name
                    for col_idx, col_name in lg_cols:
                        col_name_str = str(col_name)
                        if '.3' in col_name_str and col_idx >= 13:
                            profit_col = col_name
                            break
                        elif col_idx >= 13 and col_idx < 14 and profit_col is None:
                            profit_col = col_name
                elif target_region == 'MIEN TAY':
                    # Lấy cột LK, DT, LG thứ sáu (index 14, 15, 16)
                    for col_idx, col_name in lk_cols:
                        col_name_str = str(col_name)
                        if '.4' in col_name_str:
                            customers_col = col_name
                            break
                        elif col_idx >= 14 and col_idx < 17 and customers_col is None:
                            customers_col = col_name
                    for col_idx, col_name in dt_cols:
                        col_name_str = str(col_name)
                        if '.4' in col_name_str:
                            revenue_col = col_name
                            break
                        elif col_idx >= 15 and col_idx < 17 and revenue_col is None:
                            revenue_col = col_name
                    for col_idx, col_name in lg_cols:
                        col_name_str = str(col_name)
                        if '.4' in col_name_str:
                            profit_col = col_name
                            break
                        elif col_idx >= 16 and col_idx < 17 and profit_col is None:
                            profit_col = col_name
        
        # Parse số liệu
        def parse_value(val):
            if pd.isna(val) or val == '' or str(val).strip() == '-' or str(val).strip().upper() == 'NAN':
                return 0
            val_str = str(val).strip().replace(',', '').replace('"', '')
            # Xử lý số có dấu chấm làm dấu phân cách hàng nghìn
            if '.' in val_str and ',' not in val_str:
                if val_str.count('.') > 1:
                    val_str = val_str.replace('.', '')
                elif val_str.count('.') == 1:
                    parts = val_str.split('.')
                    if len(parts) == 2 and len(parts[1]) <= 2:
                        pass  # Có thể là số thập phân
                    else:
                        val_str = val_str.replace('.', '')
            try:
                return float(val_str)
            except:
                return 0
        
        result_df = pd.DataFrame()
        result_df['nhom_tuyen'] = df[nhom_tuyen_col].astype(str).str.strip()
        
        if customers_col:
            result_df['plan_customers'] = df[customers_col].apply(parse_value)
        else:
            result_df['plan_customers'] = 0
        
        # LƯU Ý: Dữ liệu từ Plan Tết và Plan Xuân có đơn vị DT và LG là triệu đồng (tr.đ)
        # Chuyển đổi sang VND để tính toán thống nhất với các nguồn dữ liệu khác
        if revenue_col:
            result_df['plan_revenue'] = df[revenue_col].apply(parse_value) * 1_000_000  # Chuyển từ triệu đồng sang VND
        else:
            result_df['plan_revenue'] = 0
        
        if profit_col:
            result_df['plan_profit'] = df[profit_col].apply(parse_value) * 1_000_000  # Chuyển từ triệu đồng sang VND
        else:
            result_df['plan_profit'] = 0
        
        result_df['period'] = period_name
        
        # Chỉ giữ lại các dòng có Nhóm tuyến là "Dom Total", "Out Total", "Grand Total"
        result_df = result_df[
            result_df['nhom_tuyen'].astype(str).str.contains('Dom Total|Out Total|Grand Total', case=False, na=False)
        ].copy()
        
        return result_df
        
    except Exception as e:
        return pd.DataFrame()