"""
Utility functions for Vietravel Business Intelligence Dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    """Filter only confirmed bookings (exclude cancelled/postponed)"""
    if 'status' not in df.columns:
        return pd.DataFrame(columns=df.columns)
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
    
    # Calculate actual metrics.
    # Prefer effective (post-cancellation) columns when present (revenue_effective, gross_profit_effective, num_customers_effective).
    # This ensures KPIs reflect cancellations recorded via `cancel_count` (col U) rather than raw booked values.
    if 'revenue_effective' in confirmed_data.columns:
        actual_revenue = confirmed_data['revenue_effective'].sum()
    else:
        actual_revenue = confirmed_data['revenue'].sum()

    if 'gross_profit_effective' in confirmed_data.columns:
        actual_gross_profit = confirmed_data['gross_profit_effective'].sum()
    else:
        actual_gross_profit = confirmed_data['gross_profit'].sum()

    if 'num_customers_effective' in confirmed_data.columns:
        actual_customers = confirmed_data['num_customers_effective'].sum()
    else:
        actual_customers = confirmed_data['num_customers'].sum()
    
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
    
    ly_revenue = last_year_confirmed['revenue'].sum()
    ly_gross_profit = last_year_confirmed['gross_profit'].sum()
    ly_customers = last_year_confirmed['num_customers'].sum()
    
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


def create_gauge_chart(value, title, max_value=150, threshold=100, unit_breakdown=None, is_inverse_metric=False, actual_value=None, planned_value=None):
    """Create a gauge chart for completion rate with hover info for business units"""
    
    value = value if not pd.isna(value) else 0

    # LÔ-GÍC MÀU ĐÃ ĐẢO NGƯỢC
    if is_inverse_metric:
        if value <= threshold:
            color = "#00CC96"  # Xanh lá: Tỷ lệ Tốt (Dưới ngưỡng)
            bgcolor = "rgba(0, 204, 150, 0.2)"
        elif value <= threshold * 1.5:
            color = "#FFA500"  # Cam: Cần chú ý
            bgcolor = "rgba(255, 165, 0, 0.2)"
        else:
            color = "#EF553B"  # Đỏ: Xấu (Vượt xa ngưỡng)
            bgcolor = "rgba(239, 85, 59, 0.2)"
    else:
        # Logic màu ban đầu (Doanh thu, Lượt khách)
        if value >= threshold:
            color = "#00CC96"
            bgcolor = "rgba(0, 204, 150, 0.2)"
        elif value >= threshold * 0.8:
            color = "#FFA500"
            bgcolor = "rgba(255, 165, 0, 0.2)"
        else:
            color = "#EF553B"
            bgcolor = "rgba(239, 85, 59, 0.2)"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [1, 1]},
        title = {'text': title, 'font': {'size': 15}},
        number = {
            'suffix': "%", 
            'font': {'size': 20}
        },
        gauge = {
            'axis': {'range': [None, max_value], 'ticksuffix': "%", 'tickfont': {'size': 12}},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold * 0.5], 'color': "#FFE5E5"},
                {'range': [threshold * 0.5, threshold * 0.8], 'color': "#FFF4E5"},
                {'range': [threshold * 0.8, threshold], 'color': "#E5F5E5"},
                {'range': [threshold, max_value], 'color': "#D4F1D4"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    # Add invisible scatter trace for hover info with business unit breakdown
    if unit_breakdown is not None and not unit_breakdown.empty:
        hover_text = "Chi tiết theo đơn vị:<br>"
        for _, row in unit_breakdown.iterrows():
            hover_text += f"<br>{row['business_unit']}: {row['completion']:.1f}%"
        
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[0.1],
            mode='markers',
            marker=dict(size=100, color='rgba(0,0,0,0)', opacity=0),
            hovertemplate=hover_text + "<extra></extra>",
            showlegend=False
        ))
    
    # Make the gauge larger so it reads well when displayed side-by-side
    # Reduce top margin so gauges sit closer to the page title
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10,t= 8, b=10),
        hovermode='closest'
    )
    # The invisible scatter used for hover adds default x/y axes; hide them so
    # the gauge appears clean (no -0.5..1.5 ticks etc.).
    try:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    except Exception:
        pass
    # If actual/planned numbers are provided, add a small, non-obstructive annotation
    # inside the figure (paper coordinates) so it appears "on the chart" but does
    # not cover the central percentage number.
    try:
        if actual_value is not None or planned_value is not None:
            # Format values: use currency formatting for revenue/profit gauges,
            # but use plain numbers for customer gauges (no currency symbol).
            try:
                title_lc = str(title).lower() if title else ''
                is_customer_metric = any(k in title_lc for k in ['lượt', 'khách', 'khach', 'customer'])

                use_currency = False
                if not is_customer_metric:
                    # For revenue/profit metrics prefer currency formatting when values are large
                    if (isinstance(actual_value, (int, float)) and abs(actual_value or 0) >= 1000) or \
                       (isinstance(planned_value, (int, float)) and abs(planned_value or 0) >= 1000):
                        use_currency = True

                if use_currency:
                    actual_str = format_currency(actual_value or 0)
                    planned_str = format_currency(planned_value or 0)
                else:
                    actual_str = format_number(actual_value or 0)
                    planned_str = format_number(planned_value or 0)
            except Exception:
                actual_str = str(actual_value or '')
                planned_str = str(planned_value or '')

            ann_text = f"TH: {actual_str}<br>KH: {planned_str}"
            # Keep a modest top margin and place the TH/KH annotation below the
            # chart title so it doesn't overlap. Use paper coords for stable placement.
            try:
                fig.update_layout(margin=dict(l=10, r=10, t=5, b=10))
            except Exception:
                pass

            fig.add_annotation(
                x=0.5,
                y=0.55,
                xref='paper',
                yref='paper',
                xanchor='center',
                yanchor='top',
                text=ann_text,
                showarrow=False,
                align='center',
                font=dict(size=12, color='#FFFFFF'),
                bordercolor='rgba(255,255,255,0.12)',
                borderwidth=1,
                bgcolor='rgba(0,0,0,0.5)',
                opacity=0.95
            )
    except Exception:
        # Don't let annotation errors break the chart
        pass

    return fig


def create_bar_chart(data, x, y, title, orientation='v', color=None):
    """Create a bar chart"""
    
    if orientation == 'h':
        fig = px.bar(data, x=y, y=x, orientation='h', title=title, color=color,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
    else:
        fig = px.bar(data, x=x, y=y, title=title, color=color,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
    
    fig.update_layout(
        height=400,
        showlegend=True if color else False,
        hovermode='x unified'
    )
    
    return fig


def create_pie_chart(data, values, names, title):
    """Create a pie chart"""
    
    fig = px.pie(data, values=values, names=names, title=title,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        height=400,
        showlegend=True
    )
    
    return fig


def group_small_categories(df, value_col, name_col, threshold=0.02, other_label='Khác'):
    """
    Group categories whose share is below `threshold` (fraction, e.g. 0.02 for 2%)
    into a single `other_label` row and return a new aggregated DataFrame.

    Args:
        df: DataFrame with at least [name_col, value_col]
        value_col: column name for numeric values
        name_col: column name for category labels
        threshold: fraction threshold under which categories are grouped (0-1)
        other_label: label to use for grouped small categories

    Returns:
        DataFrame with columns [name_col, value_col] where small categories
        are combined into a single row named `other_label`.
    """
    if df is None or df.empty:
        return df

    df2 = df[[name_col, value_col]].copy()
    # Ensure numeric
    df2[value_col] = pd.to_numeric(df2[value_col].fillna(0), errors='coerce').fillna(0)
    total = df2[value_col].sum()
    if total <= 0:
        return df2

    df2['pct'] = df2[value_col] / float(total)
    large = df2[df2['pct'] >= float(threshold)].copy()
    small = df2[df2['pct'] < float(threshold)].copy()

    other_sum = small[value_col].sum()

    # Prepare detail strings: for each large row include its value and pct;
    # for the 'other' row include a newline-separated list of small components.
    def fmt_row(row):
        return f"{row[name_col]}: {format_currency(row[value_col])} ({row['pct']*100:.2f}%)"

    parts = []
    if other_sum > 0:
        # build large rows
        for _, r in large.iterrows():
            parts.append({name_col: r[name_col], value_col: r[value_col], 'detail': fmt_row(r)})

        # build 'other' detail listing
        small_lines = [f"{r[name_col]}: {format_currency(r[value_col])} ({r['pct']*100:.2f}%)" for _, r in small.iterrows()]
        other_detail = "<br>".join(small_lines)
        parts.append({name_col: other_label, value_col: other_sum, 'detail': other_detail})
    else:
        for _, r in large.iterrows():
            parts.append({name_col: r[name_col], value_col: r[value_col], 'detail': fmt_row(r)})

    result = pd.DataFrame(parts)
    # Sort descending by value for consistent display
    result = result.sort_values(by=value_col, ascending=False).reset_index(drop=True)
    return result


def create_line_chart(data, x, y, title, color=None):
    """Create a line chart"""
    
    fig = px.line(data, x=x, y=y, title=title, color=color,
                  markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
    
    fig.update_xaxes(title="")
    fig.update_yaxes(title="")
    fig.update_layout(
        height=400,
        showlegend=True if color else False,
        hovermode='x unified'
    )
    
    return fig


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


def get_route_unit_breakdown(tours_df, route_name, metric='revenue'):
    """
    Get breakdown by business unit for a specific route and metric
    """
    confirmed = filter_confirmed_bookings(tours_df)
    route_data = confirmed[confirmed['route'] == route_name]
    
    if route_data.empty:
        return pd.DataFrame(columns=['business_unit', 'revenue', 'num_customers', 'gross_profit', 'percentage'])
    
    unit_breakdown = route_data.groupby('business_unit').agg({
        'revenue': 'sum',
        'num_customers': 'sum',
        'gross_profit': 'sum'
    }).reset_index()
    
    if metric == 'revenue':
        total_value = unit_breakdown['revenue'].sum()
        col_name = 'revenue'
    elif metric == 'customers':
        total_value = unit_breakdown['num_customers'].sum()
        col_name = 'num_customers'
    else: # profit
        total_value = unit_breakdown['gross_profit'].sum()
        col_name = 'gross_profit'
    
    # SỬA LỖI LOGIC: Phải dùng col_name để tính tỷ trọng, không phải luôn là 'revenue'
    unit_breakdown['percentage'] = np.where(
        total_value > 0,
        (unit_breakdown[col_name] / total_value * 100).round(1), # ĐÃ SỬA: Dùng col_name
        0
    )
    unit_breakdown = unit_breakdown.sort_values(col_name, ascending=False)
    
    return unit_breakdown


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
    cancelled_postponed = len(tours_df[tours_df['status'].isin(['Đã hủy', 'Hoãn'])])
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


def get_low_margin_tours(tours_df, threshold=5):
    """
    Get tours with profit margin below threshold
    """
    confirmed = filter_confirmed_bookings(tours_df)
    
    if confirmed.empty:
        return pd.DataFrame(columns=['route', 'gross_profit', 'revenue', 'num_customers', 'profit_margin'])

    # Group by route and calculate average margin
    route_margins = confirmed.groupby('route').agg({
        'gross_profit': 'sum',
        'revenue': 'sum',
        'num_customers': 'sum'
    }).reset_index()
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    route_margins['profit_margin'] = np.where(
        route_margins['revenue'] > 0,
        (route_margins['gross_profit'] / route_margins['revenue'] * 100),
        0
    )
    
    # Filter low margin routes
    low_margin = route_margins[route_margins['profit_margin'] < threshold].sort_values('profit_margin')
    
    return low_margin


def get_unit_performance(tours_df, plans_df, start_date, end_date):
    """
    Calculate performance by business unit
    """
    # Filter current period data
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    if confirmed_data.empty: 
        return pd.DataFrame(columns=['business_unit', 'actual_revenue', 'actual_profit', 'actual_customers', 'planned_revenue', 'planned_gross_profit', 'planned_customers', 'revenue_completion', 'customer_completion', 'profit_margin'])

    # Actual by unit
    actual_by_unit = confirmed_data.groupby('business_unit').agg({
        'revenue': 'sum',
        'gross_profit': 'sum',
        'num_customers': 'sum'
    }).reset_index()
    actual_by_unit.columns = ['business_unit', 'actual_revenue', 'actual_profit', 'actual_customers']
    
    # Plans by unit
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (plans_df['year'] == start_dt.year) & \
                (plans_df['month'] >= start_dt.month) & \
                (plans_df['month'] <= end_dt.month)
    period_plans = plans_df[plan_mask]
    
    plan_by_unit = period_plans.groupby('business_unit').agg({
        'planned_revenue': 'sum',
        'planned_gross_profit': 'sum',
        'planned_customers': 'sum'
    }).reset_index()
    
    # Merge and calculate completion
    performance = actual_by_unit.merge(plan_by_unit, on='business_unit', how='left').fillna(0)
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    performance['revenue_completion'] = np.where(
        performance['planned_revenue'] > 0,
        (performance['actual_revenue'] / performance['planned_revenue'] * 100),
        0
    )
    performance['customer_completion'] = np.where(
        performance['planned_customers'] > 0,
        (performance['actual_customers'] / performance['planned_customers'] * 100),
        0
    )
    performance['profit_margin'] = np.where(
        performance['actual_revenue'] > 0,
        (performance['actual_profit'] / performance['actual_revenue'] * 100),
        0
    )
    
    return performance


def get_unit_breakdown(tours_df, plans_df, start_date, end_date, metric='revenue'):
    """
    Get completion rate breakdown by business unit for a specific metric (Dùng cho Gauge Chart Hover)
    """
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (plans_df['year'] == start_dt.year) & \
                (plans_df['month'] >= start_dt.month) & \
                (plans_df['month'] <= end_dt.month)
    period_plans = plans_df[plan_mask]
    
    results = []
    for unit in sorted(confirmed_data['business_unit'].unique()):
        unit_data = confirmed_data[confirmed_data['business_unit'] == unit]
        unit_plans = period_plans[period_plans['business_unit'] == unit]
        
        if metric == 'revenue':
            actual = unit_data['revenue'].sum()
            planned = unit_plans['planned_revenue'].sum()
        elif metric == 'profit':
            actual = unit_data['gross_profit'].sum()
            planned = unit_plans['planned_gross_profit'].sum()
        else:  # customers
            actual = unit_data['num_customers'].sum()
            planned = unit_plans['planned_customers'].sum()
        
        completion = calculate_completion_rate(actual, planned)
        results.append({
            'business_unit': unit,
            'completion': completion
        })
    
    return pd.DataFrame(results)


def get_segment_breakdown(tours_df, start_date, end_date, metric='revenue'):
    """
    Get breakdown by segment (FIT/GIT/Inbound) for a specific metric
    """
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    if confirmed_data.empty:
        return pd.DataFrame(columns=['segment', 'value', 'percentage'])
    
    if metric == 'revenue':
        segment_data = confirmed_data.groupby('segment')['revenue'].sum().reset_index()
        segment_data.columns = ['segment', 'value']
    elif metric == 'customers':
        segment_data = confirmed_data.groupby('segment')['num_customers'].sum().reset_index()
        segment_data.columns = ['segment', 'value']
    else:  # profit
        segment_data = confirmed_data.groupby('segment')['gross_profit'].sum().reset_index()
        segment_data.columns = ['segment', 'value']
    
    total_value = segment_data['value'].sum()
    # ĐÃ SỬA: Bảo vệ chia cho 0
    segment_data['percentage'] = np.where(
        total_value > 0,
        (segment_data['value'] / total_value * 100).round(1),
        0
    )
    segment_data = segment_data.sort_values('value', ascending=False)
    
    return segment_data


def get_segment_unit_breakdown(tours_df, start_date, end_date, segment_name, metric='revenue'):
    """
    Get business unit breakdown for a specific segment (for hover tooltips)
    """
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    segment_data = confirmed_data[confirmed_data['segment'] == segment_name]
    
    if segment_data.empty:
        return pd.DataFrame(columns=['business_unit', 'value', 'percentage'])
    
    if metric == 'revenue':
        unit_breakdown = segment_data.groupby('business_unit')['revenue'].sum().reset_index()
        unit_breakdown.columns = ['business_unit', 'value']
    elif metric == 'customers':
        unit_breakdown = segment_data.groupby('business_unit')['num_customers'].sum().reset_index()
        unit_breakdown.columns = ['business_unit', 'value']
    else:  # profit
        unit_breakdown = segment_data.groupby('business_unit')['gross_profit'].sum().reset_index()
        unit_breakdown.columns = ['business_unit', 'value']
    
    total_value = unit_breakdown['value'].sum()
    # ĐÃ SỬA: Bảo vệ chia cho 0
    unit_breakdown['percentage'] = np.where(
        total_value > 0,
        (unit_breakdown['value'] / total_value * 100).round(1),
        0
    )
    unit_breakdown = unit_breakdown.sort_values('value', ascending=False)
    
    return unit_breakdown


def create_forecast_chart(tours_df, plans_df, start_date, end_date, date_option, plans_daily_df=None, plans_weekly_df=None, selected_segment=None):
    """
    Create forecast chart combining cumulative actuals (bars) and planned/forecast lines (lines) for revenue.
    Requires date_option (Tuần/Tháng/Quý/Năm) to determine the period_end_dt.
    """
    
    # --- 1. Chuẩn bị Dữ liệu và Chuẩn hóa Ngày tháng ---
    confirmed_data = filter_confirmed_bookings(tours_df)
    
    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()
    today = pd.to_datetime(datetime.now().date())
    
    # LÔ-GÍC XÁC ĐỊNH NGÀY CUỐI CÙNG VÀ ĐỘ PHÂN GIẢI
    period_end_dt = today
    
    # Xác định độ phân giải (Granularity)
    if date_option == 'Năm' or date_option == 'Quý':
        freq_unit = 'M'
        x_format = "%m/%Y"
        x_title = "Tháng"
        
        # Mở rộng kỳ Dự báo
        if date_option == 'Năm':
            period_end_dt = pd.to_datetime(datetime(start_dt.year, 12, 31))
        elif date_option == 'Quý':
            # Dự báo đến ngày cuối cùng của quý (Quý IV bắt đầu 01/10)
            if start_dt.month in [1, 4, 7, 10]:
                end_month = start_dt.month + 2
                period_end_dt = pd.to_datetime(datetime(start_dt.year, end_month, 1)) + pd.offsets.MonthEnd(0)
            else:
                period_end_dt = today
        
    elif date_option == 'Tháng':
        freq_unit = 'W' 
        x_format = "T%W"
        x_title = "Tuần"
        
        # Dự báo đến ngày cuối cùng của tháng
        period_end_dt = start_dt + pd.offsets.MonthEnd(0)
        
    elif date_option == 'Tuần' or date_option == 'Tùy chỉnh':
        freq_unit = 'D'
        x_format = "%d/%m"
        x_title = "Ngày"
        period_end_dt = end_dt
        
    else: 
        freq_unit = 'D'
        x_format = "%d/%m"
        x_title = "Ngày"
        period_end_dt = end_dt


    # Lọc dữ liệu Thực hiện đến ngày hôm nay
    period_tours = filter_data_by_date(confirmed_data, start_dt, today, date_column='booking_date')

    if period_tours.empty:
        return go.Figure().update_layout(title=f'Không có dữ liệu Thực hiện từ {start_dt.strftime("%d/%m")}', height=300)
        
    # --- 2. Xử lý Dữ liệu Lũy kế Thực hiện ---
    
    # Tổng hợp Actuals theo đơn vị thời gian (freq_unit)
    period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period(freq_unit)
    daily_actual = period_tours.groupby('period')[['revenue']].sum().reset_index()
    
    # Chuyển Period sang Timestamp để vẽ cột
    if freq_unit == 'M':
        # Dùng ngày đầu tháng để vẽ cột
        daily_actual['date'] = daily_actual['period'].apply(lambda x: x.start_time.normalize())
        # Chiều rộng 20 ngày cho cột tháng
        bar_width = 20 * 24 * 60 * 60 * 1000 
    elif freq_unit == 'W':
        daily_actual['date'] = daily_actual['period'].apply(lambda x: x.start_time.normalize())
        # Chiều rộng 5 ngày cho cột tuần
        bar_width = 5 * 24 * 60 * 60 * 1000
    else: # D (Ngày)
        daily_actual['date'] = daily_actual['period'].apply(lambda x: x.end_time.normalize())
        bar_width = None # Plotly tự quyết định cho ngày
        
    # Sắp xếp và tính lũy kế
    daily_actual = daily_actual.sort_values('date')
    daily_actual['cumulative_actual'] = daily_actual['revenue'].cumsum()
    
    actual_data_points = daily_actual.copy()
    
    # Giá trị tổng thực hiện chính xác đến hôm nay (dùng cho Run-rate)
    current_actual_revenue = period_tours['revenue'].sum() 


    # --- 3. Xử lý Dữ liệu Kế hoạch và Dự báo ---
    
    # Prefer daily/weekly plans when available for higher-fidelity planned line
    total_planned_revenue = 0
    if (date_option in ['Tuần', 'Tùy chỉnh']) and plans_daily_df is not None and not plans_daily_df.empty:
        mask = (plans_daily_df['date'] >= pd.to_datetime(start_dt).normalize()) & (plans_daily_df['date'] <= pd.to_datetime(period_end_dt).normalize())
        df_slice = plans_daily_df.loc[mask].copy()
        # Prefer company-level rows when present to avoid summing unit-level + company-level
        try:
            # Only prefer company-level totals if user hasn't filtered by a segment
            seg = None if selected_segment is None else str(selected_segment).strip().upper()
            prefer_company = seg in (None, '') or seg in ('TẤT CẢ', 'TAT CA', 'ALL')
            if prefer_company and not plans_df.empty and plans_df['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN').any():
                df_slice = df_slice[df_slice['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN')]
        except Exception:
            pass
        daily_plan_line = df_slice[['date', 'planned_revenue_daily']].copy()
        daily_plan_line = daily_plan_line.sort_values('date')
        daily_plan_line['cumulative_planned'] = daily_plan_line['planned_revenue_daily'].cumsum()
        total_planned_revenue = daily_plan_line['planned_revenue_daily'].sum()
    elif (period_length := (period_end_dt - start_dt).days + 1) <= 60 and plans_weekly_df is not None and not plans_weekly_df.empty:
        # Use weekly plan points (week_start) and cumulative sum
        ws_start = (pd.to_datetime(start_dt) - pd.to_timedelta(pd.to_datetime(start_dt).weekday(), unit='d')).normalize()
        ws_end = (pd.to_datetime(period_end_dt) - pd.to_timedelta(pd.to_datetime(period_end_dt).weekday(), unit='d')).normalize()
        mask = (plans_weekly_df['week_start'] >= ws_start) & (plans_weekly_df['week_start'] <= ws_end)
        df_slice = plans_weekly_df.loc[mask].copy()
        try:
            seg = None if selected_segment is None else str(selected_segment).strip().upper()
            prefer_company = seg in (None, '') or seg in ('TẤT CẢ', 'TAT CA', 'ALL')
            if prefer_company and not plans_df.empty and plans_df['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN').any():
                df_slice = df_slice[df_slice['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN')]
        except Exception:
            pass
        weekly = df_slice[['week_start', 'planned_revenue_week']].copy()
        weekly = weekly.sort_values('week_start')
        weekly.rename(columns={'week_start': 'date', 'planned_revenue_week': 'planned_revenue_daily'}, inplace=True)
        weekly['cumulative_planned'] = weekly['planned_revenue_daily'].cumsum()
        daily_plan_line = weekly
        total_planned_revenue = weekly['planned_revenue_daily'].sum()
    else:
        # Fallback to monthly pro-rata (original logic)
        plan_mask = (plans_df['year'] == start_dt.year) & \
                    (plans_df['month'] >= start_dt.month) & \
                    (plans_df['month'] <= period_end_dt.month)
        period_plans = plans_df[plan_mask].copy()
        # Prefer company-level monthly totals when present
        try:
            seg = None if selected_segment is None else str(selected_segment).strip().upper()
            prefer_company = seg in (None, '') or seg in ('TẤT CẢ', 'TAT CA', 'ALL')
            if prefer_company and not period_plans.empty and period_plans['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN').any():
                period_plans = period_plans[period_plans['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN')]
        except Exception:
            pass
        # Compute planned revenue for the selected period by prorating monthly totals by overlap days
        total_planned_revenue = 0.0
        try:
            from calendar import monthrange
            # Build list of months between start_dt and period_end_dt (inclusive)
            s = start_dt.replace(day=1)
            e = period_end_dt.replace(day=1)
            months = []
            cur = s
            while cur <= e:
                months.append((int(cur.year), int(cur.month)))
                # advance month
                if cur.month == 12:
                    cur = cur.replace(year=cur.year+1, month=1)
                else:
                    cur = cur.replace(month=cur.month+1)

            # For each month, get the monthly plan amount (preferring company-level rows if present)
            for (y, m) in months:
                # days in this month
                dim = monthrange(y, m)[1]
                month_start = pd.to_datetime(datetime(y, m, 1))
                month_end = pd.to_datetime(datetime(y, m, dim))
                # overlap with our period
                overlap_start = max(pd.to_datetime(start_dt), month_start)
                overlap_end = min(pd.to_datetime(period_end_dt), month_end)
                overlap_days = (overlap_end - overlap_start).days + 1
                if overlap_days <= 0:
                    continue

                # Sum planned_revenue for this month (respecting company-level preference applied earlier)
                pmask = (plans_df['year'] == y) & (plans_df['month'] == m)
                month_plans = plans_df[pmask] if not plans_df.empty else pd.DataFrame()
                if month_plans.empty:
                    continue
                # If prefer company-level totals, try to filter to company rows
                try:
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

                rev_month = month_plans['planned_revenue'].sum()
                # Add prorated portion for this month's overlap
                total_planned_revenue += rev_month * (float(overlap_days) / float(dim))
        except Exception:
            # Fallback: use naive sum if prorating fails
            total_planned_revenue = period_plans['planned_revenue'].sum()

        # If monthly period_plans are empty (or sum to zero) but an annual TOTAL exists, use/prorate it
        if (total_planned_revenue == 0) and not plans_df.empty:
            try:
                months_in_period = (period_end_dt.year - start_dt.year) * 12 + (period_end_dt.month - start_dt.month) + 1
                annual_rows = plans_df[(plans_df['year'] == start_dt.year) & (plans_df['month'] == 0)]
                if not annual_rows.empty:
                    # If user filtered to a specific segment, try match segment in annual rows
                    matched = pd.DataFrame()
                    if selected_segment and str(selected_segment).strip().upper() not in ('TẤT CẢ', 'TAT CA', 'ALL'):
                        seg_up = str(selected_segment).strip().upper()
                        seg_mask = annual_rows['segment'].fillna('').astype(str).str.upper().str.contains(seg_up)
                        if seg_mask.any():
                            matched = annual_rows[seg_mask]
                    if matched.empty:
                        comp_mask = annual_rows['business_unit'].fillna('').str.upper().str.contains('TOAN|TOÀN')
                        if comp_mask.any():
                            matched = annual_rows[comp_mask]
                    if matched.empty:
                        matched = annual_rows

                    rev_ann = matched['planned_revenue'].sum()
                    total_planned_revenue = rev_ann * (float(months_in_period) / 12.0 if months_in_period > 0 else 0)
            except Exception:
                pass

        # Tính Kế hoạch lũy kế tuyến tính (Planned Line)
        plan_date_range = pd.date_range(start=start_dt, end=period_end_dt, freq='D') 
        total_days_in_period = (period_end_dt - start_dt).days + 1
        daily_plan_rate = total_planned_revenue / total_days_in_period if total_days_in_period > 0 else 0
        daily_plan_line = pd.DataFrame({'date': plan_date_range})
        # cumulative planned: day 1 -> daily_plan_rate, day N -> N * daily_plan_rate
        daily_plan_line['cumulative_planned'] = (daily_plan_line['date'] - start_dt).dt.days * daily_plan_rate + daily_plan_rate
    
    # Tính Dự báo Run-rate 
    days_elapsed = (today - start_dt).days + 1
    daily_run_rate = current_actual_revenue / days_elapsed if days_elapsed > 0 else 0
    
    # Tạo chuỗi Dự báo (Forecast Line)
    forecast_dates = pd.date_range(start=today, end=period_end_dt, freq='D')
    forecast_line = pd.DataFrame({'date': forecast_dates})

    forecast_line['cumulative_forecast'] = current_actual_revenue + (
        (forecast_line['date'] - today).dt.days * daily_run_rate
    )
    
    # --- 4. Tạo Biểu đồ Kết hợp ---
    
    fig = go.Figure()

    # Trace 1: Thực hiện Lũy kế (Dạng cột) — keep cumulative height but show
    # per-period revenue as visible labels on each bar. This provides the
    # requested "số doanh thu thực hiện trên mỗi cột ngày" while preserving
    # the cumulative visual used for run-rate/forecast.
    # Show cumulative value on bar labels (bar height is already cumulative_actual).
    try:
        per_bar_text = [format_currency(v) for v in actual_data_points['cumulative_actual']]
    except Exception:
        # Fallback to numeric formatting of cumulative values
        per_bar_text = [format_number(v) for v in actual_data_points.get('cumulative_actual', [])]

    fig.add_trace(go.Bar(
        x=actual_data_points['date'],
        y=actual_data_points['cumulative_actual'],
        name='Thực hiện Lũy kế',
        marker_color='#636EFA',
        width=bar_width,
        text=per_bar_text,
        textposition='outside',
        textfont=dict(size=10, color='#FFFFFF'),
    hovertemplate=(f'{x_title}: %{{x|{x_format}}}'
               f'<br>Lũy kế Thực hiện: %{{y:,.0f}} ₫'
               f'<br>Trong kỳ (không lũy kế): %{{customdata[0]:,.0f}} ₫<extra></extra>'),
        customdata=np.column_stack([actual_data_points['revenue']]) if not actual_data_points.empty else None
    ))
    
    # Trace 2: Kế hoạch Lũy kế (Đường)
    fig.add_trace(go.Scatter(
        x=daily_plan_line['date'],
        y=daily_plan_line['cumulative_planned'],
        name='Kế hoạch Lũy kế',
        mode='lines',
        line=dict(color='#EF553B', width=2),
        hovertemplate='Ngày: %{x|%d/%m}<br>Kế hoạch: %{y:,.0f} ₫<extra></extra>'
    ))

    # Trace 3: Đường Dự báo Cuối kỳ (Đường nét đứt)
    anchor_point = pd.DataFrame({
        'date': [today],
        'cumulative_forecast': [current_actual_revenue]
    })
    
    forecast_dates_extended = pd.concat([anchor_point, 
                                         forecast_line[['date', 'cumulative_forecast']]], ignore_index=True)
                                         
    fig.add_trace(go.Scatter(
        x=forecast_dates_extended['date'],
        y=forecast_dates_extended['cumulative_forecast'],
        name='Dự báo Cuối kỳ',
        mode='lines',
        line=dict(color='#00CC96', width=2, dash='dot'),
        hovertemplate='Ngày: %{x|%d/%m}<br>Dự báo: %{y:,.0f} ₫<extra></extra>'
    ))
    
    # --- 5. Cập nhật Layout và Định dạng ---
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Doanh thu Lũy kế (₫)",
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=40, b=30),
        hovermode='x unified'
    )
    
    fig.update_xaxes(tickformat=x_format, title=x_title, tickangle=0)
    # --- 6. Thêm chú thích hiển thị Số thực hiện / Số kế hoạch (định dạng tiền) ---
    try:
        planned_val = float(total_planned_revenue or 0)
    except Exception:
        planned_val = 0.0
    try:
        actual_val = float(current_actual_revenue or 0)
    except Exception:
        actual_val = 0.0

    # Tính phần trăm hoàn thành an toàn
    pct = (actual_val / planned_val * 100) if planned_val > 0 else 0

    # Xác định vị trí chú thích: gần cạnh phải trên cùng của chart
    try:
        max_y = max(
            actual_data_points['cumulative_actual'].max() if not actual_data_points.empty else 0,
            daily_plan_line['cumulative_planned'].max() if (isinstance(daily_plan_line, pd.DataFrame) and 'cumulative_planned' in daily_plan_line.columns and not daily_plan_line.empty) else 0,
            forecast_dates_extended['cumulative_forecast'].max() if 'cumulative_forecast' in forecast_dates_extended.columns and not forecast_dates_extended.empty else 0
        )
    except Exception:
        max_y = max(planned_val, actual_val, 1)

    # Format text using existing format_currency helper
    info_text = f"Thực hiện: {format_currency(actual_val)} / Kế hoạch: {format_currency(planned_val)}<br>Hoàn thành: {pct:.1f}%"
    fig.add_annotation(
        x=period_end_dt,
        y=max_y * 0.95,
        xref='x',
        yref='y',
        text=info_text,
        showarrow=False,
        align='right',
        font=dict(size=12, color='#FFFFFF'),
        bordercolor='#333333',
        borderwidth=1,
        bgcolor='rgba(0,0,0,0.5)'
    )

    return fig

def create_trend_chart(tours_df, start_date, end_date, metrics=['revenue', 'customers', 'profit']):
    """
    Create a multi-line trend chart showing trends over time
    """
    # Filter confirmed bookings in period
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    # Calculate period length in days
    period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    # Determine grouping granularity
    if period_length <= 7:
        # Daily granularity for week or less
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('D')
        period_data = period_tours.groupby('period').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        period_data['period_str'] = period_data['period'].dt.strftime('%d/%m')
        x_title = "Ngày"
    elif period_length <= 60:
        # Weekly granularity for 2 months or less
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('W')
        period_data = period_tours.groupby('period').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        period_data['period_str'] = period_data['period'].apply(lambda x: f"T{x.week}")
        x_title = "Tuần"
    else:
        # Monthly granularity for longer periods
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('M')
        period_data = period_tours.groupby('period').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        period_data['period_str'] = period_data['period'].astype(str)
        x_title = "Tháng"
    
    monthly_data = period_data
    
    # Create figure
    fig = go.Figure()
    
    if 'revenue' in metrics:
        # Show numeric labels on revenue points (formatted as currency)
        try:
            rev_text = [format_currency(v) for v in monthly_data['revenue']]
        except Exception:
            rev_text = [format_number(v) for v in monthly_data['revenue']]

        fig.add_trace(go.Scatter(
            x=monthly_data['period_str'],
            y=monthly_data['revenue'],
            name='Doanh thu',
            mode='lines+markers+text',
            text=rev_text,
            textposition='top center',
            textfont=dict(size=10, color='#FFFFFF'),
            line=dict(color='#636EFA', width=2),
            yaxis='y1'
        ))
    
    if 'customers' in metrics:
        # Show numeric labels for customer points (plain integer)
        try:
            cust_text = [format_number(v) for v in monthly_data['num_customers']]
        except Exception:
            cust_text = [str(int(v)) for v in monthly_data['num_customers']]

        fig.add_trace(go.Scatter(
            x=monthly_data['period_str'],
            y=monthly_data['num_customers'],
            name='Lượt khách',
            mode='lines+markers+text',
            text=cust_text,
            textposition='top center',
            textfont=dict(size=9, color='#00FFCC'),
            line=dict(color='#00CC96', width=2),
            yaxis='y2'
        ))
    
    if 'profit' in metrics:
        # Show numeric labels on profit points (formatted as currency)
        try:
            prof_text = [format_currency(v) for v in monthly_data['gross_profit']]
        except Exception:
            prof_text = [format_number(v) for v in monthly_data['gross_profit']]

        fig.add_trace(go.Scatter(
            x=monthly_data['period_str'],
            y=monthly_data['gross_profit'],
            name='Lãi Gộp',
            mode='lines+markers+text',
            text=prof_text,
            textposition='top center',
            textfont=dict(size=9, color='#F3A6FF'),
            line=dict(color='#AB63FA', width=2),
            yaxis='y1'
        ))
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis=dict(title="Doanh thu / Lãi Gộp (₫)", side='left'),
        yaxis2=dict(title="Lượt khách", overlaying='y', side='right'),
        height=250,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=40, b=30),
        hovermode='x unified'
    )
    
    return fig


def calculate_marketing_metrics(tours_df, start_date, end_date):
    """
    Calculate marketing and sales cost metrics (OPEX)
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    total_revenue = period_tours['revenue'].sum()
    total_opex = period_tours['opex'].sum()
    total_marketing = period_tours['marketing_cost'].sum()
    total_sales = period_tours['sales_cost'].sum()
    
    opex_ratio = (total_opex / total_revenue * 100) if total_revenue > 0 else 0
    
    return {
        'total_opex': total_opex,
        'total_marketing': total_marketing,
        'total_sales': total_sales,
        'total_revenue': total_revenue,
        'opex_ratio': opex_ratio
    }


def calculate_cac_by_channel(tours_df, start_date, end_date):
    """
    Calculate Customer Acquisition Cost (CAC) by sales channel
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    channel_metrics = period_tours.groupby('sales_channel').agg({
        'opex': 'sum',
        'customer_id': 'nunique',  # Unique customers
        'revenue': 'sum'
    }).reset_index()
    
    channel_metrics.columns = ['sales_channel', 'total_opex', 'unique_customers', 'revenue']
    # ĐÃ SỬA: Bảo vệ chia cho 0
    channel_metrics['cac'] = np.where(
        channel_metrics['unique_customers'] > 0,
        channel_metrics['total_opex'] / channel_metrics['unique_customers'],
        0
    )
    channel_metrics['cac'] = channel_metrics['cac'].fillna(0)
    
    return channel_metrics


def calculate_clv_by_segment(tours_df):
    """
    Calculate Customer Lifetime Value (CLV) by segment
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    
    # Calculate CLV = Total revenue from repeat customers / Number of customers
    segment_metrics = confirmed_tours.groupby('segment').agg({
        'customer_id': 'nunique',
        'revenue': 'sum',
        'booking_id': 'count'
    }).reset_index()
    
    segment_metrics.columns = ['segment', 'unique_customers', 'total_revenue', 'total_bookings']
    # ĐÃ SỬA: Bảo vệ chia cho 0
    segment_metrics['avg_bookings_per_customer'] = np.where(
        segment_metrics['unique_customers'] > 0,
        segment_metrics['total_bookings'] / segment_metrics['unique_customers'],
        0
    )
    # ĐÃ SỬA: Bảo vệ chia cho 0
    segment_metrics['clv'] = np.where(
        segment_metrics['unique_customers'] > 0,
        segment_metrics['total_revenue'] / segment_metrics['unique_customers'],
        0
    )
    segment_metrics['clv'] = segment_metrics['clv'].fillna(0)
    
    return segment_metrics


def get_channel_breakdown(tours_df, start_date, end_date, metric='revenue'):
    """
    Get breakdown by sales channel
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    if metric == 'revenue':
        channel_data = period_tours.groupby('sales_channel').agg({
            'revenue': 'sum',
            'num_customers': 'sum'
        }).reset_index()
        # ĐÃ SỬA: Bảo vệ chia cho 0
        channel_data['avg_revenue_per_customer'] = np.where(
            channel_data['num_customers'] > 0,
            channel_data['revenue'] / channel_data['num_customers'],
            0
        )
        return channel_data
    elif metric == 'customers':
        channel_data = period_tours.groupby('sales_channel').agg({
            'num_customers': 'sum',
            'revenue': 'sum' # Giữ revenue để tính Avg Rev per customer
        }).reset_index()
        # ĐÃ SỬA: Bảo vệ chia cho 0
        channel_data['avg_revenue_per_customer'] = np.where(
            channel_data['num_customers'] > 0,
            channel_data['revenue'] / channel_data['num_customers'],
            0
        )
        return channel_data
    else:  # profit
        channel_data = period_tours.groupby('sales_channel').agg({
            'gross_profit': 'sum'
        }).reset_index()
        return channel_data


def create_profit_margin_chart_with_color(data, x_col, y_col, title):
    """
    Create horizontal bar chart with continuous color scale (temperature/heatmap style)
    """
    # Use continuous color scale based on margin values
    fig = go.Figure(go.Bar(
        x=data[x_col],
        y=data[y_col],
        orientation='h',
        marker=dict(
            color=data[x_col],
            colorscale='RdYlGn',  # Red-Yellow-Green temperature scale
            showscale=True,
            colorbar=dict(
                title=dict(text="Tỷ suất LN (%)", side="right"),
                tickmode="linear",
                tick0=0,
                dtick=2,
                len=0.7
            ),
            cmin=data[x_col].min(),
            cmax=data[x_col].max()
        ),
        text=data[x_col].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='%{y}<br>Tỷ suất: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Tỷ suất Lãi Gộp (%)",
        yaxis_title="",
        height=max(300, len(data) * 30),
        margin=dict(l=30, r=100, t=50, b=30),
        showlegend=False
    )
    
    return fig


def get_route_detailed_table(tours_df, plans_df, start_date, end_date):
    """
    Get detailed table by route with plan comparison, including occupancy and cancellation rates.
    """
    # Lấy TẤT CẢ bookings trong kỳ (để tính hủy/đổi và tổng capacity)
    period_tours_all = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_tours = filter_confirmed_bookings(period_tours_all)
    
    if period_tours_all.empty: 
        # Cập nhật danh sách cột trả về
        return pd.DataFrame(columns=['route', 'revenue', 'num_customers', 'gross_profit', 'profit_margin', 
                                     'planned_revenue', 'revenue_completion', 'occupancy_rate', 'cancel_rate'])

    # 1. Tính ACTUALS, OCCUPANCY, và CANCEL/CHANGE Rate theo tuyến
    route_metrics = period_tours_all.groupby('route').agg(
        # Thực hiện
        revenue=('revenue', lambda x: x[period_tours_all['status'] == 'Đã xác nhận'].sum()),
        gross_profit=('gross_profit', lambda x: x[period_tours_all['status'] == 'Đã xác nhận'].sum()),
        num_customers_confirmed=('num_customers', lambda x: x[period_tours_all['status'] == 'Đã xác nhận'].sum()),
        num_customers_all=('num_customers', 'sum'),
        
        # Công suất và Hủy/Đổi
        tour_capacity=('tour_capacity', 'sum'),
        num_customers_cancelled=('num_customers', lambda x: x[x.index.isin(period_tours_all[period_tours_all['status'].isin(['Đã hủy', 'Hoãn'])].index)].sum())
    ).reset_index()
    
    # Tính Tỷ lệ Lấp đầy và Hủy/Đổi
    route_metrics['occupancy_rate'] = np.where(
        route_metrics['tour_capacity'] > 0,
        (route_metrics['num_customers_all'] / route_metrics['tour_capacity'] * 100).round(1),
        0
    )
    route_metrics['cancel_rate'] = np.where(
        route_metrics['num_customers_all'] > 0,
        (route_metrics['num_customers_cancelled'] / route_metrics['num_customers_all'] * 100).round(1),
        0
    )
    
    # 2. Xử lý Plans (Giữ nguyên logic cũ)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (
        (plans_df['year'] == start_dt.year) &
        (plans_df['month'] >= start_dt.month) &
        (plans_df['month'] <= end_dt.month)
    )
    period_plans = plans_df[plan_mask]
    
    route_plan = period_plans.groupby('route').agg({
        'planned_revenue': 'sum',
        'planned_customers': 'sum',
        'planned_gross_profit': 'sum'
    }).reset_index()
    
    # 3. Merge và Final Calculation
    route_table = route_metrics.merge(route_plan, on='route', how='left').fillna(0)
    
    # Tỷ suất LN
    route_table['profit_margin'] = np.where(
        route_table['revenue'] > 0,
        (route_table['gross_profit'] / route_table['revenue'] * 100).round(1),
        0
    )

    # Tỷ lệ Hoàn thành Kế hoạch
    route_table['revenue_completion'] = np.where(
        route_table['planned_revenue'] > 0,
        (route_table['revenue'] / route_table['planned_revenue'] * 100).round(1),
        0
    )
    
    # Đổi tên cột cho phù hợp với hiển thị cũ
    route_table.rename(columns={'num_customers_confirmed': 'num_customers'}, inplace=True)
    
    # Giới hạn các cột cuối cùng (Chỉ trả về các cột cần thiết)
    return route_table[['route', 'revenue', 'num_customers', 'gross_profit', 
                        'profit_margin', 'revenue_completion', 'occupancy_rate', 'cancel_rate']]

def get_unit_detailed_table(tours_df, plans_df, start_date, end_date):
    """
    Get detailed table by business unit
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    if period_tours.empty: 
        return pd.DataFrame(columns=['business_unit', 'revenue', 'num_customers', 'gross_profit', 'profit_margin', 'avg_revenue_per_customer'])

    # Actual data
    unit_actual = period_tours.groupby('business_unit').agg({
        'revenue': 'sum',
        'num_customers': 'sum',
        'gross_profit': 'sum'
    }).reset_index()
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    unit_actual['profit_margin'] = np.where(
        unit_actual['revenue'] > 0,
        (unit_actual['gross_profit'] / unit_actual['revenue'] * 100),
        0
    )
    # ĐÃ SỬA: Bảo vệ chia cho 0
    unit_actual['avg_revenue_per_customer'] = np.where(
        unit_actual['num_customers'] > 0,
        (unit_actual['revenue'] / unit_actual['num_customers']),
        0
    )
    
    # Plan data
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (
        (plans_df['year'] == start_dt.year) &
        (plans_df['month'] >= start_dt.month) &
        (plans_df['month'] <= end_dt.month)
    )
    period_plans = plans_df[plan_mask]
    
    unit_plan = period_plans.groupby('business_unit').agg({
        'planned_revenue': 'sum',
        'planned_customers': 'sum',
        'planned_gross_profit': 'sum'
    }).reset_index()
    
    # Merge
    unit_table = unit_actual.merge(unit_plan, on='business_unit', how='left').fillna(0)
    
    return unit_table

def get_unit_breakdown_simple(tours_df, metric='revenue'):
    """
    Get breakdown by business unit for a specific metric (revenue/customers/profit) for pie chart.
    """
    confirmed = filter_confirmed_bookings(tours_df)
    
    if confirmed.empty:
        return pd.DataFrame(columns=['business_unit', 'value', 'percentage'])
    
    if metric == 'revenue':
        unit_data = confirmed.groupby('business_unit')['revenue'].sum().reset_index()
    elif metric == 'customers':
        unit_data = confirmed.groupby('business_unit')['num_customers'].sum().reset_index()
    else:  # profit
        unit_data = confirmed.groupby('business_unit')['gross_profit'].sum().reset_index()
        
    unit_data.columns = ['business_unit', 'value']
    
    total_value = unit_data['value'].sum()
    # Bảo vệ chia cho 0
    unit_data['percentage'] = np.where(
        total_value > 0,
        (unit_data['value'] / total_value * 100).round(1),
        0
    )
    unit_data = unit_data.sort_values('value', ascending=False)
    
    return unit_data

# --- HÀM MỚI CHO TAB 3 ---

def calculate_partner_breakdown_by_type(tours_df, status_filter):
    """Calculates active/expiring partner count broken down by partner_type."""
    # Logic này yêu cầu cột 'partner_type' phải có trong tours_df
    df_filtered = tours_df[tours_df['contract_status'] == status_filter].copy()
    
    # Định nghĩa các loại đối tác cố định để đảm bảo Expander hiển thị đủ các loại
    partner_types = ['Khách sạn', 'Ăn uống', 'Vận chuyển', 'Vé máy bay', 'Điểm tham quan', 'Đối tác nước ngoài']
    
    if df_filtered.empty:
        # Trả về DataFrame với count = 0 cho tất cả các loại
        return pd.DataFrame([{'type': t, 'count': 0} for t in partner_types])

    breakdown = df_filtered.groupby('partner_type')['partner'].nunique().reset_index()
    breakdown.columns = ['type', 'count']
    
    # Xử lý các loại đối tác không có trong dữ liệu
    existing_types = breakdown['type'].tolist()
    missing_types = [t for t in partner_types if t not in existing_types]
    
    if missing_types:
        df_missing = pd.DataFrame([{'type': t, 'count': 0} for t in missing_types])
        breakdown = pd.concat([breakdown, df_missing], ignore_index=True)
    
    return breakdown
    
def calculate_partner_performance(partner_df):
    """
    Calculates key performance metrics for partners used in the scatter plot.
    """
    # Lấy dữ liệu đã xác nhận (hoặc dữ liệu đã lọc theo kỳ)
    # Vì partner_df đã được lọc theo date/dimensional, ta dùng nó trực tiếp
    
    if partner_df.empty:
        return pd.DataFrame(columns=['partner', 'total_revenue', 'avg_feedback', 'total_customers'])

    partner_performance = partner_df.groupby('partner').agg(
        total_revenue=('revenue', 'sum'),
        # Giả định cột feedback_ratio là tỷ lệ phản hồi tích cực (0-1)
        avg_feedback=('feedback_ratio', 'mean'), 
        total_customers=('num_customers', 'sum')
    ).reset_index()
    
    # Chuyển đổi tỷ lệ phản hồi thành phần trăm
    partner_performance['avg_feedback'] = partner_performance['avg_feedback'] * 100

    return partner_performance

def calculate_partner_revenue_by_type(partner_df):
    """
    Calculates total revenue grouped by service_type for expander detail.
    """
    if partner_df.empty:
        return pd.DataFrame(columns=['service_type', 'revenue'])
    
    revenue_by_type = partner_df.groupby('service_type')['revenue'].sum().reset_index()
    revenue_by_type = revenue_by_type.sort_values('revenue', ascending=False)
    
    return revenue_by_type    

# HÀM MỚI CHO VÙNG 2 TAB 3: TÍNH TỔNG TỒN KHO DỊCH VỤ VÀ TỶ LỆ HỦY DỊCH VỤ
def calculate_service_inventory(tours_df, service_type=None):
    """Calculates total service units (customers) held by type."""
    df = tours_df.copy()
    
    if service_type and service_type != "Tất cả":
        df = df[df['service_type'] == service_type]
    
    # Tính tổng số lượng khách hàng sử dụng dịch vụ này (đơn vị tồn kho)
    inventory = df.groupby('service_type')['num_customers'].sum().reset_index()
    inventory.columns = ['service_type', 'total_units']
    return inventory

def calculate_service_cancellation_metrics(tours_df):
    """Calculates service cancellation rate based on contract status."""
    
    # Giả định: Các tour có trạng thái 'Đã hủy' hoặc 'Hoãn' liên quan đến hủy dịch vụ
    total_services = len(tours_df)
    
    if total_services == 0:
        return {'cancel_rate': 0, 'total_cancelled': 0}

    # Giả định: Hủy hợp đồng = Hủy/Hoãn Tour
    cancelled_services = tours_df[tours_df['status'].isin(['Đã hủy', 'Hoãn'])]
    total_cancelled = len(cancelled_services)
    
    cancellation_rate = (total_cancelled / total_services) * 100
    
    return {
        'cancel_rate': cancellation_rate,
        'total_cancelled': total_cancelled
    }



def calculate_partner_kpis(tours_df):
    """Calculate core KPIs for Partner Management (Vùng 1)"""
    
    # Lọc dữ liệu hợp đồng đang triển khai (Đơn giản hóa: dùng trạng thái hợp đồng)
    active_contracts = tours_df[tours_df['contract_status'].isin(["Đang triển khai", "Sắp hết hạn"])]
    
    total_active_partners = active_contracts['partner'].nunique()
    total_contracts = active_contracts['partner'].count()
    
    # Tình trạng hợp đồng
    contracts_status_count = active_contracts.groupby('contract_status')['partner'].count().reset_index()
    contracts_status_count.columns = ['status', 'count']
    
    # Tình trạng thanh toán
    payment_status_count = tours_df.groupby('payment_status')['partner'].count().reset_index()
    payment_status_count.columns = ['status', 'count']
    
    # Dịch vụ đang giữ
    service_inventory = tours_df.groupby('service_type')['num_customers'].sum().reset_index()
    service_inventory.columns = ['service_type', 'total_units']
    
    # Tính tổng doanh thu dịch vụ
    total_service_revenue = tours_df['revenue'].sum()
    
    return {
        'total_active_partners': total_active_partners,
        'total_contracts': total_contracts,
        'contracts_status_count': contracts_status_count,
        'payment_status_count': payment_status_count,
        'service_inventory': service_inventory,
        'total_service_revenue': total_service_revenue
    }

def calculate_partner_revenue_metrics(tours_df):
    """Calculate service price metrics (Vùng 2)"""
    
    # Tính giá dịch vụ (giá trung bình/khách)
    tours_df['service_price_per_pax'] = np.where(
        tours_df['num_customers'] > 0,
        tours_df['service_cost'] / tours_df['num_customers'],
        0
    )
    
    # Group by service type
    service_metrics = tours_df.groupby('service_type').agg(
        max_price=('service_price_per_pax', 'max'),
        avg_price=('service_price_per_pax', 'mean'),
        min_price=('service_price_per_pax', 'min'),
    ).reset_index()
    
    return service_metrics

def create_partner_trend_chart(tours_df, start_date, end_date):
    """Creates a combined bar/line chart for partner revenue and customer count (Vùng 3)"""
    
    period_tours = filter_data_by_date(tours_df, start_date, end_date, date_column='booking_date')
    
    # Tương tự như create_trend_chart, xác định granularity
    period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    if period_length <= 60:
        freq = 'W'
        x_title = "Tuần"
        date_col = 'week_start'
        period_tours['week_start'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('W').apply(lambda x: x.start_time)
        df_trend = period_tours.groupby('week_start').agg(
            revenue=('revenue', 'sum'),
            customers=('num_customers', 'sum')
        ).reset_index()
    else:
        freq = 'M'
        x_title = "Tháng"
        date_col = 'month_start'
        period_tours['month_start'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('M').apply(lambda x: x.start_time)
        df_trend = period_tours.groupby('month_start').agg(
            revenue=('revenue', 'sum'),
            customers=('num_customers', 'sum')
        ).reset_index()
        
    if df_trend.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Doanh thu (Cột - Trục Y chính)
    fig.add_trace(
        go.Bar(x=df_trend[date_col], y=df_trend['revenue'], name='Doanh thu Dịch vụ', marker_color='#636EFA'),
        secondary_y=False,
    )

    # Trace 2: Lượt khách (Đường - Trục Y phụ)
    fig.add_trace(
        go.Scatter(x=df_trend[date_col], y=df_trend['customers'], name='Số lượng Khách', mode='lines+markers', line=dict(color='#FFA15A', width=3)),
        secondary_y=True,
    )

    # Cập nhật layout
    fig.update_layout(
        title_text=f"Xu hướng Doanh thu và Số lượng Khách theo {x_title}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=50, r=50, t=50, b=30),
    )

    # Thiết lập trục Y chính (Doanh thu)
    fig.update_yaxes(title_text="Doanh thu (₫)", secondary_y=False, tickformat=".2s")
    # Thiết lập trục Y phụ (Khách)
    fig.update_yaxes(title_text="Số lượng Khách", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text=x_title)

    return fig

# HÀM MỚI CHO TAB 2  
def calculate_booking_metrics(tours_df, start_date, end_date):
    """Calculates Total Booked Customers, Success Rate, and Cancellation/Change Rate."""
    period_tours = filter_data_by_date(tours_df, start_date, end_date)
    # Use column 'cancel_count' (cột U) as number of cancelled pax when available.
    # Ignore negative values in num_customers (treat them as 0 for totals).
    # Cancellation rate = sum(cancel_count) / sum(num_customers_filtered)
    # Success rate = (sum(num_customers_filtered) - sum(cancel_count)) / sum(num_customers_filtered)

    # Ensure numeric columns
    if 'num_customers' in period_tours.columns:
        num_cust_series = pd.to_numeric(period_tours['num_customers'], errors='coerce').fillna(0)
        # Exclude negative values
        num_cust_series = num_cust_series[num_cust_series > 0]
        total_customers_all = int(num_cust_series.sum()) if not num_cust_series.empty else 0
    else:
        total_customers_all = 0

    if 'cancel_count' in period_tours.columns:
        cancel_series = pd.to_numeric(period_tours['cancel_count'], errors='coerce').fillna(0)
        # Treat negative cancel_count as 0
        cancel_series = cancel_series[cancel_series > 0]
        total_cancelled = int(cancel_series.sum()) if not cancel_series.empty else 0
    else:
        # Fallback: count cancelled bookings by status if cancel_count not present
        cancelled_changed = period_tours[period_tours['status'].isin(['Đã hủy', 'Hoãn'])]
        total_cancelled = int(cancelled_changed['num_customers'].sum()) if not cancelled_changed.empty else 0

    if total_customers_all > 0:
        cancel_change_rate = (total_cancelled / total_customers_all) * 100
        success_rate = ((total_customers_all - total_cancelled) / total_customers_all) * 100
    else:
        cancel_change_rate = 0
        success_rate = 0

    return {
        'total_booked_customers': total_customers_all,
        'success_rate': success_rate,
        'cancel_change_rate': cancel_change_rate
    }

# Trong file utils.py

def create_cancellation_trend_chart(tours_df, start_date, end_date):
    """Creates a line chart showing the trend of cancelled/changed customers (Absolute Count)."""
    cancelled_df = tours_df[tours_df['status'].isin(['Đã hủy', 'Hoãn'])].copy()
    period_cancelled = filter_data_by_date(cancelled_df, start_date, end_date)
    
    if period_cancelled.empty:
        return go.Figure().update_layout(height=250, title="Không có dữ liệu hủy/đổi tour")

    period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    # SỬA LỖI: Ưu tiên NGÀY cho kỳ ngắn (< 30 ngày)
    if period_length <= 30: # <--- Ưu tiên Ngày cho kỳ 1 tháng hoặc ít hơn
        freq_unit = 'D'
        x_title = "Ngày"
    elif period_length <= 60:
        freq_unit = 'W'
        x_title = "Tuần"
    else:
        freq_unit = 'M'
        x_title = "Tháng"
    
    period_cancelled['period'] = pd.to_datetime(period_cancelled['booking_date']).dt.to_period(freq_unit)
    
    trend_data = period_cancelled.groupby('period').agg(
        total_customers=('num_customers', 'sum')
    ).reset_index()
    
    # Định dạng trục X (quan trọng để hiển thị ngày thay vì tuần)
    if freq_unit == 'D':
        trend_data['period_str'] = trend_data['period'].dt.strftime('%d/%m')
        # Đặt tên cột y cho đúng với số lượng tuyệt đối
        y_label = "Lượt khách hủy/đổi" 
    elif freq_unit == 'W':
        trend_data['period_str'] = trend_data['period'].apply(lambda x: f"W{x.week}-{x.year}")
        y_label = "Lượt khách hủy/đổi"
    else:
        trend_data['period_str'] = trend_data['period'].astype(str)
        y_label = "Lượt khách hủy/đổi"
        
    
    fig = px.line(
        trend_data, 
        x='period_str', 
        y='total_customers', 
        title='Xu hướng Lượt khách hủy/đổi tour',
        markers=True
    )
    
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_label)
    fig.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
    
    return fig


def create_demographic_pie_chart(tours_df, grouping_col, title):
    """Creates a pie chart showing revenue share by age group or nationality."""
    # Giả định cột customer_age_group và customer_nationality tồn tại trong tours_df
    if grouping_col not in tours_df.columns:
        return go.Figure().update_layout(height=250, title=f"Thiếu cột '{grouping_col}'")
        
    confirmed = filter_confirmed_bookings(tours_df)
    
    grouped = confirmed.groupby(grouping_col).agg(
        total_revenue=('revenue', 'sum')
    ).reset_index()
    
    # Pie Chart
    fig = px.pie(
        grouped, 
        values='total_revenue', 
        names=grouping_col,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(
        textinfo='percent+label', 
        hovertemplate='<b>%{label}</b><br>Doanh thu: %{value:,.0f} ₫<br>Tỉ lệ: %{percent}<extra></extra>'
    )
    fig.update_layout(
        height=250, 
        margin=dict(t=30, b=10, l=10, r=10), 
        showlegend=False
    )
    return fig

def create_ratio_trend_chart(tours_df, start_date, end_date, metric='success_rate', title=''):
    """
    Creates a line chart showing the trend of a specified ratio metric (Success Rate or Cancellation Rate).
    Requires re-calculating the ratio for each period.
    """
    
    period_tours = filter_data_by_date(tours_df, start_date, end_date)
    if period_tours.empty:
        return go.Figure().update_layout(height=250, title=f"Không có dữ liệu {title}")

    # Xác định độ phân giải (Tuần hoặc Tháng)
    period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    # SỬA LỖI: Buộc theo NGÀY nếu kỳ báo cáo ngắn (<= 7 ngày)
    if period_length <= 7:
        freq_unit = 'D' # <--- SỬA THÀNH NGÀY
        x_title = "Ngày"
    elif period_length <= 60:
        freq_unit = 'W'
        x_title = "Tuần"
    else:
        freq_unit = 'M'
        x_title = "Tháng"
    
    # 1. Nhóm dữ liệu theo Period và tính các thành phần cần thiết
    period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period(freq_unit)

    # For the requested trend charts we show absolute counts (per your request):
    # - success: number of customers with status 'Đã xác nhận' per period
    # - cancellation: number of cancelled customers per period (use cancel_count if available, otherwise count status)
    if metric == 'success_rate':
        # compute total successful customers per period
        def sum_confirmed_customers(df):
            # sum num_customers where status == 'Đã xác nhận'
            try:
                s = df.loc[df['status'] == 'Đã xác nhận', 'num_customers']
                return pd.to_numeric(s, errors='coerce').fillna(0).sum()
            except Exception:
                return 0

        trend_data = period_tours.groupby('period').apply(lambda d: pd.Series({
            'value': sum_confirmed_customers(d)
        })).reset_index()
        y_label = "Lượt khách đặt thành công"
        color_seq = ['#636EFA']
    else:
        # cancellation count per period: prefer summing 'cancel_count' (column U) if present
        def sum_cancelled_customers(df):
            if 'cancel_count' in df.columns:
                try:
                    s = pd.to_numeric(df['cancel_count'], errors='coerce').fillna(0)
                    # treat negative as 0
                    s = s[s > 0]
                    return s.sum()
                except Exception:
                    return 0
            else:
                # fallback: sum num_customers where status in ['Đã hủy','Hoãn']
                try:
                    s = df.loc[df['status'].isin(['Đã hủy', 'Hoãn']), 'num_customers']
                    return pd.to_numeric(s, errors='coerce').fillna(0).sum()
                except Exception:
                    return 0

        trend_data = period_tours.groupby('period').apply(lambda d: pd.Series({
            'value': sum_cancelled_customers(d)
        })).reset_index()
        y_label = "Lượt khách hủy/đổi"
        color_seq = ['#EF553B']
    
    # Định dạng trục X (quan trọng để hiển thị ngày thay vì tuần)
    if freq_unit == 'D':
        trend_data['period_str'] = trend_data['period'].dt.strftime('%d/%m')
    elif freq_unit == 'W':
        trend_data['period_str'] = trend_data['period'].apply(lambda x: f"W{x.week}-{x.year}")
    else:
        trend_data['period_str'] = trend_data['period'].astype(str)
        
    
    # 3. Tạo biểu đồ đường
    fig = px.line(
        trend_data,
        x='period_str',
        y='value',
        title=title,
        markers=True,
        color_discrete_sequence=color_seq
    )
    # Force integer-like hover/labels for counts
    fig.update_traces(hovertemplate='%{x}<br>%{y:.0f}<extra></extra>')
    
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_label)
    fig.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
    
    return fig

# Trong file utils.py (Hàm create_stacked_route_chart)

def create_stacked_route_chart(tours_df, metric='revenue', title='', top_n=10):
    """
    Creates a stacked bar chart showing the metric total per Route, segmented by Business Unit (BU).
    """
    confirmed = filter_confirmed_bookings(tours_df)
    
    # 1. Lấy Top N Routes theo Doanh thu làm cơ sở sắp xếp
    try:
        top_routes = confirmed.groupby('route')['revenue'].sum().nlargest(int(top_n)).index.tolist()
    except Exception:
        # Fallback to all routes if grouping fails
        top_routes = confirmed['route'].dropna().unique().tolist()
    
    # 2. NHÓM DỮ LIỆU theo Route VÀ Đơn vị Kinh doanh (BU)
    df_grouped = confirmed[confirmed['route'].isin(top_routes)].groupby(
        ['route', 'business_unit']
    ).agg(
        revenue=('revenue', 'sum'),
        num_customers=('num_customers', 'sum'),
        gross_profit=('gross_profit', 'sum')
    ).reset_index()
    
    if df_grouped.empty:
        return go.Figure().update_layout(height=250, title=title)
        
    # Xác định các cột và định dạng
    # ... (Giữ nguyên logic xác định y_col, hover_format, yaxis_title) ...
    if metric == 'revenue':
        y_col = 'revenue'
        hover_format = '₫'
        yaxis_title = 'Doanh thu (₫)'
    elif metric == 'num_customers':
        y_col = 'num_customers'
        hover_format = ''
        yaxis_title = 'Lượt khách'
    else: # gross_profit
        y_col = 'gross_profit'
        hover_format = '₫'
        yaxis_title = 'Lãi Gộp (₫)'

    # Tạo biểu đồ Cột xếp chồng (Dùng df_grouped)
    fig = px.bar(
        df_grouped, # <--- ĐÃ SỬA: Dùng DataFrame đã nhóm
        x='route',
        y=y_col,
        color='business_unit', # Xếp chồng theo Đơn vị Kinh doanh
        title=title,
        category_orders={'route': top_routes}, # Giữ nguyên thứ tự Top N
        color_discrete_sequence=px.colors.qualitative.T10
    )
    
    # ... (Giữ nguyên update_layout và update_traces) ...
    fig.update_layout(
        barmode='stack', # Đảm bảo xếp chồng
        xaxis_title="Tuyến Tour",
        yaxis_title=yaxis_title,
        height=300,
        margin=dict(t=30, b=10, l=10, r=10),
        legend_title_text='Đơn vị KD'
    )
    
    return fig



def create_top_routes_dual_axis_chart(df_data):
    """
    Creates a grouped bar chart comparing Revenue and Profit (Y1) with Customers (Y2)
    for the top routes. Uses Dual Axis.
    """
    if df_data.empty:
        return go.Figure().update_layout(height=400)
        
    # Tạo Subplots với Trục Phụ (Secondary Y-axis)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Trace 1 & 2: Revenue and Profit (Trục Y Chính - Currency)
    # Prepare formatted text labels (with fallbacks)
    try:
        revenue_texts = [format_currency(v) for v in df_data['revenue']]
    except Exception:
        revenue_texts = [format_number(v) for v in df_data['revenue']]

    try:
        profit_texts = [format_currency(v) for v in df_data['gross_profit']]
    except Exception:
        profit_texts = [format_number(v) for v in df_data['gross_profit']]

    cust_texts = [format_number(v) for v in df_data['num_customers']]

    # Revenue bar with larger/contrasting labels
    fig.add_trace(go.Bar(
        x=df_data['route'], y=df_data['revenue'], name='Doanh thu', marker_color='#636EFA',
        text=revenue_texts, textposition='outside', textfont=dict(size=12, color='#FFFFFF'),
        hovertemplate='DT: %{y:,.0f} ₫<extra></extra>'
    ), secondary_y=False)

    # Profit bar with larger/contrasting labels (use white so it shows on dark background)
    fig.add_trace(go.Bar(
        x=df_data['route'], y=df_data['gross_profit'], name='Lãi Gộp', marker_color='#FFA15A',
        text=profit_texts, textposition='outside', textfont=dict(size=12, color='#FFFFFF'),
        hovertemplate='LN: %{y:,.0f} ₫<extra></extra>'
    ), secondary_y=False)

    # Customers line with slightly larger point labels
    fig.add_trace(go.Scatter(
        x=df_data['route'], y=df_data['num_customers'], name='Lượt khách', marker=dict(color='#00CC96', size=8),
        mode='lines+markers+text', line=dict(dash='dot', width=3),
        text=cust_texts, textposition='top center', textfont=dict(size=11, color='#00CC96'),
        hovertemplate='LK: %{y:,.0f}<extra></extra>'
    ), secondary_y=True)

    # Cấu hình Layout
    fig.update_layout(
        title_text="So sánh DT, LN (Cột) và LK (Đường)",
        barmode='group',
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'),
        xaxis=dict(title="Tuyến Tour", tickangle=45),
        yaxis=dict(title="Doanh thu / Lãi Gộp (₫)", side='left', showgrid=True),
        yaxis2=dict(title="Lượt khách", side='right', showgrid=False)
    )
    return fig

def create_top_routes_ratio_stacked(df_data):
    """
    Creates a 100% Stacked Bar Chart showing the contribution of each top route 
    to the total Revenue, Customers, and Profit (3 Metrics).
    """
    # 1. Chuyển đổi dữ liệu sang định dạng Long
    df_long = pd.melt(df_data, id_vars=['route'], 
                      value_vars=['revenue', 'num_customers', 'gross_profit'],
                      var_name='Metric', value_name='Value')
    
    df_long['Metric'] = df_long['Metric'].replace({
        'revenue': 'Doanh thu',
        'num_customers': 'Lượt khách',
        'gross_profit': 'Lãi Gộp'
    })

    # 2. Tính Tỷ trọng đóng góp của mỗi Route cho tổng thể (Metric)
    df_totals = df_long.groupby('Metric')['Value'].sum().reset_index().rename(columns={'Value': 'Total'})
    df_long = df_long.merge(df_totals, on='Metric')
    df_long['Ratio'] = df_long['Value'] / df_long['Total'] * 100

    # 3. Tạo Biểu đồ Cột Xếp chồng 100%
    fig = px.bar(
        df_long,
        x='Metric',
        y='Ratio',
        color='route',
        title='Tỷ trọng đóng góp của Top Tuyến Tour (%)',
        barmode='stack',
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data={'Ratio': ':.1f', 'Value': True}
    )
    
    fig.update_layout(
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        yaxis_title="Tỷ trọng (%)",
        xaxis_title="Chỉ số",
        yaxis_tickformat='.0f',
        legend_title_text='Tuyến Tour'
    )
    
    fig.update_traces(hovertemplate='<b>%{y:.1f}%</b><br>%{x}<br>Route: %{customdata[0]}<extra></extra>',
                      customdata=df_long[['route']])
    
    return fig   
 
def create_segment_bu_comparison_chart(df_data_long, grouping_col='segment'):
    """
    Creates a grouped bar chart comparing Revenue and Profit (Y1) with Customers (Y2)
    for Segments or Business Units.
    """
    if df_data_long.empty:
        return go.Figure().update_layout(height=350, title=f"Không có dữ liệu cho {grouping_col}")

    # 1. Chuẩn bị dữ liệu
    df_currency = df_data_long[df_data_long['Metric'].isin(['Revenue', 'Profit'])].copy()
    df_customers = df_data_long[df_data_long['Metric'] == 'Customers'].copy()
    
    # 2. Tạo Subplots với Trục Phụ (Secondary Y-axis)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 3. Trace 1 & 2: Revenue and Profit (Trục Y Chính - Currency)
    # Lấy các nhóm theo grouping_col (Segment hoặc Group)
    groups = df_data_long[grouping_col].unique()

    # Thêm Revenue
    df_rev = df_currency[df_currency['Metric'] == 'Revenue']
    # Add formatted labels on revenue bars
    try:
        rev_texts = [format_currency(v) for v in df_rev['Value']]
    except Exception:
        rev_texts = [format_number(v) for v in df_rev['Value']]

    fig.add_trace(go.Bar(
        x=df_rev[grouping_col], y=df_rev['Value'], name='Doanh thu', marker_color='#636EFA',
        text=rev_texts, textposition='outside', textfont=dict(size=8, color='#FFFFFF'),
        hovertemplate='DT: %{y:,.0f} ₫<extra></extra>'
    ), secondary_y=False)
    
    # Thêm Profit
    df_prof = df_currency[df_currency['Metric'] == 'Profit']
    try:
        prof_texts = [format_currency(v) for v in df_prof['Value']]
    except Exception:
        prof_texts = [format_number(v) for v in df_prof['Value']]

    fig.add_trace(go.Bar(
        x=df_prof[grouping_col], y=df_prof['Value'], name='Lãi Gộp', marker_color='#FFA15A',
        text=prof_texts, textposition='outside', textfont=dict(size=8, color='#000000'),
        hovertemplate='LN: %{y:,.0f} ₫<extra></extra>'
    ), secondary_y=False)

    # 4. Trace 3: Customers (Trục Y Phụ - Count)
    fig.add_trace(go.Scatter(
        x=df_customers[grouping_col], y=df_customers['Value'], name='Lượt khách', 
        marker=dict(color='#00CC96', size=8),
        mode='lines+markers+text', line=dict(width=3),
        text=[format_number(v) for v in df_customers['Value']], textposition='top center', textfont=dict(size=9, color='#00FFCC'),
        hovertemplate='LK: %{y:,.0f}<extra></extra>'
    ), secondary_y=True)

    # 5. Cấu hình Layout
    # Improve readability when there are many groups:
    # - rotate x labels, enable automargin, and increase bottom margin
    # - use adaptive height based on number of groups
    n_groups = len(groups) if groups is not None else 0
    fig_height = max(350, int(n_groups * 45))
    fig.update_layout(
        barmode='group', # Hiển thị cột cạnh nhau để so sánh
        height=fig_height,
        margin=dict(t=30, b=max(80, int(n_groups * 8)), l=40, r=40),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor='center'),
    xaxis=dict(title=grouping_col, tickangle=90, automargin=True, tickfont=dict(size=10)),
        yaxis=dict(title="Tiền tệ (₫)", side='left', showgrid=True, tickformat='.2s'),
        yaxis2=dict(title="Lượt khách", side='right', showgrid=False)
    )
    # Reduce text clutter: let Plotly decide whether to put bar text inside/outside
    # and keep bar label font small. Also reduce marker label size for customers.
    # (This helps prevent numeric labels from overlapping when bars are very tall.)
    for t in fig.data:
        # For Bar traces, use 'auto' textposition so labels move inside/outside as needed
        if isinstance(t, go.Bar):
            t.textposition = 'auto'
            t.textfont = dict(size=8, color='#000000')
        # For Scatter (customers), keep smaller marker and smaller text to avoid overlap
        if isinstance(t, go.Scatter):
            # remove inline text to reduce overlap; rely on hover for exact numbers
            try:
                t.mode = 'lines+markers'
                t.marker = dict(color=t.marker.color if hasattr(t.marker, 'color') else '#00CC96', size=6)
            except Exception:
                pass
    return fig