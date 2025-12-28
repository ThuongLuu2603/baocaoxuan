"""
Vietravel Business Intelligence Dashboard
Comprehensive tour sales performance, revenue, profit margins, and operational metrics dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz # Cần thiết cho Timezone handling
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
# Cần import make_subplots ở đây để dùng trong app.py nếu cần cho chart phức tạp
from plotly.subplots import make_subplots 
from admin_ui import render_admin_ui

# Import custom modules
from data_generator import load_or_generate_data
from utils import (
    # Các hàm Format và Core Logic
    format_currency, format_number, format_percentage,
    calculate_completion_rate, get_growth_rate, filter_data_by_date, filter_confirmed_bookings,
    
    # Các hàm KPI và Chart
    calculate_kpis, 
    create_gauge_chart, create_bar_chart, create_pie_chart, create_line_chart,
    
    # Các hàm Top/Breakdown
    get_top_routes, get_route_unit_breakdown, get_unit_breakdown,
    get_segment_breakdown, get_segment_unit_breakdown, get_channel_breakdown,
    get_unit_breakdown_simple,
    
    # Các hàm Operational và Detailed Tables
    calculate_operational_metrics, get_low_margin_tours, get_unit_performance, 
    get_route_detailed_table, get_unit_detailed_table,
    
    # Các hàm Marketing/CLV/Forecast
    create_forecast_chart, create_trend_chart, 
    calculate_marketing_metrics, calculate_cac_by_channel, calculate_clv_by_segment, 
    create_profit_margin_chart_with_color,
    calculate_partner_performance,
    
    # Các hàm Đối tác mới (ĐÃ THÊM)
    calculate_partner_kpis, calculate_partner_revenue_metrics, create_partner_trend_chart,
    calculate_partner_breakdown_by_type,calculate_service_inventory, calculate_service_cancellation_metrics,
    calculate_partner_revenue_by_type,

    # CHỨC NĂNG MỚI CHO TAB 2
    calculate_booking_metrics, 
    create_cancellation_trend_chart, 
    create_demographic_pie_chart,
    create_ratio_trend_chart,
    create_stacked_route_chart,
    create_top_routes_dual_axis_chart,
    create_top_routes_ratio_stacked,
    create_segment_bu_comparison_chart
    , group_small_categories
)

# Page configuration
st.set_page_config(
    page_title="Vietravel BI Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to reduce padding and whitespace
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1 {
        padding-top: 0rem;
        margin-top: 0rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding-top: 8px;
        padding-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Nhập nguồn dữ liệu (đặt trước khi load dữ liệu)
# Mặc định sử dụng Google Sheet với link cố định
DEFAULT_DATANET_URL = 'https://docs.google.com/spreadsheets/d/1CljNuZ4WVNXGL7J111ZhVT9FPCVZDQsB6L5UHMgYeAc/edit?gid=29056776#gid=29056776'
DEFAULT_PLAN_URL = 'https://docs.google.com/spreadsheets/d/1CljNuZ4WVNXGL7J111ZhVT9FPCVZDQsB6L5UHMgYeAc/edit?gid=322447784#gid=322447784'

with st.sidebar:
    st.markdown("---")
    st.subheader("Nguồn dữ liệu")
    
    # Khởi tạo giá trị mặc định trong session_state nếu chưa có
    if 'use_sheet' not in st.session_state:
        st.session_state['use_sheet'] = True
    if 'sheet_url' not in st.session_state:
        st.session_state['sheet_url'] = DEFAULT_DATANET_URL
    if 'plan_sheet_url' not in st.session_state:
        st.session_state['plan_sheet_url'] = DEFAULT_PLAN_URL
    
    use_sheet = st.checkbox("Dùng Google Sheet (CSV public)", value=st.session_state.get('use_sheet', True))
    
    # Expander để người dùng có thể thay đổi link nếu cần
    with st.expander("🔧 Thay đổi nguồn dữ liệu", expanded=False):
        sheet_url = st.text_input(
            "Link Google Sheet (datanet)",
            value=st.session_state.get('sheet_url', DEFAULT_DATANET_URL),
            help="Dán link Google Sheet (bấm Share → Anyone with the link → Viewer). Có thể giữ #gid hiện tại."
        )
        plan_sheet_url = st.text_input(
            "Link Google Sheet (Kế hoạch)",
            value=st.session_state.get('plan_sheet_url', DEFAULT_PLAN_URL),
            help="Link Google Sheet chứa Kế hoạch. Header ở hàng 2, các đơn vị bắt đầu từ cột E, mỗi đơn vị chiếm 4 cột (Khách, Doanh thu, Lãi Gộp)."
        )
        # Lưu lại vào session_state để sử dụng khi load
        st.session_state['use_sheet'] = use_sheet
        st.session_state['sheet_url'] = sheet_url
        st.session_state['plan_sheet_url'] = plan_sheet_url
    
    # Hiển thị thông tin nguồn đang dùng (rút gọn)
    if use_sheet:
        st.caption(f"📊 Datanet: ...{st.session_state['sheet_url'][-20:]}")
        st.caption(f"📋 Kế hoạch: ...{st.session_state['plan_sheet_url'][-20:]}")

# Initialize session state for data
# Load data when not already loaded or when explicitly requested (data_loaded flag False)
if not st.session_state.get('data_loaded', False):
    with st.spinner('Đang tải dữ liệu...'):
        # load_or_generate_data now returns (tours_df, plans_df, historical_df, meta)
        # If the user checked "Dùng Google Sheet" we will load tours from the sheet.
        spreadsheet_url = st.session_state.get('sheet_url') if st.session_state.get('use_sheet') else None
        # IMPORTANT: plan sheet may be provided independently — always pass it to the loader
        plan_sheet_url = st.session_state.get('plan_sheet_url') if st.session_state.get('plan_sheet_url') else None
        # Pass both the data sheet and optional plan sheet to the loader
        result = load_or_generate_data(spreadsheet_url, plan_spreadsheet_url=plan_sheet_url)
        # Support both old and new signatures for safety
        if isinstance(result, tuple) and len(result) == 4:
            tours_df, plans_df, historical_df, data_meta = result
        else:
            tours_df, plans_df, historical_df = result
            data_meta = {'used_excel': False, 'processed_files': [], 'parsed_rows': 0}

        # Save loaded data into session state
        st.session_state['tours_df'] = tours_df
        st.session_state['plans_df'] = plans_df
        # If loader provided daily/weekly expanded plans in meta, save them to session state
        st.session_state['plans_daily_df'] = data_meta.get('plans_daily_df') if isinstance(data_meta, dict) else None
        st.session_state['plans_weekly_df'] = data_meta.get('plans_weekly_df') if isinstance(data_meta, dict) else None
        st.session_state['historical_df'] = historical_df
        st.session_state['data_meta'] = data_meta
        st.session_state['data_loaded'] = True

    # Show a banner if data was loaded from external source
    meta = st.session_state.get('data_meta', {})
    # Show banner if tours or plan sheets were used / parsed
    if meta.get('used_excel') or meta.get('used_sheet') or meta.get('parsed_plan_rows', 0) > 0:
        files = st.session_state['data_meta'].get('processed_files', [])
        plan_files = st.session_state['data_meta'].get('processed_plan_files', [])
        parsed = st.session_state['data_meta'].get('parsed_rows', 0)
        parsed_plan = st.session_state['data_meta'].get('parsed_plan_rows', 0)
        files_str = ', '.join(files) if files else '(<no data files>)'
        plan_files_str = ', '.join(plan_files) if plan_files else '(<no plan files>)'
        st.sidebar.success(f"Dữ liệu tours: {files_str} — {parsed} dòng parsed; Kế hoạch: {plan_files_str} — {parsed_plan} dòng parsed")

# Load data from session state
tours_df = st.session_state.tours_df
plans_df = st.session_state.plans_df
historical_df = st.session_state.historical_df

# Dashboard Title
st.title("📊 VIETRAVEL - DASHBOARD KINH DOANH TOUR")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Bộ lọc dữ liệu")
    
    # Date range selector
    st.subheader("Khoảng thời gian")
    
    # Quick date range options
    date_option = st.selectbox(
        "Chọn kỳ báo cáo",
        ["Tuần", "Tháng", "Quý", "Năm", "Tùy chỉnh"]
    )
    
    # Xử lý Timezone an toàn
    vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    today = datetime.now(vietnam_tz).replace(tzinfo=None) # Naive datetime
    
    if date_option == "Tuần":
        # Tuần hiện tại (Monday - Sunday)
        # weekday(): Monday=0, Sunday=6
        days_since_monday = today.weekday()  # 0=Mon, 6=Sun
        start_date = today - timedelta(days=days_since_monday)
        start_date = datetime(start_date.year, start_date.month, start_date.day)
        end_date = start_date + timedelta(days=6)  # Sunday of current week
        end_date = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59)
    elif date_option == "Tháng":
        # Toàn bộ tháng hiện tại (01 → last day of month)
        from calendar import monthrange
        start_date = datetime(today.year, today.month, 1)
        last_day = monthrange(today.year, today.month)[1]
        end_date = datetime(today.year, today.month, last_day, 23, 59, 59)
    elif date_option == "Quý":
        # Toàn bộ quý hiện tại (first day Q → last day Q)
        from calendar import monthrange
        quarter = (today.month - 1) // 3 + 1
        start_month = 3 * quarter - 2  # Q1:1, Q2:4, Q3:7, Q4:10
        end_month = 3 * quarter        # Q1:3, Q2:6, Q3:9, Q4:12
        start_date = datetime(today.year, start_month, 1)
        last_day = monthrange(today.year, end_month)[1]
        end_date = datetime(today.year, end_month, last_day, 23, 59, 59)
    elif date_option == "Năm":
        # Toàn bộ năm hiện tại (01/01 → 31/12)
        start_date = datetime(today.year, 1, 1)
        end_date = datetime(today.year, 12, 31, 23, 59, 59)
    else:  # Tùy chỉnh
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Từ ngày",
                value=datetime(today.year, today.month, 1)
            )
        with col2:
            end_date = st.date_input(
                "Đến ngày",
                value=today
            )
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
    
    st.markdown(f"**Kỳ báo cáo:** {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
    
    # Business unit filter
    st.subheader("Đơn vị kinh doanh")
    # Prefer business_unit list coming from the Plan sheet if available (Google Sheet authoritative)
    plans_df_ss = st.session_state.get('plans_df') if st.session_state.get('plans_df') is not None else plans_df
    try:
        plan_bus_units = [] if plans_df_ss is None else list(pd.Series(plans_df_ss['business_unit'].dropna().unique()) )
    except Exception:
        plan_bus_units = []
    if plan_bus_units:
        business_units = ["Tất cả"] + sorted(plan_bus_units)
    else:
        business_units = ["Tất cả"] + sorted(tours_df['business_unit'].unique().tolist())
    selected_unit = st.selectbox("Chọn đơn vị", business_units)
    
    # Route filter
    st.subheader("Tuyến tour")
    if selected_unit != "Tất cả":
        routes = ["Tất cả"] + sorted(
            tours_df[tours_df['business_unit'] == selected_unit]['route'].unique().tolist()
        )
    else:
        routes = ["Tất cả"] + sorted(tours_df['route'].unique().tolist())
    selected_route = st.selectbox("Chọn tuyến", routes)
    
    # Segment filter (use segments defined in the Plan sheet when available)
    st.subheader("Phân khúc")
    try:
        plan_segments = [] if plans_df_ss is None else list(pd.Series(plans_df_ss['segment'].dropna().unique()))
    except Exception:
        plan_segments = []

    # Canonical segment set we want to expose
    canonical = ['FIT', 'GIT', 'Inbound', 'Khác']
    seg_opts = ["Tất cả"]
    if plan_segments:
        normalized_plan_segs = [str(s).strip().upper() for s in plan_segments]
        for c in canonical:
            if any(c.upper() in s for s in normalized_plan_segs):
                # keep original casing for display when possible
                # find first original match
                for orig in plan_segments:
                    if c.upper() in str(orig).upper():
                        display = orig if c != 'Inbound' else 'Inbound'
                        seg_opts.append(display)
                        break
        # If canonical none found, fall back to plan_segments (exclude obvious category labels like 'Nội địa')
        if len(seg_opts) == 1:
            fallback = [s for s in plan_segments if str(s).strip().upper() not in ('NỘI ĐỊA', 'NƯỚC NGOÀI')]
            seg_opts += sorted(fallback)
    else:
        # fallback to tours_df segments
        raw_segments = [str(s).strip() for s in tours_df['segment'].dropna().unique().tolist()]
        normalized = [s.upper() for s in raw_segments]
        if any('FIT' in s for s in normalized): seg_opts.append('FIT')
        if any('GIT' in s for s in normalized): seg_opts.append('GIT')
        if any('INB' in s for s in normalized) or any('INBOUND' in s for s in normalized): seg_opts.append('Inbound')
        if any('KHÁC' in s or 'KHAC' in s or s == 'KHÁC' for s in normalized): seg_opts.append('Khác')

    selected_segment = st.selectbox("Chọn phân khúc", seg_opts)
    
    # Top N selector
    st.subheader("Thiết lập hiển thị")
    # Determine number of distinct routes (cột P / 'route') in the file and set slider maximum accordingly
    if 'route' in tours_df.columns:
        try:
            num_routes = int(tours_df['route'].dropna().nunique())
        except Exception:
            num_routes = 15
    else:
        num_routes = 15

    # Slider minimum: keep 5 where possible, but if there are fewer routes allow smaller min
    slider_min = 5 if num_routes >= 5 else 1
    # Slider maximum should be based on number of routes per user request
    slider_max = num_routes if num_routes >= slider_min else slider_min
    # Default value: 10 by default, but if there are fewer routes, default to num_routes
    default_top_n = 15 if num_routes >= 15 else num_routes

    top_n = st.slider("Top N tuyến tour", min_value=slider_min, max_value=slider_max, value=default_top_n)
    
    # Bổ sung Filter cho Tab 3
    st.markdown("---")
    st.subheader("Bộ lọc Đối tác")
    partners = ["Tất cả"] + sorted(tours_df['partner'].unique().tolist())
    selected_partner = st.selectbox("Chọn Đối tác", partners)
    
    service_types = ["Tất cả"] + sorted(tours_df['service_type'].unique().tolist())
    selected_service = st.selectbox("Chọn Loại dịch vụ", service_types)

    st.markdown("---")
    
    # Refresh data button
    if st.button("🔄 Làm mới dữ liệu", width='stretch'):
        st.session_state.data_loaded = False
        st.rerun()

# Filter data based on selections (dimensional filters only, NOT date)
# Date filtering will be done inside calculate_kpis to preserve YoY data
tours_filtered_dimensional = tours_df.copy()
filtered_plans = plans_df.copy()

if selected_unit != "Tất cả":
    tours_filtered_dimensional = tours_filtered_dimensional[tours_filtered_dimensional['business_unit'] == selected_unit]
    filtered_plans = filtered_plans[filtered_plans['business_unit'] == selected_unit]

if selected_route != "Tất cả":
    tours_filtered_dimensional = tours_filtered_dimensional[tours_filtered_dimensional['route'] == selected_route]
    filtered_plans = filtered_plans[filtered_plans['route'] == selected_route]

if selected_segment != "Tất cả":
    tours_filtered_dimensional = tours_filtered_dimensional[tours_filtered_dimensional['segment'] == selected_segment]
    # Don't filter plans by segment when using Google Sheets data as it may not have segment breakdown
    # Only filter if we're using generated data or if plans actually have valid segment data
    # Use the actual data_meta key stored in session_state (data_meta)
    data_meta = st.session_state.get('data_meta', {})
    # Only attempt to filter plans by segment if the plans dataframe actually has a 'segment' column
    if 'segment' in filtered_plans.columns:
        # If data was generated locally (not from sheet), it's safe to filter by segment
        if not data_meta.get('used_sheet', False):
            filtered_plans = filtered_plans[filtered_plans['segment'] == selected_segment]
        else:
            # If using sheet, only filter when the plans dataframe contains non-empty segment values
            if not filtered_plans['segment'].isna().all():
                filtered_plans = filtered_plans[filtered_plans['segment'] == selected_segment]
    # If plans remain empty after attempted filtering, try a fallback: match plan 'business_unit' to the selected segment
    # This helps when the sheet encodes segments as business_unit rows (e.g., a 'FIT' row)
    if filtered_plans.empty and 'business_unit' in plans_df.columns:
        candidate = plans_df[plans_df['business_unit'].astype(str).str.upper() == str(selected_segment).upper()]
        if not candidate.empty:
            filtered_plans = candidate.copy()

# Áp dụng bộ lọc đối tác cho Tab 3
partner_filtered_df = tours_filtered_dimensional.copy()
if selected_partner != "Tất cả":
    partner_filtered_df = partner_filtered_df[partner_filtered_df['partner'] == selected_partner]
if selected_service != "Tất cả":
    partner_filtered_df = partner_filtered_df[partner_filtered_df['service_type'] == selected_service]

# Calculate KPIs using dimensionally filtered data (calculate_kpis will handle date filtering)
# Pass daily/weekly expanded plans from session_state when available so KPIs use correct granularity
kpis = calculate_kpis(
    tours_filtered_dimensional,
    filtered_plans,
    start_date,
    end_date,
    plans_daily_df=st.session_state.get('plans_daily_df'),
    plans_weekly_df=st.session_state.get('plans_weekly_df'),
    period_type=date_option,
    selected_segment=selected_segment
)


# Also create a date+dimension filtered version for charts that don't need historical data
filtered_tours = filter_data_by_date(tours_filtered_dimensional, start_date, end_date)

# TÍNH TOÁN BOOKING METRICS CHO TAB 2 (ĐÃ DI CHUYỂN)
booking_metrics = calculate_booking_metrics(tours_df, start_date, end_date)


if 'show_admin_ui' not in st.session_state:
    st.session_state.show_admin_ui = False

# Nút mở/đóng UI Admin (đặt ở khu vực trên cùng)
col_toggle, col_empty = st.columns([1, 4])

with col_toggle:
    if st.session_state.show_admin_ui:
        if st.button("<< Quay lại Dashboard Chính", type="secondary"):
            st.session_state.show_admin_ui = False
            st.rerun()
    else:
        if st.button("🔧 Mở UI Nhập liệu/Sửa Hợp đồng (Admin)", type="secondary"):
            st.session_state.show_admin_ui = True
            st.rerun()

# ----------------------------------------------------
# KHU VỰC HIỂN THỊ UI ADMIN LỚN
# ----------------------------------------------------
if st.session_state.show_admin_ui:
    render_admin_ui() # <--- GỌI HÀM TỪ FILE admin_ui.py







# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard theo dõi Kinh Doanh",
    "🔍 Dashboard theo dõi sản phẩm",
    "🤝 Dashboard theo dõi Đối tác" 
])

# ============================================================
# TAB 1: TỔNG QUAN (5 VÙNG THEO SPEC)
# ============================================================
with tab1:
    # ========== VÙNG 1: TỐC ĐỘ ĐẠT KẾ HOẠCH ==========
    st.markdown("### Vùng 1: Tốc độ đạt Kế hoạch")
    
    # Row: 3 Gauge charts + 1 Forecast chart
    col1, col2, col3 = st.columns(3)
    
    # Get unit breakdown data for hover tooltips
    revenue_breakdown = get_unit_breakdown(filtered_tours, filtered_plans, start_date, end_date, metric='revenue')
    profit_breakdown = get_unit_breakdown(filtered_tours, filtered_plans, start_date, end_date, metric='profit')
    customers_breakdown = get_unit_breakdown(filtered_tours, filtered_plans, start_date, end_date, metric='customers')
    
    with col1:
        fig_revenue = create_gauge_chart(
            kpis['revenue_completion'],
            "Đạt KH Doanh thu",
            unit_breakdown=revenue_breakdown,
            actual_value=kpis.get('actual_revenue'),
            planned_value=kpis.get('planned_revenue')
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        profit_completion = calculate_completion_rate(kpis['actual_gross_profit'], kpis['planned_gross_profit'])
        fig_profit = create_gauge_chart(
            profit_completion,
            "Đạt KH Lãi Gộp",
            unit_breakdown=profit_breakdown,
            actual_value=kpis.get('actual_gross_profit'),
            planned_value=kpis.get('planned_gross_profit')
        )
        st.plotly_chart(fig_profit, use_container_width=True)
    
    with col3:
        fig_customers = create_gauge_chart(
            kpis['customer_completion'],
            "Đạt KH Lượt khách",
            unit_breakdown=customers_breakdown,
            actual_value=kpis.get('actual_customers'),
            planned_value=kpis.get('planned_customers')
        )
        st.plotly_chart(fig_customers, use_container_width=True)
    
# ========== BIỂU ĐỒ DỰ BÁO HOÀN THÀNH KẾ HOẠCH (SỬA LỖI 4 ĐỐI SỐ) ==========
# Hàng 2: Tiến độ KH theo Khu vực (1 cột) | Dự báo Hoàn thành KH (2 cột)
    st.markdown("#### Phân tích Tiến độ & Dự báo")
    col1, col2 = st.columns([1, 2]) # Tỉ lệ 1:2
    
    # Lấy dữ liệu cần thiết cho Hàng 2
    unit_performance = get_unit_performance(tours_filtered_dimensional, filtered_plans, start_date, end_date)
    # Sắp xếp theo tiến độ Doanh thu (từ cao -> thấp) để biểu đồ hiển thị rõ ràng
    if not unit_performance.empty and 'revenue_completion' in unit_performance.columns:
        unit_performance = unit_performance.sort_values('revenue_completion', ascending=False).reset_index(drop=True)
    
    with col1:
        st.markdown("##### 📊 Tiến độ KH theo Khu vực")
        if not unit_performance.empty:
            fig = go.Figure()
            colors = ['#00CC96' if x >= 100 else '#FFA500' if x >= 80 else '#EF553B' 
                        for x in unit_performance['revenue_completion']]
            customdata = [[row['actual_revenue'], row['planned_revenue'], row['revenue_completion']]
                          for _, row in unit_performance.iterrows()]
            fig.add_trace(go.Bar(
                x=unit_performance['business_unit'],
                y=unit_performance['revenue_completion'],
                text=[f"{v:.1f}%" for v in unit_performance['revenue_completion']],
                textposition='outside',
                marker_color=colors,
                customdata=customdata,
                hovertemplate='<b>%{x}</b><br>DT thực hiện: %{customdata[0]:,.0f} ₫<br>DT kế hoạch: %{customdata[1]:,.0f} ₫<br>Tiến độ: %{customdata[2]:.1f}%<extra></extra>'
            ))
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="KH 100%")
            fig.update_layout(xaxis_title="", yaxis_title="Tiến độ (%)", height=300, showlegend=False, margin=dict(l=30, r=30, t=10, b=30))
            st.plotly_chart(fig)
        else:
            st.info("Không có dữ liệu tiến độ cho khu vực kinh doanh được chọn.")
    
    with col2:
        st.markdown("##### 📈 Dự báo Hoàn thành Kế hoạch")
        # Ensure forecast chart respects selected segment when preferring company totals
        fig_forecast = create_forecast_chart(
            filtered_tours,
            filtered_plans,
            start_date,
            end_date,
            date_option,
            plans_daily_df=st.session_state.get('plans_daily_df'),
            plans_weekly_df=st.session_state.get('plans_weekly_df'),
            selected_segment=selected_segment
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown("---")
    


    # ========== VÙNG 2: CHỈ SỐ TỔNG QUAN ==========
    st.markdown("###  Vùng 2: Các Chỉ số")
    
    # Row 1: 3 KPI Cards 
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💰 DOANH THU TỔNG",
            value=format_currency(kpis['actual_revenue']),
            delta=f"{format_percentage(kpis['revenue_growth'])} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**Kế hoạch:** {format_currency(kpis['planned_revenue'])}")
            st.write(f"**Thực hiện:** {format_currency(kpis['actual_revenue'])}")
            st.write(f"**Hoàn thành:** {format_percentage(kpis['revenue_completion'])}")
            st.write(f"**Cùng kỳ năm trước:** {format_currency(kpis['ly_revenue'])}")
            st.write(f"**Tăng trưởng:** {format_percentage(kpis['revenue_growth'])}")
    
    with col2:
        st.metric(
            label="💵 Lãi Gộp",
            value=format_currency(kpis['actual_gross_profit']),
            delta=f"{format_percentage(kpis['profit_growth'])} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**Kế hoạch:** {format_currency(kpis['planned_gross_profit'])}")
            st.write(f"**Thực hiện:** {format_currency(kpis['actual_gross_profit'])}")
            profit_completion = calculate_completion_rate(kpis['actual_gross_profit'], kpis['planned_gross_profit'])
            st.write(f"**Hoàn thành:** {format_percentage(profit_completion)}")
            st.write(f"**Cùng kỳ năm trước:** {format_currency(kpis['ly_gross_profit'])}")
            st.write(f"**Tăng trưởng:** {format_percentage(kpis['profit_growth'])}")
    
    with col3:
        st.metric(
            label="👥 LƯỢT KHÁCH TỔNG",
            value=format_number(kpis['actual_customers']),
            delta=f"{format_percentage(kpis['customer_growth'])} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**Kế hoạch:** {format_number(kpis['planned_customers'])}")
            st.write(f"**Thực hiện:** {format_number(kpis['actual_customers'])}")
            st.write(f"**Hoàn thành:** {format_percentage(kpis['customer_completion'])}")
            st.write(f"**Cùng kỳ năm trước:** {format_number(kpis['ly_customers'])}")
            st.write(f"**Tăng trưởng:** {format_percentage(kpis['customer_growth'])}")
    
    # Row 2: Marketing/Sales Cost and Trend Chart
    st.markdown("")
    col1, col2 = st.columns([1, 2])
    # Tính toán AOV
    aov = kpis['actual_revenue'] / kpis['actual_customers'] if kpis['actual_customers'] > 0 else 0
    ly_aov = kpis['ly_revenue'] / kpis['ly_customers'] if kpis['ly_customers'] > 0 else 0
    aov_growth = get_growth_rate(aov, ly_aov)    
    
    with col1:
        st.metric(
            label="💵 DOANH THU TB/KHÁCH (AOV)",
            value=format_currency(aov),
            delta=f"{format_percentage(aov_growth)} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**AOV Cùng kỳ:** {format_currency(ly_aov)}")
            st.write(f"**Tăng trưởng AOV:** {format_percentage(aov_growth)}")
            st.write(f"**Doanh thu Tổng:** {format_currency(kpis['actual_revenue'])}")
            st.write(f"**Lượt khách Tổng:** {format_number(kpis['actual_customers'])}")
    
    with col2:
        st.markdown("<div style='font-size: 14px; font-weight: bold; margin-bottom: 10px;'>📊 Xu hướng Doanh thu / Lượt khách / Lãi Gộp theo thời gian</div>", unsafe_allow_html=True)
        fig_trend = create_trend_chart(filtered_tours, start_date, end_date, metrics=['revenue', 'customers', 'profit'])
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    
    
    # ========== VÙNG 3: PHÂN THEO PHÂN KHÚC & ĐƠN VỊ KINH DOANH ==========
    st.markdown("### Vùng 3: Phân theo Phân khúc & Đơn vị Kinh doanh")
    SEGMENT_COLORS = ['#3CB371', '#6495ED', '#FFA07A']
    BU_COLORS = ['#3CB371', '#6495ED', '#FFA07A', '#FF6347']
    
    # --- HÀNG 1: PHÂN TÍCH THEO PHÂN KHÚC (BAR CHART NHÓM) ---
    st.markdown("#### Hàng 1: Hiệu suất theo Phân khúc (FIT / GIT / Inbound)")
    col1, col2 = st.columns(2)
    
    # 1. Chuẩn bị dữ liệu cho Phân khúc (Revenue, Customers, Profit)
    segment_revenue = get_segment_breakdown(filtered_tours, start_date, end_date, metric='revenue')
    segment_customers = get_segment_breakdown(filtered_tours, start_date, end_date, metric='customers')
    segment_profit = get_segment_breakdown(filtered_tours, start_date, end_date, metric='profit')

    # Gom dữ liệu Phân khúc
    df_segment_comp = segment_revenue[['segment', 'value']].rename(columns={'value': 'Revenue'}).merge(
        segment_customers[['segment', 'value']].rename(columns={'value': 'Customers'}), on=['segment'], how='outer'
    ).merge(
        segment_profit[['segment', 'value']].rename(columns={'value': 'Profit'}), on=['segment'], how='outer'
    ).fillna(0)
    
    # Chuyển sang định dạng long
    df_segment_long = pd.melt(df_segment_comp, id_vars=['segment'], 
                              value_vars=['Revenue', 'Customers', 'Profit'], 
                              var_name='Metric', value_name='Value')

    with col1:
        st.markdown("##### 📈 So sánh DT, LK, LN theo Phân khúc")
        fig_segment_bar = create_segment_bu_comparison_chart(df_segment_long, grouping_col='segment') # Hàm mới
        fig_segment_bar.update_layout(height=350)
        st.plotly_chart(fig_segment_bar, use_container_width=True)
        
    with col2:
        st.markdown("##### Phân bố Doanh thu ")
        # Vẫn giữ 1 Pie Chart Doanh thu để xem tỷ trọng (%)
        if not segment_revenue.empty:
            # Group small categories (<1%) into 'Khác'
            seg_grouped = group_small_categories(segment_revenue, value_col='value', name_col='segment', threshold=0.02, other_label='Khác')
            fig = go.Figure(go.Pie(
                labels=seg_grouped['segment'],
                values=seg_grouped['value'],
                textinfo='label+percent',
                marker=dict(colors=SEGMENT_COLORS),
                domain=dict(x=[0, 1], y=[0, 1])
            ))
            # Attach detailed hover info (for 'Khác' show its components)
            fig.update_traces(textfont=dict(size=12),
                              customdata=seg_grouped['detail'],
                              hovertemplate='<b>%{label}</b><br>%{percent}<br>%{customdata}<extra></extra>')
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=40), showlegend=False)
            st.plotly_chart(fig)


    st.markdown("---")
    
    # --- HÀNG 2: PHÂN TÍCH THEO KHU VỰC (BAR CHART NHÓM) ---
    st.markdown("#### Hàng 2: Hiệu suất theo Khu vực Đơn vị Kinh doanh")
    
    # 2. Chuẩn bị dữ liệu cho Đơn vị Kinh doanh
    bu_revenue = get_unit_breakdown_simple(filtered_tours, metric='revenue').rename(columns={'value': 'Revenue', 'business_unit': 'group'})
    bu_customers = get_unit_breakdown_simple(filtered_tours, metric='customers').rename(columns={'value': 'Customers', 'business_unit': 'group'})
    bu_profit = get_unit_breakdown_simple(filtered_tours, metric='profit').rename(columns={'value': 'Profit', 'business_unit': 'group'})
    
    # Gom dữ liệu Đơn vị Kinh doanh
    df_bu_comp = bu_revenue[['group', 'Revenue']].merge(
        bu_customers[['group', 'Customers']], on='group', how='inner'
    ).merge(
        bu_profit[['group', 'Profit']], on='group', how='inner'
    )
    
    df_bu_long = pd.melt(df_bu_comp, id_vars=['group'], 
                              value_vars=['Revenue', 'Customers', 'Profit'], 
                              var_name='Metric', value_name='Value')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📈 So sánh DT, LK, LN theo Khu vực")
        fig_bu_bar = create_segment_bu_comparison_chart(df_bu_long, grouping_col='group') # Hàm mới
        fig_bu_bar.update_layout(height=450)
        st.plotly_chart(fig_bu_bar, use_container_width=True)
        
    with col2:
        st.markdown("##### Phân bố Doanh thu Khu vực ")
        if not bu_revenue.empty:
            # Group small categories (<1%) into 'Khác'
            bu_grouped = group_small_categories(bu_revenue.rename(columns={'group': 'group', 'value': 'value'}).rename(columns={'group': 'group'}).assign(**{'group': bu_revenue['group'], 'value': bu_revenue['Revenue']}), value_col='value', name_col='group', threshold=0.02, other_label='Khác')
            # Use colors; if number of slices > colors, Plotly will cycle colors
            fig = go.Figure(go.Pie(
                labels=bu_grouped['group'],
                values=bu_grouped['value'],
                textinfo='label+percent',
                marker=dict(colors=BU_COLORS),
                domain=dict(x=[0, 1], y=[0, 1])
            ))
            fig.update_traces(textfont=dict(size=12),
                              customdata=bu_grouped['detail'],
                              hovertemplate='<b>%{label}</b><br>%{percent}<br>%{customdata}<extra></extra>')
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=40), showlegend=False)
            st.plotly_chart(fig)
    
    st.markdown("---")

    
    # ========== VÙNG 4: THEO ĐƠN VỊ KINH DOANH ==========
    st.markdown("### Vùng 4: Hiệu suất theo Đơn vị Kinh doanh")
    
    # Get unit data
    unit_table = get_unit_detailed_table(filtered_tours, filtered_plans, start_date, end_date)
    
    # Row 1: Revenue vs Plan comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### So sánh Doanh thu Thực hiện và Kế hoạch")
        if not unit_table.empty:
            # Sort units by actual revenue descending so highest revenue appears left-most
            unit_table = unit_table.sort_values('revenue', ascending=False).reset_index(drop=True)
            # Convert to vertical grouped bars (Doanh thu Kế hoạch vs Thực hiện)
            try:
                planned_text = [format_currency(v) for v in unit_table['planned_revenue']]
            except Exception:
                planned_text = [format_number(v) for v in unit_table['planned_revenue']]
            try:
                actual_text = [format_currency(v) for v in unit_table['revenue']]
            except Exception:
                actual_text = [format_number(v) for v in unit_table['revenue']]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=unit_table['business_unit'],
                y=unit_table['planned_revenue'],
                name='Kế hoạch',
                marker_color='#FFA15A',
                text=planned_text,
                textposition='outside',
                textfont=dict(size=9, color='#000000')
            ))

            fig.add_trace(go.Bar(
                x=unit_table['business_unit'],
                y=unit_table['revenue'],
                name='Thực hiện',
                marker_color='#636EFA',
                text=actual_text,
                textposition='outside',
                textfont=dict(size=9, color='#FFFFFF')
            ))

            height = max(400, int(len(unit_table) * 25))
            fig.update_layout(
                xaxis_title="Đơn vị",
                yaxis_title="Doanh thu (₫)",
                height=height,
                barmode='group',
                margin=dict(l=80, r=30, t=10, b=140),
                xaxis=dict(tickangle=-45, tickfont=dict(size=10), categoryorder='array', categoryarray=unit_table['business_unit'])
            )
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("#### Tỷ suất Lãi Gộp theo Đơn vị")
        if not unit_table.empty:
            unit_margin = unit_table[['business_unit', 'profit_margin']].copy()
            # Sort by profit margin descending so highest margin units appear left-most
            unit_margin = unit_margin.sort_values('profit_margin', ascending=False).reset_index(drop=True)
            # Build vertical bar chart with continuous color scale
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=unit_margin['business_unit'],
                y=unit_margin['profit_margin'],
                marker=dict(
                    color=unit_margin['profit_margin'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title=dict(text="Tỷ suất LN (%)", side="right")),
                    cmin=unit_margin['profit_margin'].min(),
                    cmax=unit_margin['profit_margin'].max()
                ),
                text=[f"{v:.1f}%" for v in unit_margin['profit_margin']],
                textposition='outside'
            ))
            fig2.update_layout(
                xaxis_title='Đơn vị',
                yaxis_title='Tỷ suất Lãi Gộp (%)',
                height=max(400, int(len(unit_margin) * 25)),
                margin=dict(l=80, r=80, t=10, b=140),
                xaxis=dict(tickangle=-45, tickfont=dict(size=10), categoryorder='array', categoryarray=unit_margin['business_unit'])
            )
            st.plotly_chart(fig2)
    
    # Row 2: Detailed table
    st.markdown("#### Bảng số liệu chi tiết theo Đơn vị")
    if not unit_table.empty:
        display_df = unit_table.copy()
        display_df = display_df[[
            'business_unit', 'revenue', 'num_customers', 'gross_profit',
            'profit_margin', 'avg_revenue_per_customer'
        ]]
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['gross_profit'] = display_df['gross_profit'].apply(format_currency)
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
        display_df['avg_revenue_per_customer'] = display_df['avg_revenue_per_customer'].apply(format_currency)
        display_df.columns = ['Đơn vị', 'Doanh thu', 'Lượt khách', 'Lãi Gộp', 'Tỷ suất LN (%)', 'DT TB/khách']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
# ========== VÙNG 5: THÔNG TIN TUYẾN TOUR ==========
    st.markdown("### Vùng 5: Thông tin tuyến tour")

    # Chuẩn bị dữ liệu cho cả 3 chỉ số
    top_revenue = get_top_routes(filtered_tours, n=10, metric='revenue')
    top_customers = get_top_routes(filtered_tours, n=10, metric='customers')
    top_profit = get_top_routes(filtered_tours, n=10, metric='profit')

    # Hợp nhất dữ liệu Top 10 vào 1 DataFrame duy nhất để so sánh
    # Đảm bảo kiểu dữ liệu nhất quán cho cột route và loại bỏ NaN
    if not top_revenue.empty:
        top_revenue = top_revenue.copy()
        top_revenue['route'] = top_revenue['route'].fillna('').astype(str).str.strip()
        top_revenue = top_revenue[top_revenue['route'] != ''].copy()
    
    if not top_customers.empty:
        top_customers = top_customers.copy()
        top_customers['route'] = top_customers['route'].fillna('').astype(str).str.strip()
        top_customers = top_customers[top_customers['route'] != ''].copy()
    
    if not top_profit.empty:
        top_profit = top_profit.copy()
        top_profit['route'] = top_profit['route'].fillna('').astype(str).str.strip()
        top_profit = top_profit[top_profit['route'] != ''].copy()
    
    # Tạo DataFrame merge từ top_revenue (đã clean)
    if not top_revenue.empty:
        df_merged_top10 = top_revenue[['route', 'revenue']].copy()
        if not top_customers.empty:
            df_merged_top10 = df_merged_top10.merge(top_customers[['route', 'num_customers']], on='route', how='left')
        if not top_profit.empty:
            df_merged_top10 = df_merged_top10.merge(top_profit[['route', 'gross_profit']], on='route', how='left')
        df_merged_top10 = df_merged_top10.fillna(0)
        df_merged_top10 = df_merged_top10.sort_values('revenue', ascending=False) # Sắp xếp theo DT
    else:
        df_merged_top10 = pd.DataFrame()

    # --- HÀNG 1: BIỂU ĐỒ 1 - SO SÁNH TUYỆT ĐỐI (TRỤC KÉP) ---
    st.markdown("#### Hàng 1: So sánh Giá trị Tuyệt đối (Doanh thu, Lượt khách, Lãi Gộp)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📈 So sánh DT, LK, LN theo Top 10 Tuyến Tour")
        if not df_merged_top10.empty:
            # Hàm mới: Biểu đồ cột nhóm/kết hợp với trục kép
            fig_dual_axis = create_top_routes_dual_axis_chart(df_merged_top10) # <--- Hàm mới
            st.plotly_chart(fig_dual_axis, use_container_width=True)
        else:
            st.info("Không có dữ liệu Top 10 Tuyến Tour.")

    # --- HÀNG 2: BIỂU ĐỒ 2 - TỶ TRỌNG ĐÓNG GÓP (100% STACKED PIE/BAR) ---
    with col2:
        st.markdown("##### 📊 Tỷ trọng Đóng góp của Top 10 Tuyến Tour")
        if not df_merged_top10.empty:
            # Hàm mới: Biểu đồ cột xếp chồng 100% cho Tỷ trọng DT, LK, LN
            fig_stacked_ratio = create_top_routes_ratio_stacked(df_merged_top10) # <--- Hàm mới
            st.plotly_chart(fig_stacked_ratio, use_container_width=True)
        else:
            st.info("Không có dữ liệu tỷ trọng.")


    st.markdown("---")
    
    # ========== VÙNG 6: CHỈ SỐ QUẢN LÝ HOẠT ĐỘNG ==========
    st.markdown("### Vùng 6: Chỉ số Quản lý Hoạt động")
    
    # Calculate operational metrics (use all-time dimensional data for accurate rates)
    ops_metrics = calculate_operational_metrics(tours_filtered_dimensional)
    
    # Row: 3 Operational gauge charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_occ = create_gauge_chart(
            ops_metrics['avg_occupancy'],
            "Tỷ lệ Lấp đầy BQ (FIT)",
            max_value=100,
            threshold=75
        )
        st.plotly_chart(fig_occ, key="gauge_tab1")
    
    with col2:
        fig_cancel = create_gauge_chart(
            ops_metrics['cancel_rate'],
            "Tỷ lệ Khách Hủy/Hoãn",
            max_value=30,
            threshold=10,
            is_inverse_metric=True
        )
        st.plotly_chart(fig_cancel)
    
    with col3:
        fig_return = create_gauge_chart(
            ops_metrics['returning_rate'],
            "Tỷ lệ Khách Quay lại",
            max_value=100,
            threshold=30
        )
        st.plotly_chart(fig_return)


# ============================================================
# TAB 2: CHI TIẾT (3 VÙNG THEO SPEC)
# ============================================================
with tab2:
    route_table = get_route_detailed_table(filtered_tours, filtered_plans, start_date, end_date)
    top_revenue = get_top_routes(filtered_tours, n=top_n, metric='revenue')
    top_customers = get_top_routes(filtered_tours, n=top_n, metric='customers')
    top_profit = get_top_routes(filtered_tours, n=top_n, metric='profit')
# ========== VÙNG 1: TÓM TẮT HIỆU SUẤT BOOKING (ĐÃ THÊM KPI VÀ TRENDS) ==========
    st.markdown("### Vùng 1: Tóm tắt Hiệu suất Booking")
    
    # --- Hàng 1: KPI Cấp cao ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Số lượng khách đã đặt",
            value=format_number(booking_metrics['total_booked_customers'])
        )

    with col2:
        st.metric(
            label="💰 Tổng Doanh thu",
            value=format_currency(kpis['actual_revenue'])
        )
    with col3:
        st.markdown("##### 📈 Tỷ lệ Lấp đầy BQ (FIT)")
        fig_occ = create_gauge_chart(
            ops_metrics['avg_occupancy'],
            "Tỷ lệ Lấp đầy BQ (FIT)",
            max_value=100, 
            threshold=75,
            is_inverse_metric=False
        )
        st.plotly_chart(fig_occ, use_container_width=True, key="gauge_tab2")
    with col4:
        st.empty()

    st.markdown("---")


    # --- Hàng 2: Tỷ lệ Thành công (Gauge & Trend) ---
    st.markdown("#### 🟢 Hiệu suất Booking Thành công")
    col1, col2 = st.columns([1, 3]) # Tỷ lệ 1:3 cho Gauge và Line Chart

    with col1:
        # Tỷ lệ booking thành công (Gauge Chart)
        fig_success = create_gauge_chart(
            booking_metrics['success_rate'],
            "Tỷ lệ booking thành công",
            max_value=100, 
            threshold=90
        )
        st.plotly_chart(fig_success, use_container_width=True)
    
    with col2:
        # Xu hướng tỷ lệ booking thành công (Line Chart)
        fig_success_trend = create_ratio_trend_chart(tours_df, start_date, end_date, 
                                                     metric='success_rate', 
                                                     title='Xu hướng Tỷ lệ Booking Thành công (Theo ngày/tuần)')
        st.plotly_chart(fig_success_trend, use_container_width=True)

    st.markdown("---")


    # --- Hàng 3: Tỷ lệ Hủy/Đổi (Gauge & Trend) ---
    st.markdown("#### 🔴 Hiệu suất Khách Hàng Hủy/Đổi")
    col1, col2 = st.columns([1, 3]) # Tỷ lệ 1:3 cho Gauge và Line Chart

    with col1:
        # Tỷ lệ khách hàng hủy tour hoặc thay đổi (Gauge Chart)
        fig_cancel = create_gauge_chart(
            booking_metrics['cancel_change_rate'],
            "Tỷ lệ Khách Hủy/Đổi",
            max_value=30, 
            threshold=15, 
            is_inverse_metric=True
        )
        st.plotly_chart(fig_cancel, use_container_width=True)
        
    with col2:
        # Xu hướng tỷ lệ khách hàng hủy tour (Line Chart)
        fig_cancel_trend_ratio = create_ratio_trend_chart(tours_df, start_date, end_date, 
                                                           metric='cancellation_rate', 
                                                           title='Xu hướng Tỷ lệ Khách Hủy/Đổi (Theo ngày/tuần)')
        st.plotly_chart(fig_cancel_trend_ratio, use_container_width=True)

    st.markdown("---")


    # ========== VÙNG 2: THEO TUYẾN ==========
    st.markdown("### Vùng 2: Phân tích theo Tuyến")
    
    # Get route data
    route_table = get_route_detailed_table(filtered_tours, filtered_plans, start_date, end_date)
    top_revenue = get_top_routes(filtered_tours, n=top_n, metric='revenue')
    top_customers = get_top_routes(filtered_tours, n=top_n, metric='customers')
    top_profit = get_top_routes(filtered_tours, n=top_n, metric='profit')
    
    # Row 1: Top tuyến Tour charts
    st.markdown("#### Top Tuyến Tour")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Doanh thu (Phân bổ BU)")
        fig_rev_stacked = create_stacked_route_chart(filtered_tours, metric='revenue', title='', top_n=top_n)
        st.plotly_chart(fig_rev_stacked, use_container_width=True, key="tab2_rev_stacked")
    
    with col2:
        st.markdown("##### Lượt khách (Phân bổ BU)")
        fig_cust_stacked = create_stacked_route_chart(filtered_tours, metric='num_customers', title='', top_n=top_n)
        st.plotly_chart(fig_cust_stacked, use_container_width=True, key="tab2_cust_stacked")
    
    with col3:
        st.markdown("##### Lãi Gộp (Phân bổ BU)")
        fig_profit_stacked = create_stacked_route_chart(filtered_tours, metric='gross_profit', title='', top_n=top_n)
        st.plotly_chart(fig_profit_stacked, use_container_width=True, key="tab2_profit_stacked")
    
    st.markdown("")

    # Row 2: Profit margin with color coding
    st.markdown("#### Tỷ suất Lãi Gộp theo Tuyến")
    if not route_table.empty:
        top_10_margin = route_table.nlargest(top_n, 'profit_margin')[['route', 'profit_margin']]
        fig = create_profit_margin_chart_with_color(top_10_margin, 'profit_margin', 'route', '')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # Row 3: Detailed table
    st.markdown("#### Bảng số liệu chi tiết theo Tuyến")
    if not route_table.empty:
        display_df = route_table.copy()
        display_df = display_df[[
            'route', 'revenue', 'num_customers', 'gross_profit', 
            'profit_margin', 'revenue_completion', 'occupancy_rate', 'cancel_rate'
        ]]
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['gross_profit'] = display_df['gross_profit'].apply(format_currency)
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
        display_df['revenue_completion'] = display_df['revenue_completion'].apply(lambda x: f"{x:.1f}%")
        display_df['occupancy_rate'] = display_df['occupancy_rate'].apply(lambda x: f"{x:.1f}%")
        display_df['cancel_rate'] = display_df['cancel_rate'].apply(lambda x: f"{x:.1f}%")
        display_df.columns = ['Tuyến', 'Doanh thu', 'Lượt khách', 'Lãi Gộp', 
                      'Tỷ suất LN (%)', 'Tiến độ KH (%)', 'Tỷ lệ Lấp đầy (%)', 'Tỷ lệ Hủy/Đổi (%)']

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("")
    

    
    # ========== VÙNG 3: THEO KÊNH BÁN VÀ PHÂN KHÚC ==========
    st.markdown("### Vùng 3: Theo Kênh bán và Phân khúc")
    
    # Get channel and segment data
    channel_revenue = get_channel_breakdown(filtered_tours, start_date, end_date, metric='revenue')
    channel_customers = get_channel_breakdown(filtered_tours, start_date, end_date, metric='customers')
    segment_revenue = get_segment_breakdown(filtered_tours, start_date, end_date, metric='revenue')
    segment_customers = get_segment_breakdown(filtered_tours, start_date, end_date, metric='customers')
    cac_data = calculate_cac_by_channel(filtered_tours, start_date, end_date)
    clv_data = calculate_clv_by_segment(tours_filtered_dimensional)
    
    # Row 1: Kênh bán pie charts
    st.markdown("#### Phân bố theo Kênh bán")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Doanh thu")
        if not channel_revenue.empty:
            fig = create_pie_chart(channel_revenue, 'revenue', 'sales_channel', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("##### Lượt khách")
        if not channel_customers.empty:
            fig = create_pie_chart(channel_customers, 'num_customers', 'sales_channel', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    with col3:
        st.markdown("##### Doanh thu TB/khách")
        if not channel_revenue.empty:
            fig = go.Figure(go.Bar(
                x=channel_revenue['sales_channel'],
                y=channel_revenue['avg_revenue_per_customer'],
                text=[format_currency(v) for v in channel_revenue['avg_revenue_per_customer']],
                textposition='outside',
                marker_color='#636EFA'
            ))
            fig.update_layout(xaxis_title="Doanh thu TB/khách (₫)", yaxis_title="", height=200, showlegend=False, margin=dict(l=30, r=30, t=10, b=60))
            st.plotly_chart(fig)
    
    # Row 2: Kênh bán detailed table
    if not channel_revenue.empty:
        display_df = channel_revenue.copy()
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['avg_revenue_per_customer'] = display_df['avg_revenue_per_customer'].apply(format_currency)
        display_df.columns = ['Kênh bán', 'Doanh thu', 'Lượt khách', 'Doanh thu TB/khách']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("")
    
    # Row 3: Phân khúc pie charts
    st.markdown("#### Phân bố theo Phân khúc")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Doanh thu")
        if not segment_revenue.empty:
            fig = create_pie_chart(segment_revenue, 'value', 'segment', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("##### Lượt khách")
        if not segment_customers.empty:
            fig = create_pie_chart(segment_customers, 'value', 'segment', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    st.markdown("")
    
    # Row 4: CAC and CLV
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Chi phí Thu hút Khách hàng (CAC) theo Kênh")
        if not cac_data.empty:
            fig = go.Figure(go.Bar(
                y=cac_data['sales_channel'],
                x=cac_data['cac'],
                orientation='h',
                text=[format_currency(v) for v in cac_data['cac']],
                textposition='outside',
                marker_color='#FFA15A'
            ))
            fig.update_layout(xaxis_title="CAC (₫)", yaxis_title="", height=200, showlegend=False, margin=dict(l=100, r=100, t=10, b=30))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("#### Giá trị Trọn đời Khách hàng (CLV) theo Phân khúc")
        if not clv_data.empty:
            fig = go.Figure(go.Bar(
                y=clv_data['segment'],
                x=clv_data['clv'],
                orientation='h',
                text=[format_currency(v) for v in clv_data['clv']],
                textposition='outside',
                marker_color='#00CC96'
            ))
            fig.update_layout(xaxis_title="CLV (₫)", yaxis_title="", height=200, showlegend=False, margin=dict(l=100, r=100, t=10, b=30))
            st.plotly_chart(fig)
    
    st.markdown("---")

# ========== VÙNG 4: XU HƯỚNG VÀ NHÂN KHẨU HỌC (MỚI) ==========
    st.markdown("### Vùng 4: Xu hướng và Nhân khẩu học")

    # Hàng 1: 2 Biểu đồ Xu hướng (Revenue Trend, Cancellation Trend)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Xu hướng Doanh thu theo thời kỳ")
        # Xu hướng doanh thu theo từng thời kỳ (Line Chart)
        fig_rev_trend = create_trend_chart(filtered_tours, start_date, end_date, metrics=['revenue'])
        st.plotly_chart(fig_rev_trend, use_container_width=True)
        
    with col2:
        st.markdown("##### Xu hướng Khách hàng hủy/đổi tour")
        # Xu hướng khách hàng hủy tour (Line Chart)
        fig_cancel_trend = create_cancellation_trend_chart(tours_df, start_date, end_date)
        st.plotly_chart(fig_cancel_trend, use_container_width=True)

    # Hàng 2: 2 Biểu đồ Tỷ trọng (Age, Nationality)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Tỷ trọng Doanh thu theo Độ tuổi")
        # Tỷ trọng doanh thu khách hàng theo độ tuổi (Pie Chart)
        # Giả định cột customer_age_group tồn tại
        fig_age_pie = create_demographic_pie_chart(filtered_tours, 'customer_age_group', '')
        st.plotly_chart(fig_age_pie, use_container_width=True)

    with col2:
        st.markdown("##### Tỷ trọng Doanh thu theo Quốc tịch")
        # Tỷ trọng doanh thu khách hàng theo quốc tịch (Pie Chart)
        # Giả định cột customer_nationality tồn tại
        fig_nat_pie = create_demographic_pie_chart(filtered_tours, 'customer_nationality', '')
        st.plotly_chart(fig_nat_pie, use_container_width=True)
        
    st.markdown("---")





# ============================================================
# TAB 3: ĐỐI TÁC (TÁI CẤU TRÚC HOÀN CHỈNH)
# ============================================================
with tab3:
    st.title("🤝 Dashboard Quản lý Dịch vụ và Đối tác (Pending)")
    
    # Lấy dữ liệu đã lọc theo Đối tác/Dịch vụ
    # Giả định các hàm tính toán đã được định nghĩa trong utils.py hoặc được import
    partner_filtered_data = filter_data_by_date(partner_filtered_df, start_date, end_date)
    partner_kpis = calculate_partner_kpis(partner_filtered_data)
    partner_revenue_metrics = calculate_partner_revenue_metrics(partner_filtered_data)
    service_cancel_metrics = calculate_service_cancellation_metrics(partner_filtered_data)
    service_inventory_total = calculate_service_inventory(partner_filtered_data)['total_units'].sum()
    partner_performance = calculate_partner_performance(partner_filtered_data) 
    
    # Dữ liệu phân tích chi tiết theo loại (cho Expander Vùng 1)
    active_breakdown = calculate_partner_breakdown_by_type(partner_filtered_data, status_filter="Đang triển khai")
    expiring_breakdown = calculate_partner_breakdown_by_type(partner_filtered_data, status_filter="Sắp hết hạn")
    
    # --- VÙNG 1: TỔNG QUAN KPIs VÀ CẢNH BÁO (ĐÃ THÊM CHI TIẾT DỊCH VỤ) ---
    st.markdown("### 🎯 Vùng 1: Tổng quan Đối tác & Cảnh báo Hợp đồng")
    
    # Hàng 1: 4 KPI Cards tập trung
    col1, col2, col3, col4 = st.columns(4)
    
    # Tổng đối tác Đang triển khai
    with col1:
        st.metric(
            label="🤝 Tổng đối tác Đang triển khai",
            delta=" Tăng 2",
            value=format_number(partner_kpis['total_active_partners'])
        )
        # THÊM CHI TIẾT: Phân theo Loại Dịch vụ
        with st.expander("Chi tiết: Đang triển khai"):
            for _, row in active_breakdown.iterrows():
                st.write(f"**{row['type']}**: {format_number(row['count'])} đối tác")
        
    # Hợp đồng Sắp hết hạn (Cảnh báo)
    with col2:
        expiring_contracts = partner_kpis['contracts_status_count'][partner_kpis['contracts_status_count']['status'] == 'Sắp hết hạn']['count'].sum()
        st.metric(
            label="🚨 Hợp đồng Sắp hết hạn",
            value=format_number(expiring_contracts),
            delta="Cần gia hạn",
            delta_color="inverse"
        )
        # THÊM CHI TIẾT: Phân theo Loại Dịch vụ
        with st.expander("Chi tiết: Sắp hết hạn"):
            for _, row in expiring_breakdown.iterrows():
                st.write(f"**{row['type']}**: {format_number(row['count'])} hợp đồng")
        
    # Tổng Doanh thu dịch vụ (Revenue)
    with col3:
        st.metric(
            label="💰 Tổng Dịch vụ đang giữ",
            delta=" Tăng 2 tỷ",
            value=format_currency(partner_kpis['total_service_revenue'])
        )
        # THÊM CHI TIẾT: Phân theo Loại Dịch vụ
        # Giả định hàm calculate_partner_revenue_by_type trả về DataFrame: type, revenue
        revenue_by_type = calculate_partner_revenue_by_type(partner_filtered_data) # <--- Cần hàm này trong utils.py
        with st.expander("Chi tiết: Doanh thu theo Loại DV"):
            for _, row in revenue_by_type.iterrows():
                st.write(f"**{row['service_type']}**: {format_currency(row['revenue'])}")
        
    # Tình trạng Hủy dịch vụ (Gauge Chart)
    with col4:
        st.markdown("##### Tỷ lệ Hủy Dịch vụ")
        fig_service_cancel = create_gauge_chart(
            service_cancel_metrics['cancel_rate'],
            "Tỷ lệ Hủy Dịch vụ",
            max_value=30, 
            threshold=10, 
            is_inverse_metric=True
        )
        st.plotly_chart(fig_service_cancel, use_container_width=True)

    st.markdown("---")
    
    
    # --- VÙNG 2: PHÂN TÍCH TÌNH TRẠNG HỢP ĐỒNG & PHÂN TÍCH DỊCH VỤ (ĐÃ SỬA CHÚ THÍCH) ---
    st.markdown("### 📊 Vùng 2: Trạng thái Hợp đồng & Phân tích Dịch vụ")
    
    # Dữ liệu cho biểu đồ tròn (Tỷ trọng Trả trước/Trả sau)
    payment_status_data = partner_filtered_data.groupby('payment_status')['partner'].count().reset_index()
    payment_status_data.columns = ['status', 'count']
    
    col_status, col_price = st.columns([1, 2])
    
    # 1. Biểu đồ: Tỷ trọng Trạng thái Thanh toán (Pie Chart)
    with col_status:
        st.markdown("##### Tỷ trọng Thanh toán Hợp đồng")
        payment_data = payment_status_data[payment_status_data['status'].isin(['Trả trước', 'Trả sau'])].copy()
        total_payment_contracts = payment_data['count'].sum() # TỔNG HỢP ĐỒNG
        
        if not payment_data.empty:
            count_prepaid = payment_data[payment_data['status'] == 'Trả trước']['count'].iloc[0] if 'Trả trước' in payment_data['status'].values else 0
            count_postpaid = payment_data[payment_data['status'] == 'Trả sau']['count'].iloc[0] if 'Trả sau' in payment_data['status'].values else 0
            
            # --- HIỂN THỊ CHÚ THÍCH MỚI ---
            st.markdown(f"""
            <div style="font-size: 14px; font-weight: bold; text-align: center; margin-bottom: 5px;">
                Tổng Hợp đồng: {format_number(total_payment_contracts)}
            </div>
            <div style="font-size: 13px; text-align: center; margin-bottom: 5px;">
                <span style="color: #636EFA;">■ Trả trước:</span> {format_number(count_prepaid)} hợp đồng
                <span style="color: #FFA15A; margin-left: 15px;">■ Trả sau:</span> {format_number(count_postpaid)} hợp đồng
            </div>
            """, unsafe_allow_html=True)
            
            # --- TẠO BIỂU ĐỒ TRÒN (TẮT CHÚ THÍCH TỰ ĐỘNG) ---
            fig_payment_pie = px.pie(
                payment_data, 
                values='count', 
                names='status',
                color_discrete_sequence=['#636EFA', '#FFA15A'],
            )
            
            fig_payment_pie.update_traces(textinfo='percent+label', 
                                            hovertemplate='<b>%{label}</b><br>Số lượng: %{value:,.0f}<br>Tỉ lệ: %{percent}<extra></extra>')
            
            fig_payment_pie.update_layout(
                height=300, # Đã chỉnh height thấp hơn
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            
            st.plotly_chart(fig_payment_pie, use_container_width=True)
        else:
            st.info("Không có dữ liệu hợp đồng Trả trước/Trả sau.")
            
        # Thống kê chi tiết
        active_breakdown = calculate_partner_breakdown_by_type(partner_filtered_data, status_filter="Đang triển khai")
        with st.expander("Phân loại Đối tác Đang triển khai"):
             for _, row in active_breakdown.iterrows():
                 st.write(f"**{row['type']}**: {format_number(row['count'])} đối tác")

    # 2. Bar Chart: Giá Dịch vụ (Giá TB/Khách)
    with col_price:
        st.markdown("##### Phân tích Giá Dịch vụ (Max, Avg, Min)")
        if not partner_revenue_metrics.empty:
            df_melted = partner_revenue_metrics.melt(
                id_vars='service_type',
                value_vars=['max_price', 'avg_price', 'min_price'],
                var_name='price_type',
                value_name='price_value'
            )
            
            df_melted['price_type'] = df_melted['price_type'].replace({
                'max_price': 'Giá Cao nhất',
                'avg_price': 'Giá Trung bình',
                'min_price': 'Giá Thấp nhất'
            })
            
            fig_price_comp = px.bar(
                df_melted,
                x='price_value',
                y='service_type',
                color='price_type',
                orientation='h',
                title='Giá Dịch vụ theo Loại (Max, Avg, Min)',
                barmode='group'
            )
            fig_price_comp.update_xaxes(title="Giá (₫)")
            fig_price_comp.update_traces(hovertemplate='%{x:,.0f} ₫<extra></extra>')
            fig_price_comp.update_layout(height=350, yaxis={'categoryorder':'total ascending'}, margin=dict(t=30))
            st.plotly_chart(fig_price_comp, use_container_width=True)
        
    st.markdown("---")


    # --- VÙNG 3: XU HƯỚNG VÀ HIỆU QUẢ HỢP TÁC ---
    st.markdown("### 📈 Vùng 3: Xu hướng và Hiệu quả Hợp tác")
    
    # Row 1: Biểu đồ Doanh thu và Số lượng khách theo thời gian
    col_trend, col_scatter = st.columns(2)
    
    with col_trend:
        st.markdown("##### Xu hướng Doanh thu và Lượt khách từ Đối tác")
        fig_partner_trend = create_partner_trend_chart(partner_filtered_df, start_date, end_date)
        st.plotly_chart(fig_partner_trend, use_container_width=True)
    
    with col_scatter:
        st.markdown("##### Đánh giá Hiệu quả Từng Đối tác")
        if not partner_performance.empty:
            # Biểu đồ Bong bóng: X=Doanh thu, Y=Tỷ lệ Phản hồi, Size=Số lượng khách
            fig_scatter = px.scatter(
                partner_performance,
                x='total_revenue',
                y='avg_feedback',
                size='total_customers',
                color='partner',
                hover_name='partner',
                title='Hiệu quả Đối tác (DT vs Phản hồi Tích cực)',
                labels={'total_revenue': 'Doanh thu (₫)', 'avg_feedback': 'Tỷ lệ phản hồi tích cực (%)', 'total_customers': 'Lượt khách'}
            )
            fig_scatter.update_traces(hovertemplate='<b>%{hovertext}</b><br>Doanh thu: %{x:,.0f} ₫<br>Phản hồi: %{y:.1%}<br>Lượt khách: %{marker.size:,.0f}<extra></extra>')
            fig_scatter.update_layout(height=400, showlegend=False, margin=dict(t=30))
            st.plotly_chart(fig_scatter, use_container_width=True)

    # Bảng chi tiết Doanh thu/Chi phí/Lãi Gộp
    st.markdown("#### Bảng Chi tiết Hợp đồng và Tỷ suất Lãi Gộp")
    
    # Lấy bảng hợp đồng chi tiết
    df_partner_revenue_detail = partner_filtered_data.groupby(['partner', 'service_type', 'payment_status', 'contract_status']).agg(
        total_revenue=('revenue', 'sum'),
        total_service_cost=('service_cost', 'sum'),
        num_bookings=('booking_id', 'count')
    ).reset_index()
    
    df_partner_revenue_detail['profit_margin'] = np.where(
        df_partner_revenue_detail['total_revenue'] > 0,
        ((df_partner_revenue_detail['total_revenue'] - df_partner_revenue_detail['total_service_cost']) / df_partner_revenue_detail['total_revenue']) * 100,
        0
    )
    
    # Áp dụng formatting
    df_partner_revenue_detail['total_revenue'] = df_partner_revenue_detail['total_revenue'].apply(format_currency)
    df_partner_revenue_detail['total_service_cost'] = df_partner_revenue_detail['total_service_cost'].apply(format_currency)
    df_partner_revenue_detail['profit_margin'] = df_partner_revenue_detail['profit_margin'].apply(lambda x: f"{x:.1f}%")

    df_partner_revenue_detail.rename(columns={
        'contract_status': 'Trạng thái HĐ', 
        'service_type': 'Loại DV', 
        'payment_status': 'Tình trạng TT', 
        'total_revenue': 'Doanh thu',
        'total_service_cost': 'Chi phí DV',
        'num_bookings': 'SL HĐ',
        'profit_margin': 'Tỷ suất LN (%)'
    }, inplace=True)
    
    # Hàm highlight_expiring (Giữ nguyên)
    def highlight_expiring(s):
        if s['Trạng thái HĐ'] == 'Sắp hết hạn':
            return ['background-color: #ffe0e0; color: red'] * len(s)
        return [''] * len(s)

    st.dataframe(
        df_partner_revenue_detail[['partner', 'Loại DV', 'Doanh thu', 'Chi phí DV', 'Tỷ suất LN (%)', 'Trạng thái HĐ', 'Tình trạng TT']]
        .style.apply(highlight_expiring, axis=1), 
        use_container_width=True, hide_index=True
    )

st.markdown("---")

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p>📊 Vietravel Business Intelligence Dashboard Ver 2</p>
        <p>Cập nhật lần cuối: {}</p>
    </div>
""".format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)