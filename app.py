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
import time


# Cached loader: fetch + parse Google Sheet (or generate) once per TTL to speed up Streamlit Cloud cold starts
@st.cache_data(ttl=3600)
def load_data_cached(spreadsheet_url, plan_spreadsheet_url):
    t0 = time.time()
    result = load_or_generate_data(spreadsheet_url, plan_spreadsheet_url=plan_spreadsheet_url)
    elapsed = time.time() - t0
    # normalize return to 4-tuple (tours_df, plans_df, historical_df, meta)
    if isinstance(result, tuple) and len(result) == 4:
        tours_df, plans_df, historical_df, data_meta = result
    else:
        tours_df, plans_df, historical_df = result
        data_meta = {'used_excel': False, 'processed_files': [], 'parsed_rows': 0}
    try:
        if isinstance(data_meta, dict):
            data_meta['loader_elapsed_sec'] = elapsed
    except Exception:
        pass
    return tours_df, plans_df, historical_df, data_meta


# --- Cached wrappers for heavy aggregations (lazy and shared across reruns) ---
@st.cache_data(ttl=600)
def cached_calculate_kpis(tours_df, plans_df, start_date, end_date, plans_daily_df, plans_weekly_df, period_type, selected_segment):
    # Convert minimal inputs to allow hashing: Streamlit will hash DataFrames by content
    return calculate_kpis(tours_df, plans_df, start_date, end_date, plans_daily_df=plans_daily_df, plans_weekly_df=plans_weekly_df, period_type=period_type, selected_segment=selected_segment)


@st.cache_data(ttl=600)
def cached_get_top_routes(tours_df, n, metric):
    return get_top_routes(tours_df, n=n, metric=metric)


@st.cache_data(ttl=600)
def cached_get_route_detailed_table(tours_df, plans_df, start_date, end_date):
    return get_route_detailed_table(tours_df, plans_df, start_date, end_date)


@st.cache_data(ttl=600)
def cached_calculate_operational_metrics(tours_df):
    return calculate_operational_metrics(tours_df)


@st.cache_data(ttl=600)
def cached_calculate_booking_metrics(tours_df, start_date, end_date):
    return calculate_booking_metrics(tours_df, start_date, end_date)

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

    # CHỨC NĂNG MỚI CHO DASHBOARD
    load_route_plan_data, load_route_performance_data, load_unit_completion_data, create_completion_progress_chart,
    calculate_booking_metrics, 
    create_cancellation_trend_chart, 
    create_demographic_pie_chart,
    create_ratio_trend_chart,
    create_stacked_route_chart,
    create_top_routes_dual_axis_chart,
    create_top_routes_ratio_stacked,
    create_segment_bu_comparison_chart,
    create_route_trend_chart,
    group_small_categories,
    
    # Hàm mới cho biểu đồ tuyến theo tuần
    classify_route_type,
    get_route_weekly_performance,
    create_route_weekly_chart,
    
    # Hàm đọc dữ liệu hoàn thành kế hoạch đơn vị
    load_unit_completion_data,
    
    # Hàm đọc dữ liệu tốc độ đạt kế hoạch theo tuyến
    load_route_performance_data,
    create_route_performance_chart,
    
    # Hàm đọc dữ liệu theo dõi chỗ bán etour
    load_etour_seats_data,
    create_seats_tracking_chart
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


DEFAULT_UNIT_COMPLETION_URL = 'https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=614149511#gid=614149511' # Toan cty
DEFAULT_ROUTE_PERFORMANCE_URL = 'https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=903527778#gid=903527778' #datanet
DEFAULT_PLAN_TET_URL = 'https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=1651160424#gid=1651160424' # Kế hoạch tuyến Tết
DEFAULT_PLAN_XUAN_URL = 'https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=212301737#gid=212301737' # Kế hoạch tuyến Xuân
DEFAULT_ETOUR_SEATS_URL = 'https://docs.google.com/spreadsheets/d/1Phksbyj11bmX9XKxYvxDJUlzq2rbblGUeqVLUtWFDuc/edit?gid=2069863260#gid=2069863260' # Dữ liệu theo dõi chỗ bán etour

with st.sidebar:
    # Khởi tạo giá trị mặc định trong session_state nếu chưa có
    if 'use_sheet' not in st.session_state:
        st.session_state['use_sheet'] = True
    if 'sheet_url' not in st.session_state:
        st.session_state['sheet_url'] = None
    if 'plan_sheet_url' not in st.session_state:
        st.session_state['plan_sheet_url'] = None
    if 'unit_completion_url' not in st.session_state:
        st.session_state['unit_completion_url'] = DEFAULT_UNIT_COMPLETION_URL
    if 'route_performance_url' not in st.session_state:
        st.session_state['route_performance_url'] = DEFAULT_ROUTE_PERFORMANCE_URL
    if 'plan_tet_url' not in st.session_state:
        st.session_state['plan_tet_url'] = DEFAULT_PLAN_TET_URL
    if 'plan_xuan_url' not in st.session_state:
        st.session_state['plan_xuan_url'] = DEFAULT_PLAN_XUAN_URL
    if 'etour_seats_url' not in st.session_state:
        st.session_state['etour_seats_url'] = DEFAULT_ETOUR_SEATS_URL
    
    # Hiển thị thông tin nguồn đang dùng (rút gọn)
    if st.session_state.get('use_sheet', True) and st.session_state.get('sheet_url'):
        st.caption(f"📊 Datanet: ...{st.session_state['sheet_url'][-20:]}")
    if st.session_state.get('use_sheet', True) and st.session_state.get('plan_sheet_url'):
        st.caption(f"📋 Kế hoạch: ...{st.session_state['plan_sheet_url'][-20:]}")
    
    st.markdown("---")
    st.subheader("🔍 Bộ lọc dữ liệu")
    
    # Load dữ liệu route_performance để có options cho bộ lọc
    route_performance_url = st.session_state.get('route_performance_url', DEFAULT_ROUTE_PERFORMANCE_URL)
    cache_key_route = f'route_performance_data_{route_performance_url}'
    
    if cache_key_route not in st.session_state:
        with st.spinner('Đang tải dữ liệu...'):
            route_performance_data = load_route_performance_data(route_performance_url)
            st.session_state[cache_key_route] = route_performance_data
    else:
        route_performance_data = st.session_state[cache_key_route]
    
    # Nếu không có dữ liệu, thử load lại
    if route_performance_data.empty:
        with st.spinner('Đang tải lại dữ liệu...'):
            route_performance_data = load_route_performance_data(route_performance_url)
            st.session_state[cache_key_route] = route_performance_data
    
    # Bộ lọc Giai đoạn
    if not route_performance_data.empty and 'period' in route_performance_data.columns:
        available_periods = sorted(route_performance_data['period'].dropna().unique().tolist())
        # Đặt "KM XUÂN" làm mặc định (nếu có), nếu không thì lấy phần tử đầu tiên
        default_index = 0
        if 'KM XUÂN' in available_periods:
            default_index = available_periods.index('KM XUÂN')
        
        current_selected_period = st.session_state.get('sidebar_period_filter', available_periods[default_index] if available_periods else 'KM XUÂN')
        if current_selected_period not in available_periods:
            current_selected_period = available_periods[default_index] if available_periods else 'KM XUÂN'
        
        selected_period = st.selectbox(
            "Giai đoạn",
            options=available_periods,
            index=available_periods.index(current_selected_period) if current_selected_period in available_periods else default_index,
            key="sidebar_period_filter"
        )
    else:
        selected_period = 'KM XUÂN'
    
    # Bộ lọc Khu vực Đơn Vị
    if not route_performance_data.empty and 'region_unit' in route_performance_data.columns:
        available_regions = ['Tất cả'] + sorted(route_performance_data['region_unit'].dropna().unique().tolist())
        selected_region = st.selectbox(
            "Khu vực Đơn Vị",
            options=available_regions,
            index=0,
            key="sidebar_region_filter"
        )
    else:
        selected_region = 'Tất cả'
    
    # Bộ lọc Đơn Vị (phụ thuộc vào Khu vực Đơn Vị)
    if not route_performance_data.empty and 'unit' in route_performance_data.columns:
        if selected_region != 'Tất cả':
            filtered_units = route_performance_data[route_performance_data['region_unit'] == selected_region]['unit'].dropna().unique().tolist()
        else:
            filtered_units = route_performance_data['unit'].dropna().unique().tolist()
        
        available_units = ['Tất cả'] + sorted(filtered_units)
        
        current_selected_unit = st.session_state.get('sidebar_unit_filter', 'Tất cả')
        if current_selected_unit not in available_units:
            current_selected_unit = 'Tất cả'
        
        selected_unit = st.selectbox(
            "Đơn Vị",
            options=available_units,
            index=available_units.index(current_selected_unit) if current_selected_unit in available_units else 0,
            key="sidebar_unit_filter"
        )
    else:
        selected_unit = 'Tất cả'
    
    # Bộ lọc Tuyến Tour
    if not route_performance_data.empty and 'route' in route_performance_data.columns:
        # Lọc tuyến theo các filter đã chọn
        temp_data = route_performance_data.copy()
        if selected_region != 'Tất cả':
            temp_data = temp_data[temp_data['region_unit'] == selected_region]
        if selected_unit != 'Tất cả':
            temp_data = temp_data[temp_data['unit'] == selected_unit]
        # Filter theo Giai đoạn (không cần kiểm tra "Tất cả" vì đã bỏ option này)
        temp_data = temp_data[temp_data['period'] == selected_period]
        
        available_routes = ['Tất cả'] + sorted(temp_data['route'].dropna().unique().tolist())
        
        current_selected_route = st.session_state.get('sidebar_route_filter', 'Tất cả')
        if current_selected_route not in available_routes:
            current_selected_route = 'Tất cả'
        
        selected_route = st.selectbox(
            "Tuyến Tour",
            options=available_routes,
            index=available_routes.index(current_selected_route) if current_selected_route in available_routes else 0,
            key="sidebar_route_filter"
        )
    else:
        selected_route = 'Tất cả'
    
    # Lưu các filter vào session_state
    st.session_state['filter_period'] = selected_period
    st.session_state['filter_region'] = selected_region
    st.session_state['filter_unit'] = selected_unit
    st.session_state['filter_route'] = selected_route

# Initialize session state for data
# Load data when not already loaded or when explicitly requested (data_loaded flag False)
if not st.session_state.get('data_loaded', False):
    # Use module-level cached loader (defined above) to fetch data
    with st.spinner('Đang tải dữ liệu (tối ưu hóa cache)...'):
        spreadsheet_url = st.session_state.get('sheet_url') if st.session_state.get('use_sheet') else None
        plan_sheet_url = st.session_state.get('plan_sheet_url') if st.session_state.get('plan_sheet_url') else None
        tours_df, plans_df, historical_df, data_meta = load_data_cached(spreadsheet_url, plan_sheet_url)

        # Save loaded data into session state
        st.session_state['tours_df'] = tours_df
        st.session_state['plans_df'] = plans_df
        st.session_state['plans_daily_df'] = data_meta.get('plans_daily_df') if isinstance(data_meta, dict) else None
        st.session_state['plans_weekly_df'] = data_meta.get('plans_weekly_df') if isinstance(data_meta, dict) else None
        st.session_state['historical_df'] = historical_df
        st.session_state['data_meta'] = data_meta
        st.session_state['data_loaded'] = True

    # Show a banner including load time if available
    meta = st.session_state.get('data_meta', {})
    loader_time = meta.get('loader_elapsed_sec') if isinstance(meta, dict) else None
    # Không hiển thị thông báo load time nữa
    # Show banner if tours or plan sheets were used / parsed
    if meta.get('used_excel') or meta.get('used_sheet') or meta.get('parsed_plan_rows', 0) > 0:
        # Lưu thông tin vào session state thay vì hiển thị
        files = st.session_state['data_meta'].get('processed_files', [])
        plan_files = st.session_state['data_meta'].get('processed_plan_files', [])
        parsed = st.session_state['data_meta'].get('parsed_rows', 0)
        parsed_plan = st.session_state['data_meta'].get('parsed_plan_rows', 0)
        # Không hiển thị thông báo

# Load data from session state
tours_df = st.session_state.tours_df
plans_df = st.session_state.plans_df
historical_df = st.session_state.historical_df
# Determine whether data came from Google Sheet
data_meta = st.session_state.get('data_meta', {}) if isinstance(st.session_state.get('data_meta', {}), dict) else {}
used_sheet = bool(data_meta.get('used_sheet', False))
# Chỉ hiển thị warning nếu có URL nhưng load thất bại (không phải khi URL là None)
sheet_url_provided = st.session_state.get('sheet_url') is not None
if not used_sheet and sheet_url_provided:
    # Inform user that sheet was not available
    st.sidebar.warning("Google Sheet chưa được đọc thành công — Một số biểu đồ có thể không hiển thị dữ liệu.")

# Dashboard Title
st.title("📊 VIETRAVEL - DASHBOARD KINH DOANH TOUR")

# Filter data based on selections (dimensional filters only, NOT date)
# Date filtering will be done inside calculate_kpis to preserve YoY data
# Enforce: if the loader did NOT successfully read the Google Sheet, lock Dashboard
# to use sheet-only data by replacing tour/plan frames with empty DataFrames so that
# downstream charts/tables show no data. This prevents fallback generated data from appearing.
data_meta = st.session_state.get('data_meta', {})
used_sheet = bool(data_meta.get('used_sheet', False))

# Set default values for old filters (for backward compatibility with existing code)
date_option = "Tuần"
vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
today = datetime.now(vietnam_tz).replace(tzinfo=None)
days_since_monday = today.weekday()
start_date = today - timedelta(days=days_since_monday)
start_date = datetime(start_date.year, start_date.month, start_date.day)
end_date = start_date + timedelta(days=6)
end_date = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59)

selected_unit = "Tất cả"
selected_units_list = tours_df['business_unit'].unique().tolist() if 'business_unit' in tours_df.columns else []
selected_route = "Tất cả"
selected_routes_list = tours_df['route'].unique().tolist() if 'route' in tours_df.columns else []
selected_segment = "Tất cả"
top_n = 15
selected_partner = "Tất cả"
selected_service = "Tất cả"

# SWAP DATA SOURCE: Nếu chọn "Kỳ Báo cáo", thay thế tours_df bằng dữ liệu từ sheet Kỳ Báo Cáo
if st.session_state.get('use_kybaocao', False):
    kybaocao_df = st.session_state.get('kybaocao_df', pd.DataFrame())
    selected_month = st.session_state.get('selected_month', None)
    report_period_col = st.session_state.get('report_period_col', None)
    
    if not kybaocao_df.empty and selected_month and report_period_col:
        # Filter theo tháng trong cột V
        # Chuyển đổi cột về số để so sánh - PHẢI tạo copy trước
        kybaocao_df_copy = kybaocao_df.copy()
        kybaocao_df_copy[report_period_col] = pd.to_numeric(kybaocao_df_copy[report_period_col], errors='coerce')
        tours_df = kybaocao_df_copy[kybaocao_df_copy[report_period_col] == int(selected_month)].copy()
        
        # COLUMN NAME MAPPING: Map Kỳ Báo Cáo column names to expected names
        # Based on exact column positions from Google Sheets
        column_mapping = {}
        
        # Map by column index (Google Sheets columns: A=0, B=1, C=2, etc.)
        # Cột E (index 4): Ngày khởi hành
        if len(tours_df.columns) > 4:
            column_mapping[tours_df.columns[4]] = 'departure_date'
        
        # Cột G (index 6): lượt khách
        if len(tours_df.columns) > 6:
            column_mapping[tours_df.columns[6]] = 'num_customers'
        
        # Cột I (index 8): Doanh Thu
        if len(tours_df.columns) > 8:
            column_mapping[tours_df.columns[8]] = 'revenue'
        
        # Cột J (index 9): Lãi gộp
        if len(tours_df.columns) > 9:
            column_mapping[tours_df.columns[9]] = 'gross_profit'
        
        # Cột P (index 15): Tuyến Tour
        if len(tours_df.columns) > 15:
            column_mapping[tours_df.columns[15]] = 'route'
        
        # Cột Q (index 16): business_unit
        if len(tours_df.columns) > 16:
            column_mapping[tours_df.columns[16]] = 'business_unit'
        
        # Cột R (index 17): segment
        if len(tours_df.columns) > 17:
            column_mapping[tours_df.columns[17]] = 'segment'
        
        # Rename columns
        tours_df = tours_df.rename(columns=column_mapping)

# SWAP DATA SOURCE: Nếu chọn "Kỳ Báo cáo", thay thế tours_df bằng dữ liệu từ sheet Kỳ Báo Cáo
if st.session_state.get('use_kybaocao', False):
    kybaocao_df = st.session_state.get('kybaocao_df', pd.DataFrame())
    selected_month = st.session_state.get('selected_month', None)
    report_period_col = st.session_state.get('report_period_col', None)
    
    if not kybaocao_df.empty and selected_month and report_period_col:
        # Filter theo tháng trong cột V
        # Chuyển đổi cột về số để so sánh - PHẢI tạo copy trước
        kybaocao_df = kybaocao_df.copy()
        kybaocao_df[report_period_col] = pd.to_numeric(kybaocao_df[report_period_col], errors='coerce')
        tours_df = kybaocao_df[kybaocao_df[report_period_col] == int(selected_month)].copy()
        
        # COLUMN NAME MAPPING: Map Kỳ Báo Cáo column names to expected names
        # Based on exact column positions from Google Sheets
        column_mapping = {}
        
        # Map by column index (Google Sheets columns: A=0, B=1, C=2, etc.)
        # Cột E (index 4): Ngày khởi hành
        if len(tours_df.columns) > 4:
            column_mapping[tours_df.columns[4]] = 'departure_date'
        
        # Cột G (index 6): lượt khách
        if len(tours_df.columns) > 6:
            column_mapping[tours_df.columns[6]] = 'num_customers'
        
        # Cột I (index 8): Doanh Thu
        if len(tours_df.columns) > 8:
            column_mapping[tours_df.columns[8]] = 'revenue'
        
        # Cột J (index 9): Lãi gộp
        if len(tours_df.columns) > 9:
            column_mapping[tours_df.columns[9]] = 'gross_profit'
        
        # Cột P (index 15): Tuyến Tour
        if len(tours_df.columns) > 15:
            column_mapping[tours_df.columns[15]] = 'route'
        
        # Cột Q (index 16): business_unit
        if len(tours_df.columns) > 16:
            column_mapping[tours_df.columns[16]] = 'business_unit'
        
        # Cột R (index 17): Tổng số khách (occu) -> tour_capacity
        if len(tours_df.columns) > 17:
            column_mapping[tours_df.columns[17]] = 'tour_capacity'
        
        # Cột S (index 18): Phân khúc
        if len(tours_df.columns) > 18:
            column_mapping[tours_df.columns[18]] = 'segment'
        
        # Cột T (index 19): Kênh bán
        if len(tours_df.columns) > 19:
            column_mapping[tours_df.columns[19]] = 'sales_channel'
        
        # Cột U (index 20): Số khách hủy
        if len(tours_df.columns) > 20:
            column_mapping[tours_df.columns[20]] = 'cancel_count'
        
        # Cột V (index 21): Kỳ báo cáo - will be used as report_period
        if len(tours_df.columns) > 21:
            column_mapping[tours_df.columns[21]] = 'report_period'
        
        # Apply column mapping
        if column_mapping:
            tours_df = tours_df.rename(columns=column_mapping)
        
        # Xóa cột report_period để tránh conflict với logic hiện tại
        if 'report_period' in tours_df.columns:
            tours_df = tours_df.drop(columns=['report_period'])
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['num_customers', 'revenue', 'gross_profit', 'tour_capacity', 'cancel_count']
        for col in numeric_cols:
            if col in tours_df.columns:
                tours_df[col] = pd.to_numeric(tours_df[col], errors='coerce').fillna(0)
        
        # Convert departure_date to datetime (for trend charts only, not for aggregation)
        if 'departure_date' in tours_df.columns:
            tours_df['departure_date'] = pd.to_datetime(tours_df['departure_date'], errors='coerce')
        
        # booking_date is for aggregation - use first day of selected month
        tours_df['booking_date'] = pd.Timestamp(2025, selected_month, 1)
        
        # Add default values for missing essential columns
        if 'cancel_count' not in tours_df.columns:
            tours_df['cancel_count'] = 0
        
        # Add other commonly needed columns with defaults (for features not in Kỳ Báo Cáo)
        if 'customer_id' not in tours_df.columns:
            # Generate unique IDs for each row
            tours_df['customer_id'] = range(1, len(tours_df) + 1)
        
        if 'partner' not in tours_df.columns:
            tours_df['partner'] = 'Unknown'
        
        if 'customer_age_group' not in tours_df.columns:
            tours_df['customer_age_group'] = 'Unknown'
        
        if 'customer_nationality' not in tours_df.columns:
            tours_df['customer_nationality'] = 'Vietnam'
        
        if 'contract_status' not in tours_df.columns:
            tours_df['contract_status'] = 'Đang triển khai'
        
        if 'payment_status' not in tours_df.columns:
            tours_df['payment_status'] = 'Đã thanh toán'
        
        if 'service_type' not in tours_df.columns:
            tours_df['service_type'] = 'Tour'
        
        if 'partner_type' not in tours_df.columns:
            tours_df['partner_type'] = 'Khách sạn'
        
        if 'feedback_ratio' not in tours_df.columns:
            tours_df['feedback_ratio'] = 0.75  # Default 75% feedback
        
        used_sheet = True  # Mark as valid data source
    else:
        # Không có dữ liệu cho tháng đã chọn - không hiện warning
        pass

if used_sheet:
    tours_filtered_dimensional = tours_df.copy()
    filtered_plans = plans_df.copy()
else:
    # create empty frames with same columns where possible to avoid KeyErrors later
    try:
        tours_filtered_dimensional = pd.DataFrame(columns=tours_df.columns)
    except Exception:
        tours_filtered_dimensional = pd.DataFrame()
    try:
        filtered_plans = pd.DataFrame(columns=plans_df.columns)
    except Exception:
        filtered_plans = pd.DataFrame()

# Apply unit filter
if selected_unit != "Tất cả":
    if 'business_unit' in tours_filtered_dimensional.columns:
        # Lọc theo danh sách các đơn vị đã chọn
        tours_filtered_dimensional = tours_filtered_dimensional[tours_filtered_dimensional['business_unit'].isin(selected_units_list)]
    if 'business_unit' in filtered_plans.columns:
        filtered_plans = filtered_plans[filtered_plans['business_unit'].isin(selected_units_list)]

# Apply route filter
if selected_route != "Tất cả":
    if 'route' in tours_filtered_dimensional.columns:
        # Lọc theo danh sách các tuyến đã chọn
        tours_filtered_dimensional = tours_filtered_dimensional[tours_filtered_dimensional['route'].isin(selected_routes_list)]
    if 'route' in filtered_plans.columns:
        filtered_plans = filtered_plans[filtered_plans['route'].isin(selected_routes_list)]

if selected_segment != "Tất cả":
    if 'segment' in tours_filtered_dimensional.columns:
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
# Nếu dùng Kỳ Báo cáo, không filter theo date trong tours_df (đã filter theo tháng rồi)
# nhưng vẫn cần start_date/end_date để lấy đúng KPI plan tháng đó
use_kybaocao = st.session_state.get('use_kybaocao', False)

if use_kybaocao:
    # Khi dùng Kỳ Báo cáo, data đã được filter theo tháng trong cột V
    # Nhưng vẫn cần start_date/end_date để calculate_kpis lấy đúng plan tháng đó
    # start_date/end_date đã được set ở trên (đầu tháng -> cuối tháng)
    kpis = cached_calculate_kpis(
        tours_filtered_dimensional,
        filtered_plans,
        start_date,
        end_date,
        st.session_state.get('plans_daily_df'),
        st.session_state.get('plans_weekly_df'),
        "Tháng",  # Force period_type = "Tháng" để lấy plan tháng
        selected_segment
    )
    # Filter thêm theo departure_date để chỉ lấy tour khởi hành trong tháng được chọn
    if 'departure_date' in tours_filtered_dimensional.columns:
        tours_filtered_dimensional['departure_date'] = pd.to_datetime(tours_filtered_dimensional['departure_date'], errors='coerce')
        filtered_tours = tours_filtered_dimensional[
            (tours_filtered_dimensional['departure_date'] >= start_date) &
            (tours_filtered_dimensional['departure_date'] <= end_date)
        ].copy()
    else:
        filtered_tours = tours_filtered_dimensional.copy()
else:
    kpis = cached_calculate_kpis(
        tours_filtered_dimensional,
        filtered_plans,
        start_date,
        end_date,
        st.session_state.get('plans_daily_df'),
        st.session_state.get('plans_weekly_df'),
        date_option,
        selected_segment
    )
    # Also create a date+dimension filtered version for charts that don't need historical data
    filtered_tours = filter_data_by_date(tours_filtered_dimensional, start_date, end_date)

# TÍNH TOÁN BOOKING METRICS CHO DASHBOARD
# Use the dimensional tours frame so booking metrics respect the sheet-only lock above
if use_kybaocao:
    # Khi dùng Kỳ Báo cáo, data đã filter theo tháng, vẫn pass start_date/end_date để tính đúng
    booking_metrics = cached_calculate_booking_metrics(tours_filtered_dimensional, start_date, end_date)
else:
    booking_metrics = cached_calculate_booking_metrics(tours_filtered_dimensional, start_date, end_date)


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
# Chỉ còn 1 tab duy nhất
tab1 = st.container()

# ============================================================
# DASHBOARD THEO DÕI KINH DOANH (TẤT CẢ NỘI DUNG)
# ============================================================
with tab1:
    # Chỉ hiển thị warning và khóa dashboard nếu có URL nhưng load thất bại
    if not used_sheet and sheet_url_provided:
        st.warning("Google Sheet chưa được đọc thành công — Một số biểu đồ có thể không hiển thị dữ liệu.")
        col_retry1, col_retry2 = st.columns([1, 5])
        with col_retry1:
            if st.button("🔄 Thử lại"):
                try:
                    load_data_cached.clear()
                except Exception:
                    pass
                st.session_state['data_loaded'] = False
                st.rerun()
        with col_retry2:
            st.info("Vui lòng kiểm tra URL/Quyền truy cập của Google Sheet rồi nhấn 'Thử lại'.")
        st.markdown("---")
    # ========== VÙNG 1: TỐC ĐỘ ĐẠT KẾ HOẠCH ==========
    st.markdown("### Vùng 1: Tốc độ đạt Kế hoạch")
    
    # Lấy dữ liệu từ Google Sheet mới (Kết quả Kinh doanh)
    # Sử dụng URL từ session_state hoặc default
    unit_completion_url = st.session_state.get('unit_completion_url', DEFAULT_UNIT_COMPLETION_URL)
    
    # Cache để tránh load lại mỗi lần rerun
    cache_key = f'unit_completion_data_{unit_completion_url}'
    if cache_key not in st.session_state:
        with st.spinner('Đang tải dữ liệu mức độ hoàn thành kế hoạch đơn vị...'):
            unit_completion_data = load_unit_completion_data(unit_completion_url)
            st.session_state[cache_key] = unit_completion_data
    else:
        unit_completion_data = st.session_state[cache_key]
    
    # Nếu không có dữ liệu, thử load lại
    if unit_completion_data.empty:
        with st.spinner('Đang tải lại dữ liệu...'):
            unit_completion_data = load_unit_completion_data(unit_completion_url)
            st.session_state[cache_key] = unit_completion_data
    
    if not unit_completion_data.empty:
        # Tách dữ liệu khu vực và đơn vị
        regions_data = unit_completion_data[unit_completion_data['is_region'] == True].copy()
        units_data = unit_completion_data[unit_completion_data['is_region'] == False].copy()
        
        # Filter: Chọn khu vực hoặc tất cả đơn vị
        available_regions = ["Tất cả", "Tất cả đơn vị"] + sorted(regions_data['business_unit'].unique().tolist())
        
        # Mặc định là "Tất cả đơn vị" (index 1)
        default_region = st.session_state.get('select_region_v1', 'Tất cả đơn vị')
        if default_region not in available_regions:
            default_region = 'Tất cả đơn vị'
        default_index = available_regions.index(default_region) if default_region in available_regions else 1
        
        col_filter1, col_filter2 = st.columns([1, 3])
        with col_filter1:
            selected_region = st.selectbox(
                "Chọn Khu vực",
                options=available_regions,
                index=default_index,
                key="select_region_v1"
            )
        
        # Lọc dữ liệu theo lựa chọn
        if selected_region == "Tất cả":
            # Hiển thị tất cả khu vực
            display_data = regions_data.copy()
            chart_title = "Mức độ hoàn thành của các Khu vực"
        elif selected_region == "Tất cả đơn vị":
            # Hiển thị tất cả đơn vị từ tất cả khu vực
            display_data = units_data.copy()
            chart_title = "Mức độ hoàn thành của tất cả Đơn vị"
        else:
            # Hiển thị các đơn vị trong khu vực được chọn
            display_data = units_data[units_data['region'] == selected_region].copy()
            chart_title = f"Mức độ hoàn thành của các đơn vị - {selected_region}"
        
        if not display_data.empty:
            # Sắp xếp theo revenue_completion để hiển thị
            display_data = display_data.sort_values('revenue_completion', ascending=False).reset_index(drop=True)
            
            # Tạo biểu đồ cột nhóm: Doanh Thu và Lãi Gộp
            fig = go.Figure()
            
            # Cột Doanh Thu (DT) - màu xanh
            fig.add_trace(go.Bar(
                name='DT',
                x=display_data['business_unit'],
                y=display_data['revenue_completion'],
                text=[f"{v:.0f}%" for v in display_data['revenue_completion']],
                textposition='outside',
                marker_color='#636EFA',  # Màu xanh
                hovertemplate='<b>%{x}</b><br>DT: %{y:.1f}%<extra></extra>'
            ))
            
            # Cột Lãi Gộp (LG) - màu cam
            fig.add_trace(go.Bar(
                name='LG',
                x=display_data['business_unit'],
                y=display_data['profit_completion'],
                text=[f"{v:.0f}%" for v in display_data['profit_completion']],
                textposition='outside',
                marker_color='#FFA15A',  # Màu cam
                hovertemplate='<b>%{x}</b><br>LG: %{y:.1f}%<extra></extra>'
            ))
            
            # Thêm đường mục tiêu 100%
            fig.add_hline(
                y=100, 
                line_dash="dash", 
                line_color="gray", 
                annotation_text="Mức mục tiêu",
                annotation_position="right"
            )
            
            # Cập nhật layout
            fig.update_layout(
                title=chart_title,
                xaxis_title="",
                yaxis_title="Mức độ hoàn thành (%)",
                barmode='group',
                height=450,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=30, r=30, t=60, b=100),
                xaxis=dict(tickangle=-45, tickfont=dict(size=10))
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Hiển thị bảng chi tiết nếu chọn khu vực cụ thể hoặc "Tất cả đơn vị"
            if selected_region != "Tất cả" and not display_data.empty:
                with st.expander("📊 Xem bảng chi tiết", expanded=False):
                    # Sắp xếp display_data TRƯỚC KHI tạo detail_df: theo Khu vực, sau đó theo DT đã bán giảm dần
                    if 'region' in display_data.columns and 'revenue_actual' in display_data.columns:
                        display_data = display_data.sort_values(['region', 'revenue_actual'], ascending=[True, False]).reset_index(drop=True)
                    elif 'region' in display_data.columns:
                        display_data = display_data.sort_values('region', ascending=True).reset_index(drop=True)
                    elif 'revenue_actual' in display_data.columns:
                        display_data = display_data.sort_values('revenue_actual', ascending=False).reset_index(drop=True)
                    
                    # Tạo bảng chi tiết với đầy đủ các cột
                    detail_cols = ['business_unit']
                    
                    # Nếu là "Tất cả đơn vị", thêm cột khu vực
                    if selected_region == "Tất cả đơn vị":
                        detail_cols.append('region')
                    
                    # Thêm các cột số liệu nếu có
                    if 'revenue_plan' in display_data.columns:
                        detail_cols.extend(['revenue_plan', 'revenue_actual', 'revenue_completion'])
                    if 'profit_plan' in display_data.columns:
                        detail_cols.extend(['profit_plan', 'profit_actual', 'profit_completion'])
                    
                    # Lọc các cột có sẵn
                    available_cols = [col for col in detail_cols if col in display_data.columns]
                    detail_df = display_data[available_cols].copy()
                    
                    # Đặt tên cột tiếng Việt
                    col_mapping = {
                        'business_unit': 'Đơn vị',
                        'region': 'Khu vực',
                        'revenue_plan': 'DT Kế hoạch (tr.đ)',
                        'revenue_actual': 'DT đã bán (tr.đ)',
                        'revenue_completion': 'Tỷ lệ đạt DT (%)',
                        'profit_plan': 'LG Kế hoạch (tr.đ)',
                        'profit_actual': 'LG đã bán (tr.đ)',
                        'profit_completion': 'Tỷ lệ đạt LG (%)'
                    }
                    
                    detail_df = detail_df.rename(columns=col_mapping)
                    
                    # Sắp xếp thứ tự cột: Khu vực (nếu có), Đơn vị, DT Kế hoạch, DT đã bán, Tỷ lệ đạt DT, LG Kế hoạch, LG đã bán, Tỷ lệ đạt LG
                    desired_order = ['Khu vực', 'Đơn vị', 'DT Kế hoạch (tr.đ)', 'DT đã bán (tr.đ)', 'Tỷ lệ đạt DT (%)', 'LG Kế hoạch (tr.đ)', 'LG đã bán (tr.đ)', 'Tỷ lệ đạt LG (%)']
                    available_order = [col for col in desired_order if col in detail_df.columns]
                    detail_df = detail_df[available_order]
                    
                    # Format các cột số
                    if 'DT Kế hoạch (tr.đ)' in detail_df.columns:
                        detail_df['DT Kế hoạch (tr.đ)'] = detail_df['DT Kế hoạch (tr.đ)'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
                    if 'DT đã bán (tr.đ)' in detail_df.columns:
                        detail_df['DT đã bán (tr.đ)'] = detail_df['DT đã bán (tr.đ)'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
                    if 'Tỷ lệ đạt DT (%)' in detail_df.columns:
                        detail_df['Tỷ lệ đạt DT (%)'] = detail_df['Tỷ lệ đạt DT (%)'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%")
                    if 'LG Kế hoạch (tr.đ)' in detail_df.columns:
                        detail_df['LG Kế hoạch (tr.đ)'] = detail_df['LG Kế hoạch (tr.đ)'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
                    if 'LG đã bán (tr.đ)' in detail_df.columns:
                        detail_df['LG đã bán (tr.đ)'] = detail_df['LG đã bán (tr.đ)'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
                    if 'Tỷ lệ đạt LG (%)' in detail_df.columns:
                        detail_df['Tỷ lệ đạt LG (%)'] = detail_df['Tỷ lệ đạt LG (%)'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%")
                    
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"Không có dữ liệu cho khu vực '{selected_region}'")
        
        # Nút refresh dữ liệu
        col_refresh1, col_refresh2 = st.columns([1, 5])
        with col_refresh1:
            if st.button("🔄 Làm mới dữ liệu", key="refresh_unit_completion"):
                unit_completion_data = load_unit_completion_data(unit_completion_url)
                st.session_state[cache_key] = unit_completion_data
                st.rerun()
    else:
        st.warning("Không thể tải dữ liệu từ Google Sheet. Vui lòng kiểm tra URL và quyền truy cập.")
        if st.button("🔄 Thử lại", key="retry_unit_completion"):
            unit_completion_data = load_unit_completion_data(unit_completion_url)
            st.session_state[cache_key] = unit_completion_data
            st.rerun()


    # ============================================================
    # PHẦN NỘI DUNG DASHBOARD THEO DÕI SẢN PHẨM - PHẦN 1
    # ============================================================
    # Chỉ hiển thị warning nếu có URL nhưng load thất bại
    if not used_sheet and sheet_url_provided:
        st.warning("Google Sheet chưa được đọc thành công — Một số biểu đồ có thể không hiển thị dữ liệu.")
        col_retry1, col_retry2 = st.columns([1, 5])
        with col_retry1:
            if st.button("🔄 Thử lại", key="retry_sheet_1"):
                try:
                    load_data_cached.clear()
                except Exception:
                    pass
                st.session_state['data_loaded'] = False
                st.rerun()
        with col_retry2:
            st.info("Vui lòng kiểm tra URL/Quyền truy cập của Google Sheet rồi nhấn 'Thử lại'.")
        st.markdown("---")
    

    
    # ========== BIỂU ĐỒ TỐC ĐỘ ĐẠT KẾ HOẠCH THEO TUYẾN ==========
    st.markdown("### Tốc độ đạt Kế hoạch theo Tuyến")
    
    # Lấy dữ liệu từ Google Sheet mới
    route_performance_url = st.session_state.get('route_performance_url', DEFAULT_ROUTE_PERFORMANCE_URL)
    
    # Cache để tránh load lại mỗi lần rerun
    cache_key_route = f'route_performance_data_{route_performance_url}'
    if cache_key_route not in st.session_state:
        with st.spinner('Đang tải dữ liệu tốc độ đạt kế hoạch theo tuyến...'):
            route_performance_data = load_route_performance_data(route_performance_url)
            st.session_state[cache_key_route] = route_performance_data
    else:
        route_performance_data = st.session_state[cache_key_route]
    
    # Nếu không có dữ liệu, thử load lại
    if route_performance_data.empty:
        with st.spinner('Đang tải lại dữ liệu...'):
            route_performance_data = load_route_performance_data(route_performance_url)
            st.session_state[cache_key_route] = route_performance_data
    
    if not route_performance_data.empty:
        # Lấy các filter từ sidebar
        selected_period = st.session_state.get('filter_period', 'KM XUÂN')
        selected_region = st.session_state.get('filter_region', 'Tất cả')
        selected_unit = st.session_state.get('filter_unit', 'Tất cả')
        selected_route = st.session_state.get('filter_route', 'Tất cả')
        
        # Áp dụng các filter từ sidebar
        filtered_data = route_performance_data.copy()
        # Filter theo Giai đoạn (không cần kiểm tra "Tất cả" vì đã bỏ option này)
        filtered_data = filtered_data[filtered_data['period'] == selected_period].copy()
        if selected_region != 'Tất cả':
            filtered_data = filtered_data[filtered_data['region_unit'] == selected_region].copy()
        if selected_unit != 'Tất cả':
            filtered_data = filtered_data[filtered_data['unit'] == selected_unit].copy()
        if selected_route != 'Tất cả':
            filtered_data = filtered_data[filtered_data['route'] == selected_route].copy()
        
        route_performance_data = filtered_data
        
        # HÀNG 1: NỘI ĐỊA (3 biểu đồ)
        st.markdown("#### Nội địa")
        
        # Filter dữ liệu Nội địa
        domestic_data = route_performance_data[route_performance_data['route_type'] == 'Nội địa'].copy()
        
        # 3 biểu đồ Nội địa
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not domestic_data.empty:
                fig_dom_cust = create_route_performance_chart(
                    domestic_data, 
                    metric='num_customers', 
                    title='Lượt Khách'
                )
                st.plotly_chart(fig_dom_cust, use_container_width=True, key="route_dom_cust")
            else:
                st.info("Không có dữ liệu")
        
        with col2:
            if not domestic_data.empty:
                fig_dom_rev = create_route_performance_chart(
                    domestic_data, 
                    metric='revenue', 
                    title='Doanh Thu'
                )
                st.plotly_chart(fig_dom_rev, use_container_width=True, key="route_dom_rev")
            else:
                st.info("Không có dữ liệu")
        
        with col3:
            if not domestic_data.empty:
                fig_dom_profit = create_route_performance_chart(
                    domestic_data, 
                    metric='gross_profit', 
                    title='Lãi Gộp'
                )
                st.plotly_chart(fig_dom_profit, use_container_width=True, key="route_dom_profit")
            else:
                st.info("Không có dữ liệu")
        
        # 3 card hiển thị % hoàn thành kế hoạch cho Nội địa
        if not domestic_data.empty:
            # Tính tổng actual và plan (nếu có plan data)
            total_customers_actual = domestic_data['num_customers'].sum()
            total_revenue_actual = domestic_data['revenue'].sum()
            total_profit_actual = domestic_data['gross_profit'].sum()
            
            # Lấy plan data nếu có
            plan_tet_url = st.session_state.get('plan_tet_url', '')
            plan_xuan_url = st.session_state.get('plan_xuan_url', '')
            
            total_customers_plan = 0
            total_revenue_plan = 0
            total_profit_plan = 0
            
            if plan_tet_url or plan_xuan_url:
                cache_key_plan_tet = f'plan_tet_data_{plan_tet_url}'
                cache_key_plan_xuan = f'plan_xuan_data_{plan_xuan_url}'
                
                if cache_key_plan_tet in st.session_state:
                    plan_tet_data = st.session_state[cache_key_plan_tet]
                    if not plan_tet_data.empty:
                        domestic_plan_tet = plan_tet_data[plan_tet_data['route_type'] == 'Nội địa']
                        total_customers_plan += domestic_plan_tet['plan_customers'].sum()
                        total_revenue_plan += domestic_plan_tet['plan_revenue'].sum()
                        total_profit_plan += domestic_plan_tet['plan_profit'].sum()
                
                if cache_key_plan_xuan in st.session_state:
                    plan_xuan_data = st.session_state[cache_key_plan_xuan]
                    if not plan_xuan_data.empty:
                        domestic_plan_xuan = plan_xuan_data[plan_xuan_data['route_type'] == 'Nội địa']
                        total_customers_plan += domestic_plan_xuan['plan_customers'].sum()
                        total_revenue_plan += domestic_plan_xuan['plan_revenue'].sum()
                        total_profit_plan += domestic_plan_xuan['plan_profit'].sum()
            
            # Tính % hoàn thành
            completion_customers = (total_customers_actual / total_customers_plan * 100) if total_customers_plan > 0 else 0
            completion_revenue = (total_revenue_actual / total_revenue_plan * 100) if total_revenue_plan > 0 else 0
            completion_profit = (total_profit_actual / total_profit_plan * 100) if total_profit_plan > 0 else 0
            
            # Hiển thị 3 card
            col_card1, col_card2, col_card3 = st.columns(3)
            
            with col_card1:
                st.metric(
                    label="Lượt Khách",
                    value=f"{completion_customers:.1f}%",
                    delta=None
                )
            
            with col_card2:
                st.metric(
                    label="Doanh Thu",
                    value=f"{completion_revenue:.1f}%",
                    delta=None
                )
            
            with col_card3:
                st.metric(
                    label="Lãi Gộp",
                    value=f"{completion_profit:.1f}%",
                    delta=None
                )
        
        st.markdown("---")
        
        # HÀNG 2: OUTBOUND (3 biểu đồ)
        st.markdown("#### Outbound")
        
        # Filter dữ liệu Outbound
        outbound_data = route_performance_data[route_performance_data['route_type'] == 'Outbound'].copy()
        
        # 3 biểu đồ Outbound
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not outbound_data.empty:
                fig_out_cust = create_route_performance_chart(
                    outbound_data, 
                    metric='num_customers', 
                    title='Lượt Khách'
                )
                st.plotly_chart(fig_out_cust, use_container_width=True, key="route_out_cust")
            else:
                st.info("Không có dữ liệu")
        
        with col2:
            if not outbound_data.empty:
                fig_out_rev = create_route_performance_chart(
                    outbound_data, 
                    metric='revenue', 
                    title='Doanh Thu'
                )
                st.plotly_chart(fig_out_rev, use_container_width=True, key="route_out_rev")
            else:
                st.info("Không có dữ liệu")
        
        with col3:
            if not outbound_data.empty:
                fig_out_profit = create_route_performance_chart(
                    outbound_data, 
                    metric='gross_profit', 
                    title='Lãi Gộp'
                )
                st.plotly_chart(fig_out_profit, use_container_width=True, key="route_out_profit")
            else:
                st.info("Không có dữ liệu")
        
        # 3 card hiển thị % hoàn thành kế hoạch cho Outbound
        if not outbound_data.empty:
            # Tính tổng actual và plan (nếu có plan data)
            total_customers_actual = outbound_data['num_customers'].sum()
            total_revenue_actual = outbound_data['revenue'].sum()
            total_profit_actual = outbound_data['gross_profit'].sum()
            
            # Lấy plan data nếu có
            plan_tet_url = st.session_state.get('plan_tet_url', '')
            plan_xuan_url = st.session_state.get('plan_xuan_url', '')
            
            total_customers_plan = 0
            total_revenue_plan = 0
            total_profit_plan = 0
            
            if plan_tet_url or plan_xuan_url:
                cache_key_plan_tet = f'plan_tet_data_{plan_tet_url}'
                cache_key_plan_xuan = f'plan_xuan_data_{plan_xuan_url}'
                
                if cache_key_plan_tet in st.session_state:
                    plan_tet_data = st.session_state[cache_key_plan_tet]
                    if not plan_tet_data.empty:
                        outbound_plan_tet = plan_tet_data[plan_tet_data['route_type'] == 'Outbound']
                        total_customers_plan += outbound_plan_tet['plan_customers'].sum()
                        total_revenue_plan += outbound_plan_tet['plan_revenue'].sum()
                        total_profit_plan += outbound_plan_tet['plan_profit'].sum()
                
                if cache_key_plan_xuan in st.session_state:
                    plan_xuan_data = st.session_state[cache_key_plan_xuan]
                    if not plan_xuan_data.empty:
                        outbound_plan_xuan = plan_xuan_data[plan_xuan_data['route_type'] == 'Outbound']
                        total_customers_plan += outbound_plan_xuan['plan_customers'].sum()
                        total_revenue_plan += outbound_plan_xuan['plan_revenue'].sum()
                        total_profit_plan += outbound_plan_xuan['plan_profit'].sum()
            
            # Tính % hoàn thành
            completion_customers = (total_customers_actual / total_customers_plan * 100) if total_customers_plan > 0 else 0
            completion_revenue = (total_revenue_actual / total_revenue_plan * 100) if total_revenue_plan > 0 else 0
            completion_profit = (total_profit_actual / total_profit_plan * 100) if total_profit_plan > 0 else 0
            
            # Hiển thị 3 card
            col_card1, col_card2, col_card3 = st.columns(3)
            
            with col_card1:
                st.metric(
                    label="Lượt Khách",
                    value=f"{completion_customers:.1f}%",
                    delta=None
                )
            
            with col_card2:
                st.metric(
                    label="Doanh Thu",
                    value=f"{completion_revenue:.1f}%",
                    delta=None
                )
            
            with col_card3:
                st.metric(
                    label="Lãi Gộp",
                    value=f"{completion_profit:.1f}%",
                    delta=None
                )
        
        # Nút refresh dữ liệu
        col_refresh1, col_refresh2 = st.columns([1, 5])
        with col_refresh1:
            if st.button("🔄 Làm mới dữ liệu", key="refresh_route_performance"):
                route_performance_data = load_route_performance_data(route_performance_url)
                st.session_state[cache_key_route] = route_performance_data
                st.rerun()
    else:
        st.warning("Không thể tải dữ liệu từ Google Sheet. Vui lòng kiểm tra URL và quyền truy cập.")
        if st.button("🔄 Thử lại", key="retry_route_performance"):
            route_performance_data = load_route_performance_data(route_performance_url)
            st.session_state[cache_key_route] = route_performance_data
            st.rerun()
    
    st.markdown("---")
    
    # ========== BẢNG TIẾN ĐỘ HOÀN THÀNH KẾ HOẠCH ==========
    st.markdown("### TIẾN ĐỘ HOÀN THÀNH KẾ HOẠCH")
    
    # Load dữ liệu kế hoạch
    plan_tet_url = st.session_state.get('plan_tet_url', DEFAULT_PLAN_TET_URL)
    plan_xuan_url = st.session_state.get('plan_xuan_url', DEFAULT_PLAN_XUAN_URL)
    
    cache_key_plan_tet = f'plan_tet_data_{plan_tet_url}'
    cache_key_plan_xuan = f'plan_xuan_data_{plan_xuan_url}'
    
    if cache_key_plan_tet not in st.session_state:
        with st.spinner('Đang tải kế hoạch Tết...'):
            plan_tet_data = load_route_plan_data(plan_tet_url, period_name='TẾT')
            st.session_state[cache_key_plan_tet] = plan_tet_data
    else:
        plan_tet_data = st.session_state[cache_key_plan_tet]
    
    if cache_key_plan_xuan not in st.session_state:
        with st.spinner('Đang tải kế hoạch Xuân...'):
            plan_xuan_data = load_route_plan_data(plan_xuan_url, period_name='KM XUÂN')
            st.session_state[cache_key_plan_xuan] = plan_xuan_data
    else:
        plan_xuan_data = st.session_state[cache_key_plan_xuan]
    
    # Gộp kế hoạch Tết và Xuân
    if not plan_tet_data.empty and not plan_xuan_data.empty:
        all_plan_data = pd.concat([plan_tet_data, plan_xuan_data], ignore_index=True)
    elif not plan_tet_data.empty:
        all_plan_data = plan_tet_data.copy()
    elif not plan_xuan_data.empty:
        all_plan_data = plan_xuan_data.copy()
    else:
        all_plan_data = pd.DataFrame()
    
    if not all_plan_data.empty and not route_performance_data.empty:
        # Merge kế hoạch với thực tế theo route và period
        # Chuẩn hóa tên route để merge
        all_plan_data['route_normalized'] = all_plan_data['route'].astype(str).str.strip().str.upper()
        route_performance_data['route_normalized'] = route_performance_data['route'].astype(str).str.strip().str.upper()
        
        # Merge
        merged_data = route_performance_data.merge(
            all_plan_data[['route_normalized', 'route_type', 'period', 'plan_customers', 'plan_revenue', 'plan_profit']],
            on=['route_normalized', 'route_type', 'period'],
            how='left',
            suffixes=('_actual', '_plan')
        )
        
        # Tính phần trăm hoàn thành
        merged_data['completion_customers'] = (merged_data['num_customers'] / merged_data['plan_customers'].replace(0, np.nan)) * 100
        merged_data['completion_revenue'] = (merged_data['revenue'] / merged_data['plan_revenue'].replace(0, np.nan)) * 100
        merged_data['completion_profit'] = (merged_data['gross_profit'] / merged_data['plan_profit'].replace(0, np.nan)) * 100
        
        # Thay thế inf và nan bằng 0
        merged_data['completion_customers'] = merged_data['completion_customers'].replace([np.inf, -np.inf, np.nan], 0)
        merged_data['completion_revenue'] = merged_data['completion_revenue'].replace([np.inf, -np.inf, np.nan], 0)
        merged_data['completion_profit'] = merged_data['completion_profit'].replace([np.inf, -np.inf, np.nan], 0)
        
        # Chia thành Nội địa và Outbound
        domestic_completion = merged_data[merged_data['route_type'] == 'Nội địa'].copy()
        outbound_completion = merged_data[merged_data['route_type'] == 'Outbound'].copy()
        
        # Hiển thị biểu đồ và bảng Nội địa
        if not domestic_completion.empty:
            st.markdown("#### Nội địa")
            
            # Tạo biểu đồ line chart
            # Nhóm theo route để tổng hợp (nếu có nhiều period cho cùng route)
            domestic_chart_data = domestic_completion.groupby('route').agg({
                'completion_customers': 'mean',
                'completion_revenue': 'mean',
                'completion_profit': 'mean',
                'revenue': 'sum',
                'plan_revenue': 'sum'
            }).reset_index()
            
            fig_domestic = create_completion_progress_chart(
                domestic_chart_data,
                title='TIẾN ĐỘ HOÀN THÀNH KẾ HOẠCH - NỘI ĐỊA'
            )
            st.plotly_chart(fig_domestic, use_container_width=True, key="completion_domestic_chart")
        
        st.markdown("---")
        
        # Hiển thị biểu đồ và bảng Outbound
        if not outbound_completion.empty:
            st.markdown("#### Outbound")
            
            # Tạo biểu đồ line chart
            # Nhóm theo route để tổng hợp (nếu có nhiều period cho cùng route)
            outbound_chart_data = outbound_completion.groupby('route').agg({
                'completion_customers': 'mean',
                'completion_revenue': 'mean',
                'completion_profit': 'mean',
                'revenue': 'sum',
                'plan_revenue': 'sum'
            }).reset_index()
            
            fig_outbound = create_completion_progress_chart(
                outbound_chart_data,
                title='TIẾN ĐỘ HOÀN THÀNH KẾ HOẠCH - OUTBOUND'
            )
            st.plotly_chart(fig_outbound, use_container_width=True, key="completion_outbound_chart")
        
        # Nút refresh
        col_refresh1, col_refresh2 = st.columns([1, 5])
        with col_refresh1:
            if st.button("🔄 Làm mới dữ liệu Kế hoạch", key="refresh_plan_data"):
                plan_tet_data = load_route_plan_data(plan_tet_url, period_name='TẾT')
                plan_xuan_data = load_route_plan_data(plan_xuan_url, period_name='KM XUÂN')
                st.session_state[cache_key_plan_tet] = plan_tet_data
                st.session_state[cache_key_plan_xuan] = plan_xuan_data
                st.rerun()
    else:
        if all_plan_data.empty:
            st.warning("Không thể tải dữ liệu kế hoạch từ Google Sheet. Vui lòng kiểm tra URL và quyền truy cập.")
        if route_performance_data.empty:
            st.warning("Không có dữ liệu thực tế để so sánh.")
    
    st.markdown("---")
    
    # ========== BIỂU ĐỒ THEO DÕI CHỖ BÁN (ETOUR) ==========
    st.markdown("### THEO DÕI SỐ CHỖ BÁN CỦA CÁC TUYẾN TRONG GIAI ĐOẠN - ETOUR")
    
    # Load dữ liệu etour
    etour_seats_url = st.session_state.get('etour_seats_url', DEFAULT_ETOUR_SEATS_URL)
    cache_key_etour = f'etour_seats_data_{etour_seats_url}'
    
    if cache_key_etour not in st.session_state:
        etour_seats_data = load_etour_seats_data(etour_seats_url)
        st.session_state[cache_key_etour] = etour_seats_data
    else:
        etour_seats_data = st.session_state[cache_key_etour]
    
    if not etour_seats_data.empty:
        # Filter dữ liệu Nội địa
        domestic_seats_data = etour_seats_data[etour_seats_data['route_type'] == 'Nội địa'].copy()
        
        # Filter dữ liệu Outbound
        outbound_seats_data = etour_seats_data[etour_seats_data['route_type'] == 'Outbound'].copy()
        
        # Hiển thị biểu đồ Nội địa
        if not domestic_seats_data.empty:
            st.markdown("#### Nội địa")
            fig_domestic_seats = create_seats_tracking_chart(
                domestic_seats_data,
                title='Theo dõi số chỗ bán của các tuyến trong giai đoạn - etour (Nội địa)'
            )
            st.plotly_chart(fig_domestic_seats, use_container_width=True, key="seats_domestic_chart")
        
        st.markdown("---")
        
        # Hiển thị biểu đồ Outbound
        if not outbound_seats_data.empty:
            st.markdown("#### Outbound")
            fig_outbound_seats = create_seats_tracking_chart(
                outbound_seats_data,
                title='Theo dõi số chỗ bán của các tuyến trong giai đoạn - etour (Outbound)'
            )
            st.plotly_chart(fig_outbound_seats, use_container_width=True, key="seats_outbound_chart")
        
        # Nút refresh dữ liệu
        col_refresh1, col_refresh2 = st.columns([1, 5])
        with col_refresh1:
            if st.button("🔄 Làm mới dữ liệu ETOUR", key="refresh_etour_seats"):
                etour_seats_data = load_etour_seats_data(etour_seats_url)
                st.session_state[cache_key_etour] = etour_seats_data
                st.rerun()
    else:
        st.warning("Không thể tải dữ liệu từ Google Sheet ETOUR. Vui lòng kiểm tra URL và quyền truy cập.")
        if st.button("🔄 Thử lại", key="retry_etour_seats"):
            etour_seats_data = load_etour_seats_data(etour_seats_url)
            st.session_state[cache_key_etour] = etour_seats_data
            st.rerun()
    
    st.markdown("---")





# ============================================================

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p>📊 Vietravel Business Intelligence Dashboard Ver 2</p>
        <p>Cập nhật lần cuối: {}</p>
    </div>
""".format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)