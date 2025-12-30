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
def cached_calculate_operational_metrics(tours_df):
    return calculate_operational_metrics(tours_df)


# Import custom modules
from data_generator import load_or_generate_data
from utils import (
    # Các hàm Format và Core Logic
    format_currency, format_number, format_percentage,
    calculate_completion_rate, get_growth_rate, filter_data_by_date, filter_confirmed_bookings,
    
    # Các hàm KPI và Chart
    calculate_kpis, 
    
    # Các hàm Top/Breakdown
    get_top_routes,
    
    # Các hàm Operational
    calculate_operational_metrics,
    
    # CHỨC NĂNG MỚI CHO DASHBOARD
    load_route_plan_data, 
    load_route_performance_data, 
    load_unit_completion_data, 
    create_completion_progress_chart,
    
    # Hàm phân loại tuyến
    classify_route_type,
    
    # Hàm tạo biểu đồ tốc độ đạt kế hoạch theo tuyến
    create_route_performance_chart,
    
    # Hàm đọc dữ liệu theo dõi chỗ bán etour
    load_etour_seats_data,
    create_seats_tracking_chart,
    
    # Hàm đọc dữ liệu cho phần Tiến độ hoàn thành kế hoạch
    load_completion_progress_actual_data,
    load_completion_progress_plan_data,
    
    # Hàm đọc tổng kế hoạch từ sheet
    load_total_plan_data
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
    h2 {
        font-size: 1.2rem !important;
    }
    h3 {
        font-size: 1.1rem !important;
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
        # Lọc bỏ các option có "LK" trong tên (Mien Bac LK, Mien Trung LK, etc.)
        all_regions = route_performance_data['region_unit'].dropna().unique().tolist()
        filtered_regions = [r for r in all_regions if 'LK' not in str(r).upper()]
        available_regions = ['Tất cả'] + sorted(filtered_regions)
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

# Dashboard Title - căn giữa
st.markdown("""
    <div style="text-align: center; margin-bottom: 5px; margin-top: 10px;">
        <h1 style="font-size: 1.6rem; margin-bottom: 2px; line-height: 1.2;">📊 VIETRAVEL - DASHBOARD THEO DÕI KINH DOANH</h1>
""", unsafe_allow_html=True)

# Hiển thị thông tin giai đoạn như header
selected_period = st.session_state.get('filter_period', 'KM XUÂN')
period_normalized = str(selected_period).strip().upper()
if period_normalized in ['KM XUÂN', 'KM XUAN', 'XUÂN', 'XUAN']:
    period_info = "Kế hoạch giai đoạn Xuân (1/12 - 28/02/2026)"
elif period_normalized in ['TẾT', 'TET']:
    period_info = "Kế hoạch giai đoạn Tết (14/02 - 22/02/2026)"
else:
    period_info = ""

if period_info:
    st.markdown(f"""
        <h2 style='color: #1f77b4; margin-top: 0; text-align: center; font-size: 1.2rem; line-height: 1.2;'>{period_info}</h2>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("</div>", unsafe_allow_html=True)

# Filter data based on selections (dimensional filters only, NOT date)
# Date filtering will be done inside calculate_kpis to preserve YoY data
# Enforce: if the loader did NOT successfully read the Google Sheet, lock Dashboard
# to use sheet-only data by replacing tour/plan frames with empty DataFrames so that
# downstream charts/tables show no data. This prevents fallback generated data from appearing.
data_meta = st.session_state.get('data_meta', {})

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

# Đảm bảo start_date, end_date và date_option luôn được định nghĩa
# Nếu chưa được định nghĩa từ sidebar, sử dụng giá trị mặc định (tháng hiện tại)
try:
    _ = start_date
    _ = end_date
    _ = date_option
except NameError:
    # Nếu start_date, end_date hoặc date_option chưa được định nghĩa, sử dụng giá trị mặc định
    vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    today = datetime.now(vietnam_tz).replace(tzinfo=None)
    from calendar import monthrange
    date_option = "Tháng"  # Giá trị mặc định
    start_date = datetime(today.year, today.month, 1)
    last_day = monthrange(today.year, today.month)[1]
    end_date = datetime(today.year, today.month, last_day, 23, 59, 59)

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
    
    # ========== LOAD DỮ LIỆU KẾ HOẠCH (Dùng chung cho ETOUR và TIẾN ĐỘ HOÀN THÀNH KẾ HOẠCH) ==========
    # Load dữ liệu kế hoạch
    plan_tet_url = st.session_state.get('plan_tet_url', DEFAULT_PLAN_TET_URL)
    plan_xuan_url = st.session_state.get('plan_xuan_url', DEFAULT_PLAN_XUAN_URL)
    
    # Lấy region_filter từ session_state
    selected_region = st.session_state.get('filter_region', 'Tất cả')
    region_filter = selected_region if selected_region != 'Tất cả' else None
    
    # Cache key bao gồm region_filter
    cache_key_plan_tet = f'plan_tet_data_{plan_tet_url}_{region_filter}'
    cache_key_plan_xuan = f'plan_xuan_data_{plan_xuan_url}_{region_filter}'
    
    # Kiểm tra xem region_filter có thay đổi không
    last_region_filter = st.session_state.get('last_region_filter', None)
    if last_region_filter != region_filter:
        # Xóa cache cũ nếu region_filter thay đổi
        old_cache_key_tet = f'plan_tet_data_{plan_tet_url}_{last_region_filter}'
        old_cache_key_xuan = f'plan_xuan_data_{plan_xuan_url}_{last_region_filter}'
        if old_cache_key_tet in st.session_state:
            del st.session_state[old_cache_key_tet]
        if old_cache_key_xuan in st.session_state:
            del st.session_state[old_cache_key_xuan]
        st.session_state['last_region_filter'] = region_filter
    
    if cache_key_plan_tet not in st.session_state:
        with st.spinner('Đang tải kế hoạch Tết...'):
            plan_tet_data = load_route_plan_data(plan_tet_url, period_name='TẾT', region_filter=region_filter)
            st.session_state[cache_key_plan_tet] = plan_tet_data
    else:
        plan_tet_data = st.session_state[cache_key_plan_tet]
    
    if cache_key_plan_xuan not in st.session_state:
        with st.spinner('Đang tải kế hoạch Xuân...'):
            plan_xuan_data = load_route_plan_data(plan_xuan_url, period_name='KM XUÂN', region_filter=region_filter)
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
    
    # Load tổng kế hoạch từ sheet (Dom Total và Out Total) theo giai đoạn đã chọn
    # Lấy giai đoạn từ filter
    selected_period = st.session_state.get('filter_period', 'KM XUÂN')
    
    # Chỉ load và dùng số kế hoạch từ sheet tương ứng với giai đoạn đã chọn
    total_plan_dom_lk = 0
    total_plan_dom_dt = 0
    total_plan_out_lk = 0
    total_plan_out_dt = 0
    
    # Chuẩn hóa tên giai đoạn
    period_normalized = str(selected_period).strip().upper()
    is_tet = period_normalized in ['TẾT', 'TET']
    is_xuan = period_normalized in ['KM XUÂN', 'KM XUAN', 'XUÂN', 'XUAN']
    
    if is_tet:
        # Chỉ lấy từ sheet Tết
        cache_key_total_plan_tet = f'total_plan_tet_{plan_tet_url}'
        if cache_key_total_plan_tet not in st.session_state:
            with st.spinner('Đang tải tổng kế hoạch Tết...'):
                total_plan_tet = load_total_plan_data(plan_tet_url, period_name='TẾT')
                st.session_state[cache_key_total_plan_tet] = total_plan_tet
        else:
            total_plan_tet = st.session_state[cache_key_total_plan_tet]
        
        total_plan_dom_lk = total_plan_tet.get('dom_lk', 0)
        total_plan_dom_dt = total_plan_tet.get('dom_dt', 0)  # Đơn vị: tr.d
        total_plan_out_lk = total_plan_tet.get('out_lk', 0)
        total_plan_out_dt = total_plan_tet.get('out_dt', 0)  # Đơn vị: tr.d
    elif is_xuan:
        # Chỉ lấy từ sheet Xuân
        cache_key_total_plan_xuan = f'total_plan_xuan_{plan_xuan_url}'
        if cache_key_total_plan_xuan not in st.session_state:
            with st.spinner('Đang tải tổng kế hoạch Xuân...'):
                total_plan_xuan = load_total_plan_data(plan_xuan_url, period_name='KM XUÂN')
                st.session_state[cache_key_total_plan_xuan] = total_plan_xuan
        else:
            total_plan_xuan = st.session_state[cache_key_total_plan_xuan]
        
        total_plan_dom_lk = total_plan_xuan.get('dom_lk', 0)
        total_plan_dom_dt = total_plan_xuan.get('dom_dt', 0)  # Đơn vị: tr.d
        total_plan_out_lk = total_plan_xuan.get('out_lk', 0)
        total_plan_out_dt = total_plan_xuan.get('out_dt', 0)  # Đơn vị: tr.d
    else:
        # Mặc định là Xuân nếu không xác định được
        cache_key_total_plan_xuan = f'total_plan_xuan_{plan_xuan_url}'
        if cache_key_total_plan_xuan not in st.session_state:
            with st.spinner('Đang tải tổng kế hoạch Xuân...'):
                total_plan_xuan = load_total_plan_data(plan_xuan_url, period_name='KM XUÂN')
                st.session_state[cache_key_total_plan_xuan] = total_plan_xuan
        else:
            total_plan_xuan = st.session_state[cache_key_total_plan_xuan]
        
        total_plan_dom_lk = total_plan_xuan.get('dom_lk', 0)
        total_plan_dom_dt = total_plan_xuan.get('dom_dt', 0)  # Đơn vị: tr.d
        total_plan_out_lk = total_plan_xuan.get('out_lk', 0)
        total_plan_out_dt = total_plan_xuan.get('out_dt', 0)  # Đơn vị: tr.d
    
    # ========== BIỂU ĐỒ THEO DÕI CHỖ BÁN (ETOUR) ==========
    st.markdown("### THEO DÕI SỐ CHỖ BÁN CỦA CÁC TUYẾN TRONG GIAI ĐOẠN - ETOUR")
    
    # Load dữ liệu etour
    etour_seats_url = st.session_state.get('etour_seats_url', DEFAULT_ETOUR_SEATS_URL)
    cache_key_etour = f'etour_seats_data_{etour_seats_url}'
    
    # Lấy region_filter để kiểm tra xem có thay đổi không
    last_region_filter_etour = st.session_state.get('last_region_filter_etour', None)
    
    # Nếu region filter thay đổi, clear cache để reload dữ liệu
    if last_region_filter_etour != selected_region:
        if cache_key_etour in st.session_state:
            del st.session_state[cache_key_etour]
        st.session_state['last_region_filter_etour'] = selected_region
    
    if cache_key_etour not in st.session_state:
        etour_seats_data = load_etour_seats_data(etour_seats_url)
        st.session_state[cache_key_etour] = etour_seats_data
    else:
        etour_seats_data = st.session_state[cache_key_etour]
    
    if not etour_seats_data.empty:
        # Merge số kế hoạch từ all_plan_data (đã filter theo region) vào etour_seats_data
        # để đảm bảo số kế hoạch đúng theo filter
        if not all_plan_data.empty:
            # Lấy period từ filter để chỉ lấy số kế hoạch đúng period
            selected_period = st.session_state.get('filter_period', 'KM XUÂN')
            
            # Lưu plan_revenue và plan_seats gốc từ etour
            etour_seats_data['plan_revenue_etour'] = etour_seats_data['plan_revenue'].copy()
            etour_seats_data['plan_seats_etour'] = etour_seats_data['plan_seats'].copy()
            
            # QUAN TRỌNG: Chỉ merge khi route match CHÍNH XÁC
            # Sử dụng route_group (Tuyến tour) để merge với all_plan_data, vì all_plan_data có route là Tuyến tour
            # Nếu không có route_group, dùng route
            merge_col = 'route_group' if 'route_group' in etour_seats_data.columns and not etour_seats_data['route_group'].isna().all() else 'route'
            
            # Chỉ normalize Unicode để xử lý encoding, KHÔNG normalize tên route
            import unicodedata
            def normalize_unicode(text):
                """Normalize Unicode để so sánh chính xác"""
                if pd.isna(text):
                    return ''
                text_str = str(text).strip()
                return unicodedata.normalize('NFC', text_str)
            
            # Normalize route names (chỉ Unicode, không thay đổi tên)
            etour_seats_data['route_normalized'] = etour_seats_data[merge_col].apply(normalize_unicode)
            
            all_plan_data_for_merge = all_plan_data.copy()
            all_plan_data_for_merge['route_normalized'] = all_plan_data_for_merge['route'].apply(normalize_unicode)
            
            # Filter theo period nếu có
            if 'period' in all_plan_data_for_merge.columns:
                all_plan_data_for_merge = all_plan_data_for_merge[all_plan_data_for_merge['period'] == selected_period].copy()
            
            # Tạo lookup table từ all_plan_data
            # QUAN TRỌNG: Nếu có nhiều route normalize thành cùng giá trị, ưu tiên route dài hơn (chính xác hơn)
            plan_lookup = all_plan_data_for_merge.copy()
            plan_lookup['route_len'] = plan_lookup['route'].astype(str).str.len()
            plan_lookup = plan_lookup.sort_values('route_len', ascending=False)
            plan_lookup = plan_lookup.groupby('route_normalized').agg({
                'plan_customers': 'first',
                'plan_revenue': 'first'
            }).reset_index()
            plan_lookup = plan_lookup.rename(columns={
                'plan_revenue': 'plan_revenue_plan',
                'plan_customers': 'plan_customers_plan'
            })
            
            # Merge plan_customers và plan_revenue từ all_plan_data
            # CHỈ merge khi route_normalized match CHÍNH XÁC
            etour_seats_data = etour_seats_data.merge(
                plan_lookup[['route_normalized', 'plan_customers_plan', 'plan_revenue_plan']],
                on='route_normalized',
                how='left'
            )
            
            # Thay thế plan_revenue và plan_seats từ file kế hoạch CHỈ KHI MATCH
            # Nếu không match, để 0 (không fallback về etour)
            if 'plan_customers_plan' in etour_seats_data.columns:
                # Chỉ dùng số từ file kế hoạch nếu match, không match thì để 0
                etour_seats_data['plan_seats'] = etour_seats_data['plan_customers_plan'].fillna(0)
            if 'plan_revenue_plan' in etour_seats_data.columns:
                # Chỉ dùng số từ file kế hoạch nếu match, không match thì để 0
                etour_seats_data['plan_revenue'] = etour_seats_data['plan_revenue_plan'].fillna(0)
                
                # Debug: Kiểm tra các route không match được
                unmatched_routes = etour_seats_data[etour_seats_data['plan_revenue_plan'].isna() & (etour_seats_data['route_type'] == 'Outbound')]
                if not unmatched_routes.empty and len(unmatched_routes) <= 20:  # Chỉ log nếu không quá nhiều
                    # Có thể log ra để debug nhưng không hiển thị cho user
                    pass
            
            # Xóa cột tạm (GIỮ LẠI plan_revenue_etour và plan_seats_etour để dùng cho tính tổng % đạt kế hoạch)
            etour_seats_data = etour_seats_data.drop(columns=[
                'route_normalized', 
                'plan_revenue_plan', 'plan_customers_plan'
            ], errors='ignore')
        else:
            # Nếu không có all_plan_data, vẫn cần tạo plan_revenue_etour và plan_seats_etour từ giá trị gốc
            if 'plan_revenue_etour' not in etour_seats_data.columns:
                etour_seats_data['plan_revenue_etour'] = etour_seats_data['plan_revenue'].copy()
            if 'plan_seats_etour' not in etour_seats_data.columns:
                etour_seats_data['plan_seats_etour'] = etour_seats_data['plan_seats'].copy()
        
        # Lấy region_filter từ session_state để filter dữ liệu (đã lấy ở trên)
        # selected_region đã được lấy ở trên
        
        # Chuẩn bị matching_regions để dùng sau
        matching_regions = []
        if selected_region != 'Tất cả':
            selected_region_normalized = str(selected_region).strip().upper()
            # Map tên region - bao gồm cả các biến thể có thể có trong CSV
            region_mapping = {
                'MIEN BAC': ['MIEN BAC', 'MIỀN BẮC', 'MIEN BAC', 'Mien Bac', 'MIENBAC'],
                'MIEN TRUNG': ['MIEN TRUNG', 'MIỀN TRUNG', 'Mien Trung', 'MIENTRUNG'],
                'MIEN NAM': ['MIEN NAM', 'MIỀN NAM', 'Mien Nam', 'MIENNAM']
            }
            # Tìm các giá trị region tương ứng
            for key, values in region_mapping.items():
                if selected_region_normalized in key or any(selected_region_normalized in v.upper() for v in values):
                    matching_regions.extend(values)
                    matching_regions.append(key)
            if not matching_regions:
                matching_regions = [selected_region_normalized]
            
            # Chuẩn hóa tất cả thành uppercase để so sánh
            matching_regions = list(set([r.upper() for r in matching_regions]))
        
        # Filter theo region nếu có
        filtered_etour_data = etour_seats_data.copy()
        if selected_region != 'Tất cả' and 'region_unit' in filtered_etour_data.columns and matching_regions:
            # Chuẩn hóa tên region để so sánh
            filtered_etour_data['region_unit_normalized'] = filtered_etour_data['region_unit'].astype(str).str.strip().str.upper()
            
            # Filter theo region - CHỈ giữ các dòng có region_unit khớp
            before_filter_count = len(filtered_etour_data)
            filtered_etour_data = filtered_etour_data[
                filtered_etour_data['region_unit_normalized'].isin(matching_regions)
            ].copy()
            after_filter_count = len(filtered_etour_data)
            
            # Debug: Kiểm tra xem có dòng nào từ region khác không
            if not filtered_etour_data.empty:
                # Kiểm tra lại để chắc chắn
                wrong_regions = filtered_etour_data[
                    ~filtered_etour_data['region_unit_normalized'].isin(matching_regions)
                ]
                if not wrong_regions.empty:
                    # Loại bỏ các dòng sai
                    filtered_etour_data = filtered_etour_data[
                        filtered_etour_data['region_unit_normalized'].isin(matching_regions)
                    ].copy()
            
            filtered_etour_data = filtered_etour_data.drop(columns=['region_unit_normalized'])
        
        # Filter theo period (Giai đoạn) nếu có
        selected_period = st.session_state.get('filter_period', 'KM XUÂN')
        if selected_period != 'Tất cả' and 'period' in filtered_etour_data.columns:
            # Chuẩn hóa tên period để so sánh
            period_normalized = str(selected_period).strip().upper()
            # Map các giá trị period có thể có - CHỈ lấy các giá trị tương ứng với period đã chọn
            period_mapping = {
                'KM XUÂN': ['KM XUÂN', 'KM XUAN'],
                'KM XUAN': ['KM XUÂN', 'KM XUAN'],
                'TẾT': ['TẾT', 'TET'],
                'TET': ['TẾT', 'TET']
            }
            matching_periods = []
            # Tìm period mapping tương ứng
            for key, values in period_mapping.items():
                if period_normalized == key.upper() or period_normalized in [v.upper() for v in values]:
                    matching_periods.extend(values)
                    matching_periods.append(key)
            if not matching_periods:
                matching_periods = [period_normalized]
            matching_periods = list(set([p.upper() for p in matching_periods]))
            
            # Filter theo period - CHỈ lấy các dòng có period khớp
            filtered_etour_data = filtered_etour_data[
                filtered_etour_data['period'].astype(str).str.strip().str.upper().isin(matching_periods)
            ].copy()
        
        # Filter dữ liệu Nội địa
        domestic_seats_data = filtered_etour_data[filtered_etour_data['route_type'] == 'Nội địa'].copy()
        
        # Filter dữ liệu Outbound
        outbound_seats_data = filtered_etour_data[filtered_etour_data['route_type'] == 'Outbound'].copy()
        
        # Debug: Kiểm tra số dòng sau khi filter
        if selected_region != 'Tất cả':
            # Đảm bảo chỉ sum các dòng có region_unit đúng
            if not domestic_seats_data.empty and 'region_unit' in domestic_seats_data.columns:
                # Kiểm tra lại filter
                domestic_seats_data = domestic_seats_data[
                    domestic_seats_data['region_unit'].astype(str).str.strip().str.upper().isin(matching_regions)
                ].copy()
            if not outbound_seats_data.empty and 'region_unit' in outbound_seats_data.columns:
                outbound_seats_data = outbound_seats_data[
                    outbound_seats_data['region_unit'].astype(str).str.strip().str.upper().isin(matching_regions)
                ].copy()
        
        # Hiển thị biểu đồ Nội địa
        if not domestic_seats_data.empty:
            st.markdown("#### Nội địa")
            # QUAN TRỌNG: Tính lại remaining_seats theo công thức: số kế hoạch - đã thực hiện
            # Đảm bảo công thức nhất quán với bảng chi tiết
            domestic_seats_data = domestic_seats_data.copy()
            domestic_seats_data['remaining_seats'] = (domestic_seats_data['plan_seats'] - domestic_seats_data['actual_seats']).clip(lower=0)
            
            # Tính tổng % đạt kế hoạch
            # QUAN TRỌNG: Dùng số kế hoạch TỔNG từ sheet "Kế hoạch xuân" và "Kế hoạch tết" (Dom Total)
            # Chuyển đổi từ tr.d sang VNĐ: total_plan_dom_dt * 1_000_000
            total_plan_revenue = total_plan_dom_dt * 1_000_000  # Đơn vị: VNĐ
            total_plan_seats = total_plan_dom_lk  # Đơn vị: LK
            
            total_actual_revenue = domestic_seats_data['actual_revenue'].sum()  # Đơn vị: VNĐ
            total_actual_seats = domestic_seats_data['actual_seats'].sum()  # Đơn vị: LK
            
            completion_revenue_pct = (total_actual_revenue / total_plan_revenue * 100) if total_plan_revenue > 0 else 0
            completion_seats_pct = (total_actual_seats / total_plan_seats * 100) if total_plan_seats > 0 else 0
            
            # Hiển thị 2 card metrics với thông tin chi tiết
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                # Format số với dấu phẩy
                total_plan_revenue_tr = total_plan_revenue / 1_000_000
                total_actual_revenue_tr = total_actual_revenue / 1_000_000
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
                                padding: 16px 20px; 
                                border-radius: 8px; 
                                border: none;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                margin-right: 5px;">
                        <div style="text-align: center; font-size: 12px; color: #6c757d; margin-bottom: 12px; font-weight: 600; letter-spacing: 0.5px;">
                            Kế hoạch Doanh Thu Đạt
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="font-size: 12px; color: #495057; line-height: 1.6;">
                                <div style="margin-bottom: 4px;">KH: <span style="font-weight: 600;">{total_plan_revenue_tr:,.0f} tr.d</span></div>
                                <div>TH: <span style="font-weight: 600;">{total_actual_revenue_tr:,.0f} tr.d</span></div>
                            </div>
                            <div style="font-size: 32px; font-weight: 700; color: #1f77b4; line-height: 1;">
                                {completion_revenue_pct:.1f}%
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
                                padding: 12px 16px; 
                                border-radius: 8px; 
                                border: none;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                margin-left: 5px;">
                        <div style="text-align: center; font-size: 11px; color: #6c757d; margin-bottom: 10px; font-weight: 600; letter-spacing: 0.5px;">
                            Kế hoạch LK đạt
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="font-size: 11px; color: #495057; line-height: 1.6;">
                                <div style="margin-bottom: 4px;">KH: <span style="font-weight: 600;">{total_plan_seats:,.0f} LK</span></div>
                                <div>TH: <span style="font-weight: 600;">{total_actual_seats:,.0f} LK</span></div>
                            </div>
                            <div style="font-size: 24px; font-weight: 700; color: #1f77b4; line-height: 1;">
                                {completion_seats_pct:.1f}%
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            fig_domestic_seats = create_seats_tracking_chart(
                domestic_seats_data,
                title='Theo dõi số chỗ bán của các tuyến trong giai đoạn - etour (Nội địa)'
            )
            st.plotly_chart(fig_domestic_seats, use_container_width=True, key="seats_domestic_chart")
            
            # Bảng chi tiết Nội địa - ETOUR
            with st.expander("📊 Xem bảng chi tiết", expanded=False):
                # Tính toán các chỉ số
                # Đảm bảo chỉ sum các dòng đã được filter theo region_unit
                # Groupby theo route_group (Tuyến tour) để sum các dòng theo tuyến tour
                # Nếu không có route_group, dùng route
                groupby_col = 'route_group' if 'route_group' in domestic_seats_data.columns and not domestic_seats_data['route_group'].isna().all() else 'route'
                
                # Đảm bảo chỉ sum các dòng có region_unit đúng (nếu đã filter)
                # domestic_seats_data đã được filter ở trên, nhưng filter lại để chắc chắn
                if selected_region != 'Tất cả' and 'region_unit' in domestic_seats_data.columns and matching_regions:
                    # Debug: Kiểm tra các giá trị region_unit có trong dữ liệu
                    unique_regions = domestic_seats_data['region_unit'].astype(str).str.strip().str.upper().unique()
                    
                    # Filter lại để chắc chắn chỉ có các dòng từ region đã chọn
                    # CHỈ sum các dòng có region_unit khớp với matching_regions
                    # QUAN TRỌNG: Phải filter TRƯỚC khi groupby để tránh sum các dòng từ các region khác
                    domestic_seats_data_filtered = domestic_seats_data[
                        domestic_seats_data['region_unit'].astype(str).str.strip().str.upper().isin(matching_regions)
                    ].copy()
                    
                    # QUAN TRỌNG: Filter thêm theo period để đảm bảo chỉ lấy dữ liệu từ period đã chọn
                    selected_period = st.session_state.get('filter_period', 'KM XUÂN')
                    if selected_period != 'Tất cả' and 'period' in domestic_seats_data_filtered.columns:
                        period_normalized = str(selected_period).strip().upper()
                        period_mapping = {
                            'KM XUÂN': ['KM XUÂN', 'KM XUAN'],
                            'KM XUAN': ['KM XUÂN', 'KM XUAN'],
                            'TẾT': ['TẾT', 'TET'],
                            'TET': ['TẾT', 'TET']
                        }
                        matching_periods = []
                        for key, values in period_mapping.items():
                            if period_normalized == key.upper() or period_normalized in [v.upper() for v in values]:
                                matching_periods.extend(values)
                                matching_periods.append(key)
                        if not matching_periods:
                            matching_periods = [period_normalized]
                        matching_periods = list(set([p.upper() for p in matching_periods]))
                        
                        domestic_seats_data_filtered = domestic_seats_data_filtered[
                            domestic_seats_data_filtered['period'].astype(str).str.strip().str.upper().isin(matching_periods)
                        ].copy()
                    
                    # Debug: Kiểm tra xem có bao nhiêu dòng sau khi filter
                    if not domestic_seats_data_filtered.empty:
                        # Kiểm tra xem có dòng nào có route_group = "Miền Bắc" không
                        if 'route_group' in domestic_seats_data_filtered.columns:
                            mien_bac_rows = domestic_seats_data_filtered[
                                domestic_seats_data_filtered['route_group'].astype(str).str.strip().str.upper() == 'MIỀN BẮC'
                            ]
                    
                    # Debug: Kiểm tra xem có dòng nào từ region khác không
                    if not domestic_seats_data_filtered.empty:
                        # Đảm bảo tất cả các dòng đều có region_unit đúng
                        wrong_region_rows = domestic_seats_data_filtered[
                            ~domestic_seats_data_filtered['region_unit'].astype(str).str.strip().str.upper().isin(matching_regions)
                        ]
                        if not wrong_region_rows.empty:
                            # Nếu có dòng sai, loại bỏ
                            domestic_seats_data_filtered = domestic_seats_data_filtered[
                                domestic_seats_data_filtered['region_unit'].astype(str).str.strip().str.upper().isin(matching_regions)
                            ].copy()
                else:
                    domestic_seats_data_filtered = domestic_seats_data.copy()
                
                # Với plan_revenue và plan_seats: dùng 'first' vì đã merge từ all_plan_data (mỗi route_group chỉ có 1 giá trị kế hoạch)
                # Với actual: dùng 'sum' để sum các dòng theo tuyến tour (chỉ các dòng đã filter)
                # QUAN TRỌNG: Đã filter theo region và period rồi, nên CHỈ cần groupby theo route_group
                # KHÔNG groupby theo region_unit và period nữa vì đã filter rồi
                # Lưu ý: plan_revenue_etour cần sum vì có thể có nhiều dòng cùng route_group với giá trị khác nhau
                agg_dict = {
                    'plan_revenue': 'first',  # Lấy giá trị đầu tiên (không sum) - từ file kế hoạch
                    'actual_revenue': 'sum',  # Sum các dòng theo tuyến tour (chỉ trong region và period đã filter)
                    'plan_seats': 'first',  # Lấy giá trị đầu tiên (không sum)
                    'actual_seats': 'sum',  # Sum các dòng theo tuyến tour (chỉ trong region và period đã filter)
                }
                if 'plan_revenue_etour' in domestic_seats_data_filtered.columns:
                    agg_dict['plan_revenue_etour'] = 'sum'  # Sum các dòng từ ETOUR (cột G)
                
                domestic_seats_detail = domestic_seats_data_filtered.groupby(groupby_col).agg(agg_dict).reset_index()
                
                # Đổi tên cột groupby về 'route' để dùng chung
                if groupby_col == 'route_group':
                    domestic_seats_detail = domestic_seats_detail.rename(columns={'route_group': 'route'})
                
                # Chuyển đổi đơn vị sang triệu đồng
                domestic_seats_detail['plan_revenue_tr'] = domestic_seats_detail['plan_revenue'] / 1_000_000
                domestic_seats_detail['actual_revenue_tr'] = domestic_seats_detail['actual_revenue'] / 1_000_000
                # Chuyển đổi plan_revenue_etour sang triệu đồng (Doanh số dự kiến từ cột G của ETOUR)
                if 'plan_revenue_etour' in domestic_seats_detail.columns:
                    domestic_seats_detail['plan_revenue_etour_tr'] = domestic_seats_detail['plan_revenue_etour'] / 1_000_000
                else:
                    domestic_seats_detail['plan_revenue_etour_tr'] = 0
                
                # Tính các chỉ số
                domestic_seats_detail['completion_revenue_pct'] = (domestic_seats_detail['actual_revenue'] / domestic_seats_detail['plan_revenue'].replace(0, np.nan) * 100).fillna(0)
                domestic_seats_detail['completion_seats_pct'] = (domestic_seats_detail['actual_seats'] / domestic_seats_detail['plan_seats'].replace(0, np.nan) * 100).fillna(0)
                
                # DT còn= DS Dự kiến - DT đã bán (nếu > 0)
                domestic_seats_detail['additional_revenue_tr'] = (domestic_seats_detail['plan_revenue_tr'] - domestic_seats_detail['actual_revenue_tr']).clip(lower=0)
                
                # Số chỗ có thể khai thác thêm = SL Dự kiến - LK đã bán
                domestic_seats_detail['additional_seats'] = (domestic_seats_detail['plan_seats'] - domestic_seats_detail['actual_seats']).clip(lower=0)
                
                # Tạo bảng chi tiết với format số có dấu phẩy
                detail_table = pd.DataFrame({
                    'STT': range(1, len(domestic_seats_detail) + 1),
                    'Tuyến tour': domestic_seats_detail['route'],
                    'Doanh thu kế hoạch (Tr.đ)': domestic_seats_detail['plan_revenue_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Doanh số dự kiến (Tr.đ)': domestic_seats_detail['plan_revenue_etour_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Doanh số đã bán (Tr.đ)': domestic_seats_detail['actual_revenue_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Tốc độ đạt kế hoạch DT (%)': domestic_seats_detail['completion_revenue_pct'].round(1).astype(str) + '%',
                    'Doanh số còn (Tr.đ)': domestic_seats_detail['additional_revenue_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'LK Kế hoạch': domestic_seats_detail['plan_seats'].fillna(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Lượt khách dự kiến': domestic_seats_detail['plan_seats'].fillna(0).astype(int).apply(lambda x: f"{x:,}"),
                    'LK đã thực hiện': domestic_seats_detail['actual_seats'].fillna(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Tốc độ đạt kế hoạch LK (%)': domestic_seats_detail['completion_seats_pct'].round(1).astype(str) + '%',
                    'LK kế hoạch còn': domestic_seats_detail['additional_seats'].fillna(0).astype(int).apply(lambda x: f"{x:,}")
                })
                
                # Sắp xếp theo DT đã bán giảm dần (dùng giá trị số thực tế, không phải string đã format)
                detail_table['_sort_revenue'] = domestic_seats_detail['actual_revenue_tr'].fillna(0)
                detail_table = detail_table.sort_values('_sort_revenue', ascending=False).reset_index(drop=True)
                detail_table = detail_table.drop(columns=['_sort_revenue'])
                detail_table['STT'] = range(1, len(detail_table) + 1)
                
                st.dataframe(detail_table, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Hiển thị biểu đồ Outbound
        if not outbound_seats_data.empty:
            st.markdown("#### Outbound")
            # QUAN TRỌNG: Tính lại remaining_seats theo công thức: số kế hoạch - đã thực hiện
            # Đảm bảo công thức nhất quán với bảng chi tiết
            outbound_seats_data = outbound_seats_data.copy()
            outbound_seats_data['remaining_seats'] = (outbound_seats_data['plan_seats'] - outbound_seats_data['actual_seats']).clip(lower=0)
            
            # Tính tổng % đạt kế hoạch
            # QUAN TRỌNG: Dùng số kế hoạch TỔNG từ sheet "Kế hoạch xuân" và "Kế hoạch tết" (Out Total)
            # Chuyển đổi từ tr.d sang VNĐ: total_plan_out_dt * 1_000_000
            total_plan_revenue = total_plan_out_dt * 1_000_000  # Đơn vị: VNĐ
            total_plan_seats = total_plan_out_lk  # Đơn vị: LK
            
            total_actual_revenue = outbound_seats_data['actual_revenue'].sum()  # Đơn vị: VNĐ
            total_actual_seats = outbound_seats_data['actual_seats'].sum()  # Đơn vị: LK
            
            completion_revenue_pct = (total_actual_revenue / total_plan_revenue * 100) if total_plan_revenue > 0 else 0
            completion_seats_pct = (total_actual_seats / total_plan_seats * 100) if total_plan_seats > 0 else 0
            
            # Hiển thị 2 card metrics với thông tin chi tiết
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                # Format số với dấu phẩy
                total_plan_revenue_tr = total_plan_revenue / 1_000_000
                total_actual_revenue_tr = total_actual_revenue / 1_000_000
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
                                padding: 12px 16px; 
                                border-radius: 8px; 
                                border: none;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                margin-right: 5px;">
                        <div style="text-align: center; font-size: 11px; color: #6c757d; margin-bottom: 10px; font-weight: 600; letter-spacing: 0.5px;">
                            Kế hoạch Doanh Thu Đạt
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="font-size: 11px; color: #495057; line-height: 1.6;">
                                <div style="margin-bottom: 4px;">KH: <span style="font-weight: 600;">{total_plan_revenue_tr:,.0f} tr.d</span></div>
                                <div>TH: <span style="font-weight: 600;">{total_actual_revenue_tr:,.0f} tr.d</span></div>
                            </div>
                            <div style="font-size: 24px; font-weight: 700; color: #1f77b4; line-height: 1;">
                                {completion_revenue_pct:.1f}%
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
                                padding: 12px 16px; 
                                border-radius: 8px; 
                                border: none;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                margin-left: 5px;">
                        <div style="text-align: center; font-size: 11px; color: #6c757d; margin-bottom: 10px; font-weight: 600; letter-spacing: 0.5px;">
                            Kế hoạch LK đạt
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="font-size: 11px; color: #495057; line-height: 1.6;">
                                <div style="margin-bottom: 4px;">KH: <span style="font-weight: 600;">{total_plan_seats:,.0f} LK</span></div>
                                <div>TH: <span style="font-weight: 600;">{total_actual_seats:,.0f} LK</span></div>
                            </div>
                            <div style="font-size: 24px; font-weight: 700; color: #1f77b4; line-height: 1;">
                                {completion_seats_pct:.1f}%
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            fig_outbound_seats = create_seats_tracking_chart(
                outbound_seats_data,
                title='Theo dõi số chỗ bán của các tuyến trong giai đoạn - etour (Outbound)'
            )
            st.plotly_chart(fig_outbound_seats, use_container_width=True, key="seats_outbound_chart")
            
            # Bảng chi tiết Outbound - ETOUR
            with st.expander("📊 Xem bảng chi tiết", expanded=False):
                # Tính toán các chỉ số
                # Đảm bảo chỉ sum các dòng đã được filter theo region_unit
                # Groupby theo route_group (Tuyến tour) để sum các dòng theo tuyến tour
                # Nếu không có route_group, dùng route
                groupby_col = 'route_group' if 'route_group' in outbound_seats_data.columns and not outbound_seats_data['route_group'].isna().all() else 'route'
                
                # Đảm bảo chỉ sum các dòng có region_unit đúng (nếu đã filter)
                if selected_region != 'Tất cả' and 'region_unit' in outbound_seats_data.columns:
                    # Filter lại để chắc chắn
                    outbound_seats_data_filtered = outbound_seats_data[
                        outbound_seats_data['region_unit'].astype(str).str.strip().str.upper().isin(matching_regions)
                    ].copy()
                else:
                    outbound_seats_data_filtered = outbound_seats_data.copy()
                
                # QUAN TRỌNG: Filter thêm theo period để đảm bảo chỉ lấy dữ liệu từ period đã chọn
                selected_period = st.session_state.get('filter_period', 'KM XUÂN')
                if selected_period != 'Tất cả' and 'period' in outbound_seats_data_filtered.columns:
                    period_normalized = str(selected_period).strip().upper()
                    period_mapping = {
                        'KM XUÂN': ['KM XUÂN', 'KM XUAN'],
                        'KM XUAN': ['KM XUÂN', 'KM XUAN'],
                        'TẾT': ['TẾT', 'TET'],
                        'TET': ['TẾT', 'TET']
                    }
                    matching_periods = []
                    for key, values in period_mapping.items():
                        if period_normalized == key.upper() or period_normalized in [v.upper() for v in values]:
                            matching_periods.extend(values)
                            matching_periods.append(key)
                    if not matching_periods:
                        matching_periods = [period_normalized]
                    matching_periods = list(set([p.upper() for p in matching_periods]))
                    
                    outbound_seats_data_filtered = outbound_seats_data_filtered[
                        outbound_seats_data_filtered['period'].astype(str).str.strip().str.upper().isin(matching_periods)
                    ].copy()
                
                # Với plan_revenue và plan_seats: dùng 'first' vì đã merge từ all_plan_data (mỗi route_group chỉ có 1 giá trị kế hoạch)
                # Với actual: dùng 'sum' để sum các dòng theo tuyến tour (chỉ các dòng đã filter)
                # QUAN TRỌNG: Đã filter theo region và period rồi, nên CHỈ cần groupby theo route_group
                # KHÔNG groupby theo region_unit và period nữa vì đã filter rồi
                # Lưu ý: plan_revenue_etour cần sum vì có thể có nhiều dòng cùng route_group với giá trị khác nhau
                agg_dict = {
                    'plan_revenue': 'first',  # Lấy giá trị đầu tiên (không sum) - từ file kế hoạch
                    'actual_revenue': 'sum',  # Sum các dòng theo tuyến tour (chỉ trong region và period đã filter)
                    'plan_seats': 'first',  # Lấy giá trị đầu tiên (không sum)
                    'actual_seats': 'sum',  # Sum các dòng theo tuyến tour (chỉ trong region và period đã filter)
                }
                if 'plan_revenue_etour' in outbound_seats_data_filtered.columns:
                    agg_dict['plan_revenue_etour'] = 'sum'  # Sum các dòng từ ETOUR (cột G)
                
                outbound_seats_detail = outbound_seats_data_filtered.groupby(groupby_col).agg(agg_dict).reset_index()
                
                # Đổi tên cột groupby về 'route' để dùng chung
                if groupby_col == 'route_group':
                    outbound_seats_detail = outbound_seats_detail.rename(columns={'route_group': 'route'})
                
                # Chuyển đổi đơn vị sang triệu đồng
                outbound_seats_detail['plan_revenue_tr'] = outbound_seats_detail['plan_revenue'] / 1_000_000
                outbound_seats_detail['actual_revenue_tr'] = outbound_seats_detail['actual_revenue'] / 1_000_000
                # Chuyển đổi plan_revenue_etour sang triệu đồng (Doanh số dự kiến từ cột G của ETOUR)
                if 'plan_revenue_etour' in outbound_seats_detail.columns:
                    outbound_seats_detail['plan_revenue_etour_tr'] = outbound_seats_detail['plan_revenue_etour'] / 1_000_000
                else:
                    outbound_seats_detail['plan_revenue_etour_tr'] = 0
                
                # Tính các chỉ số
                outbound_seats_detail['completion_revenue_pct'] = (outbound_seats_detail['actual_revenue'] / outbound_seats_detail['plan_revenue'].replace(0, np.nan) * 100).fillna(0)
                outbound_seats_detail['completion_seats_pct'] = (outbound_seats_detail['actual_seats'] / outbound_seats_detail['plan_seats'].replace(0, np.nan) * 100).fillna(0)
                
                # Doanh số KH còn = DS Dự kiến - DT đã bán (nếu > 0)
                outbound_seats_detail['additional_revenue_tr'] = (outbound_seats_detail['plan_revenue_tr'] - outbound_seats_detail['actual_revenue_tr']).clip(lower=0)
                
                # Số chỗ kế hoạch còn= SL Dự kiến - LK đã bán
                outbound_seats_detail['additional_seats'] = (outbound_seats_detail['plan_seats'] - outbound_seats_detail['actual_seats']).clip(lower=0)
                
                # Tạo bảng chi tiết với format số có dấu phẩy
                detail_table = pd.DataFrame({
                    'STT': range(1, len(outbound_seats_detail) + 1),
                    'Tuyến tour': outbound_seats_detail['route'],
                    'Doanh thu kế hoạch (Tr.đ)': outbound_seats_detail['plan_revenue_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Doanh số dự kiến (Tr.đ)': outbound_seats_detail['plan_revenue_etour_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Doanh số đã bán (Tr.đ)': outbound_seats_detail['actual_revenue_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Tốc độ đạt kế hoạch DT (%)': outbound_seats_detail['completion_revenue_pct'].round(1).astype(str) + '%',
                    'DT kế hoạch còn (Tr.đ)': outbound_seats_detail['additional_revenue_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'LK Kế hoạch': outbound_seats_detail['plan_seats'].fillna(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Lượt khách dự kiến': outbound_seats_detail['plan_seats'].fillna(0).astype(int).apply(lambda x: f"{x:,}"),
                    'LK đã thực hiện': outbound_seats_detail['actual_seats'].fillna(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Tốc độ đạt kế hoạch LK (%)': outbound_seats_detail['completion_seats_pct'].round(1).astype(str) + '%',
                    'LK kế hoạch còn': outbound_seats_detail['additional_seats'].fillna(0).astype(int).apply(lambda x: f"{x:,}")
                })
                
                # Sắp xếp theo DT đã bán giảm dần (dùng giá trị số thực tế, không phải string đã format)
                detail_table['_sort_revenue'] = outbound_seats_detail['actual_revenue_tr'].fillna(0)
                detail_table = detail_table.sort_values('_sort_revenue', ascending=False).reset_index(drop=True)
                detail_table = detail_table.drop(columns=['_sort_revenue'])
                detail_table['STT'] = range(1, len(detail_table) + 1)
                
                st.dataframe(detail_table, use_container_width=True, hide_index=True)
        
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
    
    # ========== VÙNG 1: TỐC ĐỘ ĐẠT KẾ HOẠCH ==========
    st.markdown("### TỐC ĐỘ ĐẠT KẾ HOẠCH (DATANET)")
    
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
    st.markdown("### TỐC ĐỘ ĐẠT KẾ HOẠCH TUYẾN (DATANET)")
    
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
        # QUAN TRỌNG: Lấy giá trị từ dòng subtotal tương ứng
        # - Tất cả = Total LK
        # - TPHCM & DNB = TPHCM & DNB LK
        # - Mien Trung = Mien Trung LK
        # - Mien Tay = Mien Tay LK
        # - Mien Bac = Mien Bac LK
        filtered_data = route_performance_data.copy()
        # Filter theo Giai đoạn (không cần kiểm tra "Tất cả" vì đã bỏ option này)
        filtered_data = filtered_data[filtered_data['period'] == selected_period].copy()
        
        # Map tên khu vực sang tên subtotal
        region_to_subtotal = {
            'Tất cả': 'Total LK',
            'TPHCM & DNB': 'TPHCM & DNB LK',
            'Mien Trung': 'Mien Trung LK',
            'Mien Tay': 'Mien Tay LK',
            'Mien Bac': 'Mien Bac LK'
        }
        
        # Tìm dòng subtotal tương ứng
        selected_region_normalized = str(selected_region).strip()
        subtotal_name = region_to_subtotal.get(selected_region_normalized, None)
        
        if subtotal_name and ('region_unit' in filtered_data.columns or 'unit' in filtered_data.columns):
            # Tìm dòng có region_unit hoặc unit chứa subtotal_name
            if 'region_unit' in filtered_data.columns:
                subtotal_rows = filtered_data[
                    filtered_data['region_unit'].astype(str).str.contains(subtotal_name, case=False, na=False)
                ]
            elif 'unit' in filtered_data.columns:
                subtotal_rows = filtered_data[
                    filtered_data['unit'].astype(str).str.contains(subtotal_name, case=False, na=False)
                ]
            else:
                subtotal_rows = pd.DataFrame()
            
            if not subtotal_rows.empty:
                # Lấy giá trị từ dòng subtotal
                filtered_data = subtotal_rows.copy()
            else:
                # Fallback: filter theo region_unit như cũ
                if selected_region != 'Tất cả':
                    filtered_data = filtered_data[filtered_data['region_unit'] == selected_region].copy()
        else:
            # Fallback: filter theo region_unit như cũ
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
        
        # Tách dữ liệu: loại bỏ "Dom Total", "Out Total", "Grand Total" khỏi biểu đồ
        # Nhưng vẫn giữ lại để tính phần trăm
        domestic_data_for_chart = domestic_data[
            ~domestic_data['route'].astype(str).str.contains('Total', case=False, na=False)
        ].copy()
        
        # Lấy giá trị từ "Dom Total" nếu có (để tính phần trăm)
        dom_total_row = domestic_data[
            domestic_data['route'].astype(str).str.contains('Dom Total', case=False, na=False)
        ]
        
        # 3 biểu đồ Nội địa (không hiển thị Total)
        col1, col2, col3 = st.columns(3)

    with col1:
            if not domestic_data_for_chart.empty:
                fig_dom_cust = create_route_performance_chart(
                    domestic_data_for_chart, 
                    metric='num_customers', 
                    title='Lượt Khách'
                )
                st.plotly_chart(fig_dom_cust, use_container_width=True, key="route_dom_cust")
            else:
                st.info("Không có dữ liệu")
        
    with col2:
            if not domestic_data_for_chart.empty:
                fig_dom_rev = create_route_performance_chart(
                    domestic_data_for_chart, 
                    metric='revenue', 
                    title='Doanh Thu'
                )
                st.plotly_chart(fig_dom_rev, use_container_width=True, key="route_dom_rev")
            else:
                st.info("Không có dữ liệu")
    
    with col3:
            if not domestic_data_for_chart.empty:
                fig_dom_profit = create_route_performance_chart(
                    domestic_data_for_chart, 
                    metric='gross_profit', 
                    title='Lãi Gộp'
                )
                st.plotly_chart(fig_dom_profit, use_container_width=True, key="route_dom_profit")
            else:
                st.info("Không có dữ liệu")
    
    # 3 card hiển thị % hoàn thành kế hoạch cho Nội địa
    # Sử dụng hàm load_completion_progress_actual_data và load_completion_progress_plan_data
    # Lấy actual data từ URL gid=903527778 với nhom_tuyen = "Dom Total"
    route_performance_url = st.session_state.get('route_performance_url', DEFAULT_ROUTE_PERFORMANCE_URL)
    plan_tet_url = st.session_state.get('plan_tet_url', DEFAULT_PLAN_TET_URL)
    plan_xuan_url = st.session_state.get('plan_xuan_url', DEFAULT_PLAN_XUAN_URL)
    
    # Lấy period filter
    selected_period = st.session_state.get('sidebar_period_filter') or st.session_state.get('filter_period', 'KM XUÂN')
    
    # Lấy region filter để tạo cache key
    selected_region = st.session_state.get('filter_region', 'Tất cả')
    
    # Cache key cho actual data (bao gồm region để reload khi region thay đổi)
    cache_key_actual = f'completion_actual_data_{route_performance_url}_{selected_period}_{selected_region}'
    if cache_key_actual not in st.session_state:
        actual_data = load_completion_progress_actual_data(route_performance_url)
        st.session_state[cache_key_actual] = actual_data
    else:
        actual_data = st.session_state[cache_key_actual]
    
    # Lấy giá trị actual từ "Dom Total"
    total_customers_actual = 0
    total_revenue_actual = 0
    total_profit_actual = 0
    
    if not actual_data.empty:
        # Filter theo period nếu có
        if selected_period != 'Tất cả':
            actual_data_filtered = actual_data[actual_data['period'].astype(str).str.contains(selected_period, case=False, na=False)]
        else:
            actual_data_filtered = actual_data
        
        # Map tên khu vực từ filter sang tên trong region_unit
        region_mapping = {
            'Tất cả': 'Total LK',
            'Mien Bac': 'Mien Bac LK',
            'Mien Trung': 'Mien Trung LK',
            'Mien Tay': 'Mien Tay LK',
            'TPHCM & DNB': 'TPHCM & DNB LK'
        }
        target_region_unit = region_mapping.get(selected_region, 'Total LK')
        
        # Filter theo region_unit và nhom_tuyen = "Dom Total"
        dom_total_actual = actual_data_filtered[
            (actual_data_filtered['region_unit'].astype(str).str.contains(target_region_unit, case=False, na=False)) &
            (actual_data_filtered['nhom_tuyen'].astype(str).str.contains('Dom Total', case=False, na=False))
        ]
        
        # CHỈ LẤY GIÁ TRỊ TỪ 1 DÒNG DUY NHẤT, KHÔNG SUM
        if not dom_total_actual.empty:
            total_customers_actual = dom_total_actual['num_customers'].iloc[0] if 'num_customers' in dom_total_actual.columns else 0
            total_revenue_actual = dom_total_actual['revenue'].iloc[0] if 'revenue' in dom_total_actual.columns else 0
            total_profit_actual = dom_total_actual['gross_profit'].iloc[0] if 'gross_profit' in dom_total_actual.columns else 0
    
    # Lấy plan data (cache key bao gồm region để reload khi region thay đổi)
    plan_key = f'domestic_plan_{selected_period}_{selected_region}'
    total_customers_plan = 0
    total_revenue_plan = 0
    total_profit_plan = 0
    
    # Kiểm tra cache
    if plan_key in st.session_state:
        cached_plan = st.session_state[plan_key]
        total_customers_plan = cached_plan.get('customers', 0)
        total_revenue_plan = cached_plan.get('revenue', 0)
        total_profit_plan = cached_plan.get('profit', 0)
    else:
        # Load plan data từ Plan Tết và Plan Xuân
        if selected_period == 'TẾT' or selected_period == 'Tất cả':
            plan_tet_data = load_completion_progress_plan_data(plan_tet_url, period_name='TẾT')
            if not plan_tet_data.empty:
                dom_total_plan_tet = plan_tet_data[
                    plan_tet_data['nhom_tuyen'].astype(str).str.contains('Dom Total', case=False, na=False)
                ]
                if not dom_total_plan_tet.empty:
                    # CHỈ LẤY GIÁ TRỊ TỪ 1 DÒNG DUY NHẤT, KHÔNG SUM
                    total_customers_plan += dom_total_plan_tet['plan_customers'].iloc[0] if 'plan_customers' in dom_total_plan_tet.columns else 0
                    total_revenue_plan += dom_total_plan_tet['plan_revenue'].iloc[0] if 'plan_revenue' in dom_total_plan_tet.columns else 0
                    total_profit_plan += dom_total_plan_tet['plan_profit'].iloc[0] if 'plan_profit' in dom_total_plan_tet.columns else 0
        
        if selected_period == 'KM XUÂN' or selected_period == 'Tất cả':
            plan_xuan_data = load_completion_progress_plan_data(plan_xuan_url, period_name='KM XUÂN')
            if not plan_xuan_data.empty:
                dom_total_plan_xuan = plan_xuan_data[
                    plan_xuan_data['nhom_tuyen'].astype(str).str.contains('Dom Total', case=False, na=False)
                ]
                if not dom_total_plan_xuan.empty:
                    # CHỈ LẤY GIÁ TRỊ TỪ 1 DÒNG DUY NHẤT, KHÔNG SUM
                    total_customers_plan += dom_total_plan_xuan['plan_customers'].iloc[0] if 'plan_customers' in dom_total_plan_xuan.columns else 0
                    total_revenue_plan += dom_total_plan_xuan['plan_revenue'].iloc[0] if 'plan_revenue' in dom_total_plan_xuan.columns else 0
                    total_profit_plan += dom_total_plan_xuan['plan_profit'].iloc[0] if 'plan_profit' in dom_total_plan_xuan.columns else 0
        
        # Lưu vào cache
        st.session_state[plan_key] = {
            'customers': total_customers_plan,
            'revenue': total_revenue_plan,
            'profit': total_profit_plan
        }
    
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
    
    # Tách dữ liệu: loại bỏ "Dom Total", "Out Total", "Grand Total" khỏi biểu đồ
    # Nhưng vẫn giữ lại để tính phần trăm
    outbound_data_for_chart = outbound_data[
        ~outbound_data['route'].astype(str).str.contains('Total', case=False, na=False)
    ].copy()
    
    # Lấy giá trị từ "Out Total" nếu có (để tính phần trăm)
    out_total_row = outbound_data[
        outbound_data['route'].astype(str).str.contains('Out Total', case=False, na=False)
    ]
    
    # 3 biểu đồ Outbound (không hiển thị Total)
    col1, col2, col3 = st.columns(3)

    with col1:
        if not outbound_data_for_chart.empty:
            fig_out_cust = create_route_performance_chart(
                outbound_data_for_chart, 
                metric='num_customers', 
                title='Lượt Khách'
            )
            st.plotly_chart(fig_out_cust, use_container_width=True, key="route_out_cust")
        else:
            st.info("Không có dữ liệu")

    with col2:
        if not outbound_data_for_chart.empty:
            fig_out_rev = create_route_performance_chart(
                outbound_data_for_chart, 
                metric='revenue', 
                title='Doanh Thu'
            )
            st.plotly_chart(fig_out_rev, use_container_width=True, key="route_out_rev")
        else:
            st.info("Không có dữ liệu")
    
    with col3:
        if not outbound_data_for_chart.empty:
            fig_out_profit = create_route_performance_chart(
                outbound_data_for_chart, 
                metric='gross_profit', 
                title='Lãi Gộp'
            )
            st.plotly_chart(fig_out_profit, use_container_width=True, key="route_out_profit")
        else:
            st.info("Không có dữ liệu")
    
    # 3 card hiển thị % hoàn thành kế hoạch cho Outbound
    # Sử dụng hàm load_completion_progress_actual_data và load_completion_progress_plan_data
    # Lấy actual data từ URL gid=903527778 với nhom_tuyen = "Out Total"
    # Sử dụng lại actual_data đã load ở phần Domestic
    if cache_key_actual in st.session_state:
        actual_data = st.session_state[cache_key_actual]
    else:
        actual_data = load_completion_progress_actual_data(route_performance_url)
        st.session_state[cache_key_actual] = actual_data
    
    # Lấy giá trị actual từ "Out Total"
    total_customers_actual_outbound = 0
    total_revenue_actual_outbound = 0
    total_profit_actual_outbound = 0
    
    if not actual_data.empty:
        # Filter theo period nếu có
        if selected_period != 'Tất cả':
            actual_data_filtered = actual_data[actual_data['period'].astype(str).str.contains(selected_period, case=False, na=False)]
        else:
            actual_data_filtered = actual_data
        
        # Map tên khu vực từ filter sang tên trong region_unit
        region_mapping = {
            'Tất cả': 'Total LK',
            'Mien Bac': 'Mien Bac LK',
            'Mien Trung': 'Mien Trung LK',
            'Mien Tay': 'Mien Tay LK',
            'TPHCM & DNB': 'TPHCM & DNB LK'
        }
        target_region_unit = region_mapping.get(selected_region, 'Total LK')
        
        # Filter theo region_unit và nhom_tuyen = "Out Total"
        out_total_actual = actual_data_filtered[
            (actual_data_filtered['region_unit'].astype(str).str.contains(target_region_unit, case=False, na=False)) &
            (actual_data_filtered['nhom_tuyen'].astype(str).str.contains('Out Total', case=False, na=False))
        ]
        
        # CHỈ LẤY GIÁ TRỊ TỪ 1 DÒNG DUY NHẤT, KHÔNG SUM
        if not out_total_actual.empty:
            total_customers_actual_outbound = out_total_actual['num_customers'].iloc[0] if 'num_customers' in out_total_actual.columns else 0
            total_revenue_actual_outbound = out_total_actual['revenue'].iloc[0] if 'revenue' in out_total_actual.columns else 0
            total_profit_actual_outbound = out_total_actual['gross_profit'].iloc[0] if 'gross_profit' in out_total_actual.columns else 0
    
    # Lấy plan data (cache key bao gồm region để reload khi region thay đổi)
    plan_key_outbound = f'outbound_plan_{selected_period}_{selected_region}'
    total_customers_plan_outbound = 0
    total_revenue_plan_outbound = 0
    total_profit_plan_outbound = 0
    
    # Kiểm tra cache
    if plan_key_outbound in st.session_state:
        cached_plan = st.session_state[plan_key_outbound]
        total_customers_plan_outbound = cached_plan.get('customers', 0)
        total_revenue_plan_outbound = cached_plan.get('revenue', 0)
        total_profit_plan_outbound = cached_plan.get('profit', 0)
    else:
        # Load plan data từ Plan Tết và Plan Xuân
        if selected_period == 'TẾT' or selected_period == 'Tất cả':
            plan_tet_data = load_completion_progress_plan_data(plan_tet_url, period_name='TẾT')
            if not plan_tet_data.empty:
                out_total_plan_tet = plan_tet_data[
                    plan_tet_data['nhom_tuyen'].astype(str).str.contains('Out Total', case=False, na=False)
                ]
                if not out_total_plan_tet.empty:
                    # CHỈ LẤY GIÁ TRỊ TỪ 1 DÒNG DUY NHẤT, KHÔNG SUM
                    total_customers_plan_outbound += out_total_plan_tet['plan_customers'].iloc[0] if 'plan_customers' in out_total_plan_tet.columns else 0
                    total_revenue_plan_outbound += out_total_plan_tet['plan_revenue'].iloc[0] if 'plan_revenue' in out_total_plan_tet.columns else 0
                    total_profit_plan_outbound += out_total_plan_tet['plan_profit'].iloc[0] if 'plan_profit' in out_total_plan_tet.columns else 0
        
        if selected_period == 'KM XUÂN' or selected_period == 'Tất cả':
            plan_xuan_data = load_completion_progress_plan_data(plan_xuan_url, period_name='KM XUÂN')
            if not plan_xuan_data.empty:
                out_total_plan_xuan = plan_xuan_data[
                    plan_xuan_data['nhom_tuyen'].astype(str).str.contains('Out Total', case=False, na=False)
                ]
                if not out_total_plan_xuan.empty:
                    # CHỈ LẤY GIÁ TRỊ TỪ 1 DÒNG DUY NHẤT, KHÔNG SUM
                    total_customers_plan_outbound += out_total_plan_xuan['plan_customers'].iloc[0] if 'plan_customers' in out_total_plan_xuan.columns else 0
                    total_revenue_plan_outbound += out_total_plan_xuan['plan_revenue'].iloc[0] if 'plan_revenue' in out_total_plan_xuan.columns else 0
                    total_profit_plan_outbound += out_total_plan_xuan['plan_profit'].iloc[0] if 'plan_profit' in out_total_plan_xuan.columns else 0
        
        # Lưu vào cache
        st.session_state[plan_key_outbound] = {
            'customers': total_customers_plan_outbound,
            'revenue': total_revenue_plan_outbound,
            'profit': total_profit_plan_outbound
        }
    
    # Tính % hoàn thành
    completion_customers = (total_customers_actual_outbound / total_customers_plan_outbound * 100) if total_customers_plan_outbound > 0 else 0
    completion_revenue = (total_revenue_actual_outbound / total_revenue_plan_outbound * 100) if total_revenue_plan_outbound > 0 else 0
    completion_profit = (total_profit_actual_outbound / total_profit_plan_outbound * 100) if total_profit_plan_outbound > 0 else 0
    
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
                
                # Clear completion progress actual data cache (bao gồm region)
                selected_period = st.session_state.get('sidebar_period_filter') or st.session_state.get('filter_period', 'KM XUÂN')
                selected_region = st.session_state.get('filter_region', 'Tất cả')
                cache_key_actual = f'completion_actual_data_{route_performance_url}_{selected_period}_{selected_region}'
                if cache_key_actual in st.session_state:
                    del st.session_state[cache_key_actual]
                
                # Clear plan data cache
                region_filter = selected_region if selected_region != 'Tất cả' else None
                plan_tet_url = st.session_state.get('plan_tet_url', '')
                plan_xuan_url = st.session_state.get('plan_xuan_url', '')
                cache_key_plan_tet = f'plan_tet_data_{plan_tet_url}_{region_filter}'
                cache_key_plan_xuan = f'plan_xuan_data_{plan_xuan_url}_{region_filter}'
                if cache_key_plan_tet in st.session_state:
                    del st.session_state[cache_key_plan_tet]
                if cache_key_plan_xuan in st.session_state:
                    del st.session_state[cache_key_plan_xuan]
                
                # Clear giá trị plan đã lưu trong session_state (bao gồm region)
                plan_key_domestic = f'domestic_plan_{selected_period}_{selected_region}'
                plan_key_outbound = f'outbound_plan_{selected_period}_{selected_region}'
                if plan_key_domestic in st.session_state:
                    del st.session_state[plan_key_domestic]
                if plan_key_outbound in st.session_state:
                    del st.session_state[plan_key_outbound]
                
                st.rerun()
    
    if route_performance_data.empty:
        st.warning("Không thể tải dữ liệu từ Google Sheet. Vui lòng kiểm tra URL và quyền truy cập.")
        if st.button("🔄 Thử lại", key="retry_route_performance"):
            route_performance_data = load_route_performance_data(route_performance_url)
            st.session_state[cache_key_route] = route_performance_data
            st.rerun()

    st.markdown("---")

    # ========== BẢNG TIẾN ĐỘ HOÀN THÀNH KẾ HOẠCH ==========
    st.markdown("### TIẾN ĐỘ HOÀN THÀNH KẾ HOẠCH (DATANET)")
    
    # Dữ liệu kế hoạch đã được load ở trên (all_plan_data)
    
    if not all_plan_data.empty and not route_performance_data.empty:
        # Merge kế hoạch với thực tế theo route CHÍNH XÁC
        # QUAN TRỌNG: Chỉ lấy giá trị khi match CHÍNH XÁC, không match thì để 0 (không fallback)
        # Normalize Unicode để xử lý vấn đề encoding (ví dụ: 'Ú' vs 'Ú')
        import unicodedata
        
        def normalize_unicode(text):
            """Normalize Unicode để so sánh chính xác"""
            if pd.isna(text):
                return ''
            text_str = str(text).strip()
            # Normalize về dạng NFC (Canonical Composition)
            return unicodedata.normalize('NFC', text_str)
        
        # Normalize route names để so sánh
        all_plan_data['route_normalized'] = all_plan_data['route'].apply(normalize_unicode)
        route_performance_data['route_normalized'] = route_performance_data['route'].apply(normalize_unicode)
        
        # Merge với route normalized và period - CHỈ lấy khi match CHÍNH XÁC
        merged_data = route_performance_data.merge(
            all_plan_data[['route_normalized', 'period', 'plan_customers', 'plan_revenue', 'plan_profit']],
            on=['route_normalized', 'period'],
            how='left',
            suffixes=('_actual', '_plan')
        )
        
        # Nếu không match được với period, thử merge chỉ với route normalized (không có period)
        # NHƯNG chỉ khi route normalized match CHÍNH XÁC
        unmatched_mask = merged_data['plan_revenue'].isna()
        if unmatched_mask.any():
            # Tạo lookup chỉ với route normalized (không có period)
            # QUAN TRỌNG: Nếu có nhiều route normalize thành cùng giá trị, ưu tiên route dài hơn (chính xác hơn)
            plan_lookup_no_period = all_plan_data.copy()
            plan_lookup_no_period['route_len'] = plan_lookup_no_period['route'].astype(str).str.len()
            plan_lookup_no_period = plan_lookup_no_period.sort_values('route_len', ascending=False)
            plan_lookup_no_period = plan_lookup_no_period.groupby('route_normalized').agg({
                'plan_customers': 'first',
                'plan_revenue': 'first',
                'plan_profit': 'first'
            }).reset_index()
            plan_lookup_no_period = plan_lookup_no_period.rename(columns={
                'plan_revenue': 'plan_revenue_fallback',
                'plan_customers': 'plan_customers_fallback',
                'plan_profit': 'plan_profit_fallback'
            })
            
            merged_unmatched = merged_data[unmatched_mask].copy()
            merged_unmatched = merged_unmatched.merge(
                plan_lookup_no_period[['route_normalized', 'plan_customers_fallback', 'plan_revenue_fallback', 'plan_profit_fallback']],
                on='route_normalized',
                how='left'
            )
            
            merged_data.loc[unmatched_mask, 'plan_revenue'] = merged_data.loc[unmatched_mask, 'plan_revenue'].fillna(merged_unmatched['plan_revenue_fallback'])
            merged_data.loc[unmatched_mask, 'plan_customers'] = merged_data.loc[unmatched_mask, 'plan_customers'].fillna(merged_unmatched['plan_customers_fallback'])
            merged_data.loc[unmatched_mask, 'plan_profit'] = merged_data.loc[unmatched_mask, 'plan_profit'].fillna(merged_unmatched['plan_profit_fallback'])
        
        # KHÔNG có fallback thêm nữa - nếu không match được, để NaN (sẽ hiển thị 0)
        
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
                'plan_customers': 'first',
                'num_customers': 'sum',
                'plan_revenue': 'first',
                'revenue': 'sum',
                'plan_profit': 'first',
                'gross_profit': 'sum'
            }).reset_index()
            
            # Tính lại completion rates từ tổng actual/plan
            domestic_chart_data['completion_customers'] = (domestic_chart_data['num_customers'] / domestic_chart_data['plan_customers'].replace(0, np.nan) * 100).fillna(0)
            domestic_chart_data['completion_revenue'] = (domestic_chart_data['revenue'] / domestic_chart_data['plan_revenue'].replace(0, np.nan) * 100).fillna(0)
            domestic_chart_data['completion_profit'] = (domestic_chart_data['gross_profit'] / domestic_chart_data['plan_profit'].replace(0, np.nan) * 100).fillna(0)
            
            fig_domestic = create_completion_progress_chart(
                domestic_chart_data,
                title='TIẾN ĐỘ HOÀN THÀNH KẾ HOẠCH - NỘI ĐỊA'
            )
            st.plotly_chart(fig_domestic, use_container_width=True, key="completion_domestic_chart")
            
            # Bảng chi tiết Nội địa
            with st.expander("📊 Xem bảng chi tiết", expanded=False):
                # Tính toán lại từ dữ liệu gốc để đảm bảo tính chính xác
                domestic_detail = domestic_completion.groupby('route').agg({
                    'plan_customers': 'first',
                    'num_customers': 'sum',
                    'plan_revenue': 'first',
                    'revenue': 'sum',
                    'plan_profit': 'first',
                    'gross_profit': 'sum'
                }).reset_index()
                
                # Loại bỏ các dòng "Grand Total" và "Dom Total"
                domestic_detail = domestic_detail[
                    ~domestic_detail['route'].astype(str).str.contains('Grand Total|Dom Total', case=False, na=False)
                ].copy()
                
                # Tính lại phần trăm hoàn thành
                domestic_detail['completion_customers_pct'] = (domestic_detail['num_customers'] / domestic_detail['plan_customers'].replace(0, np.nan) * 100).fillna(0)
                domestic_detail['completion_revenue_pct'] = (domestic_detail['revenue'] / domestic_detail['plan_revenue'].replace(0, np.nan) * 100).fillna(0)
                domestic_detail['completion_profit_pct'] = (domestic_detail['gross_profit'] / domestic_detail['plan_profit'].replace(0, np.nan) * 100).fillna(0)
                
                # Chuyển đổi đơn vị sang triệu đồng
                domestic_detail['plan_revenue_tr'] = domestic_detail['plan_revenue'] / 1_000_000
                domestic_detail['revenue_tr'] = domestic_detail['revenue'] / 1_000_000
                domestic_detail['plan_profit_tr'] = domestic_detail['plan_profit'] / 1_000_000
                domestic_detail['gross_profit_tr'] = domestic_detail['gross_profit'] / 1_000_000
                
                # Tạo bảng chi tiết với format số có dấu phẩy
                detail_table = pd.DataFrame({
                    'STT': range(1, len(domestic_detail) + 1),
                    'Tuyến tour': domestic_detail['route'],
                    'LK kế hoạch': domestic_detail['plan_customers'].fillna(0).astype(int).apply(lambda x: f"{x:,}"),
                    'LK thực hiện': domestic_detail['num_customers'].fillna(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Tốc độ đạt KH (%)': domestic_detail['completion_customers_pct'].round(1).astype(str) + '%',
                    'DT kế hoạch (Tr.đ)': domestic_detail['plan_revenue_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'DT đã bán (Tr.đ)': domestic_detail['revenue_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Tốc độ đạt kế hoạch (%)': domestic_detail['completion_revenue_pct'].round(1).astype(str) + '%',
                    'LG kế hoạch (tr.đ)': domestic_detail['plan_profit_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'LG thực hiện (tr.đ)': domestic_detail['gross_profit_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'tốc độ đạt kết hoạch (%)': domestic_detail['completion_profit_pct'].round(1).astype(str) + '%'
                })
                
                # Sắp xếp theo DT đã bán giảm dần (dùng giá trị số thực tế, không phải string đã format)
                detail_table['_sort_revenue'] = domestic_detail['revenue_tr'].fillna(0)
                detail_table = detail_table.sort_values('_sort_revenue', ascending=False).reset_index(drop=True)
                detail_table = detail_table.drop(columns=['_sort_revenue'])
                detail_table['STT'] = range(1, len(detail_table) + 1)
                
                st.dataframe(detail_table, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Hiển thị biểu đồ và bảng Outbound
        if not outbound_completion.empty:
            st.markdown("#### Outbound")
            
            # Tạo biểu đồ line chart
            # Nhóm theo route để tổng hợp (nếu có nhiều period cho cùng route)
            outbound_chart_data = outbound_completion.groupby('route').agg({
                'plan_customers': 'first',
                'num_customers': 'sum',
                'plan_revenue': 'first',
                'revenue': 'sum',
                'plan_profit': 'first',
                'gross_profit': 'sum'
            }).reset_index()
            
            # Tính lại completion rates từ tổng actual/plan
            outbound_chart_data['completion_customers'] = (outbound_chart_data['num_customers'] / outbound_chart_data['plan_customers'].replace(0, np.nan) * 100).fillna(0)
            outbound_chart_data['completion_revenue'] = (outbound_chart_data['revenue'] / outbound_chart_data['plan_revenue'].replace(0, np.nan) * 100).fillna(0)
            outbound_chart_data['completion_profit'] = (outbound_chart_data['gross_profit'] / outbound_chart_data['plan_profit'].replace(0, np.nan) * 100).fillna(0)
            
            fig_outbound = create_completion_progress_chart(
                outbound_chart_data,
                title='TIẾN ĐỘ HOÀN THÀNH KẾ HOẠCH - OUTBOUND'
            )
            st.plotly_chart(fig_outbound, use_container_width=True, key="completion_outbound_chart")
            
            # Bảng chi tiết Outbound
            with st.expander("📊 Xem bảng chi tiết", expanded=False):
                # Tính toán lại từ dữ liệu gốc để đảm bảo tính chính xác
                outbound_detail = outbound_completion.groupby('route').agg({
                    'plan_customers': 'first',
                    'num_customers': 'sum',
                    'plan_revenue': 'first',
                    'revenue': 'sum',
                    'plan_profit': 'first',
                    'gross_profit': 'sum'
                }).reset_index()
                
                # Loại bỏ các dòng "Grand Total", "Out Total", và các dòng có "Total" trong tên
                outbound_detail = outbound_detail[
                    ~outbound_detail['route'].astype(str).str.contains('Total', case=False, na=False)
                ].copy()
                
                # Tính lại phần trăm hoàn thành
                outbound_detail['completion_customers_pct'] = (outbound_detail['num_customers'] / outbound_detail['plan_customers'].replace(0, np.nan) * 100).fillna(0)
                outbound_detail['completion_revenue_pct'] = (outbound_detail['revenue'] / outbound_detail['plan_revenue'].replace(0, np.nan) * 100).fillna(0)
                outbound_detail['completion_profit_pct'] = (outbound_detail['gross_profit'] / outbound_detail['plan_profit'].replace(0, np.nan) * 100).fillna(0)
                
                # Chuyển đổi đơn vị sang triệu đồng
                outbound_detail['plan_revenue_tr'] = outbound_detail['plan_revenue'] / 1_000_000
                outbound_detail['revenue_tr'] = outbound_detail['revenue'] / 1_000_000
                outbound_detail['plan_profit_tr'] = outbound_detail['plan_profit'] / 1_000_000
                outbound_detail['gross_profit_tr'] = outbound_detail['gross_profit'] / 1_000_000
                
                # Tạo bảng chi tiết với format số có dấu phẩy
                detail_table = pd.DataFrame({
                    'STT': range(1, len(outbound_detail) + 1),
                    'Tuyến tour': outbound_detail['route'],
                    'LK kế hoạch': outbound_detail['plan_customers'].fillna(0).astype(int).apply(lambda x: f"{x:,}"),
                    'LK thực hiện': outbound_detail['num_customers'].fillna(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Tốc độ đạt KH (%)': outbound_detail['completion_customers_pct'].round(1).astype(str) + '%',
                    'DT kế hoạch (Tr.đ)': outbound_detail['plan_revenue_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'DT đã bán (Tr.đ)': outbound_detail['revenue_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'Tốc độ đạt kế hoạch (%)': outbound_detail['completion_revenue_pct'].round(1).astype(str) + '%',
                    'LG kế hoạch (tr.đ)': outbound_detail['plan_profit_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'LG thực hiện (tr.đ)': outbound_detail['gross_profit_tr'].fillna(0).round(0).astype(int).apply(lambda x: f"{x:,}"),
                    'tốc độ đạt kết hoạch (%)': outbound_detail['completion_profit_pct'].round(1).astype(str) + '%'
                })
                
                # Sắp xếp theo DT đã bán giảm dần (dùng giá trị số thực tế, không phải string đã format)
                detail_table['_sort_revenue'] = outbound_detail['revenue_tr'].fillna(0)
                detail_table = detail_table.sort_values('_sort_revenue', ascending=False).reset_index(drop=True)
                detail_table = detail_table.drop(columns=['_sort_revenue'])
                detail_table['STT'] = range(1, len(detail_table) + 1)
                
                st.dataframe(detail_table, use_container_width=True, hide_index=True)
        
        # Nút refresh
        col_refresh1, col_refresh2 = st.columns([1, 5])
        with col_refresh1:
            if st.button("🔄 Làm mới dữ liệu Kế hoạch", key="refresh_plan_data"):
                plan_tet_data = load_route_plan_data(plan_tet_url, period_name='TẾT', region_filter=region_filter)
                plan_xuan_data = load_route_plan_data(plan_xuan_url, period_name='KM XUÂN', region_filter=region_filter)
                st.session_state[cache_key_plan_tet] = plan_tet_data
                st.session_state[cache_key_plan_xuan] = plan_xuan_data
                st.rerun()
    else:
        if all_plan_data.empty:
            st.warning("Không thể tải dữ liệu kế hoạch từ Google Sheet. Vui lòng kiểm tra URL và quyền truy cập.")
        if route_performance_data.empty:
            st.warning("Không có dữ liệu thực tế để so sánh.")
    
    st.markdown("---")





# ============================================================

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p>📊 Vietravel Business Intelligence Dashboard Ver 2</p>
        <p>Cập nhật lần cuối: {}</p>
    </div>
""".format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)