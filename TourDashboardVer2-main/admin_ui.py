import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import random 
import io

def render_admin_ui():
    """
    Renders the dedicated Admin UI for creating new contracts or editing existing ones,
    modifying financial and status data directly in st.session_state.tours_df.
    """
    # --- Data source selector (restore "Nguồn dữ liệu" UI) ---
    st.header("⚙️ Nhập liệu/Sửa Hợp đồng")

    st.subheader("Nguồn dữ liệu")
    use_gsheet = st.checkbox("Dùng Google Sheet (CSV public)", value=False, key="use_gsheet_checkbox")
    gs_link = ""
    if use_gsheet:
        gs_link = st.text_input("Link Google Sheet", value=st.session_state.get("gs_link", ""), key="use_gsheet_link")
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Tải dữ liệu từ Google Sheet"):
                if not gs_link or gs_link.strip() == "":
                    st.error("Vui lòng nhập link Google Sheet public (CSV)")
                else:
                    try:
                        # Normalize common Google Sheets edit URL into CSV export URL
                        def to_csv_export_url(url: str) -> str:
                            import re
                            if not isinstance(url, str):
                                return url
                            # If user pasted a Google Sheets edit/view URL, convert to export CSV
                            m = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
                            if m:
                                sheet_id = m.group(1)
                                gid_m = re.search(r"[?&]gid=(\d+)", url)
                                gid = gid_m.group(1) if gid_m else None
                                export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                                if gid:
                                    export_url += f"&gid={gid}"
                                return export_url
                            return url

                        csv_url = to_csv_export_url(gs_link)

                        # Use the python engine and skip malformed lines to avoid tokenizing errors
                        # Report how many rows were read and warn if lines were skipped
                        # decode bytes defensively to avoid codec errors
                        import requests
                        resp = requests.get(csv_url, timeout=20)
                        raw = resp.content
                        text = raw.decode('utf-8', errors='replace')

                        # Try to detect header row by scanning the first 20 rows for a cell containing "Mã tour"
                        import csv
                        rows = list(csv.reader(io.StringIO(text)))
                        header_row = None
                        for i, r in enumerate(rows[:20]):
                            if any(isinstance(c, str) and 'mã tour' in c.lower() for c in r):
                                header_row = i
                                break

                        if header_row is None:
                            # fallback: assume header is row 1 (index 0) or use row 1 if you know it's row 2 (Excel 1-based)
                            # The user said header is row 2 (Excel 1-based), so prefer index 1 if no detection
                            header_row = 1 if len(rows) > 1 else 0

                        st.info(f"Detected header row: {header_row + 1} (1-based). Using that row as column names.")

                        # Read with pandas using detected header row. Use python engine and skip bad lines.
                        df = pd.read_csv(
                            io.StringIO(text),
                            engine='python',
                            header=header_row,
                            on_bad_lines='skip'
                        )

                        # Show a small preview of the parsed header to the user
                        try:
                            preview_cols = df.columns.tolist()[:12]
                            st.write("Detected columns:", preview_cols)
                            st.dataframe(df.head(5))
                        except Exception:
                            pass

                        if df is None or df.empty:
                            st.warning("Không đọc được dữ liệu từ CSV. Kiểm tra link và quyền truy cập (phải public / đúng định dạng CSV).")
                            # If the provided link was not a CSV export, give a hint
                            if csv_url != gs_link:
                                st.info("Hệ thống đã cố gắng chuyển URL Google Sheets sang dạng export CSV rồi nhưng không đọc được. Vui lòng dùng 'Publish -> CSV' hoặc URL chứa 'export?format=csv'.")
                        else:
                            # Try to normalize column names to the expected schema used by the app
                            original_cols = df.columns.tolist()

                            # expected column keys and candidate keywords to match
                            expected_map = {
                                'booking_id': ['mã tour', 'mã hợp đồng', 'booking_id', 'mã', 'ma tour', 'mã tour'],
                                'customer_id': ['mã khách', 'customer', 'customer_id', 'mã khách hàng'],
                                'booking_date': ['từ ngày', 'booking_date', 'booking date', 'ngày đặt', 'ngày'],
                                'route': ['tuyến', 'tuyen', 'route', 'tuyến tour', 'tuyến'],
                                'business_unit': ['đơn vị', 'đơn vị kinh doanh', 'chi nhánh', 'khu vực', 'business_unit', 'business unit'],
                                'sales_channel': ['kênh', 'kenh', 'sales_channel', 'kênh bán', 'kênh bán'],
                                'segment': ['phân khúc', 'segment', 'fit', 'git', 'inbound'],
                                'num_customers': ['lượt khách', 'số khách', 'số lượng', 'num_customers', 'số khách fit', 'số khách'],
                                'price_per_person': ['giá', 'price_per_person', 'giá/khách', 'giá trung bình'],
                                'revenue': ['doanh thu', 'revenue'],
                                'cost': ['cost', 'chi phí', 'cost tổng', 'chi phí dv'],
                                'gross_profit': ['lãi', 'lợi', 'lãi gộp', 'gross_profit', 'gross profit'],
                                'gross_profit_margin': ['tỷ lệ', 'margin', 'profit margin', 'gross_profit_margin'],
                                'status': ['trạng thái', 'status', 'tình trạng'],
                                'marketing_cost': ['marketing', 'chi phí marketing'],
                                'sales_cost': ['sales_cost', 'chi phí bán hàng', 'chi phí bán'],
                                'opex': ['opex'],
                                'partner': ['đối tác', 'partner'],
                                'service_type': ['loại dịch vụ', 'service_type', 'service'],
                                'contract_status': ['trạng thái hd', 'contract_status', 'hợp đồng'],
                                'payment_status': ['trạng thái tt', 'payment_status', 'thanh toán'],
                                'service_cost': ['service_cost', 'chi phí dịch vụ'],
                                'customer_age_group': ['độ tuổi', 'age', 'age_group'],
                                'customer_nationality': ['quốc tịch', 'nationality']
                            }

                            col_map = {}
                            lowered = {c: c.lower() for c in original_cols}
                            # iterate original columns first and find a matching expected column
                            for orig_col, orig_lower in lowered.items():
                                for expect_col, keywords in expected_map.items():
                                    for kw in keywords:
                                        if kw in orig_lower:
                                            col_map[orig_col] = expect_col
                                            break
                                    if orig_col in col_map:
                                        break

                            # Apply rename (if any)
                            if col_map:
                                df = df.rename(columns=col_map)
                                st.write("Ánh xạ cột (original -> mapped):")
                                st.write({k: v for k, v in col_map.items()})

                                # If multiple original columns were mapped to the same expected name,
                                # merge them by taking the first non-null value from left to right.
                                cols = list(df.columns)
                                dup_names = [c for c in cols if cols.count(c) > 1]
                                dup_names = list(dict.fromkeys(dup_names))  # unique preserve order
                                for name in dup_names:
                                    same = [i for i, c in enumerate(cols) if c == name]
                                    if len(same) <= 1:
                                        continue
                                    # select all columns with this name
                                    group = df.iloc[:, same]
                                    # merge to single column: first non-null in each row
                                    merged = group.bfill(axis=1).iloc[:, 0]
                                    # drop all these columns and set merged as single column
                                    df.drop(df.columns[same], axis=1, inplace=True)
                                    df[name] = merged

                            # Ensure required columns exist to avoid KeyError in app
                            required_defaults = {
                                'business_unit': 'Unknown',
                                'route': 'Unknown',
                                'segment': 'Tất cả',
                                'partner': 'Unknown',
                                'service_type': 'Unknown',
                                'booking_id': None,
                                'customer_id': None,
                                'revenue': 0,
                                'gross_profit': 0,
                                'num_customers': 0
                            }
                            for col, default in required_defaults.items():
                                if col not in df.columns:
                                    df[col] = default

                            # If booking_id missing or contains NaN, create unique ids
                            if df['booking_id'].isnull().all() or df['booking_id'].dtype == object and df['booking_id'].isnull().any():
                                df['booking_id'] = [f"BK{i+1:06d}" for i in range(len(df))]

                            # Save normalized DataFrame and report
                            st.session_state.tours_df = df
                            st.session_state.gs_link = gs_link
                            st.success(f"Đã nạp và chuẩn hóa dữ liệu từ Google Sheet vào session (hàng: {len(df)} cột: {len(df.columns)})")
                            st.experimental_rerun()
                    except Exception as e:
                        # Provide a clearer error message and hint about URL vs CSV
                        st.error(f"Lỗi khi đọc CSV: {e}")
                        st.info("Gợi ý: dùng URL xuất CSV (ví dụ https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=<GID>) hoặc File->Publish to web -> CSV.")
        with cols[1]:
            st.caption("Nhập link dạng xuất CSV (Publish -> CSV) hoặc URL trực tiếp tới file CSV public.")

    # If user didn't choose to load a sheet, fall back to requiring existing session data
    if 'tours_df' not in st.session_state or st.session_state.tours_df.empty:
        st.error("Lỗi: Dữ liệu Tour chưa được tải vào Session State. Vui lòng làm mới trang hoặc nạp từ Google Sheet ở trên.")
        return

    tours_df = st.session_state.tours_df

    option = st.radio(
        "Chọn chế độ:",
        ("Sửa Hợp đồng Hiện tại", "Nhập Hợp đồng Mới"),
        index=0
    )

    # Danh sách các lựa chọn mặc định từ dữ liệu hiện có
    unique_route = sorted(tours_df['route'].unique())
    unique_unit = sorted(tours_df['business_unit'].unique())
    unique_channel = sorted(tours_df['sales_channel'].unique())
    unique_segment = sorted(tours_df['segment'].unique())
    
    # Các lựa chọn cho Tab Đối tác (Giả định/Mặc định)
    # Lấy các giá trị đã tồn tại trong DataFrame để đảm bảo chế độ SỬA không bị lỗi
    partner_options = sorted(tours_df['partner'].unique().tolist()) if 'partner' in tours_df.columns and tours_df['partner'].any() else ["Đối tác 1", "Đối tác 2"]
    service_type_options = sorted(tours_df['service_type'].unique().tolist()) if 'service_type' in tours_df.columns and tours_df['service_type'].any() else ["Vé máy bay", "Khách sạn", "Vận chuyển", "Ăn uống"]
    contract_status_options = sorted(tours_df['contract_status'].unique().tolist()) if 'contract_status' in tours_df.columns and tours_df['contract_status'].any() else ["Đang triển khai", "Sắp hết hạn", "Đã thanh lý"]
    payment_status_options = sorted(tours_df['payment_status'].unique().tolist()) if 'payment_status' in tours_df.columns and tours_df['payment_status'].any() else ["Trả trước", "Trả sau", "Chưa thanh toán"]

    # Khởi tạo giá trị cho Form
    selected_contract = ""
    mode_key = "default"
    revenue_val = 0
    profit_val = 0
    status_val = "Đã xác nhận"
    marketing_cost_val = 0
    sales_cost = 0
    
    partner_val = partner_options[0] if partner_options else "N/A"
    service_type_val = service_type_options[0] if service_type_options else "N/A"
    contract_status_val = contract_status_options[0] if contract_status_options else "N/A"
    payment_status_val = payment_status_options[0] if payment_status_options else "N/A"
    service_cost_val = 0
    
    
    if option == "Sửa Hợp đồng Hiện tại":
        # CHẾ ĐỘ SỬA
        contract_ids = tours_df['booking_id'].unique().tolist()
        if not contract_ids:
            st.warning("Không có hợp đồng nào để sửa.")
            st.stop()
            
        selected_contract = st.selectbox("Chọn Mã Hợp đồng để sửa", contract_ids)
        current_contract_data = tours_df[tours_df['booking_id'] == selected_contract].iloc[0]
        mode_key = f"edit_{selected_contract}"
        
        # Lấy giá trị hiện tại từ tours_df
        revenue_val = int(current_contract_data['revenue'])
        profit_val = int(current_contract_data['gross_profit'])
        status_val = current_contract_data['status']
        marketing_cost_val = int(current_contract_data['marketing_cost'])
        sales_cost = float(current_contract_data['sales_cost']) 
        
        # Lấy giá trị đối tác (FIX LỖI: Đảm bảo giá trị hiện tại có trong options)
        partner_val = current_contract_data.get('partner', partner_options[0])
        if partner_val not in partner_options: partner_options.append(partner_val)
        
        service_type_val = current_contract_data.get('service_type', service_type_options[0])
        if service_type_val not in service_type_options: service_type_options.append(service_type_val)
        
        contract_status_val = current_contract_data.get('contract_status', contract_status_options[0])
        if contract_status_val not in contract_status_options: contract_status_options.append(contract_status_val)
        
        payment_status_val = current_contract_data.get('payment_status', payment_status_options[0])
        if payment_status_val not in payment_status_options: payment_status_options.append(payment_status_val)
        
        service_cost_val = current_contract_data.get('service_cost', revenue_val - profit_val)
        
    else:
        # CHẾ ĐỘ NHẬP MỚI
        new_id = f"NEW{datetime.now().strftime('%d%H%M%S')}"
        selected_contract = new_id
        st.text_input("Mã Hợp đồng Mới", value=selected_contract, disabled=True)
        mode_key = "new_contract"
        
        # Giá trị mặc định cho hợp đồng mới
        revenue_val = 15000000
        profit_val = 3000000
        status_val = "Đã xác nhận"
        marketing_cost_val = 150000
        sales_cost = 0
        service_cost_val = revenue_val - profit_val


    # FORM NHẬP LIỆU CHUNG
    with st.container(border=True):
        st.subheader(f"Dữ liệu {selected_contract}")
        
        with st.form(key=mode_key):
            
            # CÁC CỘT ĐƯỢC CHIA LÀM 2 CỘT NHỎ HƠN
            col_a, col_b = st.columns(2)
            
            # ----------------------------------------------------
            # CỘT A: THÔNG TIN ĐỐI TÁC & TOUR CƠ BẢN
            # ----------------------------------------------------
            with col_a:
                st.markdown("##### 1. Thông tin Đối tác & Tour")
                
                # Thông tin Đối tác/Dịch vụ
                input_partner = st.selectbox("Tên Đối tác", options=partner_options, index=partner_options.index(partner_val), key=f"{mode_key}_partner")
                input_service_type = st.selectbox("Loại Dịch vụ", options=service_type_options, index=service_type_options.index(service_type_val), key=f"{mode_key}_service_type")
                
                input_contract_status = st.selectbox("Trạng thái HĐ", options=contract_status_options, index=contract_status_options.index(contract_status_val), key=f"{mode_key}_contract_status")
                input_payment_status = st.selectbox("Tình trạng TT", options=payment_status_options, index=payment_status_options.index(payment_status_val), key=f"{mode_key}_payment_status")
                
                # Thông tin Tour (Chỉ chỉnh sửa ở chế độ nhập mới)
                if option == "Nhập Hợp đồng Mới":
                    new_customer_id = st.text_input("Mã Khách hàng", value=f"KH_A{random.randint(1000, 9999)}", key="new_cust_id")
                    new_route = st.selectbox("Tuyến Tour", options=unique_route, key="new_route")
                    new_unit = st.selectbox("Đơn vị Kinh doanh", options=unique_unit, key="new_unit")
                    new_customers_count = st.number_input("Số lượng Khách", value=4, min_value=1, key="new_cust_count")
                else:
                    st.text_input("Mã Khách hàng", value=current_contract_data['customer_id'], disabled=True)
                    st.text_input("Tuyến Tour", value=current_contract_data['route'], disabled=True)
                    st.text_input("Đơn vị KD", value=current_contract_data['business_unit'], disabled=True)
                    st.number_input("Số lượng Khách", value=int(current_contract_data['num_customers']), min_value=1, disabled=True)


            # ----------------------------------------------------
            # CỘT B: DỮ LIỆU TÀI CHÍNH
            # ----------------------------------------------------
            with col_b:
                st.markdown("##### 2. Dữ liệu Tài chính")
                
                input_revenue = st.number_input("Doanh thu (₫)", value=revenue_val, min_value=0, step=100000, key=f"{mode_key}_rev")
                input_profit = st.number_input("Lợi nhuận gộp (₫)", value=profit_val, min_value=0, step=100000, key=f"{mode_key}_profit")
                
                input_service_cost = st.number_input("Chi phí Dịch vụ (service_cost)", 
                                                     value=int(service_cost_val), 
                                                     min_value=0, step=100000, 
                                                     key=f"{mode_key}_service_cost")

                input_marketing_cost = st.number_input("Chi phí Marketing (₫)", value=marketing_cost_val, min_value=0, step=100000, key=f"{mode_key}_mkt")
                input_status = st.selectbox("Trạng thái Booking", options=["Đã xác nhận", "Đã hủy", "Hoãn"], index=["Đã xác nhận", "Đã hủy", "Hoãn"].index(status_val), key=f"{mode_key}_status")

            # --- NÚT SUBMIT (Phải nằm ngoài col_a, col_b nhưng trong form) ---
            submitted = st.form_submit_button("Lưu & Cập nhật Dashboard", type="primary")

            if submitted:
                # LÔ-GÍC CẬP NHẬT/THÊM MỚI
                
                # 1. Xác định giá trị cuối cùng cho các trường
                if option == "Nhập Hợp đồng Mới":
                    sales_cost_final = input_revenue * 0.05
                    num_cust_final = new_customers_count
                    
                    partner_final = input_partner
                    service_type_final = input_service_type
                    contract_status_final = input_contract_status
                    payment_status_final = input_payment_status
                    route_final = new_route
                    unit_final = new_unit
                    customer_id_final = new_customer_id
                    
                else:
                    sales_cost_final = sales_cost
                    num_cust_final = current_contract_data['num_customers']
                    
                    partner_final = input_partner
                    service_type_final = input_service_type
                    contract_status_final = input_contract_status
                    payment_status_final = input_payment_status
                    route_final = current_contract_data['route']
                    unit_final = current_contract_data['business_unit']
                    customer_id_final = current_contract_data['customer_id']
                
                
                new_opex = input_marketing_cost + sales_cost_final
                
                if input_revenue > 0:
                    price_per_person_final = input_revenue / num_cust_final if num_cust_final > 0 else input_revenue
                    margin_final = (input_profit / input_revenue) * 100
                else:
                    price_per_person_final = 0
                    margin_final = 0
                    
                # 2. Xây dựng Row mới (bao gồm các cột mới cho Đối tác/Dịch vụ)
                new_row = {
                    # Cột cần thiết cho Tab 3 (Đối tác)
                    'partner': partner_final,
                    'service_type': service_type_final,
                    'contract_status': contract_status_final,
                    'payment_status': payment_status_final,
                    'service_cost': input_service_cost, 
                    'feedback_ratio': np.random.uniform(0.7, 0.95), # Giả định giá trị phản hồi
                    
                    # Cột cần thiết cho Tab 1 & 2 (Tour chính)
                    'booking_id': selected_contract,
                    'customer_id': customer_id_final,
                    'booking_date': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                    'route': route_final,
                    'business_unit': unit_final,
                    'sales_channel': current_contract_data['sales_channel'] if option == 'Sửa Hợp đồng Hiện tại' else unique_channel[0],
                    'segment': current_contract_data['segment'] if option == 'Sửa Hợp đồng Hiện tại' else unique_segment[0],
                    'num_customers': num_cust_final,
                    'tour_capacity': current_contract_data['tour_capacity'] if option == 'Sửa Hợp đồng Hiện tại' else 30,
                    'price_per_person': price_per_person_final,
                    'revenue': input_revenue,
                    'cost': input_revenue - input_profit,
                    'gross_profit': input_profit,
                    'gross_profit_margin': margin_final,
                    'status': input_status,
                    'marketing_cost': input_marketing_cost,
                    'sales_cost': sales_cost_final,
                    'opex': new_opex
                }
                
                # 3. Thêm/Sửa Row vào DataFrame
                if option == "Sửa Hợp đồng Hiện tại":
                    idx = st.session_state.tours_df[st.session_state.tours_df['booking_id'] == selected_contract].index[0]
                    for key, val in new_row.items():
                        st.session_state.tours_df.loc[idx, key] = val
                    st.success(f"✅ Hợp đồng {selected_contract} đã được cập nhật!")
                else:
                    new_df = pd.DataFrame([new_row])
                    # Đồng bộ hóa các cột mới vào tours_df gốc nếu chúng chưa tồn tại
                    for col in new_df.columns:
                        if col not in st.session_state.tours_df.columns:
                            st.session_state.tours_df[col] = np.nan
                            
                    st.session_state.tours_df = pd.concat([st.session_state.tours_df, new_df], ignore_index=True)
                    st.success(f"✅ Đã thêm Hợp đồng MỚI: {selected_contract}!")
                
                st.session_state.show_admin_ui = False
                st.rerun()

    st.stop()