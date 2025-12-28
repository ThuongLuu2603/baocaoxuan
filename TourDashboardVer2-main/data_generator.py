"""
Data generator for Vietravel Business Intelligence Dashboard
Generates realistic mock data for tour sales, customers, and operations

Clean data generator for Vietravel dashboard.

Provides:
- VietravelDataGenerator: mock data generators
- load_or_generate_data(spreadsheet_url=None): loads a public Google Sheet (CSV export)
  mapping columns E,F,G,I,J,P,Q,R,S into the tours dataset. Falls back to mock data.
"""

import io
import random
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from faker import Faker

# Initialize Faker with Vietnamese locale
fake = Faker(['vi_VN'])


class VietravelDataGenerator:
    """Generates realistic mock data for Vietravel tour business"""

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)

        # Small sets used for realistic generation
        self.tour_routes = [
            "DH & ĐBSH", "Nam Trung Bộ", "Bắc Trung Bộ", "Liên Tuyến miền Tây",
            "Phú Quốc", "Thái Lan", "Trung Quốc", "Hàn Quốc", "Singapore - Malaysia",
            "Nhật Bản", "Châu Âu", "Châu Mỹ", "Châu Úc", "Châu Phi", "Tây Bắc",
            "Đông Bắc", "Tây Nguyên"
        ]

        self.business_units = ["Miền Trung", "Miền Tây", "Miền Bắc", "Trụ sở & ĐNB"]
        self.sales_channels = ["Online", "Trực tiếp VPGD", "Đại lý"]
        self.segments = ["FIT", "GIT", "Inbound"]

        self.partners = [
            ("Khách sạn A", "Khách sạn"), ("Khách sạn B", "Khách sạn"), ("Khách sạn C", "Khách sạn"),
            ("Hàng không X", "Vé máy bay"), ("Hàng không Y", "Vé máy bay"),
            ("Vận chuyển 1", "Vận chuyển"), ("Vận chuyển 2", "Vận chuyển"),
            ("Nhà hàng A", "Ăn uống"), ("Nhà hàng B", "Ăn uống"),
        ]

        # route -> possible units (simple mapping)
        self.route_unit_pairs = [(r, random.choice(self.business_units)) for r in self.tour_routes]

    def generate_tour_data(self, start_date, end_date, num_tours=1500):
        tours = []
        num_customers = int(num_tours * 0.7)
        customer_ids = [f"KH{i:06d}" for i in range(1, num_customers + 1)]

        for i in range(num_tours):
            booking_date = fake.date_time_between(start_date=start_date, end_date=end_date).replace(tzinfo=None)
            route, business_unit = random.choice(self.route_unit_pairs)
            channel = random.choices(self.sales_channels, weights=[0.35, 0.40, 0.25])[0]
            segment = random.choices(self.segments, weights=[0.35, 0.55, 0.10])[0]

            if random.random() < 0.3:
                num_customers_in_booking = random.randint(2, 4)
            else:
                num_customers_in_booking = random.randint(5, 20)

            tour_capacity = random.choice([20, 25, 30, 35, 40, 45])
            price_per_person = random.randint(3000000, 15000000)
            revenue = price_per_person * num_customers_in_booking
            cost_ratio = random.uniform(0.85, 0.95)
            cost = revenue * cost_ratio
            gross_profit = revenue - cost
            gross_profit_margin = (gross_profit / revenue * 100) if revenue > 0 else 0

            status = random.choices(["Đã xác nhận", "Đã hủy", "Hoãn"], weights=[0.75, 0.15, 0.10])[0]
            customer_id = random.choice(customer_ids)

            if channel == "Online":
                marketing_cost = revenue * random.uniform(0.02, 0.05)
                sales_cost = revenue * random.uniform(0.01, 0.02)
            else:
                marketing_cost = revenue * random.uniform(0.01, 0.03)
                sales_cost = revenue * random.uniform(0.02, 0.05)

            opex = marketing_cost + sales_cost
            partner_name, partner_type = random.choice(self.partners)

            tours.append({
                'booking_id': f"BK{i+1:06d}",
                'customer_id': customer_id,
                'booking_date': booking_date,
                'route': route,
                'business_unit': business_unit,
                'sales_channel': channel,
                'segment': segment,
                'num_customers': num_customers_in_booking,
                'tour_capacity': tour_capacity,
                'price_per_person': price_per_person,
                'revenue': revenue,
                'cost': cost,
                'gross_profit': gross_profit,
                'gross_profit_margin': gross_profit_margin,
                'status': status,
                'marketing_cost': marketing_cost,
                'sales_cost': sales_cost,
                'opex': opex,
                'partner': partner_name,
                'partner_type': partner_type,
                'service_type': partner_type,
                'contract_status': random.choice(["Đang triển khai", "Sắp hết hạn", "Đã thanh lý"]),
                'payment_status': random.choice(["Trả trước", "Trả sau", "Chưa thanh toán"]),
                'feedback_ratio': random.uniform(0.7, 0.95),
                'customer_age_group': random.choice(["18-25 (Gen Z)", "26-35 (Young Pro)", "36-55 (Mid-Career)", "56+ (Retiree)"]),
                'customer_nationality': random.choice(["Việt Nam", "Hàn Quốc", "Trung Quốc"]),
                'service_cost': cost * random.uniform(0.8, 1.2)
            })

        return pd.DataFrame(tours)

    def generate_plan_data(self, year, month=None):
        plans = []
        if month:
            periods = [(year, month)]
        else:
            periods = [(year, m) for m in range(1, 13)]

        for y, m in periods:
            for bu in self.business_units:
                for route in self.tour_routes:
                    for seg in self.segments:
                        seasonality = random.uniform(0.8, 1.2)
                        base_customers = random.randint(5, 20)
                        planned_customers = int(base_customers * seasonality)
                        avg_price = random.randint(3000000, 15000000)
                        planned_revenue = planned_customers * avg_price
                        plans.append({
                            'year': y,
                            'month': m,
                            'business_unit': bu,
                            'route': route,
                            'segment': seg,
                            'planned_customers': planned_customers,
                            'planned_revenue': planned_revenue,
                            'planned_gross_profit': planned_revenue * 0.2
                        })
        return pd.DataFrame(plans)

    def generate_historical_data(self, current_date, lookback_years=2):
        all_data = []
        for year_offset in range(lookback_years + 1):
            year_start = datetime(current_date.year - year_offset, 1, 1)
            year_end = datetime(current_date.year - year_offset, 12, 31)
            num_tours = random.randint(400, 600)
            all_data.append(self.generate_tour_data(year_start, year_end, num_tours=num_tours))
        return pd.concat(all_data, ignore_index=True)


# -- Helpers for Google Sheet parsing --
def _col_index(letter):
    """Convert Excel column letter to 0-based index (A->0)."""
    s = 0
    for ch in letter.upper():
        s = s * 26 + (ord(ch) - ord('A') + 1)
    return s - 1


def _parse_number(val):
    try:
        if pd.isna(val):
            return None
        if isinstance(val, str):
            v = val.strip()
            # Remove currency symbols and parentheses (handle negatives)
            v = v.replace('₫', '').replace('$', '').replace('đ', '').replace('Đ', '')
            is_negative = False
            if v.startswith('(') and v.endswith(')'):
                is_negative = True
                v = v[1:-1]

            # remove spaces
            v = v.replace(' ', '')

            # If both separators present, infer decimal vs thousands by position
            has_dot = '.' in v
            has_comma = ',' in v

            if has_dot and has_comma:
                # If comma appears after dot, assume comma is decimal separator (VN locale uses '.' as thousands)
                if v.rfind(',') > v.rfind('.'):
                    v = v.replace('.', '')
                    v = v.replace(',', '.')
                else:
                    v = v.replace(',', '')
            elif has_comma and not has_dot:
                # Single comma: decide by digits after comma
                parts = v.split(',')
                if len(parts[-1]) == 3:
                    # likely thousands separator
                    v = v.replace(',', '')
                else:
                    v = v.replace(',', '.')
            elif has_dot and not has_comma:
                parts = v.split('.')
                if len(parts[-1]) == 3:
                    # likely thousands separator
                    v = v.replace('.', '')
                else:
                    # dot as decimal
                    pass

            if v == '' or v == '-':
                return None
            num = float(v)
            return -num if is_negative else num
        return float(val)
    except Exception:
        return None


def _get_spreadsheet_id(url):
    try:
        return url.split('/d/')[1].split('/')[0]
    except Exception:
        return None


def load_or_generate_data(spreadsheet_url=None, plan_spreadsheet_url=None):
    """Load data from Google Sheet (public) or generate mock data.

    Returns: (tours_df, plans_df, historical_df, meta)
    meta contains keys: used_sheet (bool), processed_files (list), parsed_rows (int), parsed_counts (dict)
    """
    generator = VietravelDataGenerator()
    current_date = datetime.now()
    current_year = current_date.year

    tours_records = []
    parsed_counts = {}
    processed_files = []
    processed_plan_files = []

    if spreadsheet_url:
        sheet_id = _get_spreadsheet_id(spreadsheet_url)
        if sheet_id:
            # Try to parse gid if present in URL
            gid = None
            if 'gid=' in spreadsheet_url:
                try:
                    # Handle cases like ...edit?gid=123#gid=123 or additional params
                    gid_part = spreadsheet_url.split('gid=')[1]
                    gid = gid_part.split('&')[0].split('#')[0]
                except Exception:
                    gid = None
            gid = gid or '0'
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

            try:
                resp = requests.get(csv_url, timeout=15)
                resp.raise_for_status()
                # The Google Sheet provided by the user uses header on row 2 (1-based),
                # so tell pandas to use the second line as header (header=1).
                # Read as strings to preserve original formatting (thousands separators like '24.000')
                # and let _parse_number handle numeric conversion robustly.
                df = pd.read_csv(io.StringIO(resp.content.decode('utf-8')), header=1, dtype=str)
                processed_files.append(csv_url)

                # Column positions (0-based) from Excel letters provided by user
                # E=4, F=5, G=6, I=8, J=9, P=15, Q=16, R=17, S=18, T=19
                idx_map = {
                    'E': _col_index('E'),
                    'F': _col_index('F'),
                    'G': _col_index('G'),
                    'I': _col_index('I'),
                    'J': _col_index('J'),
                    'P': _col_index('P'),
                    'Q': _col_index('Q'),
                    'R': _col_index('R'),
                    'S': _col_index('S'),
                    'T': _col_index('T'),
                    'U': _col_index('U'),
                }

                per_file_count = 0
                # Detect a status-like column by header name (case-insensitive).
                # Many sheets don't provide a status column; when absent we default
                # to 'Đã xác nhận' to preserve previous behavior.
                import re
                status_idx = None
                for cidx, cname in enumerate(df.columns):
                    try:
                        if isinstance(cname, str) and re.search(r'status|trạng|trang thai|tình trạng|tinh trang', cname, re.I):
                            status_idx = cidx
                            break
                    except Exception:
                        continue
                # Iterate using iloc to access by positional column index
                for _, row in df.iterrows():
                    try:
                        # use iloc to be robust against header names
                        # Parse dates coming from Google Sheet which are in dd/mm/yyyy format
                        # Use dayfirst=True to correctly interpret day/month/year values
                        start_raw = row.iloc[idx_map['E']]
                        end_raw = row.iloc[idx_map['F']]
                        start_date = pd.to_datetime(start_raw, dayfirst=True, errors='coerce') if pd.notna(start_raw) else None
                        end_date = pd.to_datetime(end_raw, dayfirst=True, errors='coerce') if pd.notna(end_raw) else None
                        num_booked = int(_parse_number(row.iloc[idx_map['G']]) or 0)
                        # Parse numeric values from tours sheet (assume they are full VND already)
                        revenue = float(_parse_number(row.iloc[idx_map['I']]) or 0)
                        gross_profit = float(_parse_number(row.iloc[idx_map['J']]) or 0)
                        route = str(row.iloc[idx_map['P']]).strip() if pd.notna(row.iloc[idx_map['P']]) else None
                        business_unit = str(row.iloc[idx_map['Q']]).strip() if pd.notna(row.iloc[idx_map['Q']]) else None
                        tour_capacity = int(_parse_number(row.iloc[idx_map['R']]) or 0)
                        segment = str(row.iloc[idx_map['S']]).strip() if pd.notna(row.iloc[idx_map['S']]) else None

                        if not route or not business_unit:
                            # skip incomplete rows
                            continue

                        per_file_count += 1
                        booking_id = f"GS{per_file_count:06d}"
                        customer_id = f"KHG{per_file_count:06d}"
                        booking_date = start_date if start_date is not None else datetime.now()

                        price_per_person = revenue / num_booked if num_booked > 0 else 0
                        cost = revenue - gross_profit if (revenue and gross_profit) else revenue * 0.8
                        gross_profit_margin = (gross_profit / revenue * 100) if revenue > 0 else 0

                        # If the sheet contains a sales channel in column T, prefer that value
                        try:
                            sales_channel_val = row.iloc[idx_map.get('T')]
                        except Exception:
                            sales_channel_val = None

                        # When loading from Google Sheet, prefer ONLY the sheet value.
                        # Do not fall back to generator values for sales_channel.
                        if pd.notna(sales_channel_val) and str(sales_channel_val).strip() != '':
                            sales_channel = str(sales_channel_val).strip()
                        else:
                            # If sheet has no value, mark as Unknown rather than using generator defaults
                            sales_channel = 'Unknown'

                        # Parse cancellation count from column U if present.
                        cancel_count = 0
                        cancel_rate = 0.0
                        try:
                            if 'U' in idx_map:
                                raw_cancel = row.iloc[idx_map.get('U')]
                                if pd.notna(raw_cancel):
                                    # Users store number of cancelled seats (e.g. '2' meaning 2 customers cancelled)
                                    parsed_c = _parse_number(str(raw_cancel).strip())
                                    if parsed_c is None:
                                        parsed_c = 0
                                    cancel_count = int(round(parsed_c))
                        except Exception:
                            cancel_count = 0

                        # Compute cancel_rate relative to booked seats when possible
                        try:
                            if num_booked and num_booked > 0:
                                cancel_rate = max(0.0, min(1.0, float(cancel_count) / float(num_booked)))
                            else:
                                cancel_rate = 0.0
                        except Exception:
                            cancel_rate = 0.0

                        # If the sheet contains a status column (detected above), use it and
                        # normalize common Vietnamese variants; otherwise default to
                        # 'Đã xác nhận' to preserve existing behavior when the sheet lacks status.
                        if status_idx is not None:
                            try:
                                raw_status = row.iloc[status_idx]
                                if pd.notna(raw_status):
                                    st = str(raw_status).strip()
                                    # normalize common words
                                    if re.search(r'hủy|huy', st, re.I):
                                        status = 'Đã hủy'
                                    elif re.search(r'hoãn|hoan', st, re.I):
                                        status = 'Hoãn'
                                    elif re.search(r'xác|xac', st, re.I):
                                        status = 'Đã xác nhận'
                                    else:
                                        status = st
                                else:
                                    status = 'Đã xác nhận'
                            except Exception:
                                status = 'Đã xác nhận'
                        else:
                            status = 'Đã xác nhận'

                        if sales_channel == "Online":
                            marketing_cost = revenue * random.uniform(0.02, 0.05)
                            sales_cost = revenue * random.uniform(0.01, 0.02)
                        else:
                            marketing_cost = revenue * random.uniform(0.01, 0.03)
                            sales_cost = revenue * random.uniform(0.02, 0.04)

                        opex = marketing_cost + sales_cost

                        partner_name, partner_type = random.choice(generator.partners)
                        service_type = partner_type
                        contract_status = random.choices(["Đang triển khai", "Sắp hết hạn", "Đã thanh lý"], weights=[0.8, 0.1, 0.1])[0]
                        payment_status = random.choices(["Trả trước", "Trả sau", "Chưa thanh toán"], weights=[0.6, 0.3, 0.1])[0]
                        feedback_ratio = random.uniform(0.7, 0.95)
                        service_cost = cost * random.uniform(0.8, 1.2)

                        # compute effective counts/values after cancellations
                        num_customers_effective = int(round(max(0, num_booked - cancel_count)))
                        revenue_effective = float(revenue) * (max(0.0, (num_customers_effective / num_booked)) if num_booked > 0 else 0.0)
                        gross_profit_effective = float(gross_profit) * (max(0.0, (num_customers_effective / num_booked)) if num_booked > 0 else 0.0)

                        # If all seats canceled, mark as canceled
                        if cancel_count >= num_booked and num_booked > 0:
                            final_status = 'Đã hủy'
                        else:
                            final_status = status

                        tours_records.append({
                            'booking_id': booking_id,
                            'customer_id': customer_id,
                            'booking_date': booking_date,
                            'end_date': end_date,
                            'route': route,
                            'business_unit': business_unit,
                            'sales_channel': sales_channel,
                            'segment': segment,
                            'num_customers': int(num_booked),
                            'cancel_count': int(cancel_count),
                            'cancel_rate': float(cancel_rate),
                            'num_customers_effective': int(num_customers_effective),
                            'revenue_effective': float(revenue_effective),
                            'gross_profit_effective': float(gross_profit_effective),
                            'tour_capacity': int(tour_capacity),
                            'price_per_person': float(price_per_person),
                            'revenue': float(revenue),
                            'cost': float(cost),
                            'gross_profit': float(gross_profit),
                            'gross_profit_margin': float(gross_profit_margin),
                            'status': final_status,
                            'marketing_cost': float(marketing_cost),
                            'sales_cost': float(sales_cost),
                            'opex': float(opex),
                            'partner': partner_name,
                            'partner_type': partner_type,
                            'service_type': service_type,
                            'contract_status': contract_status,
                            'payment_status': payment_status,
                            'feedback_ratio': feedback_ratio,
                            'customer_age_group': random.choice(["18-25 (Gen Z)", "26-35 (Young Pro)", "36-55 (Mid-Career)", "56+ (Retiree)"]),
                            'customer_nationality': random.choice(["Việt Nam", "Hàn Quốc", "Trung Quốc"]),
                            'service_cost': float(service_cost)
                        })

                    except Exception:
                        # Skip problematic row but continue
                        continue

                parsed_counts[csv_url] = per_file_count

            except Exception:
                # If fetching/reading sheet failed, fall back to generator
                pass
    # --- Load plan sheet if provided ---
    plans_records = []
    if plan_spreadsheet_url:
        plan_sheet_id = _get_spreadsheet_id(plan_spreadsheet_url)
        if plan_sheet_id:
            # try to get gid if present
            gid = None
            if 'gid=' in plan_spreadsheet_url:
                try:
                    gid_part = plan_spreadsheet_url.split('gid=')[1]
                    gid = gid_part.split('&')[0].split('#')[0]
                except Exception:
                    gid = None
            gid = gid or '0'
            plan_csv_url = f"https://docs.google.com/spreadsheets/d/{plan_sheet_id}/export?format=csv&gid={gid}"
            try:
                resp = requests.get(plan_csv_url, timeout=15)
                resp.raise_for_status()
                # header on row 2 (header=1)
                # Read as strings so original cell formatting (e.g. '24.000' meaning 24.000 millions)
                # is preserved and parsed correctly by _parse_number.
                dfp = pd.read_csv(io.StringIO(resp.content.decode('utf-8')), header=1, dtype=str)
                processed_plan_files.append(plan_csv_url)

                # --- First: parse company-level totals in columns A-D (indices 0-3) if present ---
                try:
                    unit_name_company = str(dfp.columns[0]).strip()
                except Exception:
                    unit_name_company = 'TOÀN CÔNG TY'

                # Look for 'Tháng' rows in the first block A-D
                for ridx in dfp.index:
                    try:
                        label = str(dfp.iat[ridx, 0]).strip()
                    except Exception:
                        label = ''
                    if not label:
                        continue
                    if label.lower().startswith('tháng'):
                        import re
                        m = re.search(r"(\d{1,2})", label)
                        if not m:
                            continue
                        month = int(m.group(1))
                        # Customers in the sheet may use dot as thousands separator (e.g. 58.201 meaning 58,201).
                        raw_customers = dfp.iat[ridx, 1]
                        try:
                            # If pandas already parsed as float with fractional part, assume it's thousands format
                            if isinstance(raw_customers, (float, np.floating)) and (not float(raw_customers).is_integer()):
                                customers = int(round(float(raw_customers) * 1000))
                            else:
                                customers = int(_parse_number(raw_customers) or 0)
                        except Exception:
                            customers = int(_parse_number(raw_customers) or 0)

                        # Plan values in the sheet are in millions -> scale up
                        # revenue/gross_profit cells may be strings like '488.737' (dot as thousands sep)
                        # or parsed by pandas as floats (e.g. 488.737). If pandas gave a float with
                        # fractional part we assume dot was thousands separator and scale by 1000.
                        raw_revenue = dfp.iat[ridx, 2]
                        raw_profit = dfp.iat[ridx, 3]
                        try:
                            if isinstance(raw_revenue, (float, np.floating)) and (not float(raw_revenue).is_integer()):
                                revenue = float(raw_revenue) * 1000 * 1_000_000
                            else:
                                revenue = float(_parse_number(raw_revenue) or 0) * 1_000_000
                        except Exception:
                            revenue = float(_parse_number(raw_revenue) or 0) * 1_000_000
                        try:
                            if isinstance(raw_profit, (float, np.floating)) and (not float(raw_profit).is_integer()):
                                profit = float(raw_profit) * 1000 * 1_000_000
                            else:
                                profit = float(_parse_number(raw_profit) or 0) * 1_000_000
                        except Exception:
                            profit = float(_parse_number(raw_profit) or 0) * 1_000_000
                        plans_records.append({
                            'year': current_year,
                            'month': month,
                            'business_unit': unit_name_company if unit_name_company else 'TOÀN CÔNG TY',
                            'route': None,
                            'segment': None,
                            'planned_customers': int(customers),
                            'planned_revenue': float(revenue),
                            'planned_gross_profit': float(profit)
                        })
                        # --- Also parse following rows for segment-level breakdown (FIT / GIT / Nội địa / Nước ngoài / Khác)
                        # Continue reading subsequent rows until next 'Tháng' label or end of frame
                        try:
                            next_idx = ridx + 1
                            while next_idx in dfp.index:
                                try:
                                    next_label = str(dfp.iat[next_idx, 0]).strip()
                                except Exception:
                                    next_label = ''
                                if not next_label:
                                    next_idx += 1
                                    continue
                                # stop if we hit the next month header
                                if next_label.lower().startswith('tháng'):
                                    break
                                # treat this row as a segment/line item (e.g., FIT, GIT, Nội địa, Nước ngoài, Khác)
                                segment_name = next_label
                                raw_cust_s = dfp.iat[next_idx, 1]
                                try:
                                    if isinstance(raw_cust_s, (float, np.floating)) and (not float(raw_cust_s).is_integer()):
                                        seg_customers = int(round(float(raw_cust_s) * 1000))
                                    else:
                                        seg_customers = int(_parse_number(raw_cust_s) or 0)
                                except Exception:
                                    seg_customers = int(_parse_number(raw_cust_s) or 0)
                                raw_rev_s = dfp.iat[next_idx, 2]
                                raw_profit_s = dfp.iat[next_idx, 3]
                                try:
                                    if isinstance(raw_rev_s, (float, np.floating)) and (not float(raw_rev_s).is_integer()):
                                        seg_revenue = float(raw_rev_s) * 1000 * 1_000_000
                                    else:
                                        seg_revenue = float(_parse_number(raw_rev_s) or 0) * 1_000_000
                                except Exception:
                                    seg_revenue = float(_parse_number(raw_rev_s) or 0) * 1_000_000
                                try:
                                    if isinstance(raw_profit_s, (float, np.floating)) and (not float(raw_profit_s).is_integer()):
                                        seg_profit = float(raw_profit_s) * 1000 * 1_000_000
                                    else:
                                        seg_profit = float(_parse_number(raw_profit_s) or 0) * 1_000_000
                                except Exception:
                                    seg_profit = float(_parse_number(raw_profit_s) or 0) * 1_000_000

                                plans_records.append({
                                    'year': current_year,
                                    'month': month,
                                    'business_unit': unit_name_company if unit_name_company else 'TOÀN CÔNG TY',
                                    'route': None,
                                    'segment': segment_name,
                                    'planned_customers': int(seg_customers),
                                    'planned_revenue': float(seg_revenue),
                                    'planned_gross_profit': float(seg_profit)
                                })
                                next_idx += 1
                        except Exception:
                            # if anything goes wrong parsing segment rows, continue gracefully
                            pass
                    # Detect an annual TOTAL row like 'TOTAL 2025' and record as month==0 (annual)
                    elif 'total' in label.lower():
                        # Try to extract year from label, otherwise use current_year
                        import re
                        y_match = re.search(r'(20\d{2})', label)
                        y_row = int(y_match.group(1)) if y_match else current_year
                        raw_customers = dfp.iat[ridx, 1]
                        try:
                            if isinstance(raw_customers, (float, np.floating)) and (not float(raw_customers).is_integer()):
                                customers = int(round(float(raw_customers) * 1000))
                            else:
                                customers = int(_parse_number(raw_customers) or 0)
                        except Exception:
                            customers = int(_parse_number(raw_customers) or 0)

                        raw_revenue = dfp.iat[ridx, 2]
                        raw_profit = dfp.iat[ridx, 3]
                        try:
                            if isinstance(raw_revenue, (float, np.floating)) and (not float(raw_revenue).is_integer()):
                                revenue = float(raw_revenue) * 1000 * 1_000_000
                            else:
                                revenue = float(_parse_number(raw_revenue) or 0) * 1_000_000
                        except Exception:
                            revenue = float(_parse_number(raw_revenue) or 0) * 1_000_000
                        try:
                            if isinstance(raw_profit, (float, np.floating)) and (not float(raw_profit).is_integer()):
                                profit = float(raw_profit) * 1000 * 1_000_000
                            else:
                                profit = float(_parse_number(raw_profit) or 0) * 1_000_000
                        except Exception:
                            profit = float(_parse_number(raw_profit) or 0) * 1_000_000

                        plans_records.append({
                            'year': y_row,
                            'month': 0,
                            'business_unit': unit_name_company if unit_name_company else 'TOÀN CỘNG TY',
                            'route': None,
                            'segment': None,
                            'planned_customers': int(customers),
                            'planned_revenue': float(revenue),
                            'planned_gross_profit': float(profit)
                        })

                # Business units start from column E (index 4), each unit occupies 4 columns
                ncols = dfp.shape[1]
                col = _col_index('E')
                current_year = datetime.now().year
                while col < ncols:
                    try:
                        unit_name = str(dfp.columns[col]).strip() if col < ncols else None
                    except Exception:
                        unit_name = None
                    if not unit_name or unit_name.lower() in ['nan', 'unnamed: 4']:
                        col += 4
                        continue

                    # iterate rows to find 'Tháng ##' rows
                    for ridx in dfp.index:
                        try:
                            label = str(dfp.iat[ridx, col]).strip()
                        except Exception:
                            label = ''
                        if not label:
                            continue
                        # match Tháng NN (month row)
                        if label.lower().startswith('tháng'):
                            # extract month number
                            import re
                            m = re.search(r"(\d{1,2})", label)
                            if not m:
                                continue
                            month = int(m.group(1))
                            raw_customers = dfp.iat[ridx, col+1]
                            try:
                                if isinstance(raw_customers, (float, np.floating)) and (not float(raw_customers).is_integer()):
                                    customers = int(round(float(raw_customers) * 1000))
                                else:
                                    customers = int(_parse_number(raw_customers) or 0)
                            except Exception:
                                customers = int(_parse_number(raw_customers) or 0)
                            # Plan values in the sheet are in millions -> scale up
                            raw_revenue = dfp.iat[ridx, col+2]
                            raw_profit = dfp.iat[ridx, col+3]
                            try:
                                if isinstance(raw_revenue, (float, np.floating)) and (not float(raw_revenue).is_integer()):
                                    revenue = float(raw_revenue) * 1000 * 1_000_000
                                else:
                                    revenue = float(_parse_number(raw_revenue) or 0) * 1_000_000
                            except Exception:
                                revenue = float(_parse_number(raw_revenue) or 0) * 1_000_000
                            try:
                                if isinstance(raw_profit, (float, np.floating)) and (not float(raw_profit).is_integer()):
                                    profit = float(raw_profit) * 1000 * 1_000_000
                                else:
                                    profit = float(_parse_number(raw_profit) or 0) * 1_000_000
                            except Exception:
                                profit = float(_parse_number(raw_profit) or 0) * 1_000_000
                            plans_records.append({
                                'year': current_year,
                                'month': month,
                                'business_unit': unit_name,
                                'route': None,
                                'segment': None,
                                'planned_customers': int(customers),
                                'planned_revenue': float(revenue),
                                'planned_gross_profit': float(profit)
                            })
                        # ignore FIT/GIT/Khác rows for now (could be added later)
                    col += 4

            except Exception:
                # ignore plan sheet failures and leave plans_records empty
                plans_records = []

    # Build DataFrames
    if tours_records:
        tours_df = pd.DataFrame(tours_records)
        used_sheet = True
    else:
        year_start = datetime(current_year, 1, 1)
        year_end = current_date
        tours_df = generator.generate_tour_data(year_start, year_end, num_tours=1500)
        used_sheet = False

    # If we parsed plan records from provided plan sheet, build plans_df from it,
    # otherwise fall back to generated plans.
    if plans_records:
        plans_df = pd.DataFrame(plans_records)
    else:
        plans_df = generator.generate_plan_data(current_year)
    historical_df = generator.generate_historical_data(current_date, lookback_years=2)

    meta = {
        'used_sheet': used_sheet,
        'processed_files': processed_files,
        'parsed_rows': len(tours_records),
        'parsed_counts': parsed_counts,
        'processed_plan_files': processed_plan_files,
        'parsed_plan_rows': len(plans_records)
    }

    # --- Expand monthly plans into daily and weekly plans ---
    try:
        # Create empty DataFrames if plans_df is empty
        if plans_df is None or plans_df.empty:
            plans_daily_df = pd.DataFrame()
            plans_weekly_df = pd.DataFrame()
        else:
            from datetime import date, timedelta
            import calendar

            daily_records = []
            for _, prow in plans_df.iterrows():
                y = int(prow.get('year', current_year))
                m = int(prow.get('month', 1))
                # number of days in the month
                days_in_month = calendar.monthrange(y, m)[1]
                # per-day allocations
                pr_customers = float(prow.get('planned_customers') or 0) / days_in_month
                pr_revenue = float(prow.get('planned_revenue') or 0) / days_in_month
                pr_profit = float(prow.get('planned_gross_profit') or 0) / days_in_month

                start_dt = date(y, m, 1)
                for d in range(days_in_month):
                    dt = start_dt + timedelta(days=d)
                    daily_records.append({
                        'date': pd.Timestamp(dt),
                        'year': y,
                        'month': m,
                        'business_unit': prow.get('business_unit'),
                        'route': prow.get('route'),
                        'segment': prow.get('segment'),
                        'planned_customers_daily': pr_customers,
                        'planned_revenue_daily': pr_revenue,
                        'planned_gross_profit_daily': pr_profit
                    })

            plans_daily_df = pd.DataFrame(daily_records)
            if not plans_daily_df.empty:
                # compute week_start (Monday) for weekly aggregation
                plans_daily_df['week_start'] = plans_daily_df['date'].dt.to_pydatetime().astype('datetime64[ns]')
                plans_daily_df['week_start'] = plans_daily_df['date'].apply(lambda dt: (dt - timedelta(days=dt.weekday())).normalize())

                # aggregate weekly sums grouped by business_unit/route/segment and week_start
                plans_weekly_df = (
                    plans_daily_df
                    .groupby(['week_start', 'business_unit', 'route', 'segment'], dropna=False, as_index=False)
                    .agg(
                        planned_customers_week=('planned_customers_daily', 'sum'),
                        planned_revenue_week=('planned_revenue_daily', 'sum'),
                        planned_gross_profit_week=('planned_gross_profit_daily', 'sum')
                    )
                )
            else:
                plans_weekly_df = pd.DataFrame()
    except Exception:
        plans_daily_df = pd.DataFrame()
        plans_weekly_df = pd.DataFrame()

    # Attach expanded plans to meta so callers (app) can persist them in session state
    meta['plans_daily_df'] = plans_daily_df
    meta['plans_weekly_df'] = plans_weekly_df

    return tours_df, plans_df, historical_df, meta
