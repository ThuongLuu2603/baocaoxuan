# Tour Dashboard Ver2

Dashboard theo dõi kế hoạch và hiệu suất kinh doanh cho Vietravel Tour.

## Tính năng

- **VÙNG 1: Tốc độ đạt Kế hoạch**: Hiển thị mức độ hoàn thành kế hoạch theo đơn vị (Doanh Thu, Lãi Gộp)
- **Tốc độ đạt Kế hoạch theo Tuyến**: Biểu đồ hiển thị Lượt Khách, Doanh Thu, Lãi Gộp theo từng tuyến (Nội địa và Outbound)
- **TIẾN ĐỘ HOÀN THÀNH KẾ HOẠCH**: Biểu đồ bar chart theo dõi tiến độ hoàn thành kế hoạch
- **THEO DÕI SỐ CHỖ BÁN**: Biểu đồ kết hợp bar và line chart theo dõi số chỗ bán và doanh số

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd TourDashboardVer2-main
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Chạy ứng dụng:
```bash
streamlit run app.py
```

## Cấu trúc dữ liệu

Ứng dụng sử dụng Google Sheets làm nguồn dữ liệu:

- **DEFAULT_ROUTE_PERFORMANCE_URL**: Dữ liệu tốc độ đạt kế hoạch theo tuyến (DT và LG: triệu đồng)
- **DEFAULT_PLAN_TET_URL**: Kế hoạch tuyến Tết (DT và LG: triệu đồng)
- **DEFAULT_PLAN_XUAN_URL**: Kế hoạch tuyến Xuân (DT và LG: triệu đồng)
- **DEFAULT_ETOUR_SEATS_URL**: Dữ liệu theo dõi chỗ bán etour (Doanh số: VNĐ)

## Deploy lên Streamlit Cloud

1. Push code lên GitHub repository
2. Đăng nhập vào [Streamlit Cloud](https://streamlit.io/cloud)
3. Chọn "New app"
4. Kết nối với GitHub repository
5. Chọn branch và file chính: `app.py`
6. Deploy!

## Yêu cầu

- Python 3.8+
- Streamlit 1.50.0+
- Các dependencies trong `requirements.txt`

## Lưu ý

- Đảm bảo Google Sheets có quyền truy cập công khai (public) hoặc cấu hình quyền truy cập phù hợp
- Các URL Google Sheets có thể được thay đổi trong sidebar của ứng dụng
