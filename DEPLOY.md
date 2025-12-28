# Hướng dẫn Deploy lên GitHub và Streamlit Cloud

## Bước 1: Tạo repository trên GitHub

1. Đăng nhập vào [GitHub](https://github.com)
2. Click vào dấu "+" ở góc trên bên phải → "New repository"
3. Đặt tên repository (ví dụ: `tour-dashboard-ver2`)
4. Chọn Public hoặc Private
5. **KHÔNG** tích vào "Initialize this repository with a README" (vì đã có code local)
6. Click "Create repository"

## Bước 2: Push code lên GitHub

Sau khi tạo repository, GitHub sẽ hiển thị hướng dẫn. Chạy các lệnh sau (thay `<your-username>` và `<repository-name>` bằng thông tin của bạn):

```bash
# Thêm remote repository
git remote add origin https://github.com/<your-username>/<repository-name>.git

# Đổi tên branch từ master sang main (nếu cần)
git branch -M main

# Push code lên GitHub
git push -u origin main
```

Hoặc nếu bạn đã có SSH key setup:
```bash
git remote add origin git@github.com:<your-username>/<repository-name>.git
git branch -M main
git push -u origin main
```

## Bước 3: Deploy lên Streamlit Cloud

1. Đăng nhập vào [Streamlit Cloud](https://streamlit.io/cloud) bằng tài khoản GitHub
2. Click vào "New app"
3. Chọn repository vừa tạo từ danh sách
4. Chọn branch: `main` (hoặc `master`)
5. Main file path: `app.py`
6. Click "Deploy!"

## Bước 4: Cấu hình (nếu cần)

- Streamlit Cloud sẽ tự động cài đặt dependencies từ `requirements.txt`
- Nếu cần thay đổi URL Google Sheets, có thể cấu hình trong sidebar của ứng dụng sau khi deploy

## Lưu ý

- Đảm bảo Google Sheets có quyền truy cập công khai (public) hoặc cấu hình quyền truy cập phù hợp
- Streamlit Cloud sẽ tự động rebuild khi bạn push code mới lên GitHub

