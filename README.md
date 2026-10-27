# Hệ Thống Dự Đoán Giá Bất Động Sản (TP. Hồ Chí Minh)

Dự án nghiên cứu và xây dựng mô hình Machine Learning nhằm dự đoán giá bất động sản tại khu vực TP. Hồ Chí Minh dựa trên các đặc trưng về vị trí, diện tích, kết cấu nhà và mô tả chi tiết. Hệ thống cung cấp giao diện trực quan hóa dữ liệu và demo dự đoán qua ứng dụng Web Streamlit.

---

## 1. Cấu trúc thư mục dự án

```text
DO_AN_DU_BAO_GIA_NHA/
│
├── app/
│   ├── app.py                     # Ứng dụng web demo dự đoán (Streamlit)
│   └── style.css                  # Giao diện tùy chỉnh CSS
│
├── data/
│   ├── raw/
│   │   └── houses_hcm_10000_raw.csv    # Dữ liệu thô thu thập từ Hugging Face
│   └── processed/
│       └── houses_clean.csv            # Dữ liệu sau khi làm sạch và tiền xử lý
│
├── models/
│   └── house_price_model.pkl      # Pipeline mô hình Machine Learning tốt nhất (XGBoost)
│
├── notebooks/
│   ├── 01_kham_pha_du_lieu.ipynb        # Phân tích khám phá dữ liệu (EDA)
│   ├── 02_tien_xu_ly.ipynb              # Làm sạch và xử lý giá trị thiếu, ngoại lai
│   ├── 03_feature_engineering.ipynb     # Trích xuất và biến đổi đặc trưng
│   └── 04_train_model.ipynb             # Huấn luyện và so sánh các mô hình
│
├── reports/
│   ├── model_comparison.csv       # Bảng đánh giá chỉ số MAE, RMSE, R²
│   └── model_comparison_r2.png    # Biểu đồ trực quan so sánh R²
│
├── src/
│   ├── evaluate.py                # Kịch bản đánh giá mô hình
│   └── predict.py                 # Kịch bản chạy thử dự đoán trên dòng lệnh
│
├── prepare_dataset.py             # Script lấy mẫu và chuẩn bị dữ liệu ban đầu
├── requirements.txt               # Danh sách các thư viện cần thiết
├── .gitignore                     # Cấu hình bỏ qua các file thừa khi push Git
└── README.md                      # Tài liệu hướng dẫn đồ án
```

---

## 2. Kết quả huấn luyện mô hình

Hệ thống tiến hành so sánh 3 thuật toán hồi quy phổ biến:

| Mô hình | MAE (tỷ đồng) | RMSE (tỷ đồng) | R² Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 14.19 | 31.18 | 43.86% |
| **Random Forest** | 7.31 | 24.36 | 65.74% |
| **XGBoost (Được chọn)** | **6.85** | **23.11** | **69.15%** |

Mô hình **XGBoost** kết hợp kỹ thuật mã hóa One-Hot (vị trí, loại nhà, hướng) và TF-IDF (mô tả bất động sản) đạt hiệu năng tốt nhất với $R^2 \approx 69.15\%$.

---

## 3. Hướng dẫn cài đặt và khởi chạy

### Bước 1: Clone kho lưu trữ
```bash
git clone <URL_REPO_CUA_BAN>
cd DO_AN_DU_BAO_GIA_NHA
```

### Bước 2: Tạo và kích hoạt môi trường ảo
- **Trên Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **Trên macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 3: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 4: Khởi chạy ứng dụng Web Streamlit
```bash
streamlit run app/app.py
```
Sau khi chạy lệnh, trình duyệt sẽ tự động mở giao diện ứng dụng tại địa chỉ `http://localhost:8501`.

---

## 4. Kiểm tra dự đoán qua Command Line (Tùy chọn)
Nếu muốn kiểm tra nhanh mô hình mà không cần mở giao diện Web:
```bash
python src/predict.py
```

