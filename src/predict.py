
import pandas as pd
import joblib
from pathlib import Path


# ==============================
# 1. Đọc model đã huấn luyện
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"

model = joblib.load(MODEL_PATH)


# ==============================
# 2. Tạo dữ liệu căn nhà mới
# ==============================

house = {
    "name": "Nhà đẹp Quận 1",
    "description": "Nhà đẹp, vị trí trung tâm, gần trường học và chợ",

    "property_type_name": "Nhà riêng",
    "province_name": "Hồ Chí Minh",
    "district_name": "Quận 1",
    "ward_name": "Phường Bến Nghé",
    "street_name": "Nguyễn Huệ",
    "project_name": "Không rõ",

    "area": 80,
    "floor_count": 3,
    "frontage_width": 5,
    "house_depth": 16,
    "road_width": 8,

    "bedroom_count": 3,
    "bathroom_count": 2,

    "house_direction": "Đông",
    "balcony_direction": "Đông",

    "total_rooms": 5,
    "area_per_room": 80 / 5,

    "published_year": 2026,
    "published_month": 9,
    "published_dayofweek": 0,

    "description_length": len(
        "Nhà đẹp, vị trí trung tâm, gần trường học và chợ"
    ),

    "name_length": len(
        "Nhà đẹp Quận 1"
    )
}


# ==============================
# 3. Chuyển thành DataFrame
# ==============================

X_new = pd.DataFrame([house])


# ==============================
# 4. Dự đoán giá
# ==============================

predicted_price = model.predict(X_new)[0]


# ==============================
# 5. Hiển thị kết quả
# ==============================

print("===================================")
print("          DỰ ĐOÁN GIÁ NHÀ")
print("===================================")

print(
    f"Giá dự đoán: {predicted_price:,.0f} VNĐ"
)

print(
    f"Khoảng: {predicted_price / 1e9:.2f} tỷ đồng"
)

print("===================================")

