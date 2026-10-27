import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import plotly.express as px



# CẤU HÌNH TRANG


st.set_page_config(
    page_title="Dự đoán giá bất động sản",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ĐƯỜNG DẪN


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "houses_clean.csv"
CSS_PATH = Path(__file__).resolve().parent / "style.css"


# LOAD CSS

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# LOAD MODEL


try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Không thể tải model: {e}")
    st.stop()

# LOAD DATA

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Không thể đọc dữ liệu: {e}")
    st.stop()


# TIÊU ĐỀ

st.markdown(
    """
    <div class="main-title">
        HỆ THỐNG DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN
    </div>

    <div class="subtitle">
        Hệ thống dự đoán giá bất động sản sử dụng Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)


# THÔNG TIN TỔNG QUAN

col1, col2, col3 = st.columns(3)


with col1:
    st.markdown(
        """
        <div class="info-box">
            <div class="info-number">9,999</div>
            <div class="info-label">Bản ghi dữ liệu</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:
    st.markdown(
        """
        <div class="info-box">
            <div class="info-number">69.15%</div>
            <div class="info-label">R² của mô hình</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:
    st.markdown(
        """
        <div class="info-box">
            <div class="info-number">XGBoost</div>
            <div class="info-label">Mô hình được chọn</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# THỐNG KÊ DỮ LIỆU

st.markdown(
    """
    <div class="card">
        <div class="card-title">
            Thống kê dữ liệu nhà đất
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# BIỂU ĐỒ 1: PHÂN BỐ GIÁ NHÀ

st.markdown(
    """
    <div class="section-title">
        Phân bố giá nhà
    </div>
    """,
    unsafe_allow_html=True
)

price_billion = df["price"] / 1e9

fig_price = px.histogram(
    x=price_billion,
    nbins=50,
    title="Phân bố giá nhà trong tập dữ liệu",
    labels={
        "x": "Giá nhà (tỷ đồng)",
        "y": "Số lượng tin đăng"
    }
)

fig_price.update_layout(
    height=420,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig_price,
    use_container_width=True
)


# BIỂU ĐỒ 2: DIỆN TÍCH VÀ GIÁ

st.markdown(
    """
    <div class="section-title">
        Mối quan hệ giữa diện tích và giá
    </div>
    """,
    unsafe_allow_html=True
)

plot_df = df[
    (df["area"] > 0) &
    (df["price"] > 0)
].copy()

plot_df["price_billion"] = plot_df["price"] / 1e9

fig_area = px.scatter(
    plot_df,
    x="area",
    y="price_billion",
    title="Mối quan hệ giữa diện tích và giá nhà",
    labels={
        "area": "Diện tích (m²)",
        "price_billion": "Giá nhà (tỷ đồng)"
    },
    opacity=0.5
)

fig_area.update_layout(
    height=450,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig_area,
    use_container_width=True
)


# BIỂU ĐỒ 3: GIÁ TRUNG BÌNH THEO QUẬN

st.markdown(
    """
    <div class="section-title">
        Giá nhà trung bình theo quận
    </div>
    """,
    unsafe_allow_html=True
)

district_price = (
    df.groupby("district_name")["price"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

district_price["price_billion"] = (
    district_price["price"] / 1e9
)

fig_district_price = px.bar(
    district_price,
    x="district_name",
    y="price_billion",
    title="Top 10 quận có giá nhà trung bình cao",
    labels={
        "district_name": "Quận",
        "price_billion": "Giá trung bình (tỷ đồng)"
    },
    text_auto=True
)

fig_district_price.update_traces(texttemplate="%{y:.2f}")

fig_district_price.update_layout(
    height=450,
    xaxis_tickangle=-30,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig_district_price,
    use_container_width=True
)


# BIỂU ĐỒ 4: SỐ LƯỢNG TIN ĐĂNG THEO QUẬN


st.markdown(
    """
    <div class="section-title">
        Số lượng tin đăng theo quận
    </div>
    """,
    unsafe_allow_html=True
)

district_count = (
    df["district_name"]
    .value_counts()
    .head(15)
    .reset_index()
)

district_count.columns = [
    "district_name",
    "count"
]

fig_district = px.bar(
    district_count,
    x="district_name",
    y="count",
    title="Top 15 quận có nhiều tin đăng nhất",
    labels={
        "district_name": "Quận",
        "count": "Số lượng tin đăng"
    },
    text_auto=True
)

fig_district.update_layout(
    height=450,
    xaxis_tickangle=-30,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig_district,
    use_container_width=True
)


# FORM DỰ ĐOÁN

st.markdown(
    """
    <div class="section-title">
        Dự đoán giá nhà
    </div>
    """,
    unsafe_allow_html=True
)


# THÔNG TIN CƠ BẢN

st.markdown(
    """
    <div class="card">
        <div class="card-title">
            Thông tin bất động sản
        </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:
    area = st.number_input(
        "Diện tích (m²)",
        min_value=1.0,
        value=80.0,
        step=1.0
    )

    bedroom_count = st.number_input(
        "Số phòng ngủ",
        min_value=0,
        value=3,
        step=1
    )

    bathroom_count = st.number_input(
        "Số phòng tắm",
        min_value=0,
        value=2,
        step=1
    )


with col2:
    floor_count = st.number_input(
        "Số tầng",
        min_value=1,
        value=3,
        step=1
    )

    frontage_width = st.number_input(
        "Mặt tiền (m)",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

    house_depth = st.number_input(
        "Chiều sâu nhà (m)",
        min_value=0.0,
        value=16.0,
        step=0.5
    )


with col3:
    road_width = st.number_input(
        "Đường trước nhà (m)",
        min_value=0.0,
        value=8.0,
        step=0.5
    )

    house_direction = st.selectbox(
        "Hướng nhà",
        [
            "Đông",
            "Tây",
            "Nam",
            "Bắc",
            "Đông Bắc",
            "Đông Nam",
            "Tây Bắc",
            "Tây Nam",
            "Không rõ"
        ]
    )

    balcony_direction = st.selectbox(
        "Hướng ban công",
        [
            "Đông",
            "Tây",
            "Nam",
            "Bắc",
            "Đông Bắc",
            "Đông Nam",
            "Tây Bắc",
            "Tây Nam",
            "Không rõ"
        ]
    )

st.markdown("</div>", unsafe_allow_html=True)


# THÔNG TIN ĐỊA CHỈ

st.markdown(
    """
    <div class="card">
        <div class="card-title">
            Thông tin vị trí
        </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:
    district_name = st.text_input(
        "Quận / Huyện",
        value="Quận 1"
    )

    ward_name = st.text_input(
        "Phường / Xã",
        value="Phường Bến Nghé"
    )


with col2:
    street_name = st.text_input(
        "Tên đường",
        value="Nguyễn Huệ"
    )

    property_type_name = st.text_input(
        "Loại bất động sản",
        value="Nhà riêng"
    )


with col3:
    project_name = st.text_input(
        "Tên dự án",
        value="Không rõ"
    )

st.markdown("</div>", unsafe_allow_html=True)



# THÔNG TIN MÔ TẢ

st.markdown(
    """
    <div class="card">
        <div class="card-title">
            Thông tin mô tả
        </div>
    """,
    unsafe_allow_html=True
)

name = st.text_input(
    "Tên tin đăng",
    value="Nhà đẹp Quận 1"
)

description = st.text_area(
    "Mô tả bất động sản",
    value="Nhà đẹp, vị trí trung tâm, gần trường học và chợ",
    height=120
)

st.markdown("</div>", unsafe_allow_html=True)



# NÚT DỰ ĐOÁN


predict_button = st.button(
    "DỰ ĐOÁN GIÁ NHÀ"
)

# XỬ LÝ DỰ ĐOÁN

# XỬ LÝ DỰ ĐOÁN


if predict_button:

    # Ngày hiện tại
    today = datetime.now()

    published_year = today.year
    published_month = today.month
    published_dayofweek = today.weekday()

    # Feature engineering
    total_rooms = bedroom_count + bathroom_count

    area_per_room = area / max(total_rooms, 1)

    description_length = len(description)

    name_length = len(name)

    # Dữ liệu đầu vào
    house = {
        "name": name,
        "description": description,
        "property_type_name": property_type_name,
        "province_name": "Hồ Chí Minh",
        "district_name": district_name,
        "ward_name": ward_name,
        "street_name": street_name,
        "project_name": project_name,
        "area": area,
        "floor_count": floor_count,
        "frontage_width": frontage_width,
        "house_depth": house_depth,
        "road_width": road_width,
        "bedroom_count": bedroom_count,
        "bathroom_count": bathroom_count,
        "house_direction": house_direction,
        "balcony_direction": balcony_direction,
        "total_rooms": total_rooms,
        "area_per_room": area_per_room,
        "published_year": published_year,
        "published_month": published_month,
        "published_dayofweek": published_dayofweek,
        "description_length": description_length,
        "name_length": name_length
    }

    # Chuyển dữ liệu thành DataFrame
    X_new = pd.DataFrame([house])

    try:

        # Dự đoán
        predicted_price = model.predict(X_new)[0]

        # Không cho giá âm
        if predicted_price < 0:
            predicted_price = 0

        predicted_billion = predicted_price / 1e9

        # Hiển thị kết quả
        st.markdown(
            f"""<div class="result-card">
<div class="result-title">GIÁ NHÀ DỰ ĐOÁN</div>
<div class="result-price">{predicted_billion:.2f} tỷ đồng</div>
<div class="result-vnd">{predicted_price:,.0f} VNĐ</div>
</div>""",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")