import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "houses_clean.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 1. Đọc dữ liệu
df = pd.read_csv(DATA_PATH)

# 2. Feature Engineering
df["total_rooms"] = (
    df["bedroom_count"] +
    df["bathroom_count"]
)

df["area_per_room"] = (
    df["area"] /
    df["total_rooms"].replace(0, 1)
)

df["published_at"] = pd.to_datetime(
    df["published_at"],
    errors="coerce"
)

df["published_year"] = df["published_at"].dt.year
df["published_month"] = df["published_at"].dt.month
df["published_dayofweek"] = df["published_at"].dt.dayofweek

df["description_length"] = (
    df["description"].fillna("").str.len()
)

df["name_length"] = (
    df["name"].fillna("").str.len()
)


# 3. X và y
y = df["price"]

text_cols = [
    "name",
    "description"
]

categorical_cols = [
    "property_type_name",
    "province_name",
    "district_name",
    "ward_name",
    "street_name",
    "project_name",
    "house_direction",
    "balcony_direction"
]

numeric_cols = [
    "area",
    "floor_count",
    "frontage_width",
    "house_depth",
    "road_width",
    "bedroom_count",
    "bathroom_count",
    "total_rooms",
    "area_per_room",
    "published_year",
    "published_month",
    "published_dayofweek",
    "description_length",
    "name_length"
]


# 4. Xử lý dữ liệu thiếu
data = df.copy()

for col in text_cols:
    data[col] = data[col].fillna("")

for col in categorical_cols:
    data[col] = data[col].fillna("Không rõ")

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

X = data[
    text_cols +
    categorical_cols +
    numeric_cols
].copy()


# 5. Chia train/test giống notebook
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Bộ tiền xử lý
name_vectorizer = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2)
)

description_vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2)
)

onehot = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "name_tfidf",
            name_vectorizer,
            "name"
        ),
        (
            "description_tfidf",
            description_vectorizer,
            "description"
        ),
        (
            "categorical",
            onehot,
            categorical_cols
        ),
        (
            "numeric",
            "passthrough",
            numeric_cols
        )
    ]
)


# 7. Ba mô hình
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}


# 8. Huấn luyện và đánh giá
results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Đang huấn luyện: {model_name}")
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([
        model_name,
        mae / 1e9,
        rmse / 1e9,
        r2
    ])

    print(f"MAE  : {mae / 1e9:.2f} tỷ đồng")
    print(f"RMSE : {rmse / 1e9:.2f} tỷ đồng")
    print(f"R²   : {r2:.4f}")


# 9. Tạo bảng kết quả
results_df = pd.DataFrame(
    results,
    columns=[
        "Mô hình",
        "MAE (tỷ đồng)",
        "RMSE (tỷ đồng)",
        "R²"
    ]
)

print("\n" + "=" * 70)
print("                 KẾT QUẢ SO SÁNH")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE (tỷ đồng)": "{:.2f}".format,
            "RMSE (tỷ đồng)": "{:.2f}".format,
            "R²": "{:.4f}".format
        }
    )
)


# 10. Lưu bảng kết quả
results_path = REPORT_DIR / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nĐã lưu bảng: {results_path}")


# 11. Xác định mô hình có R² cao nhất
best_row = results_df.loc[
    results_df["R²"].idxmax()
]

print("\nMô hình có R² cao nhất:")
print(best_row["Mô hình"])
print(f"R²: {best_row['R²']:.4f}")


# 12. Vẽ biểu đồ so sánh R²
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Mô hình"],
    results_df["R²"]
)

plt.title("So sánh R² của các mô hình")
plt.xlabel("Mô hình")
plt.ylabel("R²")
plt.ylim(0, 1)

for i, value in enumerate(results_df["R²"]):
    plt.text(
        i,
        value + 0.02,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()

chart_path = REPORT_DIR / "model_comparison_r2.png"
plt.savefig(chart_path, dpi=200)
plt.show()

print(f"Đã lưu biểu đồ: {chart_path}")
