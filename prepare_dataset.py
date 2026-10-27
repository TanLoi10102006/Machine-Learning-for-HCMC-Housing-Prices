"""
prepare_dataset.py
==================
Tải dataset bất động sản Việt Nam từ Hugging Face, lọc TP. Hồ Chí Minh
và lấy ngẫu nhiên tối đa 10.000 tin phù hợp cho đồ án dự báo giá nhà.

Dataset nguồn:
    tinixai/vietnam-real-estates

Đầu ra:
    data/raw/houses_hcm_10000_raw.csv
"""

from pathlib import Path
import random
import sys

try:
    import pandas as pd
    from datasets import load_dataset
except ImportError as exc:
    print("THIẾU THƯ VIỆN:", exc)
    print()
    print("Hãy chạy:")
    print("pip install datasets pyarrow pandas")
    sys.exit(1)


# =========================
# CẤU HÌNH
# =========================
DATASET_ID = "tinixai/vietnam-real-estates"
TARGET_ROWS = 10_000
RANDOM_SEED = 42

OUTPUT_DIR = Path("data") / "raw"
OUTPUT_FILE = OUTPUT_DIR / "houses_hcm_10000_raw.csv"

# Những loại bất động sản phù hợp với bài toán "dự báo giá nhà".
# Có thể sửa danh sách này sau nếu giảng viên yêu cầu hẹp hơn.
PROPERTY_KEYWORDS = [
    "căn hộ",
    "nhà",
    "biệt thự",
    "shophouse",
]


def normalize_text(value) -> str:
    """Chuẩn hóa text đơn giản để lọc ổn định hơn."""
    if value is None:
        return ""
    return str(value).strip().lower()


def is_hcm(province_name) -> bool:
    """Nhận diện các cách ghi TP. Hồ Chí Minh."""
    text = normalize_text(province_name)

    aliases = {
        "hồ chí minh",
        "tp. hồ chí minh",
        "tp.hồ chí minh",
        "tp hồ chí minh",
        "thành phố hồ chí minh",
        "ho chi minh",
        "tp. hcm",
        "tp hcm",
        "hcm",
    }

    return text in aliases or "hồ chí minh" in text


def is_target_property(property_type_name) -> bool:
    """Lọc các loại BĐS được dùng cho đồ án."""
    text = normalize_text(property_type_name)
    return any(keyword in text for keyword in PROPERTY_KEYWORDS)


def valid_row(row: dict) -> bool:
    """Kiểm tra các trường tối thiểu cần thiết."""
    price = row.get("price")
    area = row.get("area")
    description = normalize_text(row.get("description"))

    if price is None or area is None:
        return False

    try:
        price = float(price)
        area = float(area)
    except (TypeError, ValueError):
        return False

    # Loại các giá/diện tích không hợp lệ.
    if price <= 0 or area <= 0:
        return False

    # Giữ lại tin có mô tả để sau này làm TF-IDF.
    if not description:
        return False

    return True


def main():
    print("=" * 70)
    print("CHUẨN BỊ DATASET DỰ BÁO GIÁ NHÀ")
    print("=" * 70)
    print(f"Dataset nguồn : {DATASET_ID}")
    print(f"Số mẫu cần lấy: {TARGET_ROWS:,}")
    print("Khu vực       : TP. Hồ Chí Minh")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Đang kết nối Hugging Face...")
    try:
        # streaming=True giúp không tải toàn bộ dataset vào RAM.
        dataset = load_dataset(
            DATASET_ID,
            split="train",
            streaming=True,
        )
    except Exception as exc:
        print("\nKHÔNG TẢI ĐƯỢC DATASET.")
        print("Chi tiết lỗi:")
        print(exc)
        print("\nHãy kiểm tra Internet và chạy:")
        print("pip install -U datasets huggingface_hub pyarrow pandas")
        sys.exit(1)

    rng = random.Random(RANDOM_SEED)

    # Reservoir sampling:
    # Quét dữ liệu và lấy ngẫu nhiên TARGET_ROWS mẫu đạt điều kiện.
    reservoir = []

    scanned = 0
    hcm_count = 0
    valid_count = 0

    print("Đang quét dữ liệu... (có thể mất một lúc vì dataset lớn)")
    print()

    for row in dataset:
        scanned += 1

        province = row.get("province_name")
        property_type = row.get("property_type_name")

        if not is_hcm(province):
            continue

        hcm_count += 1

        if not is_target_property(property_type):
            continue

        if not valid_row(row):
            continue

        valid_count += 1

        # Reservoir sampling để mẫu có tính ngẫu nhiên trên toàn bộ dữ liệu.
        if len(reservoir) < TARGET_ROWS:
            reservoir.append(row)
        else:
            j = rng.randint(0, valid_count - 1)
            if j < TARGET_ROWS:
                reservoir[j] = row

        if scanned % 50_000 == 0:
            print(
                f"Đã quét {scanned:,} dòng | "
                f"HCM: {hcm_count:,} | "
                f"hợp lệ: {valid_count:,} | "
                f"đang giữ: {len(reservoir):,}"
            )

    if not reservoir:
        print("\nKHÔNG TÌM THẤY DỮ LIỆU PHÙ HỢP.")
        print("Hãy kiểm tra tên tỉnh/thành hoặc cấu trúc dataset.")
        sys.exit(1)

    df = pd.DataFrame(reservoir)

    # Chỉ giữ các cột cần thiết + một số cột hữu ích cho phân tích.
    preferred_columns = [
        "name",
        "description",
        "property_type_name",
        "province_name",
        "district_name",
        "ward_name",
        "street_name",
        "project_name",
        "price",
        "area",
        "floor_count",
        "frontage_width",
        "house_depth",
        "road_width",
        "bedroom_count",
        "bathroom_count",
        "house_direction",
        "balcony_direction",
        "published_at",
    ]

    available_columns = [c for c in preferred_columns if c in df.columns]
    df = df[available_columns].copy()

    # Xáo trộn lần cuối trước khi lưu.
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print()
    print("=" * 70)
    print("HOÀN TẤT")
    print("=" * 70)
    print(f"File          : {OUTPUT_FILE.resolve()}")
    print(f"Số dòng       : {len(df):,}")
    print(f"Số cột        : {len(df.columns)}")
    print()
    print("Các cột:")
    print(", ".join(df.columns))
    print()
    print("5 dòng đầu:")
    print(df.head().to_string(index=False))
    print()
    print("Thiếu dữ liệu:")
    print(df.isnull().sum().to_string())


if __name__ == "__main__":
    main()
