import hashlib
import os
import re

import kagglehub
import pandas as pd

KAGGLE_DATASET = "PromptCloudHQ/flipkart-products"
SOURCE_FILENAME = "flipkart_com-ecommerce_sample.csv"
INR_TO_USD = 1 / 83.0
MIN_DESCRIPTION_LENGTH = 40
MIN_CATALOG_SIZE = 5000
MAX_DESCRIPTION_CHARS = 2000

MOQ_BANDS = [
    (2.0, [100, 200, 500]),
    (10.0, [50, 100, 200]),
    (50.0, [20, 50, 100]),
    (200.0, [5, 10, 20]),
    (float("inf"), [1, 2, 5]),
]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")


def stable_pick(key: str, options: list[int]) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return options[int(digest, 16) % len(options)]


def derive_moq(product_id: str, price_usd: float) -> int:
    for upper_bound, options in MOQ_BANDS:
        if price_usd < upper_bound:
            return stable_pick(product_id, options)
    return 1


def top_category(category_tree: str) -> str:
    if not isinstance(category_tree, str):
        return "General"
    cleaned = category_tree.strip().strip("[]").strip('"')
    return cleaned.split(">>")[0].strip() or "General"


def first_image(image_list: str) -> str:
    if not isinstance(image_list, str):
        return ""
    match = re.search(r'"(https?://[^"]+)"', image_list)
    return match.group(1) if match else ""


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def load_source() -> pd.DataFrame:
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    return pd.read_csv(os.path.join(path, SOURCE_FILENAME))


def build_catalog(raw: pd.DataFrame) -> pd.DataFrame:
    rows = raw.dropna(subset=["product_name", "description", "retail_price"])
    rows = rows.drop_duplicates(subset=["product_name", "description"])
    rows = rows[rows["description"].str.len() >= MIN_DESCRIPTION_LENGTH]

    catalog = pd.DataFrame()
    catalog["product_id"] = rows["uniq_id"]
    catalog["name"] = rows["product_name"].map(normalize_whitespace)
    catalog["description"] = (
        rows["description"].map(normalize_whitespace).str.slice(0, MAX_DESCRIPTION_CHARS)
    )
    catalog["price_usd"] = (rows["retail_price"] * INR_TO_USD).round(2)
    catalog["category"] = rows["product_category_tree"].map(top_category)
    catalog["brand"] = rows["brand"].fillna("Unbranded").map(normalize_whitespace)
    catalog["image_url"] = rows["image"].map(first_image)
    catalog["moq"] = [
        derive_moq(product_id, price)
        for product_id, price in zip(catalog["product_id"], catalog["price_usd"])
    ]
    return catalog


def main() -> None:
    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    raw = load_source()
    print(f"Raw rows: {len(raw)}")

    catalog = build_catalog(raw)

    assert len(catalog) >= MIN_CATALOG_SIZE, f"Expected {MIN_CATALOG_SIZE}+ products, got {len(catalog)}"
    assert catalog[["name", "description", "price_usd", "moq"]].notna().all().all()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    catalog.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(catalog)} products -> {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
