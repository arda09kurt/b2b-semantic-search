"""
Step 1 — Data Sourcing & Preparation
=====================================
Downloads the Flipkart Products dataset from Kaggle (20,000 real e-commerce
products with names, descriptions and prices) and prepares it for the
wholesale semantic search engine.

Dataset: https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products

Note on MOQ: public Kaggle datasets with a *native* MOQ column top out at
~1,000 rows (small Alibaba scrapes), which fails the 5,000+ product
requirement. To satisfy both requirements we use this real 20K-product
dataset and derive a deterministic wholesale MOQ per product from its price
band and a stable hash of its product ID (reproducible on every run).

Usage:
    python scripts/prepare_data.py

Output:
    data/products.csv  (cleaned, ~19K products)
"""

import hashlib
import os
import re

import kagglehub
import pandas as pd

INR_TO_USD = 1 / 83.0  # fixed conversion rate used for the whole catalog

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")

# Wholesale MOQ options per unit-price band (USD). Cheaper items are sold in
# larger minimum quantities, matching real B2B marketplace behaviour.
MOQ_BANDS = [
    (2.0, [100, 200, 500]),      # < $2
    (10.0, [50, 100, 200]),      # $2 - $10
    (50.0, [20, 50, 100]),       # $10 - $50
    (200.0, [5, 10, 20]),        # $50 - $200
    (float("inf"), [1, 2, 5]),   # > $200
]


def stable_pick(key: str, options: list) -> int:
    """Deterministically pick an option based on a stable hash of the key."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return options[int(digest, 16) % len(options)]


def derive_moq(product_id: str, price_usd: float) -> int:
    for threshold, options in MOQ_BANDS:
        if price_usd < threshold:
            return stable_pick(product_id, options)
    return 1


def top_category(tree: str) -> str:
    """Extract the top-level category from Flipkart's category tree string."""
    if not isinstance(tree, str):
        return "General"
    cleaned = tree.strip().strip("[]").strip('"')
    return cleaned.split(">>")[0].strip() or "General"


def first_image(images: str) -> str:
    if not isinstance(images, str):
        return ""
    match = re.search(r'"(https?://[^"]+)"', images)
    return match.group(1) if match else ""


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def main() -> None:
    print("Downloading Flipkart products dataset from Kaggle...")
    path = kagglehub.dataset_download("PromptCloudHQ/flipkart-products")
    csv_path = os.path.join(path, "flipkart_com-ecommerce_sample.csv")
    df = pd.read_csv(csv_path)
    print(f"Raw rows: {len(df)}")

    # Keep rows with the fields the search engine needs.
    df = df.dropna(subset=["product_name", "description", "retail_price"])
    df = df.drop_duplicates(subset=["product_name", "description"])

    # Drop rows whose description is too short to embed meaningfully.
    df = df[df["description"].str.len() >= 40]

    out = pd.DataFrame()
    out["product_id"] = df["uniq_id"]
    out["name"] = df["product_name"].map(clean_text)
    out["description"] = df["description"].map(clean_text).str.slice(0, 2000)
    out["price_usd"] = (df["retail_price"] * INR_TO_USD).round(2)
    out["category"] = df["product_category_tree"].map(top_category)
    out["brand"] = df["brand"].fillna("Unbranded").map(clean_text)
    out["image_url"] = df["image"].map(first_image)
    out["moq"] = [
        derive_moq(pid, price)
        for pid, price in zip(out["product_id"], out["price_usd"])
    ]

    # Sanity checks required by the project spec.
    assert len(out) >= 5000, f"Need 5,000+ products, got {len(out)}"
    assert out[["name", "description", "price_usd", "moq"]].notna().all().all()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} products -> {os.path.abspath(OUT_PATH)}")
    print(out[["name", "price_usd", "moq", "category"]].head(10).to_string())


if __name__ == "__main__":
    main()
