import os
import sys

import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "b2b-products"
EMBED_DIM = 384
UPSERT_BATCH = 100

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_catalog() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    catalog = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    archive = np.load(os.path.join(DATA_DIR, "embeddings.npz"), allow_pickle=True)
    ids, vectors = archive["ids"], archive["vectors"]
    if len(catalog) != len(ids):
        sys.exit("products.csv and embeddings.npz are out of sync; rerun generate_embeddings.py")
    return catalog, ids, vectors


def ensure_index(client: Pinecone):
    if not client.has_index(INDEX_NAME):
        print(f"Creating serverless index '{INDEX_NAME}' ({EMBED_DIM}-dim, cosine)...")
        client.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return client.Index(INDEX_NAME)


def to_vector(product_id: str, values: np.ndarray, row: dict) -> dict:
    return {
        "id": str(product_id),
        "values": values.tolist(),
        "metadata": {
            "name": str(row["name"])[:500],
            "description": str(row["description"])[:1200],
            "price_usd": float(row["price_usd"]),
            "moq": int(row["moq"]),
            "category": str(row["category"])[:120],
            "brand": str(row["brand"])[:120],
            "image_url": str(row["image_url"] or "")[:500],
        },
    }


def main() -> None:
    api_key = os.environ.get("PINECONE_API_KEY", "")
    if not api_key:
        sys.exit("Set the PINECONE_API_KEY environment variable first.")

    catalog, ids, vectors = load_catalog()
    index = ensure_index(Pinecone(api_key=api_key))

    records = catalog.to_dict("records")
    print(f"Upserting {len(records)} vectors in batches of {UPSERT_BATCH}...")
    for start in range(0, len(records), UPSERT_BATCH):
        batch = [
            to_vector(ids[start + offset], vectors[start + offset], row)
            for offset, row in enumerate(records[start : start + UPSERT_BATCH])
        ]
        index.upsert(vectors=batch)
        done = min(start + UPSERT_BATCH, len(records))
        if (start // UPSERT_BATCH) % 20 == 0 or done == len(records):
            print(f"  {done}/{len(records)}")

    stats = index.describe_index_stats()
    print(f"Done. Index now holds {stats.total_vector_count} vectors.")


if __name__ == "__main__":
    main()
