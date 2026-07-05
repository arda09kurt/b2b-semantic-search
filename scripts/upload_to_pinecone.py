"""
Step 3 — Vector Database
=========================
Creates a serverless Pinecone index (free tier) and upserts every product
embedding together with its metadata (name, description, price, MOQ, ...).

Usage:
    PINECONE_API_KEY=pc_xxx python scripts/upload_to_pinecone.py

Idempotent: re-running overwrites vectors with the same product IDs.
"""

import os
import sys

import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "b2b-products"
EMBED_DIM = 384
UPSERT_BATCH = 100

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main() -> None:
    api_key = os.environ.get("PINECONE_API_KEY", "")
    if not api_key:
        sys.exit("Set the PINECONE_API_KEY environment variable first.")

    df = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    data = np.load(os.path.join(DATA_DIR, "embeddings.npz"), allow_pickle=True)
    ids, vectors = data["ids"], data["vectors"]
    assert len(df) == len(ids), "products.csv and embeddings.npz are out of sync"

    pc = Pinecone(api_key=api_key)
    if not pc.has_index(INDEX_NAME):
        print(f"Creating serverless index '{INDEX_NAME}' ({EMBED_DIM}-dim, cosine)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(INDEX_NAME)

    records = df.to_dict("records")
    print(f"Upserting {len(records)} vectors in batches of {UPSERT_BATCH}...")
    for start in range(0, len(records), UPSERT_BATCH):
        batch = []
        for offset, row in enumerate(records[start : start + UPSERT_BATCH]):
            i = start + offset
            batch.append(
                {
                    "id": str(ids[i]),
                    "values": vectors[i].tolist(),
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
            )
        index.upsert(vectors=batch)
        done = min(start + UPSERT_BATCH, len(records))
        if (start // UPSERT_BATCH) % 20 == 0 or done == len(records):
            print(f"  {done}/{len(records)}")

    stats = index.describe_index_stats()
    print(f"Done. Index now holds {stats.total_vector_count} vectors.")


if __name__ == "__main__":
    main()
