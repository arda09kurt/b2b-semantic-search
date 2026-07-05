"""
Step 2 — AI Embeddings
=======================
Converts every product's text (name + category + brand + description) into a
384-dimensional vector using the free `sentence-transformers/all-MiniLM-L6-v2`
model.

Two interchangeable backends produce *identical* vectors:

  * ``hf``    — HuggingFace Inference API (free, needs HF_TOKEN). This is the
                same API the deployed app uses to embed live search queries.
  * ``local`` — runs the same model locally via the sentence-transformers
                library. Recommended for the one-off bulk embedding of the
                full catalog so you don't burn free API credits on 17K texts.

Usage:
    python scripts/generate_embeddings.py            # auto: hf if HF_TOKEN set, else local
    python scripts/generate_embeddings.py --backend local
    HF_TOKEN=hf_xxx python scripts/generate_embeddings.py --backend hf

Output:
    data/embeddings.npz  (ids + float32 vectors, aligned with products.csv)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 64

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
OUT_NPZ = os.path.join(DATA_DIR, "embeddings.npz")


def build_text(row: pd.Series) -> str:
    """The text that gets embedded for each product."""
    return (
        f"{row['name']}. Category: {row['category']}. "
        f"Brand: {row['brand']}. {row['description']}"
    )


def embed_hf(texts: list[str], token: str) -> np.ndarray:
    """Embed a batch of texts with the free HuggingFace Inference API."""
    import requests

    url = (
        "https://router.huggingface.co/hf-inference/models/"
        f"{MODEL_ID}/pipeline/feature-extraction"
    )
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(6):
        resp = requests.post(url, headers=headers, json={"inputs": texts}, timeout=120)
        if resp.status_code == 200:
            vecs = np.asarray(resp.json(), dtype=np.float32)
            # Normalize so cosine similarity matches the local backend.
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.clip(norms, 1e-12, None)
        if resp.status_code in (429, 500, 502, 503):  # model loading / rate limit
            wait = 2 ** attempt
            print(f"  HF API {resp.status_code}, retrying in {wait}s...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HF API error {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("HF API kept failing after retries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "local", "auto"], default="auto")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    backend = args.backend
    if backend == "auto":
        backend = "hf" if token else "local"
    if backend == "hf" and not token:
        sys.exit("Set the HF_TOKEN environment variable to use the HF API backend.")

    df = pd.read_csv(PRODUCTS_CSV)
    texts = df.apply(build_text, axis=1).tolist()
    ids = df["product_id"].tolist()
    print(f"Embedding {len(texts)} products with backend '{backend}'...")

    if backend == "local":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_ID, device="cpu")
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
    else:
        chunks = []
        for i in range(0, len(texts), BATCH_SIZE):
            chunks.append(embed_hf(texts[i : i + BATCH_SIZE], token))
            done = min(i + BATCH_SIZE, len(texts))
            if (i // BATCH_SIZE) % 10 == 0 or done == len(texts):
                print(f"  {done}/{len(texts)}")
        vectors = np.vstack(chunks)

    assert vectors.shape == (len(ids), EMBED_DIM), vectors.shape
    np.savez_compressed(OUT_NPZ, ids=np.array(ids), vectors=vectors)
    print(f"Saved {vectors.shape[0]} x {vectors.shape[1]} embeddings -> {os.path.abspath(OUT_NPZ)}")


if __name__ == "__main__":
    main()
