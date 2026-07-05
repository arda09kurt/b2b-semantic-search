import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 64
MAX_RETRIES = 6
RETRYABLE_STATUS = (429, 500, 502, 503)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
OUTPUT_NPZ = os.path.join(DATA_DIR, "embeddings.npz")

HF_FEATURE_EXTRACTION_URL = (
    f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}/pipeline/feature-extraction"
)


def build_text(row: pd.Series) -> str:
    return (
        f"{row['name']}. Category: {row['category']}. "
        f"Brand: {row['brand']}. {row['description']}"
    )


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def embed_via_hf_api(texts: list[str], token: str) -> np.ndarray:
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(MAX_RETRIES):
        response = requests.post(
            HF_FEATURE_EXTRACTION_URL, headers=headers, json={"inputs": texts}, timeout=120
        )
        if response.status_code == 200:
            return normalize(np.asarray(response.json(), dtype=np.float32))
        if response.status_code in RETRYABLE_STATUS:
            time.sleep(2**attempt)
            continue
        raise RuntimeError(f"HF API error {response.status_code}: {response.text[:300]}")
    raise RuntimeError("HF API kept failing after retries")


def embed_all_hf(texts: list[str], token: str) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), BATCH_SIZE):
        chunks.append(embed_via_hf_api(texts[start : start + BATCH_SIZE], token))
        done = min(start + BATCH_SIZE, len(texts))
        if (start // BATCH_SIZE) % 10 == 0 or done == len(texts):
            print(f"  {done}/{len(texts)}")
    return np.vstack(chunks)


def embed_all_local(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID, device="cpu")
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def resolve_backend(requested: str, token: str) -> str:
    if requested == "auto":
        return "hf" if token else "local"
    if requested == "hf" and not token:
        sys.exit("Set the HF_TOKEN environment variable to use the HF API backend.")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "local", "auto"], default="auto")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    backend = resolve_backend(args.backend, token)

    catalog = pd.read_csv(PRODUCTS_CSV)
    texts = catalog.apply(build_text, axis=1).tolist()
    ids = catalog["product_id"].tolist()
    print(f"Embedding {len(texts)} products with backend '{backend}'...")

    vectors = embed_all_local(texts) if backend == "local" else embed_all_hf(texts, token)

    assert vectors.shape == (len(ids), EMBED_DIM), vectors.shape
    np.savez_compressed(OUTPUT_NPZ, ids=np.array(ids), vectors=vectors)
    print(f"Saved {vectors.shape[0]} x {vectors.shape[1]} embeddings -> {os.path.abspath(OUTPUT_NPZ)}")


if __name__ == "__main__":
    main()
