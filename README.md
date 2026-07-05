# 📦 Semantic B2B Wholesale Product Search Engine

A search engine that lets wholesale buyers find products using **natural,
conversational language** instead of strict keywords — e.g.

> *"heavy duty warehouse shelving under $50 per unit"*

The engine understands the intent ("industrial storage racks"), respects the
constraints ("unit price ≤ $50") and returns the 5 most relevant products
with AI-written explanations of why each one matches.

**Live demo:** _[add Streamlit Cloud URL after deployment]_

---

## Architecture

```
buyer query ──► Groq LLM (Llama 3.3)          parse intent + price/MOQ constraints
                     │
                     ▼
             HuggingFace API                  embed query (all-MiniLM-L6-v2, 384-d)
                     │
                     ▼
             Pinecone (serverless)            cosine similarity + metadata filters
                     │
                     ▼
             Groq LLM (Llama 3.3)             re-rank top 25 → top 5 + explanations
                     │
                     ▼
             Streamlit UI                     product cards: name, price, MOQ, why-it-matches
```

| Component  | Technology                                       | Cost |
|------------|--------------------------------------------------|------|
| Dataset    | Kaggle — Flipkart Products (20K real products)   | free |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace Inference API | free |
| Vector DB  | Pinecone serverless (starter tier)               | free |
| LLM        | Groq — `llama-3.3-70b-versatile`                 | free |
| Frontend   | Streamlit Community Cloud                        | free |

## Dataset

Source: [Flipkart Products on Kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products)
— 20,000 real e-commerce products with names, full descriptions, prices,
brands and categories. After cleaning (dropping rows without descriptions or
prices, de-duplicating): **17,467 products**.

**A note on MOQ:** no public Kaggle dataset with 5,000+ products carries a
native Minimum Order Quantity column (real-MOQ Alibaba scrapes top out at
~1,000 rows). To meet both the *5,000+ products* and the *MOQ* requirements,
`scripts/prepare_data.py` derives a deterministic wholesale MOQ per product
from its price band (cheap items ⇒ high MOQ, expensive items ⇒ low MOQ),
seeded by a stable hash of the product ID so the value is reproducible on
every run. The derivation is documented in the script.

## Project structure

```
├── app.py                        # Streamlit frontend (deployed entrypoint)
├── search.py                     # core search pipeline (parse → embed → search → re-rank)
├── scripts/
│   ├── prepare_data.py           # 1. download Kaggle dataset, clean, add MOQ
│   ├── generate_embeddings.py    # 2. embed 17K descriptions (HF API or local)
│   └── upload_to_pinecone.py     # 3. create index + upsert vectors & metadata
├── data/products.csv             # cleaned catalog (committed for reproducibility)
└── requirements.txt              # lightweight deps for Streamlit Cloud
```

## Setup (one-time pipeline)

```bash
pip install -r requirements.txt kagglehub sentence-transformers

# 1. build the catalog
python scripts/prepare_data.py

# 2. embed the catalog (same model the app uses for queries)
python scripts/generate_embeddings.py            # uses HF_TOKEN if set, else local model

# 3. upload to Pinecone
set PINECONE_API_KEY=pcsk_...                    # export on macOS/Linux
python scripts/upload_to_pinecone.py
```

## Run the app

```bash
# keys: copy .env.example values into your environment or .streamlit/secrets.toml
streamlit run app.py
```

`.streamlit/secrets.toml` (used by Streamlit Cloud too):

```toml
HF_TOKEN = "hf_..."
PINECONE_API_KEY = "pcsk_..."
GROQ_API_KEY = "gsk_..."
```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. On [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the
   repo, branch `main`, entrypoint `app.py`.
3. In **Advanced settings → Secrets**, paste the three keys shown above.
4. Deploy — the app only needs the lightweight `requirements.txt`
   (no torch/ML libraries at runtime; embeddings come from the HF API).

## Example queries

- heavy duty warehouse shelving under $50 per unit
- comfortable cotton t-shirts for a retail store, need about 100 units
- elegant silver jewellery for a boutique under $10
- durable kitchen storage containers in bulk
