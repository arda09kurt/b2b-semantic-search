"""
Step 4 — Search Logic
======================
Turns a natural-language buyer query into ranked product results:

1. A Groq LLM parses the query into a clean semantic search phrase plus hard
   constraints (max price, max MOQ, quantity needed).
2. The query is embedded with the same free HuggingFace model used for the
   catalog (``all-MiniLM-L6-v2``).
3. Pinecone performs a cosine similarity search, applying the hard
   constraints as metadata filters.
4. The Groq LLM re-ranks the candidates and explains, per product, why it
   matches — the top 5 are returned.

Every step degrades gracefully: if the LLM is unavailable the raw vector
ranking is returned; the app never shows an empty error page.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import requests

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
INDEX_NAME = "b2b-products"
CANDIDATES = 25          # vectors fetched from Pinecone
TOP_N = 5                # results shown to the buyer

PARSE_PROMPT = """\
You parse wholesale product search queries for a B2B marketplace.
Extract from the buyer's query:
- "semantic_query": the product they want, rephrased as a short descriptive phrase \
(keep material/use-case words, drop price/quantity words)
- "max_price": max unit price in USD as a number, or null
- "min_price": min unit price in USD as a number, or null
- "max_moq": the largest minimum-order-quantity the buyer could accept, or null. \
If the buyer says they need N units, max_moq is N (they can't meet a bigger minimum).
Respond with ONLY a JSON object, no markdown.

Buyer query: {query}"""

RERANK_PROMPT = """\
You are a B2B sourcing assistant. A wholesale buyer searched: "{query}"

Candidate products (JSON):
{candidates}

Pick the {top_n} products that BEST satisfy the buyer's intent (relevance first,
then value). Respond with ONLY a JSON object, no markdown:
{{"picks": [{{"id": "<product id>", "reason": "<one short sentence: why this fits the buyer's need>"}}]}}"""


@dataclass
class SearchResult:
    id: str
    name: str
    description: str
    price_usd: float
    moq: int
    category: str
    brand: str
    image_url: str
    score: float
    reason: str = ""


@dataclass
class SearchResponse:
    results: list[SearchResult] = field(default_factory=list)
    parsed: dict | None = None
    llm_used: bool = False
    error: str = ""


def _get_key(name: str) -> str:
    """Read a key from the environment, falling back to Streamlit secrets."""
    val = os.environ.get(name, "")
    if val:
        return val
    try:
        import streamlit as st

        return st.secrets.get(name, "")
    except Exception:
        return ""


def _groq_json(prompt: str, api_key: str) -> dict | None:
    """Call Groq chat completion and parse a JSON object out of the reply."""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=45,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except Exception:
        return None


def embed_query(text: str, hf_token: str) -> list[float]:
    """Embed a single query string with the free HuggingFace Inference API."""
    url = (
        "https://router.huggingface.co/hf-inference/models/"
        f"{EMBED_MODEL_ID}/pipeline/feature-extraction"
    )
    headers = {"Authorization": f"Bearer {hf_token}"}
    last_err = ""
    for attempt in range(4):
        resp = requests.post(url, headers=headers, json={"inputs": [text]}, timeout=60)
        if resp.status_code == 200:
            vec = resp.json()[0]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            return [v / norm for v in vec]
        last_err = f"HF API {resp.status_code}"
        if resp.status_code in (429, 500, 502, 503):
            time.sleep(2**attempt)
            continue
        break
    raise RuntimeError(f"Query embedding failed ({last_err})")


def parse_query(query: str, groq_key: str) -> dict:
    """LLM extraction of constraints; falls back to the raw query."""
    fallback = {"semantic_query": query, "max_price": None, "min_price": None, "max_moq": None}
    if not groq_key:
        return fallback
    parsed = _groq_json(PARSE_PROMPT.format(query=query), groq_key)
    if not parsed or not parsed.get("semantic_query"):
        return fallback
    for k in ("max_price", "min_price", "max_moq"):
        v = parsed.get(k)
        parsed[k] = float(v) if isinstance(v, (int, float)) else None
    return parsed


def _metadata_filter(parsed: dict) -> dict | None:
    clauses = []
    price = {}
    if parsed.get("max_price") is not None:
        price["$lte"] = parsed["max_price"]
    if parsed.get("min_price") is not None:
        price["$gte"] = parsed["min_price"]
    if price:
        clauses.append({"price_usd": price})
    if parsed.get("max_moq") is not None:
        clauses.append({"moq": {"$lte": parsed["max_moq"]}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _to_result(match) -> SearchResult:
    md = dict(match.metadata or {})
    return SearchResult(
        id=str(match.id),
        name=md.get("name", ""),
        description=md.get("description", ""),
        price_usd=float(md.get("price_usd", 0)),
        moq=int(md.get("moq", 1)),
        category=md.get("category", ""),
        brand=md.get("brand", ""),
        image_url=md.get("image_url", ""),
        score=float(match.score or 0),
    )


def rerank(query: str, candidates: list[SearchResult], groq_key: str) -> list[SearchResult] | None:
    """LLM picks and explains the top N; returns None if the LLM fails."""
    if not groq_key or not candidates:
        return None
    payload = [
        {
            "id": c.id,
            "name": c.name,
            "category": c.category,
            "brand": c.brand,
            "price_usd": c.price_usd,
            "moq": c.moq,
            "description": c.description[:300],
        }
        for c in candidates
    ]
    out = _groq_json(
        RERANK_PROMPT.format(
            query=query, candidates=json.dumps(payload, ensure_ascii=False), top_n=TOP_N
        ),
        groq_key,
    )
    if not out or not isinstance(out.get("picks"), list):
        return None
    by_id = {c.id: c for c in candidates}
    picked = []
    for pick in out["picks"][:TOP_N]:
        c = by_id.get(str(pick.get("id", "")))
        if c:
            c.reason = str(pick.get("reason", ""))[:300]
            picked.append(c)
    return picked or None


def search(query: str) -> SearchResponse:
    """Full pipeline: parse -> embed -> vector search -> LLM re-rank."""
    from pinecone import Pinecone

    hf_token = _get_key("HF_TOKEN")
    pinecone_key = _get_key("PINECONE_API_KEY")
    groq_key = _get_key("GROQ_API_KEY")

    resp = SearchResponse()
    if not query.strip():
        return resp
    if not (hf_token and pinecone_key):
        resp.error = "Missing HF_TOKEN or PINECONE_API_KEY — check your secrets."
        return resp

    try:
        parsed = parse_query(query, groq_key)
        resp.parsed = parsed
        vector = embed_query(parsed["semantic_query"], hf_token)
        index = Pinecone(api_key=pinecone_key).Index(INDEX_NAME)
        matches = index.query(
            vector=vector,
            top_k=CANDIDATES,
            filter=_metadata_filter(parsed),
            include_metadata=True,
        ).matches
        candidates = [_to_result(m) for m in matches]

        picked = rerank(query, candidates, groq_key)
        if picked:
            resp.results = picked
            resp.llm_used = True
        else:
            resp.results = candidates[:TOP_N]
    except Exception as exc:  # surface a friendly message, never a stack trace
        resp.error = str(exc)
    return resp
