from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import requests

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_FEATURE_EXTRACTION_URL = (
    f"https://router.huggingface.co/hf-inference/models/{EMBED_MODEL_ID}/pipeline/feature-extraction"
)
INDEX_NAME = "b2b-products"
CANDIDATE_POOL = 25
TOP_N = 5
RETRYABLE_STATUS = (429, 500, 502, 503)

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


def get_secret(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, "")
    except Exception:
        return ""


def groq_json(prompt: str, api_key: str) -> dict | None:
    try:
        response = requests.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=45,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except Exception:
        return None


def embed_query(text: str, hf_token: str) -> list[float]:
    headers = {"Authorization": f"Bearer {hf_token}"}
    last_error = ""
    for attempt in range(4):
        response = requests.post(
            HF_FEATURE_EXTRACTION_URL, headers=headers, json={"inputs": [text]}, timeout=60
        )
        if response.status_code == 200:
            vector = response.json()[0]
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            return [v / norm for v in vector]
        last_error = f"HF API {response.status_code}"
        if response.status_code in RETRYABLE_STATUS:
            time.sleep(2**attempt)
            continue
        break
    raise RuntimeError(f"Query embedding failed ({last_error})")


def parse_query(query: str, groq_key: str) -> dict:
    fallback = {"semantic_query": query, "max_price": None, "min_price": None, "max_moq": None}
    if not groq_key:
        return fallback
    parsed = groq_json(PARSE_PROMPT.format(query=query), groq_key)
    if not parsed or not parsed.get("semantic_query"):
        return fallback
    for key in ("max_price", "min_price", "max_moq"):
        value = parsed.get(key)
        parsed[key] = float(value) if isinstance(value, (int, float)) else None
    return parsed


def build_metadata_filter(parsed: dict) -> dict | None:
    clauses = []
    price_bounds = {}
    if parsed.get("max_price") is not None:
        price_bounds["$lte"] = parsed["max_price"]
    if parsed.get("min_price") is not None:
        price_bounds["$gte"] = parsed["min_price"]
    if price_bounds:
        clauses.append({"price_usd": price_bounds})
    if parsed.get("max_moq") is not None:
        clauses.append({"moq": {"$lte": parsed["max_moq"]}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def to_result(match) -> SearchResult:
    metadata = dict(match.metadata or {})
    return SearchResult(
        id=str(match.id),
        name=metadata.get("name", ""),
        description=metadata.get("description", ""),
        price_usd=float(metadata.get("price_usd", 0)),
        moq=int(metadata.get("moq", 1)),
        category=metadata.get("category", ""),
        brand=metadata.get("brand", ""),
        image_url=metadata.get("image_url", ""),
        score=float(match.score or 0),
    )


def rerank(query: str, candidates: list[SearchResult], groq_key: str) -> list[SearchResult] | None:
    if not groq_key or not candidates:
        return None
    payload = [
        {
            "id": candidate.id,
            "name": candidate.name,
            "category": candidate.category,
            "brand": candidate.brand,
            "price_usd": candidate.price_usd,
            "moq": candidate.moq,
            "description": candidate.description[:300],
        }
        for candidate in candidates
    ]
    output = groq_json(
        RERANK_PROMPT.format(
            query=query, candidates=json.dumps(payload, ensure_ascii=False), top_n=TOP_N
        ),
        groq_key,
    )
    if not output or not isinstance(output.get("picks"), list):
        return None
    by_id = {candidate.id: candidate for candidate in candidates}
    picked = []
    for pick in output["picks"][:TOP_N]:
        candidate = by_id.get(str(pick.get("id", "")))
        if candidate:
            candidate.reason = str(pick.get("reason", ""))[:300]
            picked.append(candidate)
    return picked or None


def search(query: str) -> SearchResponse:
    from pinecone import Pinecone

    hf_token = get_secret("HF_TOKEN")
    pinecone_key = get_secret("PINECONE_API_KEY")
    groq_key = get_secret("GROQ_API_KEY")

    response = SearchResponse()
    if not query.strip():
        return response
    if not (hf_token and pinecone_key):
        response.error = "Missing HF_TOKEN or PINECONE_API_KEY. Check your secrets configuration."
        return response

    try:
        parsed = parse_query(query, groq_key)
        response.parsed = parsed
        vector = embed_query(parsed["semantic_query"], hf_token)
        index = Pinecone(api_key=pinecone_key).Index(INDEX_NAME)
        matches = index.query(
            vector=vector,
            top_k=CANDIDATE_POOL,
            filter=build_metadata_filter(parsed),
            include_metadata=True,
        ).matches
        candidates = [to_result(match) for match in matches]

        picked = rerank(query, candidates, groq_key)
        if picked:
            response.results = picked
            response.llm_used = True
        else:
            response.results = candidates[:TOP_N]
    except Exception as exc:
        response.error = str(exc)
    return response
