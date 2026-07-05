"""
Step 5 — UI & Deployment
=========================
Streamlit frontend for the Semantic B2B Wholesale Product Search Engine.

Run locally:
    streamlit run app.py

Secrets (env vars or .streamlit/secrets.toml):
    HF_TOKEN, PINECONE_API_KEY, GROQ_API_KEY
"""

import streamlit as st

from search import SearchResponse, search

st.set_page_config(
    page_title="B2B Wholesale Semantic Search",
    page_icon="📦",
    layout="wide",
)

EXAMPLE_QUERIES = [
    "heavy duty warehouse shelving under $50 per unit",
    "comfortable cotton t-shirts for a retail store, need about 100 units",
    "elegant silver jewellery for a boutique under $10",
    "durable kitchen storage containers in bulk",
    "leather office chairs under $200",
]

st.markdown(
    """
    <style>
    .product-card {
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        background: rgba(128, 128, 128, 0.06);
    }
    .price-tag { font-size: 1.3rem; font-weight: 700; color: #16a34a; }
    .moq-tag { color: #b45309; font-weight: 600; }
    .reason { font-style: italic; opacity: 0.85; }
    .badge {
        display: inline-block; padding: 0.1rem 0.6rem; border-radius: 999px;
        background: rgba(59, 130, 246, 0.15); font-size: 0.8rem; margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📦 B2B Wholesale Semantic Search")
st.caption(
    "Search 17,000+ wholesale products in plain English — powered by "
    "AI embeddings (HuggingFace), vector search (Pinecone) and an LLM re-ranker (Groq)."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. 🧠 **LLM** (Groq · Llama 3.3) reads your query and extracts "
        "price / quantity constraints\n"
        "2. 🔢 Your query becomes a **vector embedding** (all-MiniLM-L6-v2)\n"
        "3. 🎯 **Pinecone** finds the most semantically similar products\n"
        "4. 🏆 The LLM re-ranks them and explains each match\n"
    )
    st.divider()
    st.markdown(
        "**Try queries like:**\n"
        + "\n".join(f"- *{q}*" for q in EXAMPLE_QUERIES[:3])
    )

if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_input(
    "What are you sourcing today?",
    value=st.session_state.query,
    placeholder="e.g. heavy duty warehouse shelving under $50 per unit",
    key="search_box",
)

cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(cols, EXAMPLE_QUERIES):
    if col.button(example[:32] + "…", key=example, use_container_width=True):
        query = example
        st.session_state.query = example


def render_result(rank: int, r) -> None:
    with st.container():
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        img_col, body_col = st.columns([1, 5])
        with img_col:
            if r.image_url:
                st.image(r.image_url, use_container_width=True)
            else:
                st.markdown("### 📦")
        with body_col:
            st.markdown(f"**{rank}. {r.name}**")
            st.markdown(
                f'<span class="badge">{r.category}</span>'
                f'<span class="badge">{r.brand}</span>'
                f'<span class="badge">match {r.score:.2f}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span class="price-tag">${r.price_usd:,.2f}</span> / unit &nbsp;·&nbsp; '
                f'<span class="moq-tag">MOQ: {r.moq} units</span>',
                unsafe_allow_html=True,
            )
            if r.reason:
                st.markdown(f'<p class="reason">🤖 {r.reason}</p>', unsafe_allow_html=True)
            with st.expander("Product details"):
                st.write(r.description)
        st.markdown("</div>", unsafe_allow_html=True)


if query.strip():
    with st.spinner("Searching the catalog semantically…"):
        response: SearchResponse = search(query)

    if response.error:
        st.error(f"Search failed: {response.error}")
    elif not response.results:
        st.warning(
            "No products matched those constraints. "
            "Try loosening the price or quantity limits."
        )
    else:
        parsed = response.parsed or {}
        chips = []
        if parsed.get("max_price") is not None:
            chips.append(f"max ${parsed['max_price']:,.0f}/unit")
        if parsed.get("min_price") is not None:
            chips.append(f"min ${parsed['min_price']:,.0f}/unit")
        if parsed.get("max_moq") is not None:
            chips.append(f"MOQ ≤ {parsed['max_moq']:,.0f}")
        understood = f"Understood: **{parsed.get('semantic_query', query)}**"
        if chips:
            understood += " · " + " · ".join(chips)
        st.markdown(understood)
        if not response.llm_used:
            st.caption("ℹ️ Showing raw similarity ranking (LLM re-ranker unavailable).")

        st.subheader(f"Top {len(response.results)} matches")
        for rank, result in enumerate(response.results, start=1):
            render_result(rank, result)
else:
    st.info("Type what you need above — natural language works best. 👆")
