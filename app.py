import html

import streamlit as st

from search import SearchResponse, SearchResult, search

st.set_page_config(
    page_title="Wholesale Product Search",
    layout="centered",
    initial_sidebar_state="collapsed",
)

EXAMPLE_QUERIES = [
    "Heavy duty warehouse shelving under $50 per unit",
    "Cotton t-shirts for a retail store, around 100 units",
    "Silver jewellery for a boutique under $10",
    "Kitchen storage containers in bulk",
]

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, p, div, input, button {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 3.5rem; max-width: 860px; }

.hero-title {
    font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em;
    color: #0f172a; margin-bottom: 0.2rem;
}
.hero-subtitle { color: #64748b; font-size: 1.02rem; margin-bottom: 1.6rem; }

.stTextInput input {
    border-radius: 10px; border: 1.5px solid #e2e8f0; padding: 0.85rem 1rem;
    font-size: 1rem; background: #ffffff;
}
.stTextInput input:focus { border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12); }

div[data-testid="stButton"] button {
    border-radius: 999px; border: 1px solid #e2e8f0; background: #ffffff;
    color: #475569; font-size: 0.82rem; font-weight: 500; padding: 0.3rem 0.9rem;
}
div[data-testid="stButton"] button:hover { border-color: #2563eb; color: #2563eb; background: #eff6ff; }

.query-summary {
    background: #f1f5f9; border-radius: 10px; padding: 0.7rem 1rem;
    font-size: 0.9rem; color: #334155; margin: 1rem 0 1.4rem 0;
}
.query-summary .constraint {
    display: inline-block; background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 6px; padding: 0.1rem 0.55rem; margin-left: 0.45rem;
    font-weight: 600; color: #0f172a; font-size: 0.82rem;
}

.results-heading {
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #94a3b8; margin: 0.4rem 0 0.9rem 0;
}

.product-card {
    background: #ffffff; border: 1px solid #e8edf3; border-radius: 14px;
    padding: 1.25rem 1.4rem; margin-bottom: 0.9rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}
.product-card .top-row { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; }
.product-card .rank { color: #94a3b8; font-weight: 600; font-size: 0.85rem; margin-right: 0.5rem; }
.product-card .name { font-weight: 700; font-size: 1.05rem; color: #0f172a; line-height: 1.35; }
.product-card .meta { margin: 0.55rem 0 0.75rem 0; }
.product-card .pill {
    display: inline-block; background: #f8fafc; border: 1px solid #eef2f7;
    color: #64748b; border-radius: 6px; font-size: 0.76rem; font-weight: 500;
    padding: 0.12rem 0.55rem; margin-right: 0.4rem;
}
.product-card .price { font-size: 1.25rem; font-weight: 800; color: #0f172a; white-space: nowrap; }
.product-card .price small { font-size: 0.78rem; font-weight: 500; color: #94a3b8; }
.product-card .moq {
    font-size: 0.82rem; color: #475569; font-weight: 600;
    text-align: right; margin-top: 0.15rem;
}
.product-card .reason {
    border-left: 3px solid #2563eb; background: #f8fafc; color: #475569;
    font-size: 0.88rem; padding: 0.55rem 0.85rem; border-radius: 0 8px 8px 0;
    margin-top: 0.35rem;
}
.product-card details { margin-top: 0.7rem; }
.product-card summary {
    cursor: pointer; font-size: 0.82rem; font-weight: 600; color: #2563eb;
    list-style: none;
}
.product-card details p {
    font-size: 0.86rem; color: #64748b; line-height: 1.55; margin-top: 0.5rem;
}
.notice { color: #94a3b8; font-size: 0.84rem; margin-bottom: 0.8rem; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


def set_query(value: str) -> None:
    st.session_state.query_input = value


def format_constraints(parsed: dict) -> str:
    chips = []
    if parsed.get("max_price") is not None:
        chips.append(f'<span class="constraint">Max ${parsed["max_price"]:,.0f} / unit</span>')
    if parsed.get("min_price") is not None:
        chips.append(f'<span class="constraint">Min ${parsed["min_price"]:,.0f} / unit</span>')
    if parsed.get("max_moq") is not None:
        chips.append(f'<span class="constraint">MOQ up to {parsed["max_moq"]:,.0f}</span>')
    return "".join(chips)


def render_query_summary(parsed: dict, fallback_query: str) -> None:
    understood = html.escape(parsed.get("semantic_query") or fallback_query)
    st.markdown(
        f'<div class="query-summary">Interpreted as <strong>{understood}</strong>'
        f"{format_constraints(parsed)}</div>",
        unsafe_allow_html=True,
    )


def render_card(rank: int, result: SearchResult) -> None:
    reason_block = (
        f'<div class="reason">{html.escape(result.reason)}</div>' if result.reason else ""
    )
    st.markdown(
        f"""
        <div class="product-card">
          <div class="top-row">
            <div>
              <span class="rank">{rank:02d}</span>
              <span class="name">{html.escape(result.name)}</span>
              <div class="meta">
                <span class="pill">{html.escape(result.category)}</span>
                <span class="pill">{html.escape(result.brand)}</span>
                <span class="pill">{result.score * 100:.0f}% match</span>
              </div>
            </div>
            <div>
              <div class="price">${result.price_usd:,.2f} <small>/ unit</small></div>
              <div class="moq">MOQ {result.moq:,} units</div>
            </div>
          </div>
          {reason_block}
          <details>
            <summary>Product details</summary>
            <p>{html.escape(result.description)}</p>
          </details>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown('<div class="hero-title">Wholesale Product Search</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Describe what you need in plain language — '
    "semantic search across 17,000+ products with AI-ranked results.</div>",
    unsafe_allow_html=True,
)

query = st.text_input(
    "Search",
    key="query_input",
    placeholder="e.g. heavy duty warehouse shelving under $50 per unit",
    label_visibility="collapsed",
)

columns = st.columns(len(EXAMPLE_QUERIES))
for column, example in zip(columns, EXAMPLE_QUERIES):
    column.button(
        example,
        key=f"example-{example}",
        on_click=set_query,
        args=(example,),
        use_container_width=True,
    )

with st.sidebar:
    st.markdown("#### How it works")
    st.markdown(
        "1. A large language model (Groq, Llama 3.3) extracts price and "
        "quantity constraints from your query\n"
        "2. The query is converted to a vector embedding (all-MiniLM-L6-v2)\n"
        "3. Pinecone retrieves the most semantically similar products\n"
        "4. The model re-ranks results and explains each match"
    )

if query.strip():
    with st.spinner("Searching the catalog"):
        response: SearchResponse = search(query)

    if response.error:
        st.error(f"Search failed: {response.error}")
    elif not response.results:
        st.warning("No products matched those constraints. Try loosening the price or quantity limits.")
    else:
        render_query_summary(response.parsed or {}, query)
        if not response.llm_used:
            st.markdown(
                '<div class="notice">Showing similarity ranking — AI re-ranking unavailable.</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="results-heading">Top {len(response.results)} results</div>',
            unsafe_allow_html=True,
        )
        for rank, result in enumerate(response.results, start=1):
            render_card(rank, result)
