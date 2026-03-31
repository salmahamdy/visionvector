"""Streamlit frontend for the multimodal image search system."""

from __future__ import annotations

import os
from io import BytesIO

import httpx
import streamlit as st
from PIL import Image

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 60.0

st.set_page_config(
    page_title="Multimodal Image Search",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 8px;
        background: #fafafa;
    }
    .score-badge {
        background: #1f77b4;
        color: white;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.75rem;
    }
    .explanation-box {
        background: #f0f7ff;
        border-left: 4px solid #1f77b4;
        padding: 12px;
        border-radius: 4px;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results: list[dict] = []
if "last_query" not in st.session_state:
    st.session_state.last_query: str = ""
if "last_query_type" not in st.session_state:
    st.session_state.last_query_type: str = "text"
if "explanation" not in st.session_state:
    st.session_state.explanation: str = ""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_text_search(query: str, top_k: int, categories: list[str], rerank: bool) -> list[dict]:
    payload = {
        "query": query,
        "top_k": top_k,
        "categories": categories or None,
        "rerank": rerank,
    }
    resp = httpx.post(f"{API_BASE}/search/text", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["results"]


def _api_image_search(image_bytes: bytes, top_k: int, rerank: bool) -> list[dict]:
    files = {"file": ("upload.jpg", image_bytes, "image/jpeg")}
    data = {"top_k": str(top_k), "rerank": str(rerank).lower()}
    resp = httpx.post(f"{API_BASE}/search/image", files=files, data=data, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["results"]


def _api_explain(query: str, query_type: str, top_k: int) -> str:
    payload = {"query": query, "query_type": query_type, "top_k": top_k}
    resp = httpx.post(f"{API_BASE}/explain", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["explanation"]


def _load_result_image(result: dict) -> Image.Image | None:
    path: str = result.get("path", "")
    filename: str = result.get("filename", "")
    # Try via API static mount
    try:
        url = f"{API_BASE}/images/{filename}"
        r = httpx.get(url, timeout=5.0)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    # Try local path
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("🔍 Multimodal Image Search")
st.caption("Search images by text description or by uploading a query image.")

with st.sidebar:
    st.header("⚙️ Search Options")
    top_k = st.slider("Number of results", min_value=1, max_value=30, value=9)
    rerank = st.toggle("Re-rank results", value=True)
    categories_input = st.text_input("Filter by category (comma-separated)", "")
    categories = [c.strip() for c in categories_input.split(",") if c.strip()]

    st.divider()
    st.subheader("🤖 RAG Explanation")
    explain_top_k = st.slider("Results to explain", min_value=1, max_value=10, value=5)

    st.divider()
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=3.0).json()
        st.success(f"API online  ·  {health['model'].split('/')[-1]}")
        if not health["index_loaded"]:
            st.warning("Index not loaded — run embedding pipeline first.")
    except Exception:
        st.error("API offline — start uvicorn.")

# ---------------------------------------------------------------------------
# Search tabs
# ---------------------------------------------------------------------------
tab_text, tab_image = st.tabs(["📝 Text Search", "🖼️ Image Search"])

with tab_text:
    query = st.text_input("Enter a search query", placeholder="e.g. a golden retriever in the park")
    col_search, col_clear = st.columns([1, 5])
    with col_search:
        search_btn = st.button("Search", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear"):
            st.session_state.results = []
            st.session_state.explanation = ""
            st.rerun()

    if search_btn and query:
        with st.spinner("Searching…"):
            try:
                st.session_state.results = _api_text_search(query, top_k, categories, rerank)
                st.session_state.last_query = query
                st.session_state.last_query_type = "text"
                st.session_state.explanation = ""
            except Exception as exc:
                st.error(f"Search failed: {exc}")

with tab_image:
    uploaded = st.file_uploader(
        "Upload a query image", type=["jpg", "jpeg", "png", "webp"]
    )
    img_search_btn = st.button("Search by Image", type="primary")

    if img_search_btn and uploaded:
        image_bytes = uploaded.read()
        with st.spinner("Encoding and searching…"):
            try:
                st.session_state.results = _api_image_search(image_bytes, top_k, rerank)
                st.session_state.last_query = uploaded.name
                st.session_state.last_query_type = "image"
                st.session_state.explanation = ""
            except Exception as exc:
                st.error(f"Image search failed: {exc}")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(image_bytes, caption="Query image", use_container_width=True)

# ---------------------------------------------------------------------------
# Results grid
# ---------------------------------------------------------------------------
results = st.session_state.results

if results:
    st.divider()
    col_header, col_explain = st.columns([3, 1])
    with col_header:
        st.subheader(f"Results ({len(results)})")
    with col_explain:
        explain_btn = st.button("✨ Explain results", type="secondary", use_container_width=True)

    if explain_btn:
        with st.spinner("Generating explanation…"):
            try:
                st.session_state.explanation = _api_explain(
                    st.session_state.last_query,
                    st.session_state.last_query_type,
                    explain_top_k,
                )
            except Exception as exc:
                st.error(f"Explanation failed: {exc}")

    if st.session_state.explanation:
        st.markdown(
            f'<div class="explanation-box">{st.session_state.explanation}</div>',
            unsafe_allow_html=True,
        )

    num_cols = 3
    for row_start in range(0, len(results), num_cols):
        cols = st.columns(num_cols)
        for col, result in zip(cols, results[row_start : row_start + num_cols]):
            with col:
                img = _load_result_image(result)
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.info("🖼️ Image not available locally")

                score = result.get("rerank_score") or result.get("score", 0)
                st.markdown(
                    f"**{result['filename']}**  \n"
                    f"`{result['category']}`  "
                    f'<span class="score-badge">{score:.3f}</span>',
                    unsafe_allow_html=True,
                )
                if result.get("tags"):
                    st.caption("Tags: " + ", ".join(result["tags"]))
