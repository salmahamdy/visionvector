"""FastAPI backend exposing multimodal search endpoints."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from core.embedding_pipeline import EmbeddingPipeline
from core.rag_explainer import LLMBackend, RAGExplainer
from core.retrieval import HybridRetriever, MetadataFilter, SearchResult
from utils.cache import TTLCache, make_cache_key
from utils.logging_config import configure_logging

configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index"))
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", "embeddings"))
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "image_embeddings.npy"
MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", None)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# ---------------------------------------------------------------------------
# Shared singletons (initialised at startup)
# ---------------------------------------------------------------------------
_pipeline: EmbeddingPipeline | None = None
_retriever: HybridRetriever | None = None
_explainer: RAGExplainer | None = None
_search_cache: TTLCache[list[dict]] = TTLCache(maxsize=512, ttl_seconds=300)
_explain_cache: TTLCache[str] = TTLCache(maxsize=128, ttl_seconds=600)
_last_results: list[SearchResult] = []


def _get_pipeline() -> EmbeddingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = EmbeddingPipeline(model_name=MODEL_NAME)
    return _pipeline


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(
            index_dir=INDEX_DIR,
            metadata_path=METADATA_PATH,
            embeddings_path=EMBEDDINGS_PATH,
        )
    return _retriever


def _get_explainer() -> RAGExplainer:
    global _explainer
    if _explainer is None:
        _explainer = RAGExplainer(backend=LLMBackend(LLM_BACKEND), model=LLM_MODEL)
    return _explainer


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up model and index…")
    _get_pipeline()
    if METADATA_PATH.exists() and EMBEDDINGS_PATH.exists():
        _get_retriever()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Multimodal Search API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if DATA_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(DATA_DIR)), name="images")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(default=10, ge=1, le=50)
    categories: list[str] | None = None
    tags: list[str] | None = None
    rerank: bool = True


class ExplainRequest(BaseModel):
    query: str
    query_type: str = "text"
    top_k: int = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta_filter(categories: list[str] | None, tags: list[str] | None) -> MetadataFilter | None:
    if categories or tags:
        return MetadataFilter(categories=categories, tags=tags)
    return None


def _results_to_json(results: list[SearchResult]) -> list[dict]:
    return [r.to_dict() for r in results]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/search/text", summary="Search images by text query")
async def search_by_text(request: TextSearchRequest) -> JSONResponse:
    cache_key = make_cache_key(
        request.query, request.top_k, request.categories, request.tags, request.rerank
    )
    cached = _search_cache.get(cache_key)
    if cached is not None:
        logger.debug("Cache hit for query: %s", request.query)
        return JSONResponse({"results": cached, "cached": True})

    try:
        pipeline = _get_pipeline()
        embedding = pipeline.encode_text([request.query])[0]
        retriever = _get_retriever()
        results = retriever.search(
            embedding,
            top_k=request.top_k,
            meta_filter=_meta_filter(request.categories, request.tags),
            rerank=request.rerank,
        )
    except Exception as exc:
        logger.exception("Text search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    global _last_results
    _last_results = results
    payload = _results_to_json(results)
    _search_cache.set(cache_key, payload)
    return JSONResponse({"results": payload, "cached": False})


@app.post("/search/image", summary="Search images by uploaded image")
async def search_by_image(
    file: Annotated[UploadFile, File(description="Image file to search with")],
    top_k: Annotated[int, Form()] = 10,
    categories: Annotated[str | None, Form()] = None,
    rerank: Annotated[bool, Form()] = True,
) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    raw = await file.read()
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {exc}") from exc

    try:
        pipeline = _get_pipeline()
        embedding = pipeline.encode_image([image])[0]
        retriever = _get_retriever()
        cat_list = [c.strip() for c in categories.split(",")] if categories else None
        results = retriever.search(
            embedding,
            top_k=top_k,
            meta_filter=_meta_filter(cat_list, None),
            rerank=rerank,
        )
    except Exception as exc:
        logger.exception("Image search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    global _last_results
    _last_results = results
    return JSONResponse({"results": _results_to_json(results)})


@app.get("/results", summary="Get last search results")
async def get_last_results() -> JSONResponse:
    return JSONResponse({"results": _results_to_json(_last_results)})


@app.post("/explain", summary="RAG explanation of search results")
async def explain_results(request: ExplainRequest) -> JSONResponse:
    cache_key = make_cache_key(request.query, request.query_type, request.top_k)
    cached = _explain_cache.get(cache_key)
    if cached is not None:
        return JSONResponse({"explanation": cached, "cached": True})

    results_to_explain = _last_results[: request.top_k]
    if not results_to_explain:
        raise HTTPException(status_code=400, detail="No search results available to explain. Run a search first.")

    try:
        explainer = _get_explainer()
        explanation = explainer.explain(request.query, results_to_explain, request.query_type)
    except Exception as exc:
        logger.exception("RAG explanation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _explain_cache.set(cache_key, explanation)
    return JSONResponse({"explanation": explanation, "cached": False})


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "index_loaded": _retriever is not None,
        "llm_backend": LLM_BACKEND,
    }
