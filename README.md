# Multimodal Image Search System

Production-ready image search upgraded from OpenAI CLIP + Faiss into a full multimodal pipeline.

## Architecture

```
Text / Image Query
        │
        ▼
  EmbeddingPipeline          ← openai/clip-vit-base-patch32 (HuggingFace)
  (encode_text / encode_image)
        │
        ▼
  HybridRetriever
  ├── Faiss ANN (IVFFlat)    ← fast approximate search
  ├── Metadata filter        ← category / tag / filename
  └── Cosine re-ranker       ← exact re-ranking on top-K
        │
        ▼
  RAGExplainer               ← Groq (llama3) or Ollama (local)
        │
        ▼
  FastAPI  ←→  Streamlit UI
```

## Project Structure

```
multimodal-search/
├── core/
│   ├── embedding_pipeline.py   # Batch image encoding → .npy + metadata.json
│   ├── indexing.py             # IndexFlatIP + IndexIVFFlat builder
│   ├── retrieval.py            # Hybrid vector + metadata search + re-rank
│   └── rag_explainer.py        # LLM explanation via Groq or Ollama
├── api/
│   └── main.py                 # FastAPI: /search/text /search/image /explain
├── frontend/
│   └── app.py                  # Streamlit UI
├── utils/
│   ├── cache.py                # TTL LRU cache
│   ├── image_utils.py          # PIL helpers
│   ├── logging_config.py       # Centralised logging
│   └── download_sample_data.py # Download test images
├── data/images/                # Put your images here (nested subdirs = categories)
├── embeddings/                 # Auto-generated .npy + metadata.json
├── index/                      # Auto-generated flat.faiss + ivf.faiss
├── .env.example
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add GROQ_API_KEY (free at console.groq.com)
```

### 3. Add images

**Option A – download sample images:**
```bash
python utils/download_sample_data.py
```

**Option B – use your own:**
```
data/images/
├── dogs/
│   ├── labrador.jpg
│   └── poodle.jpg
├── cars/
│   └── sedan.jpg
```
Subdirectory names become the `category` metadata field.

### 4. Build the embedding index

```bash
# Encode images → embeddings/image_embeddings.npy + embeddings/metadata.json
python -m core.embedding_pipeline \
    --data_dir data/images \
    --embeddings_dir embeddings \
    --metadata_path embeddings/metadata.json

# Build Faiss indexes → index/flat.faiss + index/ivf.faiss
python -m core.indexing \
    --embeddings_path embeddings/image_embeddings.npy \
    --index_dir index
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Start the frontend

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/search/text` | Search by text query |
| POST | `/search/image` | Search by uploaded image |
| GET | `/results` | Retrieve last search results |
| POST | `/explain` | RAG explanation of last results |
| GET | `/health` | API status |

### Example: Text search (curl)

```bash
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "a dog running on grass", "top_k": 5, "rerank": true}'
```

### Example: Image search (curl)

```bash
curl -X POST http://localhost:8000/search/image \
  -F "file=@/path/to/query.jpg" \
  -F "top_k=5"
```

### Example: Explain results

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"query": "a dog running on grass", "query_type": "text", "top_k": 5}'
```

## Test Queries

```
"a golden retriever playing outdoors"
"red sports car on a highway"
"dense green forest with sunlight"
"pepperoni pizza on wooden table"
"person riding a bicycle"
```

## RAG Backend Options

### Groq (recommended, free tier available)
1. Sign up at https://console.groq.com
2. Set `GROQ_API_KEY=your_key` in `.env`
3. `LLM_BACKEND=groq`

### Ollama (fully local, no API key)
1. Install: https://ollama.ai
2. `ollama pull llama3`
3. Set `LLM_BACKEND=ollama` in `.env`

## Performance Notes

- Batch size: 32 images/batch (increase on GPU)
- `torch.no_grad()` used throughout inference
- GPU auto-detected via `torch.cuda.is_available()`
- Search results cached with 5-minute TTL
- IVF index ~10x faster than flat for large collections (>10k images)
- Re-ranking adds exact cosine pass on top-K only (cheap)
