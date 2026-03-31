"""RAG explanation layer: uses Groq or Ollama to explain search results."""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any

from core.retrieval import SearchResult

logger = logging.getLogger(__name__)


class LLMBackend(str, Enum):
    GROQ = "groq"
    OLLAMA = "ollama"


def _build_prompt(query: str, results: list[SearchResult], query_type: str) -> str:
    result_descriptions = "\n".join(
        f"{i + 1}. File: {r.filename} | Category: {r.category} "
        f"| Tags: {', '.join(r.tags) or 'none'} "
        f"| Similarity: {(r.rerank_score or r.score):.3f}"
        for i, r in enumerate(results)
    )
    return f"""You are a multimodal AI assistant explaining image search results.

Query type: {query_type}
User query: "{query}"

Retrieved images:
{result_descriptions}

Provide a concise explanation covering:
1. Why these images match the query (based on category, tags, filenames, similarity scores)
2. A brief description of what each result likely shows
3. How results compare to each other in relevance

Keep your explanation under 200 words. Be specific and factual."""


class RAGExplainer:
    """Generates natural language explanations for search results via LLM."""

    def __init__(
        self,
        backend: LLMBackend | str = LLMBackend.GROQ,
        model: str | None = None,
        groq_api_key: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self.backend = LLMBackend(backend)
        self.ollama_base_url = ollama_base_url

        if self.backend == LLMBackend.GROQ:
            try:
                from groq import Groq  # type: ignore

                self._groq_client = Groq(
                    api_key=groq_api_key or os.environ.get("GROQ_API_KEY")
                )
                self._groq_model = model or "llama3-8b-8192"
            except ImportError as exc:
                raise RuntimeError(
                    "Install groq: pip install groq"
                ) from exc

        elif self.backend == LLMBackend.OLLAMA:
            self._ollama_model = model or "llama3"

    def _call_groq(self, prompt: str) -> str:
        response = self._groq_client.chat.completions.create(
            model=self._groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    def _call_ollama(self, prompt: str) -> str:
        import httpx  # type: ignore

        payload: dict[str, Any] = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 300},
        }
        resp = httpx.post(
            f"{self.ollama_base_url}/api/generate",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def explain(
        self,
        query: str,
        results: list[SearchResult],
        query_type: str = "text",
    ) -> str:
        """Generate an explanation for a set of search results.

        Args:
            query: Original user query string.
            results: Ordered list of SearchResult objects.
            query_type: "text" or "image".

        Returns:
            Natural language explanation string.
        """
        if not results:
            return "No results were found to explain."

        prompt = _build_prompt(query, results, query_type)

        try:
            if self.backend == LLMBackend.GROQ:
                return self._call_groq(prompt)
            return self._call_ollama(prompt)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return (
                f"Explanation unavailable ({self.backend.value} error). "
                f"Top result: {results[0].filename} (score: {results[0].score:.3f})"
            )
