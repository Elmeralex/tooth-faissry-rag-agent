"""Wraps the existing hybrid RAG retrieval (FAISS + BM25 + RRF + cross-encoder) as an agent tool.

The heavy imports (torch, transformers, FAISS) happen on first use, so the
other tools and the tests work without them. src/preprocessing.py uses paths
relative to the repo root, so the index path is made absolute here and the
tool works from any working directory.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.tools.base import DETAILS_KEY, Tool

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
MAX_CONTEXT_CHARS = 1500
RELATED_CONTEXTS = 2
RELATED_CONTEXT_CHARS = 400
# Retrieval fields shown in the UI, the same ones the RAG chatbot shows.
DETAIL_FIELDS = (
    "accepted",
    "source_filter",
    "score",
    "vector_score",
    "bm25_score",
    "rrf_score",
    "rerank_score",
    "retrieved_chunk",
    "metadata",
    "reason",
)


@lru_cache(maxsize=1)
def _load_retrieval():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import preprocessing
    import retrieval

    store_dir = ROOT / preprocessing.VECTORSTORE_DIR
    if not (store_dir / "index.faiss").exists():
        raise FileNotFoundError(
            f"No FAISS index at {store_dir}. Build it first with `uv run python src/preprocessing.py`."
        )
    # retrieval.load_index_documents() (used for BM25) reads this module global.
    retrieval.VECTORSTORE_DIR = store_dir
    return retrieval, retrieval.load_vectorstore(store_dir)


def search_dental_knowledge(query: str, source: str = "all") -> dict[str, Any]:
    retrieval, vectorstore = _load_retrieval()
    result = retrieval.retrieve_nearest_chunk(query, vectorstore, source_filter=source)
    details = {
        "query": query,
        **{field: result.get(field) for field in DETAIL_FIELDS},
        "rerank_threshold": retrieval.RERANK_ACCEPT_THRESHOLD,
    }

    if not result["accepted"]:
        return {
            "found": False,
            "reason": result["reason"],
            "rerank_score": result["rerank_score"],
            DETAILS_KEY: details,
        }

    contexts = result["retrieved_contexts"]
    metadata = result.get("metadata") or {}
    return {
        "found": True,
        "rerank_score": round(result["rerank_score"], 2),
        "source_type": metadata.get("source_type"),
        "source_file": metadata.get("source_file"),
        "context": contexts[0][:MAX_CONTEXT_CHARS],
        # Lower-ranked chunks did not pass the rerank threshold on their own; treat as background.
        "related_contexts": [c[:RELATED_CONTEXT_CHARS] for c in contexts[1 : 1 + RELATED_CONTEXTS]],
        DETAILS_KEY: details,
    }


TOOL = Tool(
    name="search_dental_knowledge",
    description=(
        "Search the dental knowledge base (MedMCQA explanations + California Dental Association PDFs) "
        "with hybrid retrieval and reranking. Use for clinical and dental facts. Returns found=false "
        "when nothing relevant enough is in the knowledge base."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A focused search query, one topic at a time"},
            "source": {
                "type": "string",
                "enum": ["all", "pdf", "medmcqa"],
                "description": "Restrict to a source type (default: all)",
            },
        },
        "required": ["query"],
    },
    func=search_dental_knowledge,
)
