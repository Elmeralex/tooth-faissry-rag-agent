"""search_dental_knowledge with a fake retrieval module: no FAISS, no API key."""

from types import SimpleNamespace

from agent.tools import knowledge
from agent.tools.base import DETAILS_KEY


def retrieval_result(accepted):
    return {
        "retrieved_chunk": "Dental amalgam contains mercury, silver, copper and tin.",
        "retrieved_contexts": ["Dental amalgam contains mercury, silver, copper and tin.", "Related chunk"],
        "score": 10.65 if accepted else -1.44,
        "vector_score": 0.64,
        "bm25_score": 9.38,
        "rrf_score": 0.03,
        "rerank_score": 10.65 if accepted else -1.44,
        "accepted": accepted,
        "reason": None if accepted else "No sufficiently similar context found after reranking",
        "metadata": {"source_type": "pdf", "source_file": "amalgam.pdf", "chunk_id": 0},
        "source_filter": "all",
    }


def use_fake_retrieval(monkeypatch, accepted):
    fake = SimpleNamespace(
        RERANK_ACCEPT_THRESHOLD=3.0,
        retrieve_nearest_chunk=lambda query, store, source_filter: retrieval_result(accepted),
    )
    monkeypatch.setattr(knowledge, "_load_retrieval", lambda: (fake, None))


def test_accepted_search_returns_context_and_retrieval_details(monkeypatch):
    use_fake_retrieval(monkeypatch, accepted=True)
    result = knowledge.search_dental_knowledge("What is dental amalgam")

    assert result["found"] is True and result["source_file"] == "amalgam.pdf"
    details = result[DETAILS_KEY]
    assert details["query"] == "What is dental amalgam"
    assert (details["vector_score"], details["bm25_score"], details["rrf_score"]) == (0.64, 9.38, 0.03)
    assert details["rerank_threshold"] == 3.0
    assert details["metadata"]["source_file"] == "amalgam.pdf"


def test_rejected_search_still_has_retrieval_details(monkeypatch):
    use_fake_retrieval(monkeypatch, accepted=False)
    result = knowledge.search_dental_knowledge("What causes dry socket?")

    assert result["found"] is False
    assert result[DETAILS_KEY]["accepted"] is False
    assert result[DETAILS_KEY]["reason"].startswith("No sufficiently similar")
