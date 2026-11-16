"""Unit tests for the pure logic. No models, Qdrant or LLM needed: python -m pytest"""
from app.evaluation.retrieval_metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank, unique_docs
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25 import BM25Retriever, tokenize
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.services.context_builder import ContextBuilder


def doc(document_id, chunk_id, text="x", parent_id=None):
    return {"document_id": document_id, "chunk_id": chunk_id, "text": text,
            "parent_id": parent_id or f"{document_id}-p", "filename": document_id, "page": 1}


def test_rrf_rewards_agreement_between_retrievers():
    dense = [doc("a", 0), doc("b", 0), doc("c", 0)]
    sparse = [doc("d", 0), doc("b", 0), doc("e", 0)]
    fused = [d["document_id"] for d in reciprocal_rank_fusion([dense, sparse])]
    # b is 2nd in both lists (2/62), so it beats a and d (1st in one list only, 1/61)
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d", "e"}


def test_rrf_score_formula():
    fused = reciprocal_rank_fusion([[doc("a", 0)], [doc("a", 0)]], k=60)
    assert abs(fused[0]["rrf_score"] - 2 / 61) < 1e-12


def test_bm25_finds_exact_acronym():
    docs = [
        {**doc("p1", 0, "Robust machine learning for energy grids"), "title": "GRIDML"},
        {**doc("p2", 0, "Quantum photonics on chip"), "title": "PHOQUS"},
        {**doc("p3", 0, "Soil health monitoring"), "title": "SOILMON"},  # Okapi IDF is 0 with only 2 docs
    ]
    results = BM25Retriever(docs).search("PHOQUS project")
    assert results[0]["document_id"] == "p2"


def test_bm25_empty_index_and_stopwords():
    assert BM25Retriever([]).search("anything") == []
    assert tokenize("What is the AI-Act?") == ["ai", "act"]


def test_chunker_children_carry_parent_text():
    text = "Sentence number one. " * 400  # ~8,400 chars -> several parents and children
    chunks = chunk_documents([{"page": 1, "text": text}], "d1", "d1.pdf")
    assert len(chunks) > 2
    for c in chunks:
        assert c["text"] in c["parent_text"]
        assert len(c["text"]) <= 800 < len(c["parent_text"])
    assert [c["metadata"]["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_context_builder_dedupes_parents_and_numbers_sources():
    docs = [doc("a", 0, parent_id="p1"), doc("a", 1, parent_id="p1"), doc("b", 0, parent_id="p2")]
    for d in docs:
        d["parent_text"] = f"parent of {d['parent_id']}"
    context, sources = ContextBuilder().build(docs, max_chars=10_000)
    assert [s["n"] for s in sources] == [1, 2]
    assert "[1]" in context and "[2]" in context and "parent of p1" in context


def test_context_builder_respects_budget_but_keeps_first():
    docs = [doc("a", 0, "y" * 500, "p1"), doc("b", 0, "z" * 500, "p2")]
    _, sources = ContextBuilder().build(docs, max_chars=600)
    assert len(sources) == 1


def test_metrics():
    ranked = unique_docs([doc("a", 0), doc("a", 1), doc("b", 0), doc("c", 0)])
    assert ranked == ["a", "b", "c"]
    assert reciprocal_rank(ranked, ["b"]) == 0.5
    assert hit_at_k(ranked, ["c"], 2) == 0.0 and hit_at_k(ranked, ["c"], 3) == 1.0
    assert recall_at_k(ranked, ["a", "z"], 3) == 0.5
    assert precision_at_k(ranked, ["a", "b"], 2) == 1.0
