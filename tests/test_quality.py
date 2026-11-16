"""Tests for structure-aware parsing, OCR fallback, chunk-level sources, answer validation and
the feedback/review loop. Models and OCR are replaced by small fakes."""
import pymupdf
import pytest

from app.evaluation.benchmark import judge_says
from app.ingestion import parser
from app.llm.prompt_builder import NOT_FOUND
from app.services.answer_checks import review_reasons
from app.services.context_builder import ContextBuilder
from app.services.feedback_store import FeedbackStore
from app.services.rag_service import RAGService


def make_pdf(path, with_scanned_page=True):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Project Results", fontsize=20)
    page.insert_text((72, 110), "The hybrid retriever reached an MRR of 0.937 on the evaluation set.", fontsize=11)
    if with_scanned_page:
        src = pymupdf.open()
        src.new_page().insert_text((72, 100), "Scanned text", fontsize=18)
        pix = src[0].get_pixmap(dpi=100)
        scanned = doc.new_page()
        scanned.insert_image(scanned.rect, pixmap=pix)
    doc.save(path)


def test_parser_keeps_headings_and_uses_ocr_only_for_scanned_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, "ocr_page", lambda page: "OCR TEXT")
    path = str(tmp_path / "doc.pdf")
    make_pdf(path)

    pages = parser.extract_pdf_text(path)

    assert pages[0]["text"].startswith("# Project Results")
    assert pages[0]["ocr"] is False
    assert pages[1] == {"page": 2, "text": "OCR TEXT", "ocr": True}


def test_review_reasons():
    sources = [{"n": 1, "score": 0.9}, {"n": 2, "score": 0.4}]
    assert review_reasons("ENEXA is coordinated by Paderborn [1].", sources) == []
    assert review_reasons("ENEXA is coordinated by Paderborn.", sources) == ["answer contains no citations"]
    assert review_reasons("See [3].", sources) == ["answer cites a source that was not provided"]
    assert review_reasons("Maybe [1].", [{"n": 1, "score": 0.1}]) == ["low retrieval confidence"]
    assert review_reasons(NOT_FOUND, []) == ["no answer found in the documents"]


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def search(self, query, top_k):
        return self.docs


class FakeReranker:
    def rerank(self, query, docs, top_k):
        return [{**d, "rerank_score": 0.95} for d in docs][:top_k]


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer

    def chat(self, messages):
        return self.answer


def doc(chunk_id):
    return {"document_id": "101070305", "chunk_id": chunk_id, "parent_id": "p1", "filename": "CORDIS 101070305",
            "page": 1, "title": "ENEXA", "url": "https://cordis.europa.eu/project/id/101070305",
            "text": f"child text {chunk_id}", "parent_text": "parent text"}


def test_answers_have_chunk_level_sources_and_are_logged(tmp_path):
    store = FeedbackStore(tmp_path)
    rag = RAGService(FakeRetriever([doc(7)]), FakeReranker(), ContextBuilder(), FakeLLM("Paderborn [1]."), store)

    result = rag.generate_answer("Who coordinates ENEXA?")

    assert result["sources"][0]["chunk_id"] == 7
    assert result["sources"][0]["excerpt"] == "child text 7"
    assert result["needs_review"] is False
    assert store.review_queue() == []


def test_flagged_and_down_rated_answers_enter_review_queue(tmp_path):
    store = FeedbackStore(tmp_path)
    rag = RAGService(FakeRetriever([doc(1)]), FakeReranker(), ContextBuilder(), FakeLLM("No citation here."), store)
    flagged = rag.generate_answer("q1")

    rag.llm = FakeLLM("Paderborn [1].")
    good = rag.generate_answer("q2")
    store.add_feedback(good["answer_id"], "down", "wrong partner")

    queue = {a["answer_id"]: a["review_reasons"] for a in store.review_queue()}
    assert queue[flagged["answer_id"]] == ["answer contains no citations"]
    assert queue[good["answer_id"]] == ["rated down: wrong partner"]

    with pytest.raises(KeyError):
        store.add_feedback("unknown", "up")


def test_judge_verdict_parsing():
    assert judge_says(" relevant.", "RELEVANT") == 1.0
    assert judge_says("IRRELEVANT", "RELEVANT") == 0.0
