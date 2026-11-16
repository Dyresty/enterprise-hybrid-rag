"""Automatic validation of generated answers. Failing answers are flagged for human review."""
import re

from app.llm.prompt_builder import NOT_FOUND

# Below this top rerank score the retrieved context is weak (sigmoid scale, see rag_service).
LOW_CONFIDENCE_SCORE = 0.3


def cited_numbers(answer):
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


def review_reasons(answer, sources):
    if NOT_FOUND.lower() in answer.lower():
        # Unanswered questions show where data is missing - worth a curator's look.
        return ["no answer found in the documents"]

    reasons = []
    cited = cited_numbers(answer)
    if not cited:
        reasons.append("answer contains no citations")
    if cited - {s["n"] for s in sources}:
        reasons.append("answer cites a source that was not provided")
    if sources and max(s["score"] for s in sources) < LOW_CONFIDENCE_SCORE:
        reasons.append("low retrieval confidence")
    return reasons
