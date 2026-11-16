"""Benchmark retrieval strategies and answer quality on a labelled question set.

    python -m app.evaluation.benchmark                      # retrieval only (no LLM calls)
    python -m app.evaluation.benchmark --generation         # + answer checks with the LLM

Golden set format (JSONL): {"question": str, "relevant": [document_id, ...], "type": str}
Questions with "relevant": [] are out-of-scope and must be refused.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

from app import config
from app.evaluation.retrieval_metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank, unique_docs
from app.llm.prompt_builder import NOT_FOUND
from app.services.container import get_container

JUDGE_PROMPT = """You check whether an answer is supported by its sources.
Reply with one word: SUPPORTED if every factual claim in the answer is backed by the sources, otherwise UNSUPPORTED.

Sources:
{context}

Answer:
{answer}"""


def strategies(c):
    n = config.CANDIDATES
    return {
        "dense": lambda q: c.dense.search(q, n),
        "bm25": lambda q: c.bm25.search(q, n),
        "hybrid (RRF)": lambda q: c.hybrid.search(q, n),
        "hybrid + rerank": lambda q: c.reranker.rerank(q, c.hybrid.search(q, n), top_k=n),
    }


def evaluate_retrieval(c, questions, ks=(1, 5, 10)):
    rows = []
    for name, search in strategies(c).items():
        scores = {f"Hit@{k}": [] for k in ks} | {"Recall@5": [], "P@5": [], "MRR": []}
        for q in questions:
            ranked = unique_docs(search(q["question"]))
            for k in ks:
                scores[f"Hit@{k}"].append(hit_at_k(ranked, q["relevant"], k))
            scores["Recall@5"].append(recall_at_k(ranked, q["relevant"], 5))
            scores["P@5"].append(precision_at_k(ranked, q["relevant"], 5))
            scores["MRR"].append(reciprocal_rank(ranked, q["relevant"]))
        rows.append((name, {m: statistics.mean(v) for m, v in scores.items()}))
    return rows


def evaluate_generation(c, questions):
    answerable = [q for q in questions if q["relevant"]]
    unanswerable = [q for q in questions if not q["relevant"]]
    cited, correct_source, supported, refused = [], [], [], []

    for q in answerable:
        documents = c.rag.retrieve(q["question"])
        result = c.rag.generate_answer(q["question"])
        numbers = {int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])}
        cited_ids = {s["document_id"] for s in result["sources"] if s["n"] in numbers}
        cited.append(float(bool(numbers)))
        correct_source.append(float(bool(cited_ids & set(q["relevant"]))))
        context, _ = c.rag.context_builder.build(documents)
        verdict = c.rag.llm.chat(
            [{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=result["answer"])}], 0
        )
        supported.append(float(verdict.strip().upper().startswith("SUPPORTED")))

    for q in unanswerable:
        refused.append(float(NOT_FOUND.lower() in c.rag.generate_answer(q["question"])["answer"].lower()))

    return {
        "Answers with citations": statistics.mean(cited),
        "Cites a relevant document": statistics.mean(correct_source),
        "Faithful (LLM judge)": statistics.mean(supported),
        "Out-of-scope refused": statistics.mean(refused) if refused else float("nan"),
    }


def to_markdown(rows):
    metrics = list(rows[0][1])
    lines = ["| Strategy | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(f"{v[m]:.3f}" for m in metrics) + " |" for name, v in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/golden_cordis.jsonl")
    parser.add_argument("--generation", action="store_true")
    args = parser.parse_args()

    questions = [json.loads(line) for line in Path(args.golden).read_text(encoding="utf-8").splitlines() if line.strip()]
    c = get_container()

    answerable = [q for q in questions if q["relevant"]]
    report = [f"## Retrieval ({len(answerable)} questions, document level)\n", to_markdown(evaluate_retrieval(c, answerable))]
    for qtype in sorted({q["type"] for q in answerable}):
        subset = [q for q in answerable if q["type"] == qtype]
        report += [f"\n### Question type: {qtype} ({len(subset)})\n", to_markdown(evaluate_retrieval(c, subset))]

    if args.generation:
        gen = evaluate_generation(c, questions)
        report += [f"\n## Generation ({config.LLM_MODEL})\n", "| Metric | Score |", "|---|---|"]
        report += [f"| {k} | {v:.3f} |" for k, v in gen.items()]

    text = "\n".join(report)
    print(text)
    c.vector_store.close()
    Path("eval/results.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
