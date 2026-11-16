"""Build a supervised fine-tuning set for grounded, cited RAG answers (teacher distillation).

For a sample of CORDIS projects:
  1. a teacher LLM writes two questions about the project (one topical paraphrase, one factual),
  2. the real retrieval pipeline fetches context for each question (including distractor projects),
  3. the teacher answers with the production system prompt.
Answers that do not cite the source project are dropped. Off-topic questions are added with the
exact refusal sentence as target. Projects used in eval/golden_cordis.jsonl are excluded.

    python -m finetune.build_dataset --projects 250
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.llm.prompt_builder import NOT_FOUND, create_messages
from app.services.container import get_container

OUT = Path("finetune/data")

QUESTION_PROMPT = """Write two questions a researcher might ask that this EU project answers.
1. A topical question describing the research WITHOUT using the project's acronym or title words.
2. A factual question about funding, coordinator, partners, dates or call, naming the project by acronym.
Return JSON: {{"questions": ["...", "..."]}}

Project:
{text}"""

OFF_TOPIC_PROMPT = """Write {n} diverse questions that an EU research-project database cannot answer:
everyday topics, sports, cooking, local administration, company finances, travel, celebrities, general trivia.
Return JSON: {{"questions": [...]}}"""


def ask_json(llm, prompt, attempts=3):
    for _ in range(attempts):
        response = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        try:
            questions = json.loads(response.choices[0].message.content)["questions"]
            return [q for q in questions if isinstance(q, str) and q.strip()]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", type=int, default=250)
    parser.add_argument("--off-topic", type=int, default=60)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    c = get_container()
    llm = c.rag.llm
    random.seed(args.seed)

    eval_ids = {
        doc_id
        for line in Path("eval/golden_cordis.jsonl").read_text(encoding="utf-8").splitlines()
        for doc_id in json.loads(line)["relevant"]
    }
    # Not splitlines(): CORDIS texts contain U+2028 line separators inside JSON strings.
    with open("data/cordis/corpus.jsonl", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]
    projects = random.sample([p for p in corpus if p["id"] not in eval_ids], args.projects)

    # Generated questions are cached, so a rerun does not pay for them again.
    OUT.mkdir(parents=True, exist_ok=True)
    cache_file = OUT / f"questions_{args.projects}_{args.seed}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        print(f"Generating questions for {len(projects)} projects")
        with ThreadPoolExecutor(8) as pool:
            question_lists = list(pool.map(lambda p: ask_json(llm, QUESTION_PROMPT.format(text=p["text"][:4000])), projects))
        cached = {
            "projects": {p["id"]: qs[:2] for p, qs in zip(projects, question_lists)},
            "off_topic": ask_json(llm, OFF_TOPIC_PROMPT.format(n=args.off_topic)),
        }
        cache_file.write_text(json.dumps(cached, ensure_ascii=False, indent=1), encoding="utf-8")

    items = [(q, project_id) for project_id, qs in cached["projects"].items() for q in qs]
    items += [(q, None) for q in cached["off_topic"]]

    print(f"Retrieving context for {len(items)} questions")
    prompts = []
    for question, project_id in items:
        documents = c.rag.retrieve(question)
        context, sources = c.rag.context_builder.build(documents)
        prompts.append((question, project_id, create_messages(question, context), sources))

    def teacher_answer(entry):
        question, project_id, messages, sources = entry
        if project_id is None:
            return NOT_FOUND
        return llm.chat(messages)

    print("Generating teacher answers")
    with ThreadPoolExecutor(8) as pool:
        answers = list(pool.map(teacher_answer, prompts))

    examples, dropped = [], 0
    for (question, project_id, messages, sources), answer in zip(prompts, answers):
        if project_id is not None:
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
            if not any(s["n"] in cited and s["document_id"] == project_id for s in sources):
                dropped += 1
                continue
        examples.append(
            {
                "prompt": messages,
                "completion": [{"role": "assistant", "content": answer}],
                "project_id": project_id,
            }
        )

    random.shuffle(examples)
    split = int(len(examples) * 0.9)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", examples[:split]), ("val", examples[split:])):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    refusals = sum(e["project_id"] is None for e in examples)
    print(f"Kept {len(examples)} examples ({refusals} refusals), dropped {dropped} without a correct citation")
    print(f"train: {split}, val: {len(examples) - split}")
    c.vector_store.close()


if __name__ == "__main__":
    main()
