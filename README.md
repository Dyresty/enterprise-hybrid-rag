# Enterprise Hybrid RAG Platform

Question answering over document collections with **hybrid retrieval** (dense + BM25 fused with Reciprocal Rank Fusion), **cross-encoder reranking**, **parent-child chunking** and **grounded answers with numbered citations**. It runs against any OpenAI-compatible LLM, so the same code works with OpenAI or fully on-prem with vLLM / Ollama.

The demo corpus is **1,479 EU Horizon Europe research projects from CORDIS**: every project with Universität Paderborn as a participant, plus computer-science projects with German participants. That is 8,298 chunks. The system answers research-support questions such as *"Which project makes ML over knowledge graphs explainable, and how much EU funding did it get?"*, and every answer links back to the source project.

## Architecture

```
                 ┌──────────────────────── Ingestion ────────────────────────┐
 PDF upload ───► │ PyMuPDF ─┐                                                │
 CORDIS CSV ───► │ project ─┴─► parent chunks (4000) ─► child chunks (800) ─►│─► bge-base embeddings ─► Qdrant
                 │            each child stores its parent text + metadata   │      (child vector + payload)
                 └───────────────────────────────────────────────────────────┘

 Question ─┬─► dense search (Qdrant, top 30) ─┐
           └─► BM25 (title + chunk, top 30) ──┴─► RRF fusion ─► cross-encoder rerank ─► top 5 children
                                                                                          │
                     answer + [n] citations ◄── LLM ◄── system prompt ◄── dedupe by parent, send parent text
```

| Decision | Why |
|---|---|
| **Hybrid dense + BM25** | Embeddings miss exact tokens. On grant IDs and acronyms, dense search finds only 4 of 6, while BM25 finds 6 of 6 (see results). |
| **RRF instead of score blending** | Cosine and BM25 scores live on different scales. RRF uses only ranks: `Σ 1/(60 + rank)`. No normalisation, no tuning. |
| **Cross-encoder rerank** | A bi-encoder embeds query and passage separately. A cross-encoder reads them together, which is more precise but O(n), so it only re-scores the 30 fused candidates. |
| **Parent-child chunks** | Small children make retrieval precise. The LLM gets the larger parent for context. Children of the same parent are merged once. |
| **Grounding prompt + refusal** | Answer only from the sources, cite `[n]` for every claim, and use a fixed refusal sentence. Chunks below a reranker score of 0.01 are dropped before the LLM is called. |
| **OpenAI-compatible LLM client** | Set `LLM_BASE_URL` to switch to vLLM or Ollama. No code change, and no data leaves the network (GDPR). |
| **Embedded Qdrant by default** | Runs with no server. Set `QDRANT_URL` to use the docker-compose Qdrant server. |

## Evaluation

`eval/golden_cordis.jsonl` holds 42 hand-written questions:
- 29 **semantic** paraphrases that deliberately avoid the project's title words
- 6 **lexical** questions (acronym or grant ID)
- 2 **metadata** questions
- 5 **out-of-scope** questions that must be refused

The metrics are document level: several chunks of one project count once.

```bash
python -m app.evaluation.benchmark               # retrieval
python -m app.evaluation.benchmark --generation  # + citation / faithfulness (LLM judge) / refusal checks
```

**Retrieval, all 37 answerable questions:**

| Strategy | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| dense (bge-base-en-v1.5) | 0.865 | 0.946 | 0.946 | 0.905 |
| BM25 | 0.676 | 0.919 | 0.973 | 0.777 |
| hybrid (RRF) | 0.838 | 0.973 | **1.000** | 0.899 |
| **hybrid + rerank (bge-reranker-base)** | **0.892** | 0.973 | **1.000** | **0.937** |

**Lexical questions only (acronyms, grant IDs):** dense Hit@1 0.667, BM25 1.000, hybrid + rerank **1.000**.

**What the numbers show:**
- Dense retrieval alone fails on exact identifiers. Asked about a grant number, it returns semantically similar projects.
- Hybrid fixes recall: every relevant project is in the top 10. But naive fusion can push BM25 noise to rank 1 on semantic questions.
- The reranker restores precision at rank 1.
- **The reranker choice matters.** The common `ms-marco-MiniLM-L-6-v2` *lowered* semantic Hit@1 from 0.897 to 0.793 on this domain. `bge-reranker-v2-m3` gave 0.838 overall. `bge-reranker-base` was best (MRR 0.937), so it is the default. This was measured, not assumed.
- **Refusal threshold:** reranker scores of valid and out-of-scope questions overlap. A valid metadata question scored 0.042, while "Volkswagen revenue" scored 0.045. So a score threshold only removes obvious junk, and refusal relies on the prompt as a second layer.

**Generation (gpt-4.1-mini, 37 answerable + 5 out-of-scope questions):**

| Metric | Score |
|---|---|
| Answers containing `[n]` citations | 1.000 |
| Cites a relevant document | 0.973 |
| Faithful to sources (LLM judge) | 1.000 |
| Out-of-scope questions refused | 1.000 |

The single citation "miss" is a labelling issue, not a system error. *"AI support for emergency response decisions over huge real-time data streams"* was labelled CREXDATA, but the system answered with DARE, a European emergency-response project that also fits. The label was left unchanged, so the score is not tuned after the fact.

**Known weakness: aggregation questions.** *"Which projects with Paderborn University work on quantum photonics?"* returns 3 correct projects, but the corpus holds more (EPIQUE, QSNP, MicrocombSys, ...). Top-k retrieval can only answer from the 5 chunks it sees, so "list all X" questions need structure, not more chunks. Two fixes:
- Qdrant payload filters on extracted fields (participant, programme, year)
- a knowledge graph of projects, organisations and topics, queried with SPARQL or GraphRAG and routed to by query type

Limitations: 37 questions is small (one question = 2.7 points), and almost every question has a single relevant document, so P@5 tops out at 0.2. Next steps:
- a larger set, with multi-document questions
- German-language queries (switch to `bge-m3`)
- an LLM-judge faithfulness run on the full set

## Fine-tuning a local model (QLoRA)

The goal: a small open model that can answer in this pipeline **fully on-prem**, in the required format (grounded, `[n]` citations, exact refusal sentence). RAG supplies the facts. Fine-tuning teaches behaviour.

**Data (teacher distillation), `finetune/build_dataset.py`:**
1. 150 CORDIS projects are sampled, **excluding every project in the evaluation set**.
2. gpt-4.1-mini writes 2 questions per project: one topical paraphrase and one factual question.
3. The real retrieval pipeline fetches context for each question, distractor projects included.
4. The teacher answers with the production system prompt.
5. Answers that don't cite the source project are dropped (2 of 300).
6. 40 off-topic questions are added, with the refusal sentence as the target.

Result: 337 examples (303 train, 34 validation). Generated questions are cached so reruns cost nothing.

**Training, `finetune/train_qlora.py`:**
- Base model: Qwen2.5-3B-Instruct, loaded in 4-bit NF4 with double quantisation and bf16 compute.
- LoRA: r=16, alpha=32, dropout 0.05 on all attention and MLP projections. 29.9M trainable parameters (0.96%).
- Loss on the answer only (prompt/completion format).
- Optimiser: LR 2e-4 with cosine schedule, effective batch 8, gradient checkpointing, 1 epoch.
- Examples longer than 3,072 tokens are dropped rather than truncated, because truncation would cut the answer. 261 examples were used.
- Training took 17 minutes on a 12 GB laptop GPU, with a peak of 6.7 GB. Validation loss was 0.335 and validation token accuracy 90.3%.

> Long sequences × Qwen's 152k-token vocabulary make the output logits the main memory cost (about 2.3 GB in fp32 for one 3.8k-token sequence). Before the length filter and an eval batch size of 1, training spilled past 12 GB into shared system memory and slowed down about 10×, without an out-of-memory error.

**Serving.** The adapter is merged into the base weights. Two serving paths, both OpenAI-compatible, so the RAG switches with `LLM_BASE_URL` alone:
- **Ollama:** convert to GGUF q8_0 with llama.cpp's `convert_hf_to_gguf.py`, then `ollama create` with Qwen's chat template (`finetune/Modelfile`). The installed Ollama imports safetensors only through its MLX engine, which doesn't support Qwen2, so the conversion goes through GGUF.
- **`finetune/serve.py`:** a small transformers server in bf16, with no extra tooling.

Base and fine-tuned models are always served the same way. The judge stays fixed at gpt-4.1-mini.

| Metric | Base 3B (Ollama q8_0) | **Fine-tuned (Ollama q8_0)** | Base 3B (transformers bf16) | Fine-tuned (transformers bf16) | gpt-4.1-mini |
|---|---|---|---|---|---|
| Answers with `[n]` citations | 0.757 | **0.892** | 0.730 | 0.784 | 1.000 |
| Cites a relevant document | 0.730 | **0.865** | 0.703 | 0.757 | 0.973 |
| Faithful (LLM judge) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Out-of-scope refused | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Median latency | 1.4 s | 1.7 s | 4.9 s | 7.4 s | API |

How to read this:
- In both runtimes the fine-tuned model cites more often and cites the right project more often: +13.5 points in Ollama (5 of 37 questions), +5.4 points with transformers (2 of 37).
- The gain is consistent in direction. Its size varies between runs, as expected with 37 questions.
- Faithfulness and refusals were already perfect for both models.
- llama.cpp in Ollama is about 4× faster than the plain transformers server.
- A 3B model with 261 examples and 1 epoch is a proof of concept. It still trails gpt-4.1-mini.

Next steps:
- more data and epochs
- DPO on cited vs uncited answers
- a 7B model on a cluster GPU
- a larger evaluation set to measure small differences reliably

```bash
python -m finetune.build_dataset --projects 150 --off-topic 40
python -m finetune.train_qlora

# Ollama (llama.cpp checked out next to this repo)
python ../tools/llama.cpp/convert_hf_to_gguf.py finetune/out/merged --outtype q8_0 --outfile finetune/out/gguf/merged-q8_0.gguf
cd finetune && ollama create cordis-qwen -f Modelfile && cd ..
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=cordis-qwen python -m app.evaluation.benchmark --generation --out eval/results_cordis_qwen_ollama.md

# or without Ollama
python -m finetune.serve --model finetune/out/merged --name cordis-qwen --port 8100
LLM_BASE_URL=http://127.0.0.1:8100/v1 LLM_MODEL=cordis-qwen python -m app.evaluation.benchmark --generation --out eval/results_cordis_qwen.md
```

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or /cpu
pip install -r requirements.txt
cp .env.example .env                                   # set the LLM endpoint / key

# Data: https://cordis.europa.eu/data/cordis-HORIZONprojects-csv.zip -> unzip into data/cordis/
python -m scripts.ingest_cordis --limit 1500           # ~2 min on an RTX 5070 Ti
uvicorn app.main:app --reload                          # http://localhost:8000/docs
python -m pytest                                       # unit tests, no models needed
```

| Endpoint | |
|---|---|
| `POST /chat` `{"question": "...", "top_k": 5}` | Answer with `[n]` citations plus a source list (title, CORDIS URL, score) |
| `GET /search?q=...` | Retrieval only, no LLM: shows exactly what the model would see |
| `POST /upload` | Ingest a PDF (parsed, chunked, embedded, BM25 refreshed) |
| `GET /health` | Status and chunk count |

## Project layout

```
app/
  ingestion/   parser.py (PDF), chunker.py (parent-child)
  embeddings/  bge with query instruction
  retrieval/   retriever.py (dense), bm25.py, hybrid.py (RRF), reranker.py
  services/    rag_service.py, context_builder.py, ingestion_service.py, container.py (wiring)
  llm/         prompt_builder.py (system prompt), llm_service.py (OpenAI-compatible)
  evaluation/  retrieval_metrics.py, benchmark.py
scripts/ingest_cordis.py
eval/        golden_cordis.jsonl, results.md
tests/       test_core.py
```
