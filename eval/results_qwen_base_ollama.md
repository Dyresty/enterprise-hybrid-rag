## Retrieval (37 questions, document level)

| Strategy | Hit@1 | Hit@5 | Hit@10 | Recall@5 | P@5 | MRR |
|---|---|---|---|---|---|---|
| dense | 0.865 | 0.946 | 0.946 | 0.946 | 0.189 | 0.905 |
| bm25 | 0.676 | 0.919 | 0.973 | 0.919 | 0.184 | 0.777 |
| hybrid (RRF) | 0.838 | 0.973 | 1.000 | 0.973 | 0.195 | 0.899 |
| hybrid + rerank | 0.892 | 0.973 | 1.000 | 0.973 | 0.195 | 0.937 |

### Question type: lexical (6)

| Strategy | Hit@1 | Hit@5 | Hit@10 | Recall@5 | P@5 | MRR |
|---|---|---|---|---|---|---|
| dense | 0.667 | 0.667 | 0.667 | 0.667 | 0.133 | 0.667 |
| bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 |
| hybrid (RRF) | 0.667 | 1.000 | 1.000 | 1.000 | 0.200 | 0.792 |
| hybrid + rerank | 1.000 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 |

### Question type: metadata (2)

| Strategy | Hit@1 | Hit@5 | Hit@10 | Recall@5 | P@5 | MRR |
|---|---|---|---|---|---|---|
| dense | 1.000 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 |
| bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 |
| hybrid (RRF) | 1.000 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 |
| hybrid + rerank | 1.000 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 |

### Question type: semantic (29)

| Strategy | Hit@1 | Hit@5 | Hit@10 | Recall@5 | P@5 | MRR |
|---|---|---|---|---|---|---|
| dense | 0.897 | 1.000 | 1.000 | 1.000 | 0.200 | 0.948 |
| bm25 | 0.586 | 0.897 | 0.966 | 0.897 | 0.179 | 0.716 |
| hybrid (RRF) | 0.862 | 0.966 | 1.000 | 0.966 | 0.193 | 0.914 |
| hybrid + rerank | 0.862 | 0.966 | 1.000 | 0.966 | 0.193 | 0.920 |

## Generation (qwen-base-q8)

| Metric | Score |
|---|---|
| Answers with citations | 0.757 |
| Cites a relevant document | 0.730 |
| Faithful (LLM judge) | 1.000 |
| Out-of-scope refused | 1.000 |
| Median answer latency (s) | 1.378 |
