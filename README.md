# Enterprise Hybrid RAG Platform 🚀

A production-style **Retrieval-Augmented Generation (RAG)** platform designed to process enterprise documents and provide intelligent question answering using modern LLM application architecture.

This project implements a complete enterprise RAG pipeline including:

- Document ingestion
- PDF processing
- Intelligent chunking
- Dense vector retrieval
- Sparse keyword retrieval
- Hybrid search using Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Vector database integration
- Evaluation framework
- LLM-based generation pipeline

The goal is to build a system similar to internal enterprise knowledge assistants used for document search, knowledge management, and AI-powered question answering.

---

# Project Status

## Completed Features ✅

- Backend API architecture
- Enterprise document ingestion pipeline
- PDF extraction
- Intelligent chunking
- Embedding generation
- Vector database integration
- Dense semantic retrieval
- BM25 sparse retrieval
- Hybrid retrieval using Reciprocal Rank Fusion
- Cross-encoder reranking
- Retrieval pipeline architecture

## In Progress 🚧

- LLM generation layer
- Retrieval evaluation framework
- RAG evaluation metrics
- Deployment pipeline

---

# System Architecture

```
                         User Query

                              |
                              v

                    Query Processing Layer

                              |
                              v


        +---------------------+---------------------+

        |                                           |

        v                                           v


 Dense Vector Retrieval                    Sparse Retrieval

 (Embedding Search)                         (BM25 Search)


        |                                           |

        +---------------------+---------------------+

                              |

                              v


                 Reciprocal Rank Fusion (RRF)


                              |

                              v


                 Hybrid Candidate Retrieval


                              |

                              v


                  Cross Encoder Reranking


                              |

                              v


                    Relevant Context


                              |

                              v


                       LLM Generation


                              |

                              v


                       Final Answer
```

---

# Milestone 1 — Backend Foundation

Implemented:

- FastAPI backend architecture
- API routing structure
- Swagger API documentation
- Configuration management
- Modular project organization

Backend follows a scalable service-oriented structure suitable for production AI applications.

---

# Milestone 2 — Enterprise Document Ingestion Pipeline

The ingestion pipeline processes enterprise documents and converts them into searchable knowledge units.

Implemented:

- PDF upload API
- File validation
- Unique document ID generation
- Duplicate filename handling
- Document storage management
- Text extraction using PyMuPDF


## Pipeline

```
PDF Document

      |
      v

FastAPI Upload API

      |
      v

File Validation

      |
      v

Document Storage

      |
      v

PyMuPDF Extraction

      |
      v

Raw Document Text
```

---

# Milestone 3 — Intelligent Document Chunking

Large documents are split into smaller semantic units optimized for retrieval.

Implemented:

- Recursive text splitting
- Chunk metadata tracking
- Page number preservation
- Unique chunk ID generation


## Chunking Strategy

```
Chunk Size:
800 characters

Chunk Overlap:
150 characters
```


Each chunk stores metadata:

```json
{
    "document_id": "uuid",
    "chunk_id": 12,
    "page": 5,
    "text": "document content"
}
```

Benefits:

- Reduces LLM context size
- Improves retrieval accuracy
- Enables document referencing
- Preserves source information

---

# Milestone 4 — Embeddings & Vector Database

The system converts document chunks into dense vector representations for semantic retrieval.

Implemented:

- Embedding generation
- Vector indexing
- Similarity search
- Vector database integration


## Architecture

```
Document Chunk

        |
        v

Embedding Model

        |
        v

Dense Vector

        |
        v

Vector Database
```


## Technologies

- Sentence Transformers
- Qdrant Vector Database


Dense retrieval enables searching based on meaning rather than exact keywords.


Example:

Query:

```
"What caused the laboratory accident?"
```


The system can retrieve:

```
"Failure of the fire suppression system resulted in equipment damage."
```

even without exact keyword overlap.

---

# Milestone 5 — Hybrid Retrieval System

Implemented a hybrid retrieval pipeline combining:

1. Dense semantic retrieval
2. Sparse keyword retrieval


Hybrid retrieval improves reliability by combining the strengths of both approaches.

---

## Dense Retrieval

Uses:

- Transformer embeddings
- Vector similarity search


Advantages:

- Understands semantic meaning
- Handles natural language questions
- Finds conceptually similar documents


Example:

```
Query:
"vehicle stopped suddenly"

Can retrieve:

"unexpected emergency braking event"
```

---

## Sparse Retrieval

Uses:

- BM25 ranking algorithm


Advantages:

- Strong keyword matching
- Effective for technical terms
- Works well with IDs, names, and exact phrases


Example:

```
Query:

"ISO 26262 ASIL-D"

BM25 can prioritize exact technical matches.
```

---

# Hybrid Retrieval Architecture

```
                         User Query

                              |
                              |

              +---------------+---------------+

              |                               |

              v                               v


       Dense Retrieval                  BM25 Retrieval

       Vector Search                   Keyword Search


              |                               |

              +---------------+---------------+

                              |

                              v


                  Reciprocal Rank Fusion


                              |

                              v


                    Hybrid Ranked Results
```

---

# Reciprocal Rank Fusion (RRF)

Dense retrieval and BM25 produce ranking results using different scoring systems.

Example:

## Dense Retrieval

```
Document A → similarity score: 0.72

Document B → similarity score: 0.65

Document C → similarity score: 0.61
```


## BM25 Retrieval

```
Document B → BM25 score: 12.5

Document A → BM25 score: 9.8

Document D → BM25 score: 7.4
```


These scores cannot be directly combined because they exist on different scales.

Instead, the system combines rankings using Reciprocal Rank Fusion:

```
RRF Score = Σ 1 / (k + rank)
```


Where:

```
rank = document position in retrieval list

k = constant controlling rank contribution
```


Benefits:

- Combines semantic understanding from dense retrieval
- Preserves exact keyword matching from BM25
- Produces more robust retrieval results

---

# Milestone 6 — Cross Encoder Reranking

After hybrid retrieval, the system improves ranking quality using a Cross Encoder model.


Implemented:

- Candidate retrieval
- Query-document scoring
- Final relevance ranking


## Pipeline

```
User Query

      |

      v

Hybrid Retriever

      |

      v

Top-K Candidate Documents

      |

      v

Cross Encoder Reranker

      |

      v

Final Ranked Context
```


The retriever optimizes for:

```
High Recall
```

The reranker optimizes for:

```
High Precision
```


The Cross Encoder evaluates:

```
(query, document)
```

pairs together and produces a relevance score.


Benefits:

- Removes irrelevant chunks
- Improves context quality
- Provides better information for LLM generation

---

# Project Structure

```
enterprise-hybrid-rag/

│
├── app/
│
│   ├── api/
│   │   ├── upload.py
│   │   └── chat.py
│   │
│   ├── ingestion/
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   └── ocr.py
│   │
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   ├── evaluation/
│   │
│   ├── main.py
│   └── config.py
│
├── data/
│   ├── uploads/
│   └── processed/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# Technology Stack

## Backend

- Python
- FastAPI
- Uvicorn


## Document Processing

- PyMuPDF
- LangChain Text Splitters


## Retrieval Stack

- Sentence Transformers
- Qdrant
- BM25
- Reciprocal Rank Fusion
- Cross Encoder Models


## AI Stack

- Retrieval-Augmented Generation
- LLM APIs
- Prompt Engineering
- Semantic Search


## Deployment

Planned:

- Docker
- Docker Compose
- Cloud deployment
- Monitoring

---

# API Endpoints

## Health Check

```
GET /
```

Response:

```json
{
    "message": "Enterprise Hybrid RAG Platform Running"
}
```

---

## Upload Document

```
POST /upload
```


Uploads and processes enterprise PDF documents.


Processing pipeline:

```
PDF

↓

Validation

↓

Storage

↓

Text Extraction

↓

Chunking

↓

Embedding Generation

↓

Vector Storage
```


Example response:

```json
{
    "document_id": "uuid",
    "filename": "document.pdf",
    "pages": 20,
    "total_chunks": 120
}
```

---

# Running the Project

## Create Environment

```bash
conda create -n worky python=3.10

conda activate worky
```


## Install Dependencies

```bash
pip install -r requirements.txt
```


## Start Backend

```bash
uvicorn app.main:app --reload
```


Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

# Evaluation Framework

Planned evaluation framework for measuring retrieval and generation quality.

## Retrieval Evaluation

Metrics:

- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- NDCG


## RAG Evaluation

Metrics:

- Answer relevance
- Faithfulness
- Context relevance
- Hallucination rate


Tools:

- RAGAS
- Langfuse
- Phoenix

---

# Future Improvements 🚀

## Document Intelligence

- OCR support
- Table extraction
- Image understanding
- Metadata extraction


## Advanced Retrieval

- Parent-child retrieval
- Metadata filtering
- Query rewriting
- Multi-query retrieval
- Context compression


## Generation Layer

- LLM integration
- Citation generation
- Streaming responses
- Conversation memory


## Enterprise Features

- Authentication
- User permissions
- Monitoring
- Cost tracking
- Cloud deployment

---

# Learning Objectives

This project demonstrates:

- Enterprise RAG architecture
- Information retrieval systems
- Hybrid search techniques
- Vector databases
- Semantic search
- Ranking algorithms
- Cross encoder reranking
- LLM application development
- Production AI engineering practices


---

# Author

## Rochan Kumar

AI / Robotics Engineer

````
