# Enterprise Hybrid RAG Platform 🚀

A production-style **Retrieval-Augmented Generation (RAG)** platform designed to process enterprise documents and enable intelligent question answering using modern LLM application architecture.

This project demonstrates a complete enterprise RAG pipeline including document ingestion, preprocessing, hybrid retrieval, reranking, evaluation, and deployment.

The goal is to build a system similar to real-world internal knowledge assistants used in companies.

---

# Project Status

## Completed ✅

### Milestone 1 — Backend Foundation
- FastAPI application setup
- API routing architecture
- Swagger API documentation
- Project structure creation

### Milestone 2 — Document Ingestion Pipeline
- PDF upload API
- PDF file validation
- Unique document ID generation
- Duplicate filename handling
- PDF storage management
- Text extraction using PyMuPDF

### Milestone 3 — Document Chunking
- Recursive text splitting
- Chunk metadata tracking
- Page number preservation
- Chunk ID generation


## In Progress 🚧

- Embedding generation
- Vector database integration
- Semantic search
- Hybrid retrieval


## Planned Features 🔜

### Document Intelligence
- OCR for scanned documents
- Table extraction
- Image extraction
- Metadata extraction


### Retrieval System
- Dense vector search
- Sparse retrieval using BM25
- Hybrid search
- Query expansion
- Cross-encoder reranking


### Generation System
- LLM integration
- Context-aware prompting
- Citation generation
- Streaming responses
- Conversation memory


### Enterprise Features
- RAG evaluation
- Observability
- Cost tracking
- Authentication
- Docker deployment

---

# Architecture Overview

Current pipeline:

```
                    PDF Document

                         |
                         v

                  FastAPI Upload API

                         |
                         v

                 Document Processing

                         |
                         v

                  PDF Text Extraction
                     (PyMuPDF)

                         |
                         v

                  Document Chunking

                         |
                         v

              Embedding Generation
                 (Coming Soon)

                         |
                         v

                Vector Database
                 (Coming Soon)

                         |
                         v

                 LLM Generation
                 (Coming Soon)
```

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
│   │   └── vectorstore.py
│   │
│   ├── main.py
│   └── config.py
│
├── data/
│   ├── uploads/
│   └── processed/
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


## Retrieval & AI Stack

Planned:

- Sentence Transformers
- Qdrant Vector Database
- BM25 Sparse Retrieval
- Cross Encoder Reranking
- OpenAI API


## Deployment

Planned:

- Docker
- Docker Compose
- Linux deployment

---

# API Endpoints

## Health Check

### GET /

Checks whether the application is running.

Example response:

```json
{
    "message": "Enterprise Hybrid RAG Platform Running"
}
```

---

## Router Test

### GET /hello

Tests API routing.

Example response:

```json
{
    "message": "Upload router works!"
}
```

---

# PDF Upload API

## POST /upload

Uploads and processes PDF documents.

### Input

Multipart form data:

```
file: PDF document
```

---

## Processing Pipeline

```
PDF

↓

File Validation

↓

Unique Filename Generation

↓

File Storage

↓

Text Extraction

↓

Document Chunking

↓

Return Metadata
```

---

## Example Response

```json
{
    "document_id": "uuid",
    "filename": "document.pdf",
    "stored_as": "uuid_document.pdf",
    "pages": 10,
    "total_chunks": 45,
    "first_chunk": "Extracted document text..."
}
```

---

# Milestones

---

# Milestone 1 — Backend Foundation ✅

## Objective

Create the backend foundation for the RAG system.

## Implemented

- FastAPI application
- API router structure
- Swagger documentation


## Running the application

```bash
uvicorn app.main:app --reload
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

# Milestone 2 — PDF Document Ingestion ✅

## Objective

Create a pipeline for accepting enterprise documents.

## Implemented

- PDF upload endpoint
- File type validation
- Unique document identifiers
- File storage
- Text extraction


## Pipeline

```
PDF

↓

FastAPI Upload

↓

Validation

↓

Storage

↓

PyMuPDF

↓

Extracted Text
```

---

# Milestone 3 — Document Chunking ✅

## Objective

Convert large documents into smaller searchable pieces.

Large documents cannot directly be passed into LLMs because of context window limitations.

Chunking allows:

- Efficient retrieval
- Lower token usage
- Better answer accuracy


## Implementation

Used:

```
RecursiveCharacterTextSplitter
```


Configuration:

```
Chunk Size:
800 characters


Chunk Overlap:
150 characters
```


## Chunk Metadata

Each chunk contains:

```json
{
    "chunk_id": 0,
    "page": 1,
    "text": "document content"
}
```

---

# Upcoming Milestones

---

# Milestone 4 — Embeddings & Vector Database

Pipeline:

```
Document Chunks

↓

Embedding Model

↓

Vector Representation

↓

Qdrant Storage
```

Technologies:

- Sentence Transformers
- Qdrant


---

# Milestone 5 — Hybrid Retrieval

Combine:

## Dense Retrieval

Using:

- Vector embeddings
- Semantic similarity


## Sparse Retrieval

Using:

- BM25
- Keyword matching


Architecture:

```
User Query

      |
      |

+-------------+
| Dense Search|
+-------------+

      +

+-------------+
| BM25 Search |
+-------------+

      |
      v

Hybrid Results

      |
      v

Reranker
```

---

# Milestone 6 — LLM Generation

Features:

- Context injection
- Prompt templates
- Answer generation
- Citations
- Streaming responses


Pipeline:

```
Question

↓

Retriever

↓

Relevant Documents

↓

LLM

↓

Final Answer
```

---

# Milestone 7 — Enterprise Features

Planned:

- Conversation memory
- RAG evaluation
- Langfuse monitoring
- Phoenix evaluation
- Authentication
- Docker deployment
- Production deployment

---

# Installation

## Create Environment

```bash
conda activate worky
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

Open Swagger:

```
http://127.0.0.1:8000/docs
```

---

# Learning Objectives

This project demonstrates:

- Enterprise RAG architecture
- Document ingestion systems
- API development
- Vector databases
- Semantic search
- Information retrieval
- LLM application development
- Production AI engineering practices

---

# Author

Rochan Kumar

AI / Robotics Engineer
